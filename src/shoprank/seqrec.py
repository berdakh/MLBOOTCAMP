"""Sequence models for next-item prediction (notebook 10).

* :func:`click_events` - clicks in a set of days, ordered by time.
* :func:`build_sequences` - per-user, left-padded item-id arrays (0 = padding).
* :func:`causal_mask`, :func:`padding_mask`, :func:`attention_mask` - the masks,
  with the convention of ``torch.nn.MultiheadAttention``: **True = blocked**.
* :class:`SASRec` - a small self-attention next-item model (Kang & McAuley,
  2018, "Self-Attentive Sequential Recommendation"), written out by hand.
* :func:`rank_of_target`, :func:`hr_ndcg` - full-catalogue and sampled metrics.

Item ids inside sequences are shifted by +1 so that 0 can mean "padding";
``seq_id - 1`` is the ShopStream ``item_id``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch import nn

PAD = 0


def click_events(imps: pd.DataFrame, days) -> pd.DataFrame:
    """Clicks in ``days``, sorted by (user, time, request, position)."""
    days = list(days)
    clicks = imps.loc[imps["clicked"] & imps["day"].between(min(days), max(days)),
                      ["user_id", "item_id", "ts", "day", "request_id", "position"]]
    return clicks.sort_values(["user_id", "ts", "request_id", "position"], kind="stable").reset_index(drop=True)


def build_sequences(df: pd.DataFrame, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(users, seqs)``: sorted unique users and an int64 array of shape
    ``(len(users), max_len)`` holding each user's LAST ``max_len`` items in time
    order, shifted by +1 and left-padded with 0. Ties in ``ts`` keep row order."""
    d = df.sort_values(["user_id", "ts"], kind="stable")
    u = d["user_id"].to_numpy()
    it = d["item_id"].to_numpy().astype(np.int64) + 1
    users, start, counts = np.unique(u, return_index=True, return_counts=True)
    seqs = np.zeros((len(users), max_len), dtype=np.int64)
    # position of each event counted from the END of its user's history (0 = most recent)
    from_end = np.repeat(start + counts, counts) - 1 - np.arange(len(u))
    keep = from_end < max_len
    row = np.repeat(np.arange(len(users)), counts)[keep]
    seqs[row, max_len - 1 - from_end[keep]] = it[keep]
    return users, seqs


# ---------------------------------------------------------------------------
# Masks and the model
# ---------------------------------------------------------------------------


def causal_mask(n: int) -> torch.Tensor:
    """(n, n) bool, True above the diagonal: position i may not attend to j > i."""
    return torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)


def padding_mask(lengths, n: int) -> torch.Tensor:
    """(B, n) bool, True at padding. Sequences are LEFT-padded: the last
    ``lengths[b]`` positions are real."""
    lengths = torch.as_tensor(lengths).reshape(-1, 1)
    return torch.arange(n).reshape(1, -1) < (n - lengths)


def attention_mask(pad: torch.Tensor, fix_all_masked_rows: bool = True) -> torch.Tensor:
    """(B, n, n) bool mask from a (B, n) padding mask: causal OR key-is-padding.

    With left padding, a padding *query* at position i can see only keys
    j <= i, which are all padding: its whole row is blocked, and a softmax over
    a row of -inf is NaN. Unblocking the diagonal fixes it; those positions'
    outputs are never used (no loss on them), so nothing else changes.
    """
    n = pad.shape[1]
    mask = causal_mask(n).to(pad.device)[None] | pad[:, None, :]
    if fix_all_masked_rows:
        mask = mask & ~torch.eye(n, dtype=torch.bool, device=pad.device)[None]
    return mask


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, mask):                       # x (B, n, d); mask (B, n, n), True = blocked
        B, n, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (t.reshape(B, n, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)          # (B, heads, n, n)
        scores = scores.masked_fill(mask[:, None], float("-inf"))
        att = torch.softmax(scores, dim=-1)  # no dropout on attention weights: it is slow on CPU
        return self.out((att @ v).transpose(1, 2).reshape(B, n, d))


class Block(nn.Module):
    """Pre-LayerNorm transformer block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.att = CausalSelfAttention(dim, n_heads)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.att(self.ln1(x), mask))
        return x + self.drop(self.ffn(self.ln2(x)))


class SASRec(nn.Module):
    """Item embedding + learned positional embedding + L causal self-attention blocks.

    ``forward(seqs)`` returns hidden states (B, n, d); the score of item j
    after position t is ``hidden[:, t] @ item_emb[j]`` (tied input/output
    embeddings, as in the paper). Positions are right-aligned: the most recent
    item always gets position ``max_len - 1``, whatever the padding.
    """

    def __init__(self, n_items: int, max_len: int, dim: int = 48, n_heads: int = 2,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([Block(dim, n_heads, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.max_len = max_len
        # Small init so that the tied output logits start near 0 (loss ~ log n_items).
        nn.init.normal_(self.item_emb.weight, std=dim ** -0.5)
        nn.init.normal_(self.pos_emb.weight, std=dim ** -0.5)
        with torch.no_grad():
            self.item_emb.weight[PAD].zero_()

    def forward(self, seqs, pad=None, fix_all_masked_rows: bool = True):
        """``pad`` (B, n) bool defaults to ``seqs == PAD``."""
        n = seqs.shape[1]
        pad = (seqs == PAD) if pad is None else pad
        pos = torch.arange(self.max_len - n, self.max_len, device=seqs.device)[None]
        x = self.drop(self.item_emb(seqs) + self.pos_emb(pos))
        mask = attention_mask(pad, fix_all_masked_rows)
        for block in self.blocks:
            x = block(x, mask)
        return self.ln(x)

    def logits(self, hidden):
        """Scores over all items (column 0 = padding, set to -inf)."""
        out = hidden @ self.item_emb.weight.T
        out[..., PAD] = float("-inf")
        return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def rank_of_target(scores: np.ndarray, target: np.ndarray, candidates: np.ndarray | None = None) -> np.ndarray:
    """0-based rank of each row's target among all columns (``candidates=None``)
    or among the target plus the given negative columns. Ties count against the
    target (pessimistic), so a constant scorer gets no free hits."""
    rows = np.arange(len(target))
    t = scores[rows, target]
    if candidates is None:
        return (scores >= t[:, None]).sum(axis=1) - 1
    neg = scores[rows[:, None], candidates]
    return (neg >= t[:, None]).sum(axis=1)


def hr_ndcg(ranks: np.ndarray, k: int = 10) -> dict[str, float]:
    ranks = np.asarray(ranks)
    hit = ranks < k
    return {f"HR@{k}": float(hit.mean()),
            f"NDCG@{k}": float(np.where(hit, 1.0 / np.log2(ranks + 2), 0.0).mean())}


def sample_negatives(target: np.ndarray, n_items: int, n_neg: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform negatives in 1..n_items (shifted ids), never equal to the row's target."""
    neg = rng.integers(1, n_items + 1, size=(len(target), n_neg))
    clash = neg == target[:, None]
    while clash.any():
        neg[clash] = rng.integers(1, n_items + 1, size=int(clash.sum()))
        clash = neg == target[:, None]
    return neg
