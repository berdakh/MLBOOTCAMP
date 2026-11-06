"""A small decoder-only transformer (GPT), written to be read top to bottom.

Introduced in Level 3 notebook 03. Every piece of a modern decoder is here,
at toy scale:

    token + position embeddings -> N x [pre-LayerNorm -> causal self-attention
    -> residual -> pre-LayerNorm -> MLP -> residual] -> LayerNorm -> logits
    (output projection tied to the token embedding)

plus character-level data loading, sampling (temperature, top-k) and a
**KV cache** for fast generation. The structure follows GPT-2 / nanoGPT;
the code favours clarity over speed (attention is written out instead of
calling ``F.scaled_dot_product_attention``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------------------- data

#: Tiny Shakespeare (~1.1 MB of plays, public domain text) as a plain file on the Hub.
#: Card: https://huggingface.co/datasets/Trelis/tiny-shakespeare (repo licence file: MIT,
#: from Karpathy's char-rnn; the underlying Shakespeare text is public domain).
SHAKESPEARE_REPO = "Trelis/tiny-shakespeare"
SHAKESPEARE_FILE = "input.txt"
SHAKESPEARE_REVISION = "2225a5674b252d623cfb151ef099d179629c2f49"

_FALLBACK = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.
"""


def load_tiny_shakespeare(revision: str = SHAKESPEARE_REVISION) -> str:
    """Download (once, cached by huggingface_hub) and return the corpus as one string."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(SHAKESPEARE_REPO, SHAKESPEARE_FILE, repo_type="dataset", revision=revision)
    except Exception as exc:  # offline: a tiny excerpt keeps the code runnable, not the results
        print(f"Could not download {SHAKESPEARE_REPO} ({type(exc).__name__}); using a 10-line excerpt.")
        return _FALLBACK * 200
    with open(path, encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """One token per character. Vocabulary = the sorted set of characters seen."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


class RandomWindows:
    """``n_batches`` batches of random ``block_size + 1`` windows from a 1-D token tensor.

    Yields ``{"x": (B, T), "y": (B, T)}`` with ``y`` = ``x`` shifted by one.
    Has ``__len__`` so ``scaleml.train.train`` can size its LR schedule.
    """

    def __init__(self, data: torch.Tensor, block_size: int, batch_size: int, n_batches: int, seed: int = 0):
        self.data, self.block_size, self.batch_size, self.n_batches = data, block_size, batch_size, n_batches
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            yield self.sample()

    def sample(self) -> dict:
        starts = torch.randint(0, len(self.data) - self.block_size - 1, (self.batch_size,),
                               generator=self.generator)
        x = torch.stack([self.data[s:s + self.block_size] for s in starts.tolist()])
        y = torch.stack([self.data[s + 1:s + 1 + self.block_size] for s in starts.tolist()])
        return {"x": x, "y": y}


# --------------------------------------------------------------------------- model


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64  # maximum context length (number of positions)
    n_embd: int = 64  # hidden size d_model
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t may only attend to positions <= t."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head, self.head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)  # Q, K, V for all heads in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, heads, T, head_dim): each head attends independently
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        past = 0
        if cache is not None:  # KV cache: keep K and V of every position generated so far
            if "k" in cache:
                past = cache["k"].shape[2]
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, h, T, past+T)
        # query i sits at absolute position past+i; it may see keys at positions <= past+i
        q_pos = torch.arange(past, past + T, device=x.device)[:, None]
        k_pos = torch.arange(past + T, device=x.device)[None, :]
        att = att.masked_fill(k_pos > q_pos, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).reshape(B, T, C)  # concatenate the heads
        return self.drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm block: x + attn(LN(x)), then x + mlp(LN(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(cfg.n_embd), CausalSelfAttention(cfg)
        self.ln2, self.mlp = nn.LayerNorm(cfg.n_embd), MLP(cfg)

    def forward(self, x, cache: dict | None = None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # learned absolute positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying: one matrix embeds and un-embeds
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2: scale residual projections by depth
            if name.endswith("proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                caches: list[dict] | None = None):
        """``idx`` (B, T) token ids -> logits (B, T, vocab) and, if ``targets``, the mean loss.

        ``caches`` (one dict per layer, from ``new_cache``) turns on KV caching:
        ``idx`` then holds only the *new* tokens, positioned after the cached ones.
        """
        B, T = idx.shape
        past = caches[0]["k"].shape[2] if caches and "k" in caches[0] else 0
        if past + T > self.cfg.block_size:
            raise ValueError(f"sequence length {past + T} exceeds block_size {self.cfg.block_size}")
        pos = torch.arange(past, past + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches is not None else None)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def new_cache(self) -> list[dict]:
        return [{} for _ in self.blocks]


def count_non_embedding_params(model: TinyGPT) -> int:
    emb = model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
    return sum(p.numel() for p in model.parameters()) - emb


# --------------------------------------------------------------------------- sampling


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                generator: torch.Generator | None = None) -> torch.Tensor:
    """(B, V) logits -> (B,) sampled ids. ``temperature=0`` means greedy (argmax)."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(1)


@torch.no_grad()
def generate(model: TinyGPT, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, use_cache: bool = True,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """Append ``max_new_tokens`` sampled tokens to ``idx`` (B, T).

    Without a cache, every step re-runs the model on the whole (cropped)
    context: O(T) work per token for the projections and O(T^2) for attention.
    With a cache, each step runs only the newest token and reuses the stored
    K and V. When the cache reaches ``block_size`` it is rebuilt from the most
    recent half-window (a simple sliding-window policy for this toy model).
    """
    model.eval()
    bs = model.cfg.block_size
    if not use_cache:
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -bs:])
            nxt = sample_next(logits[:, -1], temperature, top_k, generator)
            idx = torch.cat([idx, nxt[:, None]], dim=1)
        return idx
    caches = model.new_cache()
    logits, _ = model(idx[:, -bs:], caches=caches)  # "prefill": the whole prompt at once
    for step in range(max_new_tokens):
        nxt = sample_next(logits[:, -1], temperature, top_k, generator)
        idx = torch.cat([idx, nxt[:, None]], dim=1)
        if step == max_new_tokens - 1:
            break
        if caches[0]["k"].shape[2] >= bs:  # cache full: re-prefill a recent window
            caches = model.new_cache()
            logits, _ = model(idx[:, -(bs // 2):], caches=caches)
        else:  # "decode": one token in, one token out
            logits, _ = model(nxt[:, None], caches=caches)
    return idx


@torch.no_grad()
def estimate_loss(model: TinyGPT, windows: RandomWindows, n_batches: int, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        b = windows.sample()
        _, loss = model(b["x"].to(device), b["y"].to(device))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)
