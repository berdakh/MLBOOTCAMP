"""Fine-tuning pretrained encoders: batching, evaluation, calibration and LoRA.

Introduced in Level 3 notebooks 05 (fine-tuning a classifier) and 06
(parameter-efficient fine-tuning). Everything is plain PyTorch on top of
``transformers`` models, so each step is a line you can read:

* ``encode`` / ``TokenBatches``: tokenise once, then yield shuffled batches
  that are *dynamically trimmed* to their longest row (``scaleml.train.train``
  accepts them directly);
* ``predict_logits`` / ``frozen_embeddings``: batched inference (fp16 autocast
  on a GPU);
* ``EarlyStopping``: keep the best weights by a validation metric and stop
  when it stops improving;
* ``expected_calibration_error``, ``reliability_table``, ``fit_temperature``:
  is a probability of 0.9 right 90 % of the time? and the one-parameter fix;
* ``LoRALinear``, ``inject_lora``, ``merge_lora_modules``: low-rank adapters
  written out, so that ``peft`` holds no surprises.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from .device import autocast

#: Pinned so that results are reproducible (see notebook 04 on revision pinning).
DISTILBERT_ID = "distilbert/distilbert-base-uncased"
DISTILBERT_REV = "12040accade4e8a0f71eabdb258fecc2e7e948be"

# --------------------------------------------------------------------------- data


def encode(tokenizer, texts, max_len: int) -> dict[str, torch.Tensor]:
    """Tokenise ``texts`` once (right-padded to ``max_len``, truncated). Batches trim the padding later."""
    enc = tokenizer(list(texts), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def _trim(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Cut right-padding columns that every row in this batch has (dynamic padding)."""
    longest = int(batch["attention_mask"].sum(1).max())
    return {k: (v[:, :longest] if v.dim() == 2 else v) for k, v in batch.items()}


class TokenBatches:
    """Mini-batches of an ``encode`` result (+ labels) for ``scaleml.train.train``.

    ``train`` calls ``batches()`` once per epoch, so calling the object returns
    itself and each iteration reshuffles with a new, seeded permutation.
    ``len()`` is the number of batches per epoch.
    """

    def __init__(self, enc: dict[str, torch.Tensor], labels=None, batch_size: int = 32,
                 shuffle: bool = True, seed: int = 0):
        self.enc = enc
        self.labels = None if labels is None else torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.batch_size, self.shuffle, self.seed, self._epoch = batch_size, shuffle, seed, 0
        self.n = len(enc["input_ids"])

    def __len__(self) -> int:
        return math.ceil(self.n / self.batch_size)

    def __call__(self):
        return self

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self._epoch)
            order = torch.randperm(self.n, generator=g)
            self._epoch += 1
        else:
            order = torch.arange(self.n)
        for start in range(0, self.n, self.batch_size):
            idx = order[start:start + self.batch_size]
            batch = {k: v[idx] for k, v in self.enc.items()}
            if self.labels is not None:
                batch["labels"] = self.labels[idx]
            yield _trim(batch)


def classification_loss(device: torch.device):
    """``loss_fn(model, batch)`` for ``train()``: move the batch, return the model's cross-entropy."""
    def loss_fn(model, batch):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        return model(**batch).loss
    return loss_fn


@torch.inference_mode()
def predict_logits(model, enc: dict[str, torch.Tensor], device: torch.device, batch_size: int = 64) -> np.ndarray:
    """Logits (float32 numpy, one row per example, original order) of a sequence classifier."""
    was_training = model.training
    model.eval()
    out = []
    for batch in TokenBatches(enc, batch_size=batch_size, shuffle=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device):
            out.append(model(**batch).logits.float().cpu())
    model.train(was_training)
    return torch.cat(out).numpy()


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-aware mean over tokens (notebook 04, Exercise 1)."""
    m = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


@torch.inference_mode()
def frozen_embeddings(encoder, enc: dict[str, torch.Tensor], device: torch.device,
                      batch_size: int = 64) -> np.ndarray:
    """Mean-pooled final hidden states of a frozen encoder (``AutoModel``), as float32 numpy."""
    encoder.eval()
    out = []
    for batch in TokenBatches(enc, batch_size=batch_size, shuffle=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device):
            h = encoder(**batch).last_hidden_state
        out.append(mean_pool(h.float(), batch["attention_mask"]).cpu())
    return torch.cat(out).numpy()


# --------------------------------------------------------------------------- early stopping


class EarlyStopping:
    """Use as ``train(..., evaluate=stopper, batches=stopper.wrap(batches))``.

    ``train`` calls ``evaluate(model)`` at the end of every epoch. The stopper
    computes ``metric_fn(model)`` (higher is better), keeps a CPU copy of the
    best weights, and after ``patience`` epochs without improvement makes the
    wrapped ``batches`` empty, so the remaining epochs do no steps. Call
    ``restore(model)`` afterwards to load the best weights.
    """

    def __init__(self, metric_fn, patience: int = 1, min_delta: float = 0.0):
        self.metric_fn, self.patience, self.min_delta = metric_fn, patience, min_delta
        self.best, self.best_epoch, self.best_state = -math.inf, -1, None
        self.bad_epochs, self.epoch, self.stopped = 0, -1, False
        self.last: dict = {}

    def wrap(self, batches):
        return lambda: [] if self.stopped else batches()

    def __call__(self, model) -> dict:
        if self.stopped:  # an epoch with no steps: nothing changed
            return {**self.last, "stopped": True}
        self.epoch += 1
        metrics = dict(self.metric_fn(model))
        score = metrics["score"]
        if score > self.best + self.min_delta:
            self.best, self.best_epoch, self.bad_epochs = score, self.epoch, 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad_epochs += 1
            self.stopped = self.bad_epochs >= self.patience
        self.last = metrics
        return {**metrics, "best_epoch": self.best_epoch, "stopped": self.stopped}

    def restore(self, model) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# --------------------------------------------------------------------------- calibration


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _bin_index(conf: np.ndarray, n_bins: int) -> np.ndarray:
    # Equal-width bins (lo, hi]; confidence 0 goes in the first bin, 1.0 in the last.
    return np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)


def expected_calibration_error(probs, labels, n_bins: int = 15) -> float:
    """ECE = sum over confidence bins of (bin size / n) * |accuracy - mean confidence|."""
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels)
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    bins = _bin_index(conf, n_bins)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def reliability_table(probs, labels, n_bins: int = 10) -> list[dict]:
    """Per confidence bin: count, mean confidence, accuracy (the data behind a reliability diagram)."""
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels)
    conf, correct = probs.max(1), (probs.argmax(1) == labels)
    bins = _bin_index(conf, n_bins)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            rows.append({"bin": f"({b / n_bins:.1f}, {(b + 1) / n_bins:.1f}]", "n": int(m.sum()),
                         "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return rows


def nll(logits, labels, temperature: float = 1.0) -> float:
    p = softmax(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(labels)), np.asarray(labels)], 1e-12, None)).mean())


def fit_temperature(logits, labels, max_iter: int = 200) -> float:
    """The temperature T > 0 minimising the NLL of softmax(logits / T); fit on *validation* data.

    Optimises log T (so T stays positive) with L-BFGS. Dividing by T never
    changes the argmax, so accuracy is untouched; only confidence moves.
    """
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


# --------------------------------------------------------------------------- LoRA


class LoRALinear(nn.Module):
    """``y = W x + b + (alpha / r) * B A x`` with ``W``, ``b`` frozen and only ``A``, ``B`` trained.

    ``A`` (r x in) starts random, ``B`` (out x r) starts at zero, so at
    initialisation the layer computes exactly what the pretrained layer did.
    Initialisation and scaling follow Hu et al. (2021) and the ``peft`` library.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)

    def merged(self) -> nn.Linear:
        """A plain ``nn.Linear`` with the adapter folded in: no extra latency at serving time."""
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                        device=self.base.weight.device, dtype=self.base.weight.dtype)
        with torch.no_grad():
            out.weight.copy_(self.merged_weight())
            if self.base.bias is not None:
                out.bias.copy_(self.base.bias)
        return out

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}"


def _set_module(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent_name) if parent_name else model, child, module)


def inject_lora(model: nn.Module, targets=("q_lin", "v_lin"), r: int = 8, alpha: float = 16,
                dropout: float = 0.0, trainable=("pre_classifier", "classifier")) -> list[str]:
    """Freeze ``model``, wrap every ``nn.Linear`` whose name ends with a ``targets`` entry in a
    ``LoRALinear``, and leave the modules named in ``trainable`` (the new task head) trainable.

    Returns the names of the wrapped layers. DistilBERT names its attention
    projections ``q_lin``, ``k_lin``, ``v_lin``, ``out_lin``; BERT uses ``query``/``value``.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.split(".")[-1] in targets]
    for n in names:
        _set_module(model, n, LoRALinear(model.get_submodule(n), r=r, alpha=alpha, dropout=dropout))
    for n, m in model.named_modules():
        if n.split(".")[-1] in trainable:
            for p in m.parameters():
                p.requires_grad_(True)
    return names


def merge_lora_modules(model: nn.Module) -> nn.Module:
    """Return a deep copy of ``model`` with every ``LoRALinear`` replaced by its merged ``nn.Linear``."""
    model = copy.deepcopy(model)
    for n in [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]:
        _set_module(model, n, model.get_submodule(n).merged())
    return model


def trainable_summary(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total, "percent": 100 * trainable / total}
