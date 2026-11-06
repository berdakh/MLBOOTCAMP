"""Making a ranker cheaper to run: ONNX, quantisation, distillation, benchmarks.

Notebook 18 uses these helpers on the pair features of notebook 17
(:data:`shoprank.serving.FEATURES`). Everything is small on purpose: a
10-feature scoring MLP trains in seconds on a CPU, which leaves time to
measure the *trade-offs* instead of waiting for training.

* :class:`RankMLP` - a scoring MLP whose input standardisation is a layer
  (buffers), so the exported ONNX graph contains the preprocessing too.
* :func:`train_mlp` - plain PyTorch loop; with ``teacher_probs`` it distils.
* :func:`distillation_loss` - soft (tempered teacher) + hard (click) BCE.
* :func:`export_onnx`, :func:`ort_session`, :func:`quantize_onnx_int8`,
  :func:`torch_int8` - the export / runtime / quantisation steps.
* :func:`bench` - median milliseconds per batch, single-threaded.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RankMLP(nn.Module):
    """x -> standardise (stored mean/std) -> MLP -> one logit per row."""

    def __init__(self, n_in: int, hidden: tuple[int, ...] = (128, 64), mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(np.zeros(n_in) if mean is None else mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.ones(n_in) if std is None else std, dtype=torch.float32))
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std).squeeze(-1)


def distillation_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor, labels: torch.Tensor,
                      alpha: float = 0.5, T: float = 2.0) -> torch.Tensor:
    """alpha * T^2 * BCE(student/T, sigmoid(teacher_logit/T)) + (1 - alpha) * BCE(student, labels).

    Binary version of Hinton et al.'s knowledge distillation. The T^2 factor
    keeps the soft term's gradient scale comparable across temperatures.
    """
    p = teacher_probs.clamp(1e-6, 1 - 1e-6)
    soft_target = torch.sigmoid(torch.logit(p) / T)
    soft = F.binary_cross_entropy_with_logits(student_logits / T, soft_target) * T * T
    hard = F.binary_cross_entropy_with_logits(student_logits, labels.float())
    return alpha * soft + (1 - alpha) * hard


def train_mlp(X: np.ndarray, y: np.ndarray, hidden=(128, 64), epochs: int = 3, batch_size: int = 1024,
              lr: float = 3e-3, seed: int = 0, teacher_probs: np.ndarray | None = None,
              alpha: float = 0.7, T: float = 2.0, verbose: bool = True) -> RankMLP:
    """Train a RankMLP on clicks (or distil from ``teacher_probs``). CPU, seconds."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = np.asarray(X, np.float32)
    model = RankMLP(X.shape[1], hidden, X.mean(0), X.std(0) + 1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(np.asarray(y, np.float32))
    tt = None if teacher_probs is None else torch.from_numpy(np.asarray(teacher_probs, np.float32))
    for epoch in range(epochs):
        order = torch.from_numpy(rng.permutation(len(X)))
        total = 0.0
        model.train()
        for lo in range(0, len(X), batch_size):
            idx = order[lo:lo + batch_size]
            logits = model(Xt[idx])
            if tt is None:
                loss = F.binary_cross_entropy_with_logits(logits, yt[idx])
            else:
                loss = distillation_loss(logits, tt[idx], yt[idx], alpha, T)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose:
            print(f"  epoch {epoch + 1}/{epochs}  loss {total / len(X):.4f}")
    model.eval()
    return model


@torch.inference_mode()
def torch_scores(model: nn.Module, X: np.ndarray) -> np.ndarray:
    return model(torch.from_numpy(np.asarray(X, np.float32))).numpy()


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def export_onnx(model: nn.Module, path: Path, n_features: int, opset: int = 17) -> Path:
    """Export with a dynamic batch dimension (TorchScript-based exporter, no extra deps)."""
    path = Path(path)
    dummy = torch.zeros(2, n_features)
    kwargs = dict(input_names=["features"], output_names=["logit"], opset_version=opset,
                  dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)
        except TypeError:  # torch < 2.5 has no `dynamo` argument
            torch.onnx.export(model, (dummy,), str(path), **kwargs)
    return path


def ort_session(path: Path, threads: int = 1):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def ort_scores(session, X: np.ndarray) -> np.ndarray:
    return session.run(None, {"features": np.asarray(X, np.float32)})[0]


def quantize_onnx_int8(src: Path, dst: Path) -> Path:
    """Dynamic int8 quantisation of weights (activations quantised on the fly)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return Path(dst)


def torch_int8(model: nn.Module) -> nn.Module:
    """PyTorch dynamic quantisation of the Linear layers (int8 weights, fp32 activations)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch.ao.quantization is being migrated to torchao
        return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def size_kb(path: Path) -> float:
    return Path(path).stat().st_size / 1024


def bench(fn, X: np.ndarray, batch_sizes=(1, 64, 1024), budget_s: float = 0.25, min_reps: int = 5) -> dict:
    """Median milliseconds per call of ``fn(X[:b])`` for each batch size (after a warm-up)."""
    out = {}
    for b in batch_sizes:
        xb = np.ascontiguousarray(X[:b], dtype=np.float32)
        fn(xb)  # warm-up
        times, start = [], time.perf_counter()
        while len(times) < min_reps or time.perf_counter() - start < budget_s:
            t = time.perf_counter()
            fn(xb)
            times.append((time.perf_counter() - t) * 1000)
            if len(times) > 5000:
                break
        out[b] = float(np.median(times))
    return out
