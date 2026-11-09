"""Reference solutions for notebook 14 · ML from scratch II. Graded by tests/test_solutions.py."""

from __future__ import annotations

import numpy as np


def ex1(x, y):
    """Best threshold on one feature for a regression stump: O(n log n) via sort + prefix sums.

    Returns (threshold, loss) where loss = SSE(left) + SSE(right) and points with
    x <= threshold go left. The threshold is the midpoint between consecutive distinct
    sorted values. With no valid split (n < 2 or x constant) returns (None, SSE(all)).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n == 0:
        return None, 0.0
    total_sse = float(((y - y.mean()) ** 2).sum())
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    s1, s2 = np.cumsum(ys), np.cumsum(ys ** 2)
    n_left = np.arange(1, n)                     # split after position i: left = first i+1 points
    sl, ql = s1[:-1], s2[:-1]
    sr, qr = s1[-1] - sl, s2[-1] - ql
    loss = (ql - sl ** 2 / n_left) + (qr - sr ** 2 / (n - n_left))
    valid = xs[:-1] < xs[1:]                     # cannot split between equal x values
    if not valid.any():
        return None, total_sse
    loss = np.where(valid, loss, np.inf)
    i = int(np.argmin(loss))
    return float((xs[i] + xs[i + 1]) / 2), float(max(loss[i], 0.0))


def _stump_predict(stump, X):
    f, t, left, right = stump
    return np.where(X[:, f] <= t, left, right)


def ex2(X, y, n_estimators=100, lr=0.1):
    """Gradient boosting with depth-1 stumps for squared loss.

    Model: {"init": mean(y), "lr": lr, "stumps": [(feature, threshold, left_value, right_value), ...]}
    predict(X) = init + lr * sum_m stump_m(X).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    init = float(y.mean())
    F = np.full(y.size, init)
    stumps = []
    for _ in range(n_estimators):
        r = y - F                                   # negative gradient of 1/2 (y - F)^2
        best = None
        for f in range(X.shape[1]):
            t, loss = ex1(X[:, f], r)
            if t is not None and (best is None or loss < best[0]):
                best = (loss, f, t)
        if best is None:
            break
        _, f, t = best
        left = X[:, f] <= t
        stump = (f, t, float(r[left].mean()), float(r[~left].mean()))
        stumps.append(stump)
        F += lr * _stump_predict(stump, X)
    return {"init": init, "lr": lr, "stumps": stumps}


def gbm_predict(model, X):
    X = np.asarray(X, dtype=float)
    out = np.full(X.shape[0], model["init"])
    for stump in model["stumps"]:
        out += model["lr"] * _stump_predict(stump, X)
    return out


def ex3(params, X, y):
    """Loss and gradients of a 2-layer MLP: X -> Linear -> ReLU -> Linear -> softmax cross-entropy."""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    z1 = X @ W1 + b1
    h = np.maximum(z1, 0.0)
    logits = h @ W2 + b2
    logits = logits - logits.max(axis=1, keepdims=True)
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(n), y].mean()
    dlogits = np.exp(log_probs)
    dlogits[np.arange(n), y] -= 1.0
    dlogits /= n
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(axis=0)
    dh = dlogits @ W2.T
    dz1 = dh * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return float(loss), {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def ex4(Q, K, V, mask=None):
    """Scaled dot-product attention. mask: boolean, True = may attend; broadcastable to (..., Lq, Lk).

    Rows with no allowed key get zero weights and a zero output (not NaN).
    Returns (output (..., Lq, dv), weights (..., Lq, Lk)).
    """
    Q, K, V = (np.asarray(a, dtype=float) for a in (Q, K, V))
    d = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d)
    if mask is not None:
        mask = np.broadcast_to(np.asarray(mask, dtype=bool), scores.shape)
        scores = np.where(mask, scores, -np.inf)
    row_max = scores.max(axis=-1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)     # fully masked row
    e = np.exp(scores - row_max)                               # exp(-inf) = 0
    denom = e.sum(axis=-1, keepdims=True)
    weights = np.divide(e, denom, out=np.zeros_like(e), where=denom > 0)
    return weights @ V, weights


def ex5(x, gamma, beta, running_mean, running_var, training, momentum=0.1, eps=1e-5):
    """BatchNorm1d forward (PyTorch conventions). Returns (out, new_running_mean, new_running_var).

    Training: normalise with the batch mean and BIASED batch variance; update running
    stats with the UNBIASED variance: new = (1 - momentum) * old + momentum * batch.
    Eval: normalise with the running stats, which are returned unchanged.
    """
    x = np.asarray(x, dtype=float)
    if training:
        n = x.shape[0]
        if n < 2:
            raise ValueError("batch norm in training mode needs more than one example per batch")
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        out = gamma * (x - mean) / np.sqrt(var + eps) + beta
        new_mean = (1 - momentum) * running_mean + momentum * mean
        new_var = (1 - momentum) * running_var + momentum * var * n / (n - 1)
        return out, new_mean, new_var
    out = gamma * (x - running_mean) / np.sqrt(running_var + eps) + beta
    return out, np.array(running_mean, dtype=float), np.array(running_var, dtype=float)


def ex6(x, p, training, rng):
    """Inverted dropout: zero each element with probability p, scale survivors by 1/(1-p)."""
    x = np.asarray(x, dtype=float)
    if not training or p == 0:
        return x.copy()
    if p >= 1:
        return np.zeros_like(x)
    keep = rng.random(x.shape) >= p
    return np.where(keep, x / (1.0 - p), 0.0)
