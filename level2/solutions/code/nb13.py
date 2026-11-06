"""Reference solutions for notebook 13 · ML from scratch I. Graded by tests/test_solutions.py."""

from __future__ import annotations

import numpy as np


def _sigmoid(z):
    # Stable for large |z|: never exponentiates a large positive number.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def ex1(X, y, lr=0.1, epochs=1000, l2=0.0):
    """Logistic regression, batch gradient descent on mean log-loss + (l2/2)||w||^2."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        p = _sigmoid(X @ w + b)
        err = p - y                       # dLoss/dlogit, per example
        grad_w = X.T @ err / n + l2 * w   # the intercept is not regularised
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, float(b)


def ex2(W, X, y):
    """Mean softmax cross-entropy and its gradient wrt W (d, C)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = X.shape[0]
    logits = X @ W
    logits = logits - logits.max(axis=1, keepdims=True)          # log-sum-exp trick
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(n), y].mean()
    probs = np.exp(log_probs)
    probs[np.arange(n), y] -= 1.0                                  # softmax - one_hot
    grad = X.T @ probs / n
    return float(loss), grad


def ex3(X, k, seed=0, n_iter=100):
    """k-means with k-means++ initialisation. Returns (centers (k, d), labels (n,))."""
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centers = [X[rng.integers(n)]]
    d2 = ((X - centers[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        total = d2.sum()
        idx = rng.integers(n) if total == 0 else rng.choice(n, p=d2 / total)
        centers.append(X[idx])
        d2 = np.minimum(d2, ((X - X[idx]) ** 2).sum(axis=1))
    centers = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for it in range(n_iter):
        dist = (X ** 2).sum(1)[:, None] - 2 * X @ centers.T + (centers ** 2).sum(1)[None, :]
        new_labels = dist.argmin(axis=1)
        if it > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = X[labels == j]
            if len(members):              # an empty cluster keeps its old center
                centers[j] = members.mean(axis=0)
    dist = (X ** 2).sum(1)[:, None] - 2 * X @ centers.T + (centers ** 2).sum(1)[None, :]
    return centers, dist.argmin(axis=1)


def ex4(X_train, y_train, X_test, k):
    """Vectorised kNN majority vote; ties between labels go to the smallest label."""
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)
    d2 = ((X_test ** 2).sum(1)[:, None] - 2 * X_test @ X_train.T
          + (X_train ** 2).sum(1)[None, :])
    nn = np.argpartition(d2, k - 1, axis=1)[:, :k] if k < X_train.shape[0] else \
        np.tile(np.arange(X_train.shape[0]), (X_test.shape[0], 1))
    classes, y_idx = np.unique(y_train, return_inverse=True)
    votes = np.zeros((X_test.shape[0], classes.size), dtype=int)
    np.add.at(votes, (np.arange(X_test.shape[0])[:, None], y_idx[nn]), 1)
    return classes[votes.argmax(axis=1)]          # argmax returns the first (smallest) on ties


def ex5(y_true, scores):
    """ROC-AUC = P(score_pos > score_neg) + 0.5 P(tie), via average ranks (Mann-Whitney U)."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC is undefined when only one class is present")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(s.size)
    # average rank (1-based) for each run of tied scores
    starts = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1]
    ends = np.r_[starts[1:], s.size]
    avg = (starts + ends + 1) / 2.0
    ranks[order] = np.repeat(avg, ends - starts)
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def ex6(scores, labels, k):
    """NDCG@k (gain 2^rel - 1) of the ranking by descending score; ties keep input order."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if labels.size == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")

    def dcg(rel):
        rel = rel[:k]
        return float(((2.0 ** rel - 1.0) / np.log2(np.arange(2, rel.size + 2))).sum())

    ideal = dcg(np.sort(labels)[::-1])
    return dcg(labels[order]) / ideal if ideal > 0 else 0.0
