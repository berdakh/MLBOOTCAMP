"""Self-checks for notebook 13 · ML from scratch I (NumPy only).

References are scikit-learn or closed forms; inputs are small, synthetic and
seeded, so every check is deterministic and takes at most a few seconds.
"""

from __future__ import annotations

import numpy as np

from churnlab.checks import run_check

from ..drills import check_scaling, numerical_gradient, relative_error
from ..metrics import ndcg_at_k


def _logreg_data(n=400, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w_true = np.array([1.5, -2.0, 0.5, 0.0][:d])
    p = 1 / (1 + np.exp(-(X @ w_true + 0.3)))
    return X, (rng.random(n) < p).astype(int)


def _logloss(X, y, w, b, l2):
    z = X @ w + b
    return float(np.mean(np.logaddexp(0, z) - y * z) + 0.5 * l2 * w @ w)


def ex1(fit_logreg):
    def test():
        from sklearn.linear_model import LogisticRegression

        X, y = _logreg_data()
        X0, y0 = X.copy(), y.copy()
        l2 = 0.01
        w, b = fit_logreg(X, y, lr=0.5, epochs=3000, l2=l2)
        w = np.asarray(w, dtype=float)
        assert w.shape == (X.shape[1],), f"w should have shape ({X.shape[1]},), got {w.shape}"
        assert np.ndim(b) == 0, "b should be a scalar (the intercept)"
        assert np.array_equal(X, X0) and np.array_equal(y, y0), "your function modified X or y in place"
        ref = LogisticRegression(C=1.0 / (len(y) * l2), tol=1e-10, max_iter=10_000).fit(X, y)
        err = np.max(np.abs(np.r_[w, b] - np.r_[ref.coef_[0], ref.intercept_[0]]))
        assert err < 0.02, (f"coefficients differ from sklearn LogisticRegression(C=1/(n*l2)) by {err:.3f}; "
                            f"yours {np.round(np.r_[w, b], 3)}, sklearn "
                            f"{np.round(np.r_[ref.coef_[0], ref.intercept_[0]], 3)}")
        losses = [_logloss(X, y, *fit_logreg(X, y, lr=0.5, epochs=e, l2=l2), l2) for e in (1, 10, 100)]
        assert losses[0] > losses[1] > losses[2], f"loss should decrease with more epochs, got {losses}"
        # no L2: a separable feature must not produce NaN (stable sigmoid / no log(0) in the update)
        Xs = np.array([[-3.0], [-1.0], [1.0], [3.0]]) * 100
        ws, bs = fit_logreg(Xs, np.array([0, 0, 1, 1]), lr=1.0, epochs=50, l2=0.0)
        assert np.all(np.isfinite(np.r_[np.asarray(ws, float), bs])), "NaN/inf on separable, large-scale data"
        # all one class: still finite, intercept positive
        w1, b1 = fit_logreg(X[:50], np.ones(50, dtype=int), lr=0.5, epochs=200, l2=l2)
        assert np.isfinite(b1) and b1 > 0, "with all labels = 1 the intercept should be finite and positive"

    return run_check("Exercise 1 · fit_logreg", test,
                     hint="grad_w = X.T @ (sigmoid(Xw+b) - y) / n + l2*w ; grad_b = mean(p - y); "
                          "sklearn's C equals 1/(n*l2) for this objective")


def ex2(softmax_grad):
    def test():
        rng = np.random.default_rng(1)
        n, d, c = 12, 5, 4
        X = rng.normal(size=(n, d))
        y = rng.integers(0, c, size=n)
        W = rng.normal(scale=0.5, size=(d, c))
        W0 = W.copy()
        loss, grad = softmax_grad(W, X, y)
        assert np.array_equal(W, W0), "your function modified W in place"
        grad = np.asarray(grad, dtype=float)
        assert grad.shape == W.shape, f"gradient should have W's shape {W.shape}, got {grad.shape}"
        z = X @ W
        ref_loss = float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(n), y]))
        assert abs(loss - ref_loss) < 1e-8, f"loss {loss:.6f} != mean cross-entropy {ref_loss:.6f}"
        num = numerical_gradient(lambda w: softmax_grad(w, X, y)[0], W.copy())
        rel = relative_error(grad, num)
        assert rel < 1e-6, f"gradient check failed: relative error {rel:.2e} (want < 1e-6)"
        # numerical stability: logits in the thousands
        big_loss, big_grad = softmax_grad(W * 1000, X, y)
        assert np.isfinite(big_loss) and np.all(np.isfinite(big_grad)), (
            "loss/gradient is NaN or inf for large logits - subtract the row max before exp (log-sum-exp)")
        zb = X @ (W * 1000)
        m = zb.max(1)
        ref_big = float(np.mean(m + np.log(np.exp(zb - m[:, None]).sum(1)) - zb[np.arange(n), y]))
        assert abs(big_loss - ref_big) < 1e-6 * max(1, abs(ref_big)), "loss is wrong for large logits"

    return run_check("Exercise 2 · softmax_grad", test,
                     hint="probs = softmax(XW) (subtract the row max first); grad = X.T @ (probs - onehot(y)) / n")


def _blobs(seed=0, k=5, per=80, d=2):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-20, 20, size=(k, d))
    return np.vstack([c + rng.normal(size=(per, d)) for c in centers])


def ex3(kmeans):
    def test():
        from sklearn.cluster import KMeans

        X = _blobs()
        X0 = X.copy()
        centers, labels = kmeans(X, 5, seed=0)
        centers, labels = np.asarray(centers, float), np.asarray(labels)
        assert np.array_equal(X, X0), "your function modified X in place"
        assert centers.shape == (5, 2), f"centers should be (k, d) = (5, 2), got {centers.shape}"
        assert labels.shape == (len(X),), f"labels should have shape ({len(X)},), got {labels.shape}"
        d2 = ((X[:, None, :] - centers[None]) ** 2).sum(-1)
        assert np.array_equal(labels, d2.argmin(1)), "labels must be each point's nearest returned center"
        ref = KMeans(5, n_init=10, random_state=0).fit(X).inertia_
        inertias = []
        for seed in range(5):
            c, lab = kmeans(X, 5, seed=seed)
            inertias.append(float(((X - np.asarray(c)[np.asarray(lab)]) ** 2).sum()))
        assert np.median(inertias) <= 1.02 * ref, (
            f"median inertia over seeds 0-4 is {np.median(inertias):.1f}, sklearn finds {ref:.1f} - "
            "is your init really k-means++ (sample proportional to squared distance)?")
        c2, _ = kmeans(X, 5, seed=3)
        assert np.allclose(c2, kmeans(X, 5, seed=3)[0]), "same seed must give the same result"
        c1, l1 = kmeans(X, 1, seed=0)
        assert np.allclose(np.asarray(c1)[0], X.mean(0)) and np.all(np.asarray(l1) == 0), "k=1 must give the mean"
        Xd = np.repeat(np.array([[0.0, 0.0], [5.0, 5.0]]), 10, axis=0)   # duplicates, 2 distinct points
        cd, ld = kmeans(Xd, 2, seed=0)
        assert np.all(np.isfinite(cd)), "NaN centers on duplicated points (empty cluster?)"
        assert len(set(np.asarray(ld).tolist())) == 2, "two distinct duplicated points should form two clusters"

    return run_check("Exercise 3 · kmeans", test,
                     hint="k-means++: first center uniform, each next center sampled with p ∝ D(x)^2 to the "
                          "nearest chosen center; then Lloyd: assign, recompute means, stop when labels stop changing")


def _knn_ref(Xtr, ytr, Xte, k):
    d2 = ((Xte[:, None, :] - Xtr[None]) ** 2).sum(-1)
    nn = np.argsort(d2, axis=1, kind="stable")[:, :k]
    out = []
    for row in ytr[nn]:
        vals, counts = np.unique(row, return_counts=True)
        out.append(vals[counts.argmax()])
    return np.array(out)


def ex4(knn_predict):
    def test():
        rng = np.random.default_rng(2)
        Xtr = rng.normal(size=(200, 3))
        ytr = (Xtr[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(int) + 2 * (Xtr[:, 1] > 1)
        Xte = rng.normal(size=(60, 3))
        for k in (1, 4, 7):
            got = np.asarray(knn_predict(Xtr, ytr, Xte, k))
            ref = _knn_ref(Xtr, ytr, Xte, k)
            assert got.shape == (60,), f"expected 60 predictions, got shape {got.shape}"
            bad = int((got != ref).sum())
            assert bad == 0, f"k={k}: {bad} of 60 predictions differ from a brute-force reference"
        assert np.array_equal(np.asarray(knn_predict(Xtr, ytr, Xtr, 1)), ytr), "k=1 on the training set must return y_train"
        # vote ties go to the smallest label; string labels must work
        Xt = np.array([[0.0], [1.0], [10.0], [11.0]])
        assert list(knn_predict(Xt, np.array(["b", "a", "b", "a"]), np.array([[0.4]]), 2)) == ["a"], (
            "a 1-1 vote tie should go to the smallest label ('a')")
        assert list(knn_predict(Xt, np.array([5, 5, 7, 7]), np.array([[100.0]]), 4)) == [5], "k = n_train must work"

        def make(n):
            r = np.random.default_rng(n)
            return r.normal(size=(n, 16)), r.integers(0, 5, n), r.normal(size=(n // 4, 16)), 5

        check_scaling(knn_predict, make, [400, 1600, 4000], budget=2.0, label="knn_predict")

    return run_check("Exercise 4 · knn_predict", test,
                     hint="squared distances for all pairs at once: |a|^2 - 2 a·b + |b|^2; np.argpartition for "
                          "the k nearest; count votes with np.add.at or np.bincount")


def ex5(roc_auc):
    def test():
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 300)
        s = y + rng.normal(size=300)
        assert abs(roc_auc(y, s) - roc_auc_score(y, s)) < 1e-12, "AUC differs from sklearn on continuous scores"
        s_ties = np.round(s)                       # heavy ties
        assert abs(roc_auc(y, s_ties) - roc_auc_score(y, s_ties)) < 1e-12, (
            f"with tied scores you give {roc_auc(y, s_ties):.4f}, sklearn {roc_auc_score(y, s_ties):.4f} - "
            "a tie between a positive and a negative counts 1/2")
        assert roc_auc([0, 1], [0.5, 0.5]) == 0.5, "all scores tied must give 0.5"
        assert roc_auc([0, 0, 1], [3, 2, 1]) == 0.0 and roc_auc([0, 1, 1], [1, 2, 3]) == 1.0
        assert abs(roc_auc(y.astype(bool), s) - roc_auc_score(y, s)) < 1e-12, "boolean labels must work"
        for bad in ([1, 1, 1], [0, 0]):
            try:
                roc_auc(bad, list(range(len(bad))))
            except ValueError:
                pass
            else:
                raise AssertionError("only one class present: raise ValueError (AUC is undefined)")

        def make(n):
            r = np.random.default_rng(n)
            yy = r.integers(0, 2, n)
            return yy, yy + r.normal(size=n)

        def validate(n, args, result):
            assert abs(result - roc_auc_score(*args)) < 1e-9, f"wrong AUC at n={n}"

        check_scaling(roc_auc, make, [1000, 4000, 16000, 64000, 256000], budget=2.0, label="roc_auc",
                      validate=validate)

    return run_check("Exercise 5 · roc_auc", test,
                     hint="AUC = (sum of the positives' ranks - n_pos(n_pos+1)/2) / (n_pos*n_neg), with tied "
                          "scores given their AVERAGE rank (scipy.stats.rankdata does this; try it without)")


def ex6(ndcg_at_k_fn):
    def test():
        rng = np.random.default_rng(4)
        for _ in range(30):
            n = int(rng.integers(1, 15))
            labels = rng.integers(0, 4, n)
            scores = rng.integers(0, 5, n).astype(float)      # ties on purpose
            k = int(rng.integers(1, 12))
            ref = ndcg_at_k(labels[np.argsort(-scores, kind="stable")], k)
            got = ndcg_at_k_fn(scores, labels, k)
            assert abs(got - ref) < 1e-12, (f"scores={scores.tolist()}, labels={labels.tolist()}, k={k}: "
                                           f"you give {got:.6f}, shoprank.metrics gives {ref:.6f}")
        assert ndcg_at_k_fn([3, 2, 1], [0, 0, 0], 3) == 0.0, "no relevant item: NDCG is 0 (by convention)"
        assert abs(ndcg_at_k_fn([1, 2, 3], [3, 2, 1], 10) - ndcg_at_k([1, 2, 3], 10)) < 1e-12, "k > n must work"
        assert ndcg_at_k_fn([0.1, 0.9], [0, 2], 1) == 1.0

    return run_check("Exercise 6 · ndcg_at_k", test,
                     hint="order = np.argsort(-scores, kind='stable'); DCG = sum((2**rel - 1) / log2(rank + 1)) over "
                          "the top k; divide by the DCG of the labels sorted descending")
