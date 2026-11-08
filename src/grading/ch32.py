"""Self-checks for Chapter 32 · ML from scratch II (NumPy only)."""

from __future__ import annotations

import numpy as np

from churnlab.checks import run_check
from shoprank.drills import check_scaling, numerical_gradient, relative_error


def _split_sse(x, y, t):
    left = x <= t
    sse = 0.0
    for part in (y[left], y[~left]):
        if part.size:
            sse += float(((part - part.mean()) ** 2).sum())
    return sse


def _ref_best_loss(x, y):
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    n = xs.size
    s1, s2 = np.cumsum(ys), np.cumsum(ys ** 2)
    nl = np.arange(1, n)
    loss = (s2[:-1] - s1[:-1] ** 2 / nl) + ((s2[-1] - s2[:-1]) - (s1[-1] - s1[:-1]) ** 2 / (n - nl))
    loss = np.where(xs[:-1] < xs[1:], loss, np.inf)
    return float(loss.min()), xs


def ex1(best_split):
    def test():
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = int(rng.integers(2, 40))
            x = rng.integers(0, 8, n).astype(float)          # many duplicate x values on purpose
            y = rng.normal(size=n) + (x > 3) * 2
            x0 = x.copy()
            t, loss = best_split(x, y)
            assert np.array_equal(x, x0), "your function modified x in place"
            if np.unique(x).size < 2:
                assert t is None, f"x is constant ({x[0]}): no valid split, expected (None, SSE)"
                assert abs(loss - _split_sse(x, y, np.inf)) < 1e-8, "with no split, loss is the total SSE"
                continue
            ref, xs = _ref_best_loss(x, y)
            assert t is not None, f"x={x.tolist()} has a valid split but you returned None"
            uniq = np.unique(xs)
            mids = (uniq[:-1] + uniq[1:]) / 2
            assert np.isclose(mids, t).any(), (f"threshold {t} is not a midpoint between consecutive distinct x "
                                               f"values {uniq.tolist()}")
            assert abs(_split_sse(x, y, t) - loss) < 1e-6 * max(1, loss), (
                f"returned loss {loss:.6f} is not the SSE of splitting at {t} ({_split_sse(x, y, t):.6f})")
            assert abs(loss - ref) < 1e-6 * max(1, ref), f"loss {loss:.6f} is not the minimum {ref:.6f}"
        t, loss = best_split(np.array([1.0, 1, 2, 2]), np.array([0.0, 10, 0, 10]))
        assert t == 1.5 and abs(loss - 100) < 1e-9, f"x=[1,1,2,2], y=[0,10,0,10]: expected (1.5, 100.0), got {(t, loss)}"
        t, loss = best_split(np.array([5.0]), np.array([3.0]))
        assert t is None and loss == 0, "a single point has no split: (None, 0.0)"

        def make(n):
            r = np.random.default_rng(n)
            x = r.normal(size=n)
            return x, np.sin(3 * x) + r.normal(size=n)

        def validate(n, args, result):
            ref, _ = _ref_best_loss(*args)
            assert abs(result[1] - ref) < 1e-6 * max(1, ref), f"wrong minimum loss at n={n}"

        check_scaling(best_split, make, [1000, 4000, 16000, 64000, 256000], budget=2.0, label="best_split",
                      validate=validate)

    return run_check("Exercise 1 · best_split", test,
                     hint="sort once; prefix sums S1=cumsum(y), S2=cumsum(y^2) give SSE of every left/right part "
                          "in O(1): SSE = S2 - S1^2/n. Skip positions where x[i] == x[i+1]")


def _grader_predict(model, X):
    out = np.full(X.shape[0], float(model["init"]))
    for f, t, left, right in model["stumps"]:
        out += model["lr"] * np.where(X[:, f] <= t, left, right)
    return out


def ex2(gbm_fit, gbm_predict=None):
    def test():
        from sklearn.ensemble import GradientBoostingRegressor

        rng = np.random.default_rng(1)
        X = rng.uniform(-2, 2, size=(200, 3))
        y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.1 * rng.normal(size=200)
        model = gbm_fit(X, y, n_estimators=50, lr=0.1)
        assert set(model) >= {"init", "lr", "stumps"}, "model must be a dict with keys 'init', 'lr', 'stumps'"
        assert len(model["stumps"]) == 50, f"expected 50 stumps, got {len(model['stumps'])}"
        assert abs(model["init"] - y.mean()) < 1e-9, "init should be mean(y) (the best constant for squared loss)"
        mses = []
        for m in range(0, 51, 5):
            partial = {**model, "stumps": model["stumps"][:m]}
            mses.append(float(np.mean((y - _grader_predict(partial, X)) ** 2)))
        assert all(a >= b - 1e-12 for a, b in zip(mses, mses[1:])), f"training MSE must not increase: {np.round(mses, 4)}"
        ref = GradientBoostingRegressor(max_depth=1, n_estimators=50, learning_rate=0.1, random_state=0).fit(X, y)
        ref_mse = float(np.mean((y - ref.predict(X)) ** 2))
        assert abs(mses[-1] - ref_mse) <= 0.01 * ref_mse, (
            f"training MSE {mses[-1]:.4f} vs sklearn GradientBoostingRegressor(max_depth=1) {ref_mse:.4f}")
        X_new = rng.uniform(-2, 2, size=(50, 3))
        diff = np.max(np.abs(_grader_predict(model, X_new) - ref.predict(X_new)))
        assert diff < 1e-3, f"predictions on new points differ from sklearn by up to {diff:.4f}"
        if gbm_predict is not None:
            assert np.allclose(gbm_predict(model, X_new), _grader_predict(model, X_new)), (
                "gbm_predict disagrees with init + lr * sum(stump outputs)")
        Xc = np.column_stack([np.ones(30), rng.normal(size=30)])        # a constant column must be skipped
        mc = gbm_fit(Xc, Xc[:, 1] * 2, n_estimators=5, lr=0.5)
        assert all(s[0] == 1 for s in mc["stumps"]), "a constant feature can never be split on"

    return run_check("Exercise 2 · gbm_fit", test,
                     hint="F = mean(y); repeat: r = y - F; best stump over features fitted to r (reuse best_split); "
                          "leaf values = mean residual on each side; F += lr * stump(X)")


def _mlp_params(rng, d=5, h=7, c=3):
    return {"W1": rng.normal(scale=0.5, size=(d, h)), "b1": rng.normal(scale=0.1, size=h),
            "W2": rng.normal(scale=0.5, size=(h, c)), "b2": rng.normal(scale=0.1, size=c)}


def ex3(mlp_grads):
    def test():
        rng = np.random.default_rng(2)
        params = _mlp_params(rng)
        X = rng.normal(size=(8, 5))
        y = rng.integers(0, 3, 8)
        snapshot = {k: v.copy() for k, v in params.items()}
        loss, grads = mlp_grads(params, X, y)
        assert all(np.array_equal(params[k], snapshot[k]) for k in params), "your function modified params in place"
        h = np.maximum(X @ params["W1"] + params["b1"], 0)
        z = h @ params["W2"] + params["b2"]
        ref = float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(8), y]))
        assert abs(loss - ref) < 1e-8, f"loss {loss:.6f} != reference {ref:.6f}"
        assert set(grads) == set(params), f"grads must have keys {sorted(params)}"
        for name in params:
            g = np.asarray(grads[name], dtype=float)
            assert g.shape == params[name].shape, f"grads[{name!r}] has shape {g.shape}, expected {params[name].shape}"

            def f(v, name=name):
                return mlp_grads({**params, name: v}, X, y)[0]

            rel = relative_error(g, numerical_gradient(f, params[name].copy()))
            assert rel < 1e-6, f"gradient check failed for {name}: relative error {rel:.2e}"
        big = {**params, "W2": params["W2"] * 1e3}
        lb, gb = mlp_grads(big, X, y)
        assert np.isfinite(lb) and all(np.all(np.isfinite(v)) for v in gb.values()), (
            "NaN/inf for large logits: use log-sum-exp in the loss")

    return run_check("Exercise 3 · mlp_grads", test,
                     hint="dlogits = (softmax - onehot)/n; dW2 = h.T@dlogits; db2 = sum; dh = dlogits@W2.T; "
                          "dz1 = dh * (z1 > 0); dW1 = X.T@dz1; db1 = sum")


def _ref_attention(Q, K, V, mask):
    s = Q @ np.swapaxes(K, -1, -2) / np.sqrt(Q.shape[-1])
    m = np.ones(s.shape, bool) if mask is None else np.broadcast_to(mask, s.shape)
    w = np.zeros_like(s)
    it = np.ndindex(*s.shape[:-1])
    for idx in it:
        row, allowed = s[idx], m[idx]
        if allowed.any():
            e = np.exp(row[allowed] - row[allowed].max())
            w[idx][allowed] = e / e.sum()
    return w @ V, w


def ex4(attention):
    def test():
        rng = np.random.default_rng(3)
        Q, K, V = rng.normal(size=(2, 3, 4, 8)), rng.normal(size=(2, 3, 5, 8)), rng.normal(size=(2, 3, 5, 6))
        causal = np.tril(np.ones((4, 5), bool))
        padding = np.ones((2, 1, 1, 5), bool)
        padding[1, ..., 3:] = False
        dead = np.ones((4, 5), bool)
        dead[2] = False                                   # a query that may attend to nothing
        for name, mask in [("no mask", None), ("causal (4,5)", causal), ("padding (2,1,1,5)", padding),
                           ("a fully masked row", dead)]:
            out, w = attention(Q, K, V, mask)
            ref_out, ref_w = _ref_attention(Q, K, V, mask)
            assert np.shape(out) == ref_out.shape, f"{name}: output shape {np.shape(out)}, expected {ref_out.shape}"
            assert np.all(np.isfinite(out)) and np.all(np.isfinite(w)), f"{name}: NaN/inf in the output or weights"
            assert np.allclose(w, ref_w, atol=1e-10), f"{name}: attention weights differ from the reference"
            assert np.allclose(out, ref_out, atol=1e-10), f"{name}: output differs from the reference"
        out, w = attention(Q, K, V, dead)
        assert np.all(np.asarray(w)[..., 2, :] == 0) and np.all(np.asarray(out)[..., 2, :] == 0), (
            "a query with no allowed key must get all-zero weights and a zero output")
        out, w = attention(Q * 100, K * 100, V, causal)
        assert np.all(np.isfinite(out)), "scores in the thousands overflow: subtract the row max before exp"
        q2, k2, v2 = rng.normal(size=(3, 4)), rng.normal(size=(6, 4)), rng.normal(size=(6, 2))
        assert np.allclose(attention(q2, k2, v2, None)[0], _ref_attention(q2, k2, v2, None)[0]), "2-D inputs must work"

    return run_check("Exercise 4 · attention", test,
                     hint="scores = Q @ K^T / sqrt(d); set disallowed scores to -inf; subtract the (finite) row max; "
                          "exp; divide by the row sum where it is > 0, else 0")


def ex5(batchnorm_forward):
    def test():
        rng = np.random.default_rng(4)
        x = rng.normal(loc=3, scale=2, size=(64, 5))
        gamma, beta = rng.uniform(0.5, 2, 5), rng.normal(size=5)
        rm, rv = np.zeros(5), np.ones(5)
        snap = (x.copy(), rm.copy(), rv.copy())
        out, nm, nv = batchnorm_forward(x, gamma, beta, rm, rv, training=True, momentum=0.1, eps=1e-5)
        assert all(np.array_equal(a, b) for a, b in zip((x, rm, rv), snap)), "inputs were modified in place; return new arrays"
        ref = gamma * (x - x.mean(0)) / np.sqrt(x.var(0) + 1e-5) + beta
        assert np.allclose(out, ref), "training output: normalise with the batch mean and the BIASED batch variance"
        assert np.allclose(nm, 0.9 * rm + 0.1 * x.mean(0)), "running mean: (1-momentum)*old + momentum*batch_mean"
        if np.allclose(nv, 0.9 * rv + 0.1 * x.var(0)):
            raise AssertionError("running var uses the biased variance; PyTorch stores the UNBIASED one (n-1)")
        assert np.allclose(nv, 0.9 * rv + 0.1 * x.var(0, ddof=1)), "running var: (1-momentum)*old + momentum*unbiased_var"
        x1 = x[:1]
        ev, em, evv = batchnorm_forward(x1, gamma, beta, nm, nv, training=False)
        assert np.allclose(ev, gamma * (x1 - nm) / np.sqrt(nv + 1e-5) + beta), "eval output must use the running stats"
        assert np.allclose(em, nm) and np.allclose(evv, nv), "eval mode must not change the running stats"
        try:
            batchnorm_forward(x1, gamma, beta, nm, nv, training=True)
        except ValueError:
            pass
        else:
            raise AssertionError("training with a batch of 1 should raise ValueError (the batch variance is 0)")
        m, v = np.zeros(5), np.ones(5)
        for _ in range(200):
            _, m, v = batchnorm_forward(rng.normal(loc=3, scale=2, size=(64, 5)), gamma, beta, m, v, training=True)
        assert np.allclose(m, 3, atol=0.3) and np.allclose(v, 4, atol=0.8), (
            f"after 200 batches running stats should approach mean 3, var 4; got {np.round(m, 2)}, {np.round(v, 2)}")

    return run_check("Exercise 5 · batchnorm_forward", test,
                     hint="training: mu, var = x.mean(0), x.var(0); out = gamma*(x-mu)/sqrt(var+eps)+beta; "
                          "running stats use var*n/(n-1). eval: use the running stats")


def ex6(inverted_dropout):
    def test():
        x = np.full((1000, 100), 3.0)
        x0 = x.copy()
        out = np.asarray(inverted_dropout(x, 0.3, True, np.random.default_rng(0)))
        assert np.array_equal(x, x0), "your function modified x in place"
        assert out.shape == x.shape, f"output shape {out.shape} != input shape {x.shape}"
        vals = np.unique(np.round(out, 9))
        assert set(vals.tolist()) <= {0.0, round(3 / 0.7, 9)}, (
            f"values should be 0 or x/(1-p) = {3 / 0.7:.4f}; found {vals[:5]}")
        frac = float((out == 0).mean())
        assert abs(frac - 0.3) < 0.01, f"dropped fraction {frac:.3f}, expected about p = 0.3"
        assert abs(out.mean() - 3.0) < 0.05, f"mean {out.mean():.3f}: inverted dropout preserves the expectation (3.0)"
        a = inverted_dropout(x, 0.3, True, np.random.default_rng(7))
        b = inverted_dropout(x, 0.3, True, np.random.default_rng(7))
        assert np.array_equal(a, b), "use the rng you are given, so the same seed gives the same mask"
        assert not np.array_equal(a, out), "different rngs should give different masks"
        assert np.array_equal(inverted_dropout(x, 0.3, False, np.random.default_rng(0)), x), "eval mode: identity"
        assert np.array_equal(inverted_dropout(x, 0.0, True, np.random.default_rng(0)), x), "p = 0: identity"
        z = np.asarray(inverted_dropout(x, 1.0, True, np.random.default_rng(0)))
        assert np.all(z == 0) and np.all(np.isfinite(z)), "p = 1: all zeros (and no division by zero)"

    return run_check("Exercise 6 · inverted_dropout", test,
                     hint="keep = rng.random(x.shape) >= p; out = where(keep, x/(1-p), 0); handle eval, p=0 and p=1 first")
