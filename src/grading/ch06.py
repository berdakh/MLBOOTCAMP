"""Self-checks for Chapter 6 · Pipelines & cross-validation."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.metrics import average_precision_score

from churnlab.checks import close, run_check
from churnlab.features import CATEGORICAL, FEATURES, LABEL, NUMERIC


def _train_valid():
    from churnlab.bootstrap import ensure_split

    split = ensure_split(verbose=False)
    return split.train, split.valid


def ex1(build_preprocessor):
    def test():
        train, valid = _train_valid()
        train = train.iloc[:3000]
        valid = valid.iloc[:500].copy()
        pre = build_preprocessor(list(NUMERIC), list(CATEGORICAL))
        assert isinstance(pre, ColumnTransformer), f"expected a ColumnTransformer, got {type(pre).__name__}"
        try:
            pre.transform(train[FEATURES])
        except NotFittedError:
            pass
        else:
            raise AssertionError("return an UNFITTED transformer: fitting is the caller's job, on the train slice only")
        pre = clone(pre).fit(train[FEATURES])
        out_train = pre.transform(train[FEATURES])
        out_valid = pre.transform(valid[FEATURES])
        dense = lambda a: a.toarray() if hasattr(a, "toarray") else np.asarray(a, dtype=float)  # noqa: E731
        out_train, out_valid = dense(out_train), dense(out_valid)
        assert not np.isnan(out_train).any(), "transformed TRAIN still contains NaN - impute numeric columns"
        assert not np.isnan(out_valid).any(), "transformed VALID still contains NaN - impute numeric columns"
        assert out_train.shape[1] == out_valid.shape[1], "train and valid outputs have different widths"
        assert out_train.shape[1] > len(NUMERIC), "categorical columns do not seem to be one-hot encoded"
        unseen = valid.copy()
        unseen["region"] = "Antarctica"
        unseen["plan"] = "platinum-plus"
        out_unseen = dense(pre.transform(unseen[FEATURES]))
        assert out_unseen.shape[1] == out_train.shape[1], "an unseen category changed the output width"
        names = pre.get_feature_names_out()
        assert len(names) == out_train.shape[1], "get_feature_names_out() does not match the output width"
        # Scaling was learned from the train slice only: train numeric means ~ 0.
        num_idx = [i for i, n in enumerate(names) if any(n.endswith("__" + c) for c in NUMERIC)]
        assert num_idx, "could not find the numeric output columns by name"
        means = out_train[:, num_idx].mean(axis=0)
        assert np.all(np.abs(means) < 1e-6), "numeric columns are not standardised on the data they were fitted on"

    return run_check("Exercise 1 · build_preprocessor", test,
                     hint="ColumnTransformer([('num', Pipeline([impute, scale]), numeric), "
                          "('cat', OneHotEncoder(handle_unknown='ignore'), categorical)])")


def _reference_scores(make_model_fn, df, n_folds):
    from churnlab.splits import rolling_origin_folds

    scores = []
    for tr, va in rolling_origin_folds(df, n_folds=n_folds):
        model = make_model_fn().fit(df.iloc[tr][FEATURES], df.iloc[tr][LABEL])
        p = model.predict_proba(df.iloc[va][FEATURES])[:, 1]
        scores.append(float(average_precision_score(df.iloc[va][LABEL], p)))
    return scores


def ex2(rolling_cv_scores):
    def test():
        from churnlab.modeling import make_model

        train, _ = _train_valid()
        # A shuffled copy with a non-default index: positions, not labels, must be used.
        df = train.sample(frac=1.0, random_state=1)
        df.index = df.index + 100_000
        make = lambda: make_model("logreg")  # noqa: E731
        got = rolling_cv_scores(make, df, 3)
        got = [float(x) for x in got]
        assert len(got) == 3, f"expected 3 fold scores for n_folds=3, got {len(got)}"
        ref = _reference_scores(make, df, 3)
        for k, (g, r) in enumerate(zip(got, ref)):
            assert close(g, r, 1e-4), f"fold {k}: got PR-AUC {g:.4f}, expected {r:.4f}"
        got2 = rolling_cv_scores(make, df, 2)
        assert len(got2) == 2, "n_folds was ignored"

    return run_check("Exercise 2 · rolling_cv_scores", test,
                     hint="for tr, va in rolling_origin_folds(df, n_folds=n_folds): fit on df.iloc[tr], "
                          "score average_precision on df.iloc[va]")

