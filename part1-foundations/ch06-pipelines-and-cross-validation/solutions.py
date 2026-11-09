"""Reference solutions for notebook 05. Graded by tests/test_solutions.py."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnlab.features import FEATURES, LABEL
from churnlab.splits import rolling_origin_folds


def ex1(numeric, categorical) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", num_pipe, list(numeric)),
        ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical)),
    ])


def ex2(make_model_fn, df, n_folds: int) -> list[float]:
    scores = []
    for train_idx, valid_idx in rolling_origin_folds(df, n_folds=n_folds):
        train, valid = df.iloc[train_idx], df.iloc[valid_idx]
        model = make_model_fn().fit(train[FEATURES], train[LABEL])
        p = model.predict_proba(valid[FEATURES])[:, 1]
        scores.append(float(average_precision_score(valid[LABEL], p)))
    return scores
