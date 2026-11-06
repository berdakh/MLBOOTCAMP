"""Model pipelines: preprocessing and estimator in ONE object.

Keeping the preprocessing inside the sklearn ``Pipeline`` is the single best
defence against training/serving skew: the object you save is the object you
serve, so the API cannot "forget" to scale a column or encode a category the
same way training did.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import CATEGORICAL, NUMERIC

#: Heavy-tailed counts get log1p before scaling for the linear model.
LOG_FEATURES = ["seats", "mrr_usd", "sessions_4w", "sessions_prev_4w", "sessions_trend",
                "tickets_90d", "avg_resolution_hours_90d"]


def _one_hot() -> OneHotEncoder:
    # handle_unknown="ignore": a category never seen in training (a new region,
    # a new plan) becomes all-zeros instead of crashing the API at 3 a.m.
    return OneHotEncoder(handle_unknown="ignore", min_frequency=20)


def make_preprocessor(kind: str = "linear", categorical=CATEGORICAL, numeric=NUMERIC) -> ColumnTransformer:
    if kind == "linear":
        logged = [c for c in numeric if c in LOG_FEATURES]
        plain = [c for c in numeric if c not in LOG_FEATURES]
        log_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ])
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])
        return ColumnTransformer([
            ("log", log_pipe, logged),
            ("num", num_pipe, plain),
            ("cat", _one_hot(), list(categorical)),
        ])
    if kind == "tree":
        # Trees need no scaling and handle NaN natively; categories become
        # integer codes, unknown categories map to a dedicated code.
        return ColumnTransformer([
            ("num", "passthrough", list(numeric)),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1), list(categorical)),
        ])
    raise ValueError(f"unknown preprocessor kind {kind!r}")


def make_model(name: str = "logreg", random_state: int = 0, **params) -> Pipeline:
    """Build an unfitted pipeline by name: ``dummy``, ``logreg`` or ``hgb``."""
    if name == "dummy":
        return Pipeline([("pre", make_preprocessor("tree")),
                         ("clf", DummyClassifier(strategy="prior"))])
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, C=params.pop("C", 1.0),
                                 class_weight=params.pop("class_weight", None), **params)
        return Pipeline([("pre", make_preprocessor("linear")), ("clf", clf)])
    if name == "hgb":
        n_num = len(NUMERIC)
        n_cat = len(CATEGORICAL)
        # Conservative defaults: with ~25k rows and a 5% positive rate, deep
        # trees overfit quickly. Day 2 searches around these values.
        defaults = dict(learning_rate=0.03, max_iter=200, max_leaf_nodes=8, min_samples_leaf=200,
                        l2_regularization=1.0, early_stopping=False)
        defaults.update(params)
        clf = HistGradientBoostingClassifier(
            categorical_features=[False] * n_num + [True] * n_cat,
            random_state=random_state, **defaults,
        )
        return Pipeline([("pre", make_preprocessor("tree")), ("clf", clf)])
    raise ValueError(f"unknown model {name!r}")


def predict_proba(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Positive-class probability, selecting the feature columns explicitly."""
    return model.predict_proba(df[CATEGORICAL + NUMERIC])[:, 1]
