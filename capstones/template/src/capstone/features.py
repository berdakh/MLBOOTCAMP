"""Feature construction.

Every feature must be computable at prediction time from information available
*before* the moment being predicted. Write that check as a test, not a hope.
"""

from __future__ import annotations

import pandas as pd

FEATURES = ["x1", "x2"]  # TODO: your feature list - it is also the API contract


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with exactly FEATURES, in order.

    TODO: lags / rolling windows (shifted so they only see the past), calendar
    features, per-entity aggregates computed on the training window only.
    """
    out = df.copy()
    return out[FEATURES]
