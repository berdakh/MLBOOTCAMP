"""Reference solutions for notebook 10. Graded by tests/test_solutions.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churnlab.api import example_payload
from churnlab.features import FEATURES
from grading.ch11 import reference_transform


def ex1(model, http_client, rows_df: pd.DataFrame) -> dict:
    threshold = float(http_client.get("/model").json()["threshold"])
    bodies = [example_payload(row) for _, row in rows_df.iterrows()]
    r = http_client.post("/predict/batch", json={"customers": bodies})
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    p_api = np.array([p["churn_probability"] for p in preds], dtype=float)
    flag_api = np.array([p["flagged"] for p in preds], dtype=bool)
    p_off = model.predict_proba(rows_df[FEATURES])[:, 1]
    return {"max_abs_diff": float(np.max(np.abs(p_off - p_api))),
            "n_decision_flips": int(np.sum((p_off >= threshold) != flag_api))}


def ex2(customers: list[dict], stats: dict) -> np.ndarray:
    # One transform for training and serving: build the same DataFrame and call it.
    # (In the notebook this is `train_transform`; the grader's copy is identical.)
    return reference_transform(pd.DataFrame(customers).reindex(columns=FEATURES), stats)
