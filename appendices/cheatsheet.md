# Cheat sheet

This appendix collects short code snippets for the operations written most
often in the book. They are reminders of syntax and of the details that are
easy to get wrong, not a substitute for being able to write the code from an
empty file. Each section names the chapters in which its material is
explained; the explanations of *why* a snippet is written as it is are there,
not here.


---

## pandas

*Explained in:* [Chapter 3](../part1-foundations/ch03-data-acquisition-and-validation/README.md), [Chapter 5](../part1-foundations/ch05-leakage-safe-splits-and-baseline/README.md), [Chapter 17](../part1-foundations/ch17-python-and-sql-drills/README.md).

```python
import pandas as pd

# Read safely: only empty fields are missing ("NA" stays a string)
df = pd.read_csv("customers.csv.gz", keep_default_na=False, na_values=[""],
                 parse_dates=["signup_date"])

df.info(); df.describe(include="all"); df.isna().mean().sort_values()
df["plan"].value_counts(dropna=False)

# Duplicates on a key
df.duplicated(subset=["customer_id", "week_start"]).sum()
df = df.drop_duplicates(subset=["customer_id", "week_start"], keep="first")

# Group and aggregate with named outputs
agg = (usage.groupby("customer_id")
             .agg(sessions=("sessions", "sum"), weeks=("week_start", "nunique"))
             .reset_index())

# Merge and check the result size
out = customers.merge(agg, on="customer_id", how="left", validate="one_to_one")

# Per-group rank / latest row per group
inv = inv.sort_values(["customer_id", "invoice_date"])
latest = inv.groupby("customer_id").tail(1)
inv["n"] = inv.groupby("customer_id").cumcount() + 1

# Lag and rolling within a group (rows must be sorted by time first)
u = usage.sort_values(["customer_id", "week_start"])
u["prev"] = u.groupby("customer_id")["sessions"].shift(1)
u["roll4"] = u.groupby("customer_id")["sessions"].transform(lambda s: s.rolling(4, min_periods=1).sum())

# As-of join: for each snapshot, the most recent invoice strictly before it
snap = snap.sort_values("snapshot_date"); inv = inv.sort_values("invoice_date")
pit = pd.merge_asof(snap, inv, left_on="snapshot_date", right_on="invoice_date",
                    by="customer_id", direction="backward", allow_exact_matches=False)

# Dates
df["month"] = df["snapshot_date"].dt.to_period("M")
mask = (t >= start) & (t < start + pd.Timedelta(days=30))
```

## SQL (DuckDB)

*Explained in:* [Chapter 4](../part1-foundations/ch04-sql-for-ml/README.md), [Chapter 17](../part1-foundations/ch17-python-and-sql-drills/README.md); at scale, [Chapters 19–20](../part2-ml-at-scale/ch19-event-logs-and-columnar-data/README.md).

```python
import duckdb
con = duckdb.connect()                   # in-memory
con.register("usage_weekly", usage)      # a DataFrame becomes a view
con.execute("SELECT COUNT(*) FROM usage_weekly").fetchone()
df = con.execute("SELECT * FROM usage_weekly WHERE sessions > ?", [10]).df()
con.execute("DESCRIBE usage_weekly").df()   # column types
```

```sql
-- Aggregation with a filter per column
SELECT customer_id,
       COUNT(*)                                      AS tickets,
       COUNT(*) FILTER (WHERE category = 'billing')  AS billing_tickets,
       AVG(resolution_hours)                         AS avg_hours
FROM tickets
GROUP BY customer_id
HAVING COUNT(*) >= 2;

-- Latest row per group
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id
                                 ORDER BY invoice_date DESC, invoice_id DESC) AS rn
    FROM invoices
) WHERE rn = 1;
-- DuckDB shortcut: SELECT ... QUALIFY rn = 1, or arg_max(amount_usd, invoice_date)

-- Lag and running totals
SELECT customer_id, week_start, sessions,
       LAG(sessions) OVER w                                        AS prev_sessions,
       SUM(sessions) OVER (w ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS sessions_4w
FROM usage_weekly
WINDOW w AS (PARTITION BY customer_id ORDER BY week_start);

-- Point-in-time: features strictly before each snapshot, weeks counted at their END
SELECT s.snapshot_date, u.customer_id, SUM(u.sessions) AS sessions_4w
FROM snapshots s
JOIN usage_weekly u
  ON  u.week_start + INTERVAL 7 DAY <= s.snapshot_date
  AND u.week_start + INTERVAL 7 DAY >  s.snapshot_date - INTERVAL 28 DAY
GROUP BY ALL;

-- Label: event in [T, T + 30 days)
CASE WHEN churn_date >= s.snapshot_date
      AND churn_date <  s.snapshot_date + INTERVAL 30 DAY THEN 1 ELSE 0 END

-- Typed literals and casts
WHERE created_at >= DATE '2024-04-01'
SELECT TRY_CAST(raw_value AS INTEGER) ...
date_diff('day', signup_date, snapshot_date)
```

Join reminders: `INNER` keeps matches only; `LEFT` keeps every left row (check
row counts before and after); filters on the right table in `WHERE` turn a
`LEFT JOIN` into an inner join, so put them in `ON`.

## scikit-learn

*Explained in:* [Chapters 5–9](../part1-foundations/ch05-leakage-safe-splits-and-baseline/README.md), [Chapter 16](../part1-foundations/ch16-final-evaluation-and-docs/README.md).

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                      ("scale", StandardScaler())]), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])
model.fit(train[features], train[label])
p = model.predict_proba(valid[features])[:, 1]
```

```python
# Time-aware CV with any iterable of (train_idx, valid_idx)
from sklearn.model_selection import cross_val_score, GroupKFold, TimeSeriesSplit
cross_val_score(model, X, y, cv=list(folds), scoring="average_precision")
GroupKFold(n_splits=5).split(X, y, groups=customer_ids)   # whole customers per fold

# Metrics for imbalance
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve
average_precision_score(y, p); roc_auc_score(y, p); brier_score_loss(y, p)

# Calibration
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")

# Save / load (pickles execute code: trusted files only)
import joblib
joblib.dump(model, "model.joblib"); model = joblib.load("model.joblib")
```

```python
# Group (cluster) bootstrap CI, by hand
import numpy as np
def group_bootstrap(y, p, groups, metric, n=500, seed=0):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    stats = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        if y[idx].min() != y[idx].max():
            stats.append(metric(y[idx], p[idx]))
    return np.quantile(stats, [0.025, 0.975])
```

Break-even threshold with value `V` per true positive and cost `C` per false
positive, for calibrated `p`: act when `p > C / (V + C)`. With the book's
assumed V = 150, C = 30: `p > 1/6`. The derivation is in [Chapter 8](../part1-foundations/ch08-thresholds-and-calibration/README.md).

## FastAPI and Pydantic

*Explained in:* [Chapter 10](../part1-foundations/ch10-fastapi-and-contracts/README.md), [Chapter 11](../part1-foundations/ch11-consistent-serving/README.md), [Chapter 12](../part1-foundations/ch12-api-clients-and-tests/README.md), [Chapter 14](../part1-foundations/ch14-health-logs-and-latency/README.md).

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")      # typo'd fields -> 422
    customer_id: str = Field(min_length=1)
    plan: str
    sessions_4w: float = Field(ge=0)
    features_used_4w: float | None = None          # optional, imputed downstream

class Prediction(BaseModel):
    customer_id: str
    churn_probability: float = Field(ge=0, le=1)
    model_version: str

def create_app(model, version: str) -> FastAPI:     # app factory: no globals
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}                    # never touches the model

    @app.get("/ready")
    def ready():
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ready", "model_version": version}

    @app.post("/predict", response_model=Prediction)
    def predict(c: Customer):
        p = float(model.predict_proba(to_frame(c))[:, 1][0])
        return Prediction(customer_id=c.customer_id, churn_probability=p, model_version=version)

    return app
```

```bash
uvicorn serve:app --host 127.0.0.1 --port 8000     # serve.py defines app = create_app(...)
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d @payload.json
```

```python
# Client: timeout always; retry only retryable statuses
import httpx, time
RETRYABLE = {429, 502, 503, 504}
def post_with_retry(client: httpx.Client, url, body, retries=3, backoff=0.2):
    for attempt in range(retries + 1):
        try:
            r = client.post(url, json=body, timeout=5.0)
            if r.status_code < 400:
                return r.json()
            if r.status_code not in RETRYABLE:
                r.raise_for_status()               # 4xx: fail now, do not retry
        except (httpx.ConnectError, httpx.ReadTimeout):
            pass
        if attempt < retries:
            time.sleep(backoff * 2 ** attempt)
    raise RuntimeError(f"gave up after {retries + 1} attempts")
```

## pytest

*Explained in:* [Chapter 10](../part1-foundations/ch10-fastapi-and-contracts/README.md), [Chapter 12](../part1-foundations/ch12-api-clients-and-tests/README.md).

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client(tmp_path):                          # fresh registry per test
    registry = build_tiny_registry(tmp_path)
    return TestClient(create_app(Settings(registry_dir=registry)))

def test_predict_returns_version(client, valid_payload):
    r = client.post("/predict", json=valid_payload)
    assert r.status_code == 200
    assert r.json()["model_version"] == "v1"

@pytest.mark.parametrize("field,value", [("sessions_4w", -1), ("plan", None)])
def test_invalid_input_is_422(client, valid_payload, field, value):
    r = client.post("/predict", json={**valid_payload, field: value})
    assert r.status_code == 422

def test_typo_field_rejected(client, valid_payload):
    body = dict(valid_payload); body["sesions_4w"] = body.pop("sessions_4w")
    assert client.post("/predict", json=body).status_code == 422

def test_api_matches_pipeline(client, model, rows):    # skew check
    api = [client.post("/predict", json=r).json()["churn_probability"] for r in rows]
    assert max(abs(a - b) for a, b in zip(api, model_scores(model, rows))) < 1e-6

def test_raises_on_bad_config():
    with pytest.raises(ValueError):
        Settings.from_env(environ={"CHURN_THRESHOLD_OVERRIDE": "1.5"})
```

```bash
python -m pytest                       # all tests
python -m pytest tests/test_api.py -k typo -x -q    # one file, name filter, stop at first failure
python -m pytest --lf                  # only last failures
```

Put shared fixtures in `tests/conftest.py`. Use `tmp_path` for anything that
writes files. Assert behaviour (status codes, invariants, ranges), not exact
model metrics.

## Monitoring

*Explained in:* [Chapter 14](../part1-foundations/ch14-health-logs-and-latency/README.md), [Chapter 15](../part1-foundations/ch15-monitoring-and-rollback/README.md).

```python
import numpy as np
def psi(ref, cur, bins=10, eps=1e-4):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), eps, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))
# rule of thumb: < 0.1 stable, 0.1-0.25 moderate, > 0.25 large
np.percentile(latencies_ms, [50, 95, 99])
```

## Environment variables

*Explained in:* [Chapter 13](../part1-foundations/ch13-packaging-and-configuration/README.md).

```python
import os
registry = os.environ.get("CHURN_REGISTRY_DIR", "artifacts/registry")
batch = int(os.environ.get("CHURN_MAX_BATCH_SIZE", "1000"))   # convert types explicitly
```

```bash
export CHURN_LOG_LEVEL=DEBUG        # Windows PowerShell: $env:CHURN_LOG_LEVEL = "DEBUG"
```

## Ranking metrics

*Explained in:* [Chapter 24](../part2-ml-at-scale/ch24-learning-to-rank/README.md), [Chapter 25](../part2-ml-at-scale/ch25-candidate-generation/README.md).

```python
import numpy as np

def dcg_at_k(rel, k):
    """rel: graded relevance of the items, in the order they were ranked."""
    rel = np.asarray(rel, dtype=float)[:k]
    return float(np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2))))

def ndcg_at_k(rel, k):
    ideal = dcg_at_k(sorted(rel, reverse=True), k)
    return dcg_at_k(rel, k) / ideal if ideal > 0 else 0.0

def recall_at_k(retrieved, relevant, k):
    relevant = set(relevant)
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0
```

Compute these per query (or per user) and average over queries. A query with
no relevant item has no meaningful NDCG; state whether such queries are
dropped or counted as zero.

## PyTorch training loop

*Explained in:* [Chapter 27](../part2-ml-at-scale/ch27-pytorch-training-done-right/README.md), [Chapter 39](../part3-deep-learning-systems/ch39-gpu-fundamentals-and-profiling/README.md), [Chapter 40](../part3-deep-learning-systems/ch40-training-efficiency-and-memory/README.md).

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
g = torch.Generator().manual_seed(0)              # dedicated generator for shuffling
loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, generator=g)

for epoch in range(n_epochs):
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            loss = loss_fn(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.item()                    # .item(): do not keep the graph alive

    model.eval()
    with torch.no_grad():
        ...                                       # evaluate on held-out data

# Save everything needed to resume, not only the weights
torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, "ckpt.pt")
```

On a GPU, call `torch.cuda.synchronize()` before reading a timer: kernels run
asynchronously. On a T4, which has no fast bfloat16 support, use
`dtype=torch.float16` together with a gradient scaler
(`torch.amp.GradScaler`).
