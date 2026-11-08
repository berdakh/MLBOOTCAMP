# Solutions · Chapter 5 · Leakage-safe splits & a baseline

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `time_split_with_embargo`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

```python
def time_split_with_embargo(df, train_end, valid_end, gap_days):
    t = df["snapshot_date"]
    gap = pd.Timedelta(days=gap_days)
    train = df[t + gap <= train_end]
    valid = df[(t >= train_end) & (t + gap <= valid_end)]
    test = df[t >= valid_end]
    return train, valid, test
```

**The idea in one line.** A row belongs to an earlier period only if its whole
label window `[T, T + gap)` ends by that period's end. Test has no later
period, so it needs no gap.

**Common mistakes.** Applying the gap to the *start* of the later split (`t >=
train_end + gap`) instead: it removes the right amount of time but from the
wrong side, and the training rows still carry labels from inside validation.
Using `t < train_end` only, which is the same as no embargo.

**Edge case worth knowing.** With `gap_days=0`, `t + 0 <= train_end` keeps a
snapshot dated exactly `train_end` in train *and* in valid (`t >= train_end`).
Use a strict inequality for the zero-gap case, or require `gap_days >= 1`.

---

## Exercise 2 · `leakage_problems`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
def leakage_problems(train, test, label_horizon_days, group_col=None):
    problems = []
    last_train, first_test = train["snapshot_date"].max(), test["snapshot_date"].min()
    if last_train + pd.Timedelta(days=label_horizon_days) > first_test:
        problems.append(f"time: last train snapshot {last_train.date()} + {label_horizon_days}d "
                        f"reaches past the first test snapshot {first_test.date()}")
    if group_col is not None:
        shared = set(train[group_col]) & set(test[group_col])
        if shared:
            problems.append(f"group: {len(shared)} {group_col} value(s) in both (e.g. {sorted(shared)[:3]})")
    return problems
```

**Design choices.** Return *all* problems as readable strings rather than
raising on the first: a reviewer wants the full list, and a test can still do
`assert leakage_problems(...) == []`. Make the group check opt-in, because
shared customers across periods are correct for the next-month question.

---

## Exercise 3 · `rule_scores`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex3`). Validation PR-AUC
0.118 (positive rate 0.052), against 0.1245 for the 5-feature logistic
regression and 0.061 for the yes/no hand rule.

```python
def rule_scores(df):
    return (-np.log(df["sessions_trend"].fillna(1.0).clip(lower=1e-3))
            + 1.0 * df["cancellation_tickets_30d"].fillna(0)
            + 0.5 * df["failed_payments_90d"].fillna(0)
            + 0.3 * df["tickets_30d"].fillna(0)).to_numpy(dtype=float)
```

**Why it works.** A continuous score gives a full ranking; PR-AUC rewards
putting the likely churners first. `-log(sessions_trend)` is high when usage
fell (last 4 weeks vs the 4 before), the strongest single signal on this data
([verified] ≈ 0.105 alone). The weights were chosen by reasoning, not fitted; if you
tune them on validation you have trained a model on validation, so then hold
out something else.

**Push further.** Negated `sessions_4w` alone also clears the bar (≈ 0.10):
check single features first, it tells you where the signal is.

---

## Exercise 4 · The cross-subject EEG analogy (written)

[open-ended] A strong answer:

| This project | EEG/BCI equivalent | Right when |
|---|---|---|
| **Time split** (train earlier months, validate later months, same customers) | train on earlier sessions/runs, test on a later session of the **same subjects** | the system will be calibrated on a user and then used by that user on later days (session-to-session transfer, non-stationarity) |
| **Group split** (whole customers held out) | **leave-subject(s)-out** cross-subject evaluation | the system must work on new users with no calibration data (plug-and-play BCI) |
| **Random row split** | shuffling epochs from all sessions and subjects into folds | almost never for a claim about deployment: neighbouring epochs share drift, artefacts and filter state, so it measures memorisation of the recording |

**Embargo equivalent.** Epochs that overlap in time (sliding windows) or share
a filter's transient, or a label defined over a window (e.g. "error within the
next 2 s") that crosses the train/test boundary within one session: drop
epochs within one window length of the boundary.

Credit answers that state the deployment question first and derive the split
from it.

---

## Checkpoint answers

**1. Time split.** The model scores existing customers each month for "cancels
next month", so training on earlier months and validating on later months
rehearses exactly that. A group split answers "how well on customers we have
never seen", a different product question.

**2. Embargo gap.** Labels look 30 days ahead, so a training row less than 30
days before the boundary has a label decided by validation-period events; such
rows are dropped. With snapshots and boundaries on the 1st, the March
snapshot's window ends on March 31, before April 1, so nothing is dropped;
with a mid-month boundary (2024-03-15) it drops the 2,246 March rows.

**3. PR-AUC 0.97.** Check the feature list for export-time columns
(`status`, `last_active_week`, `churn_date` or anything derived) and check
feature timestamps against the snapshot. The notebook measured 0.12 → 0.99
from two leaky columns *with a perfectly time-ordered split*: split checks see
row membership, not what a column encodes.

**4. Random split, small inflation.** [expected] This run: random rows 0.128-0.152 vs
time 0.1245. Features are already point-in-time (the SQL removed the big leak),
the 5-feature linear model cannot memorise customers, and the synthetic
business is fairly stationary. It is still wrong: it answers a question nobody
asked, it fails `assert_no_leakage` (label windows overlap), it hides drift,
and a higher-capacity model or entity-specific features would exploit shared
customers.

**5. Prior and hand rule.** A constant score gives every precision equal to
the positive rate, so its average precision is the positive rate (0.052): the
floor. A hand rule is the cheapest serious alternative; if a model cannot beat
it clearly, the model is not worth operating. Here the yes/no rule barely beat
the floor (0.061), and a continuous rule got 0.118.
