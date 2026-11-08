# Solutions · Chapter 3 · Data acquisition & validation

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## Exercise 1 · `find_duplicate_keys`

✅ Reference code: [`code/nb02.py`](solutions.py) (`ex1`), graded in CI.

```python
def find_duplicate_keys(df, key):
    key = [key] if isinstance(key, str) else list(key)
    return df[df.duplicated(subset=key, keep=False)].sort_values(key)
```

**Why `keep=False`.** The default `keep="first"` marks only the *second and
later* copies, so you would see one row of each pair and nothing to compare it
with. To decide which copy wins you need every copy side by side; that is how
the notebook found the one key (`C03550`, 2024-12-02) whose copies disagree.

**Common mistakes.** `df.duplicated()` with no `subset` finds exact duplicate
*rows*, which misses the disagreeing pair. `df.drop_duplicates()` before
looking destroys the evidence.

---

## Exercise 2 · `orphan_ids`

✅ Reference code: [`code/nb02.py`](solutions.py) (`ex2`).

```python
def orphan_ids(tables):
    known = set(tables["customers"]["customer_id"])
    return {name: set(tables[name]["customer_id"]) - known
            for name in ("usage_weekly", "invoices", "tickets")}
```

🔬 On the real data every set is empty: the extract is referentially intact.
Record that as a finding (and as a test). It is exactly what breaks when a
later export filters `customers` (e.g. "active only") but not the event tables.

**In SQL** this is an anti-join: `SELECT DISTINCT u.customer_id FROM
usage_weekly u LEFT JOIN customers c USING (customer_id) WHERE c.customer_id
IS NULL`.

---

## Exercise 3 · `freshness`

✅ Reference code: [`code/nb02.py`](solutions.py) (`ex3`).

```python
def freshness(tables):
    rows = {}
    for name, col in TIMESTAMP_COLUMNS.items():
        ts = tables[name][col]
        rows[name] = {"min": ts.min(), "max": ts.max(), "rows": len(ts)}
    return pd.DataFrame.from_dict(rows, orient="index")
```

`Series.min()`/`.max()` skip `NaT` by default. ✅ On the real data: sign-ups
2022-05-16 to 2024-10-31; usage 2023-01-02 to 2024-12-23 (week starts);
invoices 2023-01-02 to 2024-12-29; tickets 2022-12-31 to 2024-12-29.

**Why it matters.** The latest timestamps tell you the export date (end of
December 2024), which is the "as of" date of `status` and `last_active_week`.
In production the same check, run daily, is the cheapest data-freshness alert
there is ("no usage rows newer than 8 days").

---

## Exercise 4 · The `-1` sentinel: decision memo (written)

✍️ Open-ended; a strong memo contains the following.

> `sessions = -1` appears in 199 usage rows. It is the logging pipeline's
> marker for "count unknown", not "no activity": inactive weeks are not logged
> at all (there are no rows with 0 sessions), and the `-1` rows have a median
> of 2 active users and 2 features used, which is impossible with zero
> sessions. We set it to missing (NaN) and impute inside the model pipeline,
> optionally with a missing-indicator, so the model can learn whether "unknown"
> itself carries signal. Replacing it with 0 would assert the account was
> inactive that week; the model would learn "zero sessions → churn risk" partly
> from logging failures, and every future logging outage would raise churn
> scores for unaffected customers. Keeping -1 as a number is worse still: it
> would be read as "less than no activity".

Bonus: notice that `drop_duplicates(keep="first")` kept the `-1` copy of the
one disagreeing duplicate; a better rule prefers the non-sentinel copy.

---

## Checkpoint answers

**1. Proving a download.** Compare its SHA-256 with a hash pinned in code
(next to the code that uses it). Download to `.part`, verify, then rename
atomically, so an interrupted or corrupted download never looks complete and
a later run does not silently use half a file.

**2. 42% missing in a required field.** Distrust the number: look at
`value_counts(dropna=False)` (a known category missing?), then at the raw bytes
of a few affected rows. Here the code `NA` was parsed as missing by
`read_csv`'s default `na_values`. Fix at read time (`keep_default_na=False,
na_values=[""]`), then re-count: 87 truly missing.

**3. Data contract.** A machine-checkable description of what each column must
look like (type, nullability, allowed values, ranges, uniqueness, keys). Run it
when data enters (after extraction), after cleaning, and on prediction requests
(the API schema is the serving-side contract).

**4. `-1` vs `0`.** `-1` means unknown; 0 means observed inactivity. Treating
-1 as 0 teaches the model that logging failures look like disengagement, and
production outages then move scores. See 02-4.

**5. Label input, not feature.** `churn_date`: the label for snapshot T is
"churn_date in [T, T+30d)". As a feature it reveals the future relative to T.
The same holds for `status` and `last_active_week`, which describe the account
at export time; at T, about half of the still-active accounts already show
`cancelled`.
