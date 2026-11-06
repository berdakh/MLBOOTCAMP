# Solutions · 01 · A reproducible ML project

Try each exercise before reading its answer. Markers: ✅ verified by running ·
🔬 expected, depends on your run · ✍️ personal / open-ended.

---

## 01-1 · `config_hash`

✅ Reference code: [`code/nb01.py`](code/nb01.py) (`ex1`), graded in CI.

```python
def config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
```

**Why it works.** `sort_keys=True` sorts keys at every nesting level, so key
order cannot change the bytes. `separators` removes whitespace differences.
`default=str` keeps `Path` or `Timestamp` values from crashing the dump (they
become strings, which is acceptable for an identity).

**Common mistakes.**

* `hash(str(config))`: Python's `hash` of a string is salted per process, so
  the "same" config gets a different ID after a restart.
* Sorting only the top-level keys (`sorted(config.items())`). Nested dicts
  still vary.
* Using a list of features in the config and *sorting* it. Feature **order**
  can matter (it does for the column order of a model's input), so it should
  change the hash.

**Push further.** Twelve hex characters = 48 bits. For a few thousand runs,
the chance of a collision is negligible, but say how you would detect one
(store the full config next to the ID, and compare on write).

---

## 01-2 · `record_run`

✅ Reference code: [`code/nb01.py`](code/nb01.py) (`ex2`).

The key decisions:

* The **file name is the config hash**, so re-running the same config
  overwrites its own record instead of piling up near-duplicates. If you
  *want* every execution kept (e.g. to measure seed variance), add the seed to
  the config. It is part of what defines the run.
* `created_at` is ISO-8601 **in UTC**. Local times in logs from machines in
  different time zones are a classic source of confusion.
* The environment is captured *at run time*, not copied from a requirements
  file. What was installed is what matters.

**Common mistake.** Writing `metrics` that contain NumPy floats:
`json.dumps(np.float64(0.1))` works, but `np.float32` and NumPy integers raise
`TypeError`. Cast with `float()`/`int()` when building metrics.

---

## 01-3 · Three ways a re-run could differ (written)

✍️ Personal. A strong answer names **specific** mechanisms and the pin that
prevents each:

| What changed | Pin that prevents it |
|---|---|
| A preprocessing default changed in a new MNE / scikit-learn release (e.g. a filter design or a solver default) | environment: exact versions in a lock file, recorded with the result |
| The dataset was re-downloaded and a subject was added or re-referenced | data: checksum of the raw files, fingerprint of the epoch table |
| The subject-wise split was drawn without a seed, so a different held-out subject set | randomness: seeded split, and the list of held-out subjects saved |
| An uncommitted notebook edit produced the published number | code: commit hash, and refusing to report from a `-dirty` tree |

Credit any three with a correct pin. Extra credit for noticing that **a seed
only makes it repeatable**: the honest report is the spread over subjects or
seeds.

---

## Checkpoint · 01

**1. Four pins.** Code (git commit), data (raw checksums + table
fingerprints), environment (lock file + recorded versions), randomness
(explicit seeds). Bonus: the decisions, written down.

**2. 0.168 vs 0.163.** Compare the run records: data fingerprint (same
rows?), library versions (a changed default?), seeds and config hash. Then
ask whether 0.005 is outside the seed-to-seed spread at all. It may be noise.

**3. Seeds and comparisons.** A seed makes *one* run repeatable; it does not
tell you how much the result moves under a different seed, split or sample.
A difference smaller than that variation is not evidence of improvement.

**4. `src/`.** Importable by services and tests, reviewable diffs, no hidden
cell state. The *src layout* means the package must be installed or explicitly
put on the path, so tests cannot pass by accidentally importing files from the
current directory.

**5. Models out of git.** Large binaries, reproducible from pinned inputs, and
pickles execute code on load. They travel through a registry with metadata
and hashes (Day 2, notebook 08).
