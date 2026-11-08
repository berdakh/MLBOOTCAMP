# Solutions · Chapter 13 · Packaging & configuration

Try each exercise before reading its answer. Markers: [verified] verified by running ·
[expected] expected, depends on your run · [open-ended] personal / open-ended.

---

## Exercise 1 · `dockerfile_problems`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex1`), graded in CI.

The work is in parsing, not in the rules. Turn the text into a list of
*instructions* first, then each rule is one or two lines:

```python
def _instructions(text):
    logical, buffer = [], ""
    for raw in text.splitlines():
        line = raw.strip()
        if not buffer and (not line or line.startswith("#")):
            continue
        if buffer and line.startswith("#"):
            continue
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        logical.append(buffer + line)
        buffer = ""
    if buffer:
        logical.append(buffer)
    return [(l.split(None, 1)[0].upper(), (l.split(None, 1) + [""])[1]) for l in logical]
```

Then: `root_user` if there is no `USER` or the last one is `root`/`0`;
`no_healthcheck` if there is no `HEALTHCHECK` or the last one is `NONE`;
`latest_tag` for a `FROM` image without a tag or with `:latest` (skip
`--platform=` words, accept `@sha256:` digests and `scratch`); `copies_env_file`
if a `COPY`/`ADD` word's base name is `.env` or starts with `.env.`;
`pip_cache` for a `RUN` with `pip install` and no `--no-cache-dir`.

**Common mistakes.**

* Checking line by line without joining continuations: the `--no-cache-dir`
  on the next line is missed and a correct Dockerfile is flagged.
* Searching the whole text for `"USER"`: a comment "# no USER needed" passes.
  Only *instructions* count, and only the **last** `USER` matters.
* Splitting the image on the first `:`. `localhost:5000/churn-api` has a port,
  not a tag. Take the part after the last `/` before looking for `:`.
* Treating `environment.yml` as an env file because it contains "env".

**Push further.** Real linters exist: [hadolint](https://github.com/hadolint/hadolint)
checks these and dozens more (e.g. `apt-get install` without
`--no-install-recommends`). The point of writing your own is knowing *why*
each rule exists: unpinned bases change under you, root widens a breach, a
copied `.env` leaks secrets to anyone who can pull the image, a pip cache
bloats every layer, and without a health check nobody notices a hung process.

---

## Exercise 2 · `settings_from_env`

[verified] Reference code: [`solutions.py`](solutions.py) (`ex2`).

```python
LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

def _int_setting(env, name, default, low, high):
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer from {low} to {high}, got {raw!r}") from None
    if not low <= value <= high:
        raise ValueError(f"{name} must be from {low} to {high}, got {value}")
    return value

def settings_from_env(env):
    level = env.get("CHURN_LOG_LEVEL") or "INFO"
    if level.upper() not in LEVELS:
        raise ValueError(f"CHURN_LOG_LEVEL must be one of {', '.join(LEVELS)}, got {level!r}")
    return {"port": _int_setting(env, "CHURN_PORT", 8000, 1, 65535),
            "workers": _int_setting(env, "CHURN_WORKERS", 1, 1, 64),
            "log_level": level.upper()}
```

**Why it is written this way.** Every error names the variable, the bad value
and what was allowed, so the person reading the deployment log can fix it
without opening the code. `from None` hides Python's `invalid literal for
int()` chain, which is noise to that reader.

**Common mistakes.**

* `int(env.get("CHURN_PORT", 8000))`: an empty string (a variable that is set
  but blank, common in templated deployments) crashes instead of defaulting.
* `int(float(raw))`: silently turns `"80.5"` into 80. Refuse it.
* Reading `os.environ` inside the function instead of the argument: impossible
  to test without mutating the process environment.

**Push further.** `pydantic-settings` (`BaseSettings`) does all of this
declaratively, including `.env` files for local development. Knowing the
twenty lines it replaces makes its errors easy to read.

---

## Exercise 3 · Bake the model into the image, or load it from the registry? (written)

[open-ended] Personal. Both choices can be defended; the grade is in the reasoning. A
strong answer for this project:

> Load from the registry at start-up. The model is retrained monthly and must
> roll back fast: rewriting `CURRENT` and calling `/admin/reload` takes seconds,
> while rebuilding and redeploying an image takes minutes. Audit is covered
> because bundles are immutable, carry a SHA-256 and metadata, and every
> promotion and rollback is in `history.jsonl`. The weakness is the start-up
> dependency: if the registry is unreachable a new pod cannot load a model, so
> it stays *not ready* (`/ready` 503) and receives no traffic while old pods keep
> serving; I would also log the loaded version and hash at start-up, and pin
> `CHURN_MODEL_VERSION` in environments that need byte-for-byte reproducibility.

A good "bake it in" answer instead stresses that one image digest identifies
code + model completely, there is no runtime dependency, and rollback is
"deploy the previous image", which the platform already does well; it should
admit the slower rollback and an image rebuild for every retrain.

---

## Checkpoint answers

**1. `pyproject` vs lock.** `pyproject.toml` (and `requirements.txt`) state
*compatible ranges*: what the package needs. `requirements-lock.txt` states
*exact* versions one environment was verified with. The image installs from
the lock, so every container is identical; the package keeps ranges so it can
be installed next to other software.

**2. Environment configuration.** The same image runs in every environment;
only the variables change, nothing environment-specific is committed, and
secrets can be injected by the platform. Failing at start-up turns a typo into
a failed rollout (the old version keeps serving) instead of weeks of quietly
wrong predictions.

**3. Dockerfile walk-through.** Base pinned by tag (digest for full
reproducibility) → env flags for unbuffered logs → copy the lock and install
dependencies (**cached layer**, rebuilt only when the lock changes) → copy and
install the code (changes often, so later) → non-root `USER` (a compromised
process is not root) → default `ENV`, `EXPOSE`, `HEALTHCHECK` on `/health` →
exec-form `CMD` so uvicorn is PID 1 and receives the stop signal.

**4. `--workers 4`.** Four processes, each with its own model in memory (4×
RAM) and true CPU parallelism. In-process state is per worker: `/metrics`
reports one worker's counts, and `/admin/reload` reloads only the worker that
received it, so after a rollback some workers could still serve the old model.
Fix: restart all workers (rolling restart), use one worker per container, or
export metrics to a shared system (Prometheus scrapes each).

**5. Bake vs registry.** Bake: one immutable digest identifies code and model,
no start-up dependency. Registry: rollback in seconds by moving a pointer,
small images, no rebuild per retrain.

**6. Secrets.** From the platform's secret store, injected as environment
variables or mounted files at run time. Never in the image (`ENV`, copied
`.env`), the repository, notebooks, or logs.
