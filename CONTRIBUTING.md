# Authoring guide

How the course is built, so every notebook looks, runs and is checked the same
way. Read this before editing a notebook, a grader or a solution.

## The build

```
notebook_src/NN_slug.py   ->  tools/nbbuild.py  ->  notebooks/NN_slug.ipynb
```

* **Edit `notebook_src/*.py`, never the `.ipynb`.** Sources are percent-format
  Python (`# %%` code cells, `# %% [markdown]` markdown cells with every line
  prefixed by `# `). Run `make notebooks` and commit both files. CI fails if
  they drift (`python tools/nbbuild.py --check`).
* Files in `notebook_src/` whose names start with `_` are templates, not built.
* Committed notebooks have **no outputs** (`tools/check_notebooks.py`).
* `python tools/run_notebooks.py NN` executes a notebook in a fresh, empty
  artifact directory - the closest local approximation to a new Colab
  runtime. Every notebook must pass it.

## Notebook anatomy (in this order)

1. **Colab badge** - exactly
   `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/NN_slug.ipynb)`
   followed by one italic line on how to run it.
2. **Header** - `# NN · Title`, then `**Day D · ~MM minutes · CPU only**`
   (or `CPU core · optional GPU extension`), a two-sentence motivation, a
   numbered *"By the end you will be able to"* list, and *"You will produce"*
   (the files/artifacts this notebook leaves behind).
3. **Setup cell** - copy it verbatim from `notebook_src/_template.py`. It
   clones the repo in Colab, installs `requirements.txt`, finds the repo root,
   and chooses the artifact directory (`USE_DRIVE` switch).
4. **Catch-up cell** - call the `churnlab.bootstrap.ensure_*` function for
   what this notebook needs from earlier ones. Never assume a previous
   notebook ran in this session.
5. **Numbered sections** - `## N.1 · Title`. Short prose, then code, then an
   **Expected:** line stating what the learner should see (ranges, not
   exact floats, unless the value is deterministic).
6. **Exercises** - see below. 2-4 per notebook.
7. **Troubleshooting** - a `## 🛟 Troubleshooting` table: symptom → cause → fix,
   for the failures a learner will actually hit in *this* notebook.
8. **Checkpoint** - `## Checkpoint`: 4-6 questions a learner should answer
   aloud without notes. They are interview-style on purpose.
9. **Next** - `➡️ **Next:** [`NN_slug.ipynb`](NN_slug.ipynb) — one line why.

Diagrams are embedded from the raw GitHub URL so they also render in Colab:
`<img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/NAME.svg" alt="..." width="100%">`

## Exercises, graders and solutions

An exercise is a markdown cell starting `> **Exercise K.**` (K = 1, 2, ...),
then a code cell with a stub, then a check cell:

```python
def leakage_free_split(df):
    # YOUR CODE HERE
    raise NotImplementedError

grade.ex1(leakage_free_split)
```

* Graded exercises are **always functions** - for SQL, a function that
  returns the query string; for a threshold, a function that returns it. That
  keeps every grader's interface identical: `grade.exK(fn)`.
* `grade` is `churnlab.grading.nbNN`, imported in the notebook's catch-up cell.
  Each `exK` builds small tests and calls `churnlab.checks.run_check`, which
  prints ✅ / ❌ (with a hint) / ⏳ TODO and **never raises** - so the notebook
  always runs top to bottom with unfinished exercises.
* Written (non-code) exercises have no grader; say *"Write your answer in the
  cell below"* and provide a markdown cell.
* Every exercise needs a solution in `solutions/NN_slug.md` under a heading
  `## NN-K · short title`, marked ✅ (verified by running), 🔬 (expected; depends
  on a run) or ✍️ (personal / open-ended).
* Code solutions live in `solutions/code/nbNN.py` as functions named `exK`
  with the same signature as the stub. `tests/test_solutions.py` runs every
  grader against its solution - a grader that rejects its own solution fails CI.
* Every notebook with a `## Checkpoint` needs a `## Checkpoint · NN` section
  in its `solutions/NN_slug.md` answering each question.

Solutions are separate from the notebooks so that a learner can work without
seeing them. They are in the same repository so they stay in sync.

## Style

* **Measure, don't assert.** Every claim about a model is a number from this
  notebook's run, with its split named. Never invent results or impact.
* **Nothing is magic.** Build a thing by hand first (a split, a PSI, a retry
  loop), then show the `churnlab` function that does it, and point at its
  source file.
* **CPU first.** Core cells finish in seconds to a couple of minutes on a
  free Colab CPU. GPU material is in a clearly labelled optional section,
  guarded by `RUN_GPU_EXTENSION = False`, and must not be required later.
* **Honest about Colab.** Things that cannot run in Colab (Docker, a public
  server) are taught by generating files and explaining them; the notebook
  says plainly that running them is an optional external extension.
* British or American spelling, but be consistent within a file.
* Keep cells short. A learner should be able to predict what a cell prints.
