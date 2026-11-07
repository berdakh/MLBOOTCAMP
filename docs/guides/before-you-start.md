# Before you start

A ten-minute checklist. Do it before the diagnostic so Day 1 starts with
learning, not installing.


---

## The checklist

| # | Check | How | Done when |
|---|---|---|---|
| 1 | You can open a notebook in Colab | Open [notebook 00 in Colab](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/notebooks/00_diagnostic.ipynb), sign in with a Google account | The notebook is visible |
| 2 | The setup cell runs | Run the first code cell | It prints `repo:` and `artifacts:` paths with no error |
| 3 | You have decided where artifacts live | Read [Working in Colab](colab.md); choose `USE_DRIVE = True` or plan to export a zip | You know what happens to your work when the runtime disconnects |
| 4 | (Optional) A local install works | Follow [Setup](setup.md) and run `python -m pytest` | Tests pass |
| 5 | You have a place for notes | A text file, a notebook, or paper | It has a heading "Day 0 results" |
| 6 | You can record yourself | Phone or laptop camera; any app | You know where the recording button is. Day 5 and the rubric use it |
| 7 | Three hours are blocked in your calendar | For each of the next five sessions | They are blocked |

## What you need to know already

The course assumes:

- **Python**: functions, classes, comprehensions, virtual environments, reading
  a traceback.
- **pandas and NumPy**: selecting, grouping, merging; vectorised operations.
- **scikit-learn**: `fit`/`predict`, train/test splits, common metrics.
- **ML basics**: overfitting, regularisation, cross-validation, class imbalance.

It does **not** assume SQL, FastAPI, pytest, Docker, or any cloud platform.
Those are taught.

If you are unsure, the [diagnostic](https://github.com/berdakh/MLBOOTCAMP/blob/main/notebooks/00_diagnostic.ipynb)
will tell you within an hour.

## What you do not need

- A GPU. Every core cell runs on a CPU. GPU material is labelled optional and
  nothing later depends on it.
- A paid Colab plan.
- Docker or a cloud account. Day 4 writes and explains deployment files;
  running them is an optional extension.
- Any personal or company data. The project uses a committed synthetic dataset.

## How to use the materials

- **Day pages** say what each day is for and what proves it is done. Start each
  session there: [Daily plan](../days/index.md).
- **Notebooks** are where the work happens.
- **Handbook chapters** explain the ideas without code: [Handbook](../handbook/index.md).
- **Solutions** are in the repository's `solutions/` folder. Attempt every
  exercise first.
