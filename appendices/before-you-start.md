# Before you start

This appendix is a short checklist, to be completed before
[Chapter 1](../part1-foundations/ch01-diagnostic/README.md), so that the
first study session is spent on the material rather than on installation. It
takes about ten minutes.

---

## The checklist

| # | Check | How | Complete when |
|---|---|---|---|
| 1 | A lab opens in Colab | Open [the Chapter 1 lab in Colab](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part1-foundations/ch01-diagnostic/notebook.ipynb) and sign in with a Google account | The notebook is visible |
| 2 | The setup cell runs | Run the first code cell | It prints `repo:` and `artifacts:` paths without an error |
| 3 | You have decided where artifacts are kept | Read [Working in Colab](colab.md); either set `USE_DRIVE = True` or plan to export an archive at the end of each session | You know what happens to your work when the runtime disconnects |
| 4 | (Optional) A local installation works | Follow [Setup](setup.md) and run `python -m pytest` | The tests pass |
| 5 | You have a place for notes | A text file, a notebook, or paper | It has a heading "Chapter 1 results", for the diagnostic scores |
| 6 | You can record yourself | A phone or laptop camera and any recording application | You know how to start a recording. Chapter 18 and the [readiness rubric](readiness-rubric.md) use recordings of spoken answers |
| 7 | Study time is scheduled | Blocks of about three hours, as in the [study plan](study-plan.md) | The first several sessions are in your calendar |

## What the book assumes

The book assumes working knowledge of:

- **Python**: functions, classes, comprehensions, virtual environments, and
  reading a traceback;
- **pandas and NumPy**: selecting, grouping and merging; vectorised
  operations;
- **scikit-learn**: `fit` and `predict`, train/test splits, common metrics;
- **the basics of machine learning**: overfitting, regularisation,
  cross-validation, class imbalance;
- for Parts II and III, **the basics of deep learning**: tensors, a training
  loop, gradient descent, and the idea of an embedding.

It does **not** assume SQL, FastAPI, pytest, Docker, distributed training or
any cloud platform. These are taught from the beginning.

If you are unsure whether your background is sufficient, the
[skills diagnostic in Chapter 1](../part1-foundations/ch01-diagnostic/notebook.ipynb)
answers the question in about an hour, and
[Interpreting the diagnostic](diagnostic.md) explains how to act on the
result.

## What you do not need

- **A GPU for Parts I and II.** Every core cell of these parts runs on a CPU.
  Part III is written for the free T4 GPU that Colab provides
  (*Runtime → Change runtime type*); it also runs on a CPU with a reduced
  budget of data and model size, more slowly and with weaker results.
- **A paid Colab plan.**
- **Docker or a cloud account.** Chapter 13 generates and explains deployment
  files; building and running them is an optional extension outside Colab.
- **Personal or company data.** Parts I and II use committed or generated
  synthetic data; Part III and the capstones use public datasets.

## How the material is organised

- **Chapter text.** Each chapter folder contains a `README.md` with the
  learning objectives, the explanation, figures, equations, key terms, a
  summary, exercises and further reading. Read it first.
- **Lab.** The chapter's `notebook.ipynb` is where the work happens: runnable
  code, exercises with automatic checks, and checkpoint questions.
- **Solutions.** The chapter's `solutions.md` holds worked answers to every
  exercise and checkpoint question. Attempt each exercise before opening it;
  [About the solutions](solutions.md) explains why and how the answers are
  marked.
- **Appendices.** Reference and study material for the whole book, listed in
  the [appendices index](README.md).
