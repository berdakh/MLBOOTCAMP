# Solutions

Worked answers to every notebook exercise and checkpoint question. They are
kept out of the notebooks on purpose, so you can work without seeing them,
and kept in the same repository so they stay in sync with the notebooks.

## Attempt first

Reading a solution before attempting the exercise feels productive and teaches
very little. Recognising a correct answer is a different, much easier skill
than producing one, and interviews test producing. A good routine:

1. Attempt the exercise until the grader shows [verified], or for at least 15 minutes.
2. For written questions and checkpoints, write or say your answer **before**
   opening this folder.
3. Then read the solution and note anything you missed, or did differently
   and better. Disagreeing with a solution, for a reason you can state, is a
   good outcome.

If you are stuck for more than 20 minutes, read only the first line of the
solution (usually the key idea) and try again.

## What is here

```
solutions/
    README.md                 this file
    NN_slug.md                answers for notebook NN (same slug as notebooks/NN_slug.ipynb)
    code/
        nbNN.py               reference code for notebook NN's graded exercises
```

### `NN_slug.md`: written answers

One file per notebook, for example `03_sql_for_ml.md`, with two kinds of
section:

- **`## NN-K · short title`**: the answer to exercise K of notebook NN, with the
  reasoning, common mistakes, and for coded exercises a pointer to the
  reference function.
- **`## Checkpoint · NN`**: an answer to every checkpoint question at the end of
  notebook NN. These are interview-style; the answers show what a strong spoken
  answer would cover, not a script to memorise.

### `code/nbNN.py`: reference code, graded by CI

Coded exercises are always functions (a SQL exercise is a function that
returns the query string). The reference solution for exercise K of notebook
NN is the function `exK` in `code/nbNN.py`, with the same signature as the
notebook's stub. The test suite (`tests/test_solutions.py`) runs every
notebook grader against its reference solution, so **a reference solution that
does not pass its own grader fails CI**. The code here is therefore known to
pass the same checks you see in the notebook.

Passing the grader means the mechanics are right. It does not mean the
solution is the only correct one: if yours passes and differs, compare the
two and decide which you would rather defend.

## Confidence markers

Each answer is marked with how it was established:

| Marker | Meaning |
|---|---|
| [verified] | **Verified by running.** The code or the claim was executed and checked (for code, by CI against the grader) |
| [expected] | **Expected; depends on a run.** The answer describes what you should see, with ranges or directions rather than exact numbers, because the value depends on your run, library versions, or choices you made |
| [open-ended] | **Personal or open-ended.** There is no single right answer (a design choice, a written explanation, a reflection); the solution shows one good answer and what any good answer should contain |

Numbers quoted in solutions come from the course's **synthetic** TaskFlow data.
They describe that teaching dataset, not real customers.

## Related

- How exercises, graders and solutions fit together: [CONTRIBUTING.md](../CONTRIBUTING.md)
- Interview-style questions beyond the checkpoints: [docs/guides/interview-questions.md](interview-questions.md)
- What was executed, where, and what was not: [docs/guides/verification.md](verification.md)
