# About the solutions

Every chapter has worked answers to all of its exercises and checkpoint
questions, and every capstone project has a guided solution. The answers are
kept out of the labs deliberately, so that the exercises can be attempted
without seeing them, and they are kept in the same repository as the labs, so
that the two stay consistent. This appendix explains how the solutions are
organised, how the reference code is tested, and how the confidence of each
answer is marked.

---

## Attempt each exercise first

Reading a solution before attempting the exercise feels productive but
teaches very little. Recognising a correct answer is a different and much
easier skill than producing one, and both the work itself and interviews
require production. The following routine is recommended:

1. Attempt the exercise until its automatic check prints `[PASS]`, or for at
   least 15 minutes.
2. For written exercises and checkpoint questions, write down or say aloud
   your answer **before** opening the solutions.
3. Then read the solution and note anything you missed, or did differently
   and better. Disagreeing with a solution for a reason you can state is a
   good outcome.

If you are stuck for more than 20 minutes, read only the first line of the
solution (usually the key idea) and try again.

## Where the solutions are

Each chapter folder contains two solution files beside the chapter text and
the lab:

```
part1-foundations/ch05-leakage-safe-splits-and-baseline/
    README.md       chapter text
    notebook.ipynb  lab (built from notebook.py)
    solutions.md    worked answers to every exercise and checkpoint question
    solutions.py    reference code for the graded exercises
```

### `solutions.md`: written answers

Each chapter's `solutions.md` begins with a title of the form
`# Solutions · Chapter N · Title` and contains two kinds of section:

- **`## Exercise K · short title`**: the answer to exercise K of the
  chapter's lab, with the reasoning, common mistakes and, for coded exercises,
  a pointer to the reference function in `solutions.py`.
- **`## Checkpoint answers`**: an answer to every checkpoint question at the
  end of the lab. The checkpoint questions are in the style of interview
  questions; the answers show what a complete spoken answer would cover, not
  a script to memorise.

The exercises listed at the end of each chapter's text (`README.md`) are the
same exercises as in the lab, so their answers are in the same file.

### `solutions.py`: reference code, tested in CI

Coded exercises are always functions; a SQL exercise, for example, is a
function that returns the query string. The reference solution to exercise K
of Chapter N is the function `exK` in the chapter's `solutions.py`, with the
same signature as the stub in the lab.

The checks that the lab runs are in `src/grading/chNN.py` (imported in the
lab as `from grading import chNN as grade`, and called as `grade.exK(fn)`).
The test suite, `tests/test_solutions.py`, verifies for every chapter that:

- every exercise in the lab has a `## Exercise K ·` section in
  `solutions.md`, and a lab with a checkpoint has a `## Checkpoint answers`
  section;
- every check `exK` **accepts** the reference solution `exK`, so a reference
  solution that fails its own check fails CI;
- every check reports `[TODO]` (and does not crash) for an untouched stub.

The reference code is therefore known to pass the same checks that the reader
sees in the lab. Passing a check shows that the mechanics are right; it does
not mean that the reference solution is the only correct one. If your
solution passes and differs, compare the two and decide which you would
prefer to maintain and explain.

## Confidence markers

Each answer is marked with how it was established:

| Marker | Meaning |
|---|---|
| [verified] | **Verified by running.** The code or the claim was executed and checked (for code, by CI against the lab's check) |
| [expected] | **Expected; depends on a run.** The answer describes what should be observed, with ranges or directions rather than exact numbers, because the value depends on the run, the library versions, the hardware (in Part III, whether a GPU was used) or choices made by the reader |
| [open-ended] | **Personal or open-ended.** There is no single correct answer (a design choice, a written explanation, a reflection); the solution gives one good answer and states what any good answer should contain |

Numbers quoted in the solutions of Parts I and II come from the book's
**synthetic** data sets, TaskFlow and ShopStream. They describe those teaching
data sets, not real customers or users.

## Capstone solutions

Each capstone project in Part IV has a brief (`README.md`) and a guided
solution, `part4-capstones/capN-*/walkthrough.ipynb`, built from
`walkthrough.py` in the same folder in the same way as the chapter labs. The
walkthrough works through the project from the data download to the final
evaluation and its limitations. Because a capstone is meant to be built
independently, read the walkthrough after completing a first version of your
own, and use it to compare decisions rather than to copy code.

## Related appendices

- How labs, checks and solutions are written and kept consistent:
  [CONTRIBUTING.md](../CONTRIBUTING.md).
- Interview questions beyond the checkpoint questions:
  [Interview questions](interview-questions.md).
- What was executed, where, and what was not: [Verification](verification.md).
