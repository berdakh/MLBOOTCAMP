# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch58-full-interview-loop-and-portfolio/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 58 · Full interview loop & portfolio
#
# **Part III · Chapter 58 · about 90 minutes (plus the loop itself: about 5 hours, ideally on a separate day) · CPU is enough**
#
# Part I ended with a 45-minute mock interview (Chapter 18) and Part II with a
# four-round loop (Chapter 38). This lab simulates a complete loop as large
# technology companies commonly run it for ML engineers: a phone screen, then an
# onsite of two coding rounds, ML coding, ML system design, ML depth and a
# behavioural/leadership round, each with an interviewer script, a timer and a
# rubric whose anchors describe mid, senior and staff signals. It then packages
# the work of Parts I–III into a portfolio that a hiring manager can read in
# five minutes.
#
# By the end of this lab you should be able to:
#
# 1. run each round of a full loop from a script, on a timer, and score it against signal-level rubrics;
# 2. describe, in generic terms, what distinguishes mid, senior and staff answers in each round;
# 3. combine round scores into an overall and per-signal summary (Exercise 1) and keep a progress log across attempts;
# 4. structure capstone READMEs and generate one-page project briefs from your own artifacts, with every number tied to its split (Exercise 2);
# 5. audit your readiness across Parts I–III and choose what to practise next.
#
# **You will produce:** `artifacts/level3/loop/` (`plan.md`, `rubric.json`,
# `progress.jsonl` once you log a round) and `artifacts/level3/portfolio/`
# (`README_template.md`, `briefs.json`, one `brief_*.md` per project once
# Exercise 2 is done, `readiness.json`).
#
# **A note on honesty.** Interview formats, levels and pay differ between
# companies and change over time. Nothing here describes a specific
# company's process or ladder; the rubrics are generic "signals interviewers
# commonly look for", meant to calibrate practice.

# %% [markdown]
# Read first: [Chapter 58 text](README.md), which explains the structure of an interview loop, the scoring signals and the portfolio in more detail than this lab.

# %%
# --- Setup: works in Google Colab and on your own machine ------------------
# In Colab this clones the course repository and installs anything missing
# (30-60 seconds the first time). Locally it just finds the repository.
# Safe to re-run.
import os
import subprocess
import sys
from pathlib import Path

USE_DRIVE = False  # True: keep artifacts in Google Drive so they survive a disconnect

if "google.colab" in sys.modules:
    if not Path("/content/MLBOOTCAMP").exists():
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/berdakh/MLBOOTCAMP.git", "/content/MLBOOTCAMP"], check=True)
    os.chdir("/content/MLBOOTCAMP")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-part3.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# Part III is written for a GPU runtime: Runtime -> Change runtime type -> T4 GPU.
from scaleml.device import budget, describe, get_device

DEVICE = get_device()
BUDGET = budget(DEVICE)
print(describe(DEVICE))
print("budget:", BUDGET.as_dict())
if DEVICE.type == "cpu":
    print("No GPU found: running with the reduced CPU budget. Numbers will be much worse than on a T4.")

# %%
# --- Catch up: nothing to rebuild; this notebook reads whatever artifacts you have ---
import json
import random
import time
from datetime import datetime

import pandas as pd

from scaleml import design as ds
from grading import ch58 as grade

LOOP = ART / "level3" / "loop"
PORTFOLIO = ART / "level3" / "portfolio"
LOOP.mkdir(parents=True, exist_ok=True)
PORTFOLIO.mkdir(parents=True, exist_ok=True)
PROGRESS = LOOP / "progress.jsonl"
print(f"{len(ds.read_jsonl(PROGRESS))} round(s) already logged in {PROGRESS.name}")

# %% [markdown]
# **Expected:** `0 round(s) already logged in progress.jsonl` on a fresh
# runtime. With `USE_DRIVE = True`, your history survives between sessions;
# that matters here, because the log is meant to grow over weeks.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Loop | The full set of interview rounds for one candidate | phone screen + six onsite rounds |
# | Phone screen | A first, shorter round that gates the onsite | 45 minutes of coding |
# | Signal | One quality interviewers score, which recurs across rounds | `verification`, `communication` |
# | Rubric anchor | A short description of what a score or level sounds like | "designs the verification strategy" (staff) |
# | Mid / senior / staff | Generic levels of scope and independence, not any company's ladder | section 58.3 |
# | STAR | Situation, Task, Action, Result: a structure for behavioural stories | 4–5 stories in the behavioural round |
# | Weighted overall | A weighted mean of per-round means | 2.77 for the EXAMPLE attempt |
# | Weakest signal | The signal with the lowest mean across the rounds that score it | `verification` (2.0) |
# | JSON Lines log | A text file with one JSON record per line, appended and never rewritten | `progress.jsonl` |
# | Project brief | A one-page summary of a project with every number tied to its split | `brief_*.md` |
# | Split | Which data a number was measured on (held-out users, last 2 weeks, 5 seeds…) | "name the split" |
# | Readiness | Whether you can do a skill unaided, from a blank file, in interview time | `ready: 0/20` on a fresh run |

# %% [markdown]
# ## 58.1 · The loop
#
# | Round | Minutes | Format | Course material |
# |---|---|---|---|
# | **Phone screen · coding** | 45 | one or two data-structure/algorithm problems in a shared editor | Chapter 17, Chapters 33–34 |
# | **Onsite 1 · coding** | 45 | a medium problem plus a follow-up that changes a constraint | Chapters 33–34 |
# | **Onsite 2 · coding** | 45 | a different pattern (graphs, intervals, heaps, design-a-class) | Chapters 33–34 |
# | **Onsite 3 · ML coding** | 45 | implement a model component, loss or metric from scratch and test it | Chapters 31–32, 41, 44, 46 |
# | **Onsite 4 · ML system design** | 45–60 | an open product or platform prompt | Chapters 18, 37, 55–56 |
# | **Onsite 5 · ML depth** | 45 | drill-down on your CV's topics plus a paper or research discussion | Chapter 57 |
# | **Onsite 6 · behavioural / leadership** | 45 | 4–5 STAR stories with follow-ups; scope and influence | Chapter 18 (STAR), this chapter |
#
# Common variations (worth mentioning if asked about preparation): some
# loops swap one coding round for a second design round at senior and staff
# levels; some merge depth into the design round; some add a "project deep
# dive" or a presentation. Practise the components; the arrangement varies.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-x-interview-loop.svg" alt="The interview loop: phone screen, two coding rounds, ML coding, ML system design, ML depth, behavioural, then a debrief, each with the signals it mainly scores" width="100%">
#
# *What to notice:* the same signal names recur across rounds, so the debrief can see a *pattern* (e.g. weak verification in every coding round), which matters more than any one score.
#
# ## 58.2 · Interviewer scripts and the draw
#
# Each round has an opening prompt (drawn at random), a ladder of follow-ups
# the interviewer escalates through, and time cues. If you practise with a
# partner, give them `script(round)`; if alone, read the follow-ups only
# *after* your first answer.

# %%
ROUNDS = {
    "phone_coding": {
        "minutes": 45, "signals": ["problem_solving", "code_quality", "verification", "communication"],
        "prompts": ["Merge overlapping time intervals, then insert a new one in O(log n + k)",
                    "Top-k frequent ids in a stream; then over a sliding window of the last W events",
                    "Validate a binary search tree; then return the k-th smallest value",
                    "Group anagrams; then do it for a stream that does not fit in memory"],
        "follow_ups": ["What is the time and space complexity?", "What input breaks your solution?",
                       "Write two tests you would add", "How does this change if the input is 100x larger?"],
        "cues": "0-5 clarify · 5-30 code · 30-40 test by hand · 40-45 your questions",
    },
    "coding_1": {
        "minutes": 45, "signals": ["problem_solving", "code_quality", "verification", "communication"],
        "prompts": ["LRU cache with O(1) get/put; then add a TTL per key",
                    "Shortest path in a grid with obstacles; then allow removing up to k obstacles",
                    "Task scheduler with cooldowns: minimum time to run all tasks",
                    "Median of a data stream; then a sliding-window median"],
        "follow_ups": ["Why this data structure?", "Can you do better in time or memory?",
                       "Walk through your code on the edge case you chose", "What would you log in production?"],
        "cues": "aim for a working solution by minute 30; the follow-up is where senior signal appears",
    },
    "coding_2": {
        "minutes": 45, "signals": ["problem_solving", "code_quality", "verification", "communication"],
        "prompts": ["Course schedule: return a valid order or report a cycle",
                    "Minimum meeting rooms; then return the room assignment",
                    "Design a rate limiter class (sliding window); then make it thread-safe",
                    "Number of islands; then the number of distinct island shapes"],
        "follow_ups": ["Prove it terminates / is correct", "What changes with concurrent callers?",
                       "Refactor one part you are unhappy with", "How would you test this exhaustively?"],
        "cues": "say your plan before coding; leave 8 minutes to test",
    },
    "ml_coding": {
        "minutes": 45, "signals": ["ml_fundamentals", "code_quality", "verification", "communication"],
        "prompts": ["Multi-head causal self-attention with a padding mask (PyTorch); test the mask",
                    "A LoRA linear layer: forward, init, merge into the base weight; test merge equivalence",
                    "In-batch softmax loss for a two-tower model with a log-Q popularity correction",
                    "Top-p (nucleus) sampling with temperature; test against an exact distribution",
                    "NDCG@k with ties and empty lists; test against hand-computed cases",
                    "Logistic regression with L2 by gradient descent in NumPy; check the gradient numerically"],
        "follow_ups": ["Where can this overflow or underflow, and how do you prevent it?",
                       "How do you know it is correct beyond 'it runs'?",
                       "What is the complexity and memory in sequence length / batch size?",
                       "How would this run on a GPU at 100x the batch?"],
        "cues": "shapes first (write them as comments) · implement · test against a reference or invariant",
    },
    "ml_design": {
        "minutes": 50, "signals": ["quantification", "trade_offs", "operations", "ml_fundamentals", "communication"],
        "prompts": ["Ads ranking for a 300M-DAU feed (18.2)", "Video recommendations (18.3)",
                    "LLM customer-support assistant with human hand-off (18.4)",
                    "Real-time payment fraud at 5,000 TPS, 100 ms (18.5)",
                    "An ML platform for 50 teams: features, training, deployment, monitoring (17)",
                    "Search ranking for a marketplace (Chapter 37)"],
        "follow_ups": ["What is your peak QPS and how many replicas?", "Which label is biased, and how?",
                       "How do you know the new model is better before and after launch?",
                       "What fails first at 10x traffic?", "What would you build first, and what not at all?"],
        "cues": "0-5 clarify · 5-14 metrics + labels · 14-34 architecture + numbers + deep dives · 34-50 rollout, risks",
    },
    "ml_depth": {
        "minutes": 45, "signals": ["ml_fundamentals", "communication", "trade_offs"],
        "prompts": ["Why does warmup help transformer training? (19.2)",
                    "Batch size and learning rate: what scales, and where does it break? (19.3)",
                    "Double descent and why huge models generalise (19.5)",
                    "Position bias in implicit feedback and how to correct it",
                    "Fine-tuning vs RAG for a domain assistant",
                    "Critique the fictional SpectraFormer abstract (19.6)"],
        "follow_ups": ["Why? (repeat until the edge of knowledge)", "What experiment would show it?",
                       "When does that intuition fail?", "How does this change a production decision?"],
        "cues": "first 25 minutes on your CV topics, then a paper or research discussion",
    },
    "behavioural": {
        "minutes": 45, "signals": ["ownership", "leadership", "communication"],
        "prompts": ["A time you changed a team's technical direction without authority",
                    "A project that failed. What did you do, and what do you do differently now?",
                    "A disagreement with a senior colleague about an evaluation or a launch",
                    "A time you simplified something others had over-engineered",
                    "How you mentored someone through a hard problem",
                    "A decision you made with incomplete data that turned out wrong"],
        "follow_ups": ["What exactly did *you* do?", "What would have happened without you?",
                       "What did the other person think, and were they partly right?", "What is the measurable result?"],
        "cues": "4-5 stories, ~6 minutes each including follow-ups; STAR with 60% on Action",
    },
}

ATTEMPT = datetime.now().strftime("%Y-%m-%d %H:%M")
draw = {name: random.choice(r["prompts"]) for name, r in ROUNDS.items()}  # deliberately unseeded


def script(name: str) -> None:
    """Print the interviewer's script for one round (give this to your practice partner)."""
    r = ROUNDS[name]
    print(f"== {name} · {r['minutes']} min ==\nOpening prompt: {draw[name]}\nTime cues: {r['cues']}")
    print("Follow-up ladder (escalate in order):")
    for i, q in enumerate(r["follow_ups"], 1):
        print(f"  {i}. {q}")
    print("Score 1-4 on:", ", ".join(r["signals"]))


for name in ROUNDS:
    print(f"{name:>13} ({ROUNDS[name]['minutes']} min): {draw[name]}")
print()
script("ml_design")

# %% [markdown]
# **Expected:** one prompt per round (different every run: this is the
# notebook's one deliberately unseeded cell) and the full script for the
# design round. Total interview time is 320 minutes; a real onsite spreads
# it over a day with breaks, and a practice loop should too.
#
# ## 58.3 · Rubrics with leveling anchors
#
# Every round is scored 1–4 on a few **signals** (the same signal names
# recur across rounds, so you can see, e.g., your *communication* across the
# whole loop). The 1–4 scale is the same as Part II's (Chapter 38;
# `ds.SCORE_ANCHORS`). The table says what each signal tends to look like
# at three levels; use it to decide which level your answer *sounded* like.
#
# | Signal | Mid-level signal | Senior signal | Staff signal |
# |---|---|---|---|
# | **problem_solving** | reaches a working approach with hints | finds the efficient approach independently, explains why | reframes the problem, generalises the solution, anticipates the follow-up |
# | **code_quality** | correct, readable code | clean decomposition, idiomatic, handles edge cases unprompted | code another team could own: interfaces, invariants, failure behaviour |
# | **verification** | tests the happy path when asked | chooses tests that would catch real bugs; tests against a reference | designs the verification strategy (properties, invariants, monitoring in production) |
# | **ml_fundamentals** | correct definitions and standard methods | mechanisms and failure modes, with evidence | knows where the field's intuitions break and what experiment settles it |
# | **quantification** | some numbers when asked | QPS, latency, memory and cost estimated unprompted, assumptions stated | numbers drive decisions: sizing, phasing, cost curves, what to build first |
# | **trade_offs** | names alternatives | compares them with explicit criteria and picks one | says what evidence would change the choice; weighs organisational cost and reversibility |
# | **operations** | mentions monitoring | rollout, rollback, alerting, fallbacks, delayed labels | platform view: ownership, on-call load, cost governance, incident process |
# | **communication** | understandable, sometimes unstructured | structured, signposted, checks in with the interviewer | adapts depth to the audience, drives the conversation, summarises decisions |
# | **ownership** | describes own tasks | owns outcomes end to end, including mistakes | owns problems no one assigned, across team boundaries |
# | **leadership** | helps peers when asked | mentors, influences decisions in own team | sets direction across teams, builds consensus, makes others more effective |
#
# Write the rubric to disk next to today's plan:

# %%
SIGNAL_ANCHORS = {
    "problem_solving": ["reaches a working approach with hints", "efficient approach independently",
                        "reframes and generalises; anticipates follow-ups"],
    "code_quality": ["correct, readable", "clean, idiomatic, edge cases unprompted",
                     "ownable interfaces, invariants, failure behaviour"],
    "verification": ["happy-path tests when asked", "tests that catch real bugs; reference checks",
                     "verification strategy incl. production"],
    "ml_fundamentals": ["correct definitions", "mechanisms and failure modes with evidence",
                        "knows where intuitions break; designs the deciding experiment"],
    "quantification": ["numbers when asked", "QPS/latency/memory/cost unprompted with assumptions",
                       "numbers drive sizing, phasing and priorities"],
    "trade_offs": ["names alternatives", "explicit criteria and a choice",
                   "what evidence changes the choice; reversibility and org cost"],
    "operations": ["mentions monitoring", "rollout, rollback, fallbacks, delayed labels",
                   "ownership, on-call, cost governance, incidents"],
    "communication": ["understandable", "structured, signposted, checks in", "drives and adapts to the audience"],
    "ownership": ["own tasks", "outcomes end to end incl. mistakes", "unassigned problems across teams"],
    "leadership": ["helps when asked", "mentors; influences own team", "direction across teams; multiplies others"],
}
rubric = {"scale": ds.SCORE_ANCHORS, "levels": ds.LEVEL_ANCHORS,
          "signal_anchors_mid_senior_staff": SIGNAL_ANCHORS,
          "rounds": {n: r["signals"] for n, r in ROUNDS.items()}}
(LOOP / "rubric.json").write_text(json.dumps(rubric, indent=2))
plan = [f"# Full loop · attempt {ATTEMPT}", ""]
for name, r in ROUNDS.items():
    plan += [f"## {name} · {r['minutes']} min", "", f"Prompt: **{draw[name]}**", "",
             "Follow-ups: " + " / ".join(r["follow_ups"]), "", "Score 1-4 on: " + ", ".join(r["signals"]), ""]
(LOOP / "plan.md").write_text("\n".join(plan))
assert all(s in SIGNAL_ANCHORS for r in ROUNDS.values() for s in r["signals"]), "every round signal needs anchors"
print("wrote", sorted(p.name for p in LOOP.iterdir()))

# %% [markdown]
# **Expected:** `wrote ['plan.md', 'rubric.json']` (plus `progress.jsonl` once
# you have logged rounds).
#
# ## 58.4 · Timing and logging rounds
#
# `start(round)` when the clock starts; `log_round(round, scores, notes)` when
# it stops. Scores are whole numbers 1–4 for *every* signal of that round.
# The log is JSON Lines: one record per round, appended, never rewritten.

# %%
_started: dict = {}


def start(name: str) -> None:
    _started[name] = time.time()
    print(f"{name}: clock started ({ROUNDS[name]['minutes']} min). Prompt: {draw[name]}")


def log_round(name: str, scores: dict, notes: str = "") -> int:
    missing = set(ROUNDS[name]["signals"]) - set(scores)
    extra = set(scores) - set(ROUNDS[name]["signals"])
    if missing or extra:
        raise ValueError(f"{name}: score exactly {ROUNDS[name]['signals']} (missing {sorted(missing)}, "
                         f"unexpected {sorted(extra)})")
    ds.validate_scores(scores)
    minutes = round((time.time() - _started.pop(name)) / 60, 1) if name in _started else None
    n = ds.append_jsonl(PROGRESS, {"attempt": ATTEMPT, "date": datetime.now().date().isoformat(), "round": name,
                                   "prompt": draw[name], "minutes": minutes, "scores": scores, "notes": notes})
    print(f"logged {name} ({minutes} min); {n} round(s) in the log")
    return n


# After each round, e.g.:
# start("ml_coding")
# ... 45 minutes ...
# log_round("ml_coding", {"ml_fundamentals": 3, "code_quality": 3, "verification": 2, "communication": 3},
#           notes="forgot to test the all-padding row")
print("helpers ready: start(round), log_round(round, scores, notes), script(round)")

# %% [markdown]
# ## 58.5 · Scoring a loop
#
# A loop is not an average of everything. Interviewers write feedback per
# round; a hiring discussion then looks at each round *and* at signals that
# recur (weak communication in four rounds is a pattern, not noise). Your
# summary should show both.
#
# > **Exercise 1.** Write `loop_score(scores_by_round, weights)`.
# > `scores_by_round` maps round → {signal: score 1–4}; `weights` maps round
# > → non-negative weight. Return a dict with:
# > `overall` (weighted mean of the **per-round means**, weights normalised over
# > the rounds present), `by_round` (round → mean of its signals), `by_signal`
# > (signal → mean over the rounds where it was scored) and `weakest_signal`
# > (the signal with the lowest `by_signal`; ties broken alphabetically).
# > Raise `ValueError` if there are no rounds, a round has no scores, a score
# > is not a whole number 1–4, a round present has no weight, a weight is
# > negative, or the weights of the rounds present sum to 0. Ignore weights
# > for rounds that are not present.

# %%
def loop_score(scores_by_round, weights):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(loop_score)

# %% [markdown]
# The weights below are **illustrative** (design counted 1.5×, the phone
# screen 0.5× because it is a gate). Companies weigh rounds differently, and
# in real hiring discussions a single very weak round can matter more than
# any average. Here is the summary of your log, or of a clearly labelled
# **EXAMPLE** attempt if your log is empty (nothing is saved):

# %%
WEIGHTS = {"phone_coding": 0.5, "coding_1": 1.0, "coding_2": 1.0, "ml_coding": 1.0, "ml_design": 1.5,
           "ml_depth": 1.0, "behavioural": 1.0}
history = ds.read_jsonl(PROGRESS)
if not history:
    example = {"phone_coding": [3, 3, 2, 3], "coding_1": [3, 2, 2, 3], "coding_2": [2, 3, 2, 2],
               "ml_coding": [3, 3, 2, 3], "ml_design": [3, 3, 3, 3, 2], "ml_depth": [4, 3, 3],
               "behavioural": [3, 3, 3]}
    history = [{"attempt": "EXAMPLE", "round": r, "scores": dict(zip(ROUNDS[r]["signals"], s))}
               for r, s in example.items()]
attempts: dict = {}
for rec in history:
    attempts.setdefault(rec["attempt"], {})[rec["round"]] = rec["scores"]

rows = []
for attempt, by_round in attempts.items():
    try:
        summary = loop_score(by_round, WEIGHTS)
    except NotImplementedError:
        print("(finish Exercise 1 to see the weighted overall and per-signal summary; showing round means only)")
        summary = {"overall": None, "by_round": {r: sum(s.values()) / len(s) for r, s in by_round.items()},
                   "by_signal": {}, "weakest_signal": None}
    rows.append({"attempt": attempt, "overall": summary["overall"], "weakest_signal": summary["weakest_signal"],
                 **{r: round(v, 2) for r, v in summary["by_round"].items()}})
pd.DataFrame(rows).set_index("attempt")

# %% [markdown]
# **Expected:** on a fresh run, one **EXAMPLE** row of round means (2.25 for
# coding_2, 3.33 for ml_depth, …) and, until Exercise 1 is done, the note
# saying the weighted summary needs it. With the reference solution the
# EXAMPLE's overall is about **2.77** and the weakest signal is
# **verification** (2.0 in all four coding rounds), so next week's practice
# plan follows directly: write tests before being asked, in every coding round.
#
# > **Intuition.** A loop is scored like a decathlon judged event by event, not like a single
# > stopwatch: the committee reads each round's write-up, then looks across rounds for a recurring
# > weakness. The weighted overall is your summary of that, not the decision rule. *Where the analogy
# > breaks down:* in a decathlon points add up; in hiring one very weak round (or a "no hire" on a
# > core signal) can outweigh a high average.
#
# ## 58.6 · Packaging the portfolio
#
# A hiring manager spends minutes, not hours, on a portfolio. Each project
# README (Part I's TaskFlow churn service, Part II's shoprank ranking
# system, a Part III project and the capstones) should answer, in this order:

# %%
README_TEMPLATE = """# <Project name>: <one-line outcome, with the metric and the split>

> Public-data demonstration built for learning. Not work done for or endorsed by any company.

## Problem
Who would use this, what decision it supports, and what happens today without it (3-4 sentences).

## Results
| Metric | Value | Split / evaluation | Baseline |
|---|---|---|---|
| <primary metric> | TBD — fill from your run | <e.g. held-out last 2 weeks, users bootstrapped> | <baseline value> |

One sentence on uncertainty (interval, seeds) and one on the main limitation.

## How it works
A diagram or 5-line description of the pipeline: data -> features -> model -> serving -> monitoring.

## Reproduce
`make setup && make test && make train && make evaluate` - with data download instructions and expected runtime.

## Decisions and trade-offs
3-5 bullets: what you chose, the alternative, the criterion, the evidence.

## What I would do next
2-3 bullets, honest about limits.

## Data and licence
Source, licence, and anything the data terms forbid.
"""
(PORTFOLIO / "README_template.md").write_text(README_TEMPLATE)
print(README_TEMPLATE[:400], "...")

# %% [markdown]
# **Expected:** the first lines of the template, saved as
# `README_template.md`. The *Results* table comes second, right after the
# problem, because it is what readers look for first; every number names its
# split; empty cells say **TBD — fill from your run**, never a guess.
#
# How to present each level in one line (on a CV, or in the first minute of a
# project deep-dive):
#
# | Project | What it proves | The one number to quote (from *your* run) |
# |---|---|---|
# | Part I · TaskFlow churn service | leakage-safe evaluation, a tested API, monitoring and rollback | test PR-AUC with its interval, and p95 latency |
# | Part II · shoprank ranking cascade | retrieval + LTR at scale, position-bias correction, pre-registered evaluation | held-out NDCG/IPS metric with grouped-bootstrap interval; cascade p99 |
# | Part III · e.g. a fine-tuned classifier, a retrieval system or a CTR model | deep models on a GPU budget, efficiency, evaluation of LLM systems | accuracy/recall vs baseline on the named split; latency or cost per 1k requests |
# | Capstones | applying all of it to a new public dataset | the capstone's pre-registered primary metric |
#
# ### A one-page project brief from your artifacts
#
# A brief is the CV-sized version of a README. The generator below looks for
# result files that earlier notebooks wrote in **your** artifact directory
# and turns them into structured project dicts. It quotes numbers only from
# files it finds; anything else stays **TBD**, and every number it finds is
# marked "name the split" until you do.
#
# > **Intuition.** A number without its split is a nutrition label without the serving size:
# > "94% accuracy" means nothing until you know whether it was per trial on seen subjects or per new
# > user on a held-out month. *Where the analogy breaks down:* a serving size is a convention, while a
# > split encodes the *claim* (new users? new time?); choosing the wrong one changes the number and
# > what it proves.

# %%
def numeric_leaves(obj, prefix="", limit=6):
    """Up to `limit` (name, number) pairs from a JSON object, depth-first."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += numeric_leaves(v, f"{prefix}{k}.", limit - len(out))
            if len(out) >= limit:
                break
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append((prefix.rstrip("."), obj))
    return out[:limit]


def results_from(path: Path, key: str | None = None) -> list[dict]:
    if not path.exists():
        return []
    obj = json.loads(path.read_text())
    obj = obj.get(key, obj) if key and isinstance(obj, dict) else obj
    rel = path.relative_to(ART).as_posix()
    return [{"metric": name, "value": value, "split": f"TBD: name the split (from {rel})"}
            for name, value in numeric_leaves(obj)]


def depth_results(path: Path) -> list[dict]:
    """Notebook 19's file has a known layout, so its splits can be named exactly."""
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    warm = {(r["peak_lr"], r["warmup_steps"]): r["val_loss"] for r in d.get("warmup", [])}
    noise = {r["label_noise"]: r for r in d.get("label_noise", [])}
    out = [{"metric": f"TinyGPT val loss, LR 3e-2, warmup {w} steps", "value": warm.get((0.03, w)),
            "split": "Tiny Shakespeare last 10% (contiguous), 1 seed"} for w in (0, 100)]
    if 0.2 in noise:
        out += [{"metric": f"clean test acc, 20% label noise, {k.replace('test_acc_', '')}", "value": noise[0.2][k],
                 "split": "synthetic clean test set (3,000 rows), seed 0"}
                for k in ("test_acc_final", "test_acc_early_stopped")]
    return out


PROJECTS = [
    {"title": "TaskFlow churn prediction service (Part I)",
     "problem": "Rank 10M SaaS accounts by churn risk so customer success can intervene within capacity.",
     "data": "Synthetic, checksummed TaskFlow event logs; time-based split with an embargo.",
     "approach": "Leakage-safe features, logistic regression vs gradient boosting, cost-based threshold, "
                 "FastAPI service with contract tests, drift monitoring and rollback.",
     "results": results_from(ART / "final_report.json", "test"),
     "limitations": ["synthetic data: absolute numbers do not transfer", "single company-shaped scenario"],
     "next_steps": ["apply the same pipeline to a capstone dataset"]},
    {"title": "shoprank: retrieval and learning-to-rank cascade (Part II)",
     "problem": "Rank items for users of a simulated marketplace under position-biased click logs.",
     "data": "Simulated marketplace logs (shoprank), train/valid/test weeks; exploration traffic for propensities.",
     "approach": "Candidate generation (two-tower + co-visitation), LambdaMART ranker, IPS evaluation, "
                 "pre-registered test-week evaluation with grouped bootstrap.",
     "results": results_from(ART / "level2" / "final" / "test_results.json"),
     "limitations": ["simulated world with known ground truth", "offline only: no online A/B"],
     "next_steps": ["interleaving experiment design", "a sequence model for session intent"]},
    {"title": "Depth experiments: warmup, batch scaling, label noise, double descent (Part III)",
     "problem": "Turn four ML-depth interview topics into measured evidence.",
     "data": "Tiny Shakespeare (public domain text) and synthetic classification/regression data.",
     "approach": "Small controlled experiments on CPU/T4 with fixed seeds (Chapter 57).",
     "results": depth_results(ART / "level3" / "nb19" / "depth_demos.json"),
     "limitations": ["toy scale; one seed in the warmup sweep"], "next_steps": ["repeat the warmup sweep over 3 seeds"]},
]
found = ds.scan_artifacts(ART)
print(f"{len(found)} result-like files in your artifact directory")
for p in PROJECTS:
    print(f"- {p['title']}: {len(p['results'])} number(s) found" + ("" if p["results"] else " -> results TBD"))
(PORTFOLIO / "briefs.json").write_text(json.dumps(PROJECTS, indent=2, default=str))
print("wrote", (PORTFOLIO / "briefs.json").relative_to(ART))

# %% [markdown]
# **Expected:** in a fresh runtime (as in our verification run) only this
# notebook's own files exist (`3 result-like files`: `plan.md`,
# `rubric.json`, `README_template.md`), every project says `results TBD`, and
# `briefs.json` is written. After running Chapters 16, 38 and 57 *in the
# same artifact directory* (e.g.
# with `USE_DRIVE = True`), their numbers appear here, each waiting for you
# to name its split.
#
# > **Exercise 2.** Write `project_brief(project)` that renders one project
# > dict (keys `title`, `problem`, `data`, `approach`, `results`,
# > `limitations`, `next_steps`, optional `links` mapping name → URL) as
# > Markdown: a `# {title}` line, then the sections `## Problem`, `## Data`,
# > `## Approach`, `## Results`, `## Limitations`, `## Next steps` (and
# > `## Links` if given), in that order. Results are a table with the header
# > `| Metric | Value | Split |`; a value of `None` is written
# > `TBD — fill from your run`; floats are formatted with `f"{v:.4g}"`; if
# > there are no results the section says `TBD — fill from your run`.
# > Limitations, next steps and links are bullet lists (`- `; links as
# > `- [code](https://example.org/repo)`). Raise `ValueError` if a required key is missing, or if a
# > result has a value but no split: **every number names its split**.

# %%
def project_brief(project):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(project_brief)

# %% [markdown]
# Render and save the briefs (this needs Exercise 2):

# %%
try:
    for i, p in enumerate(PROJECTS, 1):
        md = project_brief(p)
        (PORTFOLIO / f"brief_{i}.md").write_text(md)
    print(md)
    print("wrote", [f"brief_{i}.md" for i in range(1, len(PROJECTS) + 1)])
except NotImplementedError:
    print("finish Exercise 2 to render brief_1.md ... brief_3.md (briefs.json is already saved)")

# %% [markdown]
# **Expected:** before Exercise 2, the "finish Exercise 2" line; after it, the
# third brief printed in Markdown (with `TBD — fill from your run` in a fresh
# runtime) and three `brief_*.md` files.
#
# ## 58.7 · Readiness across Parts I–III
#
# Score yourself with the full
# [readiness rubric](https://github.com/berdakh/MLBOOTCAMP/blob/main/appendices/readiness-rubric.md)
# (Explain / Code / Troubleshoot, levels 1–4). The checklist below is the
# short version, one line per skill a loop will test; set each to `True` only
# if you have **done it unaided, from a blank file, in interview time** (the
# rubric's level 3). Re-run weekly.

# %%
READINESS = {
    "L1 · leakage-safe split + baseline from a blank file": False,
    "L1 · group bootstrap CI and cost-based threshold": False,
    "L1 · FastAPI endpoint with a contract test": False,
    "L1 · drift check (PSI) and a rollback plan": False,
    "L2 · point-in-time feature join in SQL/pandas": False,
    "L2 · NDCG/MRR/recall@k from scratch, ties handled": False,
    "L2 · position bias: propensities + IPS, explained and coded": False,
    "L2 · A/B power and sample size by hand": False,
    "L2 · 4 coding patterns (two pointers, heaps, graphs, intervals) in 25 min each": False,
    "L3 · attention with masks + KV cache, explained and coded": False,
    "L3 · memory arithmetic for training (weights, grads, optimiser, activations)": False,
    "L3 · LoRA from scratch and when to use PEFT": False,
    "L3 · contrastive two-tower loss with in-batch negatives": False,
    "L3 · DDP: ranks, all-reduce, samplers, checkpoints": False,
    "L3 · serving: batching, latency vs throughput, cost per 1k requests": False,
    "L3 · CTR calibration + delayed feedback, explained": False,
    "L3 · LLM system evaluation and guardrails, explained": False,
    "L3 · one 45-minute design (17/18) with numbers, timed": False,
    "L3 · depth answers for 19.1 with your own experiment numbers": False,
    "Loop · a full logged loop with every round >= 3": False,
}
done = sum(READINESS.values())
print(f"ready: {done}/{len(READINESS)} ({done / len(READINESS):.0%})")
gaps = [k for k, v in READINESS.items() if not v]
print("next three to practise:", *gaps[:3], sep="\n  ")
_ = (PORTFOLIO / "readiness.json").write_text(json.dumps(
    {"date": datetime.now().date().isoformat(), "ready": done, "total": len(READINESS), "items": READINESS}, indent=2))

# %% [markdown]
# **Expected:** `ready: 0/20 (0%)` on a fresh run, then the first three gaps.
# Be strict: "I recognise it" is level 1 on the rubric, not level 3.
#
# ## 58.8 · "Tell me about yourself"
#
# Most loops open with it, and it frames everything after. Two minutes,
# four beats:
#
# 1. **Who you are in one line**, as the role sees it ("an ML engineer with a
#    background in time-series modelling and signal processing").
# 2. **Two proof points** with your role and a measured number, each with
#    its split: one research, one engineering (a Part I–III project or a capstone).
# 3. **What you are good at**, tied to the job description's needs (e.g.
#    rigorous evaluation, deep models under compute limits, ranking systems).
# 4. **Why this role, now**, in one or two sentences, without criticising
#    your current or past environment.
#
# > **Exercise 3 (written).** Write your two-minute "tell me about yourself"
# > for an ML engineer role, in the four beats above, using only numbers you
# > measured (name the split). Record it, time it (1:45–2:15), and log a
# > `behavioural` round for it with honest scores.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## Key equations
#
# **1 · Scoring a loop** (Exercise 1).
#
# $$
# m_r = \frac{1}{\lvert S_r \rvert} \sum_{s \in S_r} x_{r,s}, \qquad \text{overall} = \frac{\sum_r w_r\, m_r}{\sum_r w_r}, \qquad \bar{x}_s = \frac{1}{\lvert R_s \rvert} \sum_{r \in R_s} x_{r,s}
# $$
#
# *Symbols:* $x_{r,s} \in \{1, 2, 3, 4\}$ the score of signal $s$ in round $r$, $S_r$ the signals scored in round
# $r$, $R_s$ the rounds that score signal $s$, $w_r$ round weights. *Example (the EXAMPLE attempt):*
# $(0.5 \cdot 2.75 + 2.5 + 2.25 + 2.75 + 1.5 \cdot 2.8 + 3.33 + 3.0) / 7 = 19.41 / 7 = 2.77$.
#
# **2 · "What would it cost to serve at 1,000 QPS?"** (Checkpoint 6), with the fleet formula from Chapter 56.
#
# $$
# n = \max\!\left(2, \left\lceil \frac{\text{QPS}_{\text{peak}}}{\text{QPS}_{\text{replica}}\,(1 - h)} \right\rceil\right), \qquad \text{cost}_{1k} = \frac{n \cdot p \cdot 730}{\overline{\text{QPS}} \cdot 3600 \cdot 730 / 1000}
# $$
#
# *Example (assumptions, say them aloud):* peak 1,000 QPS, 100 QPS per replica measured, $h = 0.3$:
# $n = \lceil 14.3 \rceil = 15$; at \$0.50/hour that is \$5,475 a month; at an average of 500 QPS
# (1.31 billion requests a month) it is about \$0.0042 per 1,000 requests.
#
# **3 · Readiness.** $\text{readiness} = \#\{\text{skills done unaided}\} / \#\{\text{skills}\}$; track it weekly, not once.
#
# The cell below recomputes 1 and 2 by hand and with `scaleml.design`.

# %%
# Key equations, recomputed (instant). The EXAMPLE scores are copied from 58.5.
import math

example_scores = {"phone_coding": [3, 3, 2, 3], "coding_1": [3, 2, 2, 3], "coding_2": [2, 3, 2, 2],
                  "ml_coding": [3, 3, 2, 3], "ml_design": [3, 3, 3, 3, 2], "ml_depth": [4, 3, 3],
                  "behavioural": [3, 3, 3]}
means = {r: sum(v) / len(v) for r, v in example_scores.items()}
overall = sum(WEIGHTS[r] * m for r, m in means.items()) / sum(WEIGHTS[r] for r in means)
print("round means:", {r: round(m, 2) for r, m in means.items()}, f"| weighted overall {overall:.2f}")

n_rep = max(2, math.ceil(1_000 / (100 * (1 - 0.3))))
monthly = n_rep * 0.50 * 730
print(f"replicas: by hand {n_rep} | ds.replicas_needed {ds.replicas_needed(1_000, 100, target_utilisation=0.7)}")
print(f"monthly {monthly:,.0f} | per 1k requests: by hand {monthly / (500 * 3600 * 730) * 1000:.4f} | "
      f"ds.cost_per_1k_requests {ds.cost_per_1k_requests(ds.monthly_fleet_cost(n_rep, 0.50), 500):.4f}")

# %% [markdown]
# **Expected:** round means 2.75, 2.5, 2.25, 2.75, 2.8, 3.33 and 3.0, a weighted overall of `2.77` (the
# value Exercise 1's reference solution gives in section 58.5); `15` replicas both ways; a monthly cost of
# 5,475 and `0.0042` per 1,000 requests both ways.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ValueError: ml_design: score exactly [...]` from `log_round` | a signal missing or misspelled | copy the signal names printed by `script(round)` |
# | `ValueError: scores must be whole numbers 1-4` | a 3.5 or a string `"3"` | whole numbers only; half points hide indecision |
# | `progress.jsonl` gone after a break | Colab runtime recycled | `USE_DRIVE = True` in the setup cell before your next loop |
# | Every brief says `results TBD` | earlier notebooks ran in other runtimes (fresh artifact directories) | use Drive for artifacts, or re-run those notebooks in this runtime; never type numbers you did not produce |
# | Exercise 2 `ValueError` on your own data | a number without a split | name the split ("held-out last 2 weeks, 5 seeds"); that is the point |
# | Overall score looks fine, but one round is 1–2 | averaging hides a weak round | read `by_round` first; practise that round next week |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under two minutes each:
#
# 1. Give your two-minute "tell me about yourself". Then: which part would an interviewer probe first?
# 2. In a coding round, what separates a senior signal from a mid-level one when both candidates reach a working solution?
# 3. What would make your ML system design answer read as staff rather than senior? Give two concrete additions for the ads case.
# 4. Your loop log shows *verification* as your weakest signal across four rounds. What is your practice plan for the next two weeks, and how will you know it worked?
# 5. Walk through your strongest portfolio project in three minutes: problem, the one number with its split, the main trade-off and what you would do next.
# 6. An interviewer asks what your Part III model would cost to serve at 1,000 QPS. Answer with the arithmetic and your assumptions.
#
# **Next:** this completes Part III. Choose a capstone from
# [Part IV · Capstones](../../part4-capstones/README.md),
# score yourself honestly with the
# [readiness rubric](https://github.com/berdakh/MLBOOTCAMP/blob/main/appendices/readiness-rubric.md),
# and run one logged loop per week until every round scores 3 or more on fresh prompts.
