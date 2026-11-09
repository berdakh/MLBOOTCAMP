# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part4-capstones/cap5-content-moderation/walkthrough.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, then Runtime → Run all (CPU is enough; a T4 GPU makes the optional extension possible). The first code cell sets everything up.*

# %% [markdown]
# # Capstone 5 · Content moderation triage (Civil Comments) — guided walkthrough
#
# **Capstone · about 8 minutes to run on a laptop CPU (plus a 140 MB download) · CPU core, optional GPU extension**
#
# This notebook is a complete worked solution of the capstone's minimum viable product, written as if a
# tutor were sitting next to you. A classifier scores public comments so that a fixed number of human
# reviewers see as many toxic comments as possible, and we check whether it over-flags comments that
# merely mention an identity group. The model **ranks comments for human review**; it never removes
# anything on its own.
#
# **How to use this walkthrough.** Attempt each step yourself first. Every step states its goal, then a
# **Your turn.** box asks you to make a decision and suggests how long to spend on it. Only then read
# **Our solution.**, which shows our code and explains why we chose it, which alternatives we
# considered and which mistakes are common. **Check.** cells assert properties that any correct
# solution must satisfy (for example, that no training comment also appears in the test split).
#
# The steps follow the brief's milestones ([README.md](README.md)): 1. framing; 2. data acquisition;
# 3. data audit; 4. a leakage-safe split; 5. baselines; 6. a compact pretrained model and calibration;
# 7. evaluation with uncertainty; 8. subgroup and error analysis; 9. a small API; 10. write-up.
#
# **Data, licence and content warning.** We use the CivilComments-WILDS release on the Hugging Face Hub,
# [`pietrolesci/civilcomments-wilds`](https://huggingface.co/datasets/pietrolesci/civilcomments-wilds) at a
# pinned revision (the same source as Chapter 53): Civil Comments with the toxicity and identity
# annotations from the Jigsaw *Unintended Bias in Toxicity Classification* data and the WILDS train /
# validation / test splits. The original data are released under **CC0** (see the
# [`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments) dataset card); check the
# card and the WILDS terms yourself, and cite Borkan et al. (2019), arXiv:1903.04561. The comments are
# real user text and some are offensive: **this notebook never prints comment text**, only ids, lengths
# and scores. This is a public-data demonstration; it is not work for, or endorsed by, any platform.

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

# Part III notebooks are written for a GPU runtime, but this capstone's core runs on a CPU.
from scaleml.device import budget, describe, get_device

DEVICE = get_device()
BUDGET = budget(DEVICE)
print(describe(DEVICE))
print("budget:", BUDGET.as_dict())
if DEVICE.type == "cpu":
    print("No GPU found: running the CPU core. The optional fine-tuning extension is skipped.")

# %%
# --- Imports, run mode and sizes ----------------------------------------------
import json
import logging
import re
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scaleml import finetune as ft

# MLBOOTCAMP_CAPSTONE_SAMPLE=1 is for automated checks only: it replaces the download with a small
# SYNTHETIC, non-offensive table of the same columns (Step 2). Its numbers mean nothing.
SAMPLE = os.environ.get("MLBOOTCAMP_CAPSTONE_SAMPLE") == "1"
OUT = ART / "capstones" / "cap5"
OUT.mkdir(parents=True, exist_ok=True)
T_START = time.time()
TIMINGS = {}
SEED = 0
SIZES = {  # comments used at each stage; the TF-IDF model is cheap, the transformer is not
    "gpu": dict(tfidf_train=200_000, emb_train=40_000, val_sub=20_000, test_sub=40_000),
    "cpu": dict(tfidf_train=100_000, emb_train=8_000, val_sub=5_000, test_sub=10_000),
    "sample": dict(tfidf_train=3_000, emb_train=800, val_sub=500, test_sub=800),
}["sample" if SAMPLE else ("gpu" if DEVICE.type == "cuda" else "cpu")]
print("mode:", "SYNTHETIC SAMPLE (automated checks only)" if SAMPLE else "real data", "| sizes:", SIZES)
print(f"CPU threads available to torch: {torch.get_num_threads()}")

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Toxic label** | the comment is called toxic if at least half of its raters said so | `toxicity ≥ 0.5` |
# | **Fractional label** | the share of raters who applied a label; 0.5 means they split evenly | 0.3, 0.5, 0.8 |
# | **Review budget** | the share of all comments people can review | 1%, 2%, 5% of traffic |
# | **Operating point** | the threshold, and the precision and recall that come with it | the score above which a comment is queued |
# | **Precision (queue)** | the share of queued comments that are toxic: reviewer efficiency | 0.7 = 7 in 10 reviews are useful |
# | **Recall** | the share of all toxic comments that reach a reviewer | 0.3 = 70% of toxic comments are missed |
# | **PR-AUC (average precision)** | ranking quality at the top; a random ranking scores the positive rate | ≈ 0.11 for random scores |
# | **Calibration** | a score of 0.3 should mean "toxic about 30% of the time" | Platt scaling on validation |
# | **Subgroup** | comments that mention one identity (annotator score ≥ 0.5) | comments mentioning *muslim* |
# | **FPR / FNR** | share of non-toxic comments queued / share of toxic comments missed | per subgroup, at the operating point |
# | **BPSN AUC** | can the model rank toxic background comments above non-toxic subgroup comments? | low = identity over-flagging |
# | **Linear probe** | a frozen pretrained encoder plus a logistic regression on its embeddings | MiniLM + LR |

# %% [markdown]
# ## Key equations
#
# With labels $y_i \in \{0, 1\}$, scores $s_i$ and a threshold $\tau$:
#
# $$
# \operatorname{precision}(\tau) = \frac{\#\{i : s_i \ge \tau,\ y_i = 1\}}{\#\{i : s_i \ge \tau\}},
# \qquad
# \operatorname{recall}(\tau) = \frac{\#\{i : s_i \ge \tau,\ y_i = 1\}}{\#\{i : y_i = 1\}}
# $$
#
# *In words:* precision is the share of the queue that deserves review; recall is the share of toxic
# comments that reach the queue. Example: a queue of 2,000 with 1,400 toxic, out of 11,000 toxic
# comments in total: precision 0.70, recall ≈ 0.13.
#
# **Average precision** (PR-AUC as computed by scikit-learn), with comments sorted by score and $P_k$, $R_k$
# the precision and recall of the top $k$:
#
# $$
# \operatorname{AP} = \sum_k (R_k - R_{k-1})\, P_k
# $$
#
# *In words:* the precision averaged over the positions where a toxic comment is found. A random
# ranking gives the positive rate (≈ 0.11 here), not 0.5.
#
# **Capacity threshold.** With review budget $q$ (share of traffic) and validation scores $s^{\text{val}}$:
#
# $$
# \tau_q = \operatorname{quantile}_{1-q}\left(s^{\text{val}}\right)
# $$
#
# *In words:* the score above which the top $q$ of validation comments lie. Example: $q = 0.02$ takes the
# 98th percentile. If traffic or prevalence shifts, the realised queue share moves away from $q$.
#
# **Platt scaling** fits $p = \sigma(a \cdot \operatorname{logit}(s) + b)$ on validation, and the **Brier score**
# $\frac{1}{n}\sum_i (p_i - y_i)^2$ measures how well probabilities match outcomes (lower is better).
#
# **Subgroup error rates** at threshold $\tau$, for subgroup $g$:
#
# $$
# \operatorname{FPR}_g = \frac{\#\{i \in g : s_i \ge \tau,\ y_i = 0\}}{\#\{i \in g : y_i = 0\}},
# \qquad
# \operatorname{FNR}_g = \frac{\#\{i \in g : s_i < \tau,\ y_i = 1\}}{\#\{i \in g : y_i = 1\}}
# $$
#
# **BPSN AUC** for subgroup $g$ is the ROC-AUC on the set {non-toxic comments in $g$} ∪ {toxic comments
# not in $g$}; **BNSP AUC** uses {toxic in $g$} ∪ {non-toxic not in $g$}. *In words:* a low BPSN means
# benign mentions of $g$ score like real toxicity elsewhere.

# %% [markdown]
# ## Step 1 · Framing
#
# > **Your turn (15 minutes).** Write down: the unit of prediction; the label; what the score is used
# > for; the operating point and how it is set; the primary metric; and what the model must never do.
#
# **Our solution.**
#
# * **Unit:** one comment (text only). **Label:** `toxic = toxicity ≥ 0.5`; the raw fraction is kept
#   for analysis.
# * **Use:** the score orders a human review queue; bands are `auto_approve`, `human_review` and, for
#   the very top, `priority_review`. Never automatic removal.
# * **Operating point:** set by capacity. We **assume** a platform receives V = 100,000 comments a day
#   and reviewers can handle 1%, 2% or 5% of them; these are illustrative assumptions, not any
#   platform's numbers. The threshold for budget $q$ is the $(1-q)$ quantile of validation scores.
#   For the subgroup analysis and the API we pick the **5%** budget.
# * **Metrics:** PR-AUC (primary, because only ~11% of comments are toxic), ROC-AUC (comparable with the
#   literature), precision and recall at each budget, Brier score, and subgroup FPR/FNR with the bias
#   AUCs of Borkan et al.
#
# **Common mistakes.** Reporting accuracy (predicting "never toxic" scores ~89%); choosing the threshold
# 0.5 by habit (it has no relation to reviewer capacity); and treating identity columns as features.

# %%
BUDGETS = [0.01, 0.02, 0.05]
OPERATING_BUDGET = 0.05
DAILY_VOLUME = 100_000  # assumption for the queue table, not a real platform's traffic
IDENTITIES = ["male", "female", "transgender", "homosexual_gay_or_lesbian", "christian", "jewish", "muslim",
              "black", "white", "asian", "latino", "psychiatric_or_mental_illness"]
MIN_N, MIN_TOXIC = 100, 20  # subgroups below these counts are flagged, not interpreted
if os.environ.get("MLBOOTCAMP_CAPSTONE_SAMPLE") == "1":
    MIN_N, MIN_TOXIC = 20, 3  # the synthetic sample is tiny; these smaller limits only exercise the code

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap5-triage-flow.svg" alt="Comment, score, calibrate and a capacity threshold route each comment to auto-approve or to a human review queue; reviewer decisions return as labels" width="100%">
#
# *What to notice:* the model's output is an order and a band, not a decision, and only queued
# comments ever get new human labels (a selection bias to remember when monitoring).

# %% [markdown]
# ## Step 2 · Data acquisition
#
# **Goal.** Download a pinned version of the data, record what it is, and print the licence.
#
# > **Your turn (15 minutes).** Read the dataset card. Which source has identity labels? Pin a
# > revision so that the data cannot change under you.
#
# **Our solution.** The Hugging Face default config of `google/civil_comments` has no identity columns,
# so we use the WILDS release, which has them for all of its 448,000 comments. One Parquet file holds
# all three splits (column `split`). The revision hash pins the exact file.
#
# The offline sample path writes a small **synthetic** table with the same column names, built from
# neutral template sentences, so the notebook can be tested without network access. It is not Civil
# Comments.

# %%
from huggingface_hub import hf_hub_download

REPO, REVISION = "pietrolesci/civilcomments-wilds", "c227534cc0a34cf21db6a0bf0edd0f9050c8b305"
COLUMNS = ["id", "comment_text", "split", "toxicity", "identity_annotator_count", "toxicity_annotator_count", *IDENTITIES]
KEYWORDS = ["idiot", "idiots", "stupid", "moron", "morons", "dumb", "fool", "fools", "pathetic", "loser",
            "losers", "ignorant", "hypocrite", "liar", "liars", "clown", "clowns", "jerk", "crap", "disgusting"]


def make_synthetic_civil_comments(path: Path, n: int = 6000, seed: int = 0) -> Path:
    """SYNTHETIC, non-offensive stand-in with the columns this notebook reads. For tests only."""
    rng = np.random.default_rng(seed)
    topics = ["the budget", "the new bridge", "this article", "the school board", "the election", "the weather"]
    nice = ["I agree with {t}.", "Thanks for writing about {t}.", "I am not sure about {t}, but it is worth a debate.",
            "Good points on {t}; the numbers look right to me."]
    rude = ["Only a {k} would support {t}.", "What a {k} take on {t}.", "You are a {k} if you believe {t} works."]
    group_words = {"christian": "christian", "muslim": "muslim", "jewish": "jewish", "female": "women",
                   "male": "men", "black": "black", "white": "white", "homosexual_gay_or_lesbian": "gay"}
    rows = []
    for i in range(n):
        toxic = rng.random() < 0.12
        t = topics[rng.integers(len(topics))]
        text = (rude[rng.integers(3)].format(k=KEYWORDS[rng.integers(len(KEYWORDS))], t=t) if toxic
                else nice[rng.integers(4)].format(t=t))
        ident = {g: 0.0 for g in IDENTITIES}
        if rng.random() < 0.4:
            g = list(group_words)[rng.integers(len(group_words))]
            text = f"As a {group_words[g]} reader: " + text
            ident[g] = float(rng.choice([0.5, 0.75, 1.0]))
        tox = float(np.clip(rng.normal(0.75 if toxic else 0.1, 0.12), 0, 1))
        split = rng.choice(["train", "val", "test"], p=[0.6, 0.15, 0.25])
        rows.append({"id": 900000 + i, "comment_text": text, "split": split, "toxicity": tox,
                     "identity_annotator_count": 4, "toxicity_annotator_count": 10, **ident})
    pd.DataFrame(rows)[COLUMNS].to_parquet(path)
    return path


t0 = time.time()
if SAMPLE:
    path = make_synthetic_civil_comments(OUT / "civilcomments_SYNTHETIC.parquet")
else:
    path = hf_hub_download(REPO, "raw/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
raw = pd.read_parquet(path, columns=COLUMNS)
TIMINGS["load_s"] = time.time() - t0
print(f"loaded {len(raw):,} comments in {TIMINGS['load_s']:.0f} s from "
      f"{'a synthetic sample' if SAMPLE else f'https://huggingface.co/datasets/{REPO} (revision {REVISION[:7]})'}")
print("licence: CC0 for the original Civil Comments data (google/civil_comments card); cite Borkan et al. 2019.")

# %% [markdown]
# **Expected (real data):** 448,000 comments, a ~140 MB download the first time (cached afterwards).

# %% [markdown]
# ## Step 3 · Data audit
#
# **Goal.** Count before you model: split sizes, prevalence, missing text, how contested the labels
# are, how many comments mention each identity, and duplicates.
#
# > **Your turn (30 minutes).** Produce these counts without printing a single comment. How many
# > comments have a toxicity fraction between 0.4 and 0.6? How many texts appear in more than one split?
#
# **Our solution.**

# %%
raw["comment_text"] = raw["comment_text"].fillna("")
df = raw.assign(y=(raw.toxicity >= 0.5).astype(int), length=raw.comment_text.str.len())
for g in IDENTITIES:
    df[g] = df[g].fillna(0) >= 0.5
summary = df.groupby("split").agg(comments=("y", "size"), toxic_rate=("y", "mean"),
                                  contested_04_06=("toxicity", lambda t: t.between(0.4, 0.6).mean()),
                                  any_identity=("male", lambda s: df.loc[s.index, IDENTITIES].any(axis=1).mean()),
                                  median_length=("length", "median"))
print(f"empty texts: {(raw.comment_text.str.strip() == '').sum()} | raters per comment (median): "
      f"{int(df.toxicity_annotator_count.median())}")
summary.round(3)

# %%
def normalise(text: pd.Series) -> pd.Series:
    return text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


df["norm"] = normalise(df.comment_text)
eval_texts = set(df.norm[df.split != "train"])
cross = df.split.eq("train") & df.norm.isin(eval_texts)
val_test = df.split.eq("val") & df.norm.isin(set(df.norm[df.split == "test"]))
print(f"duplicate normalised texts overall: {df.norm.duplicated().sum():,}")
print(f"train comments whose text also appears in val or test: {cross.sum():,}  (removed from train in Step 4)")
print(f"val comments whose text also appears in test: {val_test.sum():,}  (kept: official splits)")
ident = pd.DataFrame({g: {"test_n": int(df[g][df.split == "test"].sum()),
                          "test_toxic": int((df[g] & df.y.astype(bool))[df.split == "test"].sum()),
                          "toxic_rate": df.y[df[g] & (df.split == "test")].mean()} for g in IDENTITIES}).T
ident.sort_values("test_n", ascending=False).round(3)

# %% [markdown]
# **Expected (real data, measured):** 269,038 train, 45,180 val and 133,782 test comments, each with a
# toxic rate of 0.11; about 11% of comments have a contested toxicity fraction between 0.4 and 0.6;
# about 41% mention at least one identity; median length about 257 characters; two missing texts; a
# median of 4 raters per comment. 3,449 texts are exact duplicates after normalisation; 1,369 training
# comments repeat a validation or test text, and 188 validation comments repeat a test text. In the
# test split, mentions range from 693 (*latino*) to 16,449 (*female*), and the toxic rate is much higher
# among comments mentioning *black* (0.32), *homosexual_gay_or_lesbian* (0.29), *white* (0.28) or
# *muslim* (0.23) than overall (0.11): identity mention and toxicity are correlated in the *labels*,
# which is exactly what a model will learn.
#
# > **Intuition.** A fractional label of 0.5 means five of ten raters said "toxic" and five did not.
# > Calling it toxic is a convention, and comments near the boundary are where any model "errs" most.
# > *Where this breaks down:* raters are not a random sample of readers, so even a 0.9 is an opinion of
# > a particular crowd, not a fact about the comment.

# %% [markdown]
# ## Step 4 · A leakage-safe split for this problem
#
# **Goal.** Keep the official splits, remove cross-split duplicates from training, subsample training
# to a documented CPU budget, and make sure only text reaches the model.
#
# > **Your turn (20 minutes).** Decide what each split is for. Why is a duplicate across train and test
# > a problem even if its label is correct? Which columns must never be features?
#
# **Our solution.** Train fits models; validation chooses models, calibrates and sets thresholds; test is
# scored once. A duplicate in train and test lets the model "remember" a test answer, so we drop such
# training rows (and repeated texts inside train). The TF-IDF model uses a stratified training subsample
# (same toxic rate as the full train split, fixed seed); the transformer uses a smaller stratified
# subsample, and both are compared on the same random subsamples of validation and test.

# %%
from sklearn.model_selection import train_test_split

train_all = df[(df.split == "train") & ~cross].drop_duplicates("norm")
val = df[df.split == "val"].reset_index(drop=True)
test = df[df.split == "test"].reset_index(drop=True)
tr_tfidf, _ = train_test_split(train_all, train_size=min(SIZES["tfidf_train"], len(train_all) - 2),
                               stratify=train_all.y, random_state=SEED)
tr_emb, _ = train_test_split(tr_tfidf, train_size=min(SIZES["emb_train"], len(tr_tfidf) - 2),
                             stratify=tr_tfidf.y, random_state=SEED)
val_sub = val.sample(min(SIZES["val_sub"], len(val)), random_state=SEED)
test_sub = test.sample(min(SIZES["test_sub"], len(test)), random_state=SEED)
print(f"train after de-duplication {len(train_all):,} | TF-IDF subsample {len(tr_tfidf):,} (toxic {tr_tfidf.y.mean():.3f}) "
      f"| transformer subsample {len(tr_emb):,} (toxic {tr_emb.y.mean():.3f})")
print(f"validation {len(val):,} (subsample {len(val_sub):,}) | test {len(test):,} (subsample {len(test_sub):,})")
print("example training ids:", tr_tfidf.id.head(3).tolist(), "| lengths:", tr_tfidf.length.head(3).tolist())

# %% [markdown]
# **Check.** No training text appears in validation or test; the subsamples keep the toxic rate.

# %%
assert not set(tr_tfidf.norm) & (set(val.norm) | set(test.norm))
assert abs(tr_tfidf.y.mean() - train_all.y.mean()) < 0.005 and abs(tr_emb.y.mean() - train_all.y.mean()) < 0.01
assert set(tr_emb.id) <= set(tr_tfidf.id)
print("no cross-split duplicates in training; subsamples are stratified")

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap5-split-protocol.svg" alt="Train is de-duplicated and subsampled to fit; validation sets calibration, thresholds and model choice; test is scored once; only text is a model input and identity columns only slice the evaluation" width="100%">
#
# *What to notice:* identity columns appear only on the evaluation side; every threshold is fitted on
# validation.

# %% [markdown]
# ## Step 5 · Baselines
#
# **Goal.** A transparent keyword rule and the main baseline, TF-IDF + logistic regression, evaluated on
# validation.
#
# > **Your turn (40 minutes).** Build both. For TF-IDF, use word 1–2-grams and character 3–5-grams in a
# > scikit-learn `Pipeline`, so the vocabulary and IDF weights are fitted on training text only.
#
# **Our solution.** The keyword list is deliberately small and mild (printed below) and the score is the
# number of matches. The TF-IDF model uses `class_weight="balanced"`, so its raw scores are inflated
# for the minority class; we calibrate in Step 6.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline, make_union

KW = re.compile(r"\b(" + "|".join(KEYWORDS) + r")\b")


def keyword_score(texts: pd.Series) -> np.ndarray:
    return texts.str.lower().str.count(KW).to_numpy(dtype=float)


def pr_roc(y, s) -> dict:
    return {"PR-AUC": average_precision_score(y, s), "ROC-AUC": roc_auc_score(y, s)}


print("keyword list:", ", ".join(KEYWORDS))
kw_val = keyword_score(val.comment_text)
print("keyword rule on validation:", {k: round(v, 3) for k, v in pr_roc(val.y, kw_val).items()},
      f"| flags {np.mean(kw_val > 0):.1%} of comments")

t0 = time.time()
tfidf = make_pipeline(
    make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True),
               TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=200_000, sublinear_tf=True)),
    LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
tfidf.fit(tr_tfidf.comment_text, tr_tfidf.y)
TIMINGS["tfidf_fit_s"] = time.time() - t0
t0 = time.time()
s_val = tfidf.predict_proba(val.comment_text)[:, 1]
TIMINGS["tfidf_ms_per_comment"] = 1000 * (time.time() - t0) / len(val)
print(f"TF-IDF + LR: fitted on {len(tr_tfidf):,} comments in {TIMINGS['tfidf_fit_s']:.0f} s; scoring "
      f"{TIMINGS['tfidf_ms_per_comment']:.2f} ms per comment")
print("TF-IDF on validation:", {k: round(v, 3) for k, v in pr_roc(val.y, s_val).items()})

# %% [markdown]
# **Check.** The pipeline's input is a column of text, so identity or annotation columns cannot reach
# it; and its vocabulary was built from the training subsample alone.

# %%
vocab_words = len(tfidf[0].transformer_list[0][1].vocabulary_)
refit = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True).fit(tr_tfidf.comment_text)
assert vocab_words == len(refit.vocabulary_)
assert getattr(tfidf, "feature_names_in_", None) is None  # fitted on a text Series, not a DataFrame of columns
print(f"word vocabulary {vocab_words:,} terms, identical to a vectoriser fitted on training text only")

# %% [markdown]
# **Expected (real data, validation, measured):** the keyword rule reaches PR-AUC 0.27 and ROC-AUC 0.63
# (it flags 4.7% of comments and misses most toxicity, which rarely uses these exact words). TF-IDF + LR
# reaches PR-AUC 0.65 and ROC-AUC 0.91, fitted in about a minute (54 s) on 100,000 comments with two
# CPU threads, and scores a comment in about 0.4 ms.
#
# **Common mistakes.** Fitting the vectoriser on all text "because it is unsupervised" (it still leaks
# test vocabulary and IDF statistics); judging the model by accuracy; forgetting that
# `class_weight="balanced"` changes the meaning of the scores.

# %% [markdown]
# ## Step 6 · A compact pretrained model, and calibration
#
# **Goal.** Test whether a small pretrained encoder beats TF-IDF, at a cost that runs on a CPU, and
# calibrate the scores reviewers will see.
#
# > **Your turn (45 minutes).** Embed comments with a frozen `all-MiniLM-L6-v2` encoder (mean pooling,
# > 128 tokens) and train a logistic regression on the embeddings (a linear probe). Measure throughput.
# > For a fair comparison, also train TF-IDF on the *same* small subsample.
#
# **Our solution.** Fine-tuning a transformer on a CPU is slow, so the CPU core uses a linear probe:
# the expensive encoder runs once per comment, and only a logistic regression is trained. Sorting
# comments by length before batching reduces padding. Full fine-tuning is the guarded GPU extension
# below. The model's licence is Apache-2.0 (see its model card); verify it yourself.

# %%
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

MINILM_ID, MINILM_REV, MAX_LEN = "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", 128
tok = AutoTokenizer.from_pretrained(MINILM_ID, revision=MINILM_REV)
encoder = AutoModel.from_pretrained(MINILM_ID, revision=MINILM_REV).to(DEVICE)


def embed(texts: pd.Series) -> np.ndarray:
    texts = texts.tolist()
    order = np.argsort([len(t) for t in texts])  # similar lengths per batch = less padding
    enc = ft.encode(tok, [texts[i] for i in order], MAX_LEN)
    e = ft.frozen_embeddings(encoder, enc, DEVICE, batch_size=64)
    out = np.empty_like(e)
    out[order] = e
    return out


t0 = time.time()
E = {name: embed(part.comment_text) for name, part in [("train", tr_emb), ("val", val_sub), ("test", test_sub)]}
n_emb = sum(len(v) for v in E.values())
TIMINGS["embed_s"] = time.time() - t0
TIMINGS["embed_comments_per_s"] = n_emb / TIMINGS["embed_s"]
probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))
probe.fit(E["train"], tr_emb.y)
small_tfidf = make_pipeline(
    make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
               TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")).fit(tr_emb.comment_text, tr_emb.y)
print(f"embedded {n_emb:,} comments in {TIMINGS['embed_s']:.0f} s ({TIMINGS['embed_comments_per_s']:.0f} comments/s "
      f"on {DEVICE.type}, {torch.get_num_threads()} threads)")

SUB_VAL = {"tfidf_full": tfidf.predict_proba(val_sub.comment_text)[:, 1],
           "tfidf_small": small_tfidf.predict_proba(val_sub.comment_text)[:, 1],
           "minilm_probe": probe.predict_proba(E["val"])[:, 1],
           "keyword": keyword_score(val_sub.comment_text)}
pd.DataFrame({m: pr_roc(val_sub.y, s) for m, s in SUB_VAL.items()}).T.round(3)

# %% [markdown]
# **Expected (real data, validation subsample, measured):** MiniLM embeddings at about 80 comments per
# second on two CPU threads (about 4.5 minutes for 23,000 comments). PR-AUC on the 5,000-comment
# validation subsample: TF-IDF on 100,000 comments 0.64, TF-IDF on the same 8,000 comments 0.50, the
# MiniLM probe on those 8,000 comments 0.38, keywords 0.30. A frozen general-purpose sentence encoder
# was trained to capture *topic and meaning*, and toxicity is often carried by a few words and their
# tone, which mean pooling dilutes; a fine-tuned encoder would do much better, but not on this budget.
# This comparison is the honest answer to "should we ship the transformer?": not this one, at this
# budget.
#
# **Alternatives considered.** Fine-tuning DistilBERT (better, but needs a GPU to use enough data; see
# the extension); a larger sentence encoder (slower still on CPU); TF-IDF with more data (the
# cheapest improvement, used here).

# %%
# --- Optional GPU extension: fine-tune DistilBERT on a larger subsample ------------
RUN_GPU_EXTENSION = False  # set True on a GPU runtime (Runtime -> Change runtime type -> T4 GPU)

if RUN_GPU_EXTENSION and DEVICE.type == "cuda":
    from transformers import AutoModelForSequenceClassification

    from scaleml.train import TrainConfig, train

    ft_tok = AutoTokenizer.from_pretrained(ft.DISTILBERT_ID, revision=ft.DISTILBERT_REV)
    clf = AutoModelForSequenceClassification.from_pretrained(ft.DISTILBERT_ID, revision=ft.DISTILBERT_REV, num_labels=2)
    ft_train = tr_tfidf.sample(min(BUDGET.train_examples, len(tr_tfidf)), random_state=SEED)
    batches = ft.TokenBatches(ft.encode(ft_tok, ft_train.comment_text.tolist(), MAX_LEN), ft_train.y.to_numpy(),
                              batch_size=BUDGET.batch_size, seed=SEED)
    cfg = TrainConfig(epochs=1, lr=3e-5, warmup_steps=max(1, len(batches) // 10), mixed_precision=True,
                      log_every=max(1, len(batches) // 5))
    result = train(clf, batches, ft.classification_loss(DEVICE), DEVICE, cfg)
    for name, part in [("val_sub", val_sub), ("test_sub", test_sub)]:
        logits = ft.predict_logits(clf, ft.encode(ft_tok, part.comment_text.tolist(), MAX_LEN), DEVICE)
        print(name, {k: round(v, 3) for k, v in pr_roc(part.y, ft.softmax(logits)[:, 1]).items()})
    print(f"fine-tuned on {len(ft_train):,} comments in {result.seconds:.0f} s")
else:
    print("GPU extension skipped (RUN_GPU_EXTENSION is False or no GPU). Not part of the reported results.")

# %% [markdown]
# **Calibration.** Reviewers read scores, and thresholds should stay stable across retrains, so we map
# raw scores to probabilities with Platt scaling fitted on **validation**. A monotone map does not change
# the ranking, so PR-AUC and the capacity thresholds are unaffected; only the meaning of the number
# changes.

# %%
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s_val)[:, None], val.y)


def calibrated(raw_scores):
    return platt.predict_proba(logit(raw_scores)[:, None])[:, 1]


p_val = calibrated(s_val)
print(f"validation Brier: raw {brier_score_loss(val.y, s_val):.4f} → calibrated {brier_score_loss(val.y, p_val):.4f} "
      f"(fitted here, so this is optimistic; the test number in Step 7 is the honest one)")
THRESHOLDS = {q: float(np.quantile(p_val, 1 - q)) for q in BUDGETS}
print("capacity thresholds from validation (calibrated scale):", {f"{q:.0%}": round(t, 3) for q, t in THRESHOLDS.items()})

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap5-capacity-threshold.svg" alt="Illustrative curves: as the review budget grows, recall rises and precision falls; dashed lines at 1, 2 and 5 percent" width="100%">
#
# *What to notice:* the budget, not a habit like 0.5, chooses the point on these curves.

# %% [markdown]
# ## Step 7 · Evaluation on the test split, once, with uncertainty
#
# **Goal.** Score the full test split with the TF-IDF model (the CPU-deployable one), report PR-AUC,
# ROC-AUC, Brier and precision/recall at each budget with bootstrap 95% intervals, and the head-to-head
# comparison on the test subsample.
#
# > **Your turn (30 minutes).** Apply the validation thresholds to test. Is the realised queue share
# > exactly the budget? Why not?
#
# **Our solution.**

# %%
def bootstrap(fn, *arrays, B=200, seed=SEED):
    """Point estimate and 95% interval of fn(*arrays), resampling comments with replacement."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    est = fn(*arrays)
    draws = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        draws.append(fn(*(a[idx] for a in arrays)))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return est, lo, hi


t0 = time.time()
y_test = test.y.to_numpy()
s_test = tfidf.predict_proba(test.comment_text)[:, 1]
p_test = calibrated(s_test)
TIMINGS["tfidf_test_scoring_s"] = time.time() - t0
overall = {"PR-AUC": bootstrap(average_precision_score, y_test, p_test),
           "ROC-AUC": bootstrap(roc_auc_score, y_test, p_test),
           "Brier (calibrated)": bootstrap(brier_score_loss, y_test, p_test)}
print(f"test: {len(test):,} comments, toxic rate {y_test.mean():.3f}; Brier of raw balanced scores "
      f"{brier_score_loss(y_test, s_test):.4f}")
for k, (e, lo, hi) in overall.items():
    print(f"  {k:20} {e:.4f}  (95% CI {lo:.4f}–{hi:.4f})")


def queue_metrics(y, s, thr):
    flag = s >= thr
    tp = (flag & (y == 1)).sum()
    return {"queue_share": flag.mean(), "precision": tp / max(flag.sum(), 1), "recall": tp / max((y == 1).sum(), 1)}


rows = []
for q in BUDGETS:
    thr = THRESHOLDS[q]
    m = queue_metrics(y_test, p_test, thr)
    _, plo, phi = bootstrap(lambda a, b: queue_metrics(a, b, thr)["precision"], y_test, p_test)
    _, rlo, rhi = bootstrap(lambda a, b: queue_metrics(a, b, thr)["recall"], y_test, p_test)
    rows.append({"budget": f"{q:.0%}", "threshold": thr, "queue share (test)": m["queue_share"],
                 "precision": m["precision"], "precision CI": f"{plo:.3f}–{phi:.3f}",
                 "recall": m["recall"], "recall CI": f"{rlo:.3f}–{rhi:.3f}"})
for thr in [0.5]:
    m = queue_metrics(y_test, p_test, thr)
    rows.append({"budget": "fixed 0.5", "threshold": thr, "queue share (test)": m["queue_share"],
                 "precision": m["precision"], "precision CI": "", "recall": m["recall"], "recall CI": ""})
BUDGET_TABLE = pd.DataFrame(rows)
BUDGET_TABLE.round(3)

# %%
bins = np.minimum((p_test * 10).astype(int), 9)
rel = pd.DataFrame({"bin": bins, "p": p_test, "y": y_test}).groupby("bin").agg(
    comments=("y", "size"), mean_score=("p", "mean"), toxic_rate=("y", "mean"))
print("reliability on test (10 equal-width bins of the calibrated score): mean score should match toxic rate")
print(rel.round(3).to_string())

# %% [markdown]
# **Head-to-head on the test subsample** (the same comments for every model, paired bootstrap of the
# PR-AUC difference to the full TF-IDF model).

# %%
SUB_TEST = {"tfidf_full": tfidf.predict_proba(test_sub.comment_text)[:, 1],
            "tfidf_small": small_tfidf.predict_proba(test_sub.comment_text)[:, 1],
            "minilm_probe": probe.predict_proba(E["test"])[:, 1],
            "keyword": keyword_score(test_sub.comment_text)}
ys = test_sub.y.to_numpy()
H2H = []
for name, s in SUB_TEST.items():
    ap = bootstrap(average_precision_score, ys, s)
    d = bootstrap(lambda yy, a, b: average_precision_score(yy, a) - average_precision_score(yy, b),
                  ys, s, SUB_TEST["tfidf_full"])
    H2H.append({"model": name, "train comments": {"tfidf_full": len(tr_tfidf), "keyword": 0}.get(name, len(tr_emb)),
                "PR-AUC": ap[0], "CI": f"{ap[1]:.3f}–{ap[2]:.3f}", "ROC-AUC": roc_auc_score(ys, s),
                "Δ PR-AUC vs tfidf_full": d[0], "Δ CI": f"{d[1]:+.3f} to {d[2]:+.3f}"})
H2H = pd.DataFrame(H2H)
H2H.round(3)

# %% [markdown]
# **Expected (real data, test, measured):** PR-AUC 0.644 (95% CI 0.636–0.652) and ROC-AUC 0.911
# (0.909–0.913) on 133,782 comments. Calibration lowers the Brier score from 0.091 (raw balanced
# scores) to 0.063, and the reliability table shows mean scores close to observed toxic rates, slightly
# over-confident at the top. The realised queue shares on test are 1.0%, 1.9% and 5.0% because
# validation and test come from the same distribution. Precision falls from 0.965 (1% queue) to 0.918
# (2%) and 0.779 (5%), while recall rises from 0.084 to 0.154 and 0.341. Even the 5% queue misses two
# thirds of toxic comments, because only ~11% of comments are toxic: recall at budget $q$ can never
# exceed $q / 0.11$. On the 10,000-comment subsample the MiniLM probe's PR-AUC (0.41) is 0.25 below the
# full TF-IDF model (0.66), and the interval of the difference excludes zero.
#
# > **Intuition.** A capacity threshold is a fixed-size net. Making the model better changes *which*
# > comments fall into the net (precision goes up), but recall stays capped by the size of the net. To
# > catch more toxic comments you need more reviewers or a different policy, not just a better model.

# %% [markdown]
# ## Step 8 · Subgroup and error analysis
#
# **Goal.** At the operating point (5% budget), compare FPR and FNR across identity subgroups with
# intervals, compute the bias AUCs, and see how the queue is composed.
#
# > **Your turn (40 minutes).** Build the subgroup table with counts next to every number and a flag for
# > groups below 100 comments or 20 toxic comments. Which groups have the lowest BPSN AUC?
#
# **Our solution.**

# %%
def rate(mask, flag):
    return flag[mask].mean() if mask.sum() else np.nan


THR = THRESHOLDS[OPERATING_BUDGET]
flag = p_test >= THR
fpr_all, fnr_all = rate(y_test == 0, flag), rate(y_test == 1, ~flag)
rows = []
for g in IDENTITIES:
    m = test[g].to_numpy()
    neg, pos = m & (y_test == 0), m & (y_test == 1)
    row = {"subgroup": g, "n": int(m.sum()), "toxic": int(pos.sum()), "flag": ""}
    if m.sum() < MIN_N or pos.sum() < MIN_TOXIC or neg.sum() == 0:
        row["flag"] = "too small"
        rows.append(row)
        continue
    _, flo, fhi = bootstrap(lambda f: f.mean(), flag[neg])
    _, nlo, nhi = bootstrap(lambda f: 1 - f.mean(), flag[pos])
    bp = (m & (y_test == 0)) | (~m & (y_test == 1))
    bn = (m & (y_test == 1)) | (~m & (y_test == 0))
    row.update({"FPR": flag[neg].mean(), "FPR CI": f"{flo:.3f}–{fhi:.3f}", "FNR": 1 - flag[pos].mean(),
                "FNR CI": f"{nlo:.3f}–{nhi:.3f}", "subgroup AUC": roc_auc_score(y_test[m], p_test[m]),
                "BPSN AUC": roc_auc_score(y_test[bp], p_test[bp]), "BNSP AUC": roc_auc_score(y_test[bn], p_test[bn]),
                "queue share": flag[m].sum() / flag.sum(), "share of toxic": pos.sum() / (y_test == 1).sum()})
    rows.append(row)
SUBGROUPS = pd.DataFrame(rows).set_index("subgroup").reindex(columns=[
    "n", "toxic", "flag", "FPR", "FPR CI", "FNR", "FNR CI", "subgroup AUC", "BPSN AUC", "BNSP AUC",
    "queue share", "share of toxic"]).sort_values("BPSN AUC")
print(f"operating point: budget {OPERATING_BUDGET:.0%}, threshold {THR:.3f} | overall FPR {fpr_all:.3f}, FNR {fnr_all:.3f}")
SUBGROUPS.round(3)

# %% [markdown]
# **Expected (real data, measured):** at the 5% budget (threshold 0.618) the overall FPR is 0.012 and
# FNR 0.659. Benign comments mentioning *black* have an FPR of 0.102 (95% CI 0.093–0.111) and those
# mentioning *white* 0.085: seven to eight times the overall rate. *homosexual_gay_or_lesbian* (0.067)
# and *muslim* (0.047) follow. The same groups have the lowest BPSN AUCs (0.685, 0.693, 0.764, 0.773)
# and high BNSP AUCs (0.93–0.96): the identity-term over-flagging signature, benign mentions scored like
# toxicity elsewhere. Comments mentioning *black* are 15.8% of the queue but 10.1% of all toxic
# comments. *christian* is the reverse case (FPR 0.010, BPSN 0.92, the lowest BNSP 0.88). Every group
# passes the minimum-count rule on the full test split.
#
# **Which errors are they?** We never print the text, but we can describe the most over-flagged
# non-toxic comments by id, length, score and their raw toxicity fraction. Many have fractions just
# below 0.5: raters disagreed, and the "false positive" is partly label noise.

# %%
worst_group = SUBGROUPS.index[0]
m = test[worst_group].to_numpy() & (y_test == 0)
top_fp = test.loc[m].assign(score=p_test[m]).sort_values("score", ascending=False).head(8)
print(f"highest-scored non-toxic comments mentioning '{worst_group}' (ids, lengths, scores and raw toxicity only):")
print(top_fp[["id", "length", "score", "toxicity"]].round(3).to_string(index=False))
fp_all = (y_test == 0) & flag
print(f"\nall false positives at the operating point: {fp_all.sum():,}; share with a toxicity fraction ≥ 0.3: "
      f"{(test.toxicity.to_numpy()[fp_all] >= 0.3).mean():.0%}")
conf = (test.toxicity <= 0.1).to_numpy() | (test.toxicity >= 0.8).to_numpy()
print(f"PR-AUC on confidently labelled comments (toxicity ≤ 0.1 or ≥ 0.8, {conf.mean():.0%} of test): "
      f"{average_precision_score(y_test[conf], p_test[conf]):.3f} vs {average_precision_score(y_test, p_test):.3f} on all")

# %%
QUEUE = pd.DataFrame([{
    "budget": f"{q:.0%}", "comments/day (assumed)": DAILY_VOLUME, "queue size/day": round(DAILY_VOLUME * q),
    "expected toxic in queue": round(DAILY_VOLUME * q * r["precision"]),
    "expected toxic missed": round(DAILY_VOLUME * y_test.mean() * (1 - r["recall"])),
    "precision": r["precision"], "recall": r["recall"]}
    for q, r in ((q, queue_metrics(y_test, p_test, THRESHOLDS[q])) for q in BUDGETS)])
print("queue analysis under the stated assumption of 100,000 comments a day, with test-set rates:")
QUEUE.round(3)

# %% [markdown]
# **Expected (measured rates):** at 5% of 100,000 comments (5,000 reviews a day), about 3,900 of the
# queued comments are toxic and about 7,500 toxic comments a day are not queued; at 1%, 965 of 1,000
# queued comments are toxic but about 10,400 are missed. The most over-flagged benign comments
# mentioning *black* all have toxicity fractions between 0.3 and 0.47, and 63% of all false positives
# have a fraction of at least 0.3: many "errors" are comments the raters themselves disagreed on. On
# confidently labelled comments PR-AUC rises from 0.644 to 0.762. These are consequences of the
# assumptions, written down so that a team lead can change them, not predictions about any platform.
#
# **Common mistakes.** Hiding subgroup sizes (an FPR of 0.2 on 30 comments means little); choosing
# per-group thresholds, which need the identity at prediction time and trade one error rate for another
# (Chapter 53 shows this); quoting offensive examples in a report.

# %% [markdown]
# ## Step 9 · A small API
#
# **Goal.** Serve scores and bands with the threshold and model version in every response, with the
# validation rules from the brief, and without logging raw text.
#
# > **Your turn (40 minutes).** Write a FastAPI app factory with `/health`, `/ready` and `POST /score`
# > (1–64 comments, unique ids, non-empty text up to 5,000 characters, unknown fields rejected). Test it
# > with `TestClient`, including a test that the log contains no comment text.
#
# **Our solution.** Texts longer than 5,000 characters are rejected with 422 (we chose rejection over
# silent truncation, so `truncated` is always false). The `priority_review` band is the top 1% budget;
# there is no removal band. Logs carry ids, lengths and scores only.

# %%
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LOG = logging.getLogger("cap5.api")
LOG.propagate = False  # this demo collects records itself; a service would configure a handler
MAX_CHARS = 5000


class Comment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=64)
    text: str = Field(max_length=MAX_CHARS)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    comments: list[Comment] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [c.id for c in self.comments]
        if len(ids) != len(set(ids)):
            raise ValueError("comment ids must be unique within a request")
        return self


def create_app(score_fn, meta: dict | None) -> FastAPI:
    app = FastAPI(title="comment triage", version="1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if meta is None:
            raise HTTPException(503, "model not loaded")
        return {"status": "ready", **{k: meta[k] for k in ("model_version", "threshold", "review_budget")}}

    @app.post("/score")
    def score(req: ScoreRequest):
        if meta is None:
            raise HTTPException(503, "model not loaded")
        scores = score_fn(pd.Series([c.text for c in req.comments]))
        results = []
        for c, s in zip(req.comments, scores):
            band = ("priority_review" if s >= meta["priority_threshold"] else
                    "human_review" if s >= meta["threshold"] else "auto_approve")
            results.append({"id": c.id, "score": round(float(s), 4), "band": band, "truncated": False})
            LOG.info("scored id=%s length=%d score=%.4f band=%s", c.id, len(c.text), s, band)
        return {"model_version": meta["model_version"], "threshold": round(meta["threshold"], 4),
                "review_budget": meta["review_budget"], "results": results}

    return app


META = {"model_version": "tfidf-lr-v1", "threshold": THRESHOLDS[OPERATING_BUDGET], "review_budget": OPERATING_BUDGET,
        "priority_threshold": THRESHOLDS[0.01]}
(OUT / "model_meta.json").write_text(json.dumps(META, indent=2))
client = TestClient(create_app(lambda texts: calibrated(tfidf.predict_proba(texts)[:, 1]), META))

records = []
handler = logging.Handler()
handler.emit = records.append
LOG.addHandler(handler)
LOG.setLevel(logging.INFO)
secret = "a neutral sentence about the city budget meeting"
r = client.post("/score", json={"comments": [{"id": "c-001", "text": secret}, {"id": "c-002", "text": "Thank you."}]})
LOG.removeHandler(handler)
body = r.json()
assert r.status_code == 200 and body["review_budget"] == OPERATING_BUDGET and len(body["results"]) == 2
assert all(x["band"] in {"auto_approve", "human_review", "priority_review"} for x in body["results"])
assert records and not any(secret in rec.getMessage() for rec in records), "raw text must not be logged"
bad = [{"comments": []}, {"comments": [{"id": "a", "text": "   "}]}, {"comments": [{"id": "a", "text": "x" * 5001}]},
       {"comments": [{"id": "a", "text": "hi"}, {"id": "a", "text": "hello"}]},
       {"comments": [{"id": "a", "text": "hi", "user": "u1"}]}, {"comments": [{"id": str(i), "text": "hi"} for i in range(65)]}]
assert all(client.post("/score", json=b).status_code == 422 for b in bad)
assert client.get("/ready").json()["threshold"] == META["threshold"]
assert TestClient(create_app(None, None)).get("/ready").status_code == 503
print("API: all checks pass; example response:", json.dumps(body)[:220])

# %% [markdown]
# **Common mistakes.** Logging the request body for debugging (it then contains user text forever);
# returning a "remove" band; computing the threshold at request time instead of shipping it with the
# model version, so that a rollback silently keeps the new operating point.

# %% [markdown]
# ## Step 10 · Write-up: results tables, CPU budget, limitations, next steps
#
# **Goal.** Produce the README results tables *from this run*, never by hand, and state the compute used.
#
# > **Your turn (30 minutes).** Generate the markdown in code; then write the limitations and the next
# > steps.
#
# **Our solution.** In sample mode the tables are labelled synthetic and must not be pasted anywhere.

# %%
TIMINGS["notebook_s"] = time.time() - T_START


def results_markdown() -> str:
    src = "SYNTHETIC SAMPLE — not a result" if SAMPLE else f"CivilComments-WILDS ({REPO}@{REVISION[:7]})"
    pr, roc, br = overall["PR-AUC"], overall["ROC-AUC"], overall["Brier (calibrated)"]
    out = [f"Data: {src}; official test split, {len(test):,} comments, toxic rate {y_test.mean():.3f}. "
           f"Model: TF-IDF (word 1–2 + char 3–5) + logistic regression on a stratified {len(tr_tfidf):,}-comment "
           "training subsample, Platt-calibrated and thresholded on validation. 95% CIs: bootstrap over comments (200).", "",
           "| Metric | Value (95% CI) |", "|---|---|",
           f"| PR-AUC | {pr[0]:.3f} ({pr[1]:.3f}–{pr[2]:.3f}) |", f"| ROC-AUC | {roc[0]:.3f} ({roc[1]:.3f}–{roc[2]:.3f}) |",
           f"| Brier (calibrated) | {br[0]:.4f} ({br[1]:.4f}–{br[2]:.4f}) |", "",
           "| Review budget | Threshold | Queue share (test) | Precision (95% CI) | Recall (95% CI) |", "|---|---|---|---|---|"]
    for _, r in BUDGET_TABLE.iterrows():
        out.append(f"| {r.budget} | {r.threshold:.3f} | {r['queue share (test)']:.3f} | {r.precision:.3f}"
                   + (f" ({r['precision CI']})" if r["precision CI"] else "") + f" | {r.recall:.3f}"
                   + (f" ({r['recall CI']})" if r["recall CI"] else "") + " |")
    out += ["", f"Head-to-head on a random {len(test_sub):,}-comment test subsample (same comments for every model):", "",
            "| Model | Training comments | PR-AUC (95% CI) | ROC-AUC | Δ PR-AUC vs TF-IDF full (95% CI) |", "|---|---|---|---|---|"]
    for _, r in H2H.iterrows():
        out.append(f"| {r.model} | {r['train comments']:,} | {r['PR-AUC']:.3f} ({r.CI}) | {r['ROC-AUC']:.3f} | "
                   f"{r['Δ PR-AUC vs tfidf_full']:+.3f} ({r['Δ CI']}) |")
    out += ["", f"Subgroups at the {OPERATING_BUDGET:.0%} budget (threshold {THR:.3f}); overall FPR {fpr_all:.3f}, "
            f"FNR {fnr_all:.3f}. Groups with fewer than {MIN_N} comments or {MIN_TOXIC} toxic comments are not interpreted.", "",
            "| Subgroup | n | Toxic | FPR (95% CI) | FNR (95% CI) | Subgroup AUC | BPSN AUC | BNSP AUC |", "|---|---|---|---|---|---|---|---|"]
    for g, r in SUBGROUPS.iterrows():
        if r.flag:
            out.append(f"| {g} | {r.n} | {r.toxic} | {r.flag} | | | | |")
        else:
            out.append(f"| {g} | {r.n:,} | {r.toxic:,} | {r.FPR:.3f} ({r['FPR CI']}) | {r.FNR:.3f} ({r['FNR CI']}) | "
                       f"{r['subgroup AUC']:.3f} | {r['BPSN AUC']:.3f} | {r['BNSP AUC']:.3f} |")
    out += ["", f"Queue under the assumption of {DAILY_VOLUME:,} comments a day:", "",
            "| Budget | Queue/day | Expected toxic in queue | Expected toxic missed | Precision | Recall |", "|---|---|---|---|---|---|"]
    for _, r in QUEUE.iterrows():
        out.append(f"| {r.budget} | {r['queue size/day']:,} | {r['expected toxic in queue']:,} | "
                   f"{r['expected toxic missed']:,} | {r.precision:.3f} | {r.recall:.3f} |")
    out += ["", f"CPU budget used: device {DEVICE.type}, {torch.get_num_threads()} threads; TF-IDF fit "
            f"{TIMINGS['tfidf_fit_s']:.0f} s, scoring {TIMINGS['tfidf_ms_per_comment']:.2f} ms/comment; MiniLM embeddings "
            f"{TIMINGS['embed_comments_per_s']:.0f} comments/s ({TIMINGS['embed_s']:.0f} s); whole notebook "
            f"{TIMINGS['notebook_s'] / 60:.1f} min."]
    return "\n".join(out)


RESULTS_MD = results_markdown()
(OUT / "results.md").write_text(RESULTS_MD + "\n", encoding="utf-8")
print(RESULTS_MD)

# %% [markdown]
# **Limitations** (state them in your README):
#
# 1. **Label noise and disagreement.** Labels are fractions from a small crowd of raters; the 0.5 cut-off
#    is a convention. Step 8 showed that many "false positives" have fractions just below 0.5, and that
#    PR-AUC is much higher on confidently labelled comments.
# 2. **Identity-term over-flagging.** Benign comments that mention some identities are queued several
#    times more often than other benign comments (low BPSN AUC, high subgroup FPR).
# 3. **Identity labels are partial and noisy;** small subgroups have wide intervals and some cannot be
#    evaluated.
# 4. **Domain and time.** One commenting platform in one period (2015–2017); language and norms drift.
# 5. **Toxicity is not a policy.** A platform's rules are more specific and context-dependent.
# 6. **Capacity numbers are assumptions;** the transformer comparison is at a CPU budget (8,000
#    training comments), not the best a transformer can do.
#
# **Next steps (extensions).** Fine-tune DistilBERT on a GPU (the guarded cell); soft-label training on
# the toxicity fraction; bias mitigation (reweight or augment benign identity mentions) with a before /
# after subgroup table; per-subgroup calibration; monitoring of queue share, score drift (PSI) and
# reviewer agreement, remembering that only queued comments receive new labels.
#
# **What to put in your repository.** The data card (source, revision, CC0 notice, split and duplicate
# counts, subsample seed), the evaluation code with unit tests on tiny hand-made cases, the tables
# printed above, a model card whose intended use is "triage for human review" and whose out-of-scope
# use is "automatic removal", and the API with its tests.
#
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | Download is slow or stalls | first download of ~140 MB | wait; it is cached afterwards. Locally, `HF_HUB_DISABLE_XET=1` can help behind proxies |
# | The notebook takes much longer than 10 minutes | fewer CPU threads, or another job running | reduce `SIZES` (the "cpu" row) and say so in your write-up |
# | `MemoryError` while fitting TF-IDF | too little RAM for 100,000 comments with character n-grams | lower `tfidf_train` or `max_features` |
# | Results differ slightly from the expected lines | library versions, thread count | conclusions, not third decimals, should match |
