# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch53-fairness-robustness-and-privacy/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 53 · Fairness, robustness & privacy
#
# **Part III · Chapter 53 · about 90 minutes · T4 GPU (runs on CPU with a reduced budget)**
#
# A model with a high overall AUC can still flag comments that merely *mention* an identity as
# toxic, degrade sharply when the data drift or someone misspells a slur, and reveal which records it
# was trained on. Questions on these topics now appear in most ML interview loops ("how would you
# evaluate this classifier for bias?", "what is differential privacy and what does it cost?"), and
# the expected answers are quantitative.
#
# > **Content warning.** This notebook trains on real online comments labelled for toxicity, some
# > of which are offensive. It **never prints comment text**: only ids, lengths, counts and metrics.
# > The identity terms used in the template tests are neutral words.
#
# By the end of this lab you should be able to:
#
# 1. compute subgroup AUC, **BPSN** and **BNSP** AUC (Borkan et al., 2019) and FPR/FNR gaps, and explain what each reveals;
# 2. show how one global threshold produces different error rates per group, and what per-group thresholds trade;
# 3. test robustness to **distribution shift** (time and source) and to **perturbations** (typos, identity-term swaps);
# 4. explain PII handling and **membership inference**, and measure an attack on an over-fitted model;
# 5. implement **DP-SGD** (per-example clipping + Gaussian noise) and show the privacy/utility trade-off.
#
# **You will produce:** `artifacts/level3/nb15/subgroup_metrics.csv` and `artifacts/level3/nb15/results.json`.
#
# **Measured here vs on a T4.** Everything in this notebook is CPU-friendly (TF-IDF + logistic
# regression, NumPy DP-SGD); the GPU budget only uses more data. Numbers in **Expected:** lines
# were printed by our CPU-budget run. Fine-tuning a transformer on this data is the natural
# extension; on a T4 it would typically raise overall AUC by a few points, but whether it closes the
# subgroup gaps must be measured, not assumed (not measured in our verification runs).

# %% [markdown]
# Read first: [Chapter 53 text](README.md), which explains subgroup metrics, robustness testing, membership inference and differential privacy in more detail than this lab.

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

# Level 3 is written for a GPU runtime: Runtime -> Change runtime type -> T4 GPU.
from scaleml.device import budget, describe, get_device

DEVICE = get_device()
BUDGET = budget(DEVICE)
print(describe(DEVICE))
print("budget:", BUDGET.as_dict())
if DEVICE.type == "cpu":
    print("No GPU found: running with the reduced CPU budget. Numbers will be much worse than on a T4.")

# %%
# --- Catch up: nothing needed from earlier notebooks; download the dataset --
import json
import re
import time

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from grading import ch53 as grade

OUT = ART / "level3" / "nb15"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = {}

REPO, REVISION = "pietrolesci/civilcomments-wilds", "c227534cc0a34cf21db6a0bf0edd0f9050c8b305"
IDENTITIES = ["male", "female", "homosexual_gay_or_lesbian", "christian", "jewish", "muslim", "black", "white",
              "psychiatric_or_mental_illness"]
t0 = time.perf_counter()
path = hf_hub_download(REPO, "raw/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
raw = pd.read_parquet(path, columns=["id", "comment_text", "split", "created_date", "publication_id",
                                     "toxicity", *IDENTITIES])
print(f"loaded {len(raw):,} comments in {time.perf_counter() - t0:.0f} s")
print(f"dataset: https://huggingface.co/datasets/{REPO} (revision {REVISION[:7]}): the CivilComments-WILDS")
print("release of the Jigsaw 'Unintended Bias in Toxicity Classification' data (Civil Comments platform);")
print("the original data are released under CC0 (see https://huggingface.co/datasets/google/civil_comments).")

# %% [markdown]
# **Expected:** 448,000 comments (a ~140 MB download the first time, then cached). Each has a
# `toxicity` score (the fraction of ~10 crowd raters who called it toxic) and, for this subset,
# identity scores (the fraction of raters who said the comment *mentions* that identity). As in
# the Jigsaw competition and WILDS, we binarise both at 0.5.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Subgroup | The comments that mention one identity | comments mentioning *muslim* |
# | Background | All test comments that do *not* mention that identity | the comparison population for BPSN / BNSP |
# | Subgroup AUC | Ranking quality measured only inside one subgroup | 0.78–0.80 for four groups |
# | BPSN AUC | Toxic background vs non-toxic subgroup comments: low means benign mentions score like toxicity | 0.67–0.75 for *black*, *muslim* |
# | BNSP AUC | Non-toxic background vs toxic subgroup comments: low means toxic posts about the group are missed | 0.91–0.95 (high) |
# | FPR / FNR | Share of negatives flagged / share of positives missed, at a threshold | FPR 0.083 vs 0.010 at 0.5 |
# | Distribution shift | Test data drawn from a different time or source than training | 2016 → late 2017 comments |
# | Counterfactual swap | Replacing one identity term with another and checking the score does not move | "gay" → "straight" |
# | PII | Personally identifiable information: names, emails, phone numbers | the regex redactor in section 6 |
# | Membership inference | Guessing whether a record was in the training set from the model's behaviour on it | loss-threshold attack, AUC 0.64 |
# | Differential privacy (ε, δ) | A guarantee that one record barely changes what training outputs; smaller ε = stronger | ε ≤ 1.0 at σ = 8 |
# | DP-SGD | SGD with per-example gradient clipping and Gaussian noise | `clip_and_noise` in Exercise 2 |

# %% [markdown]
# ## 1 · Data and a baseline classifier
#
# We keep the official WILDS `split` column (train/val/test are random splits) and sample to the
# budget. The model is TF-IDF (word 1–2-grams) + logistic regression: fast, strong for its cost,
# and transparent, which makes its biases easy to diagnose.

# %%
df = raw.assign(y=(raw.toxicity >= 0.5).astype(int), year=raw.created_date.str[:4],
                month=raw.created_date.str[:7], length=raw.comment_text.str.len())
for c in IDENTITIES:
    df[c] = df[c].fillna(0) >= 0.5
del raw
N_TRAIN = {"gpu": 200_000, "cpu": 60_000, "smoke": 5_000}[BUDGET.name]
N_TEST = {"gpu": 100_000, "cpu": 40_000, "smoke": 5_000}[BUDGET.name]
train = df[df.split == "train"].sample(N_TRAIN, random_state=0)
test = df[df.split == "test"].sample(N_TEST, random_state=0)
print(f"train {len(train):,} | test {len(test):,} | toxic rate train {train.y.mean():.3f}, test {test.y.mean():.3f}")
print(f"median length {int(train.length.median())} characters; example ids: {test.id.head(3).tolist()}")
print(test[IDENTITIES].agg(["sum", "mean"]).T.assign(toxic_rate=[test.y[test[c]].mean() for c in IDENTITIES])
      .round(3).to_string())

# %%
t0 = time.perf_counter()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
X_train = vec.fit_transform(train.comment_text)
X_test = vec.transform(test.comment_text)
clf = LogisticRegression(C=4.0, max_iter=2000).fit(X_train, train.y)
s_test = clf.predict_proba(X_test)[:, 1]
auc_all = roc_auc_score(test.y, s_test)
RESULTS["overall_auc"] = round(auc_all, 4)
print(f"TF-IDF {X_train.shape[1]:,} features, trained in {time.perf_counter() - t0:.1f} s | test AUC {auc_all:.4f}")

# %% [markdown]
# **Expected (CPU budget, measured):** 60,000 training and 40,000 test comments, ~11% toxic; identity
# mentions range from 486 (psychiatric or mental illness) to ~5,000 (female) test comments, and the
# toxic rate is much higher among comments mentioning *black*, *white*, *homosexual* or *muslim*
# (23–33%) than overall (11%). ~190,000 TF-IDF features, trained in ~15–35 s; test AUC ≈ 0.893.
#
# ## 2 · Subgroup metrics: subgroup AUC, BPSN, BNSP
#
# Overall AUC hides *where* the errors are. Borkan et al. (2019) define three AUCs per identity
# subgroup `g`, each on a different slice of the test set:
#
# * **Subgroup AUC**: only comments that mention `g`. Low = the model separates toxic from non-toxic
#   poorly *within* conversations about `g`.
# * **BPSN** (Background Positive, Subgroup Negative): non-toxic comments that mention `g` together
#   with toxic comments that do not. Low = non-toxic mentions of `g` score higher than real toxicity
#   elsewhere: **false positives driven by the identity term**.
# * **BNSP** (Background Negative, Subgroup Positive): toxic comments that mention `g` with non-toxic
#   ones that do not. Low = toxic comments about `g` score *lower* than benign comments elsewhere:
#   false negatives.
#
# None of them depends on a threshold.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-x-bpsn-bnsp.svg" alt="Score distributions for background and subgroup comments; the subgroup's non-toxic comments are shifted towards higher scores; three boxes define subgroup AUC, BPSN and BNSP" width="100%">
#
# *What to notice:* a shift of the subgroup's *non-toxic* scores (the dashed bump moving right) lowers BPSN but leaves BNSP high: exactly the pattern measured below.
#
# > **Intuition.** Think of a smoke detector that also beeps at toast. Subgroup AUC asks "inside the
# > kitchen, does it tell smoke from toast?"; BPSN asks "is kitchen toast scored below real smoke in
# > the rest of the house?" If not, the *room* (the identity mention) is triggering it. BNSP asks the
# > reverse: "is real kitchen smoke scored above harmless air elsewhere?" *Where the analogy breaks
# > down:* identity mentions are not a fixed room; they co-occur with other words, so a low BPSN
# > says *that* the term pushes scores up, not *why* (label bias, sampling, or the model).

# %% [markdown]
# > **Exercise 1.** Write `bpsn_bnsp_auc(y, s, subgroup)` returning the tuple `(bpsn, bnsp)` for labels
# > `y` (0/1), scores `s` and a boolean `subgroup` mask (all array-likes of equal length). Use
# > `sklearn.metrics.roc_auc_score`.

# %%
def bpsn_bnsp_auc(y, s, subgroup):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(bpsn_bnsp_auc)

# %%
def subgroup_report(y, s, frame, threshold=0.5):
    y, s = np.asarray(y), np.asarray(s)
    fpr_all = np.mean(s[y == 0] >= threshold)
    fnr_all = np.mean(s[y == 1] < threshold)
    rows = []
    for g in IDENTITIES:
        m = frame[g].to_numpy()
        bp = (m & (y == 0)) | (~m & (y == 1))  # BPSN slice
        bn = (m & (y == 1)) | (~m & (y == 0))  # BNSP slice
        rows.append({"subgroup": g, "n": int(m.sum()), "toxic_rate": y[m].mean(),
                     "subgroup_auc": roc_auc_score(y[m], s[m]), "bpsn_auc": roc_auc_score(y[bp], s[bp]),
                     "bnsp_auc": roc_auc_score(y[bn], s[bn]),
                     "FPR": np.mean(s[m & (y == 0)] >= threshold), "FNR": np.mean(s[m & (y == 1)] < threshold)})
    out = pd.DataFrame(rows).set_index("subgroup")
    out["FPR gap"] = out.FPR - fpr_all
    out["FNR gap"] = out.FNR - fnr_all
    return out, fpr_all, fnr_all


sub, fpr_all, fnr_all = subgroup_report(test.y, s_test, test)
print(f"overall: AUC {auc_all:.4f} | FPR {fpr_all:.3f} | FNR {fnr_all:.3f} at threshold 0.5")
print(sub.round(3).sort_values("bpsn_auc").to_string())
# the Jigsaw competition's summary: power mean (p = -5) of each bias AUC, which emphasises the worst groups
pm = lambda v, p=-5: float(np.mean(np.asarray(v) ** p) ** (1 / p))  # noqa: E731
final = 0.25 * auc_all + 0.25 * (pm(sub.subgroup_auc) + pm(sub.bpsn_auc) + pm(sub.bnsp_auc))
print(f"Jigsaw 'final metric' (0.25 overall AUC + 0.25 × power means of the three bias AUCs): {final:.4f}")
sub.to_csv(OUT / "subgroup_metrics.csv")
RESULTS["subgroups"] = sub.round(4).to_dict(orient="index")
RESULTS["jigsaw_final_metric"] = round(final, 4)

# %% [markdown]
# **Expected (measured):** overall AUC 0.893, but subgroup AUCs of 0.78–0.80 for *black*,
# *homosexual_gay_or_lesbian*, *muslim* and *white*, and **BPSN AUCs as low as 0.67–0.75** for them,
# while their BNSP AUCs are high (0.91–0.95). That pattern (low BPSN, high BNSP) is the signature of
# the identity term itself pushing scores up: benign comments mentioning these groups score like toxic
# ones. *christian* shows the opposite (BPSN 0.91, BNSP 0.86). At threshold 0.5 the FPR on benign
# comments mentioning *black* is 0.083 against 0.010 overall (an 8× ratio). The Jigsaw summary metric
# is ≈ 0.848. Note also the overall FNR of 0.72 at 0.5: with an 11% base rate a logistic regression
# rarely scores above 0.5, so a real deployment would pick a lower threshold (next section).

# %% [markdown]
# ## 3 · Threshold effects
#
# Deployment needs a threshold, and one global threshold gives each group its own error rates. Below:
# FPR per group at a few thresholds, then **per-group thresholds** chosen to give every group the
# same FPR as the background (an "equalised false-positive rate" policy) and what that does to FNR.
# Per-group thresholds need the identity at prediction time (often unavailable, sometimes not
# legal to use), and equalising one error rate generally un-equalises others: when base rates differ,
# FPR, FNR and calibration cannot all be equalised at once (Kleinberg et al., 2016; Chouldechova, 2017).
#
# > **Intuition.** One threshold is one height bar for everyone; if one group's benign comments are
# > systematically scored higher, the same bar catches more of them. Moving the bar per group can
# > equalise false positives, but then the toxic comments of that group must clear a higher bar and
# > more slip through. *Where the analogy breaks down:* with different base rates, no choice of bars
# > makes FPR, FNR and calibration all equal at once (the impossibility results).

# %%
groups = ["muslim", "homosexual_gay_or_lesbian", "black", "christian"]
y_t = test.y.to_numpy()
rows = []
for thr in [0.3, 0.5, 0.7]:
    row = {"threshold": thr, "overall FPR": np.mean(s_test[y_t == 0] >= thr)}
    for g in groups:
        m = test[g].to_numpy()
        row[f"FPR {g[:10]}"] = np.mean(s_test[m & (y_t == 0)] >= thr)
    rows.append(row)
print(pd.DataFrame(rows).round(3).to_string(index=False))

bg = ~test[IDENTITIES].any(axis=1).to_numpy()
target_fpr = np.mean(s_test[bg & (y_t == 0)] >= 0.5)
rows = []
for g in groups:
    m = test[g].to_numpy()
    thr_g = np.quantile(s_test[m & (y_t == 0)], 1 - target_fpr)  # threshold giving the background's FPR
    rows.append({"subgroup": g, "global thr FPR": np.mean(s_test[m & (y_t == 0)] >= 0.5),
                 "global thr FNR": np.mean(s_test[m & (y_t == 1)] < 0.5), "group threshold": thr_g,
                 "group thr FPR": np.mean(s_test[m & (y_t == 0)] >= thr_g),
                 "group thr FNR": np.mean(s_test[m & (y_t == 1)] < thr_g)})
print(f"background (no identity mentioned) FPR at 0.5: {target_fpr:.3f}")
print(pd.DataFrame(rows).round(3).to_string(index=False))

# %% [markdown]
# **Expected (measured):** at every threshold the benign comments mentioning *black*, *homosexual* or
# *muslim* are flagged 4–8× more often than benign comments overall (at 0.3: 13–22% vs 3.5%); *christian*
# is close to the overall rate. Background comments (no identity mentioned) have an FPR of only 0.004 at
# 0.5. Per-group thresholds equalise FPR at 0.004 but need thresholds of 0.81–0.88 for those groups, and
# their FNR rises to 0.91–0.94: most toxic comments about these groups would then go unflagged. Equal
# FPR was bought with unequal (and worse) FNR, which is the trade-off the impossibility results predict.

# %% [markdown]
# ## 4 · Robustness: distribution shift
#
# A random split measures interpolation. Production data comes from the future and from new
# sources. Two shift tests, each training a fresh model on one slice and testing it on held-out data
# from the **same** slice and from a **different** one:
#
# * **time**: train on 2016 comments, test on 2016 (held out) vs the last months of 2017;
# * **source**: train on the largest publication, test on it vs all other publications.

# %%
def fit_eval(tr, tests):
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
    m = LogisticRegression(C=4.0, max_iter=2000).fit(v.fit_transform(tr.comment_text), tr.y)
    out = {}
    for name, te in tests.items():
        s = m.predict_proba(v.transform(te.comment_text))[:, 1]
        out[name] = {"n": len(te), "toxic_rate": te.y.mean(), "AUC": roc_auc_score(te.y, s),
                     "mean score / toxic rate": s.mean() / te.y.mean()}
    return out


N_SHIFT = {"gpu": 60_000, "cpu": 25_000, "smoke": 3_000}[BUDGET.name]
N_SHIFT_TEST = N_SHIFT // 2
y2016 = df[df.year == "2016"].sample(frac=1.0, random_state=1)
late2017 = df[df.month >= "2017-09"]
time_res = fit_eval(y2016.iloc[:N_SHIFT], {"2016 (held out)": y2016.iloc[N_SHIFT:N_SHIFT + N_SHIFT_TEST],
                                            "2017-09 onwards": late2017.sample(N_SHIFT_TEST, random_state=1)})
top_pub = df.publication_id.value_counts().index[0]
pub = df[df.publication_id == top_pub].sample(frac=1.0, random_state=2)
other = df[df.publication_id != top_pub]
src_res = fit_eval(pub.iloc[:N_SHIFT], {f"publication {top_pub} (held out)": pub.iloc[N_SHIFT:N_SHIFT + N_SHIFT_TEST],
                                         "other publications": other.sample(N_SHIFT_TEST, random_state=2)})
shift = pd.DataFrame({**{("time", k): v for k, v in time_res.items()}, **{("source", k): v for k, v in src_res.items()}}).T
RESULTS["shift"] = {f"{a}/{b}": v for (a, b), v in shift.round(4).to_dict(orient="index").items()}
print(shift.round(3).to_string())

# %% [markdown]
# **Expected (measured):** both shifts cost a little ranking quality (AUC 0.868 → 0.855 for time, 0.866
# → 0.859 for source) and more **calibration**: the ratio of mean score to toxic rate drops from 0.89 to
# 0.82 on late-2017 comments (whose toxic rate is higher, 0.124 vs 0.108). On a random split none of
# this would be visible. A transformer would probably degrade less on vocabulary drift, but that
# has to be measured with the same protocol.
#
# ## 5 · Robustness: perturbations and counterfactual identity swaps
#
# Two perturbation tests on the main model:
#
# * **Typos**: swap two adjacent letters inside ~20% of the words of toxic test comments. A word-level
#   TF-IDF model has never seen the misspelt tokens: how much recall survives? (Adversaries do this on
#   purpose: "id1ot".)
# * **Counterfactual fairness**: replace one identity term with another in the *same* sentence
#   (e.g. "muslim" ↔ "christian"); the toxicity score should not change, because the meaning of the
#   comment (as far as toxicity goes) did not. We use (i) neutral templates (Dixon et al., 2018) and
#   (ii) real non-toxic test comments that contain the term.

# %%
def typo(text, rate, rng):
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 3 and rng.random() < rate:
            j = rng.integers(1, len(w) - 2)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


rng = np.random.default_rng(0)
toxic = test[test.y == 1]
s_clean = clf.predict_proba(vec.transform(toxic.comment_text))[:, 1]
s_typo = clf.predict_proba(vec.transform([typo(t, 0.2, rng) for t in toxic.comment_text]))[:, 1]
RESULTS["typos"] = {"recall_clean": float(np.mean(s_clean >= 0.5)), "recall_typos": float(np.mean(s_typo >= 0.5))}
print(f"{len(toxic):,} toxic test comments: recall at 0.5 clean {np.mean(s_clean >= 0.5):.3f} -> "
      f"with typos {np.mean(s_typo >= 0.5):.3f}; mean score {s_clean.mean():.3f} -> {s_typo.mean():.3f}")

TEMPLATES = ["I am a {t} person.", "My neighbour is {t}.", "Being {t} is part of who I am.",
             "We had dinner with a {t} family last night.", "Our new teacher is {t} and very kind.",
             "{T} people live in my town."]
TERMS = ["christian", "muslim", "jewish", "atheist", "gay", "straight", "black", "white", "young", "old", "american"]
tmpl_scores = pd.DataFrame({t: clf.predict_proba(vec.transform([x.format(t=t, T=t.capitalize()) for x in TEMPLATES]))[:, 1]
                            for t in TERMS}).mean().sort_values(ascending=False)
RESULTS["template_scores"] = tmpl_scores.round(4).to_dict()
print("mean toxicity score of neutral template sentences, by identity term:")
print(tmpl_scores.round(3).to_string())

# %%
PAIRS = [("muslim", "christian"), ("gay", "straight"), ("black", "white")]
benign = test[test.y == 0]
rows = []
for a, b in PAIRS:
    pat = re.compile(rf"\b{a}\b", flags=re.I)
    has = benign[benign.comment_text.str.contains(pat)]
    swapped = has.comment_text.str.replace(pat, b, regex=True)
    s_a = clf.predict_proba(vec.transform(has.comment_text))[:, 1]
    s_b = clf.predict_proba(vec.transform(swapped))[:, 1]
    rows.append({"swap": f"{a} -> {b}", "n non-toxic comments": len(has), "mean score before": s_a.mean(),
                 "after": s_b.mean(), "flagged before (>=0.5)": np.mean(s_a >= 0.5), "after ": np.mean(s_b >= 0.5)})
swaps = pd.DataFrame(rows)
RESULTS["swaps"] = swaps.round(4).to_dict(orient="records")
print(swaps.round(3).to_string(index=False))

# %% [markdown]
# **Expected (measured):** typos in 20% of the words reduce recall on toxic comments from 0.285 to
# 0.254 (−11% relative), with a word-level model that has never seen the misspelt tokens (character
# n-grams or subword models are more robust). The template test gives a clear result: neutral sentences score
# 0.50 with "gay", 0.30 with "black", 0.20 with "muslim", against ≈ 0.03–0.05 for "american",
# "straight", "christian" or "young". Swapping the term in real non-toxic comments lowers their mean
# score (e.g. "gay" → "straight": 0.187 → 0.077, flag rate at 0.5: 5.7% → 0.7%): the model's decision
# depends on *who* is mentioned, which is exactly what counterfactual fairness forbids.

# %% [markdown]
# ## 6 · Privacy: PII, membership inference
#
# **PII.** Training data and logs contain names, emails, phone numbers and addresses. Standard
# practice: minimise what is collected, redact or tokenise PII before it reaches training data and
# logs, restrict access, set retention limits, and honour deletion requests (which is hard once a
# record has been trained on: one more reason to keep PII out). Regexes catch structured PII (emails,
# phone numbers, card numbers); names and addresses need an NER model. Below: a regex redactor on
# synthetic strings, then how many test comments it would touch.

# %%
PII = {"EMAIL": r"[\w.+-]+@[\w-]+\.[\w.]+", "PHONE": r"\b(?:\+?1[ -.]?)?\(?\d{3}\)?[ -.]?\d{3}[ -.]?\d{4}\b",
       "URL": r"https?://\S+"}


def redact(text):
    for tag, pat in PII.items():
        text = re.sub(pat, f"<{tag}>", text)
    return text


print(redact("Contact jane.doe@example.com or (555) 123-4567, details at https://example.com/x"))
counts = {tag: int(test.comment_text.str.contains(pat, regex=True).sum()) for tag, pat in PII.items()}
print("test comments containing each PII pattern:", counts)

# %% [markdown]
# **Membership inference.** If a model behaves differently on its training examples than on unseen
# ones (lower loss, more confident), an attacker who can query it can guess whether a given record
# was in the training set: a privacy leak even if no data is ever shown (Shokri et al., 2017). The
# simplest attack thresholds the per-example loss; its AUC (members vs non-members) measures the
# leak: 0.5 = no leak. Over-fitting is the driver, so we compare a weakly and a strongly
# regularised model trained on 5,000 comments.

# %%
mi_train, mi_out = train.iloc[:5000], test.iloc[:5000]
Xm, Xo = vec.transform(mi_train.comment_text), vec.transform(mi_out.comment_text)
rows = []
for C in [100.0, 1.0, 0.1]:
    m = LogisticRegression(C=C, max_iter=3000).fit(Xm, mi_train.y)
    pm_, po_ = m.predict_proba(Xm)[:, 1], m.predict_proba(Xo)[:, 1]
    loss_in = -np.log(np.where(mi_train.y == 1, pm_, 1 - pm_) + 1e-12)
    loss_out = -np.log(np.where(mi_out.y == 1, po_, 1 - po_) + 1e-12)
    attack_auc = roc_auc_score(np.r_[np.ones(len(loss_in)), np.zeros(len(loss_out))], -np.r_[loss_in, loss_out])
    rows.append({"C (inverse regularisation)": C, "train AUC": roc_auc_score(mi_train.y, pm_),
                 "test AUC": roc_auc_score(mi_out.y, po_), "mean loss members": loss_in.mean(),
                 "mean loss non-members": loss_out.mean(), "attack AUC": attack_auc})
mi = pd.DataFrame(rows)
RESULTS["membership_inference"] = mi.round(4).to_dict(orient="records")
print(mi.round(3).to_string(index=False))

# %% [markdown]
# **Expected (measured):** our regex redactor rewrites the synthetic string; in 40,000 test comments it
# finds no emails or phone numbers (the dataset was already filtered) but ~1,100 URLs. With C = 100 the
# model fits the training set perfectly (train AUC 1.000), members have a mean loss of 0.016 against 0.365
# for non-members, and the loss-threshold attack reaches AUC ≈ 0.64. Stronger regularisation (C = 0.1)
# brings the attack down to ≈ 0.57 at a small cost in test AUC (0.785 → 0.774). The attack AUC tracks
# the train/test loss gap, i.e. memorisation.
#
# ## 7 · Differential privacy: DP-SGD
#
# **Differential privacy** (DP) bounds how much any single training record can change what the
# training algorithm outputs: for neighbouring datasets `D`, `D'` (one record added or removed)
# and every set of outcomes `S`, `P[A(D) ∈ S] ≤ e^ε · P[A(D') ∈ S] + δ`. Small ε = strong privacy;
# membership inference then provably cannot do much better than guessing.
#
# **DP-SGD** (Abadi et al., 2016) makes SGD differentially private with two changes per step:
# 1. **clip each example's gradient** to L2 norm ≤ `C`, so no single record can move the model much;
# 2. **add Gaussian noise** `N(0, σ²C²)` to the *sum* of clipped gradients before averaging.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-x-dp-sgd.svg" alt="One DP-SGD step: per-example gradients, clip each to norm C, sum, add Gaussian noise with standard deviation sigma times C, average and step; a circle of radius C shows clipping" width="100%">
#
# *What to notice:* clipping bounds what any single record can contribute (at most C); the noise is scaled to that bound, so it hides any one record's contribution.
#
# > **Intuition.** A vote where each person's ballot can move the result by at most one unit
# > (clipping), and the official count is then reported with random jitter of a few units (noise):
# > looking at the result, nobody can tell whether any particular person voted. More voters (a larger
# > batch) make the same jitter matter less. *Where the analogy breaks down:* training repeats this
# > thousands of times on overlapping data, and the privacy loss accumulates; computing the total ε
# > needs an accountant, not intuition.
#
# The privacy cost ε then follows from the noise multiplier σ, the sampling rate and the number of
# steps, computed by a **privacy accountant** (Opacus, TensorFlow Privacy). We implement DP-SGD for
# logistic regression on 96-dimensional SVD features of the TF-IDF matrix (per-example gradients of
# logistic regression are `(p − y)·x`, so no special library is needed).

# %% [markdown]
# > **Exercise 2.** Write `clip_and_noise(per_example_grads, clip, sigma, rng)`: `per_example_grads` is
# > an `(n, d)` array. Clip each row to L2 norm at most `clip` (rows already within the bound are
# > unchanged), sum the rows, add `rng.normal(0, sigma * clip, size=d)`, and divide by `n`. Return the
# > `(d,)` noisy mean gradient.

# %%
def clip_and_noise(per_example_grads, clip, sigma, rng):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(clip_and_noise)

# %%
def reference_clip_and_noise(g, clip, sigma, rng):
    """The same operation as Exercise 2, so this section runs before you solve it."""
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    clipped = g * np.minimum(1.0, clip / np.maximum(norms, 1e-12))
    return (clipped.sum(0) + rng.normal(0.0, sigma * clip, size=g.shape[1])) / len(g)


t0 = time.perf_counter()
svd = TruncatedSVD(96, random_state=0).fit(X_train[:20_000])  # fit on a subset to keep it fast
Z_tr, Z_te = svd.transform(X_train), svd.transform(X_test)
mu, sd = Z_tr.mean(0), Z_tr.std(0) + 1e-8
Z_tr = np.c_[(Z_tr - mu) / sd, np.ones(len(Z_tr))]  # standardise, add a bias column
Z_te = np.c_[(Z_te - mu) / sd, np.ones(len(Z_te))]
y_tr = train.y.to_numpy()
print(f"SVD features in {time.perf_counter() - t0:.1f} s; non-private LR on them: test AUC "
      f"{roc_auc_score(test.y, LogisticRegression(max_iter=2000).fit(Z_tr[:, :-1], y_tr).predict_proba(Z_te[:, :-1])[:, 1]):.4f}")


def dp_sgd(sigma, clip=1.0, epochs=3, batch=256, lr=0.5, seed=0, private=True):
    rng = np.random.default_rng(seed)
    w = np.zeros(Z_tr.shape[1])
    for _ in range(epochs):
        for idx in np.array_split(rng.permutation(len(Z_tr)), len(Z_tr) // batch):
            p = 1 / (1 + np.exp(-Z_tr[idx] @ w))
            g = (p - y_tr[idx])[:, None] * Z_tr[idx]  # per-example gradients, (B, d)
            w -= lr * (reference_clip_and_noise(g, clip, sigma, rng) if private else g.mean(0))
    return roc_auc_score(test.y, Z_te @ w)


def epsilon_zcdp(sigma, epochs, delta):
    """A simple, conservative bound: each example is in one batch per epoch (Gaussian mechanism with
    sensitivity C and noise sigma*C -> rho = 1/(2 sigma^2) per use), composed over epochs in zCDP:
    rho = epochs / (2 sigma^2), then eps = rho + 2 sqrt(rho ln(1/delta)) (Bun & Steinke, 2016).
    It ignores privacy amplification by sampling, which an accountant would exploit (smaller eps)."""
    rho = epochs / (2 * sigma**2)
    return rho + 2 * np.sqrt(rho * np.log(1 / delta))


EPOCHS, DELTA = 3, 1 / len(Z_tr)
rows = [{"setting": "non-private SGD", "sigma": 0.0, "test AUC": dp_sgd(0.0, private=False), "epsilon (bound)": np.inf}]
rows.append({"setting": "clipping only", "sigma": 0.0, "test AUC": dp_sgd(0.0), "epsilon (bound)": np.inf})
for sigma in [0.5, 1.0, 2.0, 4.0, 8.0]:
    rows.append({"setting": "DP-SGD", "sigma": sigma, "test AUC": dp_sgd(sigma),
                 "epsilon (bound)": epsilon_zcdp(sigma, EPOCHS, DELTA)})
dp = pd.DataFrame(rows)
RESULTS["dp_sgd"] = dp.round(4).replace({np.inf: None}).to_dict(orient="records")
print(f"n = {len(Z_tr):,}, batch 256, {EPOCHS} epochs, clip C = 1, delta = 1/n = {DELTA:.1e}")
print(dp.round(4).to_string(index=False))
(OUT / "results.json").write_text(json.dumps(RESULTS, indent=2, default=float))
print("saved", OUT / "results.json")

# %% [markdown]
# **Expected (measured):** the 96 SVD dimensions keep only part of the signal (non-private logistic
# regression on them: test AUC ≈ 0.73, vs 0.89 on the full TF-IDF; DP-SGD on sparse high-dimensional
# features is possible but slower to write by hand). Plain SGD reaches ≈ 0.72 in 3 epochs; clipping alone
# changes little (≈ 0.72). Adding noise costs accuracy smoothly: σ = 1 (ε ≤ 9.6 by our bound) ≈ 0.72,
# σ = 2 (ε ≤ 4.4) ≈ 0.71, σ = 4 (ε ≤ 2.1) ≈ 0.68, σ = 8 (ε ≤ 1.0) ≈ 0.63. That is the privacy/utility
# trade-off: strong guarantees (ε around 1) cost real accuracy unless much more data is available (the
# noise on the mean gradient shrinks as 1/batch size) or a pretrained model (fine-tune few parameters).
# The ε column comes from the stated zCDP formula, a loose upper bound for this simplified sampling
# scheme; a real privacy claim needs an accountant (e.g. Opacus with Poisson sampling), which would
# report a smaller ε for the same σ.

# %% [markdown]
# > **Exercise 3.** A toxicity filter's overall AUC is 0.95, but BPSN AUC for comments mentioning
# > *muslim* is 0.80 and the FPR at the production threshold is 3× the background's. (a) Explain to a
# > product manager what that means for users. (b) Give three likely causes in the data or model.
# > (c) Propose three mitigations (data, model, policy) and the trade-off of each. (d) What would you
# > monitor after launch, and what would make you roll back? Write your answer in the cell below.

# %% [markdown]
# *Your answer (Exercise 3):* ...

# %% [markdown]
# ## Key equations
#
# **1 · AUC as a probability, and the three bias AUCs.**
#
# $$
# \mathrm{AUC}(P, N) = \Pr\big(s(x^{+}) > s(x^{-})\big), \quad x^{+} \sim P,\ x^{-} \sim N \quad (\text{ties count } \tfrac12)
# $$
#
# $$
# \begin{aligned}
# \text{Subgroup AUC} &= \mathrm{AUC}(\text{SP},\ \text{SN}) \\
# \text{BPSN AUC} &= \mathrm{AUC}(\text{BP},\ \text{SN}) \\
# \text{BNSP AUC} &= \mathrm{AUC}(\text{SP},\ \text{BN})
# \end{aligned}
# $$
#
# *Symbols:* S/B = subgroup / background, P/N = toxic / non-toxic (so SN = non-toxic comments that
# mention the group). *Example:* BPSN = 0.70 means that in 30% of (toxic background, benign
# subgroup) pairs, the benign subgroup comment scores *higher*.
#
# **2 · Error rates per group at threshold $t$.**
#
# $$
# \mathrm{FPR}_g(t) = \Pr(s \ge t \mid y = 0, g), \qquad \mathrm{FNR}_g(t) = \Pr(s < t \mid y = 1, g)
# $$
#
# *Example (section 2):* $\mathrm{FPR}_{\text{black}}(0.5) = 0.083$ vs 0.010 overall: an 8× ratio.
#
# **3 · Jigsaw summary (power mean, $p = -5$).** $M_p(m_1, \dots, m_G) = \big(\tfrac{1}{G} \sum_g m_g^{\,p}\big)^{1/p}$;
# a negative $p$ pulls the mean towards the *worst* group. *Example:* $M_{-5}(0.9, 0.7) = 0.765$, while the plain mean is 0.80.
#
# **4 · Differential privacy.**
#
# $$
# \Pr[A(D) \in S] \le e^{\varepsilon}\, \Pr[A(D') \in S] + \delta \quad \text{for all neighbouring } D, D' \text{ and all } S
# $$
#
# *In words:* adding or removing one record changes the probability of any outcome by at most a factor
# $e^{\varepsilon}$ (plus a small slack $\delta$). *Example:* $\varepsilon = 1$ means at most a factor $e \approx 2.7$.
#
# **5 · One DP-SGD step.**
#
# $$
# \bar{g}_i = g_i \cdot \min\!\left(1, \frac{C}{\lVert g_i \rVert_2}\right), \qquad \tilde{g} = \frac{1}{n}\left( \sum_{i=1}^{n} \bar{g}_i + \mathcal{N}(0, \sigma^2 C^2 I) \right), \qquad \theta \leftarrow \theta - \eta\, \tilde{g}
# $$
#
# *Symbols:* $C$ the clipping norm, $\sigma$ the noise multiplier, $n$ the batch size. The noise on the
# mean gradient has standard deviation $\sigma C / n$ per coordinate. *Example:* $C = 1$, $\sigma = 2$,
# $n = 256$: $2/256 \approx 0.0078$.
#
# The cell below checks 1, 3 and 5 numerically on this notebook's scores.

# %%
# Key equations, checked numerically (seconds on a CPU).
y_arr, m = test.y.to_numpy(), test["muslim"].to_numpy()
bp, sn = s_test[~m & (y_arr == 1)], s_test[m & (y_arr == 0)]
rng_chk = np.random.default_rng(0)
bp_s = rng_chk.choice(bp, min(2000, len(bp)), replace=False)
pairwise = np.mean(bp_s[:, None] > sn[None, :]) + 0.5 * np.mean(bp_s[:, None] == sn[None, :])
sk = roc_auc_score(np.r_[np.ones(len(bp_s)), np.zeros(len(sn))], np.r_[bp_s, sn])
print(f"BPSN (muslim) as P(toxic background > benign subgroup): pairwise {pairwise:.4f} | roc_auc_score {sk:.4f}")
print(f"power mean M_-5(0.9, 0.7) = {pm([0.9, 0.7]):.3f} | plain mean {np.mean([0.9, 0.7]):.3f}")

g = rng_chk.normal(0, 3, size=(256, 20))  # 256 per-example gradients, most longer than C = 1
norms_before = np.linalg.norm(g, axis=1)
clipped = g * np.minimum(1.0, 1.0 / norms_before)[:, None]
noise_only = [reference_clip_and_noise(np.zeros((256, 20)), 1.0, 2.0, rng_chk) for _ in range(2000)]
print(f"norms before clipping: median {np.median(norms_before):.1f} | after: max {np.linalg.norm(clipped, axis=1).max():.3f}")
print(f"std of the noise on the mean gradient: measured {np.std(noise_only):.5f} | sigma*C/n {2.0 * 1.0 / 256:.5f}")

# %% [markdown]
# **Expected:** the pairwise probability and `roc_auc_score` agree (to about three decimals, on the
# same subsample; both close to the BPSN for *muslim* in section 2); `M_-5 = 0.765` vs a plain mean of 0.800;
# gradient norms around 13 before clipping and at most `1.000` after; and a measured noise standard
# deviation within ~1% of `0.00781`.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `scaleml` | setup cell not run | run the first code cell |
# | Download fails or is very slow | Hub outage or no network | re-run later; the file is cached after the first success |
# | `ValueError: Only one class present in y_true` | a subgroup slice with no toxic (or no non-toxic) examples | use more test data; skip or merge tiny subgroups and report their size |
# | Subgroup metrics jump between runs | small subgroups (a few hundred examples, a few dozen positives) | bootstrap confidence intervals; never act on a gap smaller than its CI |
# | DP-SGD accuracy collapses | noise too large for the batch size (noise on the mean is σC/B) | larger batches, fewer steps, tune C; accept the privacy/utility trade-off |
# | A privacy claim without an ε from an accountant | the bound used is informal | use Opacus / TF Privacy accounting with the exact sampling scheme |
# | Memory error loading the parquet | the full file has 448k comments | load fewer columns (as here), or sample by row group |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Define subgroup AUC, BPSN AUC and BNSP AUC. Which one detects "mentions of an identity are flagged as toxic", and why?
# 2. Why can't one threshold equalise false-positive rates across groups, and why can't you equalise every error rate at once?
# 3. How would you test a classifier's robustness before launch? Name a shift test and a perturbation test, and what each would catch.
# 4. What is membership inference, what makes a model vulnerable, and how do you measure it?
# 5. Explain DP-SGD step by step. What do the clipping norm and the noise multiplier control, and what does ε mean?
# 6. Design question: you are launching a comment-moderation model for a new country. What fairness, robustness and privacy checks go into the launch review?
#
# **Next chapter:** [Chapter 54 · Evaluating LLM systems & guardrails](../ch54-evaluating-llm-systems-and-guardrails/notebook.ipynb) — the same evaluation discipline for systems built on language models: eval sets, judges, prompt injection and guardrails.
