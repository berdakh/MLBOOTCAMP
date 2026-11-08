# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 29 · A/B testing fundamentals
#
# **Day 6 · ~90 minutes · CPU only**
#
# Offline metrics nominate a ranker; an online experiment decides whether it
# ships. Most bad launch decisions are not caused by exotic statistics. They
# come from a wrong randomisation unit, an underpowered test, a broken
# assignment, a peek, or twenty metrics read at once. This notebook *measures*
# each of these failures on simulated experiments where, unlike in real life,
# you know the true effect.
#
# By the end you will be able to:
#
# 1. explain why the randomisation unit drives the variance, and show a naive test's false-positive rate under an A/A test;
# 2. choose between per-user CTR and pooled CTR, and say what each one estimates;
# 3. compute a sample size in closed form and check it by simulation;
# 4. run a t-test and a bootstrap and say when they disagree;
# 5. detect a sample-ratio mismatch and explain why it invalidates the result;
# 6. quantify how peeking and multiple metrics inflate false positives, and correct for them.
#
# **You will produce:** `artifacts/level2/nb11/aa_tests.csv`, the measured false-positive
# rates of every test in this notebook, and the simulator
# [`src/shoprank/experiments.py`](../../src/shoprank/experiments.py), reused in notebook 12.

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-part2.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Catch up: only the ShopStream *world* is needed (no logs), < 2 s ------
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from shoprank import experiments as ex
from shoprank.data import WorldConfig, make_world
from grading import ch29 as grade

OUT = ART / "level2" / "nb11"
OUT.mkdir(parents=True, exist_ok=True)

world = make_world(WorldConfig.scale("small"))
pop = ex.user_population(world)
print(pop[["activity", "base_ctr"]].describe().round(3).T[["mean", "std", "min", "50%", "max"]])

# %% [markdown]
# **Expected:** mean activity about 1.4 (heavy-tailed, max above 10) and mean
# base CTR about 0.14 (spread 0.04–0.3), close to the CTR of the logging policy in
# ShopStream's logs.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Randomisation unit | the thing one coin flip assigns to an arm | the user: all of a user's requests see the same ranker |
# | A/A test | an experiment where both arms get the identical treatment, so every "win" is a false positive | 400 simulated A/A runs measure each test's false-positive rate |
# | Exposed (triggered) user | an assigned user who actually reached the tested surface | about 5% of assigned users never make a request |
# | Pooled vs per-user CTR | total clicks ÷ total impressions, versus the average of each user's own click rate | both about 0.145 per arm in 11.2 |
# | Standard error (SE) | the typical size of an estimate's random error across repeated experiments | naive SE 0.0006 vs a true spread of 0.0017 |
# | Significance level α | the false-positive rate you accept when there is no effect | α = 0.05 |
# | Power (1 − β) | the chance the test detects an effect that is really there | 80% power for 10% → 11% at 14,749 users per arm |
# | MDE | minimum detectable effect: the smallest true effect the test is sized to catch | +1 percentage point of conversion |
# | Sample-ratio mismatch (SRM) | arm sizes too unequal to be chance, which means broken assignment or logging | 0.505 of users in control, SRM p ≈ 3e-6 |
# | Peeking | checking repeatedly and stopping at the first significant result | 14 daily looks give a false-positive rate of about 0.2 |
# | FWER / FDR | the chance of at least one false positive / the expected share of false ones among discoveries | Bonferroni controls FWER, Benjamini–Hochberg controls FDR |
#
# The whole notebook on one page: each section below measures what goes wrong when one of these steps is skipped.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-x-ab-test-lifecycle.svg" alt="Six steps of an A/B test: hypothesis, power, randomise, SRM check, analyse, decide, each with the failure caused by skipping it" width="100%">
#
# *What to notice:* steps 1–4 all happen before any metric is visible; that ordering, not the formula, is what protects the result.

# %% [markdown]
# ## 11.1 · The simulator
#
# `ex.simulate_users(pop, n, rng, lift)` draws `n` users from this
# population, simulates a two-week **pre-period** (no treatment) and a two-week
# **experiment period**, and randomises each user to control (arm 0) or
# treatment (arm 1). Requests per user are Poisson with the user's activity;
# each request shows 10 items; clicks are binomial with the user's CTR, which
# drifts a little between periods and is multiplied by `1 + lift` in treatment.
# Read the function: 40 lines, no magic.
#
# Our "new ranker" improves CTR by **2% relative** (0.140 → 0.143): a typical,
# worthwhile ranking win, and a small one to detect.

# %%
LIFT = 0.02
rng = np.random.default_rng(0)
one = ex.simulate_users(pop, 40_000, rng, lift=LIFT)
exposed = one[one["impressions"] > 0]
print(f"{len(one):,} users assigned, {len(exposed):,} with at least one request "
      f"({len(one) - len(exposed):,} never showed up)")
one.head(3)

# %% [markdown]
# **Expected:** about 5% of assigned users make no request in two weeks. An
# analysis that counts them would dilute any effect; standard practice is to
# analyse users who were **exposed** (triggered), assigning them *before*
# the outcome, so the filter cannot depend on treatment.
#
# ## 11.2 · Which CTR? Per-user vs pooled
#
# * **Pooled CTR** = Σ clicks / Σ impressions over all users of an arm. Heavy users
#   carry more weight. It answers "what share of impressions get clicked?"
# * **Per-user CTR** = mean over users of (clicks_u / impressions_u). Every user
#   counts once. It answers "how does the typical user respond?"
#
# They are different *estimands*. Either can be right, but you must pick one
# before the test, and analyse it with the correct unit of variance.

# %%
by_arm = exposed.groupby("arm").apply(lambda g: pd.Series({
    "users": len(g),
    "pooled_ctr": g["clicks"].sum() / g["impressions"].sum(),
    "per_user_ctr": (g["clicks"] / g["impressions"]).mean(),
}), include_groups=False)
by_arm.loc["relative lift"] = by_arm.loc[1] / by_arm.loc[0] - 1
print(by_arm.round(4))

# %% [markdown]
# **Expected:** both CTRs are about 0.145 in each arm, and the measured relative lifts are
# noisy: typically anywhere from 0% to +4% in one experiment of this size (we saw
# +3.1% pooled and +3.3% per-user), although the true lift is exactly 2%.
# (In this simulator activity and CTR are independent, so the two CTRs are
# close. In real products heavy users usually click differently, and the two
# metrics can move in different directions.) How noisy, and which test is
# honest about it, is the next section.
#
# ## 11.3 · The randomisation unit decides the variance
#
# Clicks from the same user are correlated: a user with a high CTR clicks a
# lot on *every* request. If you randomise **users** but test the pooled
# CTR as if every **impression** were independent (the two-proportion z-test
# you learned first), the standard error is too small and the test fires too often.
#
# The way to *measure* that is an **A/A test**: both arms get the same
# ranker (`lift=0`), so every "significant" result is a false positive.
# A valid test at α = 0.05 should fire about 5% of the time. We run 400 A/A
# experiments of 20,000 users each:

# %%
N_SIMS, N_USERS = 400, 20_000
rng = np.random.default_rng(1)
rows = []
start = time.time()
for _ in range(N_SIMS):
    d = ex.simulate_users(pop, N_USERS, rng, lift=0.0)
    d = d[d["impressions"] > 0]
    a, b = d[d["arm"] == 0], d[d["arm"] == 1]
    naive = ex.two_proportion_ztest(a["clicks"].sum(), a["impressions"].sum(), b["clicks"].sum(), b["impressions"].sum())
    per_user = ex.welch_ttest(a["clicks"] / a["impressions"], b["clicks"] / b["impressions"])
    r = ex.simulate_request_randomized(pop, N_USERS, rng, lift=0.0)
    req = ex.two_proportion_ztest(r["clicks_0"].sum(), r["imps_0"].sum(), r["clicks_1"].sum(), r["imps_1"].sum())
    rows.append({"user-rand, impression z-test (pooled CTR)": naive["p"],
                 "user-rand, user t-test (per-user CTR)": per_user["p"],
                 "request-rand, impression z-test (pooled CTR)": req["p"],
                 "diff_user": naive["diff"], "se_naive": naive["se"], "diff_req": req["diff"]})
aa = pd.DataFrame(rows)
tests = [c for c in aa.columns if "test" in c]
fpr = (aa[tests] < 0.05).mean()
print(f"{N_SIMS} A/A experiments in {time.time() - start:.0f} s. False-positive rate at alpha = 0.05:")
print(fpr.round(3).to_string())
print(f"\npooled-CTR difference, user-randomised   : true SD across experiments {aa['diff_user'].std():.5f} "
      f"vs naive SE {aa['se_naive'].mean():.5f}")
print(f"pooled-CTR difference, request-randomised: true SD across experiments {aa['diff_req'].std():.5f}")

# %% [markdown]
# **Expected (our runs):**
#
# * the impression-level z-test on user-randomised data fires about **50%**
#   of the time instead of 5%: its SE (about 0.0006) is too small by a factor of
#   about 2.8 compared with the true spread of the estimate (about 0.0017);
# * the user-level t-test fires about 4–5% (400 runs give about ±2% simulation noise);
# * with **request** randomisation the true SD of the estimate is much *smaller*
#   (about 0.0007): each user appears in both arms, so their personal CTR largely
#   cancels out of the difference. The impression-level test is much closer to
#   valid but still fires about 9% in our runs: heavy users' requests split
#   unevenly by chance, which a binomial SE ignores.
#
# So why not always randomise requests? Because the unit must also match **what the
# treatment does and what you measure**:
#
# * a user who sees ranker A in the morning and B at noon has an
#   inconsistent experience, and *carry-over* (B taught them to scroll more)
#   contaminates A's requests: **interference** between units;
# * user-level outcomes (retention, sessions per week, purchases over a
#   month) do not exist at the request level;
# * marketplaces have interference *between users* too (two rankers
#   competing for the same limited inventory or ad budget), which needs
#   cluster or switchback designs.
#
# Rule: randomise at the level of the unit whose behaviour the treatment
# changes (usually the user), and compute variance **at that unit**: a
# per-user metric with a t-test, or a pooled ratio with the delta method (notebook 12).
#
# > **Intuition.** Treating 1,000 impressions from 50 users as 1,000 independent
# > data points is like polling 50 people 20 times each and reporting 1,000 opinions:
# > a person's answers largely repeat, so the honest sample size is much closer to 50.
# > *Where it breaks down:* repeated impressions are not exact copies, so the effective
# > sample size lies somewhere between users and impressions; the delta method
# > (notebook 12) computes exactly where.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-x-randomization-unit.svg" alt="Request, user and cluster randomisation compared on variance and interference risk" width="100%">
#
# *What to notice:* moving right (request → user → cluster) trades variance for protection against interference; the analysis unit must move with it.

# %%
fpr.rename("false_positive_rate").to_frame().to_csv(OUT / "aa_tests.csv")

# %% [markdown]
# ## 11.4 · Power and sample size
#
# For a metric that is a **proportion** over independent units (did the user
# convert?), the sample size per arm to detect an absolute difference `δ`
# with a two-sided test at level α and power 1-β is
#
# $$ n = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2 \, \big(p_0(1-p_0) + p_1(1-p_1)\big)}{\delta^2}, \qquad p_1 = p_0 + \delta. $$
#
# Read it as an interview answer: halving the effect **quadruples** the sample
# size; the `z` terms are about 2.8 for α = 0.05 and 80% power; and the
# variance term is largest for proportions near 0.5.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-x-power.svg" alt="Two bell curves of the test statistic under no effect and under the true effect, with alpha, beta and power shaded" width="100%">
#
# *What to notice:* power is the green area of the right curve beyond the line; it grows when δ moves the curves apart or when more users make both curves narrower.
#
# > **Intuition.** Detecting a lift is hearing a whisper in a noisy room. The
# > whisper is δ; the noise is the standard error, which falls only like 1/√n. A whisper
# > half as loud needs four times the listeners. *Where it breaks down:* the "noise"
# > also depends on the metric (σ), which is why CUPED (notebook 12) can lower it without
# > adding users.

# %% [markdown]
# > **Exercise 1.** Write `sample_size_per_arm(p0, mde_abs, alpha=0.05, power=0.8)` using the
# > formula above (two-sided; `scipy.stats.norm.ppf` gives the `z` values). Round **up** to an integer.

# %%
def sample_size_per_arm(p0, mde_abs, alpha=0.05, power=0.8):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(sample_size_per_arm)

# %% [markdown]
# A formula is a claim; simulation checks it. The next cell uses its own copy of
# the formula (so it runs before you finish the exercise), simulates 2,000
# experiments with a 10% → 11% conversion rate at that sample size, and
# measures the power:

# %%
def n_formula(p0, d, alpha=0.05, power=0.8):
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil(z**2 * (p0 * (1 - p0) + (p0 + d) * (1 - p0 - d)) / d**2))


p0, d = 0.10, 0.01
n = n_formula(p0, d)
rng = np.random.default_rng(2)
conv_a = rng.binomial(n, p0, 2000)            # converted users per arm, 2,000 experiments
conv_b = rng.binomial(n, p0 + d, 2000)
pvals = ex.two_proportion_ztest(conv_a, n, conv_b, n)["p"]
print(f"n per arm = {n:,} | measured power = {np.mean(pvals < 0.05):.3f} (target 0.80)")
print(f"half the effect (0.5 pp) needs n = {n_formula(p0, d / 2):,} per arm")

# %% [markdown]
# **Expected:** n = 14,749 per arm, measured power about 0.80 (±0.02), and
# about 4x the users for half the effect.
#
# **Beyond proportions.** For per-user CTR the units are users and the metric is
# continuous, so use the general form `n = 2 σ² (z₁₋α/₂ + z₁₋β)² / δ²` with σ
# the per-user standard deviation, estimated **from pre-period data** before
# the test starts (never from the test itself). Our 2% relative lift:

# %%
rng = np.random.default_rng(3)
pre = ex.simulate_users(pop, 50_000, rng)          # historical data, no treatment
pre = pre[pre["pre_impressions"] > 0]
ctr_pre = pre["pre_clicks"] / pre["pre_impressions"]
sigma, mean = ctr_pre.std(), ctr_pre.mean()
delta = LIFT * mean
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
n_ctr = int(np.ceil(2 * sigma**2 * z**2 / delta**2))
print(f"per-user CTR: mean {mean:.4f}, sd {sigma:.4f} -> n = {n_ctr:,} exposed users per arm for a {LIFT:.0%} lift")

start, hits = time.time(), []
for _ in range(200):
    dd = ex.simulate_users(pop, int(2 * n_ctr / 0.95), rng, lift=LIFT)  # assign extra: ~5% never show up
    dd = dd[dd["impressions"] > 0]
    ctr = dd["clicks"] / dd["impressions"]
    hits.append(ex.welch_ttest(ctr[dd["arm"] == 0], ctr[dd["arm"] == 1])["p"] < 0.05)
print(f"measured power over 200 simulated experiments: {np.mean(hits):.2f} ({time.time() - start:.0f} s)")

# %% [markdown]
# **Expected:** a per-user CTR SD of about 0.097, about 17,500 exposed users per arm,
# and a measured power of about 0.78–0.80 (200 runs give about ±0.06 of noise). Two practical notes:
#
# * In real products the pre-period SD is often a little smaller than the
#   experiment's (novelty, seasonality, new users), so a planned test can be
#   slightly under-powered. Pad the estimate, or plan for 85–90% power.
# * Variance reduction (CUPED, notebook 12) cuts that sample size substantially
#   using the same pre-period data.
#
# ## 11.5 · t-test vs bootstrap
#
# The t-test assumes the *mean* is approximately normal (true for large n by
# the CLT, even for skewed per-user metrics). The bootstrap assumes only that the
# sample represents the population, and works for any statistic (medians,
# ratios, percentiles), at a computational cost.

# %%
rng = np.random.default_rng(4)
d = ex.simulate_users(pop, 60_000, rng, lift=LIFT)
d = d[d["impressions"] > 0]
ctr = d["clicks"] / d["impressions"]
a, b = ctr[d["arm"] == 0].to_numpy(), ctr[d["arm"] == 1].to_numpy()
t = ex.welch_ttest(a, b)
start = time.time()
lo, hi = ex.bootstrap_diff_ci(a, b, n_boot=1000, rng=rng)
print(f"difference {t['diff']:+.5f} | t-test 95% CI [{t['diff'] - 1.96 * t['se']:+.5f}, {t['diff'] + 1.96 * t['se']:+.5f}]"
      f" p={t['p']:.3f}")
print(f"bootstrap 95% CI [{lo:+.5f}, {hi:+.5f}] ({time.time() - start:.1f} s for 1,000 resamples)")

# %% [markdown]
# **Expected:** a difference of about +0.003 with both intervals excluding zero
# and agreeing to within about 0.0001; the bootstrap takes well under a second
# here (a few seconds for larger arms). With ~28,000 users per arm the CLT has
# done its job, and the t-test is the cheaper choice. Reach for the bootstrap when the
# statistic has no simple standard error (a median, a percentile, a ratio of
# sums), when samples are small and skewed, or as a check on a hand-derived
# variance, as notebook 12 does for the delta method. Always resample the
# **randomisation unit** (users), never impressions.

# %% [markdown]
# ## 11.6 · Sample-ratio mismatch: check the assignment before the result
#
# You asked for a 50/50 split. If the arms' user counts are too unequal to be
# chance, something between assignment and logging is broken, and **any**
# metric difference is suspect, because the missing users are rarely missing at random.
#
# > **Intuition.** An SRM is a survey where, in one arm only, unhappy customers hang
# > up before answering. The answers you do get look happier, and no amount of
# > statistics on the answers can tell you that. *Where it breaks down:* a mismatch can
# > also be harmless (a bot filter applied evenly but noisily); you only know after
# > finding the cause, which is why you investigate rather than reweight.
#
# A planted bug, modelled on real ones: the new ranker's client code logs
# the exposure event lazily, on the user's first interaction. Users who never
# click are logged only half the time in treatment. The ranker itself is
# **identical** in both arms (an A/A test).

# %%
rng = np.random.default_rng(5)
d = ex.simulate_users(pop, 200_000, rng, lift=0.0)
d = d[d["impressions"] > 0]
lost = (d["arm"] == 1) & (d["clicks"] == 0) & (rng.random(len(d)) < 0.5)
logged = d[~lost]
n_a, n_b = (logged["arm"] == 0).sum(), (logged["arm"] == 1).sum()
ctr = logged["clicks"] / logged["impressions"]
res = ex.welch_ttest(ctr[logged["arm"] == 0], ctr[logged["arm"] == 1])
print(f"users: control {n_a:,} | treatment {n_b:,} | share in control {n_a / (n_a + n_b):.4f}")
print(f"per-user CTR 'lift' {res['diff'] / ctr[logged['arm'] == 0].mean():+.2%}, p = {res['p']:.2g}  <- an A/A test!")

# %% [markdown]
# **Expected:** about 2% fewer users in treatment (a share in control of
# about 0.505), a "lift" of roughly +2% with a p-value around 1e-10: a
# highly significant win in an A/A test, as large as the real ranker's effect. The bias is large because the lost users are
# exactly the ones with the lowest CTR. Whether a 0.505 split is suspicious is a statistics
# question, which you answer with a **chi-square goodness-of-fit test** of the
# observed counts against the expected split.

# %% [markdown]
# > **Exercise 2.** Write `srm_pvalue(n_a, n_b, expected_ratio=0.5)`: the p-value of a chi-square
# > goodness-of-fit test (1 degree of freedom) that the counts `(n_a, n_b)` come from a split
# > where a fraction `expected_ratio` of users should land in arm A.

# %%
def srm_pvalue(n_a, n_b, expected_ratio=0.5):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(srm_pvalue)

# %%
p_srm = stats.chisquare([n_a, n_b], f_exp=[(n_a + n_b) / 2] * 2).pvalue
clean = d.groupby("arm").size()
print(f"SRM p-value with the bug: {p_srm:.1e} | without it: {stats.chisquare(clean.to_numpy()).pvalue:.2f}")

# %% [markdown]
# **Expected:** an SRM p-value of about 1e-6 (3e-6 in our run) with the bug, and an unremarkable
# one (anything from 0.05 to 1) without it. With only 40,000 users the same bug gave
# an SRM p-value of about 0.02 in our runs (not alarming at p < 0.001) while the
# biased metric was already "significant": a small mismatch can hide a large bias,
# so treat any SRM as a reason to investigate. Teams commonly alert at
# p < 0.001, because the check runs on every experiment and false alarms are costly.
# When it fires, **do not** "re-weight the arms" and read the result anyway:
# find the cause (logging, bots filtered differently, redirects, crashes in one
# arm), fix it, and restart the experiment.
#
# ## 11.7 · Peeking
#
# A dashboard updates daily. Someone checks it every morning and stops the test
# the first day it is significant. Each look is a fresh chance for noise to
# cross the threshold, so the false-positive rate of the *procedure* is far
# above α. Simulated A/A tests: 14 daily looks, conversion rate 10%, 1,000 new users per arm per day.

# %%
rng = np.random.default_rng(6)
n_exp, days, per_day, p = 4000, 14, 1000, 0.10
conv = rng.binomial(per_day, p, size=(n_exp, 2, days)).cumsum(axis=2)   # cumulative conversions
n_cum = per_day * np.arange(1, days + 1)
pv = ex.two_proportion_ztest(conv[:, 0], n_cum, conv[:, 1], n_cum)["p"]  # (n_exp, days)
peek_fpr = [(pv[:, :look] < 0.05).any(axis=1).mean() for look in range(1, days + 1)]
bonf = (pv < 0.05 / days).any(axis=1).mean()
print(f"test once at day 14 : FPR {np.mean(pv[:, -1] < 0.05):.3f}")
print(f"peek daily, stop at first p<0.05: FPR {peek_fpr[-1]:.3f}")
print(f"peek daily with Bonferroni alpha/14 : FPR {bonf:.3f} (valid, but very conservative)")
plt.figure(figsize=(5, 3))
plt.plot(range(1, days + 1), peek_fpr, marker="o")
plt.axhline(0.05, color="gray", ls="--")
plt.xlabel("number of daily looks")
plt.ylabel("A/A false-positive rate")
plt.title("Peeking inflates false positives")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Expected:** about 0.05 for a single test at the end, about **0.2** for daily
# peeking with stopping (rising with every look; see the plot), and about 0.02 with Bonferroni.
# The right tools if you need to look early are **sequential tests** designed for
# it: group-sequential boundaries with alpha spending (O'Brien–Fleming spends little
# alpha early) or always-valid p-values (mSPRT). Most experimentation platforms
# offer one. Otherwise: fix the duration from the power calculation, look only for
# guardrail breakages (crashes, latency), and read the primary metric once.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-x-peeking.svg" alt="A z-statistic random walk in an A/A test crosses 1.96 on day 5 and ends near zero on day 14; a sequential boundary starts high and narrows" width="100%">
#
# *What to notice:* the day-5 crossing is pure noise; a sequential boundary demands much stronger evidence early, so the same walk would never have stopped.
#
# > **Intuition.** Peeking is rolling a die every morning and stopping the first time
# > it shows a six: run long enough and you always "find" one. *Where it breaks down:*
# > successive looks are correlated (day 6 contains day 5's data), so the inflation grows
# > more slowly than for independent rolls: about 0.2 after 14 looks, not 1 − 0.95¹⁴ ≈ 0.51.

# %% [markdown]
# > **Exercise 3.** Write `aa_false_positive_rate(n_experiments, n_users_per_arm, n_looks, alpha=0.05, p=0.1, seed=0)`.
# > Simulate `n_experiments` A/A tests with conversion rate `p` in both arms; users arrive in
# > `n_looks` equal batches (`n_users_per_arm // n_looks` per arm per look). After each batch, run a
# > two-sided pooled two-proportion z-test on the cumulative counts. Return the fraction of
# > experiments in which **any** look had `p-value < alpha`. Use `np.random.default_rng(seed)`
# > so that the result is reproducible.

# %%
def aa_false_positive_rate(n_experiments, n_users_per_arm, n_looks, alpha=0.05, p=0.1, seed=0):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(aa_false_positive_rate)

# %% [markdown]
# ## 11.8 · Many metrics, many chances
#
# A launch review shows 20 metrics. Even if the ranker changes nothing, each
# has a 5% chance of a "significant" move. Below, 2,000 simulated experiments
# with 20 metrics (pairwise correlation 0.3), of which 3 have a true effect
# of 3 standard errors and 17 have none. We simulate the 20 test statistics
# directly from their joint normal distribution, which is what the CLT gives for
# per-user means; it is the same as simulating users, just faster.

# %%
rng = np.random.default_rng(7)
m, n_true, n_exp = 20, 3, 2000
corr = np.full((m, m), 0.3) + 0.7 * np.eye(m)
effect = np.r_[np.full(n_true, 3.0), np.zeros(m - n_true)]
zs = rng.multivariate_normal(np.zeros(m), corr, size=n_exp) + effect
pvals = 2 * stats.norm.sf(np.abs(zs))
methods = {
    "no correction": pvals < 0.05,
    "Bonferroni (alpha/20)": pvals < 0.05 / m,
    "Benjamini-Hochberg (q=0.05)": np.array([ex.benjamini_hochberg(row, 0.05) for row in pvals]),
}
summary = {}
for name, rej in methods.items():
    false_rej = rej[:, n_true:]
    summary[name] = {
        "P(any false positive)": false_rej.any(axis=1).mean(),
        "false discovery proportion": np.mean(false_rej.sum(1) / np.maximum(rej.sum(1), 1)),
        "power (true effects found)": rej[:, :n_true].mean(),
    }
print(pd.DataFrame(summary).T.round(3).to_string())

# %% [markdown]
# **Expected:** without correction, about half the experiments show at least
# one false "win" among the 17 null metrics (about 0.49; less than the
# `1 - 0.95^17 ≈ 0.58` of independent metrics because they are correlated), and
# about 19% of all "significant" metrics are false. Bonferroni brings the chance
# of any false positive to about 0.03 at the cost of power (0.86 → 0.50);
# Benjamini–Hochberg controls the *proportion* of false discoveries (about 3%)
# and keeps a little more power (about 0.59).
#
# In practice, the cleanest defence is a design decision, not a correction:
# **one** pre-registered primary metric decides the launch; a handful of
# **guardrails** (latency, crashes, revenue) can only block it; everything else is
# exploratory and labelled as such. Notebook 12 turns that into launch criteria.

# %% [markdown]
# ## Key equations
#
# A revision sheet: the formulas behind this notebook, each with a tiny worked example. The code cell after it checks every example numerically.
#
# **1. Two-proportion z-test** (pooled; valid only if the n trials are independent units)
#
# $$
# z = \frac{\hat p_B - \hat p_A}{\sqrt{\hat p\,(1-\hat p)\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}}, \qquad \hat p = \frac{x_A + x_B}{n_A + n_B}
# $$
#
# *In words:* the observed difference in rates, measured in standard errors computed as if both arms shared one rate (the null hypothesis).
# *Symbols:* $x$ = conversions, $n$ = units, $\hat p_A = x_A / n_A$, $\hat p$ = the pooled rate.
# *Example:* 1,000 / 10,000 vs 1,100 / 10,000 → $\hat p = 0.105$, SE = 0.00434, z = 2.31, p = 0.021.
#
# **2. Standard error of a difference, and its confidence interval**
#
# $$
# \operatorname{SE}(\bar y_B - \bar y_A) = \sqrt{\frac{\sigma_A^2}{n_A} + \frac{\sigma_B^2}{n_B}}, \qquad \text{95\% CI} = (\bar y_B - \bar y_A) \pm 1.96\,\operatorname{SE}
# $$
#
# *In words:* variances of independent estimates add, so the SE of a difference is the root of the summed squared SEs.
# *Symbols:* $\sigma^2$ = per-unit variance in each arm ($p(1-p)$ for a proportion), $n$ = units per arm.
# *Example:* $\sqrt{0.1 \cdot 0.9/10^4 + 0.11 \cdot 0.89/10^4} = 0.00433$, so the CI for +0.01 is [+0.0015, +0.0185].
#
# **3. Sample size per arm**
#
# $$
# n \approx \frac{2\,(z_{1-\alpha/2} + z_{1-\beta})^2\, p(1-p)}{\delta^2}
# $$
#
# *In words:* (how many standard errors of separation you need)² × (noise per unit) ÷ (effect)². The 2 is there because both arms are noisy.
# *Symbols:* $z_{1-\alpha/2} = 1.96$ for α = 0.05, $z_{1-\beta} = 0.84$ for 80% power, $p$ = baseline rate, $\delta$ = absolute MDE. For a continuous metric replace $p(1-p)$ with $\sigma^2$.
# *Example:* $p = 0.10$, $\delta = 0.01$: $2 \cdot 2.80^2 \cdot 0.09 / 0.0001 \approx 14{,}128$; the exact two-variance form used in 11.4 gives 14,749. Halve δ and n quadruples.
#
# **4. Sample-ratio mismatch: chi-square goodness of fit**
#
# $$
# \chi^2 = \sum_{\text{arms}} \frac{(O - E)^2}{E}, \qquad \text{1 degree of freedom for two arms}
# $$
#
# *In words:* how far the observed arm counts are from the planned split, scaled by the expected counts.
# *Symbols:* $O$ = observed users in an arm, $E$ = total × planned share.
# *Example:* 50,600 vs 49,400 users on a 50/50 split: $\chi^2 = 2 \cdot 600^2 / 50{,}000 = 14.4$, p ≈ 1.5e-4, below the usual 0.001 alarm.
#
# **5. Multiple testing: Bonferroni and Benjamini–Hochberg**
#
# $$
# \text{Bonferroni: reject } H_i \text{ if } p_i \le \frac{\alpha}{m}; \qquad \text{BH: } k = \max\Big\{ i : p_{(i)} \le \frac{i}{m}\,q \Big\}, \text{ reject } p_{(1)}, \dots, p_{(k)}
# $$
#
# *In words:* Bonferroni splits α evenly over m tests (controls the chance of **any** false positive); BH sorts the p-values and compares each with a rising threshold (controls the **share** of false discoveries). With m independent null tests, $P(\text{any false positive}) = 1 - (1-\alpha)^m$.
# *Symbols:* $m$ = number of tests, $p_{(i)}$ = the i-th smallest p-value, $q$ = target false-discovery rate.
# *Example:* p = (0.001, 0.008, 0.012, 0.041, 0.30), α = q = 0.05: Bonferroni's threshold 0.01 rejects 2; BH's thresholds (0.01, 0.02, 0.03, 0.04, 0.05) reject 3. And $1 - 0.95^{17} = 0.58$ for 17 independent null metrics.

# %%
# --- Check the key equations numerically -------------------------------------
xa, na, xb, nb_ = 1_000, 10_000, 1_100, 10_000
p_pool = (xa + xb) / (na + nb_)
se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / na + 1 / nb_))
z_hand = (xb / nb_ - xa / na) / se_pool
print(f"1. z = {z_hand:.3f}, p = {2 * stats.norm.sf(abs(z_hand)):.4f} by hand | "
      f"ex.two_proportion_ztest p = {ex.two_proportion_ztest(xa, na, xb, nb_)['p']:.4f}")
se_diff = np.sqrt(0.1 * 0.9 / na + 0.11 * 0.89 / nb_)
print(f"2. SE of the difference {se_diff:.5f}, 95% CI [{0.01 - 1.96 * se_diff:+.4f}, {0.01 + 1.96 * se_diff:+.4f}]")
zsum = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
print(f"3. n per arm: simplified {np.ceil(2 * zsum**2 * 0.1 * 0.9 / 0.01**2):,.0f} | "
      f"two-variance form {np.ceil(zsum**2 * (0.1 * 0.9 + 0.11 * 0.89) / 0.01**2):,.0f} | n_formula {n_formula(0.10, 0.01):,}")
counts = np.array([50_600, 49_400])
chi2 = ((counts - counts.sum() / 2) ** 2 / (counts.sum() / 2)).sum()
print(f"4. SRM chi-square {chi2:.1f}, p = {stats.chi2.sf(chi2, df=1):.1e} | scipy.stats.chisquare p = {stats.chisquare(counts).pvalue:.1e}")
p_demo = np.array([0.001, 0.008, 0.012, 0.041, 0.30])
print("5. Bonferroni rejects", (p_demo <= 0.05 / len(p_demo)).astype(int),
      "| BH rejects", ex.benjamini_hochberg(p_demo, 0.05).astype(int),
      f"| 1 - 0.95^17 = {1 - 0.95**17:.2f}")

# %% [markdown]
# **Expected:** 1. z = 2.307 and p = 0.0211 both ways; 2. SE 0.00433 and CI [+0.0015, +0.0185];
# 3. 14,128 (simplified) and 14,749 twice; 4. chi-square 14.4 with p = 1.5e-04 from both;
# 5. Bonferroni `[1 1 0 0 0]`, BH `[1 1 1 0 0]`, and 0.58.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `shoprank` | setup cell not run | run the first code cell |
# | A/A tests "significant" far more often than 5% | variance computed at the wrong unit (impressions instead of users), or peeking | test at the randomisation unit; one pre-planned look |
# | Measured power far below the plan | σ estimated on a different population, or the effect diluted by never-exposed users | estimate σ on comparable pre-period users; analyse exposed (triggered) users |
# | SRM alert | logging, bot filtering, redirects or crashes differ by arm | stop, find the cause, restart; never re-weight |
# | `RuntimeWarning: divide by zero` in per-user CTR | users with zero impressions | filter `impressions > 0` (exposure) before dividing |
# | Bootstrap runs out of memory | resampling matrix of n_boot × n users | resample in chunks, as `ex.bootstrap_diff_ci` does |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. You randomise by user but compute CTR's confidence interval treating impressions as independent. What goes wrong, in which direction, and how would you show it?
# 2. When would you randomise by request, by user, or by cluster (e.g. city, seller)? Name one interference problem for each.
# 3. Your baseline conversion is 5% and you want to detect +0.25 percentage points. Roughly how many users per arm do you need, and what happens if the PM wants to detect half of that?
# 4. The experiment shows a significant lift, and the arms have 50.4% / 49.6% of 2 million users. What do you do?
# 5. A stakeholder wants to check results daily and stop when significant. Explain the risk with a number, and offer two alternatives.
# 6. A launch review shows 3 of 25 metrics significantly up. How do you decide whether the launch is a win?
#
# **Next chapter:** [`12_advanced_experimentation.ipynb`](../ch30-advanced-experimentation/notebook.ipynb) — make experiments cheaper (CUPED, interleaving), analyse ratio metrics correctly, and decide what to launch.
