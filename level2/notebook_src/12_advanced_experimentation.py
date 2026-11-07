# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/level2/notebooks/12_advanced_experimentation.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # 12 · Advanced experimentation
#
# **Day 6 · ~90 minutes · CPU only**
#
# Notebook 11 made experiments *valid*. This one makes them *cheap and
# decisive*: reuse pre-period data to cut variance (CUPED), analyse ratio
# metrics such as CTR with the right variance (delta method), compare rankers with a
# fraction of the traffic (interleaving), and face the uncomfortable fact that
# offline metrics can rank models in the wrong order. It ends where real
# experiments end: a launch decision with written criteria.
#
# By the end you will be able to:
#
# 1. apply CUPED and predict its variance reduction from one correlation;
# 2. compute the variance of a ratio metric with the delta method and check it against the bootstrap;
# 3. explain why slicing results by segment produces "effects" that are not there;
# 4. implement team-draft interleaving and measure how many fewer users it needs than an A/B test;
# 5. show an offline metric disagreeing with online behaviour, and explain why;
# 6. write launch criteria with a primary metric, guardrails and decision rules.
#
# **You will produce:** `artifacts/level2/nb12/launch_criteria.md`, a launch-criteria
# document filled with this notebook's measured numbers, and `artifacts/level2/nb12/power_curves.csv`.

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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "level2/requirements.txt"], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "churnlab").is_dir())
os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from churnlab.colab import artifact_dir

ART = artifact_dir(use_drive=USE_DRIVE)
print("repo:", ROOT)
print("artifacts:", ART)

# %%
# --- Catch up: ShopStream world + logs (logs rebuilt in ~30 s on a fresh runtime)
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from shoprank import experiments as ex
from shoprank import store
from shoprank.data import K
from shoprank.grading import nb12 as grade

OUT = ART / "level2" / "nb12"
OUT.mkdir(parents=True, exist_ok=True)

world, users, items, imps = store.load("small")
pop = ex.user_population(world)
print(f"{len(imps):,} logged impressions | population of {len(pop):,} users, mean base CTR {pop['base_ctr'].mean():.3f}")

# %% [markdown]
# **Expected:** about 2.45 million impressions and a mean base CTR of about 0.139.
# The simulator is notebook 11's `ex.simulate_users`.

# %% [markdown]
# ## 📖 Words to know
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Covariate | a quantity measured before assignment that predicts the outcome | each user's clicks in the two-week pre-period |
# | CUPED | subtract θ × (covariate − its mean) from the metric to remove the predictable part of its variance | variance factor about 0.5 for clicks per user |
# | Correlation ρ | how tightly two quantities move together, from −1 to 1 | ρ ≈ 0.71 between pre-period and experiment clicks |
# | Ratio metric | a sum divided by a sum, where both vary per user | pooled CTR = Σ clicks ÷ Σ impressions |
# | Delta method | a straight-line (first-order Taylor) approximation that gives the variance of a function of means | SE 0.00168 vs a true spread of 0.00169 |
# | Bootstrap | resample units with replacement and recompute, to see how much the estimate moves | 500 resamples of users validate the delta method |
# | Heterogeneous effect | a treatment effect that differs between segments | 15 country × device slices whose true lift is identical |
# | Interleaving | merge two rankers' lists into one and credit each click to the ranker that contributed the item | about 20× fewer users than an A/B test for 80% power |
# | Offline–online gap | offline and online evaluations disagreeing about which model is better | logging policy first offline, third online |
# | Guardrail metric | a metric that can block a launch but never justify one | p95 latency, crash rate, purchases per user |
# | Launch criteria | the decision rule, written before the experiment starts | `launch_criteria.md` in 12.6 |

# %% [markdown]
# ## 12.1 · CUPED: subtract what you already knew
#
# Much of the variance in "clicks per user during the experiment" is
# *predictable* from the same user's clicks **before** the experiment: heavy
# users stay heavy. CUPED (Deng et al., 2013) uses a pre-period covariate `x`:
#
# $$ y^{cv}_i = y_i - \theta\,(x_i - \bar x), \qquad \theta = \frac{\mathrm{cov}(x, y)}{\mathrm{var}(x)} $$
#
# Because `x` is measured **before** randomisation, it is independent of the
# arm, so the adjusted metric has the same expected difference between arms
# (no bias) and its variance is multiplied by `1 - ρ²`, where ρ = corr(x, y).
# It is linear regression adjustment, and the same reasoning gives the same estimator.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/l2-x-cuped.svg" alt="Scatter of experiment clicks against pre-period clicks with a regression line, and the narrower distribution of the CUPED-adjusted metric" width="100%">
#
# *What to notice:* CUPED keeps only the vertical distance from each point to the line; the adjusted metric has the same centre and a narrower spread.
#
# > 💡 **Intuition.** Judging a diet by weight after two weeks is hopeless when people
# > differ by 30 kg; judging it by each person's *change* from their starting weight
# > removes that spread. CUPED is the refined version: it subtracts θ × (starting value),
# > with θ chosen to minimise the variance (θ = 1 is the plain change score).
# > *Where it breaks down:* for users with no history, or a metric the past does not
# > predict (ρ ≈ 0), there is nothing to subtract and CUPED gains nothing.

# %% [markdown]
# > **Exercise 1.** Write `cuped_adjust(y, x)`: return the CUPED-adjusted array
# > `y - θ (x - mean(x))` with `θ = cov(x, y) / var(x)` estimated from the given data
# > (use the same `ddof` for both). Return a NumPy array of the same length. If `x` is constant, return `y` unchanged.

# %%
def cuped_adjust(y, x):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(cuped_adjust)

# %% [markdown]
# Now measure it: 300 simulated experiments of 20,000 users with a true 2%
# lift; metric = clicks per exposed user; covariate = the same user's clicks in the two-week pre-period.
# θ is estimated on both arms pooled, as in practice.

# %%
def cuped(y, x):
    """Reference implementation used below (so the notebook runs before Exercise 1 is done)."""
    theta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean())


LIFT = 0.02
rng = np.random.default_rng(0)
rows, start = [], time.time()
for _ in range(300):
    d = ex.simulate_users(pop, 20_000, rng, lift=LIFT)
    d = d[d["impressions"] > 0]
    y, x, arm = d["clicks"].to_numpy(float), d["pre_clicks"].to_numpy(float), d["arm"].to_numpy() == 1
    y_cv = cuped(y, x)
    rows.append({"raw": y[arm].mean() - y[~arm].mean(), "cuped": y_cv[arm].mean() - y_cv[~arm].mean(),
                 "rho": np.corrcoef(x, y)[0, 1], "control_mean": y[~arm].mean()})
cv = pd.DataFrame(rows)
true_effect = LIFT * cv["control_mean"].mean()  # clicks scale with CTR, so +2% CTR = +2% clicks
rho = cv["rho"].mean()
print(f"300 experiments in {time.time() - start:.0f} s | corr(pre_clicks, clicks) = {rho:.3f}")
print(f"true effect ≈ {true_effect:.4f} clicks/user | mean estimate raw {cv['raw'].mean():.4f}, CUPED {cv['cuped'].mean():.4f}")
print(f"SD of the estimate: raw {cv['raw'].std():.4f} | CUPED {cv['cuped'].std():.4f}")
print(f"variance ratio measured {cv['cuped'].var() / cv['raw'].var():.3f} vs predicted 1 - rho^2 = {1 - rho**2:.3f}")
print(f"power at this size: raw {(cv['raw'] / cv['raw'].std() > 1.96).mean():.2f} | "
      f"CUPED {(cv['cuped'] / cv['cuped'].std() > 1.96).mean():.2f}")

# %% [markdown]
# **Expected:** a correlation of about 0.71; both estimators centred on the true
# effect (about 0.21 clicks per user, on a mean of about 10.4) within simulation
# noise (±0.02 for the mean of 300 estimates); the CUPED estimate's variance about
# half the raw one (measured 0.50 vs predicted `1 - ρ²` = 0.49); and power
# rising from about 0.2 to about 0.35 at the same sample size. Halving the
# variance halves the users you need, or the duration of the test.
#
# Two limits worth saying in an interview:
#
# * CUPED helps as much as the covariate predicts the metric. For **per-user CTR**
#   (a noisier, bounded metric) the pre/post correlation in this simulator is only
#   about 0.35, so the reduction is about 12%. New users have no pre-period at all
#   (use 0 plus an indicator, or other covariates known before assignment).
# * The covariate must be unaffected by treatment. Pre-period data is safe by
#   construction; "clicks during the experiment on another surface" is not.
#
# ## 12.2 · Ratio metrics and the delta method
#
# Pooled CTR per arm is `R = Σ clicks / Σ impressions = mean(c) / mean(n)`, a
# ratio of two per-user means. Its unit of randomisation is the user, but it is not a
# mean of per-user values, so the t-test does not apply directly, and
# notebook 11 showed that the impression-level binomial SE is far too small.
# The **delta method** linearises the ratio around the means:
#
# $$ \mathrm{Var}(\hat R) \approx \frac{1}{N}\left( \frac{\sigma_c^2}{\mu_n^2} - \frac{2\mu_c\,\sigma_{cn}}{\mu_n^3} + \frac{\mu_c^2\,\sigma_n^2}{\mu_n^4} \right) $$
#
# with means, variances and covariance of per-user clicks `c` and impressions `n`,
# and N users in the arm.
#
# > 💡 **Intuition.** For small wobbles, a ratio of two averages behaves like a straight
# > line in (mean clicks, mean impressions): a 1% rise in clicks raises CTR by about 1%,
# > a 1% rise in impressions lowers it by about 1%. The variance of a straight-line
# > combination is a textbook formula, and that is the delta method. The covariance term
# > matters: users with more impressions also have more clicks, and the two partly cancel.
# > *Where it breaks down:* with few users or very heavy tails the wobbles are not small,
# > and the straight line is a poor fit; check with a bootstrap.

# %% [markdown]
# > **Exercise 2.** Write `delta_method_ratio_var(num, den)`: the delta-method variance of
# > `mean(num) / mean(den)` for per-unit arrays `num` and `den` (use `ddof=1` for the variances
# > and the covariance). The grader compares your value with the formula and with a bootstrap.

# %%
def delta_method_ratio_var(num, den):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(delta_method_ratio_var)

# %%
def delta_var(c, n):
    """Reference implementation used below."""
    mc, mn = c.mean(), n.mean()
    cov = np.cov(c, n)
    return (cov[0, 0] / mn**2 - 2 * mc * cov[0, 1] / mn**3 + mc**2 * cov[1, 1] / mn**4) / len(c)


rng = np.random.default_rng(1)
rows = []
for _ in range(400):  # A/A experiments
    d = ex.simulate_users(pop, 20_000, rng, lift=0.0)
    d = d[d["impressions"] > 0]
    g = [d[d["arm"] == k] for k in (0, 1)]
    c = [x["clicks"].to_numpy(float) for x in g]
    n = [x["impressions"].to_numpy(float) for x in g]
    r = [ci.sum() / ni.sum() for ci, ni in zip(c, n)]
    se_delta = np.sqrt(delta_var(c[0], n[0]) + delta_var(c[1], n[1]))
    pooled = (c[0].sum() + c[1].sum()) / (n[0].sum() + n[1].sum())
    se_naive = np.sqrt(pooled * (1 - pooled) * (1 / n[0].sum() + 1 / n[1].sum()))
    rows.append({"diff": r[1] - r[0], "se_delta": se_delta, "se_naive": se_naive})
ratio = pd.DataFrame(rows)

# one bootstrap on the last experiment (resampling USERS within each arm)
idx = [rng.integers(0, len(ci), (500, len(ci))) for ci in c]
boot = (c[1][idx[1]].sum(1) / n[1][idx[1]].sum(1)) - (c[0][idx[0]].sum(1) / n[0][idx[0]].sum(1))
print(f"true SD of the pooled-CTR difference over 400 A/A experiments: {ratio['diff'].std():.5f}")
print(f"  delta-method SE (mean)       : {ratio['se_delta'].mean():.5f}")
print(f"  bootstrap SE (one experiment): {boot.std():.5f}")
print(f"  naive binomial SE (mean)     : {ratio['se_naive'].mean():.5f}")
z = ratio["diff"] / ratio["se_delta"]
print(f"A/A false-positive rate: delta method {np.mean(np.abs(z) > 1.96):.3f} | "
      f"naive {np.mean(np.abs(ratio['diff'] / ratio['se_naive']) > 1.96):.3f}")

# %% [markdown]
# **Expected:** the delta-method SE (0.00168) and the bootstrap SE (0.00170) both match the
# true SD (0.00169) within a few percent; the naive SE (0.00061) is about 2.8x too small; the
# delta-method z-test fires about 4–5% of the time in A/A tests and the naive one about 50%.
# The delta method costs one pass over the data; the bootstrap costs hundreds,
# which matters when an experimentation platform computes thousands of
# metric × experiment × day cells every night. That is why platforms use the
# delta method for ratio metrics (CTR, revenue per session, clicks per query),
# and the bootstrap to *validate* it.
#
# ## 12.3 · Heterogeneous effects: a caution
#
# After every experiment someone asks "did it work better for iOS users in
# Germany?" Slicing is legitimate for **pre-registered** segments with a
# reason. Here is why slicing *after the fact* is dangerous. One large
# experiment, **true lift exactly +2% for everyone**, sliced by country × device (15 segments):

# %%
rng = np.random.default_rng(2)
d = ex.simulate_users(pop, 300_000, rng, lift=LIFT)
d = d[d["impressions"] > 0].assign(ctr=lambda t: t["clicks"] / t["impressions"])
rows = []
for (country, device), g in d.groupby(["country", "device"]):
    a, b = g.loc[g["arm"] == 0, "ctr"], g.loc[g["arm"] == 1, "ctr"]
    t = ex.welch_ttest(a, b)
    rows.append({"segment": f"{country}/{device}", "users": len(g), "lift": t["diff"] / a.mean(), "p": t["p"]})
seg = pd.DataFrame(rows).sort_values("lift")
overall = ex.welch_ttest(d.loc[d["arm"] == 0, "ctr"], d.loc[d["arm"] == 1, "ctr"])
print(f"overall: lift {overall['diff'] / d.loc[d['arm'] == 0, 'ctr'].mean():+.2%}, p = {overall['p']:.1e}")
print(f"segment lifts range from {seg['lift'].min():+.1%} to {seg['lift'].max():+.1%}; "
      f"{(seg['p'] < 0.05).sum()} of 15 significant at 0.05; {(seg['lift'] < 0).sum()} negative")
print(seg.round(4).to_string(index=False))

# %% [markdown]
# **Expected:** the overall lift is close to +2% and highly significant. The segment
# estimates spread widely around it (in our run from +0.1% to +5.4%, a 50-fold range), about
# half the segments are "significant" and half are not, and the extremes tend to be small
# segments (BR/web, the smallest, has the largest lift). A team reading this table would
# "discover" that the ranker works on iOS in Great Britain and India but not on US web. Every
# difference between segments here is noise: the simulator applies the same 2% to everyone.
#
# What to do instead: pre-register a few segments with a hypothesis; correct for
# multiple comparisons (notebook 11); expect small segments to be extreme
# (shrink them towards the overall effect); and treat a surprising segment
# result as a hypothesis for a **new** experiment, not as a finding. Real heterogeneity exists
# (`ex.simulate_users(..., lift_by={"device": {"ios": 0.05}})` simulates it): try
# it, and see how many users you need before the iOS segment is reliably different.
#
# ## 12.4 · Interleaving: let each user judge both rankers
#
# An A/B test compares *different users* under two rankers, so it must
# average away how much users differ from each other. **Interleaving** shows
# each user a single list that merges both rankers' results and credits each
# click to the ranker that contributed the item. The comparison is *within*
# a user and a request, and between-user variance cancels.
#
# **Team-draft interleaving** (Radlinski, Kurup & Joachims, 2008): like picking
# teams in a playground. In each round, a coin decides which ranker picks
# first; each ranker then adds its highest-ranked item not already in the
# list, and that item is credited to its team.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/l2-x-team-draft.svg" alt="Two rankings merged by team draft into one shown list, with clicks credited to the team that picked each item" width="100%">
#
# *What to notice:* each shown item belongs to exactly one team, and the coin flip per round keeps either ranker from always owning the top slot.
#
# > 💡 **Intuition.** An A/B test is two groups each tasting one cola; interleaving is
# > every person sipping both, side by side. Differences between tasters (some love any
# > sugar) cancel within the person. *Where it breaks down:* a sip-test cannot tell you
# > whether people will buy more cola over a month; interleaving cannot measure session-
# > or user-level outcomes such as revenue or retention.

# %% [markdown]
# > **Exercise 3.** Write `team_draft_interleave(list_a, list_b, rng)` for two rankings of equal length
# > `k`. Return `(items, teams)`: the interleaved list of length `k` (no duplicates) and, for
# > each position, `"A"` or `"B"`, the team that picked it. Team-draft rule: the team with fewer
# > picks so far picks next; on a tie, `rng.random() < 0.5` means A picks first. A team always
# > picks its highest-ranked item not already in `items`.

# %%
def team_draft_interleave(list_a, list_b, rng):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex3(team_draft_interleave)

# %% [markdown]
# The package version is `ex.team_draft_interleave`. Two example interleavings of the same
# pair of rankings:

# %%
demo_rng = np.random.default_rng(0)
a_list, b_list = ["a1", "s1", "a2", "s2", "a3"], ["s1", "b1", "s2", "b2", "a1"]
for _ in range(2):
    print(ex.team_draft_interleave(a_list, b_list, demo_rng))

# %% [markdown]
# Now the experiment. Two rankers score each request's 100 candidates by the
# **true** relevance plus noise: ranker A with noise SD 2.0, ranker B with 2.2, so
# A is genuinely better. Clicks come from the world's true click model
# (examination by position × true click probability). For 2,000 users ×
# 6 requests we precompute both rankers' lists, each list's click probabilities, and
# two interleavings per request; each simulated experiment then draws users
# (with their real activity levels) and their clicks.

# %%
rng = np.random.default_rng(3)
U, R, M, N_CAND = 2000, 6, 2, 100
pool_users = rng.integers(0, len(users), U)
activity = users["activity"].to_numpy()[pool_users]
cand = rng.integers(0, len(items), (U, R, N_CAND))
true_logit = world.true_logit(pool_users[:, None, None], cand)
true_p = 1 / (1 + np.exp(-true_logit))
exam = world.examination_prob(np.arange(K))


def top_k(noise_sd):
    return np.argsort(-(true_logit + rng.normal(0, noise_sd, true_logit.shape)), axis=2)[..., :K]


list_A, list_B = top_k(2.0), top_k(2.2)
pA = np.take_along_axis(true_p, list_A, axis=2) * exam      # P(click) per position, ranker A
pB = np.take_along_axis(true_p, list_B, axis=2) * exam
pI = np.empty((U, R, M, K))
teamA = np.empty((U, R, M, K), dtype=bool)
for u in range(U):
    for r in range(R):
        for m in range(M):
            merged, teams = ex.team_draft_interleave(list(list_A[u, r]), list(list_B[u, r]), rng)
            pI[u, r, m] = true_p[u, r, merged] * exam
            teamA[u, r, m] = np.array(teams) == "A"
ctr_a, ctr_b = pA.sum(-1).mean() / K, pB.sum(-1).mean() / K
print(f"true CTR: ranker A {ctr_a:.4f} | ranker B {ctr_b:.4f} | A better by {ctr_a / ctr_b - 1:.1%}")


def draw_users(n):
    """n exposed users: each makes Poisson(activity x 0.35 x 14) >= 1 requests."""
    u = rng.integers(0, U, n)
    n_req = np.maximum(rng.poisson(activity[u] * 0.35 * 14), 1)
    owner = np.repeat(np.arange(n), n_req)
    return np.repeat(u, n_req), rng.integers(0, R, len(owner)), owner


def ab_test(n):
    uu, rr, owner = draw_users(n)
    arm_a = rng.random(n) < 0.5
    p = np.where(arm_a[owner][:, None], pA[uu, rr], pB[uu, rr])
    clicks = np.bincount(owner, (rng.random(p.shape) < p).sum(1), minlength=n)
    res = stats.ttest_ind(clicks[arm_a], clicks[~arm_a], equal_var=False)
    return res.pvalue < 0.05 and clicks[arm_a].mean() > clicks[~arm_a].mean()


def interleaving_test(n):
    uu, rr, owner = draw_users(n)
    mm = rng.integers(0, M, len(owner))
    clicked = rng.random((len(owner), K)) < pI[uu, rr, mm]
    ta = teamA[uu, rr, mm]
    credit = np.bincount(owner, (clicked & ta).sum(1) - (clicked & ~ta).sum(1), minlength=n)  # A minus B per user
    res = stats.ttest_1samp(credit, 0)
    return res.pvalue < 0.05 and credit.mean() > 0


start, curves = time.time(), []
for n in (2000, 4000, 8000, 16000, 32000):
    curves.append({"design": "A/B test", "users": n, "power": np.mean([ab_test(n) for _ in range(100)])})
for n in (125, 250, 500, 1000, 2000):
    curves.append({"design": "interleaving", "users": n, "power": np.mean([interleaving_test(n) for _ in range(100)])})
curves = pd.DataFrame(curves)
curves.to_csv(OUT / "power_curves.csv", index=False)
print(f"({time.time() - start:.0f} s)")
print(curves.pivot(index="users", columns="design", values="power").to_string())

needed = {d: float(np.interp(0.8, g["power"].cummax(), g["users"])) for d, g in curves.groupby("design")}
print(f"users for 80% power (interpolated): A/B ≈ {needed['A/B test']:,.0f} | "
      f"interleaving ≈ {needed['interleaving']:,.0f} | ratio ≈ {needed['A/B test'] / needed['interleaving']:.0f}x")
fig, ax = plt.subplots(figsize=(5, 3))
for design, g in curves.groupby("design"):
    ax.plot(g["users"], g["power"], marker="o", label=design)
ax.axhline(0.8, color="gray", ls="--")
ax.set_xscale("log")
ax.set_xlabel("users in the experiment")
ax.set_ylabel("power (100 simulations each)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Expected:** ranker A's true CTR is about 4–5% higher than B's (0.156 vs 0.149). The A/B
# test needs roughly 15,000–20,000 users for 80% power; interleaving needs
# roughly 700–1,000, more than an order of magnitude fewer (about 20x in our runs).
# Each power value is from 100 simulations (±0.05), so read the ratio, not the third digit.
#
# Why: the A/B test's per-user clicks vary enormously between users (activity
# alone varies 100-fold), while the interleaving credit difference compares two
# rankers *for the same user on the same request*. That is why large search
# and recommendation teams use interleaving to **screen** many ranker candidates
# quickly, then confirm the winner with an A/B test. Interleaving says
# which ranker users prefer; it cannot measure effects on whole-session metrics
# (revenue, retention, latency), and it needs a clean credit-assignment rule.
#
# ## 12.5 · The offline–online gap
#
# Before any of this, a ranker is chosen offline: NDCG on logged clicks. Here
# that choice goes wrong. Take logged **validation-week** requests (logging-policy traffic),
# and re-rank each request's 10 shown items with four rankers. Offline: NDCG@10 with the logged
# clicks as labels. Online: serve that order to the same users and compute the **expected
# CTR** under the true click model, which we can do because ShopStream keeps the truth.

# %%
val = imps[imps["day"].between(42, 48) & ~imps["is_random"]]
n_req = len(val) // K
sample = np.random.default_rng(4).choice(n_req, 30_000, replace=False)
grid = lambda col: val[col].to_numpy().reshape(n_req, K)[sample]  # noqa: E731 - (requests, 10)
u, it, clicked, pscore = grid("user_id"), grid("item_id"), grid("clicked").astype(float), grid("policy_score")
truth = world.true_click_prob(u, it)
train_clicks = imps.loc[imps["day"] < 42].groupby("item_id")["clicked"].sum()
popularity = train_clicks.reindex(np.arange(len(items)), fill_value=0).to_numpy()[it].astype(float)

rankers = {
    "logging policy": pscore,
    "true relevance": truth,
    "item popularity (train clicks)": popularity,
    "random": np.random.default_rng(5).random(truth.shape),
}
has_click = clicked.sum(1) > 0
rows = []
for name, score in rankers.items():
    order = np.argsort(-score, axis=1, kind="stable")
    online_ctr = (np.take_along_axis(truth, order, axis=1) * exam).sum(1).mean() / K
    offline = np.nanmean(ex.ndcg_rows(score[has_click], clicked[has_click]))
    rows.append({"ranker": name, "offline NDCG@10 (logged clicks)": offline, "online CTR (true click model)": online_ctr})
gap = pd.DataFrame(rows).set_index("ranker")
print(f"{has_click.sum():,} of 30,000 sampled requests have a click (used for NDCG)")
print(gap.round(4).to_string())

# %% [markdown]
# **Expected:** offline, the **logging policy** scores highest (about 0.76) and true
# relevance clearly lower (about 0.68). Online, the order flips: true relevance
# has by far the highest CTR (about 0.18) and the logging policy is below it (about 0.14),
# below even a plain popularity ranker (about 0.15), which offline ranks third. Offline evaluation on
# logged clicks rewards a ranker for *agreeing with the logging policy*: clicks
# happened mostly at the top positions the policy chose (position bias, notebook 05),
# so reproducing the policy's order "recovers" them.
#
# This is the offline–online gap in one table. Defences, in order of cost:
# debiased offline metrics (inverse propensity weighting with exploration traffic,
# notebook 05); offline metrics validated against past A/B results ("does offline
# NDCG predict the online winner?"); interleaving as a cheap online filter; and
# finally the A/B test, which is the only thing that measures the decision you are making.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/docs/assets/diagrams/l2-x-offline-online-gap.svg" alt="Slope chart: offline NDCG on logged clicks ranks the logging policy first; online CTR ranks true relevance first and the logging policy third" width="100%">
#
# *What to notice:* the red line falls from first to third: logged clicks measured agreement with the old policy, not quality.
#
# ## 12.6 · Guardrails and launch criteria
#
# A launch decision should be written **before** the experiment starts, so the
# result cannot shape the rule. The file below fills a template with numbers measured
# in notebooks 11–12; read it, then write your own memo in Exercise 4.

# %%
cuped_factor = cv["cuped"].var() / cv["raw"].var()
criteria = f"""# Launch criteria: ranker A vs production (template, filled by notebook 12)

## Decision metric (one, pre-registered)
* Clicks per exposed user over 14 days, user-randomised 50/50, analysed with CUPED
  (pre-period clicks; measured variance factor {cuped_factor:.2f}, i.e. about
  {1 / cuped_factor:.1f}x fewer users than an unadjusted test).
* Minimum detectable effect: +2% relative, alpha = 0.05 two-sided, power 0.8.
  Sample size from pre-period variance; the test runs the full 14 days (two weekly cycles), no early stopping.

## Validity checks (any failure = result void, investigate, rerun)
* Sample-ratio mismatch: chi-square p < 0.001 on assigned and on exposed users.
* A/A on the same pipeline in the previous week: false-positive rate consistent with 5%.
* Ratio metrics (CTR) analysed with the delta method at the user level, never impression-level SEs.

## Guardrails (can block a launch, cannot justify one)
| Guardrail | Rule |
|---|---|
| p95 feed latency | no increase > 5% (one-sided test) |
| crash / error rate | no significant increase |
| purchases per user | non-inferior: lower 95% bound > -1% |
| catalogue coverage | no drop > 10% (popularity collapse) |

## Decision rules
* Ship: decision metric significantly positive AND no guardrail violated AND checks pass.
* Do not ship: decision metric flat or negative, or any guardrail violated.
* Segment results and the other metrics: reported, labelled exploratory, BH-corrected; never the reason to ship.

## Evidence so far
* Interleaving screen: ranker A preferred; 80% power needed ~{needed['interleaving']:,.0f} users vs
  ~{needed['A/B test']:,.0f} for an A/B test (simulated).
* Offline NDCG on logged clicks is NOT used for the decision: it ranked the logging policy above true relevance.
"""
(OUT / "launch_criteria.md").write_text(criteria)
print(criteria)

# %% [markdown]
# **Expected:** the document prints with a CUPED variance factor of about 0.5 and the
# interleaving and A/B user counts from 12.4.

# %% [markdown]
# > **Exercise 4 (written).** Write a one-page **launch memo** for this scenario: the A/B test of
# > ranker A ran for 14 days; clicks per user +1.8% (CUPED, 95% CI +0.6% to +3.0%); purchases per
# > user -0.4% (95% CI -1.3% to +0.5%); p95 latency +7%; SRM p = 0.21; two of eleven secondary
# > metrics significant, one of them "iOS users in Germany +9%". State the decision under the criteria
# > above, what you would do next, and what you would tell the team that built the ranker.
# >
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer here.*

# %% [markdown]
# ## 🧮 Key equations
#
# **1. CUPED adjustment and its variance**
#
# $$
# y^{cv}_i = y_i - \theta\,(x_i - \bar x), \qquad \theta = \frac{\operatorname{cov}(x, y)}{\operatorname{var}(x)}, \qquad \operatorname{Var}(y^{cv}) = \operatorname{Var}(y)\,(1 - \rho^2)
# $$
#
# *In words:* regress the metric on its pre-period value and keep the residual (plus the mean). Because $x$ is fixed before randomisation, the difference between arms is unchanged; only the noise shrinks, by the factor $1 - \rho^2$, and so does the sample size you need.
# *Symbols:* $y$ = metric during the experiment, $x$ = the same user's pre-period value, $\rho = \operatorname{corr}(x, y)$.
# *Example:* $x = (1, 2, 3, 4)$, $y = (2, 3, 5, 6)$: θ = 2.33 / 1.67 = 1.4, $y^{cv} = (4.1, 3.7, 4.3, 3.9)$, variance 3.33 → 0.067 $= 3.33 \cdot (1 - 0.98)$. In 12.1, ρ = 0.71 gives a factor of about 0.5: half the users.
#
# **2. Delta method for a ratio of means**
#
# $$
# \operatorname{Var}\!\left(\frac{\bar c}{\bar n}\right) \approx \frac{1}{N}\left( \frac{\sigma_c^2}{\mu_n^2} - \frac{2\,\mu_c\,\sigma_{cn}}{\mu_n^3} + \frac{\mu_c^2\,\sigma_n^2}{\mu_n^4} \right)
# $$
#
# *In words:* linearise $R = \mu_c / \mu_n$ around the means; the variance of the linear approximation is $\nabla R^\top \Sigma\, \nabla R / N$ with gradient $\nabla R = (1/\mu_n,\ -\mu_c/\mu_n^2)$.
# *Symbols:* per-user clicks $c$ and impressions $n$ with means $\mu_c, \mu_n$, variances $\sigma_c^2, \sigma_n^2$, covariance $\sigma_{cn}$; $N$ users in the arm.
# *Example:* $\mu_c = 1.4$, $\mu_n = 10$, $\sigma_c^2 = 4$, $\sigma_n^2 = 36$, $\sigma_{cn} = 9$, $N = 10{,}000$: $(0.04 - 0.0252 + 0.0071) / 10^4 = 2.19 \times 10^{-6}$, so SE ≈ 0.0015.
#
# **3. The interleaving statistic**
#
# $$
# \Delta = \frac{1}{U} \sum_{u=1}^{U} \left( \text{clicks}_{A,u} - \text{clicks}_{B,u} \right), \qquad t = \frac{\Delta}{s_\Delta / \sqrt{U}}
# $$
#
# *In words:* per user, credited clicks for A minus credited clicks for B; a one-sample t-test asks whether the mean difference is zero.
# *Symbols:* $U$ users, $s_\Delta$ = the standard deviation of the per-user differences.
# *Example:* 400 users with mean difference 0.05 and SD 0.8: t = 0.05 / (0.8 / 20) = 1.25, not significant; 1,600 users with the same numbers give t = 2.5.

# %%
# --- Check the key equations numerically -------------------------------------
x4, y4 = np.array([1.0, 2, 3, 4]), np.array([2.0, 3, 5, 6])
theta4 = np.cov(x4, y4)[0, 1] / np.var(x4, ddof=1)
rho4 = np.corrcoef(x4, y4)[0, 1]
print(f"1. theta = {theta4:.2f} | adjusted {np.round(cuped(y4, x4), 2)} | variance {np.var(y4, ddof=1):.3f} -> "
      f"{np.var(cuped(y4, x4), ddof=1):.4f}; Var(y)(1 - rho^2) = {np.var(y4, ddof=1) * (1 - rho4**2):.4f}")

mc, mn, vc, vn, cn, N = 1.4, 10.0, 4.0, 36.0, 9.0, 10_000
print(f"2. worked example: SE = {np.sqrt((vc / mn**2 - 2 * mc * cn / mn**3 + mc**2 * vn / mn**4) / N):.5f}")
rng = np.random.default_rng(12)                       # 5,000 synthetic users
imp_u = (rng.poisson(10, 5_000) + 1).astype(float)
clk_u = rng.binomial(imp_u.astype(int), rng.beta(2, 12, 5_000)).astype(float)
idx = rng.integers(0, 5_000, (1_000, 5_000))          # resample USERS
boot_r = clk_u[idx].sum(1) / imp_u[idx].sum(1)
print(f"   synthetic users: delta-method SE {np.sqrt(delta_var(clk_u, imp_u)):.5f} | bootstrap SE {boot_r.std():.5f}")
print(f"3. interleaving t: {0.05 / (0.8 / np.sqrt(400)):.2f} with 400 users, {0.05 / (0.8 / np.sqrt(1600)):.2f} with 1,600")

# %% [markdown]
# **Expected:** 1. θ = 1.40, adjusted `[4.1 3.7 4.3 3.9]`, variance 3.333 → 0.0667 both ways;
# 2. SE = 0.00148 for the worked example, and for the synthetic users a delta-method SE of
# about 0.0019 matching the bootstrap within a few percent; 3. t = 1.25 and 2.50.

# %% [markdown]
# ## 🛟 Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `shoprank` | setup cell not run | run the first code cell |
# | CUPED estimate biased | covariate affected by treatment (measured after assignment) | use only data from before each user's first exposure |
# | CUPED gives almost no reduction | weak covariate (low ρ), e.g. for per-user CTR or new users | use a better-predicting covariate (same metric, pre-period), or several (regression adjustment) |
# | Delta-method SE far from bootstrap SE | per-unit arrays at the wrong level (impressions instead of users), or a heavy tail with few units | aggregate to the randomisation unit; check with a bootstrap |
# | Interleaving always returns ties | identical lists, or credit counted on shared items for both teams | each item belongs to exactly one team in team-draft |
# | `ValueError` in `np.interp` | a power curve never reaches 0.8 | extend the user grid |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Explain CUPED to a product manager in two sentences, then say exactly what variance reduction to expect and when it fails.
# 2. Why can't you put pooled CTR's per-impression clicks into a t-test? What does the delta method do instead, and how would you validate it?
# 3. An experiment shows +2% overall and +11% for one of 30 segments. What do you tell the team that wants to launch only for that segment?
# 4. How does team-draft interleaving attribute clicks, why does it need far fewer users than an A/B test, and what can it not measure?
# 5. Your new ranker wins offline NDCG on logged clicks but loses the A/B test. Give two mechanisms, and how you would make the offline metric more trustworthy.
# 6. What belongs in launch criteria, and why must they be written before the experiment starts?
#
# ➡️ **Next:** [`13_ml_from_scratch_1.ipynb`](13_ml_from_scratch_1.ipynb) — Day 7 starts the ML coding interview: logistic regression, k-means, kNN, AUC and NDCG from scratch in NumPy.
