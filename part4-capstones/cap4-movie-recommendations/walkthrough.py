# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part4-capstones/cap4-movie-recommendations/walkthrough.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Capstone 4 · Movie recommendations (MovieLens) — guided walkthrough
#
# **CPU only · about 2 minutes of compute on ml-latest-small (plus the download) · the brief is [README.md](README.md)**
#
# This notebook is a complete worked solution of the capstone's minimum viable
# product: a top-10 recommender for returning MovieLens users, evaluated with a
# protocol that does not let the model see the future, compared honestly with
# popularity baselines, broken down by slices (cold-start users, long-tail
# items), and served by a small tested API.
#
# ### How to use this walkthrough
#
# Treat it as a tutor sitting next to you, not as code to copy. Every step has
# the same shape:
#
# 1. **Goal** — what the step must produce and why it matters.
# 2. **Your turn.** — a decision to make or a piece of code to write yourself,
#    with a rough time budget. Close the solution, try it, then come back.
# 3. **Our solution.** — the code we chose, followed by *why* we chose it, the
#    alternatives we considered and the mistakes people commonly make.
# 4. **Check.** — assertions that must hold. If one fails on your own version,
#    you have found a bug, which is the point of having it.
#
# The steps follow the brief's milestones: framing → data → audit → split →
# baselines → models → evaluation → slices → API → write-up. Relevant
# chapters: [Chapter 5](../../part1-foundations/ch05-leakage-safe-splits-and-baseline/notebook.ipynb)
# (leakage-safe splits), [Chapter 23](../../part2-ml-at-scale/ch23-implicit-feedback-and-position-bias/notebook.ipynb)
# (implicit feedback), [Chapter 25](../../part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb)
# (candidate generation) and [Chapter 37](../../part2-ml-at-scale/ch37-recsys-and-search-system-design/notebook.ipynb)
# (recommender system design).

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

# %% [markdown]
# ### Data and licence notice
#
# The data are the **MovieLens** research datasets published by **GroupLens
# Research** (University of Minnesota), downloaded by this notebook from the
# official host `files.grouplens.org` and from nowhere else. The terms below
# are summarised from the dataset README and are marked **to verify** in the
# brief: read the README of the release you download before you use it.
#
# * use for **research purposes**, under the README's conditions;
# * **no commercial or revenue-bearing use** without separate permission;
# * no statement or implication of endorsement by the University of Minnesota
#   or GroupLens;
# * acknowledge the dataset in publications (Harper and Konstan, 2015,
#   *ACM TiiS* 5(4), DOI 10.1145/2827872);
# * **do not redistribute** the files except under the same licence. So: never
#   commit them to your repository and never download them from a mirror.
#   This notebook caches them in your artifact directory only.
#
# This project is a non-commercial demonstration. It is not work for, or
# endorsed by, GroupLens, Netflix, Meta, Amazon or any other platform, and the
# API in step 9 must not be deployed as part of a product.
#
# **Offline sample mode (for the course's automated checks only).** If the
# environment variable `MLBOOTCAMP_CAPSTONE_SAMPLE=1` is set, step 2 does not
# download anything. It writes a small **synthetic** dataset with exactly the
# official file names, columns and formats (`ratings.csv`, `movies.csv`,
# `tags.csv`, `links.csv`), generated from users with hidden tastes, and every
# step runs on that. Numbers printed in sample mode describe a toy world we
# invented; they are **not results** and must never go into a README. The
# course build machine cannot reach `files.grouplens.org`, so this notebook was
# verified there in sample mode only; the real-data run is yours to do.

# %%
# --- Configuration -----------------------------------------------------------
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import scipy.sparse as sp

SAMPLE = os.environ.get("MLBOOTCAMP_CAPSTONE_SAMPLE") == "1"

RELEASE = "ml-latest-small"  # development. For reported results: "ml-32m" with USER_SAMPLE set
USER_SAMPLE = None           # e.g. 6000 for ml-32m (a documented random sample of users); None = all
SEED = 2026
POS_THRESHOLD = 4.0          # a rating >= 4.0 counts as a positive (step 1 explains why)
K = 10                       # slate size: Recall@10, NDCG@10
MIN_HISTORY = 5              # users with fewer training positives are "cold" (step 8 checks this choice)
MODEL_MIN_ITEM_POS = 1 if USER_SAMPLE is None else 5  # items a CF model may score (memory, step 6)
BASE_URL = "https://files.grouplens.org/datasets/movielens/"
# Fill this after your first download (step 2 prints the value) so later runs fail
# loudly if the file ever changes. Leave it empty until you have seen the file.
EXPECTED_SHA256: dict[str, str] = {}

CAP = ART / "cap4"
CAP.mkdir(parents=True, exist_ok=True)
LABEL = "SYNTHETIC SAMPLE - not a result" if SAMPLE else RELEASE
DAY = 86_400
T_START = time.time()
print(f"mode: {'offline synthetic sample' if SAMPLE else 'official download'} | release: {RELEASE} | "
      f"user sample: {USER_SAMPLE} | cache: {CAP}")

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Implicit feedback** | a signal of interest that is not a direct answer to "how much do you like this?"; we only know that something happened | "user 12 rated movie 318 with at least 4 stars" becomes a 1 in a user × item matrix |
# | **Request time** | the moment a recommendation is asked for; nothing after it may be used | the timestamp of the held-out rating |
# | **Leave-last-out** | per user, hold out the chronologically last positive for test and the one before for validation | user 7's last 4-star rating is their test item |
# | **Global temporal split** | one cut-off time *T* for everybody: train before *T*, test after | train before 2017-x, test on each user's first positive after it |
# | **Candidate set** | the items the model is allowed to recommend at request time | movies that already existed and that the user has not rated |
# | **Full ranking** | rank the target against *all* candidates, not a sample of 100 negatives | about 9,000 candidates per user on ml-latest-small |
# | **Recall@10 (hit rate)** | did the held-out movie appear in the top 10? | 1 or 0 per user, averaged |
# | **NDCG@10** | like Recall@10, but a hit at rank 1 counts more than a hit at rank 10 | rank 1 → 1.0, rank 3 → 0.5 |
# | **Popularity baseline** | recommend what most people liked, recently or over all time | "most 4-star ratings in the last 90 days" |
# | **Item-kNN / co-occurrence** | movies liked by the same people are similar; recommend neighbours of what you liked | cosine similarity of two item columns |
# | **EASE** | a linear item-to-item model with a closed-form solution | one matrix inverse of size items × items |
# | **ALS (implicit MF)** | factorise the user × item matrix into user and item vectors, alternating between the two | 64-dimensional tastes |
# | **Cold start** | too little history to personalise (new user) or to learn an item (new movie) | fewer than 5 training positives |
# | **Coverage** | share of the catalogue that appears in anyone's top 10 | low coverage = blockbusters for everyone |
# | **Paired bootstrap** | resample *users*, recompute the difference between two models on the same users | 95% CI of NDCG(model) − NDCG(popularity) |

# %% [markdown]
# ## Key equations
#
# **1 · Recall@K with one held-out item per user.** In words: the share of users
# whose held-out item is ranked in the top $K$.
#
# $$
# \operatorname{Recall@}K = \frac{1}{\lvert U \rvert} \sum_{u \in U} \mathbb{1}\left[\,r_u \le K\,\right]
# $$
#
# * $U$: evaluated users; $r_u$: 1-based rank of user $u$'s held-out item among
#   all candidates. With one relevant item, Recall@K equals the hit rate.
#   *Example:* ranks 1, 4 and 37 for three users: $(1 + 1 + 0)/3 = 0.667$.
#
# **2 · NDCG@K with one held-out item.** In words: a hit is worth less the lower
# it is ranked, with a logarithmic discount; the ideal ranking puts the item
# first, so the ideal DCG is 1.
#
# $$
# \operatorname{NDCG@}K = \frac{1}{\lvert U \rvert} \sum_{u \in U} \frac{\mathbb{1}\left[\,r_u \le K\,\right]}{\log_2(r_u + 1)}
# $$
#
# * *Example:* the same ranks 1, 4, 37: $(1/\log_2 2 + 1/\log_2 5 + 0)/3 = (1 + 0.431)/3 = 0.477$.
#
# **3 · Cosine similarity of two items.** In words: how many users liked both,
# relative to how many liked each.
#
# $$
# s_{ij} = \frac{\mathbf{x}_i^\top \mathbf{x}_j}{\lVert \mathbf{x}_i \rVert \, \lVert \mathbf{x}_j \rVert}
# $$
#
# * $\mathbf{x}_i$: the binary column of item $i$ (which users liked it).
#   *Example:* 30 users liked $i$, 20 liked $j$, 12 liked both: $12/\sqrt{30 \cdot 20} = 0.49$.
#
# **4 · EASE (Steck, 2019).** In words: learn item-to-item weights $B$ that
# rebuild each user's row from the *other* items they liked, with a ridge
# penalty and a zero diagonal; the solution is one matrix inverse.
#
# $$
# P = \left(X^\top X + \lambda I\right)^{-1}, \qquad B_{ij} = -\frac{P_{ij}}{P_{jj}} \;(i \ne j), \qquad B_{ii} = 0, \qquad \hat{\mathbf{s}}_u = \mathbf{x}_u B
# $$
#
# * $X$: users × items binary matrix; $\lambda$: ridge strength (tuned on
#   validation). Memory is items², so the item count decides feasibility:
#   10,000 items → $10^8$ float64 values → 0.8 GB.
#
# **5 · Implicit ALS (Hu, Koren and Volinsky, 2008).** In words: fit
# user vectors $\mathbf{p}_u$ and item vectors $\mathbf{q}_i$ so that their dot
# product is near 1 for liked items and near 0 otherwise, trusting observed
# positives $1+\alpha$ times more than unobserved cells.
#
# $$
# \min_{P,Q} \sum_{u,i} c_{ui}\left(y_{ui} - \mathbf{p}_u^\top \mathbf{q}_i\right)^2 + \lambda\left(\sum_u \lVert \mathbf{p}_u \rVert^2 + \sum_i \lVert \mathbf{q}_i \rVert^2\right), \qquad c_{ui} = 1 + \alpha\, y_{ui}
# $$
#
# With $Q$ fixed, each user's vector has a closed form, and vice versa:
#
# $$
# \mathbf{p}_u = \left(Q^\top Q + \alpha\, Q_u^\top Q_u + \lambda I\right)^{-1} (1 + \alpha)\, Q_u^\top \mathbf{1}
# $$
#
# * $y_{ui} \in \{0, 1\}$: liked or not; $Q_u$: rows of $Q$ for the items user
#   $u$ liked. The trick $Q^\top Q$ (shared by all users) plus a small
#   correction is what makes ALS fast on sparse data.
#
# **6 · Percentile bootstrap CI over users.** In words: resample users with
# replacement $B$ times, recompute the mean each time, and report the 2.5th and
# 97.5th percentiles.
#
# $$
# \text{CI}_{95\%} = \left[\, Q_{0.025}\big(\bar{m}^{*}_1, \ldots, \bar{m}^{*}_B\big),\; Q_{0.975}\big(\bar{m}^{*}_1, \ldots, \bar{m}^{*}_B\big) \,\right]
# $$
#
# * $\bar{m}^{*}_b$: the metric's mean over the $b$-th resample of users. For a
#   *paired* comparison resample users once and take the difference of the two
#   models on the same resampled users.

# %% [markdown]
# ## Step 1 · Framing the problem
#
# **Goal.** Write down, before touching data, exactly what is predicted, for
# whom, when, and how it is scored. Every later decision (split, metric,
# baseline) follows from these sentences.
#
# **Your turn.** (15 minutes) Fill in: question, unit of prediction, label,
# horizon, candidate set, metric. Decide whether a 2-star rating should count
# as "the user engaged" or "the user did not like it".
#
# **Our solution.**
#
# | Item | Our choice | Why |
# |---|---|---|
# | Question | For a returning user, which 10 movies they have not rated yet should we show next so that they are likely to like at least one? | The product shows a ranked slate; the decision is "which 10" |
# | Unit of prediction | a (user, request time) pair → a ranked list of 10 movie IDs | one slate per request |
# | Label | the user's next **positive** interaction: a rating ≥ 4.0 | see below |
# | Horizon | the next positive after the request time | leave-last-out takes exactly one |
# | Candidates | every movie that had been rated by anyone **before** the request time, minus movies this user already rated | a model cannot recommend a movie that did not exist yet, and a slate of movies already seen is wasted |
# | Metric | Recall@10 and NDCG@10 by full ranking; coverage and mean popularity alongside | the slate is ranked and only the top 10 is shown; RMSE would score rating prediction for items the user chose, which is not the decision |
#
# **Why rating ≥ 4 as a positive.** MovieLens ratings are explicit, but the
# product question is implicit: "will the user engage with and like this?". A
# rating of any value tells us the user watched (or at least chose to rate) the
# movie; a rating of 1–2 stars tells us they did not like it, and
# recommending more of what users dislike is not success. Thresholding at 4.0
# keeps the "liked it" interactions (on a 0.5–5 scale, 4 is the first value
# that is clearly above the typical user mean of roughly 3.5) and treats the
# rest as *not positive* rather than *negative*: unrated and low-rated cells
# are both zeros in the training matrix. The brief's alternative, **any rating
# = interaction**, answers a different question ("what will they rate next?"),
# and step 8 reports it as a secondary analysis.
#
# *Alternatives considered.* Weighting positives by the rating value
# (confidence $c_{ui} = 1 + \alpha \cdot \text{rating}$) — sensible, one more
# parameter to tune; per-user thresholds (above the user's own mean) — needs
# the user's mean as of request time, which is easy to leak.
#
# *Common mistakes.* Optimising RMSE and calling it a recommender; defining the
# label after looking at which choice gives the best number; forgetting that
# "not rated" is not "disliked".

# %% [markdown]
# ## Step 2 · Data acquisition
#
# **Goal.** Download the release from the official host, check its integrity,
# cache it in the artifact directory (never in the repository), and print the
# licence from the README inside the archive.
#
# **Your turn.** (20 minutes) Write `download_release(release)` that is safe to
# re-run: it downloads once, computes a SHA-256, compares it with a pinned
# value if you have one, and extracts the archive.
#
# **Our solution.** GroupLens publishes one zip per release at
# `https://files.grouplens.org/datasets/movielens/<release>.zip`; for some
# releases an `.md5` file sits next to it (**to verify** for yours — the code
# below tries it and says what it found). We record the SHA-256 for the data
# card either way. In sample mode, the function after it writes the synthetic
# stand-in instead.

# %%
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_release(release: str, dest: Path = CAP) -> Path:
    """Download <release>.zip from files.grouplens.org once, verify it, extract it."""
    url = f"{BASE_URL}{release}.zip"
    zip_path = dest / f"{release}.zip"
    if not zip_path.exists():
        print("downloading", url)
        tmp = zip_path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=300) as response, open(tmp, "wb") as f:
            shutil.copyfileobj(response, f)
        tmp.rename(zip_path)
    digest = sha256_of(zip_path)
    print(f"{zip_path.name}: {zip_path.stat().st_size / 1e6:.2f} MB, sha256 {digest}")
    try:  # an official checksum next to the file, if GroupLens publishes one for this release
        with urllib.request.urlopen(url + ".md5", timeout=30) as response:
            published = response.read().decode().split()[0].strip().lower()
        md5 = hashlib.md5(zip_path.read_bytes()).hexdigest()
        print(f"published md5 {published} | ours {md5} | match: {published == md5}")
        if published != md5:
            raise RuntimeError("MD5 mismatch: delete the zip and download again")
    except OSError as exc:
        print(f"no published .md5 reachable ({type(exc).__name__}); record the SHA-256 above in DATA_CARD.md")
    if release in EXPECTED_SHA256 and EXPECTED_SHA256[release] != digest:
        raise RuntimeError(f"{release}.zip changed since you pinned it: the results would not be comparable")
    folder = dest / release
    if not (folder / "ratings.csv").exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)  # the archive contains a top-level <release>/ folder
    return folder


def print_licence(folder: Path) -> None:
    """Print the usage-licence section of the README shipped inside the archive."""
    readme = next((p for p in (folder / "README.txt", folder / "README.html") if p.exists()), None)
    if readme is None:
        print("README not found in the archive: read the licence on the official page before continuing")
        return
    lines = readme.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next((i for i, ln in enumerate(lines) if "usage license" in ln.lower()), 0)
    print("\n".join(lines[start:start + 40]))

# %% [markdown]
# The synthetic stand-in, used **only** when `MLBOOTCAMP_CAPSTONE_SAMPLE=1`.
# It imitates the official format (comma-separated, header row,
# `userId,movieId,rating,timestamp` with half-star ratings and Unix-second
# timestamps; `movieId,title,genres` with the year in parentheses and
# pipe-separated genres including `(no genres listed)`) and the behaviours the
# brief warns about: onboarding bursts, tied timestamps, old movies rated
# recently, every user with at least 20 ratings. Each simulated user has a
# hidden taste vector, so a collaborative model *can* beat popularity here; new
# movies get a temporary boost, so time matters. None of this says anything
# about real MovieLens.

# %%
def write_synthetic_movielens(folder: Path, n_users: int = 480, n_items: int = 1600, seed: int = 7) -> Path:
    """Write a SYNTHETIC dataset in the ml-latest-small file format. Not real data."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("1996-03-29", tz="UTC").timestamp()
    end = pd.Timestamp("2018-09-24", tz="UTC").timestamp()
    d = 6
    item_f = rng.normal(size=(n_items, d))
    year = np.clip(np.round(2018 - rng.exponential(16, n_items)), 1915, 2018).astype(int)
    year_ts = np.array([pd.Timestamp(f"{y}-01-01", tz="UTC").timestamp() for y in year])
    entry = np.maximum(start, year_ts) + rng.exponential(150 * DAY, n_items)
    order = np.argsort(entry)  # movieIds grow with the date a movie enters the catalogue
    item_f, year, entry = item_f[order], year[order], entry[order]
    movie_ids = np.sort(rng.choice(np.arange(1, 200_000), n_items, replace=False))
    appeal = rng.normal(0, 1.0, n_items)
    quality = rng.normal(0, 0.4, n_items)
    user_f = rng.normal(size=(n_users, d))
    join = start + (end - start - 60 * DAY) * rng.beta(1.4, 1.0, n_users)
    n_target = np.minimum(20 + rng.lognormal(3.3, 0.9, n_users).astype(int), 600)
    rows = []
    for u in range(n_users):
        affinity = item_f @ user_f[u] / np.sqrt(d)
        n_sess = 1 + rng.poisson(2.5)
        sizes = rng.multinomial(n_target[u], rng.dirichlet([4.0] + [1.0] * (n_sess - 1)))
        t_sess = join[u] + np.concatenate([[0], np.cumsum(rng.exponential(400 * DAY, n_sess - 1))])
        rated = np.zeros(n_items, bool)
        for size, t0 in zip(sizes, t_sess):
            if t0 > end or size == 0:
                continue
            avail = (entry <= t0) & ~rated
            age = np.where(avail, (t0 - entry) / (365 * DAY), np.inf)
            logit = 1.6 * affinity + 0.9 * appeal + 2.0 * np.exp(-age)
            g = np.where(avail, logit + rng.gumbel(size=n_items), -np.inf)
            chosen = np.argsort(-g)[: min(size, int(avail.sum()))]
            rated[chosen] = True
            gaps = rng.choice([0, 1, 4, 20, 45, 90, 300], size=len(chosen))
            ts = (t0 + np.cumsum(gaps)).astype(np.int64)
            raw = 3.0 + 0.9 * affinity[chosen] + quality[chosen] + rng.normal(0, 0.7, len(chosen))
            stars = np.clip(np.round(raw * 2) / 2, 0.5, 5.0)
            rows.append(pd.DataFrame({"userId": u + 1, "movieId": movie_ids[chosen], "rating": stars,
                                      "timestamp": np.minimum(ts, int(end))}))
    ratings = pd.concat(rows, ignore_index=True)
    keep = ratings.groupby("userId")["movieId"].transform("size") >= 20  # the release's own rule
    ratings = ratings[keep].sort_values(["userId", "movieId"]).reset_index(drop=True)
    genres = np.array(["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
                       "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi"])
    top2 = np.argsort(-np.abs(item_f), axis=1)[:, :2]
    names = ["|".join(genres[2 * j + (item_f[i, j] > 0)] for j in top2[i]) for i in range(n_items)]
    names = [n if rng.random() > 0.02 else "(no genres listed)" for n in names]
    titles = [f"Synthetic Film {m} ({y})" if rng.random() > 0.01 else f"Synthetic Film {m}"
              for m, y in zip(movie_ids, year)]
    movies = pd.DataFrame({"movieId": movie_ids, "title": titles, "genres": names})
    links = pd.DataFrame({"movieId": movie_ids, "imdbId": [f"{x:07d}" for x in rng.integers(1, 9_999_999, n_items)],
                          "tmdbId": rng.integers(1, 500_000, n_items)})
    tagged = ratings.sample(300, random_state=seed)
    tags = pd.DataFrame({"userId": tagged["userId"], "movieId": tagged["movieId"],
                         "tag": rng.choice(["funny", "slow", "classic", "twist ending", "atmospheric"], 300),
                         "timestamp": tagged["timestamp"] + 60})
    ratings.to_csv(folder / "ratings.csv", index=False)
    movies.to_csv(folder / "movies.csv", index=False)
    links.to_csv(folder / "links.csv", index=False)
    tags.to_csv(folder / "tags.csv", index=False)
    (folder / "README.txt").write_text(
        "SYNTHETIC SAMPLE written by the MLBOOTCAMP capstone 4 walkthrough for offline checks.\n"
        "It is NOT MovieLens data and has no relation to real users or movies.\n\n"
        "Usage License\n=============\nNot applicable (synthetic). The real terms are in the README of the\n"
        "official release at https://files.grouplens.org/datasets/movielens/\n", encoding="utf-8")
    return folder

# %%
t0 = time.time()
if SAMPLE:
    DATA_DIR = write_synthetic_movielens(CAP / "synthetic-ml-latest-small")
    print("SYNTHETIC SAMPLE written to", DATA_DIR, "- nothing was downloaded")
else:
    DATA_DIR = download_release(RELEASE)
print_licence(DATA_DIR)
print(f"files: {sorted(p.name for p in DATA_DIR.iterdir())} | {time.time() - t0:.1f}s")

# %% [markdown]
# **Check.** The four CSV files exist. On the real release, read the printed
# licence section in full now and quote its key condition in your
# `DATA_CARD.md`; then paste the printed SHA-256 into `EXPECTED_SHA256`.
#
# *Why cache in the artifact directory?* The licence does not allow
# redistribution, so the files must never land in git; the artifact directory
# is ignored by git and survives re-runs. *Common mistakes:* downloading from a
# Kaggle or GitHub mirror (against the terms, and you cannot prove which
# version you have); trusting a checksum computed after you modified the file.

# %%
assert {"ratings.csv", "movies.csv", "tags.csv", "links.csv"} <= {p.name for p in DATA_DIR.iterdir()}

# %% [markdown]
# ## Step 3 · Data audit
#
# **Goal.** Know the data before modelling: schema and types, missing values,
# value ranges, how activity is distributed, and the oddities the brief warns
# about (bursts, tied timestamps, old movies rated recently, genre and title
# formats). Each claim below is printed as evidence, not asserted from memory.
#
# **Your turn.** (30 minutes) List five things that could break a time-aware
# evaluation and write one line of pandas that measures each.
#
# **Our solution.**

# %%
ratings_raw = pd.read_csv(DATA_DIR / "ratings.csv",
                          dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"})
movies = pd.read_csv(DATA_DIR / "movies.csv", dtype={"movieId": "int32", "title": "string", "genres": "string"})
if USER_SAMPLE:  # a documented random sample of users, for the large releases
    chosen = np.random.default_rng(SEED).choice(ratings_raw["userId"].unique(), USER_SAMPLE, replace=False)
    ratings_raw = ratings_raw[ratings_raw["userId"].isin(chosen)].reset_index(drop=True)
    print(f"user sample: {USER_SAMPLE} users, seed {SEED}")

print(f"[{LABEL}]")
print("ratings:", ratings_raw.shape, dict(ratings_raw.dtypes.astype(str)))
print("movies: ", movies.shape, dict(movies.dtypes.astype(str)))
print("missing values:", int(ratings_raw.isna().sum().sum()), "in ratings,", int(movies.isna().sum().sum()), "in movies")
print("duplicate (userId, movieId) pairs:", int(ratings_raw.duplicated(["userId", "movieId"]).sum()))
print("rating values:", sorted(ratings_raw["rating"].unique().tolist()))
ts = pd.to_datetime(ratings_raw["timestamp"], unit="s", utc=True)
print(f"time range: {ts.min():%Y-%m-%d} to {ts.max():%Y-%m-%d}")
per_user = ratings_raw.groupby("userId").size()
per_item = ratings_raw.groupby("movieId").size()
print(f"users {per_user.size:,} | movies rated {per_item.size:,} of {len(movies):,} listed | ratings {len(ratings_raw):,}")
print(f"ratings per user: min {per_user.min()}, median {per_user.median():.0f}, max {per_user.max()}")
print(f"ratings per movie: median {per_item.median():.0f}; movies with a single rating: {(per_item == 1).mean():.1%}")
print(f"share of ratings >= {POS_THRESHOLD}: {(ratings_raw['rating'] >= POS_THRESHOLD).mean():.1%}")

# %%
# The oddities the brief warns about, measured
r = ratings_raw.sort_values(["userId", "timestamp", "movieId"])
same_second = (r.groupby("userId")["timestamp"].diff() == 0).mean()
first_ts = r.groupby("userId")["timestamp"].transform("min")
first_day = ((r["timestamp"] - first_ts) < DAY).mean()
year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("float")
rated_year = r.merge(movies[["movieId"]].assign(year=year), on="movieId", how="left")
age_at_rating = pd.to_datetime(rated_year["timestamp"], unit="s").dt.year - rated_year["year"]
print(f"consecutive ratings by a user in the same second (ties): {same_second:.1%}")
print(f"ratings made within a day of the user's first rating (onboarding burst): {first_day:.1%}")
print(f"titles without a parsable (year): {year.isna().sum()} | '(no genres listed)': "
      f"{(movies['genres'] == '(no genres listed)').sum()}")
print(f"median years between release and rating: {age_at_rating.median():.0f}; "
      f"share rated 20+ years after release: {(age_at_rating >= 20).mean():.1%}")
print("genres per movie (pipe-separated):", movies["genres"].str.count(r"\|").add(1).describe()[["mean", "max"]].round(2).to_dict())

# %% [markdown]
# **Check.** The release's README promises at least 20 ratings per user (**to
# verify** for yours), half-star values in 0.5–5 and no duplicate pairs. If
# any of these fails, stop and find out why before modelling.
#
# **What to read in these numbers.** A large share of ratings within a day of
# the first one means that "the user's next interaction" is often part of the
# sign-up flow, where users rate movies the site showed them; that is a
# limitation of the label, not something to fix. Tied timestamps mean
# "chronological order" is not unique: we break ties by `movieId` and say so.
# Movies rated decades after release mean "new movie" must be defined by the
# first rating in the data, not by the release year.
#
# *Common mistakes:* reading the large releases into a dense user × item matrix
# (it does not fit); assuming `movieId` is contiguous (it is not, so we build an
# index); treating the title year as the date a movie became available here.

# %%
assert per_user.min() >= 20, "fewer than 20 ratings for some user: check the release README"
assert ratings_raw["rating"].between(0.5, 5.0).all() and (ratings_raw["rating"] * 2 % 1 == 0).all()
assert not ratings_raw.duplicated(["userId", "movieId"]).any()

# %% [markdown]
# ## Step 4 · A leakage-safe split for this problem
#
# **Goal.** Build training, validation and test sets that imitate deployment:
# at request time, the system knows only what happened before.
#
# **Your turn.** (45 minutes) Choose between a **global temporal split** (one
# cut-off *T*) and a **per-user leave-last-out by timestamp**. Write down what
# each one leaks and what each one costs.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap4-time-split.svg" alt="Three splits on a time axis: a random split hides a past rating and trains on the future; leave-last-out holds out each user's last positive but trains on other users' later events; a global temporal split trains on everything before one time T" width="100%">
#
# *What to notice:* only the global temporal split uses no event after any
# test request; leave-last-out leaks other users' future, a random split leaks
# the user's own future.
#
# **Our solution: leave-last-out by timestamp as the primary protocol, the
# global temporal split as the robustness check.**
#
# * **Why not random.** A random held-out rating has, in most cases, later
#   ratings of the same user in training: the model learns what the user did
#   next and is asked about something in between. It also learns from
#   popularity that had not happened yet. Scores go up for reasons that do not
#   exist at serving time. Step 7 measures this once.
# * **Why leave-last-out.** Every user with at least three positives gets one
#   test request, so the evaluation covers all kinds of users (light and heavy,
#   early and late). Its weakness is *global* leakage: user A's test may be in
#   2001 while user B's training ratings from 2015 are in the training matrix.
#   We reduce it as the brief asks: (1) candidates must have been rated by
#   someone **before** the request time; (2) popularity baselines count only
#   events before the request time; (3) we report the global temporal split as
#   well and compare.
# * **Why not only the global split.** With one cut-off, only users active on
#   both sides of *T* can be tested; on ml-latest-small that is a small,
#   unrepresentative group (heavy long-term users), so the confidence
#   intervals are wide.
#
# Concretely: sort each user's ratings by (timestamp, movieId). **Test** = the
# last positive; **validation** = the second-to-last positive; **training for
# the test stage** = every rating of that user before the test positive (in
# this order), plus the same for all other users; **training for the
# validation stage** = every rating before the validation positive. Ratings
# after a user's held-out positive are discarded for that stage: they are that
# user's future.

# %%
def index_ratings(raw: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Contiguous user/item indices, chronological order per user (ties broken by movieId)."""
    df = raw.sort_values(["userId", "timestamp", "movieId"], kind="mergesort").reset_index(drop=True)
    df["u"] = np.searchsorted(USER_IDS, df["userId"].to_numpy())
    df["i"] = np.searchsorted(ITEM_IDS, df["movieId"].to_numpy())
    df["seq"] = df.groupby("u").cumcount()
    df["pos"] = df["rating"] >= threshold
    return df


USER_IDS = np.sort(ratings_raw["userId"].unique())
ITEM_IDS = np.sort(ratings_raw["movieId"].unique())
N_USERS, N_ITEMS = len(USER_IDS), len(ITEM_IDS)
ratings = index_ratings(ratings_raw, POS_THRESHOLD)
FIRST_TS = ratings.groupby("i")["timestamp"].min().reindex(range(N_ITEMS)).to_numpy()  # when each movie "appears"
TITLE = movies.set_index("movieId")["title"].reindex(ITEM_IDS).fillna("?").to_numpy()
print(f"{N_USERS:,} users x {N_ITEMS:,} rated movies | positives: {ratings['pos'].sum():,}")

# %% [markdown]
# A **stage** holds everything one evaluation needs: the training table, the
# users evaluated with their target item and request time, the binary training
# matrix, and the items each user had already rated before the request.

# %%
class Stage:
    """One evaluation: training events, evaluated users, their target and request time."""

    def __init__(self, name, train, users, targets, times, n_users=None):
        self.name, self.train = name, train
        self.users, self.targets, self.times = np.asarray(users), np.asarray(targets), np.asarray(times, float)
        pos = train[train["pos"]]
        self.X = sp.csr_matrix((np.ones(len(pos), np.float32), (pos["u"], pos["i"])),
                               shape=(n_users or N_USERS, N_ITEMS))
        tu, ti = train["u"].to_numpy(), train["i"].to_numpy()
        order_u = np.argsort(tu, kind="stable")
        bounds = np.searchsorted(tu[order_u], [self.users, self.users + 1])
        self.seen = [ti[order_u[lo:hi]] for lo, hi in zip(*bounds)]  # movies rated before the request
        order = np.argsort(pos["timestamp"].to_numpy(), kind="stable")
        self.pop_ts = pos["timestamp"].to_numpy()[order]  # training positives sorted by time,
        self.pop_items = pos["i"].to_numpy()[order]       # for "as of request time" popularity
        self.n_train_pos = np.asarray(self.X[self.users].sum(1)).ravel()

    def __repr__(self):
        return (f"Stage({self.name}: {len(self.users):,} users, {len(self.train):,} training ratings, "
                f"{int(self.X.sum()):,} training positives)")


def leave_last_out(df: pd.DataFrame, min_pos: int = 3) -> dict[str, Stage]:
    """Per-user, time-ordered leave-last-out: test = last positive, validation = second-to-last."""
    pos = df[df["pos"]]
    g = pos.groupby("u")
    eligible = g.size()[lambda s: s >= min_pos].index.to_numpy()
    last = g.nth(-1).set_index("u").loc[eligible]
    second = g.nth(-2).set_index("u").loc[eligible]
    stages = {}
    for name, held in (("validation", second), ("test", last)):
        cut = pd.Series(np.iinfo(np.int64).max, index=range(N_USERS))
        cut.loc[eligible] = held["seq"].to_numpy()  # users with too few positives keep all events
        train = df[df["seq"].to_numpy() < cut.loc[df["u"]].to_numpy()]
        stages[name] = Stage(name, train, eligible, held["i"].to_numpy(), held["timestamp"].to_numpy())
    return stages


t0 = time.time()
STAGES = leave_last_out(ratings)
VAL, TEST = STAGES["validation"], STAGES["test"]
print(VAL)
print(TEST)
print(f"users not evaluated (fewer than 3 positives): {N_USERS - len(TEST.users)} | {time.time() - t0:.1f}s")

# %% [markdown]
# **Check.** The tests a reviewer would ask for, run on the actual split:
# no training rating of an evaluated user at or after their held-out position,
# no training rating later than their request time, the held-out item not in
# their training rows, and validation strictly before test for every user.

# %%
def check_stage(stage: Stage, df: pd.DataFrame) -> None:
    held_seq = df.set_index(["u", "i"])["seq"]
    tr = stage.train[stage.train["u"].isin(stage.users)]
    lim = pd.Series(held_seq.loc[list(zip(stage.users, stage.targets))].to_numpy(), index=stage.users)
    tlim = pd.Series(stage.times, index=stage.users)
    assert (tr["seq"].to_numpy() < lim.loc[tr["u"]].to_numpy()).all(), "a user's future is in training"
    assert (tr["timestamp"].to_numpy() <= tlim.loc[tr["u"]].to_numpy()).all(), "training rating after request time"
    assert not any(t in s for t, s in zip(stage.targets, stage.seen)), "target already in the user's history"


for st in (VAL, TEST):
    check_stage(st, ratings)
v_seq = ratings.set_index(["u", "i"])["seq"]
assert (v_seq.loc[list(zip(VAL.users, VAL.targets))].to_numpy()
        < v_seq.loc[list(zip(TEST.users, TEST.targets))].to_numpy()).all()

# How much *global* future does leave-last-out leave in the test training table?
later = np.searchsorted(np.sort(TEST.train["timestamp"].to_numpy()), TEST.times, side="right")
share_future = 1 - later / len(TEST.train)
print(f"per test user, share of training ratings (all users) made after their request time: "
      f"median {np.median(share_future):.1%}, mean {share_future.mean():.1%}")
print(f"test targets that no one had rated before the request time (unrecommendable): "
      f"{(FIRST_TS[TEST.targets] >= TEST.times).mean():.1%}")

# %% [markdown]
# The first number is the size of the global leakage that leave-last-out
# leaves in: it is why step 7 also runs the global temporal split. The second
# is the share of test targets that *no* recommender could have shown, because
# the movie had never been rated before the request — an honest ceiling below
# 1.0 for every model.
#
# *Common mistakes:* splitting by row index after sorting by user only (ties
# and order are then arbitrary); building the "already rated" filter from the
# whole dataset (it removes the test item from the candidates: an automatic
# miss) or not at all (the slate is full of movies the user has seen);
# computing popularity on all data.

# %% [markdown]
# ## Step 5 · Metrics and baselines
#
# **Goal.** A full-ranking evaluator, unit-tested on cases you can compute by
# hand, and the two popularity baselines every model must beat.
#
# **Your turn.** (40 minutes) Write `evaluate(stage, score_fn)` that masks
# movies that did not exist yet and movies the user already rated, finds the
# rank of the target among all remaining candidates, and returns per-user
# Recall@10 and NDCG@10. Decide how to break score ties.
#
# **Our solution.** Ties are broken by item index (deterministic, and the same
# rule for the reported rank and for the served top 10). A target whose score
# is $-\infty$ (masked, or an item the model cannot score) is a miss.

# %%
def evaluate(stage: Stage, score_fn, k: int = K, batch: int = 256) -> dict:
    """Full ranking against every candidate. Returns per-user ranks (0-based) and each top-k list."""
    n = len(stage.users)
    ranks, topk = np.full(n, np.inf), np.zeros((n, k), dtype=np.int64)
    item_idx = np.arange(N_ITEMS)
    for s in range(0, n, batch):
        rows = np.arange(s, min(n, s + batch))
        S = np.array(score_fn(stage, rows), dtype=np.float64)
        S[FIRST_TS[None, :] >= stage.times[rows, None]] = -np.inf  # movie not in the catalogue yet
        for j, r in enumerate(rows):
            S[j, stage.seen[r]] = -np.inf                           # already rated before the request
        tgt = stage.targets[rows]
        st = S[np.arange(len(rows)), tgt]
        rk = (S > st[:, None]).sum(1) + ((S == st[:, None]) & (item_idx[None, :] < tgt[:, None])).sum(1)
        ranks[rows] = np.where(np.isfinite(st), rk, np.inf)
        topk[rows] = np.argsort(-S, axis=1, kind="stable")[:, :k]
    return {"rank": ranks, "topk": topk}


def recall_at_k(rank, k=K):
    return (np.asarray(rank) < k).astype(float)


def ndcg_at_k(rank, k=K):
    rank = np.asarray(rank, dtype=float)
    out = np.zeros(rank.shape)
    hit = rank < k
    out[hit] = 1.0 / np.log2(rank[hit] + 2)  # 0-based rank r -> 1 / log2(r + 2)
    return out

# %% [markdown]
# **Check.** Hand-computed cases: ranks 1, 4 and 37 (1-based) give Recall@10
# 0.667 and NDCG@10 0.477 (key equations 1 and 2). And a five-item toy stage
# where we know the answer: the target has the second-highest score, but the
# highest-scoring movie was already rated, so the target must be ranked first.

# %%
assert np.isclose(recall_at_k([0, 3, 36]).mean(), 2 / 3)
assert np.isclose(ndcg_at_k([0, 3, 36]).mean(), (1 + 1 / np.log2(5)) / 3)
assert ndcg_at_k([10])[0] == 0 and ndcg_at_k([np.inf])[0] == 0

toy = Stage.__new__(Stage)
toy.users, toy.targets, toy.times = np.array([0]), np.array([2]), np.array([np.inf])
toy.seen = [np.array([0])]
toy_scores = np.zeros((1, N_ITEMS))
toy_scores[0, :5] = [9, 1, 5, 3, 7]  # item 0 was rated before -> masked; item 4 (7) beats the target (5)
res = evaluate(toy, lambda st, rows: toy_scores)
assert res["rank"][0] == 1 and list(res["topk"][0, :2]) == [4, 2]
print("metric and evaluator checks passed")

# %% [markdown]
# **The baselines.**
#
# 1. **Global popularity (as of request time):** count each movie's positives
#    in the stage's training table *before the request time*, recommend the
#    most counted.
# 2. **Recent popularity:** the same count restricted to the last $N$ days
#    before the request time, ties broken by global popularity; $N$ is chosen
#    on validation.
#
# Popularity uses only the stage's training table, so validation and test
# targets never count towards it.

# %%
def popularity_fn(window_days=None):
    def score(stage: Stage, rows):
        out = np.zeros((len(rows), N_ITEMS))
        for j, r in enumerate(rows):
            t = stage.times[r]
            hi = np.searchsorted(stage.pop_ts, t, side="left")  # strictly before the request
            total = np.bincount(stage.pop_items[:hi], minlength=N_ITEMS).astype(float)
            if window_days is None:
                out[j] = total
            else:
                lo = np.searchsorted(stage.pop_ts, t - window_days * DAY, side="left")
                recent = np.bincount(stage.pop_items[lo:hi], minlength=N_ITEMS)
                out[j] = recent + total / (total.max() + 1)  # ties broken by all-time popularity
        return out
    return score


def summarise(res: dict, stage: Stage) -> dict:
    rec, nd = recall_at_k(res["rank"]), ndcg_at_k(res["rank"])
    train_pop = np.asarray(stage.X.sum(0)).ravel()
    return {"Recall@10": rec.mean(), "NDCG@10": nd.mean(),
            "coverage": len(np.unique(res["topk"])) / N_ITEMS,
            "mean_pop": train_pop[res["topk"]].mean()}


t0 = time.time()
val_rows = {}
VAL_RES = {"pop_global": evaluate(VAL, popularity_fn())}
val_rows["pop_global"] = summarise(VAL_RES["pop_global"], VAL)
for days in (7, 30, 90, 365, 1095):
    VAL_RES[f"pop_recent_{days}d"] = evaluate(VAL, popularity_fn(days))
    val_rows[f"pop_recent_{days}d"] = summarise(VAL_RES[f"pop_recent_{days}d"], VAL)
print(f"[{LABEL}] validation, {len(VAL.users)} users")
print(pd.DataFrame(val_rows).T.round(4).to_string())
BEST_DAYS = max((7, 30, 90, 365, 1095), key=lambda d: val_rows[f"pop_recent_{d}d"]["NDCG@10"])
print(f"chosen recent-popularity window on validation: {BEST_DAYS} days | {time.time() - t0:.1f}s")

# %% [markdown]
# **Why these baselines.** Popularity is the strongest simple recommender in
# most offline studies, and recent popularity is often much stronger: tastes
# and catalogues move, and "what people liked last month" is a good guess for
# "what this person will like next". If a learned model cannot beat recent
# popularity on the same users, it is not worth serving.
#
# *Common mistakes:* counting popularity over all data (the test targets are
# then part of the score — step 7 shows how much that inflates it); choosing
# the window on the test set; forgetting to remove already-rated movies from
# the popularity list.

# %% [markdown]
# ## Step 6 · Improved models: item co-occurrence, EASE and implicit ALS
#
# **Goal.** Three collaborative-filtering models of increasing flexibility,
# tuned on validation only, behind one interface: `score(X_rows)` takes the
# users' binary history rows and returns a score for every movie
# ($-\infty$ for movies the model cannot score).
#
# **Your turn.** (2 hours) Implement item-kNN with cosine similarity and either
# EASE or ALS in NumPy. Estimate the memory of every items × items matrix
# before you allocate it.
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap4-implicit-models.svg" alt="The binary user by item matrix feeds three models: item-kNN (cosine of item columns), EASE (closed-form item-to-item weights) and ALS (user and item factor vectors); each produces a score for every movie, then already-rated and not-yet-existing movies are masked" width="100%">
#
# *What to notice:* all three models only see the binary training matrix; the
# masking rules of step 5 are applied afterwards, identically for every model
# and baseline.
#
# **Our solution.** Item-kNN and EASE are item-to-item models: a user's scores
# are a weighted sum over the movies they liked. ALS learns a vector per user
# and per movie; to score a user we *fold in* their history (solve for the user
# vector with item vectors fixed), which also handles users who were not in the
# training matrix — exactly what the API needs for new users.

# %%
def model_items(X, min_pos=MODEL_MIN_ITEM_POS):
    counts = np.asarray(X.sum(0)).ravel()
    return np.flatnonzero(counts >= min_pos)


class ItemKNN:
    """Cosine similarity between item columns of the binary matrix; score = history @ S."""

    name = "item_knn"

    def fit(self, X):
        self.items = model_items(X)
        Xm = X[:, self.items]
        G = (Xm.T @ Xm).toarray().astype(np.float32)
        norms = np.sqrt(np.diag(G)) + 1e-9
        self.S = G / norms[:, None] / norms[None, :]
        np.fill_diagonal(self.S, 0.0)
        return self

    def score(self, X_rows):
        out = np.full((X_rows.shape[0], N_ITEMS), -np.inf)
        out[:, self.items] = X_rows[:, self.items] @ self.S
        return out


class EASE:
    """Steck (2019): B = -P / diag(P), P = (X'X + lambda I)^-1, zero diagonal."""

    name = "ease"

    def __init__(self, lam=200.0):
        self.lam = lam

    def fit(self, X):
        self.items = model_items(X)
        Xm = X[:, self.items]
        G = (Xm.T @ Xm).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.lam
        P = np.linalg.inv(G)
        self.B = (-P / np.diag(P)[None, :]).astype(np.float32)
        np.fill_diagonal(self.B, 0.0)
        return self

    def score(self, X_rows):
        out = np.full((X_rows.shape[0], N_ITEMS), -np.inf)
        out[:, self.items] = X_rows[:, self.items] @ self.B
        return out


class ImplicitALS:
    """Hu, Koren & Volinsky (2008) with binary preferences and confidence 1 + alpha."""

    name = "als"

    def __init__(self, factors=64, reg=1.0, alpha=10.0, iters=10, seed=SEED):
        self.f, self.reg, self.alpha, self.iters, self.seed = factors, reg, alpha, iters, seed

    def _solve(self, C, Y):
        """One half-step: for every row r of C, the closed-form vector given Y fixed."""
        YtY = Y.T @ Y + self.reg * np.eye(self.f)
        out = np.zeros((C.shape[0], self.f))
        for r in range(C.shape[0]):
            idx = C.indices[C.indptr[r]:C.indptr[r + 1]]
            if idx.size:
                Yr = Y[idx]
                out[r] = np.linalg.solve(YtY + self.alpha * Yr.T @ Yr, (1 + self.alpha) * Yr.sum(0))
        return out

    def fit(self, X):
        self.items = model_items(X)
        Cui = X[:, self.items].tocsr()
        Ciu = Cui.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.V = rng.normal(0, 0.01, (len(self.items), self.f))
        for _ in range(self.iters):
            U = self._solve(Cui, self.V)
            self.V = self._solve(Ciu, U)
        return self

    def score(self, X_rows):
        U = self._solve(sp.csr_matrix(X_rows[:, self.items]), self.V)  # fold-in: works for new users too
        out = np.full((X_rows.shape[0], N_ITEMS), -np.inf)
        out[:, self.items] = U @ self.V.T
        return out


def model_fn(model):
    return lambda stage, rows: model.score(stage.X[stage.users[rows]])


def clone(m):
    """A fresh, unfitted model with the same hyperparameters (refit on another stage)."""
    if m.name == "ease":
        return EASE(m.lam)
    if m.name == "als":
        return ImplicitALS(factors=m.f, reg=m.reg, alpha=m.alpha, iters=m.iters)
    return ItemKNN()


n_model = len(model_items(VAL.X))
print(f"items a CF model may score on validation: {n_model:,} -> one dense items x items float64 matrix "
      f"= {n_model ** 2 * 8 / 1e9:.2f} GB")

# %% [markdown]
# **Check.** Before running EASE, the printed size must fit comfortably in
# memory (Colab's free tier has about 12 GB). If it does not, raise
# `MODEL_MIN_ITEM_POS`: dropping movies with very few positives costs little
# Recall, and step 8 measures what it costs.
#
# Now tune on validation. The grids are deliberately small; each fit is
# seconds to a minute on ml-latest-small.

# %%
t0 = time.time()
CANDIDATES = [ItemKNN()] + [EASE(lam) for lam in (20.0, 100.0, 500.0)] + [
    ImplicitALS(factors=f, reg=reg, alpha=a) for f, reg, a in ((32, 1.0, 10.0), (64, 5.0, 10.0), (64, 5.0, 30.0))]
for m in CANDIDATES:
    t1 = time.time()
    m.fit(VAL.X)
    key = f"{m.name}" + (f"(lam={m.lam:g})" if m.name == "ease" else "") + (
        f"(f={m.f},reg={m.reg:g},a={m.alpha:g})" if m.name == "als" else "")
    VAL_RES[key] = evaluate(VAL, model_fn(m))
    val_rows[key] = summarise(VAL_RES[key], VAL) | {"fit_s": time.time() - t1}
    m.key = key
VAL_TABLE = pd.DataFrame(val_rows).T
print(f"[{LABEL}] validation, {len(VAL.users)} users (model selection only)")
print(VAL_TABLE.round(4).to_string())
best_of = {name: max((m for m in CANDIDATES if m.name == name), key=lambda m: val_rows[m.key]["NDCG@10"])
           for name in ("item_knn", "ease", "als")}
BEST = max(best_of.values(), key=lambda m: val_rows[m.key]["NDCG@10"])
print("best per family:", {k: m.key for k, m in best_of.items()})
print(f"selected model (validation NDCG@10): {BEST.key} | {time.time() - t0:.0f}s")

# %% [markdown]
# **Why these three.** Item-kNN is the classic co-occurrence baseline and is
# easy to explain to a product owner ("people who liked X also liked Y"). EASE
# is often surprisingly strong for its cost and has a single parameter. ALS is
# the standard implicit-feedback matrix factorisation; its vectors double as
# embeddings for candidate generation ([Chapter 25](../../part2-ml-at-scale/ch25-candidate-generation/notebook.ipynb)).
# A LightGBM re-ranker on top (features: model score, popularity, recency,
# genre match) is the natural next step and is listed as an extension.
#
# *Alternatives considered.* BPR (pairwise, trained by SGD; more knobs, similar
# results in practice), the `implicit` library (faster, but hides the algebra
# this capstone wants you to own), sequence models (need a GPU at scale).
#
# *Common mistakes:* tuning on the test users; giving the models a different
# candidate filter from the baselines; letting $\lambda$ in EASE go to zero
# (the inverse then mostly fits noise); forgetting the zero diagonal in EASE,
# which makes every movie recommend itself.

# %% [markdown]
# ## Step 7 · Evaluation on the test stage, with uncertainty
#
# **Goal.** Refit the chosen configurations on the test stage's training
# table, evaluate every model **once** on the test users, and report 95%
# bootstrap confidence intervals and paired differences against recent
# popularity.
#
# **Your turn.** (30 minutes) Write the bootstrap. Decide what is resampled
# (users, not ratings: users are the independent units here).
#
# **Our solution.**

# %%
def bootstrap_ci(values, n_boot=1000, seed=SEED):
    values = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(values), (n_boot, len(values)))
    means = values[idx].mean(1)
    return values.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def fmt(ci):
    return f"{ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]"


t0 = time.time()
TEST_FNS = {"pop_global": popularity_fn(), f"pop_recent_{BEST_DAYS}d": popularity_fn(BEST_DAYS)}
TEST_MODELS = {}
for name, m in best_of.items():
    TEST_MODELS[m.key] = clone(m).fit(TEST.X)
    TEST_FNS[m.key] = model_fn(TEST_MODELS[m.key])
TEST_RES = {name: evaluate(TEST, fn) for name, fn in TEST_FNS.items()}
REF = f"pop_recent_{BEST_DAYS}d"
rows = []
for name, res in TEST_RES.items():
    s = summarise(res, TEST)
    nd, nd_ref = ndcg_at_k(res["rank"]), ndcg_at_k(TEST_RES[REF]["rank"])
    rows.append({"model": name, "Recall@10": fmt(bootstrap_ci(recall_at_k(res["rank"]))),
                 "NDCG@10": fmt(bootstrap_ci(nd)), "NDCG diff vs recent pop": fmt(bootstrap_ci(nd - nd_ref)),
                 "coverage": f"{s['coverage']:.3f}", "mean popularity": f"{s['mean_pop']:.1f}"})
TEST_TABLE = pd.DataFrame(rows).set_index("model")
print(f"[{LABEL}] TEST (leave-last-out by timestamp), {len(TEST.users)} users, full ranking over {N_ITEMS:,} movies")
print(TEST_TABLE.to_string())
print(f"{time.time() - t0:.0f}s")

# %% [markdown]
# **How to read the table.** A paired difference whose interval excludes 0
# says the model beat (or lost to) recent popularity *on these users*; an
# interval that includes 0 says the data cannot tell them apart, and that is a
# result worth reporting too. Look at coverage and mean popularity next to the
# accuracy columns: a model can raise Recall by recommending blockbusters to
# everyone, which the popularity column exposes.
#
# **Robustness check 1: the global temporal split.** One cut-off *T* (the 80th
# percentile of rating times), the same hyperparameters (no re-tuning), request
# time *T* for every test user, target = each user's first positive after *T*.

# %%
def global_temporal(df: pd.DataFrame, q: float = 0.8) -> Stage:
    T = float(np.quantile(df["timestamp"], q))
    train = df[df["timestamp"] < T]
    after = df[(df["timestamp"] >= T) & df["pos"]].sort_values(["u", "timestamp", "movieId"])
    first_after = after.groupby("u").head(1)
    has_hist = train[train["pos"]].groupby("u").size()
    first_after = first_after[first_after["u"].isin(has_hist.index)]
    return Stage(f"global T={pd.to_datetime(T, unit='s'):%Y-%m-%d}", train, first_after["u"].to_numpy(),
                 first_after["i"].to_numpy(), np.full(len(first_after), T))


t0 = time.time()
GLOBAL = global_temporal(ratings)
assert (GLOBAL.train["timestamp"] < GLOBAL.times[0]).all()  # nothing after T in training, for anyone
glob_rows = {}
GLOBAL_FNS = {"pop_global": popularity_fn(), REF: popularity_fn(BEST_DAYS)}
GLOBAL_FNS |= {key: model_fn(clone(m).fit(GLOBAL.X)) for key, m in TEST_MODELS.items()}
for name, fn in GLOBAL_FNS.items():
    rk = evaluate(GLOBAL, fn)["rank"]
    glob_rows[name] = recall_at_k(rk), ndcg_at_k(rk)
print(f"[{LABEL}] {GLOBAL}")
print(pd.DataFrame({k: {"Recall@10": fmt(bootstrap_ci(r)), "NDCG@10": fmt(bootstrap_ci(n))}
                    for k, (r, n) in glob_rows.items()}).T.to_string())
print(f"{time.time() - t0:.0f}s")

# %% [markdown]
# If the two protocols rank the models the same way, the global leakage left
# in leave-last-out is not changing conclusions. If they disagree, trust the
# global split for the *ordering* (it has no future at all) and leave-last-out
# for the *coverage of users*, and say so in the README. Expect wider intervals
# here: fewer users are active on both sides of *T*.
#
# **Robustness check 2: what a random split would have told you.** Hold out one
# random positive per user, train on everything else (future included), no
# existence filter, popularity over all training data. This is the protocol
# the brief tells you to show *once* and then never use.

# %%
t0 = time.time()
rng = np.random.default_rng(SEED)
pos_tbl = ratings[ratings["pos"] & ratings["u"].isin(TEST.users)]
held = pos_tbl.groupby("u").sample(1, random_state=SEED)
RANDOM = Stage("random holdout", ratings.drop(index=held.index), held["u"].to_numpy(), held["i"].to_numpy(),
               np.full(len(held), np.inf))
leak_rows = {"pop_global (time-aware, test)": ndcg_at_k(TEST_RES["pop_global"]["rank"]),
             "pop_global (random split)": ndcg_at_k(evaluate(RANDOM, popularity_fn())["rank"]),
             f"{BEST.key} (time-aware, test)": ndcg_at_k(TEST_RES[BEST.key]["rank"])}
leak_rows[f"{BEST.key} (random split)"] = ndcg_at_k(evaluate(RANDOM, model_fn(clone(BEST).fit(RANDOM.X)))["rank"])
# The second classic trap: popularity counted over ALL data, test period and targets included
all_counts = np.bincount(ratings.loc[ratings["pos"], "i"], minlength=N_ITEMS).astype(float)
leak_rows["pop_global (counted on all data, leaky)"] = ndcg_at_k(
    evaluate(TEST, lambda st, rows: np.tile(all_counts, (len(rows), 1)))["rank"])
print(f"[{LABEL}] NDCG@10 on the same users under different protocols")
for k_, v in leak_rows.items():
    print(f"  {k_:<48} {fmt(bootstrap_ci(v))}")
print(f"{time.time() - t0:.0f}s")

# %% [markdown]
# **Why the random split numbers are not comparable.** They answer an easier
# question: the model has seen what the user did *after* the held-out rating
# and what everyone else did later. Any gain over the time-aware numbers is
# measurement error that would disappear at launch. The "counted on all data"
# line is the most common real-world version of the same mistake: the
# popularity baseline includes the test targets in its own counts.
#
# *Common mistakes:* reporting the random-split number because it is larger;
# tuning on the global split after looking at the test; comparing models on
# different user sets.

# %% [markdown]
# ## Step 8 · Error and slice analysis
#
# **Goal.** Find where the model wins and where it fails: by user activity,
# for cold-start users, for long-tail and new movies, and under the alternative
# label definition. Every slice comes with its count, because a slice of 12
# users does not support a conclusion.
#
# **Your turn.** (45 minutes) Before running the cells, write down your
# hypothesis for each slice. Then compare.
#
# **Our solution — slice 1: user activity** (training positives at request time).

# %%
def slice_table(stage, results, groups, names, order):
    out = {}
    for g in [o for o in order if o in set(groups)]:
        mask = groups == g
        row = {"users": int(mask.sum())}
        for n in names:
            row[n] = fmt(bootstrap_ci(recall_at_k(results[n]["rank"][mask])))
        out[g] = row
    return pd.DataFrame(out).T


activity = pd.cut(TEST.n_train_pos, [0, 4, 19, 49, np.inf], labels=["01-04", "05-19", "20-49", "50+"]).astype(str)
print(f"[{LABEL}] Recall@10 by training positives (test users)")
print(slice_table(TEST, TEST_RES, activity, [REF, BEST.key], ["01-04", "05-19", "20-49", "50+"]).to_string())

# %% [markdown]
# **Slice 2: simulated cold-start users.** Releases where every user has at
# least 20 ratings contain no truly cold users, so we simulate them on the
# **validation** stage (selection data, so the threshold we choose is not
# tuned on test). For a random 20% of users we pretend we met them when they
# had exactly *k* positives: the request time becomes the time of their
# (*k*+1)-th positive, that positive is the target, and everything they did
# from then on is removed from training. Users with too few training positives
# are skipped for that *k*. Then we refit the selected model and compare it
# with recent popularity on exactly those users.

# %%
t0 = time.time()
rng = np.random.default_rng(SEED)
cold_users = rng.choice(VAL.users, max(1, len(VAL.users) // 5), replace=False)
cold_tbl = {}
for k_hist in (1, 3, 5, 10):
    tr = VAL.train
    nxt = tr[tr["pos"] & tr["u"].isin(cold_users)].groupby("u").nth(k_hist)  # the (k+1)-th positive
    cut = pd.Series(np.iinfo(np.int64).max, index=range(N_USERS))
    cut.loc[nxt["u"].to_numpy()] = nxt["seq"].to_numpy()
    truncated = tr[tr["seq"].to_numpy() < cut.loc[tr["u"]].to_numpy()]
    st = Stage(f"cold k={k_hist}", truncated, nxt["u"].to_numpy(), nxt["i"].to_numpy(), nxt["timestamp"].to_numpy())
    assert (st.n_train_pos == k_hist).all()
    mdl = clone(BEST).fit(st.X)
    cold_tbl[k_hist] = {"users": len(st.users),
                        REF: fmt(bootstrap_ci(ndcg_at_k(evaluate(st, popularity_fn(BEST_DAYS))["rank"]))),
                        BEST.key: fmt(bootstrap_ci(ndcg_at_k(evaluate(st, model_fn(mdl))["rank"])))}
print(f"[{LABEL}] NDCG@10 for the next positive of users who have exactly k positives (validation users)")
print(pd.DataFrame(cold_tbl).T.to_string())
print(f"{time.time() - t0:.0f}s")

# %% [markdown]
# Read the table from the top: the smallest *k* at which the model reliably
# beats popularity is the evidence for `MIN_HISTORY` (we started from the
# brief's 5). Below it, serve popularity and say so in the response
# (`"strategy": "popularity_fallback"`). A **brand-new** user (*k* = 0) has no
# row at all: only popularity, or an onboarding question ("pick three movies
# you like"), can serve them — and the ALS fold-in can use those three movies
# immediately.
#
# **Slice 3: cold and long-tail items.** Collaborative models cannot recommend
# a movie with no training positives. How often is the test target such a
# movie, and how does each model do on the tail?

# %%
train_cnt = np.asarray(TEST.X.sum(0)).ravel()[TEST.targets]
not_yet = FIRST_TS[TEST.targets] >= TEST.times
item_slice = np.where(not_yet, "not yet rated by anyone", pd.cut(train_cnt, [-1, 0, 4, 19, np.inf],
                      labels=["0 positives", "1-4", "5-19", "20+"]).astype(str))
print(f"[{LABEL}] Recall@10 by the target's training positives (test users)")
print(slice_table(TEST, TEST_RES, item_slice, [REF] + list(TEST_MODELS),
                  ["not yet rated by anyone", "0 positives", "1-4", "5-19", "20+"]).to_string())

# %% [markdown]
# A target with zero training positives is a guaranteed miss for every
# collaborative model, whatever its quality; only content features (genres,
# tags *applied before the request time*) can rank it. That is the brief's
# hybrid extension, and this slice is the number that justifies it — or does
# not, if the slice is tiny.
#
# **Slice 4: the alternative label ("any rating is an interaction").** Same
# protocol, every rating a positive, same hyperparameters (re-tuning would be
# the honest next step).

# %%
t0 = time.time()
any_df = index_ratings(ratings_raw, threshold=0.0)
ANY_TEST = leave_last_out(any_df)["test"]
mdl_any = clone(BEST).fit(ANY_TEST.X)
any_tbl = {}
for name, fn in {REF: popularity_fn(BEST_DAYS), BEST.key: model_fn(mdl_any)}.items():
    rk = evaluate(ANY_TEST, fn)["rank"]
    any_tbl[name] = {"Recall@10": fmt(bootstrap_ci(recall_at_k(rk))), "NDCG@10": fmt(bootstrap_ci(ndcg_at_k(rk)))}
print(f"[{LABEL}] any rating = interaction, test stage, {len(ANY_TEST.users)} users")
print(pd.DataFrame(any_tbl).T.to_string())
print(f"{time.time() - t0:.0f}s")

# %% [markdown]
# The two labels answer different questions — "what will they rate next?"
# versus "what will they rate next *and like*?" — so their numbers are not
# comparable with each other, only the model-versus-baseline comparison within
# each. If the model's advantage shrinks under "any rating", part of what it
# learned is taste, not just exposure; if it grows, the model is mostly
# predicting which movies people get around to rating.
#
# *Common mistakes:* reporting only the slice where the model wins; slices
# with a handful of users and no interval; concluding "the model is bad for
# cold users" when the slice is defined by the number of positives the model
# itself needs.

# %% [markdown]
# ## Step 9 · A small recommendation API
#
# **Goal.** Serve the selected model behind a FastAPI app factory with
# `/health`, `/ready`, `GET /recommendations/{user_id}`,
# `POST /recommendations` (request-time context) and `GET /similar/{movie_id}`,
# with the validation rules from the brief, the model version and the serving
# strategy in every response, and tests run in-process with `TestClient`
# ([Chapter 10](../../part1-foundations/ch10-fastapi-and-contracts/notebook.ipynb)).
#
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/cap4-serving.svg" alt="Serving design: a nightly batch job precomputes top-200 candidates per known user, item neighbours and recent popularity; at request time the API merges recent items, filters exclusions and already-rated movies, falls back to popularity for unknown or cold users, and returns the top k with the strategy used" width="100%">
#
# *What to notice:* the expensive work is precomputed; the request path only
# merges, filters and truncates, and always says which path served it.
#
# **Your turn.** (1 hour) Decide what is precomputed and what is computed per
# request, then write the response schema before the code.
#
# **Our solution.** The serving model is refitted on **all** ratings (trained
# through the last timestamp). A nightly job would produce the bundle below.

# %%
t0 = time.time()
FULL = Stage("serving", ratings, np.arange(N_USERS), np.zeros(N_USERS, int), np.full(N_USERS, np.inf))
serve_model = clone(BEST).fit(FULL.X)
knn_full = ItemKNN().fit(FULL.X)
trained_through = int(ratings["timestamp"].max())
TOP_N = 200


def precompute_topn(stage, model, n=TOP_N, batch=256):
    out = {}
    for s in range(0, N_USERS, batch):
        rows = np.arange(s, min(N_USERS, s + batch))
        S = model.score(stage.X[rows])
        for j, u in enumerate(rows):
            S[j, stage.seen[u]] = -np.inf
        top = np.argsort(-S, axis=1, kind="stable")[:, :n]
        for j, u in enumerate(rows):
            keep = top[j][np.isfinite(S[j, top[j]])]
            out[int(USER_IDS[u])] = [(int(ITEM_IDS[i]), float(S[j, i])) for i in keep]
    return out


# recent popularity as of the end of the data: the same scorer as the baseline, request time "now"
now = Stage.__new__(Stage)
now.times, now.pop_ts, now.pop_items = np.array([trained_through + 1.0]), FULL.pop_ts, FULL.pop_items
pop_now = popularity_fn(BEST_DAYS)(now, np.array([0]))[0]
POPULAR = [(int(ITEM_IDS[i]), float(pop_now[i])) for i in np.argsort(-pop_now, kind="stable")[:500]]
NEIGH = {}
for col, i in enumerate(knn_full.items):
    row = knn_full.S[col]
    nb = np.argsort(-row, kind="stable")[:20]
    NEIGH[int(ITEM_IDS[i])] = [(int(ITEM_IDS[knn_full.items[c]]), float(row[c])) for c in nb if row[c] > 0]

BUNDLE = {
    "model_version": f"cap4-{BEST.key}-{pd.to_datetime(trained_through, unit='s'):%Y%m%d}",
    "trained_through": pd.to_datetime(trained_through, unit="s", utc=True).isoformat(),
    "strategy_model": BEST.name,
    "topn": precompute_topn(FULL, serve_model),
    "seen": {int(USER_IDS[u]): set(ITEM_IDS[FULL.seen[u]].tolist()) for u in range(N_USERS)},
    "n_pos": {int(USER_IDS[u]): int(c) for u, c in enumerate(np.asarray(FULL.X.sum(1)).ravel())},
    "popular": POPULAR,
    "neighbours": NEIGH,
    "known_movies": set(ITEM_IDS.tolist()),
    "titles": dict(zip(ITEM_IDS.tolist(), TITLE.tolist())),
}
print(f"bundle {BUNDLE['model_version']}: {len(BUNDLE['topn']):,} users x top-{TOP_N}, "
      f"{len(NEIGH):,} item neighbour lists | {time.time() - t0:.0f}s")

# %%
from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RecRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: int = Field(gt=0)
    k: int = Field(10, ge=1, le=100)
    recent_movie_ids: list[int] = Field(default_factory=list, max_length=200)
    exclude_movie_ids: list[int] = Field(default_factory=list, max_length=1000)

    @field_validator("recent_movie_ids", "exclude_movie_ids")
    @classmethod
    def no_duplicates(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("duplicate movie ids")
        return v


class Item(BaseModel):
    movie_id: int
    title: str
    score: float
    rank: int


class RecResponse(BaseModel):
    model_version: str
    user_id: int
    strategy: str
    items: list[Item]
    ignored_movie_ids: list[int] = []


def rrf(*lists, c=60):
    """Reciprocal-rank fusion: merge ranked lists whose scores are not comparable."""
    fused = {}
    for lst in lists:
        for r, (mid, _) in enumerate(lst):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (c + r + 1)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def create_app(bundle: dict | None) -> FastAPI:
    """App factory. bundle=None simulates 'model not loaded yet' (ready -> 503)."""
    app = FastAPI(title="cap4 movie recommendations (non-commercial demo)")

    def require():
        if bundle is None:
            raise HTTPException(503, "model not loaded")
        return bundle

    def recommend(req: RecRequest) -> RecResponse:
        b = require()
        recent = [m for m in req.recent_movie_ids if m in b["known_movies"]]
        ignored = [m for m in req.recent_movie_ids if m not in b["known_movies"]]
        blocked = set(req.exclude_movie_ids) | set(recent) | b["seen"].get(req.user_id, set())
        neighbours = rrf(*[b["neighbours"].get(m, []) for m in recent]) if recent else []
        if b["n_pos"].get(req.user_id, 0) >= MIN_HISTORY:
            base, strategy = b["topn"][req.user_id], b["strategy_model"]
        else:
            base, strategy = b["popular"], "popularity_fallback"
        if neighbours:
            ranked, strategy = rrf(base, neighbours), strategy + "+recent_items"
        else:
            ranked = base
        items = [(m, s) for m, s in ranked if m not in blocked][: req.k]
        return RecResponse(model_version=b["model_version"], user_id=req.user_id, strategy=strategy,
                           ignored_movie_ids=ignored,
                           items=[Item(movie_id=m, title=b["titles"].get(m, "?"), score=round(s, 6), rank=r + 1)
                                  for r, (m, s) in enumerate(items)])

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        b = require()
        return {"ready": True, "model_version": b["model_version"], "trained_through": b["trained_through"]}

    @app.get("/recommendations/{user_id}", response_model=RecResponse)
    def get_recs(user_id: int = PathParam(gt=0), k: int = Query(10, ge=1, le=100)):
        return recommend(RecRequest(user_id=user_id, k=k))

    @app.post("/recommendations", response_model=RecResponse)
    def post_recs(req: RecRequest):
        return recommend(req)

    @app.get("/similar/{movie_id}")
    def similar(movie_id: int = PathParam(gt=0), k: int = Query(10, ge=1, le=100)):
        b = require()
        if movie_id not in b["neighbours"]:
            raise HTTPException(404, "unknown movie or no neighbours")
        return {"model_version": b["model_version"], "movie_id": movie_id,
                "items": [{"movie_id": m, "score": round(s, 6), "rank": r + 1}
                          for r, (m, s) in enumerate(b["neighbours"][movie_id][:k])]}

    return app

# %% [markdown]
# **Check.** The tests a reviewer would run: liveness without a model,
# not-ready state, a known user served by the model, an unknown user served by
# the fallback, exclusions and already-rated movies never returned, unknown
# recent IDs ignored and reported, each validation rule rejected with 422, and
# the model version in every response.

# %%
client = TestClient(create_app(BUNDLE))
not_ready = TestClient(create_app(None))
heavy = int(USER_IDS[np.argmax(np.asarray(FULL.X.sum(1)).ravel())])

assert not_ready.get("/health").json() == {"status": "ok"}
assert not_ready.get("/ready").status_code == 503
assert not_ready.get(f"/recommendations/{heavy}").status_code == 503
assert client.get("/ready").json()["model_version"] == BUNDLE["model_version"]

r1 = client.get(f"/recommendations/{heavy}", params={"k": 10}).json()
assert r1["strategy"] == BEST.name and len(r1["items"]) == 10 and r1["model_version"] == BUNDLE["model_version"]
assert not {it["movie_id"] for it in r1["items"]} & BUNDLE["seen"][heavy], "already-rated movie served"
assert [it["rank"] for it in r1["items"]] == list(range(1, 11))

unknown_user = int(USER_IDS.max()) + 1000
r2 = client.get(f"/recommendations/{unknown_user}").json()
assert r2["strategy"] == "popularity_fallback" and len(r2["items"]) == 10

top_movie = r1["items"][0]["movie_id"]
recent = [BUNDLE["popular"][0][0]]  # the most popular movie: it certainly has neighbours
body = {"user_id": heavy, "k": 5, "recent_movie_ids": recent + [999_999_999], "exclude_movie_ids": [top_movie]}
r3 = client.post("/recommendations", json=body).json()
assert top_movie not in {it["movie_id"] for it in r3["items"]} and recent[0] not in {it["movie_id"] for it in r3["items"]}
assert r3["ignored_movie_ids"] == [999_999_999] and r3["strategy"].endswith("+recent_items") and len(r3["items"]) == 5

for bad in ({"user_id": 0}, {"user_id": heavy, "k": 0}, {"user_id": heavy, "k": 101},
            {"user_id": heavy, "recent_movie_ids": [1, 1]}, {"user_id": heavy, "recent_movie_ids": list(range(201))},
            {"user_id": heavy, "surprise": 1}):
    assert client.post("/recommendations", json=bad).status_code == 422, bad
assert client.get("/recommendations/-3").status_code == 422
assert client.get(f"/similar/{recent[0]}", params={"k": 3}).status_code == 200
assert client.get("/similar/999999999").status_code == 404
print("API tests passed. Example slate for user", heavy, f"[{LABEL}]:")
for it in r1["items"][:5]:
    print(f"  {it['rank']:>2}. {it['title']}")

# %% [markdown]
# **Why this design.** Precomputing top-200 candidates per known user keeps the
# request path cheap and predictable; the request-time part (recent movies,
# exclusions) is merged with reciprocal-rank fusion because the scores of two
# different models are not on the same scale. The already-rated filter is
# applied **in serving with the same rule as in evaluation** — filtering only
# in evaluation would make offline numbers describe a system you do not run.
# Freshness is the price: a rating made today changes the precomputed list only
# after tonight's batch, which is why the request may carry `recent_movie_ids`.
#
# *Common mistakes:* one endpoint that silently returns popularity for unknown
# users without saying so; `/health` that fails when the model is missing (then
# the orchestrator restarts a healthy process); validating only types, not
# ranges and duplicates.

# %% [markdown]
# ## Step 10 · Write-up: results table, limitations, next steps
#
# **Goal.** Produce the README's results table **from the computed numbers**
# (never typed by hand), with the split and the release named, and write the
# limitations a sceptical reviewer would agree with.
#
# **Our solution.** The cell below builds the markdown. On a real run it is
# also saved to the artifact directory for you to paste into the README; in
# sample mode it is printed with a warning and saved nowhere.

# %%
def results_markdown() -> str:
    lines = [("Data: SYNTHETIC SAMPLE (not MovieLens)" if SAMPLE else f"Release: `{RELEASE}`") + (f", random sample of {USER_SAMPLE} users (seed {SEED})" if USER_SAMPLE else "")
             + f"; label: rating >= {POS_THRESHOLD}; split: time-aware leave-last-out (test = each user's last "
             f"positive), {len(TEST.users)} users; full ranking over {N_ITEMS:,} movies; 95% bootstrap CIs over users.",
             "", "| Model | Recall@10 | NDCG@10 | NDCG@10 − recent popularity | Coverage | Mean popularity |",
             "|---|---|---|---|---:|---:|"]
    for name, r in TEST_TABLE.iterrows():
        lines.append(f"| {name} | {r['Recall@10']} | {r['NDCG@10']} | {r['NDCG diff vs recent pop']} | "
                     f"{r['coverage']} | {r['mean popularity']} |")
    lines += ["", f"Global temporal split ({GLOBAL.name}, {len(GLOBAL.users)} users), NDCG@10:", ""]
    lines += [f"* {k}: {fmt(bootstrap_ci(n))}" for k, (_, n) in glob_rows.items()]
    lines += ["", "Random split vs time-aware (same users), NDCG@10:", ""]
    lines += [f"* {k}: {fmt(bootstrap_ci(v))}" for k, v in leak_rows.items()]
    return "\n".join(lines)


md = results_markdown()
if SAMPLE:
    print("=" * 78 + "\nSYNTHETIC SAMPLE: the table below describes invented data. Do NOT copy it into a README.\n" + "=" * 78)
else:
    (CAP / "results_table.md").write_text(md + "\n", encoding="utf-8")
    print("saved", CAP / "results_table.md", "- paste it into README.md under 'Results'")
print(md)
print(f"\ntotal runtime: {(time.time() - T_START) / 60:.1f} min")

# %% [markdown]
# **Limitations** (specific to this data and protocol):
#
# * **Ratings are not consumption.** A rating is an explicit act by a subset of
#   viewers, often long after watching. Recall here predicts "what will they
#   rate (and like) next", not "what will they watch".
# * **Onboarding bursts.** Step 3 measured how many ratings fall within a day
#   of the first one; for those, "the next positive" is partly an artefact of
#   the MovieLens sign-up flow.
# * **Selection.** The release keeps only users with at least 20 ratings, so
#   genuinely cold users are absent; step 8 simulated them, which is not the
#   same as observing them.
# * **Global leakage in leave-last-out.** Step 4 measured the share of training
#   ratings that come after a user's request time; the global temporal split is
#   the check, with fewer users and wider intervals.
# * **Offline is not online.** Position bias, popularity feedback loops and
#   novelty are not captured; a launch decision needs an A/B test
#   ([Chapter 29](../../part2-ml-at-scale/ch29-ab-testing-fundamentals/notebook.ipynb)).
# * **ml-latest-small changes over time** and its README says it is not for
#   shared research results: report final numbers on a fixed release (for
#   example ml-32m with a documented user sample), with its checksum.
# * **Licence.** Research, non-commercial use only; the API is a demonstration.
#
# **What to do next (extensions).**
#
# 1. Re-run on ml-32m: set `RELEASE = "ml-32m"`, `USER_SAMPLE = 6000`, keep
#    `MODEL_MIN_ITEM_POS = 5`, and check the EASE memory line in step 6 before
#    fitting. (Not executed for this walkthrough; budget 10–20 minutes on
#    Colab, mostly reading the CSV.)
# 2. A LightGBM re-ranker over the top-200 ALS/EASE candidates, with features
#    computed as of request time (model score, recent popularity, genre match
#    with the user's history, movie age) — [Chapter 24](../../part2-ml-at-scale/ch24-learning-to-rank/notebook.ipynb).
# 3. Genre and tag content similarity for the cold-item slice, with tags
#    restricted to those applied before the request time.
# 4. Diversity or novelty re-ranking, reported as a trade-off curve against
#    NDCG@10 and coverage.
# 5. Version the model and its precomputed tables together, and monitor the
#    share of fallback responses (the brief's churnlab stretch).

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `URLError` / TLS error in step 2 | no access to `files.grouplens.org` from this machine | run in Colab; do not switch to a mirror (the licence forbids redistribution) |
# | `MemoryError` in EASE or item-kNN | too many items for a dense items × items matrix | raise `MODEL_MIN_ITEM_POS` or use `USER_SAMPLE`; check the printed size first |
# | Every model has Recall@10 near 0 | the "already rated" filter includes the target, or IDs were not mapped with `searchsorted` | run the step 4 and step 5 checks |
# | Random-split numbers far above time-aware ones | that is the leakage the brief asks you to show | report it once, do not use it for decisions |
# | ALS barely beats popularity | too little regularisation or too few iterations; or the data really favour popularity | tune `reg`, `alpha` on validation; report the negative result |
#
# ## Checkpoint
#
# Answer aloud in about two minutes each, using your own numbers:
#
# 1. Why Recall@10 and NDCG@10 instead of RMSE?
# 2. Describe your split. How could it still leak, and how did you measure it?
# 3. By how much did you beat recent popularity, and is the difference distinguishable from zero?
# 4. How does your system handle a brand-new user? A brand-new movie?
# 5. Explain EASE or ALS in two minutes. When would it fail?
# 6. What does the MovieLens licence let you do with this project?
