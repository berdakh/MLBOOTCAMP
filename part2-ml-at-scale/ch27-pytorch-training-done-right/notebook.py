# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part2-ml-at-scale/ch27-pytorch-training-done-right/notebook.ipynb)
#
# *Runs in a fresh Google Colab CPU runtime: click the badge, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 27 · PyTorch training done right
#
# **Day 5 · ~90 minutes · CPU (a GPU only speeds up the optional parts)**
#
# You have trained PyTorch models before; this notebook is about the parts
# that decide whether a training run can be *trusted, repeated and resumed*
# by someone else, which is what reviewers and interviewers probe. The running
# example is a click model on ShopStream impressions: small enough to train in
# seconds on a CPU, real enough to show every failure.
#
# By the end you will be able to:
#
# 1. choose between map-style and iterable datasets, write a collate function, and reason about `num_workers` and `pin_memory`;
# 2. write the training loop by hand and say exactly where `train()`/`eval()`, `no_grad` and `zero_grad` go and why;
# 3. predict the loss at initialisation and use it, plus "overfit one batch" and gradient norms, to debug a model that won't learn;
# 4. make two runs produce **bitwise-identical** loss curves, and prove a resumed run equals an uninterrupted one;
# 5. use warmup + cosine learning-rate schedules and gradient clipping;
# 6. find planted bugs in a training script from its symptoms.
#
# **You will produce:** `artifacts/level2/nb09/click_mlp.pt` (a full checkpoint:
# model, optimizer, scheduler and RNG states) and the small helper module
# [`src/shoprank/torchkit.py`](../../src/shoprank/torchkit.py), which later
# notebooks reuse.

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
# --- Catch up: ShopStream (rebuilt in ~30 s if this runtime has never seen it)
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from shoprank import store
from shoprank import torchkit as tk
from grading import ch27 as grade

torch.set_num_threads(2)  # Colab CPU runtimes have 2 cores; more threads only add contention
OUT = ART / "level2" / "nb09"
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
world, users, items, imps = store.load("small")
print(f"{len(imps):,} impressions loaded in {time.time() - t0:.0f} s | torch {torch.__version__}")

# %% [markdown]
# **Expected:** about 2.45 million impressions; 10-60 s the first time (the
# data is generated), a few seconds after that.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | **Map-style dataset** | has `__len__` and `__getitem__(i)`; the sampler picks indices | `RowDataset`, `tk.TensorBatchDataset` |
# | **IterableDataset** | only `__iter__`; it decides the order itself | `tk.ParquetDayStream` |
# | **Collate function** | stacks individual samples into one batch | the default `collate_fn` for per-row datasets |
# | **`num_workers`** | separate processes that load batches in parallel | 0 vs 2 on a 2-core runtime |
# | **Logit** | the raw score before the sigmoid or softmax | the click model's single output |
# | **`BCEWithLogitsLoss`** | binary cross-entropy that applies the sigmoid itself, stably | the click model's loss |
# | **`train()` / `eval()`** | switch layer behaviour (dropout, batch norm) | (3) and (2) in the loop |
# | **`no_grad()`** | stop recording the graph for backprop | (1) around validation |
# | **Gradient clipping** | rescale the gradient when its norm exceeds a threshold | `clip_grad_norm_(…, 1.0)` |
# | **Warmup + cosine** | ramp the learning rate up linearly, then decay it along a cosine | `tk.warmup_cosine` |
# | **Checkpoint** | everything needed to continue a run exactly | `click_mlp.pt`: model, optimizer, scheduler, RNG |
# | **Bitwise reproducible** | two runs give identical floats, not just close ones | `==` on the loss lists in 9.5 |

# %% [markdown]
# ## 9.1 · From a log to tensors
#
# The click model predicts `P(click)` for one impression from four inputs:
# the user, the item, the item's category and the **position** it was shown at.
# Position is a feature during training so that the model can explain away
# position bias (notebook 05); at serving time you would fix it to one value.
#
# We train on the last two training weeks (days 28–41) and validate on the
# validation week (42–48), subsampled so every cell runs in seconds. The split is by
# **time**, as in every Level 2 notebook; section 9.9 shows what goes wrong otherwise.

# %%
cat_of_item = torch.as_tensor(items["category"].to_numpy(), dtype=torch.long)


def to_tensors(df: pd.DataFrame):
    """(user, item, position, clicked) as tensors - the column order is the contract."""
    return (torch.as_tensor(df["user_id"].to_numpy(), dtype=torch.long),
            torch.as_tensor(df["item_id"].to_numpy(), dtype=torch.long),
            torch.as_tensor(df["position"].to_numpy(), dtype=torch.long),
            torch.as_tensor(df["clicked"].to_numpy(), dtype=torch.float32))


cols = ["day", "user_id", "item_id", "position", "clicked"]
train_df = imps.loc[imps["day"].between(28, 41), cols].sample(n=240_000, random_state=0).sort_index()
valid_df = imps.loc[imps["day"].between(42, 48), cols].sample(n=60_000, random_state=0).sort_index()
train_t, valid_t = to_tensors(train_df), to_tensors(valid_df)
POS_RATE = float(train_t[3].mean())
print(f"train {len(train_df):,} rows (days 28-41) | valid {len(valid_df):,} rows (days 42-48)"
      f" | click rate {POS_RATE:.3f}")

# %% [markdown]
# **Expected:** 240,000 training and 60,000 validation rows; a click rate of about 0.137.
#
# ## 9.2 · Datasets and DataLoaders
#
# A **map-style** `Dataset` has `__len__` and `__getitem__(i)`: the loader's
# *sampler* chooses indices (shuffled or not), workers fetch them, and a
# **collate function** stacks the samples into a batch. An **iterable**
# `IterableDataset` has only `__iter__`: it decides the order itself, which
# suits streams that do not fit in memory (Parquet files, a message queue),
# but shuffling and sharding across workers become *your* job.
#
# The textbook map-style dataset returns one row at a time:

# %%
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler


class RowDataset(Dataset):
    """One impression per __getitem__ call - the textbook way."""

    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, i):
        return tuple(t[i] for t in self.tensors)


def dict_collate(samples):
    """A custom collate_fn: list of (u, i, p, y) tuples -> one dict of stacked tensors."""
    u, i, p, y = zip(*samples)
    return {"user": torch.stack(u), "item": torch.stack(i), "pos": torch.stack(p), "y": torch.stack(y)}


def time_one_pass(loader, max_batches=None):
    start, n = time.perf_counter(), 0
    for b, batch in enumerate(loader):
        n += len(batch["y"] if isinstance(batch, dict) else batch[-1])
        if max_batches and b + 1 >= max_batches:
            break
    return n, time.perf_counter() - start


subset = tuple(t[:50_000] for t in train_t)
n, sec_rows = time_one_pass(DataLoader(RowDataset(subset), batch_size=1024, shuffle=True, collate_fn=dict_collate))
print(f"per-row Dataset    : {n:,} rows in {sec_rows:.2f} s")

# Same data, but __getitem__ receives a whole list of indices (one call per batch).
batch_ds = tk.TensorBatchDataset(*subset)
sampler = BatchSampler(RandomSampler(batch_ds), batch_size=1024, drop_last=False)
n, sec_batch = time_one_pass(DataLoader(batch_ds, sampler=sampler, batch_size=None))
print(f"per-batch Dataset  : {n:,} rows in {sec_batch:.3f} s  ({sec_rows / sec_batch:.0f}x faster)")

# %% [markdown]
# **Expected:** the per-row dataset takes about 0.5–0.8 s for 50,000 rows; the
# per-batch one about 0.03 s, 15–25x faster in our runs.
# For small tabular rows the Python overhead of one `__getitem__` + collate per row
# *is* the cost. For images, decoding dominates instead and per-row is fine.
#
# `batch_size=None` turns off the loader's own batching: the sampler already
# yields lists of indices. This pattern is what you want for tabular and
# embedding-lookup models.
#
# **`num_workers`.** Workers are separate *processes* that run
# `__getitem__` + collate in parallel and pass batches back through shared
# memory. They help when loading is CPU-heavy (decoding JPEGs, tokenising text)
# and you have spare cores. They cost process start-up per epoch (unless
# `persistent_workers=True`), memory (each worker holds a copy of the dataset
# object), and inter-process transfer. A free Colab runtime has **2 CPU
# cores**, so `num_workers=2` is the most that can help there. Measure, don't guess:

# %%
for workers in (0, 2):
    loader = DataLoader(RowDataset(subset), batch_size=1024, shuffle=True, collate_fn=dict_collate,
                        num_workers=workers, persistent_workers=False)
    n, sec = time_one_pass(loader)
    print(f"per-row Dataset, num_workers={workers}: {sec:.2f} s")

# %% [markdown]
# **Expected:** on a 2-core machine the two times are of the same order (0.5–2 s); two
# workers may be slightly faster or slower, depending on how much of the cost is
# worker start-up and inter-process copying. The batched dataset with
# `num_workers=0` beats both by a wide margin, so fix the data path before
# adding processes.
#
# **`pin_memory=True`** allocates batches in page-locked host memory so that
# `.to("cuda", non_blocking=True)` can overlap the copy with compute. It only
# matters with a GPU; on a CPU-only runtime it does nothing useful.
#
# **Iterable datasets and workers.** With an `IterableDataset`, *every*
# worker runs `__iter__` from the start. Unless you shard, you get each row
# `num_workers` times. ShopStream's impressions are stored as Parquet part files;
# `tk.ParquetDayStream` streams them and shards files across workers by
# `get_worker_info()`:

# %%
files = sorted((store.dataset_dir("small") / "impressions").rglob("*.parquet"))
for shard in (False, True):
    stream = tk.ParquetDayStream(files, columns=["clicked"], batch_rows=50_000, shard=shard)
    rows = sum(len(b["clicked"]) for b in DataLoader(stream, batch_size=None, num_workers=2))
    print(f"shard={shard!s:5}: {rows:,} rows streamed with 2 workers (the log has {len(imps):,})")

# %% [markdown]
# **Expected:** `shard=False` streams every row twice (about 4.9 million rows);
# `shard=True` streams exactly the 2,446,900-or-so rows of the log. Nothing
# raises in the broken case: the only symptom is an epoch that is
# suspiciously long and a model that has seen duplicates.
#
# ## 9.3 · The model and the loop, by hand
#
# The click model: embeddings for user, item, category and position, concatenated,
# then a small MLP with dropout. It outputs a **logit**; the loss is
# `BCEWithLogitsLoss`, which applies the sigmoid inside in a numerically stable way.

# %%


class ClickMLP(nn.Module):
    def __init__(self, n_users, n_items, n_cats, n_pos=10, dim=16, hidden=64, dropout=0.2):
        super().__init__()
        self.user = nn.Embedding(n_users, dim)
        self.item = nn.Embedding(n_items, dim)
        self.cat = nn.Embedding(n_cats, 8)
        self.pos = nn.Embedding(n_pos, 4)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim + 8 + 4, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, user, item, pos):
        x = torch.cat([self.user(user), self.item(item), self.cat(cat_of_item[item]), self.pos(pos)], dim=1)
        return self.mlp(x).squeeze(1)  # logits, shape (batch,)


def new_model(seed=0, bias_init=True):
    torch.manual_seed(seed)
    model = ClickMLP(len(users), len(items), int(items["category"].max()) + 1)
    if bias_init:  # see 9.4: start at the base rate, not at 50%
        with torch.no_grad():
            model.mlp[-1].bias.fill_(float(np.log(POS_RATE / (1 - POS_RATE))))
    return model


def make_loader(tensors, batch_size=2048, shuffle=True, generator=None):
    ds = tk.TensorBatchDataset(*tensors)
    base = RandomSampler(ds, generator=generator) if shuffle else range(len(ds))
    return DataLoader(ds, sampler=BatchSampler(base, batch_size, drop_last=False), batch_size=None)


model = new_model()
print(model)
print(f"{tk.count_parameters(model):,} trainable parameters")

# %% [markdown]
# Now the loop, written out once so every line can be questioned:

# %%
from sklearn.metrics import roc_auc_score

loss_fn = nn.BCEWithLogitsLoss()


@torch.no_grad()  # (1) no autograd graph in evaluation: less memory, faster
def predict(model, tensors, batch_size=8192):
    model.eval()  # (2) dropout off (and BatchNorm would use running stats)
    out = [model(*(t[i:i + batch_size] for t in tensors[:3])) for i in range(0, len(tensors[0]), batch_size)]
    return torch.cat(out)


def valid_report(model):
    logits = predict(model, valid_t)
    return {"valid_loss": round(loss_fn(logits, valid_t[3]).item(), 4),
            "valid_auc": round(float(roc_auc_score(valid_t[3].numpy(), logits.numpy())), 4)}


model = new_model()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
train_loader = make_loader(train_t, generator=tk.make_generator(0))
start = time.time()
for epoch in range(3):
    model.train()  # (3) EVERY epoch: predict() above left the model in eval mode
    total, count = 0.0, 0
    for user, item, pos, y in train_loader:
        optimizer.zero_grad(set_to_none=True)  # (4) clear last step's grads BEFORE backward
        loss = loss_fn(model(user, item, pos), y)
        loss.backward()  # (5) accumulate d loss / d params into .grad
        optimizer.step()  # (6) update using .grad
        total += loss.item() * len(y)  # (7) .item(): a float, not a tensor holding the graph
        count += len(y)
    print(f"epoch {epoch}: train_loss {total / count:.4f} | {valid_report(model)}")
print(f"3 epochs in {time.time() - start:.1f} s")

# %% [markdown]
# **Expected:** training loss falls from about 0.357 to about 0.336; validation
# loss about 0.335 and AUC about 0.76 after three epochs, about a second per epoch.
# The training loss in epoch 0 is an average over a model that was changing,
# and computed *with dropout on*, which is why train and valid
# losses are not directly comparable.
#
# The numbered comments are the interview questions:
#
# * **(3) `model.train()` inside the epoch loop.** Evaluation switched the model
#   to eval mode; forgetting to switch back silently trains without dropout
#   (and, with BatchNorm, with frozen statistics).
# * **(4) `zero_grad` placement.** Anywhere between the previous `step()` and
#   this `backward()` is correct. Omit it and gradients add up across steps.
#   (Sometimes that is deliberate: *gradient accumulation* over several micro-batches.)
#   `set_to_none=True` frees the memory instead of filling it with zeros.
# * **(1) `no_grad` vs (2) `eval`.** They are independent: `eval()` changes
#   layer *behaviour*; `no_grad()` stops *recording* the graph. Evaluation needs both.
# * **(7) `loss.item()`.** Summing `loss` tensors keeps every step's graph
#   alive: the classic slow memory leak.

# %% [markdown]
# > **Intuition.** `model.eval()` puts the car in parking mode (it changes how the car *behaves*: dropout off, batch-norm statistics frozen); `torch.no_grad()` switches off the dashcam (nothing is *recorded* for a later replay, i.e. backprop). Validation wants both: parked, and not recording. *Where it breaks down:* a parked car cannot move, but a model in eval mode can still be trained, silently and wrongly, if you forget to switch back.

# %% [markdown]
# > **Exercise 1.** Package the loop as `train_one_epoch(model, loader, optimizer, loss_fn)`.
# > The loader yields `(*inputs, y)` tuples; call `model(*inputs)`. Put the model in training
# > mode, do one optimizer step per batch, and return the **mean loss per example** over the
# > epoch as a Python float (weight each batch's mean loss by its size). The grader trains a
# > tiny linear model and compares your weights, step for step, with a reference loop.

# %%
def train_one_epoch(model, loader, optimizer, loss_fn):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(train_one_epoch)

# %% [markdown]
# `tk.train_epoch` is the same loop with three additions you will use below:
# an optional per-step scheduler, gradient clipping, and a per-step log of
# the loss, gradient norm and learning rate. Read it in
# [`src/shoprank/torchkit.py`](../../src/shoprank/torchkit.py).
#
# ## 9.4 · Know the loss before you train
#
# Before the first step, a sensibly initialised model should predict roughly
# the base rate for everyone. For binary cross-entropy with a positive rate
# `p`, the best *constant* prediction is `p` itself, and its loss is the
# entropy `-(p log p + (1-p) log(1-p))`. For `C` balanced classes with
# softmax cross-entropy, it is `log C`. A loss at step 0 far from this number
# means a bug (wrong labels, wrong loss, logits on the wrong scale)
# *before* any training has happened.

# %%
first = next(iter(make_loader(train_t, generator=tk.make_generator(0))))
for bias_init in (False, True):
    with torch.no_grad():
        m = new_model(bias_init=bias_init).eval()
        init_loss = loss_fn(m(*first[:3]), first[3]).item()
    print(f"bias_init={bias_init!s:5}: loss at step 0 = {init_loss:.4f}")
p = POS_RATE
print(f"log(2) = {np.log(2):.4f} | base-rate entropy H({p:.3f}) = {-(p * np.log(p) + (1 - p) * np.log(1 - p)):.4f}")

# %% [markdown]
# **Expected:** about 0.6–0.7 (close to log 2: random small logits put every
# prediction near 50%) without the bias initialisation, and about 0.40 with it,
# matching the entropy line. A model that starts near 0.69 wastes its
# first hundreds of steps just learning the base rate. With rare positives
# (ads CTR at 1%) the gap is large (0.69 vs 0.056) and that early
# "hockey stick" in the loss curve hides everything else.

# %% [markdown]
# > **Exercise 2.** Write `expected_initial_loss(pos_rate)`: the binary cross-entropy (natural log)
# > of the best constant prediction for a dataset with positive rate `pos_rate`, i.e. what a
# > correctly initialised model should show at step 0. Handle `pos_rate` of exactly 0 or 1.

# %%
def expected_initial_loss(pos_rate):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(expected_initial_loss)

# %% [markdown]
# ## 9.5 · Reproducibility: bitwise-identical runs
#
# A training run draws random numbers in four places: weight initialisation,
# data order, dropout masks, and anything in data-loading workers. To repeat it
# exactly:
#
# | Source | Control |
# |---|---|
# | init, dropout (torch global RNG) | `torch.manual_seed(seed)` before building the model |
# | data order | a **dedicated** `torch.Generator` passed to the sampler |
# | NumPy / `random` inside workers | `worker_init_fn=tk.seed_worker`, plus `generator=` on the DataLoader |
# | non-deterministic kernels (mostly GPU) | `torch.use_deterministic_algorithms(True)`, cuDNN flags (`tk.set_deterministic`) |
#
# Here is one complete training run as a function, then two calls with the same seed:

# %%
def run(seed, epochs=2, extra_draw=False):
    tk.seed_everything(seed)
    model = new_model(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loader = make_loader(train_t, generator=tk.make_generator(seed))
    log = []
    for epoch in range(epochs):
        if extra_draw and epoch > 0:
            torch.rand(1)  # e.g. a debugging line someone added that draws from the global RNG
        tk.train_epoch(model, loader, optimizer, loss_fn, log=log)
    return [row["loss"] for row in log], tk.state_fingerprint(model)


tk.set_deterministic(True)
start = time.time()
losses_a, fp_a = run(seed=1)
losses_b, fp_b = run(seed=1)
losses_c, fp_c = run(seed=2)
print(f"3 runs in {time.time() - start:.1f} s | {len(losses_a)} steps each")
print("same seed, identical loss curves (exact ==):", losses_a == losses_b, "| weights:", fp_a == fp_b)
print("different seed                             :", losses_a == losses_c, "| weights:", fp_a == fp_c)
diff = np.abs(np.array(losses_a) - np.array(losses_c))
print(f"max |loss difference| between seeds 1 and 2: {diff.max():.4f}")

# %% [markdown]
# **Expected:** `True | weights: True` for the same seed, `False | weights:
# False` for a different one, with per-step losses differing by up to about 0.1
# (different batches, different initial weights).
# The equality test is `==` on Python floats: **bitwise**, not "close enough".
#
# Why the dedicated generator matters: the next run adds a single extra draw from the *global*
# RNG between epochs (the sort of line someone adds while debugging).

# %%
losses_d, fp_d = run(seed=1, extra_draw=True)
n_steps = len(losses_a) // 2
print("epoch 1 identical:", losses_a[:n_steps] == losses_d[:n_steps],
      "| epoch 2 identical:", losses_a[n_steps:] == losses_d[n_steps:])

# %% [markdown]
# **Expected:** epoch 1 identical `True`, epoch 2 `False`. The data order is still
# the same (it comes from the dedicated generator), but the dropout masks in
# epoch 2 shifted because they share the global RNG with that extra draw. Had
# the sampler used the global RNG too, the data order would have changed as well.
# Exact reproducibility is fragile by nature: it holds for *this code, these
# library versions, this hardware*. Across a GPU model or a PyTorch upgrade,
# expect "statistically the same", which is why you report the spread over seeds.

# %% [markdown]
# ## 9.6 · Learning-rate schedules and gradient clipping
#
# **Warmup + cosine.** Adam's second-moment estimates are poor in the first
# steps, so large early updates can push embeddings or attention weights somewhere bad;
# a short linear **warmup** avoids that. A **cosine decay** afterwards lowers the
# step size so the last epochs refine rather than bounce. Schedules in PyTorch are
# stepped either per epoch or per optimizer step; `tk.warmup_cosine` is per **step**.
#
# > **Intuition.** Warmup is not flooring the accelerator on a cold engine: Adam's running averages start empty, so its first few step sizes are guesses. Cosine decay is easing off the pedal as you approach the parking space. *Where it breaks down:* an engine warms up in minutes regardless; how many warmup steps a model needs depends on the batch size and the optimiser, so treat it as a hyper-parameter, not a constant.
#
# **Gradient clipping** rescales the gradient when its global norm exceeds a
# threshold. It protects against rare huge gradients (a bad batch, a spike in
# a sequence model) and is cheap. `clip_grad_norm_` returns the norm *before*
# clipping, which is the number to monitor.

# %%
import matplotlib.pyplot as plt

EPOCHS = 4
steps_per_epoch = len(make_loader(train_t))
total_steps = EPOCHS * steps_per_epoch


def run_schedule(schedule: bool, clip=None, lr=3e-3, seed=0, epochs=EPOCHS):
    tk.seed_everything(seed)
    model = new_model(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sched = tk.warmup_cosine(optimizer, warmup_steps=steps_per_epoch // 2, total_steps=total_steps) if schedule else None
    loader = make_loader(train_t, generator=tk.make_generator(seed))
    log = []
    for _ in range(epochs):
        tk.train_epoch(model, loader, optimizer, loss_fn, scheduler=sched, clip=clip, log=log)
    return model, pd.DataFrame(log)


start = time.time()
results = {}
for name, kwargs in {"constant lr 3e-3": dict(schedule=False),
                     "warmup+cosine, clip 1.0": dict(schedule=True, clip=1.0)}.items():
    model, log = run_schedule(**kwargs)
    results[name] = (log, valid_report(model))
    print(f"{name:24s} -> {results[name][1]}")
print(f"{steps_per_epoch} steps/epoch, {time.time() - start:.1f} s")

fig, axes = plt.subplots(1, 3, figsize=(13, 3))
for name, (log, _) in results.items():
    axes[0].plot(log["lr"], label=name)
    axes[1].plot(log["loss"].rolling(10).mean(), label=name)
    axes[2].plot(log["grad_norm"], label=name, alpha=0.7)
axes[0].set_title("learning rate per step")
axes[1].set_title("train loss (10-step mean)")
axes[2].set_title("gradient norm before clipping")
axes[2].set_yscale("log")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Expected:** both runs reach a validation AUC around 0.76. In our runs the schedule gave
# a slightly lower validation loss than the constant rate (0.335 vs 0.336); on this small model the effect is modest, so do not over-read
# one run: compare over seeds before claiming a gain. The plots show the
# warmup ramp and the cosine tail, and gradient norms mostly below 1 with
# occasional larger steps, which clipping at 1.0 caps.
#
# ## 9.7 · Checkpoint and resume, proven
#
# Colab disconnects; spot instances get pre-empted; long jobs crash at hour 11.
# A checkpoint that holds only the weights is **not** enough to continue the
# same run: Adam's moment estimates, the scheduler's position and every RNG
# state (including the shuffle generator) are part of the training state.
# The test of a checkpoint is simple: an interrupted-and-resumed run must equal an
# uninterrupted one, bitwise.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-training-loop-checkpoint.svg" alt="The PyTorch training step, and a full checkpoint versus a weights-only one" width="100%">
#
# *What to notice:* the loop at the top has state in four places (weights, optimizer moments, scheduler position, random generators). The green box saves all four; the red box saves one, which is why its curve departs below.
#
# > **Intuition.** A checkpoint is a saved game: you need not only where your character stands (the weights) but the inventory (Adam's moments), the quest log (the scheduler's step) and the dice (every random generator). Load only the position and the game continues, just not *your* game. *Where it breaks down:* a game reloads instantly; a checkpoint is worth saving only as often as losing the work in between would hurt.

# %%
CKPT = OUT / "click_mlp.pt"


def build(seed):
    """Fresh model, optimizer, scheduler and shuffle generator."""
    model = new_model(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = tk.warmup_cosine(optimizer, warmup_steps=steps_per_epoch // 2, total_steps=total_steps)
    gen = tk.make_generator(seed)
    return model, optimizer, sched, gen


def epochs_of(model, optimizer, sched, gen, n):
    loader = make_loader(train_t, generator=gen)
    return [tk.train_epoch(model, loader, optimizer, loss_fn, scheduler=sched, clip=1.0) for _ in range(n)]


start = time.time()
# A: uninterrupted, 4 epochs
tk.seed_everything(0)
model_a, *rest_a = build(0)
curve_a = epochs_of(model_a, *rest_a, EPOCHS)

# B: 2 epochs, save, "crash", rebuild everything with a DIFFERENT seed, load, 2 more epochs
tk.seed_everything(0)
model_b, opt_b, sched_b, gen_b = build(0)
curve_b = epochs_of(model_b, opt_b, sched_b, gen_b, 2)
tk.save_checkpoint(CKPT, model_b, opt_b, sched_b, generator=gen_b, epoch=2)
del model_b, opt_b, sched_b, gen_b
tk.seed_everything(123)
model_b, opt_b, sched_b, gen_b = build(123)  # wrong seed on purpose: the checkpoint must override it
extra = tk.load_checkpoint(CKPT, model_b, opt_b, sched_b, generator=gen_b)
curve_b += epochs_of(model_b, opt_b, sched_b, gen_b, EPOCHS - extra["epoch"])

# C: weights-only "resume": fresh optimizer, scheduler restarted, RNG not restored
tk.seed_everything(0)
model_c, opt_c, sched_c, gen_c = build(0)
curve_c = epochs_of(model_c, opt_c, sched_c, gen_c, 2)
weights = {k: v.clone() for k, v in model_c.state_dict().items()}
model_c, opt_c, sched_c, gen_c = build(123)
model_c.load_state_dict(weights)
curve_c += epochs_of(model_c, opt_c, sched_c, gen_c, 2)

print(f"checkpoint: {CKPT.name}, {CKPT.stat().st_size / 1e6:.1f} MB | 12 epochs in {time.time() - start:.1f} s")
print("uninterrupted :", [round(x, 6) for x in curve_a], tk.state_fingerprint(model_a))
print("full resume   :", [round(x, 6) for x in curve_b], tk.state_fingerprint(model_b))
print("weights only  :", [round(x, 6) for x in curve_c], tk.state_fingerprint(model_c))
print("full resume == uninterrupted (bitwise):", curve_a == curve_b and
      tk.state_fingerprint(model_a) == tk.state_fingerprint(model_b))
print("weights-only == uninterrupted         :", curve_a == curve_c)

# %% [markdown]
# **Expected:** the full resume reproduces the uninterrupted curve and weight
# fingerprint exactly (`True`); the weights-only resume does not (`False`):
# its curve departs from the third epoch on, because it restarts the schedule
# (warmup again) with empty Adam moments and a different shuffle order. Here
# the difference is small, about 0.0003 in loss; on a long run with a decayed
# learning rate, restarting at the peak rate can undo hours of training.
# Train-loss numbers to expect: 0.3675, 0.3396, 0.3365, 0.3347 (may differ in
# the last digits on other hardware or library versions; the `True`/`False` must not). The checkpoint is about 2.5 MB:
# Adam stores two moments per parameter, so it is ~3x the weights.
#
# Two habits visible in `tk.save_checkpoint`: write to a temporary file and
# **rename** (a crash during the save never corrupts the last good
# checkpoint), and store the epoch so the resumed loop knows where to start.
# `torch.load(..., weights_only=False)` executes pickled code: only load checkpoints you wrote.

# %% [markdown]
# ## 9.8 · A debugging playbook for a model that won't learn
#
# In order, cheapest first:
#
# 1. **Loss at step 0** equals the number from 9.4? If not, the bug is in labels, loss or output scale.
# 2. **Overfit one batch.** A model with enough capacity must drive the loss on a
#    single fixed batch towards zero. If it cannot, the bug is in the model or the
#    loop (detached tensors, wrong loss, frozen parameters), *not* in the data or
#    the hyper-parameters.
# 3. **Gradient norms per layer.** Zero means a broken graph or a dead layer; NaN or
#    exploding norms mean the learning rate, an unstable op, or bad inputs.
# 4. Only then: learning rate sweep, data checks, capacity.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l2-debug-flowchart.svg" alt="Debugging a model that won't learn: step-0 loss, overfit one batch, gradient norms, then tune" width="100%">
#
# *What to notice:* each blue check takes seconds and, when it fails, points to a different part of the code (red). Hyper-parameters come last because they cannot fix any of the red boxes.

# %%
torch.manual_seed(0)
one_batch = tuple(t[:256] for t in train_t)
m = new_model(0)
m.eval()  # dropout off: we want to know if the model CAN memorise
opt = torch.optim.Adam(m.parameters(), lr=1e-2)
trace = []
for step in range(201):
    opt.zero_grad()
    loss = loss_fn(m(*one_batch[:3]), one_batch[3])
    loss.backward()
    if step in (0, 200):
        per_layer = {name: f"{p.grad.norm().item():.1e}" for name, p in m.named_parameters()
                     if name in ("user.weight", "item.weight", "mlp.0.weight", "mlp.6.weight")}
        print(f"step {step:3d} grad norms: {per_layer}")
    opt.step()
    trace.append(loss.item())
print("overfit one batch, loss at steps 0/10/20/50/200:", [round(trace[i], 4) for i in (0, 10, 20, 50, 200)])

# %% [markdown]
# **Expected:** the loss drops from about 0.32 to about 0.06 by step 10 and
# essentially 0 by step 20; every listed layer has a non-zero gradient at step 0,
# and all gradients are ~1e-7 at step 200 (nothing left to learn). Embedding
# gradients are small at step 0 because each embedding row receives
# gradient only from the rows of the batch that use it.

# %% [markdown]
# ## 9.9 · Find the bug
#
# A colleague's experiment script is below. It runs without errors and prints
# numbers that look plausible. It contains **three** bugs, each a mistake real
# teams make. Read it, run it, and look at the symptoms.

# %%
import inspect


class TwoClassClickNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.user, self.item, self.pos = nn.Embedding(len(users), 16), nn.Embedding(len(items), 16), nn.Embedding(10, 4)
        self.head = nn.Sequential(nn.Linear(36, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, u, i, p):
        return self.head(torch.cat([self.user(u), self.item(i), self.pos(p)], dim=1))


def colleague_experiment(imps, epochs=3, seed=0):
    torch.manual_seed(seed)
    df = imps[imps["day"] < 49].sample(n=150_000, random_state=seed)  # train + valid weeks
    perm = np.random.default_rng(seed).permutation(len(df))
    train, valid = df.iloc[perm[:120_000]], df.iloc[perm[120_000:]]
    tr = [torch.as_tensor(train[c].to_numpy(), dtype=torch.long) for c in ("user_id", "item_id", "position", "clicked")]
    va = [torch.as_tensor(valid[c].to_numpy(), dtype=torch.long) for c in ("user_id", "item_id", "position", "clicked")]
    model = TwoClassClickNet()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total = 0.0
        for idx in torch.randperm(len(train)).split(2048):
            opt.zero_grad()
            logits = model(tr[0][idx], tr[1][idx], tr[2][idx])
            loss = loss_fn(torch.softmax(logits, dim=1), tr[3][idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            p_click = torch.softmax(model(va[0], va[1], va[2]), dim=1)[:, 1]
        auc = roc_auc_score(va[3].numpy(), p_click.numpy())
        print(f"epoch {epoch}: train loss {total / len(train):.4f} | valid AUC {auc:.4f}")


src = inspect.getsource(colleague_experiment).splitlines()
print("\n".join(f"{n:2d}  {line}" for n, line in enumerate(src, 1)))

# %%
start = time.time()
colleague_experiment(imps)
print(f"({time.time() - start:.1f} s)")

# %% [markdown]
# **Expected:** a training loss of about 0.48 that settles around 0.44, *above*
# the ~0.40 base-rate entropy from 9.4 (a constant predictor would beat it), and a
# validation AUC of about 0.61–0.63, far below the 0.76 of section 9.3's model.
# A training loss that cannot get below the trivial baseline is itself a clue.

# %% [markdown]
# > **Exercise 3.** Which statements describe bugs that are **actually present** in
# > `colleague_experiment`? Return a list of letters from `find_bugs()`. The grader
# > stores only hashes of the answer and tells you how many you found and how many
# > false alarms you raised.
# >
# > * **A.** `opt.zero_grad()` is in the wrong place, so gradients accumulate across steps.
# > * **B.** The validation pass should not be wrapped in `torch.no_grad()`.
# > * **C.** `torch.softmax` is applied to the logits before `nn.CrossEntropyLoss`, which applies log-softmax itself.
# > * **D.** `model.eval()` is never undone, so from the second epoch on the model trains with dropout switched off.
# > * **E.** The random permutation mixes all seven weeks, so validation rows are not later in time than training rows: the score does not estimate performance on future traffic.
# > * **F.** `torch.manual_seed` is called too late for model initialisation to be reproducible.
# > * **G.** `nn.CrossEntropyLoss` needs float labels; `long` labels are silently mis-read.
# > * **H.** The learning rate 3e-3 is far too high for Adam on embeddings, which is why the loss is stuck.

# %%
def find_bugs():
    # YOUR CODE HERE: return a list of letters, e.g. ["A", "B"]
    raise NotImplementedError


grade.ex3(find_bugs)

# %% [markdown]
# For each bug you find, predict what fixing it alone would change in the printed
# numbers, then check: copy the function, fix one line, re-run. The written
# solution has the measured effect of each fix.
#
# ## 9.10 · Optional GPU extension: mixed precision (AMP)
#
# On a GPU, `torch.autocast` runs matmuls in float16 and `GradScaler` scales
# the loss so small float16 gradients do not underflow. The loop barely changes.
# This cell does nothing unless you have a CUDA GPU **and** set the flag.
# We did not run it in our verification runs (CPU only). On a T4 you should
# expect a speed-up only for much larger models than this one; with a 300k-parameter
# MLP the GPU is limited by Python overhead, not arithmetic.

# %%
RUN_GPU_EXTENSION = False

if RUN_GPU_EXTENSION and torch.cuda.is_available():
    device = torch.device("cuda")
    cat_of_item = cat_of_item.to(device)
    gpu_model = new_model(0).to(device)
    opt = torch.optim.Adam(gpu_model.parameters(), lr=3e-3)
    scaler = torch.amp.GradScaler("cuda")
    loader = make_loader(train_t, generator=tk.make_generator(0))
    for epoch in range(2):
        gpu_model.train()
        start = time.time()
        for user, item, pos, y in loader:
            user, item, pos, y = (t.to(device, non_blocking=True) for t in (user, item, pos, y))
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(gpu_model(user, item, pos), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)  # so clipping sees the true gradient norm
            torch.nn.utils.clip_grad_norm_(gpu_model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        print(f"epoch {epoch}: last loss {loss.item():.4f}, {time.time() - start:.1f} s")
    cat_of_item = cat_of_item.cpu()
else:
    print("GPU extension skipped (RUN_GPU_EXTENSION=False or no CUDA device).")

# %% [markdown]
# ## Key equations
#
# **1 · Binary cross-entropy on a logit, and its value at initialisation.** In words: the negative log of the probability the model gave to the true label; the best constant predictor scores the label entropy.
#
# $$
# \ell(z, y) = -\big[y \log \sigma(z) + (1-y) \log(1-\sigma(z))\big], \qquad \ell^{*}_\text{const} = -\big[p \log p + (1-p) \log(1-p)\big]
# $$
#
# * $z$: logit; $\sigma(z) = 1/(1+e^{-z})$; $y \in \{0, 1\}$; $p$: the positive rate. *Example:* $p = 0.137$ → 0.40; random small logits give $\log 2 = 0.69$; initialise the output bias to $\log\frac{p}{1-p} = -1.84$ to start at 0.40.
#
# **2 · Softmax and cross-entropy.** In words: turn $C$ logits into probabilities, and pay $-\log$ of the probability of the right class.
#
# $$
# \operatorname{softmax}(\mathbf{z})_c = \frac{e^{z_c}}{\sum_{k=1}^{C} e^{z_k}}, \qquad \ell = -\log \operatorname{softmax}(\mathbf{z})_y
# $$
#
# * *Example:* logits (2, 0.5, −1), true class 0: probabilities 0.79, 0.18, 0.04, loss $-\log 0.79 = 0.24$. Uniform guessing over $C$ classes: $\log C$.
#
# **3 · The AdamW update.** In words: keep running averages of the gradient and of its square, correct their start-up bias, step by their ratio, and shrink the weights separately (decoupled weight decay).
#
# $$
# \begin{aligned}
# m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \\
# \hat m_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t} \\
# \theta_t &= \theta_{t-1} - \eta \lambda\, \theta_{t-1} - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
# \end{aligned}
# $$
#
# * $g_t$: gradient; $\eta$: learning rate; $\lambda$: weight decay; $\beta_1 = 0.9$, $\beta_2 = 0.999$. *Example:* at $t = 1$, $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so every parameter moves by about $\eta$ whatever the size of its gradient: that is why early steps are risky and warmup helps.
#
# **4 · Linear warmup, then cosine decay** (as `tk.warmup_cosine`). In words: ramp up over $T_w$ steps, then follow half a cosine down to a floor.
#
# $$
# \eta_t = \eta_{\max} \cdot \begin{cases} \dfrac{t+1}{T_w} & t < T_w \\[6pt] r + (1-r)\,\dfrac{1}{2}\Big(1 + \cos\big(\pi\, \tfrac{t - T_w}{T - T_w}\big)\Big) & t \ge T_w \end{cases}
# $$
#
# * $T$: total steps; $r$: the floor as a fraction of $\eta_{\max}$ (0.05 here). *Example:* $T_w = 10$, $T = 100$: step 4 → $0.5\,\eta_{\max}$; step 55 (halfway through the decay) → $0.525\,\eta_{\max}$.
#
# **5 · Gradient-norm clipping.** In words: if the whole gradient is longer than $c$, shrink it to length $c$; its direction does not change.
#
# $$
# \mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1, \frac{c}{\lVert \mathbf{g} \rVert_2}\right)
# $$
#
# * *Example:* $\mathbf{g} = (3, 4)$, $c = 1$: norm 5, so $\mathbf{g} \to (0.6, 0.8)$.
#
# The cell below checks all five against PyTorch.

# %%
# Check the five equations against PyTorch
z, y = torch.tensor([-2.0, 0.5, 3.0]), torch.tensor([0.0, 1.0, 1.0])
bce_hand = -(y * torch.log(torch.sigmoid(z)) + (1 - y) * torch.log(1 - torch.sigmoid(z))).mean()
p = 0.137
print(f"1. BCE by hand {bce_hand.item():.6f} | BCEWithLogitsLoss {nn.BCEWithLogitsLoss()(z, y).item():.6f}"
      f" | entropy at p=0.137: {-(p * np.log(p) + (1 - p) * np.log(1 - p)):.4f}")

logits_chk = torch.tensor([[2.0, 0.5, -1.0]])
ce_hand = -torch.log(torch.softmax(logits_chk, dim=1)[0, 0])
print(f"2. CE by hand {ce_hand.item():.6f} | CrossEntropyLoss "
      f"{nn.CrossEntropyLoss()(logits_chk, torch.tensor([0])).item():.6f}")

w0, grad = torch.tensor([1.0, -2.0]), torch.tensor([0.5, -0.1])
lr, b1, b2, eps, wd = 0.1, 0.9, 0.999, 1e-8, 0.01
param = nn.Parameter(w0.clone())
opt_chk = torch.optim.AdamW([param], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
param.grad = grad.clone()
opt_chk.step()
m_hat, v_hat = (1 - b1) * grad / (1 - b1), (1 - b2) * grad**2 / (1 - b2)
w_hand = w0 - lr * wd * w0 - lr * m_hat / (v_hat.sqrt() + eps)
print(f"3. AdamW by hand {w_hand.tolist()} | torch {param.data.tolist()} | equal: "
      f"{torch.allclose(w_hand, param.data)}")

dummy = nn.Parameter(torch.zeros(1))
opt_s = torch.optim.SGD([dummy], lr=1.0)
sched = tk.warmup_cosine(opt_s, warmup_steps=10, total_steps=100, min_ratio=0.05)
lib_lr = []
for _ in range(100):
    lib_lr.append(opt_s.param_groups[0]["lr"])
    opt_s.step()
    sched.step()
t = np.arange(100)
hand_lr = np.where(t < 10, (t + 1) / 10,
                   0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * (t - 10) / 90)))
print(f"4. warmup-cosine max |hand - tk| = {np.abs(hand_lr - np.array(lib_lr)).max():.2e} "
      f"| lr at steps 4, 55: {lib_lr[4]:.3f}, {lib_lr[55]:.3f}")

clip_p = nn.Parameter(torch.zeros(2))
clip_p.grad = torch.tensor([3.0, 4.0])
norm_before = torch.nn.utils.clip_grad_norm_([clip_p], max_norm=1.0)
print(f"5. clip: norm before {norm_before.item():.1f}, gradient after {clip_p.grad.tolist()}")

# %% [markdown]
# **Expected:** 1–2: each pair agrees to six decimals, entropy 0.3995; 3: `equal: True` (both about `[0.899, -1.898]`); 4: a maximum difference of about 1e-16, learning rates 0.500 and 0.525; 5: norm 5.0 and a gradient of about `[0.6, 0.8]`.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `shoprank` | setup cell not run | run the first code cell |
# | Loss at step 0 far from `log 2` / base-rate entropy | labels wrong dtype or range, softmax/sigmoid applied twice, wrong reduction | print one batch; check the loss against 9.4 before training |
# | Two runs with the same seed differ | the sampler uses the global RNG; a library draws random numbers; non-deterministic GPU kernels | dedicated `torch.Generator` for shuffling, `tk.set_deterministic(True)`, seed before building the model |
# | Resumed run differs from uninterrupted | optimizer/scheduler/RNG state not saved or not restored | save with `tk.save_checkpoint` and restore with `tk.load_checkpoint` into freshly built objects |
# | DataLoader with `num_workers>0` hangs or errors in a notebook | dataset/collate defined in an unpicklable way (spawn start method), or a worker crashed | use `num_workers=0` to see the real traceback; define datasets in a module |
# | Every row seen twice per epoch | `IterableDataset` not sharded across workers | shard by `get_worker_info()` as in `ParquetDayStream` |
# | Memory grows every step | accumulating `loss` tensors instead of `loss.item()` | `.item()` or `.detach()` before storing |
# | `RuntimeError: ... deterministic` | an op has no deterministic implementation (GPU) | `warn_only=True` (as in `tk.set_deterministic`), or replace the op |

# %% [markdown]
# ## Checkpoint
#
# Answer aloud, without notes, in under a minute each:
#
# 1. Walk through one training step. Where exactly must `zero_grad` go, and when would you deliberately *not* zero gradients every step?
# 2. What do `model.eval()` and `torch.no_grad()` each do, and what goes wrong if you use only one of them during validation?
# 3. Your binary classifier with a 2% positive rate shows a loss of 0.69 at step 0. Is that a bug? What should it be, and how do you fix the initialisation?
# 4. A job restarted from a checkpoint ends with a worse model than an uninterrupted run. List everything a checkpoint must contain to make resume exact.
# 5. Two runs with the same seed give different results on a GPU. Name the possible sources and how you would make them repeatable, and when bitwise repeatability is not worth its cost.
# 6. Your model's loss won't go down. Describe your first 15 minutes of debugging, in order.
#
# **Next chapter:** [`10_sequence_models_for_user_behavior.ipynb`](../ch28-sequence-models-for-user-behavior/notebook.ipynb) — the same loop, applied to a model that reads a user's click history in order.
