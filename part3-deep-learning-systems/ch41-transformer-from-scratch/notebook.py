# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch41-transformer-from-scratch/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 41 · A transformer from scratch
#
# **Part III · Chapter 41 · about 90 minutes · T4 GPU (runs on CPU with a reduced budget)**
#
# "Implement multi-head attention" and "what does a KV cache change?" are among the
# most common coding and depth questions for engineers who work with language models.
# In this lab we build a small GPT-style decoder in which every line can be explained,
# train it on Shakespeare, sample from it, and make generation faster with a KV cache.
#
# After this lab you should be able to:
#
# 1. implement causal multi-head self-attention from the formula, and prove the mask works with a perturbation test;
# 2. assemble a pre-LayerNorm decoder block (attention, MLP, residuals) with weight tying;
# 3. predict the loss at initialisation and check it before training;
# 4. train with the course's training loop and read the loss curve;
# 5. sample with temperature and top-k, and explain how each changes the output;
# 6. implement a KV cache, verify it changes nothing but speed, and measure the speed-up.
#
# **You will produce:** `artifacts/level3/nb03/tinygpt.pt` (a trained checkpoint) and
# `artifacts/level3/nb03/samples.txt`. The model code is
# [`src/scaleml/tinygpt.py`](../../src/scaleml/tinygpt.py): read it alongside this notebook.
#
# **Measured here vs on a T4.** Our verification runs are CPU-only, with the CPU
# budget (a 64-wide, 2-layer model, 1,500 steps). Numbers in **Expected:** lines were
# printed by such a run unless marked *"on a T4 you should see … (not measured in
# our verification runs)"*.

# %% [markdown]
# Read first: [Chapter 41 text](README.md) explains the ideas behind this lab (A transformer from scratch) in more depth; [solutions](solutions.md) has worked answers.

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
# --- Catch up: nothing needed from earlier notebooks; download the corpus ----
import math
import time

import torch

from scaleml import profiling as prof
from scaleml import tinygpt as tg
from scaleml.device import count_parameters, seed_everything, sync
from grading import ch41 as grade
from scaleml.train import TrainConfig, train

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
OUT = ART / "level3" / "nb03"
OUT.mkdir(parents=True, exist_ok=True)

text = tg.load_tiny_shakespeare()
print(f"dataset: https://huggingface.co/datasets/{tg.SHAKESPEARE_REPO} (revision {tg.SHAKESPEARE_REVISION[:7]})")
print("licence: the repo ships Karpathy's char-rnn MIT licence; the text itself (Shakespeare) is public domain.")
print(f"{len(text):,} characters")
print(text[:200])

# %% [markdown]
# **Expected:** 1,115,394 characters, starting with "First Citizen: Before we proceed any further,
# hear me speak." The file comes from the Hugging Face Hub
# ([card](https://huggingface.co/datasets/Trelis/tiny-shakespeare)), pinned to one
# revision so that everyone trains on the same bytes.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Token / vocabulary | the unit the model reads (here one character) / the set of all of them | 65 distinct characters |
# | Embedding | a learned vector looked up for each token id (and each position) | `tok_emb`, `pos_emb`, 64 numbers each on CPU |
# | Query, key, value | three vectors per token: what it is looking for, what it offers, what it hands over | `x @ w_qkv` split in three (Exercise 1) |
# | Attention head | one independent attention computation on a slice of the channels | 4 heads of 16 channels each |
# | Causal mask | −∞ added to scores of future positions so a token cannot see what comes after it | the perturbation test in 41.3 |
# | Residual stream | the running vector each block reads from and adds its output to | `x = x + Attention(LayerNorm(x))` |
# | LayerNorm | rescales each token's vector to mean 0 and variance 1, then applies a learned scale and shift | pre-LN: inside each branch |
# | Logits | unnormalised scores, one per vocabulary entry; softmax turns them into probabilities | `(B, T, 65)` from the model |
# | Cross-entropy / perplexity | average −log probability of the correct next token / its exponential | ln 65 ≈ 4.17 at init → perplexity 65 |
# | Temperature, top-k | divide logits by τ before softmax / keep only the k most likely tokens | τ = 0.8, k = 10 in 41.6 |
# | KV cache | stored keys and values of past tokens, so each new token is computed once | 2–5× faster generation in 41.7 |
# | Weight tying | the output layer reuses the input embedding matrix | `weights tied: True` |

# %% [markdown]
# ## 41.1 · Characters as tokens
#
# A character-level tokenizer keeps the focus on the model: the vocabulary is
# the set of distinct characters (65 here). Real models use subword
# tokenizers (Chapter 42), which make sequences ~4× shorter for English text.
# We hold out the last 10% of the text for validation: a *contiguous* block, so
# validation windows never overlap training windows.

# %%
tok = tg.CharTokenizer(text)
data = torch.tensor(tok.encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]
print(f"vocab {tok.vocab_size}: {''.join(tok.chars)!r}")
print(f"train {len(train_data):,} tokens | val {len(val_data):,} tokens")
print("round trip:", tok.decode(tok.encode("To be, or not to be")))

# %% [markdown]
# **Expected:** vocabulary 65 (newline, space, punctuation, letters), 1,003,854
# training and 111,540 validation tokens, and a perfect round trip.
#
# ## 41.2 · Causal self-attention
#
# For each position `t`, attention computes a weighted average of *value*
# vectors, weighted by how well the position's *query* matches each *key*:
#
# `Attention(Q, K, V) = softmax(Q Kᵀ / √d + M) V`
#
# where `d` is the head dimension and `M` is the **causal mask**: 0 where
# key position ≤ query position, −∞ otherwise, so that after the softmax each
# token attends only to itself and the past. That is what makes next-token
# training valid: the prediction at position `t` cannot peek at token `t+1`.
# Dividing by `√d` keeps the dot products' variance near 1 so the softmax
# does not saturate at initialisation.
#
# **Multi-head**: split the `C` channels into `H` heads of size `d = C/H`, run
# attention in each independently (different heads learn different patterns:
# previous token, start of word, matching quote…), concatenate, and mix with an
# output projection.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-causal-attention.svg" alt="A 5 by 5 attention grid for 'the cat sat on mat': cells above the diagonal are masked with minus infinity; the row for 'sat' has weights 0.2, 0.5, 0.3; beside it the four steps scores, mask, softmax, weighted sum of V" width="100%">
#
# *What to notice:* each row is a probability distribution over the past only; the upper triangle is never zero *before* the softmax, it is −∞, so it becomes exactly 0 *after* it.
#
# > **Intuition.** A soft library search. Your **query** is what you type; every earlier token
# > has a **key** (its catalogue entry) and a **value** (its content). You get back a blend of
# > contents, weighted by how well each key matched. The causal mask means you may only search
# > books already on the shelf. *Where it breaks down:* the catalogue entries are learned, and
# > every token is at once a reader (query) and a book (key and value).
#
# > **Why divide by √d? A worked example.** A dot product of two random vectors with $d$
# > independent unit-variance entries has standard deviation $\sqrt d$: 8 for $d = 64$. Scores
# > spread over ±16 make the softmax almost one-hot, and its gradient almost zero. Dividing by
# > $\sqrt{d}$ brings the spread back to about ±2.

# %% [markdown]
# > **Exercise 1.** Write `causal_self_attention(x, w_qkv, w_out, n_heads)` as a pure function
# > (no `nn.Module`, no biases). `x` is `(B, T, C)`, `w_qkv` is `(C, 3C)` and `w_out` is `(C, C)`.
# > Compute `x @ w_qkv` and split it into Q, K, V **in that order** along the last dimension;
# > split each into `n_heads` heads; apply scaled, causally masked softmax attention; merge
# > the heads back to `(B, T, C)` and multiply by `w_out`. The grader compares with a reference
# > and runs a perturbation test (changing token 4 must not change outputs 0–3).

# %%
def causal_self_attention(x, w_qkv, w_out, n_heads):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(causal_self_attention)

# %% [markdown]
# ## 41.3 · The decoder
#
# `tg.TinyGPT` (open [`src/scaleml/tinygpt.py`](../../src/scaleml/tinygpt.py)) is:
#
# ```
# tokens ─► token embedding + position embedding
#        ─► N × Block:  x = x + Attention(LayerNorm(x))
#                       x = x + MLP(LayerNorm(x))        # MLP: Linear(C→4C), GELU, Linear(4C→C)
#        ─► LayerNorm ─► Linear(C→vocab) whose weight IS the token embedding (weight tying)
# ```
#
# * **Pre-LayerNorm** (normalise *inside* the residual branch) keeps a clean
#   identity path from output to input, so deep stacks train stably without
#   careful warm-up; the original Transformer's post-LN needed it.
# * **Residual connections** let gradients flow straight through the stack.
# * **Weight tying** shares one `vocab × C` matrix between input and output:
#   fewer parameters, and in practice equal or better perplexity on small data.
# * Initialisation follows GPT-2: N(0, 0.02), with residual output projections
#   scaled by `1/√(2·layers)` so the residual stream's variance does not grow with depth.
#
# Model size comes from the budget: CPU 64 wide × 2 layers; GPU 256 wide × 4 layers.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-decoder-block.svg" alt="A pre-LayerNorm decoder block left to right: x, LayerNorm, causal attention, add back to x, LayerNorm, MLP, add back again, with two residual skip paths above" width="100%">
#
# *What to notice:* the residual paths above the row never pass through a LayerNorm or a weight matrix, so the output is always 'input + corrections': that clean path is why pre-LN stacks train stably.
#
# > **Intuition.** The residual stream is a shared notebook passed down the stack. Each block
# > reads it (through LayerNorm), and *adds* a note; nothing is ever erased. Attention lets a
# > token copy notes from earlier tokens, the MLP rewrites the token's own notes.

# %%
CFG = tg.GPTConfig(vocab_size=tok.vocab_size, block_size=BUDGET.max_seq_len, n_embd=BUDGET.hidden,
                   n_head=4, n_layer=BUDGET.layers, dropout=0.1 if DEVICE.type == "cuda" else 0.0)
seed_everything(0)
model = tg.TinyGPT(CFG).to(DEVICE)
print(CFG)
print(f"{count_parameters(model):,} parameters ({tg.count_non_embedding_params(model):,} outside the embeddings)")
print("weights tied:", model.head.weight.data_ptr() == model.tok_emb.weight.data_ptr())

# %% [markdown]
# **Expected (CPU budget):** `n_embd=64, n_layer=2, block_size=64`, 108,352 parameters,
# 100,096 of them outside the embeddings, and `weights tied: True`. (On the GPU budget:
# 256 × 4 layers, ~3.2M parameters.)
#
# ### Test the mask on the real model
#
# The same perturbation test the grader used, now on the full model: change
# one token in the middle of a sequence and check that the logits *before* it do
# not move, while the logits from it onwards do. This is the unit test that
# catches the most harmful transformer bug, a leaky mask, which otherwise shows up
# only as a suspiciously low training loss and a model that generates meaningless text.

# %%
model.eval()
g = torch.Generator().manual_seed(0)
idx = torch.randint(0, tok.vocab_size, (2, CFG.block_size), generator=g).to(DEVICE)
t_change = CFG.block_size // 2
idx2 = idx.clone()
idx2[:, t_change] = (idx2[:, t_change] + 1) % tok.vocab_size
with torch.no_grad():
    a, _ = model(idx)
    b, _ = model(idx2)
before = (a[:, :t_change] - b[:, :t_change]).abs().max().item()
after = (a[:, t_change:] - b[:, t_change:]).abs().max().item()
print(f"changed token {t_change}: max logit change before it {before:.2e} | from it on {after:.2e}")
assert before < 1e-5 < after, "causal mask broken!"

# %% [markdown]
# **Expected:** a change of exactly 0 (or ~1e-7) before position 32 and a clearly non-zero
# change (~0.8 in our run) from position 32 on.
#
# ## 41.4 · The loss at initialisation
#
# A freshly initialised model with small weights outputs nearly uniform
# probabilities over the vocabulary, so its cross-entropy should be about
# **ln(vocab)** = ln 65 ≈ 4.17. Much higher means overconfident wrong predictions
# (bad initialisation, e.g. weights too large); much lower before training means a bug
# (most often a leaky mask, or targets not shifted by one).

# %%
train_windows = tg.RandomWindows(train_data, CFG.block_size, BUDGET.batch_size, n_batches=0, seed=0)
val_windows = tg.RandomWindows(val_data, CFG.block_size, BUDGET.batch_size, n_batches=0, seed=1)
init_val = tg.estimate_loss(model, val_windows, n_batches=10, device=DEVICE)
print(f"loss at init {init_val:.3f} | ln(vocab) = {math.log(tok.vocab_size):.3f}")
# a baseline to beat: always predict the training set's character frequencies (no context at all)
freq = torch.bincount(train_data, minlength=tok.vocab_size).float() / len(train_data)
unigram = -(torch.log(freq.clamp_min(1e-12))[val_data]).mean().item()
print(f"unigram baseline (character frequencies, no context): {unigram:.3f} on validation")

# %% [markdown]
# **Expected:** about 4.19 vs ln 65 = 4.174, and a unigram baseline of about 3.3:
# a model must beat that to have learned anything about context.
#
# ## 41.5 · Training
#
# We train with `scaleml.train.train` (read [`src/scaleml/train.py`](../../src/scaleml/train.py)):
# AdamW, linear warm-up then cosine decay, gradient clipping at 1.0, and fp16
# autocast + GradScaler on a T4 (a no-op on CPU). Each "epoch" is a fixed number of
# random 64-character windows (`tg.RandomWindows`).

# %%
STEPS = {"gpu": 4000, "cpu": 1500, "smoke": 30}[BUDGET.name]
LR = 3e-3 if BUDGET.name != "gpu" else 1e-3  # small models tolerate (and need) a higher LR
config = TrainConfig(epochs=1, lr=LR, warmup_steps=100, max_grad_norm=1.0, log_every=max(1, STEPS // 6))


def lm_loss(model, batch):
    _, loss = model(batch["x"].to(DEVICE, non_blocking=True), batch["y"].to(DEVICE, non_blocking=True))
    return loss


def evaluate(model):
    return {"val_loss": round(tg.estimate_loss(model, val_windows, n_batches=20, device=DEVICE), 4)}


seed_everything(0)
result = train(model, lambda: tg.RandomWindows(train_data, CFG.block_size, BUDGET.batch_size, STEPS, seed=0),
               lm_loss, DEVICE, config, evaluate=evaluate)
for h in result.history:
    print(h)
tokens_per_s = STEPS * BUDGET.batch_size * CFG.block_size / result.seconds
print(f"{result.steps} steps in {result.seconds:.0f} s | {tokens_per_s:,.0f} training tokens/s on {DEVICE}")
torch.save({"config": CFG.__dict__, "chars": tok.chars, "model": model.state_dict()}, OUT / "tinygpt.pt")
print("saved", OUT / "tinygpt.pt")

# %% [markdown]
# **Expected (CPU budget, measured):** the training loss falls from ~2.6–2.7 (average
# over the first 250 steps) to ~1.7, and the validation loss ends around 1.83–1.86 (our run: 1.833),
# in roughly 35–60 s (about 50,000–90,000 training tokens/s), far below the unigram
# baseline from 3.4.
#
# **On a T4 you should see** the GPU budget (256 × 4 layers, 128-character context,
# 4,000 steps) reach a validation loss around 1.5 in a few minutes, and
# samples with mostly real words and plausible speaker names (not measured in our
# verification runs). `tinygpt.pt` holds the config, the vocabulary and the weights:
# enough to sample, not to *resume* training (that also needs the optimizer and
# scheduler state; see `scaleml.train.save_checkpoint`).
#
# ## 41.6 · Sampling: temperature and top-k
#
# The model outputs logits for the next token. To generate, divide them by a
# **temperature** τ (τ < 1 sharpens the distribution, τ > 1 flattens it, τ → 0 is
# greedy), optionally keep only the **top-k** most likely tokens (removing the long tail
# of unlikely ones that produce meaningless output), sample, append, repeat.

# %% [markdown]
# > **Exercise 2.** Write `top_k_sample(logits, k, temperature, generator)`: `logits` is
# > `(B, V)`; return a `(B,)` `torch.long` tensor of sampled token ids. Divide by `temperature`
# > (> 0), keep only the `k` largest logits per row (all of them if `k >= V`), and sample from
# > the softmax with `torch.multinomial(..., generator=generator)` so results are reproducible.

# %%
def top_k_sample(logits, k, temperature, generator):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(top_k_sample)

# %%
prompt = torch.tensor([tok.encode("ROMEO:\n")], device=DEVICE)
samples = []
for temperature, top_k in [(0.0, None), (0.8, 10), (1.5, None)]:
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    out = tg.generate(model, prompt, max_new_tokens=160, temperature=temperature, top_k=top_k, generator=gen)
    sample = tok.decode(out[0].tolist())
    samples.append(f"--- temperature={temperature}, top_k={top_k}\n{sample}\n")
    print(samples[-1])
_ = (OUT / "samples.txt").write_text("\n".join(samples))

# %% [markdown]
# **Expected (CPU budget):** greedy decoding (temperature 0) quickly falls into
# a loop, repeating the same phrase. Temperature 0.8 with top-k 10 gives
# Shakespeare-shaped text: speaker names in capitals followed by a colon, line
# breaks, short words, many misspelt. Temperature 1.5 without top-k is noticeably
# more garbled (rare characters, broken words). The exact text depends on your run.
#
# ## 41.7 · The KV cache
#
# Generating token `t+1` without a cache re-runs the model on all `t` previous
# tokens, although their keys and values have not changed (causal masking means
# later tokens never affect earlier positions). A **KV cache** stores each layer's K and V
# for the tokens processed so far; each new step computes Q, K, V for *one* token,
# appends K and V to the cache and attends over it. Read `CausalSelfAttention.forward`
# and `generate` in `tinygpt.py`: the cache is a dict per layer, and the mask accounts
# for the query's absolute position `past + i`.
#
# First, correctness: greedy decoding with and without the cache must produce the same tokens.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-kv-cache.svg" alt="Prefill runs the whole prompt at once and writes keys and values for every token into the cache; each decode step computes q, k, v for one new token, appends k and v, and attends over the cache" width="100%">
#
# *What to notice:* prefill is one big parallel pass; decode is a long series of tiny passes that each read the *whole* cache, which is why serving long contexts is limited by memory, not FLOPs.
#
# > **Intuition.** Without a cache, writing each new word means re-reading the whole book from
# > page one. With a cache you keep a one-line note per page (its K and V) and only read the
# > notes. *Where it breaks down:* the notes are not small: for a 7B model they take 0.5 MB per
# > token, so the "notes" can outgrow the model on long contexts.

# %%
with_cache = tg.generate(model, prompt, 50, temperature=0.0, use_cache=True)
without = tg.generate(model, prompt, 50, temperature=0.0, use_cache=False)
print("identical greedy outputs:", torch.equal(with_cache, without))
# and the logits themselves: one prefill + 10 single-token steps vs a full forward pass
seq = with_cache[:, :CFG.block_size]
with torch.no_grad():
    full, _ = model(seq)
    caches = model.new_cache()
    part, _ = model(seq[:, :-10], caches=caches)
    steps = [model(seq[:, i:i + 1], caches=caches)[0] for i in range(seq.shape[1] - 10, seq.shape[1])]
cached = torch.cat([part] + steps, dim=1)
print(f"max |logit difference| cached vs full: {(cached - full).abs().max().item():.1e}")

# %% [markdown]
# **Expected:** `True`, and a logit difference around 1e-6–1e-5 (our run: 9.4e-06; floating-point
# summation order), i.e. the cache changes nothing but speed.
#
# Now speed. The benefit grows with the context length, so we measure on an
# *untrained* model with the same width and depth but a 512-token context (speed does not
# depend on the weights' values), generating from an 8-token prompt.

# %%
bench_cfg = tg.GPTConfig(tok.vocab_size, block_size=512, n_embd=CFG.n_embd, n_head=CFG.n_head, n_layer=CFG.n_layer)
bench_model = tg.TinyGPT(bench_cfg).to(DEVICE).eval()
start_ids = torch.randint(0, tok.vocab_size, (1, 8), device=DEVICE)
speed = {}
for new_tokens in (64, 256, 500):
    for use_cache in (False, True):
        sync(DEVICE)
        t0 = time.perf_counter()
        tg.generate(bench_model, start_ids, new_tokens, temperature=0.0, use_cache=use_cache)
        sync(DEVICE)
        speed[(new_tokens, use_cache)] = new_tokens / (time.perf_counter() - t0)
    print(f"{new_tokens:4} new tokens: no cache {speed[(new_tokens, False)]:7.0f} tok/s | "
          f"KV cache {speed[(new_tokens, True)]:7.0f} tok/s | speed-up "
          f"{speed[(new_tokens, True)] / speed[(new_tokens, False)]:.1f}x")
kv_bytes = 2 * bench_cfg.n_layer * bench_cfg.block_size * bench_cfg.n_embd * 4
print(f"KV cache size at 512 tokens, batch 1, fp32: {prof.human_bytes(kv_bytes)}")

# %% [markdown]
# **Expected (CPU, measured):** the cache is faster at every length and the gap grows
# with the context: in our runs 2.1–2.2× at 64 new tokens, 3.5–4.6× at 256 and 3.8–5.0× at
# 500 (single timings on a shared CPU: expect ±30%). Without a cache, tokens/s falls steadily
# (from ~850–1,100 to ~300) because every step re-processes the whole context. With the cache it
# also falls at long lengths, for two reasons: attention over the cache still grows
# with the context, and this simple implementation `torch.cat`s the cache every step
# (re-copying it); production code pre-allocates the cache. The cache for this toy model
# is only 512 KB.
#
# **On a T4 you should see** a similar trend for this tiny model, with absolute
# rates limited by kernel-launch overhead rather than arithmetic (dozens of small
# kernels per token); for real LLMs the cache is what makes generation practical
# (not measured in our verification runs). The cost moves to memory: for a 7B model
# (32 layers, 4096 wide) in fp16, the cache is 2 × 32 × 4096 × 2 bytes = 0.5 MB *per
# token*: 2 GB for one 4k-token sequence. Serving systems are designed around that
# number (Chapter 50: paged KV caches, continuous batching; multi-query and
# grouped-query attention share K and V across heads to shrink it).

# %% [markdown]
# > **Exercise 3.** (a) Give the time complexity of one forward pass of self-attention over a
# > sequence of length `n` with hidden size `d`, and its memory for the attention matrix.
# > (b) Generating `n` tokens one at a time: what is the total attention cost without a KV cache,
# > and with one? What about the MLP and projection layers? (c) What does the KV cache cost, and
# > why is long-context generation usually limited by memory bandwidth rather than FLOPs?
# > Write your answer in the cell below.

# %% [markdown]
# *Your answer (Exercise 3):* ...

# %% [markdown]
# ## Key equations
#
# **1. Scaled dot-product attention.**
#
# $$
# \operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\Big(\frac{Q K^\top}{\sqrt{d_k}} + M\Big)\, V
# $$
#
# In words: score every query against every key, scale, hide the future, normalise each row,
# average the values. Symbols: $Q, K, V \in \mathbb{R}^{T \times d_k}$; $M_{ij} = 0$ if $j \le i$,
# $-\infty$ otherwise. Example: query $(1, 0)$, keys $(1, 0)$ and $(0, 1)$, $d_k = 2$: scores
# $0.707$ and $0$, softmax weights $0.67$ and $0.33$.
#
# **2. Multi-head attention.**
#
# $$
# \operatorname{MHA}(X) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_H)\, W^O, \qquad
# \text{head}_i = \operatorname{Attention}(X W_i^Q, X W_i^K, X W_i^V)
# $$
#
# In words: $H$ attentions on $d_k = C/H$ channels each, side by side, mixed by $W^O$. Example:
# $C = 64$, $H = 4$ gives $d_k = 16$; the weights $W^{Q,K,V,O}$ hold $4C^2 = 16{,}384$ parameters
# per layer (no biases), whatever $H$ is.
#
# **3. LayerNorm.**
#
# $$
# \operatorname{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad
# \mu = \frac{1}{C}\sum_c x_c, \quad \sigma^2 = \frac{1}{C}\sum_c (x_c - \mu)^2
# $$
#
# In words: per token, subtract the mean and divide by the standard deviation over channels,
# then apply a learned scale $\gamma$ and shift $\beta$. Example: $x = (1, 2, 3)$: $\mu = 2$,
# $\sigma^2 = 2/3$, output $\approx (-1.22, 0, 1.22)$ with $\gamma = 1, \beta = 0$.
#
# **4. Cross-entropy and perplexity.**
#
# $$
# \mathcal{L} = -\frac{1}{N} \sum_{t=1}^{N} \log p_\theta\big(x_{t+1} \mid x_{\le t}\big), \qquad
# \operatorname{PPL} = e^{\mathcal{L}}
# $$
#
# In words: the average surprise at the true next token; perplexity is "as unsure as a uniform
# choice among PPL options". Example: uniform over 65 characters gives $\mathcal L = \ln 65 = 4.17$
# and PPL = 65; our trained validation loss of 1.83 is a perplexity of $e^{1.83} \approx 6.2$.
#
# **5. Softmax with temperature.**
#
# $$
# p_i = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}
# $$
#
# Example: logits $(2, 1, 0)$ give $(0.665, 0.245, 0.090)$ at $\tau = 1$ and
# $(0.867, 0.117, 0.016)$ at $\tau = 0.5$: lower temperature, sharper choice.
#
# **6. KV-cache size.**
#
# $$
# \text{bytes} = 2 \cdot L \cdot H \cdot d_{\text{head}} \cdot T \cdot B \cdot b
# $$
#
# In words: a key and a value (the 2) per layer, head, channel, position and sequence. Symbols:
# $L$ layers, $H$ heads, $d_{\text{head}}$ channels per head, $T$ positions, $B$ sequences, $b$
# bytes per number. Example: 7B model ($L = 32$, $H = 32$, $d_{\text{head}} = 128$), fp16, one
# 4,096-token sequence: $2 \cdot 32 \cdot 32 \cdot 128 \cdot 4096 \cdot 2 \approx 2.1 \times 10^9$ bytes.
#
# The next cell checks 1, 3 and 4 against PyTorch's own implementations, and 6 against the toy
# model's cache from 3.7.

# %%
import torch.nn.functional as F

torch.manual_seed(0)
q, k, v = torch.randn(3, 1, 2, 5, 8).unbind(0)  # (batch, heads, T, d_head)
scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
manual = scores.masked_fill(future, float("-inf")).softmax(-1) @ v
print("attention  manual vs F.scaled_dot_product_attention:",
      f"{(manual - F.scaled_dot_product_attention(q, k, v, is_causal=True)).abs().max().item():.1e}")

x = torch.randn(4, 16)
ln = torch.nn.LayerNorm(16)
manual_ln = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + ln.eps)
print(f"LayerNorm  manual vs nn.LayerNorm: {(manual_ln - ln(x)).abs().max().item():.1e}")

logits, y = torch.randn(6, 65), torch.randint(0, 65, (6,))
manual_ce = -logits.log_softmax(-1)[torch.arange(6), y].mean()
print(f"cross-entropy manual vs F.cross_entropy: {(manual_ce - F.cross_entropy(logits, y)).abs().item():.1e}")
print(f"perplexity at init: exp({init_val:.3f}) = {math.exp(init_val):.1f} (vocabulary {tok.vocab_size})")


def kv_cache_bytes(layers, heads, d_head, seq, batch, bytes_per):
    return 2 * layers * heads * d_head * seq * batch * bytes_per


print(f"KV cache, 7B fp16, 4,096 tokens: {kv_cache_bytes(32, 32, 128, 4096, 1, 2) / 2**30:.1f} GiB")
print(f"KV cache, the 41.7 benchmark model (512 tokens, fp32): "
      f"{kv_cache_bytes(CFG.n_layer, CFG.n_head, CFG.n_embd // CFG.n_head, 512, 1, 4) / 2**10:.0f} KiB")

# %% [markdown]
# **Expected:** differences of ~1e-7 or smaller for attention, LayerNorm and cross-entropy;
# perplexity at init about 66 (≈ the vocabulary size: a uniform guess); 2.0 GiB for the 7B cache
# and 512 KiB for the toy model on the CPU budget (the number quoted in 41.7).

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `scaleml` | setup cell not run | run the first code cell |
# | "Could not download Trelis/tiny-shakespeare" and a tiny corpus | no internet / Hub outage | re-run when online; the fallback only keeps the code runnable |
# | Loss at init far above ln(vocab) | weights initialised too large; missing `1/√d` | check `_init` and the attention scale |
# | Training loss drops far below ~1.0 within a few hundred steps | causal mask leaks the future, or `y` not shifted by one | run the perturbation test in 41.3 |
# | `ValueError: sequence length ... exceeds block_size` | prompt + generation longer than the context with `use_cache=True` and a manual loop | crop to the last `block_size` tokens (as `generate` does) |
# | Loss becomes `nan` on a T4 | fp16 overflow with a too-high LR | lower `LR`; GradScaler skips inf steps but cannot fix a diverging run |
# | Cached and uncached generations differ | position offset or mask wrong in the cached path | compare logits step by step as in 41.7 |
# | Sampling gives the same text every time | temperature 0 (greedy) or a fixed generator seed | expected; vary the seed or temperature |

# %% [markdown]
# ## Checkpoint
#
# Answer each question without notes; a complete answer takes about a minute:
#
# 1. Write the attention formula and explain the role of each part: Q, K, V, the `√d` scaling, the mask, the softmax, multiple heads.
# 2. Pre-LayerNorm vs post-LayerNorm: what is the difference and why do modern decoders use pre-LN?
# 3. Your new language model's training loss is 0.05 after 200 steps on a character dataset. What do you suspect and how do you test it?
# 4. What does temperature do to the next-token distribution? Top-k vs top-p (nucleus): when would you prefer each? Why does greedy decoding repeat itself?
# 5. Explain the KV cache: what is stored, what computation it saves, what it costs in memory, and how multi-query/grouped-query attention reduce that cost.
# 6. Why tie the input embedding and output projection weights? When might you not?
#
# **Next chapter:** [Chapter 42 · Pretrained transformers & tokenisation](../ch42-pretrained-transformers-and-tokenization/notebook.ipynb) — the same architecture at scale, pretrained: subword tokenizers, padding, pooling and batched inference with Hugging Face.
