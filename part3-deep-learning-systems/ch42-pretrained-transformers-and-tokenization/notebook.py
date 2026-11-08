# %% [markdown]
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berdakh/MLBOOTCAMP/blob/main/part3-deep-learning-systems/ch42-pretrained-transformers-and-tokenization/notebook.ipynb)
#
# *Runs in a fresh Google Colab runtime: click the badge, choose Runtime → Change runtime type → T4 GPU, then Runtime → Run all. The first code cell sets everything up.*

# %% [markdown]
# # Chapter 42 · Pretrained transformers & tokenisation
#
# **Part III · Chapter 42 · about 90 minutes · T4 GPU (runs on CPU with a reduced budget)**
#
# In practice a transformer is rarely trained from scratch: one downloads a
# pretrained model, and the work lies in connecting it correctly to the data. Most
# production bugs with pretrained models are not in the model itself but in
# tokenisation, padding, truncation, pooling and loading. This lab covers these
# steps, and how to make batched inference fast, which at scale is a question of cost.
#
# After this lab you should be able to:
#
# 1. explain subword tokenisation (WordPiece, byte-level BPE), special tokens, attention masks, padding side and truncation;
# 2. run a pretrained encoder and read its hidden states;
# 3. pool token embeddings into a sentence embedding (CLS vs mask-aware mean) and say why mask-awareness matters;
# 4. measure inference throughput against batch size and sequence length;
# 5. cut padding waste with dynamic padding and length bucketing, and measure the speed-up;
# 6. load models safely (safetensors, pinned revisions) and find the licence on the model card.
#
# **You will produce:** `artifacts/level3/nb04/throughput.json` (throughput and padding
# measurements on your device).
#
# **Models** (both downloaded from the Hugging Face Hub, pinned to a revision):
# [`distilbert/distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased)
# (Apache-2.0; 66M parameters; the full model) and
# [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2) (MIT; tokenizer only).
# **Data:** a sample of [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) for
# realistic text lengths (licence: see its card; for research and teaching use).
#
# **Measured here vs on a T4.** Our verification runs are CPU-only. Numbers in
# **Expected:** lines were printed by a CPU run unless marked *"on a T4 you should
# see … (not measured in our verification runs)"*.

# %% [markdown]
# Read first: [Chapter 42 text](README.md) explains the ideas behind this lab (Pretrained transformers & tokenisation) in more depth; [solutions](solutions.md) has worked answers.

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
# --- Catch up: download (once) the tokenizers and the model --------------------
import json
import time

import torch
from transformers import AutoModel, AutoTokenizer

from scaleml import profiling as prof
from scaleml.device import autocast, count_parameters, seed_everything, sync
from grading import ch42 as grade

if DEVICE.type == "cpu":
    torch.set_num_threads(min(2, os.cpu_count() or 1))
OUT = ART / "level3" / "nb04"
OUT.mkdir(parents=True, exist_ok=True)
seed_everything(0)

BERT_ID, BERT_REV = "distilbert/distilbert-base-uncased", "12040accade4e8a0f71eabdb258fecc2e7e948be"
GPT2_ID, GPT2_REV = "openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"
for model_id in (BERT_ID, GPT2_ID):
    print(f"model card: https://huggingface.co/{model_id}")

t0 = time.time()
bert_tok = AutoTokenizer.from_pretrained(BERT_ID, revision=BERT_REV)
gpt2_tok = AutoTokenizer.from_pretrained(GPT2_ID, revision=GPT2_REV)
bert = AutoModel.from_pretrained(BERT_ID, revision=BERT_REV, use_safetensors=True).to(DEVICE).eval()
print(f"loaded in {time.time() - t0:.0f} s | {type(bert).__name__}: {count_parameters(bert) / 1e6:.1f}M parameters")

# %% [markdown]
# **Expected:** two model-card links, then `DistilBertModel: 66.4M parameters`, loaded in
# a few seconds once cached (the first download is ~270 MB). Recent `transformers`
# versions print a *load report* listing `vocab_*` weights as "unexpected": those belong to
# the masked-language-modelling head, which `AutoModel` (the bare encoder) does not use.

# %% [markdown]
# ## Key terms
#
# | Term | Plain meaning | Example here |
# |---|---|---|
# | Subword tokenisation | splitting text into frequent pieces: common words whole, rare words in parts | `un ##bel ##ie ##va ##bly` |
# | WordPiece / byte-level BPE | BERT's and GPT-2's ways of learning those pieces from a corpus | 30,522 vs 50,257 tokens |
# | Token id | the integer index of a piece in the vocabulary; what the model actually receives | `[CLS]` = 101, `[SEP]` = 102, `[PAD]` = 0 |
# | Padding | filler tokens that make every row of a batch the same length | `padding=True` pads to the longest row |
# | Attention mask | 1 for real tokens, 0 for padding; masked keys get −∞ before the softmax | `batch["attention_mask"]` |
# | Truncation | cutting sequences longer than the model's limit | `max_length=16`; DistilBERT's limit is 512 |
# | Hidden state | the contextual vector the encoder outputs for each token | `last_hidden_state`, `(batch, seq, 768)` |
# | Pooling | turning per-token vectors into one vector per text | CLS (position 0) vs mask-aware mean |
# | Cosine similarity | the angle-based similarity of two vectors, from −1 to 1 | 0.89 vs 0.69 for related vs unrelated pairs |
# | Length bucketing | grouping texts of similar length into the same batch to cut padding | 59% → 17% waste in 42.5 |
# | Safetensors | a weight format that stores only tensors, so loading cannot run code | `use_safetensors=True` |
# | Revision pinning | loading a model at a fixed commit so files cannot change under you | `revision=BERT_REV` |

# %% [markdown]
# ## 42.1 · Subword tokenisation
#
# Word-level vocabularies cannot cover new words; character-level ones make
# sequences long. Subword tokenizers split rare words into frequent pieces:
#
# * **WordPiece** (BERT, DistilBERT): a fixed vocabulary (30,522 here) learned to maximise
#   the training-corpus likelihood; continuation pieces are marked `##`; this checkpoint
#   is *uncased* (lower-cases and strips accents first).
# * **Byte-level BPE** (GPT-2, most LLMs): starts from the 256 bytes and merges the most
#   frequent pairs (50,257 tokens); any string can be encoded, nothing is ever
#   "unknown"; `Ġ` marks a leading space.

# %%
sentence = "Tokenization isn't magic: unbelievably, EEG-based BCIs work in 2024!"
for name, tk in [("distilbert (WordPiece)", bert_tok), ("gpt2 (byte-level BPE)", gpt2_tok)]:
    pieces = tk.tokenize(sentence)
    print(f"{name}: {len(pieces)} tokens, vocab {tk.vocab_size:,}")
    print("  ", pieces)
enc = bert_tok(sentence)
print("distilbert input_ids:", enc["input_ids"])
print("with special tokens: ", bert_tok.convert_ids_to_tokens(enc["input_ids"])[:4], "...",
      bert_tok.convert_ids_to_tokens(enc["input_ids"])[-2:])
print("decode:", bert_tok.decode(enc["input_ids"]))

# %% [markdown]
# **Expected:** DistilBERT splits the sentence into 24 lower-cased WordPieces
# (`token`, `##ization`, `isn`, `'`, `t`, …, `un`, `##bel`, `##ie`, `##va`, `##bly`, …,
# `202`, `##4`) and wraps them in `[CLS] … [SEP]` (ids 101 and 102); decoding gives back
# *lower-cased* text with changed spacing (`isn ' t`): the tokenizer is lossy. GPT-2 needs
# 17 pieces, keeps the case and the spaces (`Ġunbelievably`, `Ġ2024`), and decodes back to
# exactly the input. Note how differently the two split numbers and acronyms.
#
# Practical consequences: sequence-length limits are in **tokens**, not words;
# numbers, code, non-English text and domain jargon tokenise into many more pieces;
# and a model only works with the exact tokenizer it was trained with.

# %%
sample = ("The European Central Bank kept interest rates unchanged on Thursday, "
          "saying inflation risks remained balanced as growth slowed across the euro zone.")
for name, tk in [("distilbert", bert_tok), ("gpt2", gpt2_tok)]:
    n = len(tk.tokenize(sample))
    print(f"{name:10}: {n} tokens for {len(sample.split())} words ({len(sample) / n:.1f} characters/token)")

# %% [markdown]
# **Expected:** 24 tokens for 22 words (6.2 characters per token) for both:
# ordinary news English is cheap. The common rule of thumb "1 token ≈ ¾ of a word"
# comes from mixed web text.
#
# ## 42.2 · Batches: padding, attention masks, truncation
#
# A batch must be a rectangle. Shorter sequences are **padded** with a pad token,
# and the **attention mask** (1 = real token, 0 = padding) tells the model to ignore
# the padding: inside attention, masked keys get −∞ before the softmax.
# Sequences longer than `max_length` are **truncated** (DistilBERT's limit is 512
# positions). Two padding choices:
#
# * `padding=True` (dynamic): pad to the longest sequence *in this batch*;
# * `padding="max_length"`: pad every batch to `max_length`, wasteful unless
#   you need fixed shapes (some compilers and accelerators do).
#
# **Padding side.** Encoders (BERT) pad on the right. Decoder-only models
# (GPT-2) used for *generation* must pad on the **left**, so the last position
# of every row is a real token from which to continue. GPT-2 has no pad token; the
# usual fix is to reuse the end-of-text token as padding (the attention mask makes it
# harmless).

# %%
texts = ["Short one.", "A somewhat longer sentence about central banks and interest rates.",
         "Medium length text here."]
batch = bert_tok(texts, padding=True, truncation=True, max_length=16, return_tensors="pt")
print({k: tuple(v.shape) for k, v in batch.items()})
print(batch["input_ids"])
print(batch["attention_mask"])
print("longest row decoded:", bert_tok.decode(batch["input_ids"][1]))

gpt2_tok.pad_token = gpt2_tok.eos_token  # GPT-2 ships without a pad token
gpt2_tok.padding_side = "left"  # for generation
left = gpt2_tok(texts, padding=True, return_tensors="pt")
print("gpt2, left-padded attention mask:\n", left["attention_mask"])

# %% [markdown]
# **Expected:** shapes `(3, 13)`: dynamic padding to the longest row, which has
# 13 tokens including `[CLS]` and `[SEP]` (under the 16 limit). The other two rows are padded
# with id 0 (`[PAD]`) and their masks end in zeros. The GPT-2 mask has its zeros on the
# **left**. Try `max_length=8` to see truncation cut the longest row and still end it with `[SEP]`.
#
# ## 42.3 · The forward pass, hidden states and pooling
#
# `AutoModel` returns the encoder without a task head: `last_hidden_state` is
# `(batch, seq, 768)`, one contextual vector per token. With
# `output_hidden_states=True` you also get the embeddings and every layer's output
# (7 tensors for DistilBERT's 6 layers): lower layers are more lexical, upper layers
# more task- and context-specific.
#
# To get **one vector per sentence**, pool:
#
# * **CLS pooling**: take position 0. Meaningful only if the model was trained so
#   that this vector summarises the input (BERT's next-sentence objective, or a
#   fine-tuned classifier). DistilBERT was distilled *without* that objective.
# * **Mean pooling**: average the token vectors, **using the attention mask** so
#   padding does not count. Sentence-transformer models are trained this way.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-text-to-vector.svg" alt="Seven steps from text to tokens, ids, embeddings, a six-layer encoder, hidden states of shape B by T by 768, and mask-aware mean pooling to one 768-dimensional vector" width="100%">
#
# *What to notice:* the tokenizer and the pooling step are outside the model, and both can silently break it: a different tokenizer gives different ids, a mask-unaware mean gives a different vector.
#
# > **Intuition.** Subword tokenisation works like a box of building bricks: a fixed set of common pieces
# > (frequent words kept whole, rare words built from smaller bricks) can build any word. *Where
# > it breaks down:* the piece boundaries are statistical, not linguistic (`2024` → `202`, `##4`),
# > which is one reason models are bad at arithmetic and spelling.
#
# > **Intuition.** Mean pooling is a vote of every token in the room; the attention mask makes
# > sure the empty seats (padding) do not vote. A worked example: hidden values 2 and 4 for two
# > real tokens and 100 for a pad give a masked mean of 3 and a naive mean of 35.3.

# %%
batch = bert_tok(texts, padding=True, return_tensors="pt").to(DEVICE)
with torch.inference_mode():
    out = bert(**batch, output_hidden_states=True)
print("last_hidden_state:", tuple(out.last_hidden_state.shape), "| hidden states:", len(out.hidden_states))

# The same sentence, alone vs padded inside a batch: a naive mean changes, a masked mean does not.
alone = bert_tok(texts[0], return_tensors="pt").to(DEVICE)
with torch.inference_mode():
    h_alone = bert(**alone).last_hidden_state
h_batch, m = out.last_hidden_state, batch["attention_mask"]
naive_alone, naive_batch = h_alone.mean(1)[0], h_batch.mean(1)[0]
masked_batch = (h_batch * m.unsqueeze(-1)).sum(1)[0] / m[0].sum()
cos = torch.nn.functional.cosine_similarity
print(f"cos(naive mean alone, naive mean padded)  = {cos(naive_alone, naive_batch, dim=0):.4f}")
print(f"cos(naive mean alone, masked mean padded) = {cos(naive_alone, masked_batch, dim=0):.4f}")
print(f"max |difference|, masked mean           = {(naive_alone - masked_batch).abs().max():.1e}")

# %% [markdown]
# **Expected:** the naive mean of the padded copy is a *different* vector (cosine
# 0.85 in our run: padding positions still have hidden states, and the
# naive mean averages them in), while the masked mean matches the unpadded sentence up
# to ~1e-6 (our run: 6.1e-07). A model whose embeddings change with whatever else is in the batch gives
# different answers online (batch of 1) and offline (big batches): a typical
# train/serve skew bug.

# %% [markdown]
# > **Exercise 1.** Write `mean_pool(last_hidden, attention_mask)`: `last_hidden` is
# > `(B, T, H)`, `attention_mask` is `(B, T)` of 0/1 integers. Return `(B, H)`: the mean of
# > the hidden states at positions where the mask is 1. A row with no tokens must give a
# > finite result (zeros), not NaN.

# %%
def mean_pool(last_hidden, attention_mask):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex1(mean_pool)

# %% [markdown]
# Is CLS or mean pooling a better *sentence embedding* for an off-the-shelf
# DistilBERT? Measure on a few pairs: related pairs should be more similar than
# unrelated ones. (A tiny, hand-made set: a sanity check, not a benchmark. Chapter 45
# evaluates retrieval properly with models trained for it.)

# %%
pairs = [
    ("A man is playing a guitar.", "Someone is playing an instrument.", 1),
    ("The stock market fell sharply today.", "Shares dropped steeply this afternoon.", 1),
    ("The cat sleeps on the sofa.", "A kitten is napping on the couch.", 1),
    ("He is cooking pasta for dinner.", "Dinner tonight is spaghetti he made.", 1),
    ("A man is playing a guitar.", "Shares dropped steeply this afternoon.", 0),
    ("The stock market fell sharply today.", "A kitten is napping on the couch.", 0),
    ("The cat sleeps on the sofa.", "Dinner tonight is spaghetti he made.", 0),
    ("He is cooking pasta for dinner.", "Someone is playing an instrument.", 0),
]


def embed(sentences, pooling):
    enc = bert_tok(sentences, padding=True, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        h = bert(**enc).last_hidden_state
    if pooling == "cls":
        return h[:, 0]
    mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)  # the mask-aware mean, as in Exercise 1
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


for pooling in ("cls", "mean"):
    a = embed([p[0] for p in pairs], pooling)
    b = embed([p[1] for p in pairs], pooling)
    sims = cos(a, b, dim=1)
    labels = torch.tensor([p[2] for p in pairs], device=sims.device)
    rel, unrel = sims[labels == 1], sims[labels == 0]
    print(f"{pooling:4} pooling: related {rel.mean():.3f} | unrelated {unrel.mean():.3f} | gap {rel.mean() - unrel.mean():.3f}"
          f" | related > every unrelated: {int((rel[:, None] > unrel[None, :]).all(1).sum())}/{len(rel)}")

# %% [markdown]
# **Expected (measured):** both poolings give high cosine similarity to *everything*
# (raw BERT embeddings occupy a narrow cone, so absolute cosine values mean little):
# CLS gave 0.96 for related and 0.90 for unrelated pairs (gap 0.06), mean pooling 0.89 vs
# 0.69 (gap 0.20). Both rank every related pair above every unrelated one on this easy
# set, but mean pooling separates them about 3× more clearly. This is why off-the-shelf sentence embeddings come
# from models *trained* for similarity (Chapters 45–46), and why a classifier on top of CLS
# works only after fine-tuning (Chapter 43).
#
# ## 42.4 · Inference throughput: batch size and sequence length
#
# Throughput (sequences/s) rises with batch size until the hardware is saturated,
# then flattens while latency per batch keeps growing. Cost per sequence grows
# with length: linearly for the projections and MLP, quadratically for attention
# scores (which matter only at long lengths for a model of this width).

# %%
def throughput(batch_size, seq_len, min_seconds=1.0):
    ids = torch.randint(1000, 20000, (batch_size, seq_len), device=DEVICE)
    mask = torch.ones_like(ids)

    def run():
        with torch.inference_mode(), autocast(DEVICE):  # fp16 on a T4, fp32 on CPU
            bert(input_ids=ids, attention_mask=mask)

    r = prof.benchmark(run, DEVICE, warmup=1, iters=2, min_seconds=min_seconds)
    return {"batch": batch_size, "seq": seq_len, "ms_per_batch": r["median_s"] * 1e3,
            "seq_per_s": batch_size / r["median_s"], "tok_per_s": batch_size * seq_len / r["median_s"]}


IS_CUDA = DEVICE.type == "cuda"
SEQ = BUDGET.max_seq_len  # 64 on CPU, 128 on GPU
grid = [throughput(b, SEQ) for b in ([1, 8, 32, 128] if IS_CUDA else [1, 8, 32])]
grid += [throughput(16, s) for s in ([32, 128, 512] if IS_CUDA else [16, 64, 128])]
print(f"{'batch':>5} {'seq':>4} {'ms/batch':>9} {'seq/s':>8} {'tokens/s':>9}")
for r in grid:
    print(f"{r['batch']:>5} {r['seq']:>4} {r['ms_per_batch']:>9.1f} {r['seq_per_s']:>8.1f} {r['tok_per_s']:>9.0f}")

# %% [markdown]
# **Expected (CPU, 2 threads, measured):** batch 1 at 64 tokens is the least
# efficient (~1,300 tokens/s in our run); batch 8 and 32 reach ~1,900–2,000 tokens/s and
# flatten: two CPU cores saturate quickly. At batch 16, tokens/s stays roughly flat
# (2,100–2,500) from 16 to 128 tokens, so ms per batch grows about linearly with length:
# at these lengths attention's quadratic term is small next to the MLPs.
#
# **On a T4 you should see** tens of thousands of tokens/s with fp16 autocast, and a much
# larger gain from batching (batch 1 leaves most of the GPU idle; batch 32–128 at
# 128 tokens fills it), with attention's quadratic cost becoming visible at 512 tokens
# (not measured in our verification runs).
#
# ## 42.5 · Padding waste: dynamic padding and length bucketing
#
# Real text lengths vary widely. Every pad token costs the same compute as a
# real one, so the batching strategy can matter more than any model optimisation.
# We embed AG News texts three ways: padding everything to `max_length`; dynamic
# padding (to the longest text in each batch) in the data's own order; and dynamic
# padding after **sorting by length**, so each batch holds texts of similar length.

# %% [markdown]
# <img src="https://raw.githubusercontent.com/berdakh/MLBOOTCAMP/main/figures/l3-padding-bucketing.svg" alt="The same six texts batched three ways: padding to 128 wastes 52 percent of positions, dynamic padding in data order 41 percent, sorting by length first 25 percent" width="100%">
#
# *What to notice:* the model and the data are identical in all three panels; only the grouping of rows changes, and the dashed (wasted) area shrinks as similar lengths share a batch.

# %%
from scaleml.textdata import load_text_classification

N_TEXTS = 2000 if IS_CUDA else 192
MAX_LEN = 2 * BUDGET.max_seq_len  # 128 on CPU, 256 on GPU
_, news, label_names = load_text_classification("ag_news", n_train=100, n_eval=N_TEXTS, seed=0)
news_texts = news["text"].tolist()
lengths = [len(ids) for ids in bert_tok(news_texts, truncation=True, max_length=MAX_LEN)["input_ids"]]
print(f"{len(news_texts)} AG News texts | tokens: min {min(lengths)}, median {sorted(lengths)[len(lengths) // 2]}, "
      f"max {max(lengths)} (truncated at {MAX_LEN})")


def run_batches(batches, pad_to_max):
    """Embed every text once; return seconds and the number of token positions processed."""
    positions = 0
    sync(DEVICE)
    start = time.perf_counter()
    for idx in batches:
        enc = bert_tok([news_texts[i] for i in idx], truncation=True, max_length=MAX_LEN, return_tensors="pt",
                       padding="max_length" if pad_to_max else True).to(DEVICE)
        positions += enc["input_ids"].numel()
        with torch.inference_mode(), autocast(DEVICE):
            bert(**enc)
    sync(DEVICE)
    return time.perf_counter() - start, positions


BS = BUDGET.batch_size
in_order = [list(range(i, min(i + BS, len(lengths)))) for i in range(0, len(lengths), BS)]
by_length = sorted(range(len(lengths)), key=lengths.__getitem__)
bucketed = [by_length[i:i + BS] for i in range(0, len(by_length), BS)]
real = sum(lengths)
padding_rows = {}
for name, batches, pad_max in [("pad to max_length", in_order, True), ("dynamic padding", in_order, False),
                               ("sorted by length", bucketed, False)]:
    seconds, positions = run_batches(batches, pad_max)
    padding_rows[name] = {"seconds": seconds, "positions": positions, "pad_fraction": 1 - real / positions}
    print(f"{name:18}: {seconds:5.1f} s | {positions:7,} positions, {1 - real / positions:5.1%} padding")
base = padding_rows["pad to max_length"]["seconds"]
print(f"speed-up vs pad-to-max: dynamic {base / padding_rows['dynamic padding']['seconds']:.1f}x, "
      f"sorted {base / padding_rows['sorted by length']['seconds']:.1f}x")

(OUT / "throughput.json").write_text(json.dumps({"device": describe(DEVICE), "grid": grid,
                                                 "padding": padding_rows}, indent=2))
print("wrote", OUT / "throughput.json")

# %% [markdown]
# **Expected (CPU, measured):** the sampled AG News texts are 24–128 tokens (median 50;
# a few hit the 128 limit). Padding to 128 wastes 59% of the positions; dynamic padding
# in the data's order still wastes 50% (each batch of 32 almost always contains one long
# text); sorting by length cuts it to 17%. Time follows the positions processed: in our runs
# 10.4–13.5 s → 8.4–8.8 s → 5.0 s, i.e. dynamic padding 1.2–1.5× and sorting 2.1–2.7× faster
# than padding to the maximum.
#
# **On a T4 you should see** the same ordering; the absolute speed-up can be smaller for
# short texts, where a GPU is under-used anyway, and larger at long lengths (not
# measured in our verification runs).
#
# **For training**, sorting the whole dataset by length is a problem: every batch would
# contain similar-length texts in the same order every epoch, which biases gradient
# steps (and lengths often correlate with labels). The standard compromise: shuffle,
# take a large chunk (e.g. 100 batches' worth), sort within the chunk, cut batches,
# then shuffle the *batches*. Inference has no such constraint: sort freely, then
# restore the original order of the outputs.

# %% [markdown]
# > **Exercise 2.** Write `bucket_by_length(lengths, batch_size)`: given token lengths, return a
# > list of batches (lists of indices into `lengths`) that uses every index exactly once,
# > uses `ceil(n / batch_size)` batches of at most `batch_size`, and minimises the padding
# > waste (`Σ_batches max_len × batch_len − Σ lengths`). The grader requires a waste no worse
# > than sorting by length and chunking.

# %%
def bucket_by_length(lengths, batch_size):
    # YOUR CODE HERE
    raise NotImplementedError


grade.ex2(bucket_by_length)

# %% [markdown]
# ## 42.6 · Loading models safely: model cards, licences, safetensors, revisions
#
# Before using a pretrained model in a product, read its **model card**: intended
# use, training data, known biases and limitations, evaluation, and **licence**.
# DistilBERT is Apache-2.0 and GPT-2 is MIT, both permissive; many models are not
# (research-only, non-commercial, or custom licences with use restrictions), and
# a dataset's licence can restrict a model trained on it.
#
# * **Safetensors.** Legacy `pytorch_model.bin` files are Python pickles: loading
#   one can execute arbitrary code. `.safetensors` files are plain tensors with a
#   JSON header: no code execution, and memory-mappable (fast loading).
#   `use_safetensors=True` makes loading fail rather than fall back to a pickle.
# * **Revision pinning.** `from_pretrained(id)` fetches whatever the `main` branch is
#   *today*; the owner can change weights or tokenizer files at any time. Pin a commit hash
#   (`revision=...`), as this notebook does, for reproducible results, and treat a revision
#   change like a dependency upgrade: re-run your evaluation.
# * **`trust_remote_code=True`** runs Python code from the model repository on your
#   machine. Leave it off unless you have reviewed that code (and pin the revision when
#   you must use it).

# %%
from huggingface_hub import HfApi

api = HfApi()
for model_id, rev in [(BERT_ID, BERT_REV), (GPT2_ID, GPT2_REV)]:
    info = api.model_info(model_id, revision=rev)
    weights = sorted(s.rfilename for s in info.siblings
                     if s.rfilename.endswith((".safetensors", ".bin", ".h5", ".msgpack", ".ckpt")))
    licence = (info.card_data or {}).get("license") if info.card_data else None
    print(f"{model_id} @ {rev[:7]} | licence: {licence} | https://huggingface.co/{model_id}")
    print("   weight files:", weights)

# %% [markdown]
# **Expected:** `apache-2.0` for DistilBERT and `mit` for GPT-2, and four weight
# files per repository (`model.safetensors` next to the legacy pickle `pytorch_model.bin`,
# TensorFlow `tf_model.h5` and Flax `flax_model.msgpack`): `use_safetensors=True` picks the safe one.

# %% [markdown]
# > **Exercise 3.** A teammate's semantic-search service embeds queries one at a time online,
# > and embeds the document corpus offline in batches of 256 padded to `max_length=512`, with
# > `last_hidden_state.mean(dim=1)` as the embedding. They also load
# > `AutoModel.from_pretrained("some-org/encoder")` without a revision. List every problem you
# > see (correctness, cost, reproducibility, safety), with the fix for each. Write your answer
# > in the cell below.

# %% [markdown]
# *Your answer (Exercise 3):* ...

# %% [markdown]
# ## Key equations
#
# **1. Mask-aware mean pooling.**
#
# $$
# \mathbf{s} = \frac{\sum_{t=1}^{T} m_t\, \mathbf{h}_t}{\sum_{t=1}^{T} m_t}
# $$
#
# In words: average the hidden vectors of the real tokens only. Symbols: $\mathbf h_t$ = hidden
# state at position $t$; $m_t \in \{0, 1\}$ = attention mask. (In code, guard the denominator
# against 0 for an empty row.) Example: values $(2, 4, 100)$ with mask $(1, 1, 0)$ give 3; the
# naive mean gives 35.3.
#
# **2. The attention mask inside attention.**
#
# $$
# \text{score}_{ij} \;\leftarrow\; \text{score}_{ij} + (1 - m_j)\cdot(-\infty)
# $$
#
# In words: a padded key $j$ gets $-\infty$, so after the softmax its weight is exactly 0 for
# every query. Example: scores $(1.0, 0.5, 2.0)$ with mask $(1, 1, 0)$ → softmax over the first two
# only: $(0.62, 0.38, 0)$.
#
# **3. Cosine similarity.**
#
# $$
# \cos(\mathbf a, \mathbf b) = \frac{\mathbf a \cdot \mathbf b}{\lVert \mathbf a \rVert \, \lVert \mathbf b \rVert}
# $$
#
# In words: the angle between two vectors, ignoring their lengths; for unit vectors it is just
# the dot product. Example: $(1, 0)$ and $(1, 1)$: $1 / \sqrt 2 \approx 0.71$.
#
# **4. Padding waste.**
#
# $$
# \text{waste} = 1 - \frac{\sum_i \ell_i}{\sum_{\text{batches } b} \lvert b \rvert \cdot \max_{i \in b} \ell_i}
# $$
#
# In words: the share of computed positions that are padding. Symbols: $\ell_i$ = length of text
# $i$; $\lvert b\rvert$ = rows in batch $b$. Example (the figure above): 370 real tokens in
# $6 \times 128 = 768$ positions: waste $= 1 - 370/768 = 52\%$; sorted, $3 \cdot 45 + 3 \cdot 120 = 495$
# positions: 25%.

# %% [markdown]
# ## Troubleshooting
#
# | Symptom | Likely cause | Fix |
# |---|---|---|
# | `ModuleNotFoundError: churnlab` / `scaleml` | setup cell not run | run the first code cell |
# | `ValueError: Asking to pad but the tokenizer does not have a padding token` | GPT-2 has no pad token | `tok.pad_token = tok.eos_token` |
# | `Token indices sequence length is longer than the specified maximum` | text longer than 512 tokens, no truncation | `truncation=True, max_length=...` |
# | Embeddings differ between batch and single-example inference | naive mean over padding, or right-padding a decoder | mask-aware pooling; left-pad decoders for generation |
# | Load report lists "UNEXPECTED" / "MISSING" weights | loading a checkpoint into a different head (e.g. MLM weights into `AutoModel`, or a new classifier head) | expected when the head changes; a *missing encoder* weight is a real problem |
# | `OSError` / timeouts downloading from the Hub | network, Hub outage, rate limit | retry; set `HF_TOKEN`; the cache (`HF_HOME`) avoids re-downloads |
# | `use_safetensors=True` fails | the repo has only pickle weights | prefer another revision/repo; if you must, load only from a trusted source |
# | Throughput numbers jump around | shared CPU, first-call overhead | warm up, repeat, compare within one session |

# %% [markdown]
# ## Checkpoint
#
# Answer each question without notes; a complete answer takes about a minute:
#
# 1. Why do modern models use subword tokenisation? Compare WordPiece and byte-level BPE, and name a failure mode of tokenisation (numbers, other languages, code).
# 2. What is the attention mask for, and what goes wrong if you pad a decoder-only model on the right before generating?
# 3. CLS pooling vs mean pooling: when is each appropriate, and why must mean pooling be mask-aware?
# 4. You need to embed 100M documents with a BERT-sized encoder. What determines throughput, and how do you maximise it (batching, padding strategy, precision, sequence length, hardware)?
# 5. How do you load a third-party model safely and reproducibly in production?
# 6. Your classifier's accuracy dropped after a dependency upgrade, with no code change. What could have changed in the tokenizer or model loading, and how would you have caught it?
#
# **Next chapter:** [Chapter 43 · Fine-tuning a classifier](../ch43-fine-tuning-a-classifier/notebook.ipynb) — put a head on this encoder and fine-tune it, against a linear probe and a TF-IDF baseline.
