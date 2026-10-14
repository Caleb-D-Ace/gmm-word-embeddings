# gmm-word-embeddings

A modern PyTorch implementation of [Gaussian Mixture Word Embeddings based on Athiwaratkun and Wilson (ACL 2017)](https://arxiv.org/abs/1704.08424).

This repository is an offline training pipeline that turns a raw text corpus into a language model of multi-sense, probabilistic word distributions: instead of one vector per word, each word is represented as a Gaussian mixture over the embedding space, so words with multiple senses (e.g. "bank") can occupy multiple distinct modes.

## Table of Contents

- [Installation](#installation)
- [How it works](#how-it-works)
- [How to use the resulting word embedding](#how-to-use-the-resulting-word-embedding)
  - [The data's shape](#the-datas-shape)
  - [Looking up a word](#looking-up-a-word)
  - [Comparing two words](#comparing-two-words)
    - [Maximum cosine similarity](#maximum-cosine-similarity)
    - [Energy from Python](#energy-from-python)
    - [Energy from `.npz`, or another language](#energy-from-npz-or-another-language)
- [Training reference](docs/TRAINING.md) — full CLI flags, checkpointing/resuming, and the metrics log (separate doc)
- [Differences from the original implementation](docs/DIFFERENCES.md) (separate doc)
- [Project status & roadmap](docs/ROADMAP.md) (separate doc)


## Installation

1. Install PyTorch matching your hardware. Only install the CUDA build if you have an NVIDIA GPU — the CPU build will run much slower.

   ```bash
   # For CUDA 12.1 (NVIDIA GPU)
   pip install torch --index-url https://download.pytorch.org/whl/cu121

   # For CPU-only
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. Install the remaining dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Choose a corpus source:
   - **Bring your own corpus** — place `.txt`, `.jsonl`, `.csv`/`.tsv`, `.parquet`, or `.arrow` files (any subfolder structure, the first four optionally `.gz`/`.bz2`-compressed) inside `data/raw/`.
   - **Or download one from Hugging Face instead** — skip this step and pass `--hf_dataset` in step 4.

4. Run preprocessing from the project root:

   ```bash
   # Using your own local corpus in data/raw/
   python src/preprocess.py

   # Or downloading a dataset from Hugging Face instead
   python src/preprocess.py --hf_dataset wikimedia/wikipedia --hf_config 20231101.simple
   ```

   This produces `data/processed/sorted_vocab.json` and `data/processed/corpus_index.bin`. Run `python src/preprocess.py --help` for the full list of options, or see the [full preprocessing flag reference](docs/TRAINING.md#preprocessing-flags) (custom `--raw_dir`/`--processed_dir`, `--text_key`, `--hf_split`, `--stopwords`, etc.). Re-run it whenever you change the corpus or the stopword setting, since the vocabulary changes with them.

5. Run training from the project root:

   ```bash
   # With default parameters (reads data/processed/, writes data/model/)
   python main.py

   # Or override any of them
   python main.py --embedding_dim 100 --k 3 --epochs 100 --lr 0.02
   ```

   Every flag has a default, so a bare `python main.py` works once `data/processed/` exists from the preprocessing step. The most commonly adjusted ones:

   | Flag | Default | Meaning |
   |---|---|---|
   | `--embedding_dim` | `50` | Dimensionality (`D`) of each Gaussian component |
   | `--k` | `2` | Number of mixture components (`K`) per word |
   | `--epochs` | `50` | Number of training epochs |
   | `--lr` | `0.01` | Initial learning rate (decays linearly to `--lr_final`) |
   | `--checkpoint_every` | `0` | Save a snapshot every N epochs (`0` disables it) |
   | `--resume_from` | `None` | Continue training from a previous checkpoint directory |

   This produces `data/model/gmm_embeddings.npz` and `data/model/gmm_embeddings.pt`. Run `python main.py --help`, or see the [full training flag reference, checkpointing/resuming, and the metrics log](docs/TRAINING.md) for everything else, including the less commonly changed flags.


## How it works

Training happens in two independent stages:

1. **Preprocessing** (`src/preprocess.py`) — tokenizes a corpus from either local files or a Hugging Face dataset, drops stopwords, and builds a vocabulary of every remaining word occurring at least `MIN_FREQ` times (40 by default — currently a constant at the top of the file, not yet a command-line option). Stopwords come from a built-in English list by default and can be changed with `--stopwords` (see [Differences from the original implementation](docs/DIFFERENCES.md)). The corpus can come from either of two sources:
   - **Local files** (default) — every file under `--raw_dir` (default `data/raw/`), searched recursively (hidden files like `.gitkeep` are skipped). Plain `.txt` files are tokenized directly; `.jsonl`/`.csv`/`.tsv`/`.parquet`/`.arrow` files have a configurable field or column (`--text_key`, default `text`) pulled out of each record first, so a pre-downloaded dump in any of those formats works without conversion. `.txt`/`.jsonl`/`.csv`/`.tsv` may also be `.gz`- or `.bz2`-compressed.
   - **A Hugging Face Hub dataset** (`--hf_dataset`, e.g. `wikimedia/wikipedia`, with optional `--hf_config`/`--hf_split`) — downloaded and cached by the [`datasets`](https://pypi.org/project/datasets/) library instead of being saved into `data/raw/`. `datasets` is an optional dependency, only needed for this path.

   Either way, it writes:
   - `data/processed/sorted_vocab.json` — a single mapping of `word -> (id, frequency count)`, ordered by descending frequency, so a word's position in the file is also its integer id.
   - `data/processed/corpus_index.bin` — the entire corpus re-encoded as a flat binary array of those integer ids (2 bytes each, or 4 when the vocabulary exceeds 65,535 words), for fast memory-mapped reading during training.

2. **Training** (`main.py` / `src/gmm_training.py`) — loads the vocabulary and binary corpus, then trains a `GMMWordEmbedding` model: each word is represented by `K` Gaussian components (mean, variance, and mixture weight) in a `D`-dimensional embedding space. Training uses a skip-gram-style objective — for each (target word, real context word) pair, a negative context word is sampled by unigram frequency^0.75 (as in word2vec), and a max-margin ranking loss pushes the real pair's Gaussian-mixture "energy" (expected likelihood overlap) above the negative pair's. It optimizes with Adam, with a learning rate that decays linearly over the whole run, and can checkpoint and resume mid-run (see [Training reference](docs/TRAINING.md#checkpointing-and-resuming)). Every batch's loss and several diagnostic statistics are written to a CSV log (see [the metrics log](docs/TRAINING.md#metrics-log)). The learned parameters are saved to both `gmm_embeddings.npz` (raw numpy arrays — `mu`, `var`, `mix_weights` — usable from any language) and `gmm_embeddings.pt` (a PyTorch `state_dict`, for loading straight back into `GMMWordEmbedding` in Python).

## How to use the resulting word embedding

Training writes four files into `data/model/` (or wherever `--output_path` pointed):

- **`sorted_vocab.json`** — the `word -> [id, frequency count]` mapping needed to interpret the other files.
- **`gmm_embeddings.pt`** — a PyTorch `state_dict`, for loading straight back into `GMMWordEmbedding` in Python.
- **`gmm_embeddings.npz`** — a NumPy archive with the same trained values as plain arrays, readable from any language.
- **`training_state.pt`** — optimizer/scheduler state used only to resume training (see [Checkpointing and resuming](docs/TRAINING.md#checkpointing-and-resuming)); not needed to use the embedding itself.

You'll always need `sorted_vocab.json` alongside whichever model file you use, since word ids only mean something in the context of that mapping. Which model file to select depends on your environment:

- **Python + PyTorch** — use `gmm_embeddings.pt`.
- **Anything else** (plain NumPy, or another language entirely) — use `gmm_embeddings.npz`. You'll need a library capable of reading the `.npz` zip-of-arrays format for your language, e.g. NumPy itself in Python, or [`cnpy`](https://github.com/rogersce/cnpy) in C++. Search for a library in your language.

### The data's shape

Every array is indexed by word id — the same integer assigned in `sorted_vocab.json`, and the same row order used to size the model when it was trained. Row `i` in every array corresponds to word id `i`:

- **`mu`** — shape `(vocab_size, K, D)`. `mu[i]` gives word `i`'s `K` Gaussian component means, each a `D`-dimensional vector — up to `K` separate "sense locations" for that word.
- **`var`** — shape `(vocab_size, K, D)`. `var[i]` gives the spread of each of those `K` components, per dimension (a diagonal covariance — each dimension is independent within a component).
- **`mix_weights`** — shape `(vocab_size, K)`, each row summing to 1. `mix_weights[i]` gives how much probability mass each of word `i`'s `K` components carries. A dominant weight (e.g. `[0.95, 0.05]`) means the word is effectively single-sense; a more even split (e.g. `[0.55, 0.45]`) means the model is genuinely using both components — this is how it represents polysemy (e.g. "bank" splitting into a financial-institution sense and a riverbank sense), which a standard word2vec-style embedding can't do.

### Looking up a word
Because the data in the model is represented as integers, you need to use both the `gmm_embeddings.npz` file AND the `sorted_vocab.json` file in tandem. The `.npz`  file will pull the relationship information, and `sorted_vocab.json` allows you to translate between id and word string. An example using numpy is given below:

```python
import json
import numpy as np

with open("data/model/sorted_vocab.json") as f:
    vocab = json.load(f)

data = np.load("data/model/gmm_embeddings.npz")
mu, var, mix_weights = data["mu"], data["var"], data["mix_weights"]

word_id, _count = vocab["bank"]
print(mu[word_id])           # shape (K, D) — "bank"'s K component means
print(mix_weights[word_id])  # shape (K,)   — how much weight each sense carries
```
- Step 1: Load the `sorted_vocab` data into a variable `vocab` using json
- Step 2: Load the `.npz` file into a variable `data`, then separate it by the `"mu"`, `"var"`, and `"mix_weights"` fields into separate arrays.
- Step 3: Fetch the word ID (and optional word count) variables for your chosen word (here, that would be `"bank"`).
- Step 4. Using that word ID, you can now get that word's mean, variation, and weight information from their respective arrays.

### Comparing two words

Every word has `K` component means rather than a single vector, so a plain cosine similarity or dot product between two "word vectors" isn't defined. The original paper (Table 2) evaluates three ways of scoring a pair of words, each of which looks across component pairs:

- **Maximum cosine similarity** — the highest cosine similarity between any component of word A and any component of word B, using only the means. The paper found this the best-performing measure on most word-similarity benchmarks, and its nearest-neighbor examples use it too.
- **Minimum Euclidean distance** between any pair of component means. It did better than maximum cosine on a couple of datasets.
- **Expected likelihood energy** — the function training itself optimizes. It also uses `var` and `mix_weights`, so it accounts for how uncertain each sense is. It scored below maximum cosine on most of the paper's similarity benchmarks, but it is the score the model was actually trained to make high for related words.

#### Maximum cosine similarity

Only `mu` is needed, so this works from the `.npz` in any language:

```python
def max_cosine(mu, a, b):
    """Highest cosine similarity between any component of word id a and any component of word id b."""
    A = mu[a] / np.linalg.norm(mu[a], axis=1, keepdims=True)   # (K, D), unit-length rows
    B = mu[b] / np.linalg.norm(mu[b], axis=1, keepdims=True)
    return (A @ B.T).max()

max_cosine(mu, vocab["bank"][0], vocab["river"][0])
```

#### Energy from Python

A `state_dict` is just a dict of tensors — it can only be loaded into an already-constructed instance of the matching class, so you'll need `GMMWordEmbedding`'s class definition available in your own project (copy `src/gmm_word_embedding.py` in, or depend on this repo directly):

```python
import json
import torch
from gmm_word_embedding import GMMWordEmbedding  # copied from src/

with open("data/model/sorted_vocab.json") as f:
    vocab = json.load(f)

# embedding_dim/K must match exactly what you trained with — a mismatch makes
# load_state_dict fail, since the saved tensor shapes won't line up otherwise.
model = GMMWordEmbedding(vocab_size=len(vocab), embedding_dim=50, K=2)
model.load_state_dict(torch.load("data/model/gmm_embeddings.pt"))
model.eval()

bank_id, river_id = vocab["bank"][0], vocab["river"][0]
with torch.no_grad():
    energy = model.forward(torch.tensor([bank_id]), torch.tensor([river_id]))
```

No need to reimplement `log_overlap`/`gmm_energy` yourself — call them, or just `forward`, directly.

#### Energy from `.npz`, or another language

You only have the raw `mu`/`var`/`mix_weights` arrays here, so the energy function has to be implemented yourself. For two words with means/variances `(mu1, var1)` and `(mu2, var2)` (each shape `(K, D)`) and mixture weights `w1`, `w2` (each shape `(K,)`):

1. **Pairwise component log-overlap** — for every pair of components `(k, m)`, one from each word, the expected likelihood kernel between those two Gaussians:

   ```
   log_overlap[k, m] = -0.5 * sum_d( log(var1[k,d] + var2[m,d]) + (mu1[k,d] - mu2[m,d])^2 / (var1[k,d] + var2[m,d]) )
                        - (D / 2) * log(2 * pi)
   ```

2. **Combine into a single energy score**, weighting each component pair by how much mixture weight each side gives it:

   ```
   energy = log( sum_k sum_m( w1[k] * w2[m] * exp(log_overlap[k, m]) ) )
   ```

   Compute this in log space with a `logsumexp` (`energy = logsumexp_{k,m}( log(w1[k]) + log(w2[m]) + log_overlap[k, m] )`) rather than calling `exp` directly. The log-overlaps are large negative numbers (around -60 at `D=50`), and `exp` of those is tiny enough that adding any small epsilon before the `log` would swamp the real value and flatten every energy to the same number.

Higher `energy` means more similar. This is exactly `GMMWordEmbedding.log_overlap`/`gmm_energy` in `src/gmm_word_embedding.py`, if you want to cross-check a port against the reference implementation.
