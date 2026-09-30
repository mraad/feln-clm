# feln-clm

Text → FELN over an OKF project catalog with a LoRA-tuned Qwen3 whose decoding is
constrained to the catalog. The model writes a query as a few short pieces, each one
choice from a closed set (layer, subtype, condition, relation + distance); SQL is rendered
from the OKF hints, never generated. Training and cross-validation run on a CUDA host
(gc5); inference runs on the Mac (MLX). The goal is accuracy, precision and recall.

v1 of this repo (CLM: frozen Qwen3-8B embeddings + choice heads answering four rounds of
typed questions) is in git history at `441c423`; its numbers stay in the table below.

## How it works

```
request ──► spans: quoted strings, numbers, distances (number + unit)
        ──► hints: where each quoted literal occurs in the project data
            ("'GASSCO AS' = Pipelines current operator", from values.json),
            and which column a synonym names ("'depth' = Wells well water depth", OKF "Also called")
        ──► prompt:  Request: … / Values: … / Columns: … / FELN:
        ──► constrained beam search (8 beams) over pieces:
               Pipelines [unknown] where current operator is 'GASSCO AS'; within 5 miles of Wells [any].
               └ head: layer [subtype]  └ condition (column, op, literal)   └ head: relation, distance, layer [subtype]
            each step allows only tokens that continue a legal piece (token trie per slot):
              · layers and subtypes from the OKF, ≤ 3 distinct layers
              · conditions from the column's kind (LIKE / UPPER / codes / yes-no / numbers / years)
              · literals and distances are spans of the request; a literal serves one condition
              · several distances in the request → each relation takes a different one
        ──► FELN {layers, where, relations}; SQL rendered by grammar.compose (casts, LIKE, codes)
```

Every output is a valid FELN over the catalog. `confidence` is the probability of the
winning sequence (renormalised over legal tokens); the other finished beams are returned
as `alternatives`. Pieces are tokenized one at a time in training and decoding, so gold is
always reachable (the oracle test decodes all 3,000 FELN.json queries back to gold).

## Results

### 200 dev requests (feln-laya's held-out split; never trained on, no recipe choice made on it)

| | **feln-clm v2** (Qwen3-4B + LoRA; Mac MLX run) | v1 (CLM, 5 heads) | feln-laya | feln-lora v2 |
|---|---:|---:|---:|---:|
| Exact FELN (`FELN.same`) | **91.5%** | 81.5% | 74.0% | 60.0%¹ |
| — a gold layer never named in the text (110) | **86.4%** | 73.6% | — | — |
| Atom precision / recall (layers, predicates, relations) | **0.961 / 0.962** | 0.939 / 0.924 | — | — |
| Accuracy / coverage at confidence ≥ 0.8 | **95.9% / 85.0%** | 92.1% / 63.0% | 91.6% / 53.5% | — |
| Accuracy / coverage at confidence ≥ 0.9 | **98.1% / 80.0%** | 94.8% / 48.0% | — | — |
| Executed on NorthSea.gdb: same feature set | **95.0%** | 94.0% | 92.0% | 84.0% |
| — same set, requests with non-empty gold (77) | **88.3%** | 85.7% | 81.8% | 66.2% |
| — mean Jaccard | **0.961** | 0.955 | 0.942 | 0.859 |
| — macro precision / recall of feature IDs | **0.952 / 0.937** | 0.898 / 0.922 | 0.924 / 0.908 | 0.831 / 0.734 |
| — micro precision / recall of feature IDs | 0.531 / 0.696 | 0.462 / 0.738 | **0.687 / 0.806** | 0.646 / 0.715 |
| Invalid outputs | 0 | 0 | — | 20 |
| Median latency | 0.59 s (gc5) · 1.56 s (Mac MLX; p90 2.5 s) | 4.0 s (Mac) | 0.16 s | 2.4 s |

¹ after ILIKE→LIKE and INT→INTEGER normalisation (49.5% raw). feln-lora (Nemotron-4B LoRA,
free-form JSON under a shape-only grammar) was trained on templated questions and its own
catalog, so this is its transfer to these reworded requests, not its in-domain score.

Mac MLX and gc5 torch predictions agree on 199/200 dev requests (one near-tie flips). The
first v2 training, before column synonyms, also scored 91.5% exact, with execution 95.5%
(non-empty 89.6%, micro precision 0.654): retraining reshuffled six near-ties (two fixed,
two broken, among them one coin flip that now returns a large feature set). Micro feature precision/recall is dominated by a few requests with very large
result sets (a wrong subtype on a wells query returns ~1,900 wells); per request (macro)
v2 leads on both. Execution metrics run gold and predicted FELN with DuckDB spatial on the
GDB (EPSG:3035 metres); 123 of 200 gold queries return no features.

### 5-fold cross-validation (the other 2,800 FELN.json requests; laya-generated rows always train)

| | Qwen3-4B | Qwen3-8B |
|---|---:|---:|
| Exact | **88.8%** | 88.8% |
| — every layer named, gold consistent (1,195) | 97.8% | 97.4% |
| — a gold layer never named (1,567) | 83.7% | 84.0% |
| — gold contradicts text (87, `noisy`) | 29.9% | 21.8% |
| Atom precision / recall | 0.951 / 0.950 | 0.952 / 0.950 |
| Accuracy / coverage at confidence ≥ 0.8 | 94.4% / 83.6% | 94.5% / 83.7% |

The sizes tie (54 requests only 4B gets right, 53 only 8B), so the smaller, faster 4B is
the model. An unused-literal penalty (0, 2, 4 nats) changed no answer on fold 0 and was
removed. Nearly all remaining errors are requests that name a related layer only by a
subtype word several layers share, or whose gold contradicts the text.

Why v2 beats v1 (v1's 37 dev misses: 11 undecidable from text, 8 gold contradicting text,
18 fixable): the generator decides layer, relation, distance and subtype jointly, so it
binds "more than 1000 meters from gas/condensate pipelines" to the right layer; the value
hints resolve which column a quoted literal belongs to; and the adapter tunes the LM
itself instead of heads over frozen embeddings.

## Limits

- Grammar = FELN.json's: subtype + at most one extra condition per layer, at most 3
  layers. 715 of 2,857 laya-generated training rows (two conditions) are skipped.
- A related layer named only by a subtype word that several layers share ("within 15 km
  of **oil**": wells, pipelines and discoveries all have *oil*) is a guess; the data
  generator picked one at random. `confidence` is low on most of them, and the
  alternatives list the other readings.
- Some gold contradicts its text (the humanizer changed "discoveries" to "wells", dropped a
  subtype). Rows tagged `noisy` are left out of training and reported separately.
- The adapter is tied to the OKF hash and to `values.json`; a changed catalog or data needs
  `prepare` + retraining (~8 min on one RTX PRO 6000).

## Column synonyms

When requests call a column something other than its OKF alias, list the other names in
the column's Query hints, comma-separated:

```
## `water_depth`

- Depth of the well in meters
- Also called: water depth, depth
```

The prompt then names the column when a request uses one of them, and `prepare` adds
training copies of the requests on that column with the name swapped for each other one
(same gold). Editing the OKF changes its hash: re-run `prepare` and retrain.

With `water_depth` "Also called: water depth, depth", "oil wells with a depth over 100",
"gas wells where depth is less than 70.5" and "wells with depth between 60 and 80" decode
to well water depth at confidence 1.00 (before: "wellbore name is not blank", 0.18–0.43);
fold 0 exact 90.2% → 90.5%, dev unchanged at 91.5%. A comparative ("deeper than 120") is
not a name and is still misread: it needs training requests that say it.

## Setup (Mac)

Keep `../feln` beside this checkout (editable path dependency). Project data and weights
are not in git:

```sh
N="$HOME/Documents/ArcGIS/Projects/NorthSea"
mkdir -p data && cp -R "$N/okf" data/okf && cp "$N/FELN.json" data/FELN.json
cp ../feln-laya/out/expanded-v2/heldout.jsonl data/laya-heldout.jsonl       # the 200 dev texts
cp ../feln-laya/out/expanded-v2/generated.jsonl data/laya-generated.jsonl   # extra training rows
uv sync --extra mlx --extra exec --extra dev
```

A model dir (`feln-clm.json`, `adapter.safetensors`, `values.json`; 128 MB for `models/q4b`) comes from
training below. The base model downloads from Hugging Face on first use.

## Use (Mac)

```sh
uv run feln-clm ask data/okf models/q4b "Find gas/condensate wells within 5 km of injection pipelines."
uv run feln-clm serve data/okf models/q4b          # single-page app: http://127.0.0.1:8710/
uv run feln-clm evaluate data/okf models/q4b out/rows --split dev --output results/x.jsonl
uv run feln-clm execute data/okf results/x.jsonl     # execution precision/recall on the GDB
```

`ask` prints the FELN (`meta`), `status` (`accepted` when `confidence` ≥ `--threshold`,
default 0.5), the pieces with their probabilities, and the alternatives. `evaluate` writes
one JSONL line per request plus `.summary.json` (exact, atom P/R, per-tag exact, selective
accuracy) and refuses to overwrite. `execute` builds `out/project.duckdb` once from the OKF
`resource` feature classes. The app is one static page on the stdlib HTTP server
(loopback, one request at a time): `GET /api/info`, `POST /api/ask {"text": …}`.

Ctrl-C, SIGTERM and SIGHUP stop every command cleanly: `serve` stops accepting
connections, finishes the requests in flight, then exits; `evaluate` keeps its JSONL and
writes a summary of the requests it finished (`"interrupted": "n/N"`). Exit code is
128 + the signal number (130 Ctrl-C, 143 SIGTERM).

## Train and evaluate (gc5)

Prepare rows on the Mac (needs the GDB for `values.json`), copy them over, then train:

```sh
uv run python -m feln_clm.prepare data/okf data/FELN.json data/laya-heldout.jsonl out/rows \
    --extra data/laya-generated.jsonl --db out/project.duckdb
rsync -a out/rows gc5:feln-clm/

# gc5, once: uv venv .venv && uv pip install --python .venv/bin/python torch "transformers>=5" safetensors
#            uv pip install --python .venv/bin/python -e ../feln && uv pip install --python .venv/bin/python --no-deps -e .
python -m feln_clm.train rows models/q4b-f0 --base Qwen/Qwen3-4B --holdout fold0      # CV run
python -m feln_clm.cli evaluate okf models/q4b-f0 rows --split fold0 --backend torch --output results/q4b-f0.jsonl
python -m feln_clm.train rows models/q4b --base Qwen/Qwen3-4B                             # final: all but dev
```

`rows.jsonl` splits: `dev` (the 200 laya held-out texts, never trained on), `fold0`..`fold4`
(the other 2,800 FELN.json requests, for recipe choices), `extra` (laya-generated, training
only). Defaults: LoRA rank 16, alpha 32 on every attention and MLP projection, lr 2e-4
cosine, batch 16, 3 epochs, loss on target tokens only. Copy the model dir (128 MB for 4B) to
the Mac; MLX and torch merge the same adapter into bf16 weights.

## Layout

```
feln_clm/okf.py        OKF markdown → catalog (columns, kinds, domains, hints, GDB resource)
feln_clm/grammar.py    spans, options, decompose / compose / SQL, pieces, slot candidates, hints, prompt, tags
feln_clm/prepare.py    splits + folds, training rows, values.json
feln_clm/train.py      LoRA SFT (torch, CUDA host)
feln_clm/lm.py         base + merged adapter on MLX (Mac) or torch (CUDA): prefill / beam step
feln_clm/decode.py     Translator: constrained beam search over piece tries
feln_clm/evaluate.py   exact / atom P-R / per-tag / selective accuracy;  execute.py: GDB execution P-R
feln_clm/server.py, static/index.html   the single-page app
feln_clm/cli.py        ask · evaluate · execute · serve
```

## Tests

```sh
uv run pytest -q   # span edge cases; oracle: all 3,000 FELN.json queries decode back to gold
uv run ruff check . && uv run ruff format --check .
```
