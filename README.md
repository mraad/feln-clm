# feln-clm

Text → FELN with a fine-tuned **Contrastive Language Model** (CLM) over an OKF project
catalog. A request becomes a few typed multiple-choice questions; CLM (frozen Qwen3-8B
encoder + fine-tuned projection heads) answers them; a joint decoder assembles the best
consistent FELN query. Training and tuning run on a CUDA host (gpu-host, vLLM); inference runs
on the Mac (native MLX). The goal is accuracy, precision and recall, not latency.

## How it works

```
request ──► spans (quoted strings, numbers, distances) + subtype mentions ("oil", "gas shows")
        ──► CLM questions, 4 rounds:
              layers        primary layer + related layers           (12 options)
              subtype       among the subtypes the request names + "any"
              column / op   extra attribute condition, split in two  (≤27, ≤12 options)
              relation      named with the bound subtype ("…relative to the oil pipelines?")
              distance/value which span of the request fills it (only when ambiguous)
        ──► joint decode: sum of log-probs over 3 layer structures, with constraints
              · a subtype name is bound to at most as many layers as the request says it
              · a condition needs a literal the request offers; literals are not shared
              · two distance relations take different distances
              · every quoted literal / bare number left unexplained costs 2 nats
        ──► FELN {layers, where, relations}; SQL rendered from the OKF hints (LIKE, UPPER, casts, codes)
```

Every question is framed as a Qwen3 chat turn that **lists the options** and ends in
`Answer:`, so the pooled last token is the one that predicts the answer. The CLM action
head embeds each option's text. Five heads trained with different seeds share the same
state embeddings; their distributions are averaged (geometric mean).

Reused unchanged: `../CLM` (`clm.schema`, heads, `Engine`, `train/finetune.py --task choice`,
warm-started from CLM-v0.1-8B) and `../feln` (`FELN.same`, `FELNCompare`, `FELNToDuckDB`).
New here: OKF reader, grammar (decompose/compose), MLX + vLLM encoders behind CLM's
`Embedder`, decoder, evaluation, execution metrics, stdlib server + one-page app.

## Results — 200 dev requests (feln-laya's held-out split)

| | feln-laya (BGE-small cross-encoder) | **feln-clm** (5-head ensemble) |
|---|---:|---:|
| Exact FELN (`FELN.same`) | 74.0% | **81.5%** (gpu-host and Mac MLX) |
| 1 / 2 / 3-layer exact | 32/37 · 87/113 · 29/50 | **35/37 · 94/113 · 34/50** |
| Atom precision / recall (layers, predicates, relations) | — | 0.939 / 0.924 |
| Accuracy / coverage at confidence ≥ 0.8 | 91.6% / 53.5% | 92.1% / **63.0%** |
| Accuracy / coverage at confidence ≥ 0.9 | — | 94.8% / 48.0% |
| Executed on NorthSea.gdb: same feature set | 92.0% | **94.0%** |
| — same set, requests with non-empty gold (77) | 81.8% | **85.7%** |
| — mean Jaccard | 0.942 | **0.955** |
| — macro precision / recall of feature IDs | **0.924** / 0.908 | 0.898 / **0.922** |
| Median latency | 0.16 s (MLX) | 0.18 s (gpu-host) · 7.9 s (Mac MLX) |

Table numbers are the Mac MLX run unless marked gpu-host; MLX and gpu-host predictions are identical
on 196/200 requests (near-ties flip), with the same exact score. Per request (gpu-host): 25
requests only feln-clm gets right, 10 only feln-laya. Execution metrics
run gold and predicted FELN with DuckDB spatial on the GDB (EPSG:3035 metres); 123 of 200
gold queries return no features, so the non-empty row is the informative one. feln-clm's
lower ID precision comes from a few large sets: e.g. "Show wells within 5 miles of gas
pipelines" (gold filters *shows* wells; the text never says so) returns all 1,881 wells
near gas pipelines. Most such requests carry low confidence (0.12 there).

Recipe progression (per-question test accuracy, then end-to-end exact):

| change | question acc | exact |
|---|---:|---:|
| raw text state, CLM defaults | 0.804 | — |
| lr 2e-3, 60 epochs, InfoNCE | 0.862 | — |
| chat turn listing the options | 0.886 | — |
| end the turn on `Answer:` (CLM strips trailing whitespace) | 0.943 | 73.0% |
| subtypes restricted to named ones, joint binding, relation names the subtype | 0.941 | 78.0% |
| + literal penalty (tuned on finetune's validation split, 70.5→72.3%) + 5 seeds | 0.95 each | 81.5% |

The dev split was used to compare recipes, so it is a development set, not a pristine
test. Decoder knobs were tuned only on the finetune validation split (285 requests).

## Limits

- Grammar = FELN.json's: subtype + at most one extra condition per layer (`=, <>, LIKE
  contains/starts/ends, either-of-two, blank, <, >, <=, >=, BETWEEN`, year ranges). 748 of
  2,857 generated training rows with two conditions were skipped.
- A subtype the request never names cannot be predicted (5.5% of FELN.json golds; mostly
  humanizer noise such as "oil/gas wells" for gold *oil/gas shows*).
- Ambiguous requests ("within 15 km of oil" — wells, pipelines or discoveries?) stay
  ambiguous; `confidence` (joint probability of the decisions) flags many of them.
- MLX latency is ~8 s per request (median) on an M4 Max (≈10k prompt tokens: options are listed in
  every question). The models are bf16; no quantization was evaluated.
- Checkpoints are tied to the OKF catalog hash; a changed OKF needs `prepare` + retraining.

## Setup

Keep `../CLM` and `../feln` beside this checkout (editable path dependencies; CLM's vLLM
pin is dropped on macOS). Project data and weights are not in git:

```sh
N="$HOME/Documents/ArcGIS/Projects/NorthSea"
mkdir -p data && cp -R "$N/okf" data/okf && cp "$N/FELN.json" data/FELN.json
cp ../feln-laya/out/expanded-v2/heldout.jsonl data/laya-heldout.jsonl       # the 200 dev texts
cp ../feln-laya/out/expanded-v2/generated.jsonl data/laya-generated.jsonl   # extra training rows
uv sync --extra mlx --extra exec --extra dev
```

`models/options4` (`feln-clm.json` + `heads/*.pt`, 400 MB) comes from training below, copied
from the training host. Qwen3-8B (bf16, ~16 GB) downloads from Hugging Face on first use.
A model refuses an OKF whose hash differs from the one it was trained on.

## Use (Mac)

```sh
uv run feln-clm ask data/okf models/options4 "Find gas/condensate wells within 5 km of injection pipelines."
uv run feln-clm serve data/okf models/options4          # single-page app: http://127.0.0.1:8710/
uv run feln-clm evaluate data/okf models/options4 data/FELN.json data/laya-heldout.jsonl --output results/x.jsonl
uv run feln-clm execute data/okf results/x.jsonl        # execution precision/recall on the GDB
```

`ask` prints the FELN (`meta`), `status` (`accepted` when `confidence` ≥ `--threshold`,
default 0.5), per-question decisions with probabilities, and the runner-up structures.
`evaluate` writes one JSONL line per request plus `.summary.json`, and refuses to overwrite.
`execute` builds `out/project.duckdb` once from the OKF `resource` feature classes.

The app is one static page served by the stdlib HTTP server (loopback only, one request at
a time): `GET /api/info` (catalog, examples), `POST /api/ask {"text": …}` → the `ask` result.
It shows the query per layer, confidence, every decision with its probability, the FELN
JSON and the alternatives.

## Train (gpu-host)

Environment on the CUDA host (once):

```sh
uv venv .venv && uv pip install --python .venv/bin/python vllm pyarrow huggingface_hub transformers
uv pip install --python .venv/bin/python -e ../CLM -e ../feln && uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/hf download Qwen/Qwen3-8B
.venv/bin/hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt --local-dir ~/.cache/clm
```

Prepare rows on the Mac (or there), then fine-tune five seeds and package:

```sh
uv run python -m feln_clm.prepare data/okf data/FELN.json data/laya-heldout.jsonl out/rows-options4 \
    --extra data/laya-generated.jsonl --format options
export PY=.venv/bin/python CLM=../CLM
scripts/finetune-gpu-host.sh s1 out/rows-options4 0 emb --seed 1   # first run embeds (~3 min)
for s in 2 3 4 5; do scripts/finetune-gpu-host.sh s$s out/rows-options4 $((s % 2)) emb --seed $s; done
$PY -m feln_clm.cli package out/rows-options4 models/options4 runs/s1 runs/s2 runs/s3 runs/s4 runs/s5
$PY -m feln_clm.cli evaluate data/okf models/options4 data/FELN.json data/laya-heldout.jsonl \
    --encoder vllm --device cuda --output results/gpu-host.jsonl
$PY scripts/sweep.py models/options4 out/rows-options4 tag '{"penalty": 0}' '{"penalty": 2}'  # val split only
```

Then copy `models/options4` to the Mac. Each head takes ~5 min on one RTX PRO 6000.
MLX and vLLM embeddings agree to cosine ≥ 0.99993, so the heads run unchanged on the Mac.
`prepare` skips (and counts) requests the grammar cannot reproduce; it never rewrites them.

## Layout

```
feln_clm/okf.py          OKF markdown → catalog (columns, kinds, domains, hints, GDB resource)
feln_clm/grammar.py      spans, mentions, options, questions, framing, decompose / compose / SQL
feln_clm/prepare.py      split + CLM typed-decision parquet (finetune.py --task choice input)
feln_clm/embedders.py    Qwen3-8B on MLX (Mac) or vLLM (CUDA) behind CLM's Embedder
feln_clm/decode.py       Translator: question rounds, head ensemble, joint constrained decode
feln_clm/evaluate.py     exact / atom P-R / selective accuracy;  execute.py: GDB execution P-R
feln_clm/server.py, static/index.html   the single-page app
feln_clm/cli.py          ask · evaluate · execute · serve · package
scripts/                 finetune-gpu-host.sh (one head), sweep.py (decoder knobs on the val split)
results/                 *.summary.json / *.execution.json per run (per-request JSONL stays local)
tasks/                   todo.md (progress), lessons.md
```

## Tests

```sh
uv run pytest -q   # span edge cases, oracle roundtrip of all 3,000 FELN.json queries, decoder constraints
uv run ruff check . && uv run ruff format --check .
```
