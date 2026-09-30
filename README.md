# feln-clm

Text → FELN over an OKF project catalog with a LoRA-tuned Qwen3 whose decoding is
constrained to the catalog. The model writes a query as a few short pieces, each one
choice from a closed set (layer, subtype, condition, relation + distance); SQL is rendered
from the OKF hints, never generated. Training and cross-validation run on a CUDA host
(gpu-host); inference runs on the Mac (MLX). The goal is accuracy, precision and recall.

v1 of this repo (CLM: frozen Qwen3-8B embeddings + choice heads answering four rounds of
typed questions) is in Git history; its numbers stay in the table below.

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
| Exact FELN (`FELN.same`) | **92.0%** | 81.5% | 74.0% | 60.0%¹ |
| — a gold layer never named in the text (110) | **88.2%** | 73.6% | — | — |
| Atom precision / recall (layers, predicates, relations) | **0.968 / 0.966** | 0.939 / 0.924 | — | — |
| Accuracy / coverage at confidence ≥ 0.8 | **94.4% / 88.5%** | 92.1% / 63.0% | 91.6% / 53.5% | — |
| Accuracy / coverage at confidence ≥ 0.9 | **97.0% / 82.0%** | 94.8% / 48.0% | — | — |
| Executed on NorthSea.gdb: same feature set | **96.5%** | 94.0% | 92.0% | 84.0% |
| — same set, requests with non-empty gold (77) | **92.2%** | 85.7% | 81.8% | 66.2% |
| — mean Jaccard | **0.971** | 0.955 | 0.942 | 0.859 |
| — macro precision / recall of feature IDs | **0.964 / 0.950** | 0.898 / 0.922 | 0.924 / 0.908 | 0.831 / 0.734 |
| — micro precision / recall of feature IDs | 0.656 / 0.702 | 0.462 / 0.738 | **0.687 / 0.806** | 0.646 / 0.715 |
| Invalid outputs | 0 | 0 | — | 20 |
| Median latency | 0.59 s (gpu-host) · 0.59 s (Mac MLX; p90 0.9 s)² | 4.0 s (Mac) | 0.16 s | 2.4 s |

¹ after ILIKE→LIKE and INT→INTEGER normalisation (49.5% raw). feln-lora (Nemotron-4B LoRA,
free-form JSON under a shape-only grammar) was trained on templated questions and its own
catalog, so this is its transfer to these reworded requests, not its in-domain score.

Mac MLX and gpu-host torch predictions agree on 200/200 dev requests. Single trainings are
noisy by about a point: the same recipe with another seed moves fold-0 exact by ~1 point
(synonyms 90.5% / 89.5%, comparatives 89.3% / 90.2%), and each retraining flips 18–27 of
the 560 fold-0 requests, nearly all implicit-layer coin flips. The dev score stayed within
that band as the OKF gained synonyms and comparatives (91.5% → 91.5% → 91.0% → 92.0%).
Compare recipes over at least two seeds.

² Earlier Mac runs of the same decoder measured 1.5–1.6 s median; this run and a 30-request
re-time measured 0.55–0.59 s. The decoder did not change, so the difference is load on the
Mac during the earlier runs, not a speed-up. Micro feature precision/recall is dominated by a few requests with very large
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

## Column synonyms and comparatives

When requests call a column something other than its OKF alias, or compare it with a
word of its own, say so in the column's Query hints:

```
## `water_depth`

- Depth of the well in meters
- Also called: water depth, depth
- Comparatives: deeper than = greater, shallower than = less, at least as deep as = at least, no shallower than = at least, at most as deep as = at most, no deeper than = at most
```

The prompt then names the column (and, for a comparative, the operator: `'deeper than'
= Wells well water depth >`) when a request uses one of these phrases; longer phrases win,
so "no deeper than" is not also read as "deeper than". `prepare` adds
training copies of the requests on that column with the name swapped for each other one,
and with "<name> greater than / more than / over x" said with each `greater` comparative
("deeper than x"), and likewise for `less` (<), `at least` (>=: "at least", "no less
than") and `at most` (<=: "at most", "no more than"); the gold is unchanged. Editing the OKF changes its hash: re-run `prepare` and retrain.

Unquoted multiword text samples are normalized to quoted catalog values before building
the prompt, in both preparation and inference. For example, `in-service` becomes
`'IN SERVICE'`, so the existing value hints identify the pipeline's `current_phase`.
Existing quoted literals and single-word values are preserved. With `models/q4b`,
"Show all wells with depth > 350 m and within 5 km of an in-service pipeline" produces:

```json
{"layers": ["Wells", "Pipelines"],
 "where": ["water_depth > cast(350 as DOUBLE PRECISION)", "current_phase = 'IN SERVICE'"],
 "relations": ["withinDistance 5 kilometers"]}
```

Run the optional GPU regression with
`FELN_CLM_TEST_MODEL=models/q4b uv run pytest -q -k local_model`.

With `water_depth` "Also called: water depth, depth" and "Comparatives: deeper than =
greater, shallower than = less", these decode correctly at confidence 1.00 (before: "wellbore name is
not blank", 0.18–0.59):

| Request | Pieces |
|---|---|
| Find wells deeper than 120 within 5 km of oil pipelines. | `Wells [any] where well water depth > 120; within 5 kilometers of Pipelines [oil]` |
| Show gas wells shallower than 70.5. | `Wells [gas] where well water depth < 70.5` |
| Show oil wells with a depth over 100. | `Wells [oil] where well water depth > 100` |
| Wells with depth between 60 and 80. | `Wells [any] where well water depth between 60 and 80` |

All 9 held-out fold-0 rewrites (synonym and comparative) decode to their gold.

With the `>=` / `<=` phrases these decode correctly too, at confidence 1.00:

| Request | Pieces |
|---|---|
| Find wells at least as deep as 100. | `Wells [any] where well water depth >= 100` |
| Show gas wells no deeper than 70.5 within 5 km of oil pipelines. | `Wells [gas] where well water depth <= 70.5; within 5 kilometers of Pipelines [oil]` |
| List oil shows no shallower than 90 meters. | `Wells [oil shows] where well water depth >= 90` |
| Water wells at most as deep as 110. | `Wells [water] where well water depth <= 110` |

Fold-0 exact over two seeds: 90.4% / 89.3% (before: 89.3% / 90.2%).

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

## Train and evaluate (gpu-host)

Prepare rows on the Mac (needs the GDB for `values.json`), copy them over, then train:

```sh
uv run python -m feln_clm.prepare data/okf data/FELN.json data/laya-heldout.jsonl out/rows \
    --extra data/laya-generated.jsonl --db out/project.duckdb
rsync -a out/rows gpu-host:feln-clm/

# gpu-host, once: uv venv .venv && uv pip install --python .venv/bin/python torch "transformers>=5" safetensors
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

## Candidate recall before contrastive reranking

Measure whether the correct FELN is available to a future reranker, without training one:

```sh
uv run python -m feln_clm.candidates data/okf models/q4b out/rows-cmp3 \
    --split dev --beam 8 --k 8 --output out/candidates-q4b-dev-top8.jsonl
```

`Translator.ask(text, nbest=8)` collects up to eight semantically distinct completions.
Search stops when the eighth completion beats every live beam, or search is exhausted.
This remains beam-pruned search, not exhaustive top-k enumeration. Default `ask(text)`
keeps the existing production behavior and response format.

The diagnostic runs both default decoding and candidate collection on each request,
records raw candidate log scores and readable query pieces, and reports exact FELN
recall at ranks 1–8, recoverable baseline errors, latency, and separate noisy/implicit
slices. Raw rows stay under ignored `out/`; they are marked `evaluation_only` and must
not become reranker training data. Existing outputs are never overwritten; interrupted
runs write partial summaries. Model and prepared-row catalog hashes must match.

Use a model's held-out fold for recipe selection; the CLI rejects other non-dev splits.
A fixed run on `dev` is descriptive only. Do not use it to choose beam width, reranker
architecture, score weights, or training examples. For training, generate candidates on
folds excluded from the generating model's training, and split paraphrases together.

Fixed diagnostic on the 200 development requests (`q4b`, beam 8; no reranker trained):

| Metric | Result |
|---|---:|
| Default top-1 exact FELN | 184/200 (92.0%) |
| Gold available at rank ≤ 2 | 197/200 (98.5%) |
| Gold available at rank ≤ 3 | 199/200 (99.5%) |
| Gold available at rank ≤ 8 | 200/200 (100%) |
| Top-1 changes from extended search | 0 |
| Median default / top-8 decoding | 0.667 s / 0.958 s |

These are candidate-availability ceilings against existing labels, not reranker accuracy.
Manual inspection of the 16 baseline misses found 2 clear semantic errors (both corrected
at rank 2), 8 ambiguous requests, and 6 conflicting gold labels. This audit is an analyst
judgment, not a relabeling of the development set. Automatic `noisy` tags miss some label
problems, including imperative “Show wells” mislabeled as the SHOWS subtype.
The useful contrastive examples involve dropping a named literal and assigning an
explicit subtype to the wrong layer. Test that hypothesis on clean held-out folds before
training or choosing a reranker recipe; the current local 4B model has trained on all
five folds. Aggregate evidence: `results/candidate-recall-q4b-dev.summary.json`.
Raw candidates and the annotated mistake audit remain in ignored `out/`.

## Contrastive reranker pilot (gpu-host)

`feln_clm.rerank` is an offline experiment; it does not change production inference.
Collect candidates from each generator's held-out fold and mark the output explicitly:

```sh
python -m feln_clm.candidates okf models/q4b-ge-f0 rows-cmp3 --backend torch \
    --split fold0 --usage reranker_pool --output candidates/fold0-s0.jsonl
python -m feln_clm.candidates okf models/q4b-ge-f0-s1 rows-cmp3 --backend torch \
    --split fold0 --usage reranker_pool --output candidates/fold0-s1.jsonl
python -m feln_clm.rerank okf candidates/fold0-s0.jsonl candidates/fold0-s1.jsonl \
    --rows rows-cmp3 --out artifacts/pilot
```

The pilot rejects development/evaluation-only candidate files. It excludes implicit or
noisy requests, checks imperative “Show” against the SHOWS subtype, and excludes gold
query groups found in generator training or development data. These filters are
conservative heuristics, not human certification that every remaining label is correct.
Equivalent gold atom sets stay together in a deterministic 60/20/20 train/validation/test
split. Atom sets may also group some non-equivalent Boolean queries; that is conservative
for leakage prevention and is never used as the correctness metric.

Frozen Qwen3-4B last-token embeddings feed a shared 128-dimensional linear projection.
Training uses per-request contrastive cross entropy at temperature 0.1, AdamW at 0.001,
and 30 epochs. There are no in-batch negatives. Validation selects the epoch and a blend
of the generator log score with the contrastive score; weight zero retains the baseline.
Two projection seeds (0 and 1) are evaluated on the same untouched test groups. Candidate
sets from both generator seeds remain in the same partition, and results distinguish
candidate-set counts from unique request counts. Missing-gold training sets are skipped;
missing-gold validation/test sets count as incorrect. This is a pilot within fold 0,
not five-fold cross-validation and not an evaluation on the 200 development requests.

Run remote collection and training in a named tmux session so a disconnect does not
stop the jobs. Use your own SSH alias in place of `gpu-host`, and keep experiment
artifacts in a private directory outside the tracked source files. For example:

```sh
ssh -t gpu-host 'tmux new-session -A -s feln-contrastive'
```

Run the collection and training commands inside that session. Keep logs, checkpoints,
partition manifests, and test predictions in ignored artifact directories.

Completed pilot: 159 training, 44 validation, and 38 test requests after filtering,
with two candidate sets per request. Both projection seeds selected epoch 2 on validation
(85/88 correct versus baseline 84/88). On the untouched test groups, both scored 73/76
(96.1%), exactly matching baseline: one correction and one regression each. The correction
removed an unrequested Wells layer; the regression dropped a requested multilateral
filter. All 76 candidate sets contained the labeled answer, so recall was not the test
bottleneck. Both rankings returned the same gold feature sets on the current database
(76/76, including 30 nonempty-gold cases); that does not erase their semantic differences.
This small pilot does not justify enabling the reranker in production. Aggregate results
are in `results/contrastive-fold0-pilot.summary.json`; local checkpoints and raw test
predictions are under ignored `out/`.
