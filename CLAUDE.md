# CLAUDE.md

`README.md` covers what this is, results and commands. This file is what breaks when you
edit things.

## Commands

```bash
uv sync --extra mlx --extra exec --extra dev
uv run pytest -q                       # needs data/ for the oracle test
uv run ruff check . && uv run ruff format --check .
```

## Training/inference contract (do not drift)

- **Piece text is the model.** The adapter is trained on the exact strings of
  `grammar.prompt`, `grammar.hints`, `grammar.head`, `grammar.condition` (option phrases in
  `_options`, `RELATIONS`) and `END`. Any wording change silently degrades a trained model:
  re-run `prepare` and retrain. `feln-clm.json` records the catalog hash, not the wording.
- **Pieces are tokenized one at a time**, in training (`train.py`) and in the decoder's
  tries (`decode.Translator.ids`), both through `lm.encode`. Never tokenize the joined
  target string: BPE merges across piece boundaries would make gold unreachable.
- The oracle test decodes all 3,000 FELN.json queries through the real beam search with a
  byte tokenizer and an LM that prefers gold. Keep it at 100%: it guards `decompose`,
  `pieces`, the slot candidates and the decoder's hard rules together.
- Hard rules live only in `Translator._trie` (≤ `MAX_LAYERS` distinct layers, a literal
  serves one condition, distinct distances when the request has several). There is no
  unused-literal penalty: 0, 2 and 4 nats gave identical fold0 exact for both sizes.
- Secondaries keep gold order (`decompose` no longer sorts them): the text follows it in
  94% of 3-layer requests, so the model learns to write layers in the order they are said.
- The prompt hints come from `values.json` (lower-cased distinct values of every text
  column, built from the GDB by `prepare`). It travels with the model dir; rebuild both
  if the data changes.
- A model refuses an OKF whose `Catalog.sha` differs; do not bypass by editing the config.

## Evaluation hygiene

- The 200 `laya-heldout` requests (`split == "dev"`) are a **development** set, never
  trained on. Recipe choices (base size, epochs, noisy filter, `beam`) are made
  on `fold0`..`fold4` (cross-validation over the other FELN.json requests), not on dev.
- `tags`: `implicit` (a gold layer never named) and `noisy` (gold contradicts the text).
  Report exact per tag; `noisy` rows are excluded from training by default.
- `evaluate` never overwrites; use a new output path per run.
- Execution metrics reproject the GDB (EPSG:4326 assumed) to EPSG:3035 metres; 123 of 200
  dev gold queries return no features, so read the non-empty-gold rows.

## Remote host

- Training and CV evaluation run on gc5 (2× RTX PRO 6000) with the `torch` backend; the
  Mac runs `mlx`. Both merge the same adapter into bf16 weights.
- `ssh gc5 'pkill -f PATTERN; …'` kills its own shell; use `pkill -f '[p]attern'` or PIDs.
- Detach long jobs with `setsid nohup … < /dev/null > log 2>&1 &` or ssh blocks. In a
  `cd X && … &` chain the `&` backgrounds the whole chain: later commands run elsewhere.
- NorthSea-derived data (rows, values.json) may go to gc5 (user-approved); never into git.
