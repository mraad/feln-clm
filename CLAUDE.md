# CLAUDE.md

`README.md` covers what this is, results and commands. This file is what breaks when you
edit things.

## Commands

```bash
uv sync --extra mlx --extra exec --extra dev
uv run pytest -q                       # needs data/ for the oracle + decoder tests
uv run ruff check . && uv run ruff format --check .
```

## Training/inference contract (do not drift)

- **Question text is the model.** Heads are trained on the exact state/option strings built by
  `grammar.q_*` and `grammar.frame`. Any wording change to a question, option description,
  role phrase or the `_ASSISTANT` suffix silently degrades a trained model: re-run `prepare`
  and retrain. `feln-clm.json` records only the framing name, not a hash of the wording.
- CLM's `schema.state_text` **strips trailing whitespace**; the framed turn must end on a
  visible token (`Answer:`). Ending on `</think>\n\n` cost ~6 points of question accuracy.
- Question ids are keys shared by `prepare.questions` and `decode.Translator`
  (`subtype:L@P`, `column:L@P`, `op:L@P|col`, `value:L@P|col|op`, `relation:L@P=code`,
  `distance:L@P=code`). Change both or neither.
- `options` framing lists every option in the state; ≤52 options per question also keeps
  the `ids` framing valid (test enforces it). The condition is split into column → op for
  that reason; do not merge them back into one 127-option question.
- `decompose` searches the same space `render_*` produces, so the oracle roundtrip test
  (3,000/3,000 FELN.json) is the guard for any grammar change. Keep it at 100%.
- Subtype questions only offer subtypes the request names (`mentions`), then `any`.
  Relation/distance questions name the bound subtype, so they are asked after the joint
  subtype decode (round 3). Training teacher-forces the gold subtype there.
- A model refuses an OKF whose `Catalog.sha` differs; do not bypass by editing the config.

## Evaluation hygiene

- The 200 `laya-heldout` requests are a **development** set (recipe comparison). Tune decoder
  knobs (`penalty`, `beam`, `top`) only with `scripts/sweep.py` on `val:` (finetune's own
  validation split, seed 1234). Seed-ensemble members used other seeds, so ensembles leak
  into that split: tune knobs with the seed-1234 head only.
- `evaluate` and the embedding caches never overwrite; use a new output path per run.
- Execution metrics reproject the GDB (EPSG:4326 assumed) to EPSG:3035 metres; 123 of 200
  gold queries return no features, so read the non-empty-gold rows.

## Remote host

- Training runs on gc5 (2× RTX PRO 6000) with the in-process vLLM encoder; the Mac runs MLX.
  Parity checked at cosine ≥ 0.99993; 196/200 dev predictions identical (near-ties flip).
- `ssh gc5 'pkill -f PATTERN; …'` kills its own shell; use `pkill -f '[p]attern'` or PIDs.
- Detach long jobs with `setsid nohup … < /dev/null > log 2>&1 &` or ssh blocks.
- NorthSea-derived data may go to gc5 (user-approved for training); never into git.
