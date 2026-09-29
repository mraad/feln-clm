# feln-clm v2 — constrained LoRA generator (replaces the CLM question rounds)

Goal: beat v1 (CLM ensemble, 81.5% exact / 94.0% execution on the 200 dev requests) and
feln-lora v2 (60.0% / 84.0%) on accuracy, precision and recall. Max 3 layers. Train and
evaluate on gc5; inference on the Mac (MLX). v1 stays in git history (441c423).

Why (dev error analysis of v1, 37 misses): 11 undecidable from text, 8 gold contradicts
text, 18 fixable: 9 relation/distance binding, 5 literal→column (the GDB knows), 4 subtype.

## Steps

- [x] data checks: gold literal/distance coverage, secondary order vs text, noisy-gold flags
- [x] grammar: drop question/framing code; target pieces + per-slot candidates (catalog only)
- [x] value hints: literal → layer.column index built from the GDB (DuckDB)
- [x] prepare: prompt + pieces rows, dev excluded, 5 folds for recipe choice
- [x] train (gc5): hand-rolled LoRA on Qwen3, prompt-masked SFT, adapter only
- [x] decode: token-level beam over a per-slot trie, joint constraints, confidence, alternatives
      backends: MLX (Mac) and torch (gc5)
- [x] evaluate: tags (implicit layer, noisy gold), per-tag exact; CV summary
- [x] cli/server/SPA; drop CLM dependency, embedders, heads
- [x] tests: oracle roundtrip through pieces, trie prefix-free, decoder constraints
- [x] gc5: 4B vs 8B by 5-fold CV; final model on all non-dev; dev eval gc5 + Mac; execution
- [x] README, CLAUDE.md; commit

Result: dev exact 91.5% (v1 81.5%, laya 74.0%, feln-lora 60.0%), execution match 95.5%;
5-fold CV 88.8% (4B = 8B); Mac = gc5 on 200/200; Mac median 1.64 s.
