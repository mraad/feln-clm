# feln-clm — text → FELN with a fine-tuned CLM

Goal: beat feln-laya (74% exact FELN on its 200-request dev split) on accuracy,
precision and recall. Speed is secondary. Train/tune on gc5, infer on the Mac (MLX).

## Steps

- [x] uv project; Qwen3-8B + CLM head; MLX embedder parity vs vLLM (cos ≥ 0.99993)
- [x] OKF reader → catalog (kinds: subtype/code/yesno/flag/like/upper/number/date)
- [x] decompose/compose; oracle roundtrip 3000/3000 on FELN.json (generated: 2109/2857,
      rest use two ANDed/ORed conditions — outside FELN.json's grammar)
- [x] questions + CLM typed-decision parquet; gc5 fine-tune with CLM `finetune.py` unchanged
- [x] framing experiments (per-question test acc): raw 0.80 → lr/epochs 0.862 →
      chat+options 0.886 → "Answer:" suffix 0.943 (CLM strips trailing whitespace)
- [x] joint decoder + evaluate: options3 1 head = 73.0% exact (laya 74%);
      1-layer 95% / 2-layer 79% / 3-layer 44% (laya 86/77/58)
- [x] options4: subtype options restricted to mentioned subtypes + joint count constraint;
      relation/distance questions name the bound subtype (targets 3-layer binding)
- [x] literal penalty 2.0 (val split) + 5-seed ensemble: 81.5% exact (gc5 = Mac MLX)
- [x] Mac MLX eval: 81.5%, 196/200 identical to gc5; execution P/R on the GDB vs laya
- [x] SPA smoke test (API + page JS); README with measured results; lessons
- [ ] (declined for now) ONNX export of heads
