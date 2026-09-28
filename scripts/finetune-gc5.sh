#!/bin/sh
# One CLM head fine-tune on the training host, logged to $RUNS/NAME/run.log.
#   scripts/finetune-gc5.sh NAME ROWS_DIR GPU EMB_DIR [extra finetune.py args...]
# e.g. scripts/finetune-gc5.sh s1 out/rows-options4 0 emb --seed 1
# Runs sharing EMB_DIR reuse its embedding cache (vLLM loads only for missing texts).
# Env: CLM (../CLM checkout), PY (venv python with vllm), RUNS (default runs), INIT (warm start).
set -eu
name=$1 rows=$2 gpu=$3 emb=$4; shift 4
CLM=${CLM:-../CLM} PY=${PY:-python} RUNS=${RUNS:-runs} INIT=${INIT:-$HOME/.cache/clm/CLM_v0.1-8B.pt}
mkdir -p "$RUNS/$name"
CUDA_VISIBLE_DEVICES=$gpu "$PY" "$CLM/train/finetune.py" --task choice \
  --data "$rows" --workflow feln --init-ckpt "$INIT" \
  --out-dir "$RUNS/$name" --embed-cache "$emb" --gpu 0 --gpu-mem 0.45 \
  --loss infonce --lr 2e-3 --epochs 60 --patience 10 "$@" > "$RUNS/$name/run.log" 2>&1
