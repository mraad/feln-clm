"""Decoder knob sweep on the finetune validation split, one Translator (shared embedding cache).

    python scripts/sweep.py MODEL_DIR ROWS_DIR TAG '{"penalty": 0}' '{"penalty": 2, "beam": 5}' ...

Never tune on the dev split: the validation split is the one CLM's finetune.py held out
(same seed), restricted to the original FELN examples. Run on the CUDA host (vLLM encoder).
"""

import json
import sys

from feln_clm.decode import Translator
from feln_clm.embedders import make
from feln_clm.evaluate import run
from feln_clm.prepare import val_texts

model, rows, tag = sys.argv[1:4]
gold = {x["text"]: x for x in json.load(open("data/FELN.json"))}
examples = [gold[t] for t in val_texts(rows)]
tr = Translator("data/okf", model.split(","), embedder=make("vllm"), device="cuda")
for spec in map(json.loads, sys.argv[4:]):
    for k, v in spec.items():
        setattr(tr, k, v)
    name = "-".join(f"{k}{v}" for k, v in spec.items())
    out = run(tr, examples, f"results/val-{tag}-{name}.jsonl")
    keys = ("exact", "atom_precision", "atom_recall", "accuracy@0.8", "coverage@0.8")
    print("RESULT", spec, {k: round(out[k], 4) for k in keys}, flush=True)
