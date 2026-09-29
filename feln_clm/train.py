"""LoRA fine-tuning of the generator on prepared rows (CUDA host).

    python -m feln_clm.train out/rows models/q4b --base Qwen/Qwen3-4B [--holdout fold0]

Trains on every row except ``dev``, the ``--holdout`` split and (unless ``--keep-noisy``)
rows whose gold contradicts the text. Loss is on the target tokens only; the prompt is
masked. Target ids are the concatenation of each piece tokenized alone, exactly as the
decoder builds its tries. Writes ``adapter.safetensors``, ``feln-clm.json`` and
``values.json`` (the prompt hints' index) into OUT.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .decode import ADAPTER, CONFIG, VALUES
from .lm import TARGETS, encode


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, scale: float):
        super().__init__()
        self.base, self.scale = base, scale
        dev = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=dev))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x.float() @ self.lora_a.T @ self.lora_b.T * self.scale).to(x.dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("rows", help="prepare output dir")
    ap.add_argument("out")
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--holdout", default="", help="split left out for evaluation, e.g. fold0")
    ap.add_argument("--keep-noisy", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rows_dir, out = Path(a.rows), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in open(rows_dir / "rows.jsonl")]
    rows = [r for r in rows if r["split"] not in ("dev", a.holdout)
            and (a.keep_noisy or "noisy" not in r["tags"])]  # fmt: skip

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16).cuda()
    model.requires_grad_(False)
    scale = a.alpha / a.rank
    adapted = {}
    for name, mod in list(model.named_modules()):
        for child, lin in list(mod.named_children()):
            if child in TARGETS and isinstance(lin, nn.Linear):
                adapted[f"{name}.{child}"] = lora = LoRA(lin, a.rank, scale)
                setattr(mod, child, lora)
    params = [p for m in adapted.values() for p in (m.lora_a, m.lora_b)]

    data = []
    for r in rows:
        p = encode(tok, r["prompt"])
        t = [i for piece in r["pieces"] for i in encode(tok, piece)]
        data.append((p + t, [-100] * len(p) + t))
    steps = a.epochs * math.ceil(len(data) / a.batch)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps)))
    )
    rng, t0, step, losses = random.Random(a.seed), time.time(), 0, []
    model.train()
    for epoch in range(a.epochs):
        rng.shuffle(data)
        for i in range(0, len(data), a.batch):
            batch = data[i : i + a.batch]
            width = max(len(ids) for ids, _ in batch)
            ids = torch.tensor([x + [0] * (width - len(x)) for x, _ in batch]).cuda()
            lab = torch.tensor([y + [-100] * (width - len(y)) for _, y in batch]).cuda()
            mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in batch]).cuda()
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            loss = nn.functional.cross_entropy(logits.transpose(1, 2), lab[:, 1:])
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            losses.append(loss.item())
            if step % 50 == 0 or step == steps:
                print(f"[train] epoch {epoch} step {step}/{steps} loss {sum(losses[-50:]) / len(losses[-50:]):.4f} "
                      f"{time.time() - t0:.0f}s", flush=True)  # fmt: skip

    save_file({f"{n}.{k}": getattr(m, k).detach().contiguous().cpu()
               for n, m in adapted.items() for k in ("lora_a", "lora_b")}, str(out / ADAPTER))  # fmt: skip
    shutil.copy(rows_dir / "values.json", out / VALUES)
    prep = json.loads((rows_dir / "prepare.json").read_text())
    cfg = {"base": a.base, "catalog_sha": prep["catalog_sha"], "scale": scale, "rank": a.rank,
           "holdout": a.holdout, "keep_noisy": a.keep_noisy, "epochs": a.epochs, "lr": a.lr,
           "batch": a.batch, "seed": a.seed, "rows": len(data), "steps": steps,
           "final_loss": sum(losses[-50:]) / len(losses[-50:]), "seconds": round(time.time() - t0)}  # fmt: skip
    (out / CONFIG).write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
