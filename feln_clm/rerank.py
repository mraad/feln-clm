"""Offline contrastive reranker pilot on out-of-fold candidates; never consumes dev.

Frozen causal-LM embeddings, a shared linear projection, and per-request contrastive
cross entropy. Query groups split 60/20/20; validation selects epoch and LM-score blend.
The untouched test partition is evaluated only after selection, for two training seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from . import grammar as g
from . import okf
from .evaluate import atoms


def group_key(meta: dict) -> str:
    # Atom sets can merge some non-equivalent Boolean queries: conservative grouping,
    # never a correctness test. Exact labels still use FELN.same via gold_rank.
    key = json.dumps(sorted(atoms(meta), key=repr), sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()


def partition(group: str) -> str:
    bucket = int(hashlib.sha256(("rerank-pilot-v1:" + group).encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else "validation" if bucket < 8 else "test"


def prepare(cat, paths, blocked_groups=frozenset()):
    rows = [json.loads(line) for p in paths for line in Path(p).read_text().splitlines()]
    texts, lookup, selected, excluded = [], {}, [], Counter()
    text_groups = {}
    for row in rows:
        if (
            row["usage"] != "reranker_pool"
            or row["split"] == "dev"
            or row["split"] != row["generator_holdout"]
        ):
            raise ValueError("only explicitly marked, held-out-fold reranker pools are allowed")
        text_groups.setdefault(g.normalize(row["text"]).lower(), set()).add(group_key(row["gold"]))

    def index(text):
        if text not in lookup:
            lookup[text] = len(texts)
            texts.append(text)
        return lookup[text]

    for row in rows:
        if group_key(row["gold"]) in blocked_groups:
            excluded["generator_training_or_dev_gold_group"] += 1
            continue
        if {"implicit", "noisy"} & set(row["tags"]):
            excluded["implicit_or_noisy"] += 1
            continue
        if len(text_groups[g.normalize(row["text"]).lower()]) > 1:
            excluded["conflicting_labels_for_same_text"] += 1
            continue
        try:
            d = g.decompose(cat, g.ground_samples(cat, row["text"]), row["gold"])
        except ValueError:
            excluded["unreachable_gold"] += 1
            continue
        # Do not mistake the imperative "Show" for the SHOWS subtype.
        said = re.sub(r"^\s*show\b", "", row["text"], flags=re.I)
        if {"implicit", "noisy"} & set(g.tags(cat, said, d)):
            excluded["additional_text_gold_conflict"] += 1
            continue
        group = group_key(row["gold"])
        selected.append(
            {
                **row,
                "group": group,
                "partition": partition(group),
                "query_index": index("Request: " + row["text"]),
                "candidate_indices": [
                    index("FELN: " + "; ".join(c["pieces"])) for c in row["candidates"]
                ],
            }
        )
    return texts, selected, dict(excluded)


def embed(texts, base, out):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(base, dtype=torch.bfloat16).cuda().eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), 16):
            batch = tok(texts[start : start + 16], padding=True, return_tensors="pt").to("cuda")
            if batch.input_ids.shape[1] > 2048:
                raise ValueError("embedding input exceeds 2048 tokens; refusing silent truncation")
            pos = batch.attention_mask.cumsum(-1) - 1
            pos.masked_fill_(batch.attention_mask == 0, 0)
            hidden = model(**batch, position_ids=pos, use_cache=False).last_hidden_state[:, -1]
            vectors.append(torch.nn.functional.normalize(hidden.float(), dim=-1).cpu().numpy())
            if start % 256 == 0:
                print(f"[embed] {min(start + 16, len(texts))}/{len(texts)}", flush=True)
    matrix = np.concatenate(vectors)
    np.save(out / "embeddings.npy", matrix)
    del model
    torch.cuda.empty_cache()
    return matrix


def train(matrix, rows, out, *, seed, epochs):
    import torch
    from torch import nn
    from torch.nn import functional as F

    torch.manual_seed(seed)
    rng = random.Random(seed)
    vectors = torch.tensor(matrix, device="cuda")
    projection = nn.Linear(matrix.shape[1], 128, bias=False).cuda()
    opt = torch.optim.AdamW(projection.parameters(), lr=1e-3, weight_decay=0.01)
    parts = {p: [r for r in rows if r["partition"] == p] for p in ("train", "validation", "test")}
    training = [r for r in parts["train"] if r["gold_rank"] > 0]
    if not training or not parts["validation"] or not parts["test"]:
        raise ValueError("need nonempty train, validation, and test partitions")

    def scores(batch):
        q = F.normalize(projection(vectors[[r["query_index"] for r in batch]]), dim=-1)
        lengths = [len(r["candidate_indices"]) for r in batch]
        width = max(lengths)
        indices = [
            r["candidate_indices"]
            + [r["candidate_indices"][0]] * (width - len(r["candidate_indices"]))
            for r in batch
        ]
        c = F.normalize(projection(vectors[torch.tensor(indices, device="cuda")]), dim=-1)
        result = torch.einsum("bd,bkd->bk", q, c) / 0.1
        mask = (
            torch.arange(width, device="cuda")[None]
            >= torch.tensor(lengths, device="cuda")[:, None]
        )
        return result.masked_fill(mask, -torch.inf)

    @torch.no_grad()
    def predict(batch, weight):
        values = scores(batch)
        for i, r in enumerate(batch):
            n = len(r["candidates"])
            lm = torch.tensor([c["log_score"] for c in r["candidates"]], device="cuda")
            values[i, :n] = lm + weight * values[i, :n]
        return (values.argmax(-1) + 1).cpu().tolist()

    def correct(batch, ranks):
        return sum(r["gold_rank"] == rank for r, rank in zip(batch, ranks))

    validation = parts["validation"]
    best = correct(validation, [1] * len(validation))
    choice = {"epoch": 0, "weight": 0.0, "validation_correct": best}
    state = {k: v.detach().clone() for k, v in projection.state_dict().items()}
    history = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(training)
        losses = []
        for start in range(0, len(training), 32):
            batch = training[start : start + 32]
            targets = torch.tensor([r["gold_rank"] - 1 for r in batch], device="cuda")
            loss = F.cross_entropy(scores(batch), targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "validation": {}}
        for weight in (0.25, 0.5, 1.0, 2.0):
            hits = correct(validation, predict(validation, weight))
            record["validation"][str(weight)] = hits
            if hits > best:
                best = hits
                choice = {"epoch": epoch, "weight": weight, "validation_correct": hits}
                state = {k: v.detach().clone() for k, v in projection.state_dict().items()}
        history.append(record)
        print(
            f"[train] seed {seed} epoch {epoch}/{epochs} loss {record['loss']:.4f} best {choice}",
            flush=True,
        )
    projection.load_state_dict(state)
    torch.save({k: v.cpu() for k, v in state.items()}, out / f"projection-s{seed}.pt")
    test = parts["test"]
    ranks = predict(test, choice["weight"])
    records = [
        {
            **r,
            "selected_rank": rank,
            "same": r["gold_rank"] == rank,
            "baseline_same": r["gold_rank"] == 1,
            "pred": r["candidates"][rank - 1]["meta"],
        }
        for r, rank in zip(test, ranks)
    ]
    with (out / f"test-s{seed}.jsonl").open("x") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def summary(selected):
        return {
            "n_candidate_sets": len(selected),
            "n_requests": len({r["text"] for r in selected}),
            "baseline_correct": sum(r["baseline_same"] for r in selected),
            "reranked_correct": sum(r["same"] for r in selected),
            "candidate_ceiling": sum(r["gold_rank"] > 0 for r in selected),
            "wins": sum(r["same"] and not r["baseline_same"] for r in selected),
            "losses": sum(r["baseline_same"] and not r["same"] for r in selected),
        }

    report = {
        "seed": seed,
        "selected_on_validation": choice,
        "test": summary(records),
        "by_generator_seed": {
            str(s): summary([r for r in records if r["generator_seed"] == s])
            for s in sorted({r["generator_seed"] for r in records})
        },
        "history": history,
    }
    (out / f"train-s{seed}.summary.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("okf")
    ap.add_argument("pools", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default="rows", help="generator's prepared rows directory")
    ap.add_argument("--base", default="Qwen/Qwen3-4B")
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cat = okf.load(a.okf)
    prepared = Path(a.rows)
    if json.loads((prepared / "prepare.json").read_text())["catalog_sha"] != cat.sha:
        raise ValueError("prepared rows must match the candidate catalog")
    source = [json.loads(line) for line in (prepared / "rows.jsonl").read_text().splitlines()]
    blocked = {group_key(r["meta"]) for r in source if r["split"].rstrip("+") == "dev"}
    for p in a.pools:
        report = json.loads(Path(p).with_suffix(".summary.json").read_text())
        if "interrupted" in report or report["catalog_sha"] != cat.sha:
            raise ValueError("candidate pools must be complete and catalog-compatible")
        cfg = report["model"]
        blocked.update(
            group_key(r["meta"])
            for r in source
            if r["split"].rstrip("+") not in ("dev", cfg["holdout"])
            and (cfg["keep_noisy"] or "noisy" not in r["tags"])
        )
    texts, rows, excluded = prepare(cat, a.pools, blocked)
    manifest = {
        "catalog_sha": cat.sha,
        "base": a.base,
        "epochs": a.epochs,
        "grouping": "gold atom sets, deterministic hash split 60/20/20",
        "filter": "explicit layers, no noisy tags, imperative Show checked; heuristic, not human-certified",
        "excluded_candidate_sets": excluded,
        "embedding_texts": len(texts),
        "partitions": {
            p: {
                "candidate_sets": sum(r["partition"] == p for r in rows),
                "groups": len({r["group"] for r in rows if r["partition"] == p}),
                "requests": len({r["text"] for r in rows if r["partition"] == p}),
            }
            for p in ("train", "validation", "test")
        },
        "input_sha256": [
            {"file": Path(p).name, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
            for p in a.pools
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "texts.json").write_text(json.dumps(texts, ensure_ascii=False))
    (out / "splits.json").write_text(json.dumps(rows, ensure_ascii=False))
    print(json.dumps(manifest, indent=2), flush=True)
    matrix = embed(texts, a.base, out)
    reports = [train(matrix, rows, out, seed=s, epochs=a.epochs) for s in (0, 1)]
    print(
        json.dumps([{k: v for k, v in r.items() if k != "history"} for r in reports], indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
