"""End-to-end evaluation: exact FELN match, atom precision/recall, selective accuracy.

Atoms of a FELN are its primary layer, each related layer, each WHERE predicate (DNF atom
per layer, literal casts stripped) and each relation (kind, metres). Precision and recall
are micro-averaged over all requests. One JSONL line per request is flushed as it is
produced; the summary lands next to it.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from feln import FELN, FELNCompare, parse_relation
from feln.compare import _groups
from feln.model import canon_kind
from feln.units import to_meters

THRESHOLDS = (0.5, 0.8, 0.9, 0.95)


def atoms(meta: dict) -> set:
    out = {("primary", meta["layers"][0].lower())}
    for i, (name, where) in enumerate(zip(meta["layers"], meta["where"])):
        name = name.lower()
        if i:
            out.add(("layer", name))
        if where.strip():
            out |= {
                ("where", name, a)
                for grp in _groups(where.replace(" ILIKE ", " LIKE "))
                for a in grp
            }
    for name, rel in zip(meta["layers"][1:], meta["relations"]):
        r = parse_relation(rel)
        metres = round(to_meters(r.distance, r.unit), 2) if r.unit else None
        out.add(("relation", name.lower(), canon_kind(r.kind), metres))
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    tp = sum(r["atoms_tp"] for r in rows)
    pred = sum(r["atoms_pred"] for r in rows)
    gold = sum(r["atoms_gold"] for r in rows)
    p, rc = tp / max(pred, 1), tp / max(gold, 1)
    out = {
        "n": n,
        "exact": sum(r["same"] for r in rows) / n,
        "partial": statistics.mean(r["partial"] for r in rows),
        "atom_precision": p,
        "atom_recall": rc,
        "atom_f1": 2 * p * rc / max(p + rc, 1e-9),
        "median_seconds": statistics.median(r["seconds"] for r in rows),
    }
    for t in THRESHOLDS:
        acc = [r for r in rows if r["confidence"] >= t]
        out[f"coverage@{t}"] = len(acc) / n
        out[f"accuracy@{t}"] = sum(r["same"] for r in acc) / max(len(acc), 1)
    return out


def run(translator, examples: list[dict], output: str) -> dict:
    path = Path(output)
    if path.exists():
        raise SystemExit(f"{output} exists; evidence is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with path.open("w") as f:
        for i, x in enumerate(examples):
            res = translator.ask(x["text"])
            pred, gold = res["meta"], x["meta"]
            a, b = atoms(pred), atoms(gold)
            row = {
                "index": i, "text": x["text"], "gold": gold, "pred": pred,
                "same": FELN(**pred).same(FELN(**gold)),
                "partial": FELNCompare.partial(FELN(**gold), FELN(**pred)),
                "atoms_tp": len(a & b), "atoms_pred": len(a), "atoms_gold": len(b),
                "confidence": res["confidence"], "status": res["status"],
                "seconds": res["seconds"], "decisions": res["decisions"],
            }  # fmt: skip
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(
                    f"[eval] {i + 1}/{len(examples)} exact {summarize(rows)['exact']:.3f}",
                    flush=True,
                )
    summary = summarize(rows)
    path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    return summary
