"""Offline candidate-recall diagnostic; outputs are evaluation evidence, not training data.

python -m feln_clm.candidates data/okf models/q4b out/rows-cmp3 \
    --split dev --output out/candidates-dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from . import grammar as g
from .decode import Translator


def summarize(rows: list[dict], k: int) -> dict:
    def rates(selected):
        n = len(selected)
        return {
            "n": n,
            "baseline_correct": sum(r["baseline_same"] for r in selected),
            "recall": {
                str(i): sum(0 < r["gold_rank"] <= i for r in selected) / n for i in range(1, k + 1)
            }
            if n
            else {},
            "recoverable_baseline_errors": sum(
                not r["baseline_same"] and r["gold_rank"] > 0 for r in selected
            ),
        }

    return {
        **rates(rows),
        "clean": rates([r for r in rows if "noisy" not in r["tags"]]),
        "explicit_clean": rates([r for r in rows if not {"noisy", "implicit"} & set(r["tags"])]),
        "implicit_clean": rates(
            [r for r in rows if "implicit" in r["tags"] and "noisy" not in r["tags"]]
        ),
        "noisy": rates([r for r in rows if "noisy" in r["tags"]]),
        "top1_changed": sum(not g.same(r["baseline"], r["candidates"][0]["meta"]) for r in rows),
        "median_baseline_seconds": statistics.median(r["baseline_seconds"] for r in rows),
        "median_candidate_seconds": statistics.median(r["candidate_seconds"] for r in rows),
        "min_candidates": min(len(r["candidates"]) for r in rows),
        "max_candidates": max(len(r["candidates"]) for r in rows),
    }


def run(
    tr, examples: list[dict], output: str, *, k: int, split: str, usage: str = "evaluation_only"
) -> dict:
    if usage == "reranker_pool" and (split == "dev" or split != tr.cfg.get("holdout")):
        raise ValueError("reranker pools require the generating model's held-out fold")
    path = Path(output)
    summary_path = path.with_suffix(".summary.json")
    if path.exists() or summary_path.exists():
        raise ValueError("output already exists; evidence is never overwritten")
    if not examples:
        raise ValueError("no examples in selected split")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with path.open("x") as f:
        try:
            for x in examples:
                baseline = tr.ask(x["text"])
                res = tr.ask(x["text"], nbest=k)
                candidates = res["candidates"]
                gold = x["meta"]
                row = {
                    "id": x["id"],
                    "split": split,
                    "usage": usage,
                    "generator_holdout": tr.cfg.get("holdout", ""),
                    "generator_seed": tr.cfg.get("seed"),
                    "text": x["text"],
                    "tags": x.get("tags", []),
                    "gold": gold,
                    "baseline": baseline["meta"],
                    "baseline_same": g.same(baseline["meta"], gold),
                    "baseline_confidence": baseline["confidence"],
                    "baseline_seconds": baseline["seconds"],
                    "candidate_seconds": res["seconds"],
                    "gold_rank": next(
                        (i for i, c in enumerate(candidates, 1) if g.same(c["meta"], gold)), 0
                    ),
                    "candidates": candidates,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                rows.append(row)
                if len(rows) % 20 == 0:
                    print(f"[candidates] {len(rows)}/{len(examples)}", flush=True)
        finally:
            if rows:
                report = {
                    "model": tr.cfg,
                    "catalog_sha": tr.cat.sha,
                    "split": split,
                    "beam": tr.beam,
                    "k": k,
                    "usage": usage,
                    "search": "distinct completions, kth-score stopping, beam-pruned",
                    **summarize(rows, k),
                }
                if len(rows) != len(examples):
                    report["interrupted"] = f"{len(rows)}/{len(examples)}"
                with summary_path.open("x") as f:
                    json.dump(report, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("okf")
    ap.add_argument("model")
    ap.add_argument("rows")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--output", required=True)
    ap.add_argument("--backend", choices=("mlx", "torch"), default="mlx")
    ap.add_argument(
        "--usage", choices=("evaluation_only", "reranker_pool"), default="evaluation_only"
    )
    a = ap.parse_args()
    if not 2 <= a.k <= a.beam:
        ap.error("require 2 <= k <= beam")
    cfg = json.loads((Path(a.model) / "feln-clm.json").read_text())
    if a.split != "dev" and a.split != cfg.get("holdout"):
        ap.error("select dev or the model's held-out fold; training rows inflate recall")
    if a.usage == "reranker_pool" and a.split == "dev":
        ap.error("development requests cannot become reranker training data")
    prep = json.loads((Path(a.rows) / "prepare.json").read_text())
    if prep["catalog_sha"] != cfg["catalog_sha"]:
        ap.error("prepared rows and model must use the same catalog")
    examples = [json.loads(line) for line in (Path(a.rows) / "rows.jsonl").read_text().splitlines()]
    examples = [r for r in examples if r["split"] == a.split]
    tr = Translator(a.okf, a.model, beam=a.beam, backend=a.backend)
    print(json.dumps(run(tr, examples, a.output, k=a.k, split=a.split, usage=a.usage), indent=2))


if __name__ == "__main__":
    main()
