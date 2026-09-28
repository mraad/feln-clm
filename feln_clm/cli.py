"""feln-clm: ask, evaluate, package and serve the text -> FELN translator."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .decode import CONFIG, Translator


def translator(a) -> Translator:
    from .embedders import make

    return Translator(a.okf, a.model, embedder=make(a.encoder), device=a.device,
                      beam=a.beam, top=a.top, threshold=a.threshold, penalty=a.penalty)  # fmt: skip


def package(rows: str, runs: list[str], out: str) -> None:
    """Prepared rows + finetune runs (one framing, any seeds) -> model dir."""
    prep = json.loads((Path(rows) / "prepare.json").read_text())
    (Path(out) / "heads").mkdir(parents=True, exist_ok=False)
    members = []
    for i, run in enumerate(runs):
        summary = json.loads((Path(run) / "finetune_summary.json").read_text())
        shutil.copy(Path(run) / "best_head.pt", Path(out) / "heads" / f"{i}.pt")
        members.append({"run": Path(run).name, "val_acc": summary["val_acc"],
                        "test_question_acc": summary["test"]["acc"], "best_epoch": summary["best_epoch"],
                        "args": summary["args"]})  # fmt: skip
    cfg = {"format": prep["format"], "catalog_sha": prep["catalog_sha"], "encoder": "Qwen/Qwen3-8B",
           "heads": members}  # fmt: skip
    (Path(out) / CONFIG).write_text(json.dumps(cfg, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ask", "evaluate", "serve"):
        p = sub.add_parser(name)
        p.add_argument("okf")
        p.add_argument("model", nargs="+", help="model dir(s): feln-clm.json + heads/*.pt")
        p.add_argument(
            "--encoder", default="mlx", help="mlx | vllm | http://host:port/v1/embeddings"
        )
        p.add_argument("--device", default="cpu", help="where the projection heads run")
        p.add_argument("--beam", type=int, default=3)
        p.add_argument("--top", type=int, default=4)
        p.add_argument("--threshold", type=float, default=0.5)
        p.add_argument("--penalty", type=float, default=2.0, help="nats per unexplained literal")
        if name == "ask":
            p.add_argument("text")
        if name == "evaluate":
            p.add_argument("examples", help="FELN.json holding the gold queries")
            p.add_argument(
                "heldout",
                help="jsonl whose texts are evaluated, or val:ROWS_DIR[:SEED] for "
                "the finetune validation split of those rows (FELN examples)",
            )
            p.add_argument("--output", required=True)
        if name == "serve":
            p.add_argument("--port", type=int, default=8710)
    p = sub.add_parser("execute", help="execution precision/recall of an evaluate JSONL")
    p.add_argument("okf")
    p.add_argument("results", help="evaluate output (gold + pred per line)")
    p.add_argument(
        "--db", default="out/project.duckdb", help="DuckDB file, built from the OKF resources"
    )
    p.add_argument("--pred-key", default="pred", help="JSON key of the predicted FELN")
    p = sub.add_parser("package")
    p.add_argument("rows", help="prepared rows dir (prepare.json)")
    p.add_argument("out")
    p.add_argument("runs", nargs="+", help="CLM finetune --out-dir(s) of that framing")
    a = ap.parse_args()

    if a.cmd == "execute":
        from . import execute, okf

        out = execute.evaluate(okf.load(a.okf), a.results, a.db, a.pred_key)
        Path(a.results).with_suffix(".execution.json").write_text(json.dumps(out, indent=1))
        return print(json.dumps(out, indent=1))
    if a.cmd == "package":
        return package(a.rows, a.runs, a.out)
    if a.cmd == "evaluate":
        from .evaluate import run

        gold = {x["text"]: x for x in json.load(open(a.examples))}
        if a.heldout.startswith("val:"):
            from .prepare import val_texts

            rows, _, seed = (a.heldout[4:] + ":1234").split(":")[:3]
            texts = val_texts(rows, int(seed))
        else:
            texts = [json.loads(line)["text"] for line in open(a.heldout)]
        examples = [gold[t] for t in texts if t in gold]
        print(json.dumps(run(translator(a), examples, a.output), indent=1))
    elif a.cmd == "ask":
        res = translator(a).ask(a.text)
        print(json.dumps({k: res[k] for k in ("meta", "status", "confidence", "decisions", "alternatives", "seconds")},
                         indent=1, ensure_ascii=False))  # fmt: skip
    else:
        from .server import serve

        serve(translator(a), a.port)


if __name__ == "__main__":
    main()
