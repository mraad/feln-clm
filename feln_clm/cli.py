"""feln-clm: ask, evaluate, execute and serve the text -> FELN translator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def translator(a):
    from .decode import Translator

    return Translator(a.okf, a.model, backend=a.backend, beam=a.beam, threshold=a.threshold)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ask", "evaluate", "serve"):
        p = sub.add_parser(name)
        p.add_argument("okf")
        p.add_argument("model", help="model dir: feln-clm.json, adapter.safetensors, values.json")
        p.add_argument("--backend", default="mlx", help="mlx (Mac) | torch (CUDA)")
        p.add_argument("--beam", type=int, default=8)
        p.add_argument("--threshold", type=float, default=0.5)
        if name == "ask":
            p.add_argument("text")
        if name == "evaluate":
            p.add_argument("rows", help="prepare output dir (rows.jsonl)")
            p.add_argument("--split", default="dev", help="dev | fold0 .. fold4")
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
    a = ap.parse_args()

    if a.cmd == "execute":
        from . import execute, okf

        out = execute.evaluate(okf.load(a.okf), a.results, a.db, a.pred_key)
        Path(a.results).with_suffix(".execution.json").write_text(json.dumps(out, indent=1))
        return print(json.dumps(out, indent=1))
    if a.cmd == "evaluate":
        from .evaluate import run

        rows = [json.loads(line) for line in open(Path(a.rows) / "rows.jsonl")]
        examples = [r for r in rows if r["split"] == a.split]
        print(json.dumps(run(translator(a), examples, a.output), indent=1))
    elif a.cmd == "ask":
        print(json.dumps(translator(a).ask(a.text), indent=1, ensure_ascii=False))
    else:
        from .server import serve

        serve(translator(a), a.port)


if __name__ == "__main__":
    main()
