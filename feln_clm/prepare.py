"""Split examples and write the generator's training rows.

    python -m feln_clm.prepare data/okf data/FELN.json data/laya-heldout.jsonl out/rows \
        --extra data/laya-generated.jsonl --db out/project.duckdb

``OUT/rows.jsonl``: one row per request with ``split`` (``dev`` = the held-out texts,
``fold0``..``fold4`` = the other FELN examples, ``extra`` = training only), ``prompt``,
``pieces`` (the target, one string per choice), ``tags`` and the gold ``meta``.
``OUT/values.json`` is the literal index the prompt hints come from. Requests the grammar
cannot reproduce are skipped and counted, never rewritten.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random

from . import grammar as g
from . import okf

FOLDS = 5


def value_sets(index: dict) -> dict[str, dict[str, set[str]]]:
    return {n: {c: set(v) for c, v in cols.items()} for n, cols in index.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("okf")
    ap.add_argument("examples", help="FELN.json: [{text, meta}]")
    ap.add_argument("heldout", help="jsonl whose texts form the dev split")
    ap.add_argument("out")
    ap.add_argument("--extra", nargs="*", default=[], help="jsonl added to training only")
    ap.add_argument("--db", default="out/project.duckdb", help="DuckDB built from the OKF")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    from .execute import connect, value_index

    os.makedirs(a.out, exist_ok=False)
    cat = okf.load(a.okf)
    index = value_index(cat, connect(cat, a.db))
    sets = value_sets(index)
    examples = json.load(open(a.examples))
    held = {json.loads(line)["text"] for line in open(a.heldout)}
    if missing := held - {x["text"] for x in examples}:
        raise SystemExit(f"{len(missing)} heldout texts are missing from {a.examples}")
    rest = [x for x in examples if x["text"] not in held]
    random.Random(a.seed).shuffle(rest)
    seen = {x["text"] for x in examples}
    parts = [("dev", x) for x in examples if x["text"] in held]
    parts += [(f"fold{i % FOLDS}", x) for i, x in enumerate(rest)]
    parts += [("extra", x) for f in a.extra for line in open(f)
              if (x := json.loads(line))["text"] not in seen]  # fmt: skip

    counts, skipped = collections.Counter(), collections.Counter()
    with open(os.path.join(a.out, "rows.jsonl"), "w") as f:
        for i, (split, x) in enumerate(parts):
            text = g.normalize(x["text"])
            try:
                d = g.decompose(cat, text, x["meta"])
            except ValueError as e:
                skipped[f"{split[:4]}/{str(e).split(':')[0]}"] += 1
                continue
            tags = g.tags(cat, text, d)
            counts[split] += 1
            counts.update(f"{split[:4]}/{t}" for t in tags)
            row = {"id": i, "split": split, "text": x["text"], "meta": x["meta"], "tags": tags,
                   "prompt": g.prompt(text, g.hints(cat, sets, g.spans(text))),
                   "pieces": g.pieces(cat, d)}  # fmt: skip
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    json.dump(index, open(os.path.join(a.out, "values.json"), "w"), ensure_ascii=False)
    report = {"catalog_sha": cat.sha, "okf": os.path.abspath(a.okf), "seed": a.seed,
              "rows": dict(counts), "skipped": dict(skipped)}  # fmt: skip
    json.dump(report, open(os.path.join(a.out, "prepare.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
