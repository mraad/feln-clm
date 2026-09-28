"""Split examples and write CLM typed-decision rows for ``CLM/train/finetune.py --task choice``.

    python -m feln_clm.prepare data/okf data/FELN.json data/laya-heldout.jsonl out/rows \
        --extra data/laya-generated.jsonl

Rows land in ``OUT/feln/{train,test}-00000.parquet`` with ``state`` (the request),
``questions`` and ``gold`` (JSON), one row per request. Questions are teacher-forced on
the gold structure: the ones inference asks once the gold decisions are taken. Requests
the grammar cannot reproduce are skipped and counted, never rewritten.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from . import grammar as g
from . import okf


def questions(cat: okf.Catalog, text: str, d: g.Decisions) -> tuple[dict, dict]:
    """(questions, gold labels) for one request under its gold decisions."""
    qs, gold = {"layers": g.q_layers(cat)}, {"layers": d.structure}
    found, named = g.spans(text), g.mentions(cat, text)
    p = d.primary
    for name in d.layers:
        ly = cat[name]
        codes = g.subtype_codes(cat, name, named)
        if d.subtype[name] in codes and len(codes) > 1:  # unnamed gold: nothing to learn
            qs[f"subtype:{name}@{p}"] = g.q_subtype(cat, name, p, codes)
            gold[f"subtype:{name}@{p}"] = d.subtype[name]
        opt = g.options(ly)[d.condition[name]]
        qs[f"column:{name}@{p}"] = g.q_column(cat, name, p)
        gold[f"column:{name}@{p}"] = opt.col or g.NONE
        if opt.col:
            qs[f"op:{name}@{p}|{opt.col}"] = g.q_op(cat, name, p, opt.col)
            gold[f"op:{name}@{p}|{opt.col}"] = opt.key
        if opt.needs:
            col = ly.columns[opt.col]
            cands = g.values(opt, col, found, text)
            if len(cands) > 1:
                qid = f"value:{name}@{p}|{opt.key}"
                qs[qid] = g.q_value(cat, name, p, opt, cands)
                gold[qid] = g.value_label(d.value[name])
    for name in d.secondaries:
        tag = f"{name}@{p}={d.subtype[name]}"
        qs[f"relation:{tag}"] = g.q_relation(cat, name, p, d.subtype[name])
        gold[f"relation:{tag}"] = d.relation[name]
        if d.relation[name] in g.DISTANCE:
            q = g.q_distance(cat, name, p, d.subtype[name], found)
            if len(q["criteria"]) > 1:
                qs[f"distance:{tag}"] = q
                gold[f"distance:{tag}"] = d.distance[name]
    for qid, label in gold.items():
        assert label in qs[qid]["criteria"], (qid, label)
    return qs, gold


def val_texts(rows_dir: str, seed: int = 1234, frac: float = 0.1) -> list[str]:
    """Texts of the validation rows ``CLM/train/finetune.py --task choice`` holds out (same
    id shuffle), restricted to the original FELN examples."""
    import random

    rows = pq.read_table(os.path.join(rows_dir, "feln", "train-00000.parquet")).to_pylist()
    ids = sorted({r["id"] for r in rows})
    random.Random(seed).shuffle(ids)
    val = set(ids[: max(1, int(round(frac * len(ids))))])
    return [r["text"] for r in rows if r["id"] in val and "-feln-" in r["id"]]


def build(cat, examples, source, fmt):
    rows, skipped = [], collections.Counter()
    for i, x in enumerate(examples):
        try:
            d = g.decompose(cat, x["text"], x["meta"])
        except ValueError as e:
            skipped[str(e).split(":")[0]] += 1
            continue
        qs, gold = questions(cat, x["text"], d)
        state, qs = g.frame(fmt, x["text"], qs)
        rows.append({
            "id": f"{source}-{i}", "workflow": source, "state": state, "text": x["text"],
            "questions": json.dumps(qs, ensure_ascii=False),
            "gold": json.dumps({k: {"label": v} for k, v in gold.items()}, ensure_ascii=False),
        })  # fmt: skip
    return rows, skipped


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("okf")
    ap.add_argument("examples", help="FELN.json: [{text, meta}]")
    ap.add_argument("heldout", help="jsonl whose texts form the test split")
    ap.add_argument("out")
    ap.add_argument("--extra", nargs="*", default=[], help="jsonl added to training only")
    ap.add_argument(
        "--format", choices=g.FORMATS, default="options", help="how questions are framed"
    )
    a = ap.parse_args()

    cat = okf.load(a.okf)
    examples = json.load(open(a.examples))
    held = {json.loads(line)["text"] for line in open(a.heldout)}
    test = [x for x in examples if x["text"] in held]
    train = [x for x in examples if x["text"] not in held]
    if len(test) != len(held):
        raise SystemExit(f"{len(held) - len(test)} heldout texts are missing from {a.examples}")
    seen = {x["text"] for x in examples}
    extra = [x for f in a.extra for line in open(f)
             if (x := json.loads(line))["text"] not in seen]  # fmt: skip

    os.makedirs(os.path.join(a.out, "feln"), exist_ok=True)
    report = {"catalog_sha": cat.sha, "okf": os.path.abspath(a.okf), "format": a.format}
    splits = {"train": [(train, "feln"), (extra, "generated")], "test": [(test, "feln")]}
    for split, parts in splits.items():
        rows = []
        for data, source in parts:
            got, skipped = build(cat, data, f"{split}-{source}", a.format)
            rows += got
            report[f"{split}/{source}"] = {
                "examples": len(data),
                "rows": len(got),
                "skipped": dict(skipped),
            }
        pq.write_table(
            pa.Table.from_pylist(rows), os.path.join(a.out, "feln", f"{split}-00000.parquet")
        )
        report[f"{split}/questions"] = sum(len(json.loads(r["gold"])) for r in rows)
    json.dump(report, open(os.path.join(a.out, "prepare.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
