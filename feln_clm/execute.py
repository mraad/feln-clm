"""Execution precision/recall: run gold and predicted FELN on the project data.

Each OKF layer's ``resource`` (a File GDB feature class) is loaded once into a DuckDB file
with its geometry reprojected to EPSG:3035 (metres), so distance relations are metric; the
table takes the layer's ``table_name``. Queries compile with feln's ``FELNToDuckDB``.

Per request the returned OBJECTID sets are compared. Reported: execution match (same set;
also over requests whose gold set is non-empty), mean Jaccard (two empty sets count 1),
macro precision over requests with a non-empty prediction and macro recall over requests
with a non-empty gold set, and micro precision/recall (summed intersections over summed
predicted / gold sizes). A query that fails to
compile or run returns the empty set and is counted.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

from feln import FELNToDuckDB, parse_relation

from . import okf

SQL = FELNToDuckDB(objectid="OBJECTID", geometry="geometry")


def connect(cat: okf.Catalog, db: str, crs: str = "EPSG:4326"):
    import duckdb

    fresh = not Path(db).exists()
    con = duckdb.connect(db)
    con.execute("INSTALL spatial; LOAD spatial;")
    if fresh:
        for ly in cat.layers.values():
            path = unquote(urlparse(ly.resource).path)
            gdb, fc = str(Path(path).parent), Path(path).name
            cols = [
                r[0]
                for r in con.execute(
                    f"DESCRIBE SELECT * FROM ST_Read('{gdb}', layer='{fc}')"
                ).fetchall()
            ]
            geom = next(c for c in cols if c.lower() in ("shape", "geom", "geometry"))
            con.execute(
                f'CREATE TABLE "{ly.table}" AS SELECT * EXCLUDE ("{geom}"), '
                f"ST_Transform(\"{geom}\", '{crs}', 'EPSG:3035', always_xy := true) AS geometry "
                f"FROM ST_Read('{gdb}', layer='{fc}')"
            )
    return con


def run(con, cat: okf.Catalog, meta: dict) -> tuple[set[int], str]:
    try:
        layers = [SimpleNamespace(name=n, table_name=cat[n].table) for n in meta["layers"]]
        sql = SQL(layers, list(meta["where"]), [parse_relation(r) for r in meta["relations"]])
        return {r[0] for r in con.execute(sql).fetchall()}, ""
    except Exception as e:  # a bad prediction is a result (empty), not a crash
        return set(), f"{type(e).__name__}: {e}"[:300]


def evaluate(cat: okf.Catalog, results: str, db: str, pred_key: str = "pred") -> dict:
    con = connect(cat, db)
    rows = [json.loads(line) for line in open(results)]
    inter = n_pred = n_gold = match = errors = 0
    jac, prec, rec, nonempty = [], [], [], []
    for r in rows:
        gold, gerr = run(con, cat, r["gold"])
        pred, perr = run(con, cat, r[pred_key]) if r.get(pred_key) else (set(), "no prediction")
        if gerr:
            raise SystemExit(f"gold query failed for {r['text']!r}: {gerr}")
        errors += bool(perr)
        inter, n_pred, n_gold = inter + len(gold & pred), n_pred + len(pred), n_gold + len(gold)
        match += gold == pred
        jac.append(len(gold & pred) / len(gold | pred) if gold | pred else 1.0)
        if pred:
            prec.append(len(gold & pred) / len(pred))
        if gold:
            rec.append(len(gold & pred) / len(gold))
            nonempty.append(gold == pred)
    p, rc = inter / max(n_pred, 1), inter / max(n_gold, 1)
    mp, mr = sum(prec) / max(len(prec), 1), sum(rec) / max(len(rec), 1)
    return {"n": len(rows), "execution_match": match / len(rows), "mean_jaccard": sum(jac) / len(jac),
            "gold_nonempty": len(nonempty), "execution_match_nonempty": sum(nonempty) / max(len(nonempty), 1),
            "macro_precision": mp, "macro_recall": mr, "macro_f1": 2 * mp * mr / max(mp + mr, 1e-9),
            "micro_precision": p, "micro_recall": rc, "micro_f1": 2 * p * rc / max(p + rc, 1e-9),
            "prediction_errors": errors}  # fmt: skip
