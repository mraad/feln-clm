"""Text -> FELN: ask CLM the typed questions, then decode the best consistent query.

Four rounds of questions: layer structure; per-layer subtype and constrained column plus
relation/distance for the ``beam`` likeliest structures; the condition on each likely
column; the literal values of each likely condition. Each structure is scored by the sum
of its decisions' log-probabilities under hard constraints: a condition needs literals the
request offers, two layers never take the same literal, and two distance relations take
different distances when the request has more than one. Each quoted string or bare number
of the request that no condition uses costs ``penalty`` nats. ``confidence`` is the joint
probability of the winning decisions (penalties included).
"""

from __future__ import annotations

import json
import math
import time
from itertools import product
from pathlib import Path

from . import grammar as g
from . import okf

CONFIG = "feln-clm.json"
EPS = 1e-9


def _lp(p: float) -> float:
    return math.log(max(p, EPS))


class Translator:
    """``model_dirs``: one or more dirs, each a framing (feln-clm.json) with heads/*.pt.

    Every head answers every question; the answer is the renormalised geometric mean of
    their distributions. Heads of one framing share its state embeddings, so extra seeds
    cost only a projection.
    """

    def __init__(self, okf_dir: str, model_dirs: list[str], embedder=None, device: str = "cpu",
                 beam: int = 3, top: int = 4, threshold: float = 0.5, penalty: float = 2.0):  # fmt: skip
        from clm.engine import DEFAULT_MODEL, Engine

        self.cat = okf.load(okf_dir)
        self.members: list[tuple[str, list[str]]] = []  # (framing, head names)
        heads = {}
        for i, d in enumerate(map(Path, model_dirs)):
            cfg = json.loads((d / CONFIG).read_text())
            if cfg["catalog_sha"] != self.cat.sha:
                raise ValueError(
                    f"{d} was trained on catalog {cfg['catalog_sha']}, OKF is {self.cat.sha}"
                )
            names = {f"m{i}-{p.stem}": str(p) for p in sorted((d / "heads").glob("*.pt"))}
            if not names:
                raise ValueError(f"{d}/heads has no .pt checkpoints")
            heads.update(names)
            self.members.append((cfg["format"], list(names)))
        self.fmt = "+".join(f for f, _ in self.members)
        self.beam, self.top, self.threshold, self.penalty = beam, top, threshold, penalty
        self.labels = {
            lab.lower()
            for ly in self.cat.layers.values()
            for lab in ly.columns[ly.subtype].domain.values()
        }
        # the first head is Engine's default model, so CLM's reference head is never loaded
        first = next(iter(heads))
        self.members[0][1][0] = DEFAULT_MODEL
        self.engine = Engine(
            embedder=embedder, checkpoint=heads.pop(first), models=heads, device=device
        )
        self.structs = g.structures(self.cat)

    def warmup(self) -> int:
        """Embed every fixed option text once, so first requests pay only for their states."""
        cat, qs = self.cat, {"layers": g.q_layers(self.cat)}
        for name, ly in cat.layers.items():
            p = name  # option texts do not depend on which layer is primary
            qs[f"subtype:{name}"] = g.q_subtype(
                cat, name, p, [*ly.columns[ly.subtype].domain, g.ANY]
            )
            qs[f"column:{name}"] = cq = g.q_column(cat, name, p)
            for col in cq["criteria"]:
                if col != g.NONE:
                    qs[f"op:{name}|{col}"] = g.q_op(cat, name, p, col)
            qs[f"relation:{name}"] = g.q_relation(cat, name, p, g.ANY)
        texts = {
            t
            for fmt, _ in self.members
            for q in g.frame(fmt, "", qs)[1].values()
            for t in q["criteria"].values()
        }
        self.engine.embedder.embed(sorted(texts))
        return len(texts)

    def _answer(self, text: str, qs: dict) -> dict:
        if not qs:
            return {}
        logs: dict[str, dict[str, float]] = {qid: {} for qid in qs}
        n = sum(len(names) for _, names in self.members)
        for fmt, names in self.members:
            state, framed = g.frame(fmt, text, qs)
            for name in names:
                for qid, a in self.engine.answer(state, framed, model=name)["answers"].items():
                    for k, p in a["probabilities"].items():
                        logs[qid][k] = logs[qid].get(k, 0.0) + _lp(p) / n
        out = {}
        for qid, lg in logs.items():
            z = sum(math.exp(v) for v in lg.values())
            out[qid] = {k: math.exp(v) / z for k, v in lg.items()}
        return out

    def _top(self, dist: dict) -> list[str]:
        return sorted(dist, key=dist.get, reverse=True)[: self.top]

    def ask(self, text: str) -> dict:
        t0 = time.perf_counter()
        text = g.normalize(text)
        cat, found, named = self.cat, g.spans(text), g.mentions(self.cat, text)
        probs = self._answer(text, {"layers": g.q_layers(cat)})
        ranked = sorted(probs["layers"], key=probs["layers"].get, reverse=True)[: self.beam]

        qs = {}
        for key in ranked:
            p, secs = self.structs[key]
            for name in (p, *secs):
                codes = g.subtype_codes(cat, name, named)
                if len(codes) > 1:
                    qs[f"subtype:{name}@{p}"] = g.q_subtype(cat, name, p, codes)
                qs[f"column:{name}@{p}"] = g.q_column(cat, name, p)
        probs.update(self._answer(text, qs))
        subtypes = {key: self._subtypes(key, probs, named) for key in ranked}

        qs = {}
        for key in ranked:
            p, secs = self.structs[key]
            for name in secs:
                code = subtypes[key][0][name]
                qs[f"relation:{name}@{p}={code}"] = g.q_relation(cat, name, p, code)
                dq = g.q_distance(cat, name, p, code, found)
                if len(dq["criteria"]) > 1:
                    qs[f"distance:{name}@{p}={code}"] = dq
        for qid in [q for q in probs if q.startswith("column:")]:
            name, p = qid[7:].split("@")
            for col in self._top(probs[qid]):
                if col != g.NONE:
                    qs[f"op:{name}@{p}|{col}"] = g.q_op(cat, name, p, col)
        probs.update(self._answer(text, qs))

        qs, offers = {}, {}
        for qid in [q for q in probs if q.startswith("op:")]:
            name, p = qid[3:].split("|")[0].split("@")
            ly = cat[name]
            for key in self._top(probs[qid]):
                opt = g.options(ly)[key]
                offers[(name, p, key)] = cands = g.values(opt, ly.columns[opt.col], found, text)
                if len(cands) > 1:
                    qs[f"value:{name}@{p}|{key}"] = g.q_value(cat, name, p, opt, cands)
        probs.update(self._answer(text, qs))

        scored = sorted((self._decode(key, probs, subtypes[key], offers, found) for key in ranked),
                        key=lambda r: r["score"], reverse=True)  # fmt: skip
        best = scored[0]
        confidence = math.exp(best["score"])
        return {
            "text": text,
            "meta": best["meta"],
            "status": "accepted" if confidence >= self.threshold else "abstain",
            "confidence": round(confidence, 4),
            "decisions": best["decisions"],
            "alternatives": [{"meta": r["meta"], "confidence": round(math.exp(r["score"]), 4)} for r in scored[1:]],
            "probabilities": probs,
            "seconds": round(time.perf_counter() - t0, 3),
        }  # fmt: skip

    def _subtypes(self, key: str, probs: dict, named: dict[str, int]) -> tuple[dict, float]:
        """Best joint subtype codes of a structure: no subtype name used more often than said."""
        p, secs = self.structs[key]
        layers = (p, *secs)
        alts = []
        for name in layers:
            dist = probs.get(f"subtype:{name}@{p}", {g.ANY: 1.0})
            alts.append(sorted(dist.items(), key=lambda kv: -kv[1])[: self.top])

        def fits(combo) -> bool:
            used: dict[str, int] = {}
            for name, (code, _) in zip(layers, combo):
                lab = g.subtype_label(self.cat, name, code)
                used[lab] = used.get(lab, 0) + (lab != "")
            return all(n <= named.get(lab, 0) for lab, n in used.items() if lab)

        combos = [c for c in product(*alts) if fits(c)] or list(product(*alts))
        best = max(combos, key=lambda c: sum(_lp(pr) for _, pr in c))
        return {name: code for name, (code, _) in zip(layers, best)}, sum(_lp(pr) for _, pr in best)

    def _decode(
        self, key: str, probs: dict, subtypes: tuple[dict, float], offers: dict, found: list
    ) -> dict:
        p, secs = self.structs[key]
        d = g.Decisions(p, secs)
        score = _lp(probs["layers"][key]) + subtypes[1]
        chosen = {"layers": (key, probs["layers"][key])}
        for name in d.layers:
            d.subtype[name] = subtypes[0][name]
            pr = probs.get(f"subtype:{name}@{p}", {g.ANY: 1.0}).get(d.subtype[name], 0.0)
            chosen[f"subtype:{name}@{p}"] = (d.subtype[name], pr)

        per_layer = []  # [(logp, option key, values)] per layer, feasible only
        for name in d.layers:
            cq = probs[f"column:{name}@{p}"]
            alts = [(_lp(cq[g.NONE]), g.NONE, ())]
            for col in self._top(cq):
                oq = probs.get(f"op:{name}@{p}|{col}", {})
                for key in self._top(oq):
                    vq = probs.get(f"value:{name}@{p}|{key}")
                    for vals in offers.get((name, p, key), []):
                        lp = (
                            _lp(cq[col])
                            + _lp(oq[key])
                            + (_lp(vq[g.value_label(vals)]) if vq else 0.0)
                        )
                        alts.append((lp, key, vals))
            per_layer.append(sorted(alts, reverse=True)[:6])
        literals = {
            s.text.lower()
            for s in found
            if s.kind == "num" or (s.kind == "str" and s.text.lower() not in self.labels)
        }

        def value(combo) -> float:  # log-prob minus a penalty per request literal left unexplained
            used = {v.lower() for _, _, vals in combo for v in vals}
            return sum(x[0] for x in combo) - self.penalty * len(literals - used)

        best = max((c for c in product(*per_layer) if _distinct([v for _, _, v in c])), key=value)
        score -= self.penalty * len(literals - {v.lower() for _, _, vals in best for v in vals})
        for name, (lp, opt_key, vals) in zip(d.layers, best):
            d.condition[name], d.value[name] = opt_key, vals
            score += lp
            chosen[f"condition:{name}@{p}"] = (
                f"{opt_key} {g.value_label(vals)}".strip(),
                math.exp(lp),
            )

        dists = list(dict.fromkeys(s.label for s in found if s.kind == "dist"))
        per_sec = []
        for name in secs:
            tag = f"{name}@{p}={d.subtype[name]}"
            rq, dq = probs[f"relation:{tag}"], probs.get(f"distance:{tag}")
            alts = []
            for kind, pr in rq.items():
                if kind not in g.DISTANCE:
                    alts.append((_lp(pr), kind, ""))
                for lab in dists if kind in g.DISTANCE else []:
                    alts.append((_lp(pr) + (_lp(dq[lab]) if dq else 0.0), kind, lab))
            per_sec.append(sorted(alts, reverse=True)[:8])
        if secs:
            ok = lambda c: len(dists) < 2 or _distinct([(lab,) for _, _, lab in c if lab])  # noqa: E731
            combos = [c for c in product(*per_sec) if ok(c)] or list(product(*per_sec))
            for name, (lp, kind, lab) in zip(secs, max(combos, key=lambda c: sum(x[0] for x in c))):
                d.relation[name], d.distance[name] = kind, lab
                score += lp
                chosen[f"relation:{name}@{p}"] = (f"{kind} {lab}".strip(), math.exp(lp))

        decisions = {q: {"choice": c, "probability": round(pr, 4)} for q, (c, pr) in chosen.items()}
        return {"meta": g.compose(self.cat, d), "score": score, "decisions": decisions}


def _distinct(value_tuples: list[tuple[str, ...]]) -> bool:
    flat = [v for vals in value_tuples for v in vals]
    return len(flat) == len(set(flat))
