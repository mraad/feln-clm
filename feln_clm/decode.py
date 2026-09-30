"""Text -> FELN: constrained beam search of the fine-tuned generator over catalog pieces.

The model writes a query as pieces (``grammar.pieces``). At every step only tokens that
continue a legal piece are allowed (a token trie of the slot's candidates, renormalised),
so the output is always a valid FELN over the OKF catalog, literals are always spans of the
request, and SQL is rendered by ``grammar.compose``. Hard rules: at most ``MAX_LAYERS``
distinct layers, a literal serves one condition, and when the request gives several
distances each relation takes a different one. ``confidence`` is the probability of the
winning sequence; ``alternatives`` are the other finished beams.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from . import grammar as g
from . import lm, okf
from .prepare import value_sets

CONFIG = "feln-clm.json"
ADAPTER = "adapter.safetensors"
VALUES = "values.json"


class Translator:
    def __init__(self, okf_dir: str, model_dir: str, backend: str = "mlx", beam: int = 8,
                 threshold: float = 0.5, model=None):  # fmt: skip
        self.cat = okf.load(okf_dir)
        d = Path(model_dir)
        self.cfg = json.loads((d / CONFIG).read_text())
        if self.cfg["catalog_sha"] != self.cat.sha:
            raise ValueError(f"{d} was trained on catalog {self.cfg['catalog_sha']}, OKF is {self.cat.sha}")  # fmt: skip
        self.values = value_sets(json.loads((d / VALUES).read_text()))
        self.lm = model or lm.load(backend, self.cfg["base"], str(d / ADAPTER), self.cfg["scale"])
        self.beam, self.threshold = beam, threshold
        self._ids: dict[str, tuple[int, ...]] = {}

    def ids(self, piece: str) -> tuple[int, ...]:
        if piece not in self._ids:
            self._ids[piece] = tuple(lm.encode(self.lm.tok, piece))
        return self._ids[piece]

    # ------------------------------------------------------------------ grammar state
    def _trie(self, acts: tuple) -> dict:
        """Token trie of the pieces allowed after ``acts``; a leaf is (piece, action)."""
        heads = [a for _, a, _ in acts if a[0] == "head"]
        used = [a[1] for a in heads]
        taken = frozenset(v.lower() for _, a, _ in acts if a[0] == "cond" for v in a[2])
        key = (tuple(used), bool(acts) and acts[-1][1][0] == "head", taken,
               frozenset(a[4] for a in heads if a[4]))  # fmt: skip
        if key in self._tries:
            return self._tries[key]
        cands = []
        if key[1]:
            cands += [(p, ("cond", *a)) for p, a in self._conds(used[-1])
                      if not taken & {v.lower() for v in a[1]}]  # fmt: skip
        if len(used) < min(g.MAX_LAYERS, len(self.cat.layers)):
            dists = (
                self.dists if len(self.dists) < 2 else [x for x in self.dists if x not in key[3]]
            )
            cands += [(p, ("head", *a)) for p, a in g.heads(self.cat, used, dists)]
        if used:
            cands.append((g.END, ("end",)))
        root: dict = {}
        for piece, act in cands:
            node = root
            for t in self.ids(piece):
                node = node.setdefault(t, {})
            node[-1] = (piece, act)  # -1 marks a complete piece (a node may also continue)
        self._tries[key] = root
        return root

    def _conds(self, name: str) -> list:
        if name not in self._cond_cache:
            self._cond_cache[name] = g.conditions(self.cat, name, self.found, self.text)
        return self._cond_cache[name]

    def _moves(self, acts: tuple, node: dict, part: float) -> dict:
        """token -> (acts, piece log-prob so far, node after the token)."""
        moves = {t: (acts, part, child) for t, child in node.items() if t != -1}
        if -1 in node:
            piece, act = node[-1]
            done = (*acts, (piece, act, part))
            if act[0] != "end":
                for t, child in self._trie(done).items():
                    if t != -1:
                        moves.setdefault(t, (done, 0.0, child))
        return moves

    def _meta(self, acts: tuple) -> dict:
        heads = [a for _, a, _ in acts if a[0] == "head"]
        d = g.Decisions(heads[0][1], tuple(h[1] for h in heads[1:]))
        name = ""
        for _, a, _ in acts:
            if a[0] == "head":
                _, name, d.subtype[name], kind, dist = a
                d.condition[name] = g.NONE
                if kind:
                    d.relation[name], d.distance[name] = kind, dist
            elif a[0] == "cond":
                d.condition[name], d.value[name] = a[1], a[2]
        return g.compose(self.cat, d)

    # ------------------------------------------------------------------ search
    def ask(self, text: str) -> dict:
        t0 = time.perf_counter()
        self.text = text = g.normalize(text)
        self.found = g.spans(text)
        self.dists = list(dict.fromkeys(s.label for s in self.found if s.kind == "dist"))
        self._tries, self._cond_cache = {}, {}
        prompt = g.prompt(
            text, g.hints(self.cat, self.values, self.found), g.column_hints(self.cat, text)
        )
        logits = self.lm.start(lm.encode(self.lm.tok, prompt))
        beams = [((), self._trie(()), 0.0, 0.0)]  # (acts, node, score, current piece log-prob)
        finished = []
        while beams:
            cands = []
            for row, (acts, node, score, part) in enumerate(beams):
                moves = self._moves(acts, node, part)
                toks = list(moves)
                x = logits[row, toks]
                lp = x - (x.max() + math.log(np.exp(x - x.max()).sum()))
                cands += [(score + float(p), row, t, moves[t]) for t, p in zip(toks, lp)]
            cands.sort(key=lambda c: -c[0])
            nxt, parents, tokens = [], [], []
            for score, row, t, (acts, part, child) in cands:
                if len(nxt) == self.beam:
                    break
                part += score - beams[row][2]
                if -1 in child and child[-1][1][0] == "end":
                    done = (*acts, (child[-1][0], child[-1][1], part))
                    finished.append((score, done))
                    continue
                nxt.append((acts, child, score, part))
                parents.append(row)
                tokens.append(t)
            best = max((f[0] for f in finished), default=-math.inf)
            if not nxt or best >= nxt[0][2]:
                break
            logits = self.lm.step(parents, tokens)
            beams = nxt
        finished.sort(key=lambda f: -f[0])
        score, acts = finished[0]
        meta, seen, alts = self._meta(acts), [], []
        for s, a in finished[1:]:
            m = self._meta(a)
            if len(alts) < 4 and not any(g.same(m, x) for x in [meta, *seen]):
                seen.append(m)
                alts.append({"meta": m, "confidence": round(math.exp(s), 4)})
        confidence = math.exp(score)
        return {
            "text": text,
            "meta": meta,
            "status": "accepted" if confidence >= self.threshold else "abstain",
            "confidence": round(confidence, 4),
            "decisions": [
                {"piece": p.strip(" ;"), "probability": round(math.exp(lp), 4)}
                for p, a, lp in acts
                if a[0] != "end"
            ],  # fmt: skip
            "alternatives": alts,
            "seconds": round(time.perf_counter() - t0, 3),
        }
