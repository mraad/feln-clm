import json
from pathlib import Path

import numpy as np
import pytest

from feln_clm import grammar as g
from feln_clm import okf

DATA = Path(__file__).resolve().parent.parent / "data"
needs_data = pytest.mark.skipif(not (DATA / "okf").exists(), reason="data/ snapshot not present")


def test_spans_quotes_contractions_and_units():
    text = "Get wells that aren't from 'NL Oil and GAS Portal', they're within 1,000.5 meters of 'X', depth 68.0."
    found = g.spans(text)
    assert [(s.kind, s.label) for s in found] == [
        ("str", "NL Oil and GAS Portal"), ("dist", "1000.5 meters"), ("str", "X"), ("num", "68.0"),
    ]  # fmt: skip


def test_value_candidates_include_measured_distances():
    opt = g.Option("water_depth|gt", "water_depth", "gt", "num", "")
    assert g.values(opt, None, g.spans("depth over 68.0 meters, 5 km away"), "") == [
        ("68.0",),
        ("5",),
    ]


class Bytes:
    """A byte tokenizer, so piece token ids are just their UTF-8 bytes."""

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class Oracle:
    """An LM that strongly prefers the next token of one target sequence."""

    tok = Bytes()

    def __init__(self, target: list[int]):
        self.target = target

    def _logits(self):
        out = np.zeros((len(self.hist), 256), dtype=np.float32)
        for i, h in enumerate(self.hist):
            if h == self.target[: len(h)] and len(h) < len(self.target):
                out[i, self.target[len(h)]] = 30.0
        return out

    def start(self, ids):
        self.hist = [[]]
        return self._logits()

    def step(self, parents, tokens):
        self.hist = [self.hist[p] + [t] for p, t in zip(parents, tokens)]
        return self._logits()


@needs_data
def test_oracle_every_example_decodes_to_its_gold():
    """decompose -> pieces -> constrained beam search -> compose reproduces FELN.json gold, so
    every gold piece is a legal candidate of its slot and the decoder state machine is right."""
    from feln_clm.decode import Translator

    cat = okf.load(DATA / "okf")
    tr = object.__new__(Translator)
    tr.cat, tr.values, tr.beam, tr.threshold, tr._ids = cat, {}, 4, 0.5, {}
    misses = []
    for x in json.load(open(DATA / "FELN.json")):
        text = g.normalize(x["text"])
        d = g.decompose(cat, text, x["meta"])
        assert g.same(g.compose(cat, d), x["meta"]), text
        tr.lm = Oracle(list("".join(g.pieces(cat, d)).encode()))
        if not g.same(tr.ask(text)["meta"], x["meta"]):
            misses.append(text)
    assert not misses, misses[:5]
