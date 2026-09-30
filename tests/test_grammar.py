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


@needs_data
def test_comparatives_hint_the_column_and_rewrite_training_text():
    """OKF 'Comparatives: deeper = greater': the prompt names column and operator, and a
    training request on water_depth > x is also written as 'deeper than x'."""
    from feln_clm.prepare import variants

    cat = okf.load(DATA / "okf")
    assert (
        g.column_hints(cat, "wells deeper than 120") == "'deeper than' = Wells well water depth >"
    )
    assert (
        g.column_hints(cat, "wells no deeper than 9")
        == "'no deeper than' = Wells well water depth <="
    )
    assert (
        g.column_hints(cat, "at least as deep as 9")
        == "'at least as deep as' = Wells well water depth >="
    )
    text = "Show me gas wells with a well water depth greater than 111.0, within 5 km of oil."
    d = g.decompose(cat, text, {"layers": ["Wells"], "where": [
        "content_type = cast(2 as SMALLINT) and (water_depth > cast(111.0 as DOUBLE PRECISION))"],
        "relations": []})  # fmt: skip
    assert "Show me gas wells deeper than 111.0, within 5 km of oil." in variants(cat, text, d)
    assert not any("shallower" in v for v in variants(cat, text, d))
    quoted = "Show gas wells named 'depth over 5' with a well water depth greater than 111.0."
    d = g.decompose(cat, quoted, {"layers": ["Wells"], "where": [
        "content_type = cast(2 as SMALLINT) and (water_depth > cast(111.0 as DOUBLE PRECISION))"],
        "relations": []})  # fmt: skip
    assert all("'depth over 5'" in v for v in variants(cat, quoted, d))  # literals untouched
    assert "Show gas wells named 'depth over 5' deeper than 111.0." in variants(cat, quoted, d)


@needs_data
def test_at_least_and_at_most_comparatives_rewrite_their_own_rows():
    from feln_clm.prepare import variants

    cat = okf.load(DATA / "okf")
    for where, said, want in [
        ("water_depth >= cast(70.0 as DOUBLE PRECISION)", "where the well water depth is at least 70.0",
         {"Find gas wells at least as deep as 70.0.", "Find gas wells no shallower than 70.0."}),
        ("water_depth <= cast(72.0 as DOUBLE PRECISION)", "where well water depth is no more than 72.0",
         {"Find gas wells at most as deep as 72.0.", "Find gas wells no deeper than 72.0."}),
    ]:  # fmt: skip
        text = f"Find gas wells {said}."
        d = g.decompose(cat, text, {"layers": ["Wells"], "where": [
            f"content_type = cast(2 as SMALLINT) and ({where})"], "relations": []})  # fmt: skip
        got = variants(cat, text, d)
        assert want <= set(got), got
        assert not any("deeper than" in v and "no deeper" not in v for v in got), got  # not > / <


def test_catalog_sample_grounding_preserves_literals_and_subtypes():
    phase = okf.Column(
        "current_phase",
        "current phase",
        "String",
        samples=["IN SERVICE", "ABANDONED IN PLACE"],
        kind="upper",
    )
    purpose = okf.Column("purpose", "purpose", "String", samples=["INJECTION"], kind="upper")
    cat = okf.Catalog(
        {
            "Pipelines": okf.Layer(
                "Pipelines",
                "",
                "",
                {
                    "current_phase": phase,
                    "purpose": purpose,
                },
            )
        },
        "test",
    )
    for phrase in ["in-service", "in‐service", "in‑service", "in service", "IN  SERVICE"]:
        text = f"Find injection pipelines {phrase}."
        assert g.ground_samples(cat, text) == "Find injection pipelines 'IN SERVICE'."
        assert g.sample_spans(text, phase)[0].text == "IN SERVICE"
    text = "Find 'in-service' pipelines named \"abandoned in place\"."
    assert g.ground_samples(cat, text) == text
    assert g.ground_samples(cat, "in 'some other' service") == "in 'some other' service"
    assert g.ground_samples(cat, "not in-serviceable") == "not in-serviceable"
    assert g.ground_samples(cat, "not in-service") == "not 'IN SERVICE'"
    assert g.ground_samples(cat, "abandoned-in-place") == "'ABANDONED IN PLACE'"


@needs_data
def test_water_depth_near_in_service_pipeline():
    """The complete request must be reachable, with phase rather than pipeline subtype."""
    from feln_clm.decode import Translator

    text = "Show all wells with depth > 350 m and within 5 km of an in-service pipeline"
    expected = {
        "layers": ["Wells", "Pipelines"],
        "where": ["water_depth > cast(350 as DOUBLE PRECISION)", "current_phase = 'IN SERVICE'"],
        "relations": ["withinDistance 5 kilometers"],
    }
    cat = okf.load(DATA / "okf")
    grounded = g.ground_samples(cat, text)
    assert "'IN SERVICE'" in grounded
    assert g.column_hints(cat, grounded) == "'depth' = Wells well water depth"
    d = g.decompose(cat, grounded, expected)
    tr = object.__new__(Translator)
    tr.cat, tr.values, tr.beam, tr.threshold, tr._ids = cat, {}, 4, 0.5, {}
    tr.lm = Oracle(list("".join(g.pieces(cat, d)).encode()))
    result = tr.ask(text)
    assert result["text"] == text
    assert g.same(result["meta"], expected)


@needs_data
def test_local_model_water_depth_near_in_service_pipeline():
    """Opt-in real-model regression: FELN_CLM_TEST_MODEL=models/q4b pytest -k local_model."""
    import os

    from feln_clm.decode import Translator

    model = os.environ.get("FELN_CLM_TEST_MODEL")
    if not model:
        pytest.skip("set FELN_CLM_TEST_MODEL to run the local GPU model")
    tr = Translator(str(DATA / "okf"), model)
    for phase in ("in-service", "in service", "in‑service"):
        result = tr.ask(f"Show all wells with depth > 350 m and within 5 km of an {phase} pipeline")
        assert g.same(
            result["meta"],
            {
                "layers": ["Wells", "Pipelines"],
                "where": [
                    "water_depth > cast(350 as DOUBLE PRECISION)",
                    "current_phase = 'IN SERVICE'",
                ],
                "relations": ["withinDistance 5 kilometers"],
            },
        )
