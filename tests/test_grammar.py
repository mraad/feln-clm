import json
from pathlib import Path

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


@needs_data
def test_oracle_roundtrip_every_example():
    """decompose -> compose reproduces every FELN.json query (FELN.same), and questions build."""
    from feln_clm.prepare import questions

    cat = okf.load(DATA / "okf")
    for x in json.load(open(DATA / "FELN.json")):
        d = g.decompose(cat, x["text"], x["meta"])
        assert g.same(g.compose(cat, d), x["meta"]), x["text"]
        qs, _ = questions(cat, x["text"], d)
        g.frame("ids", x["text"], qs)  # every question fits the letter alphabet
        for q in [
            *g.frame("prefix", x["text"], qs)[1].values(),
            *g.frame("qprefix", x["text"], qs)[1].values(),
        ]:  # one cacheable, request-free head
            head, _, tail = q["instructions"].partition(g.PREFIX_END)
            assert tail.startswith(x["text"]) and g.PREFIX_END not in tail


@needs_data
def test_decoder_binds_each_named_subtype_once_and_explains_literals():
    """'oil' is said once: two layers cannot both take it; a quoted literal pulls in its condition."""
    from feln_clm.decode import Translator

    cat = okf.load(DATA / "okf")
    tr = object.__new__(Translator)
    tr.cat, tr.structs, tr.top, tr.penalty = cat, g.structures(cat), 4, 2.0
    tr.labels = {
        lab.lower() for ly in cat.layers.values() for lab in ly.columns[ly.subtype].domain.values()
    }
    text = "Find oil/gas wells within 5 km of oil where the field label is 'VIGDIS'."
    key = "Wells>Pipelines,Discoveries"
    probs = {
        "layers": {key: 1.0},
        "subtype:Wells@Wells": {"5": 0.9, g.ANY: 0.1},
        "subtype:Pipelines@Wells": {"4": 0.6, g.ANY: 0.4},
        "subtype:Discoveries@Wells": {"3": 0.7, "4": 0.2, g.ANY: 0.1},
    }
    codes, _ = tr._subtypes(key, probs, g.mentions(cat, text))
    assert codes == {
        "Wells": "5",
        "Pipelines": g.ANY,
        "Discoveries": "3",
    }  # oil once, to the likelier

    found = g.spans(text)
    for name, col in (("Wells", {g.NONE: 1.0}), ("Pipelines", {g.NONE: 1.0})):
        probs[f"column:{name}@Wells"] = col
    probs["column:Discoveries@Wells"] = {g.NONE: 0.7, "field_label": 0.3}
    probs["op:Discoveries@Wells|field_label"] = {"field_label|is": 1.0}
    for name in ("Pipelines", "Discoveries"):
        probs[f"relation:{name}@Wells={codes[name]}"] = {"withinDistance": 1.0}
    offers = {("Discoveries", "Wells", "field_label|is"): [("VIGDIS",)]}
    meta = tr._decode(key, probs, (codes, 0.0), offers, found)["meta"]
    assert meta["where"][2] == "discovery_type = cast(3 as INTEGER) and (field_label = 'VIGDIS')"
