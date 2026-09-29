"""FELN as typed CLM choice questions over an OKF catalog.

A request is answered by a handful of choice questions (CLM wire format) whose options are
catalog text, plus literal spans read from the request itself:

    layers                      primary layer and the set of related layers
    subtype:<L>@<P>             subtype code of layer L among those the request names, or "any"
    relation:<L>@<P>=<code>     spatial relation from P to L, named with its subtype
    distance:<L>@<P>=<code>     which distance span of the request applies to L
    column:<L>@<P>              attribute of L with an extra condition, or "none"
    op:<L>@<P>|<column>         condition on that column (column|op, values folded in for codes)
    value:<L>@<P>|<column|op>   which literal span(s) fill that condition

``decompose`` turns a gold FELN into these decisions (for training and the oracle) by
searching the same space ``render`` produces, so both directions agree by construction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

from feln import FELN, parse_relation
from feln.identical import identical
from feln.units import to_meters

from .okf import Catalog, Column, Layer

KINDS = {
    "withinDistance": "within a given distance of them",
    "notWithinDistance": "farther than a given distance from all of them",
    "intersects": "intersecting them",
    "contains": "containing one or more of them",
    "within": "lying within (inside) one or more of them",
}
DISTANCE = ("withinDistance", "notWithinDistance")
UNITS = {
    "kilometers": "kilometers", "kilometer": "kilometers", "kilometres": "kilometers",
    "kilometre": "kilometers", "km": "kilometers", "meters": "meters", "meter": "meters",
    "metres": "meters", "metre": "meters", "m": "meters", "miles": "miles", "mile": "miles",
    "mi": "miles", "feet": "feet", "foot": "feet", "ft": "feet",
}  # fmt: skip
NONE = "none"
ANY = "any"

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_UNIT = "|".join(sorted(UNITS, key=len, reverse=True))
_DIST_RE = re.compile(rf"(?<![\w.])({_NUM})\s*-?\s*({_UNIT})\b", re.I)
_NUM_RE = re.compile(rf"(?<![\w.])-?(?:{_NUM})(?![\w])")
_QUOTE_RE = re.compile(r"(?<!\w)'([^'\n]*?)'(?!\w)|(?<!\w)\"([^\"\n]*?)\"(?!\w)")


# --------------------------------------------------------------------------- spans
@dataclass(frozen=True)
class Span:
    kind: str  # str | num | dist
    text: str  # literal as it will be rendered (quotes stripped, commas removed)
    start: int
    unit: str = ""  # dist only, canonical plural

    @property
    def label(self) -> str:
        return f"{self.text} {self.unit}" if self.kind == "dist" else self.text

    @property
    def year(self) -> bool:
        return self.kind == "num" and re.fullmatch(r"(19|20)\d\d", self.text) is not None


def normalize(text: str) -> str:
    return text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')


def spans(text: str) -> list[Span]:
    """Quoted strings, distances (number + unit) and bare numbers, in text order."""
    text = normalize(text)
    out: list[Span] = []
    masked = list(text)
    for m in _QUOTE_RE.finditer(text):
        g = 1 if m.group(1) is not None else 2
        out.append(Span("str", m.group(g), m.start(g)))
        masked[m.start() : m.end()] = " " * (m.end() - m.start())
    rest = "".join(masked)
    for m in _DIST_RE.finditer(rest):
        out.append(Span("dist", m.group(1).replace(",", ""), m.start(), UNITS[m.group(2).lower()]))
        masked[m.start() : m.end()] = " " * (m.end() - m.start())
    rest = "".join(masked)
    for m in _NUM_RE.finditer(rest):
        out.append(Span("num", m.group(0).replace(",", ""), m.start()))
    return sorted(out, key=lambda s: s.start)


def sample_spans(text: str, col: Column) -> list[Span]:
    """Unquoted mentions of the column's catalog sample values (case-insensitive)."""
    low = normalize(text).lower()
    found = []
    for v in col.samples:
        if len(v) < 2:
            continue
        m = re.search(rf"(?<!\w){re.escape(v.lower())}(?!\w)", low)
        if m:
            found.append(Span("str", v, m.start()))
    return found


# --------------------------------------------------------------------------- options
@dataclass(frozen=True)
class Option:
    key: str  # column|op, or "none"
    col: str
    op: str
    needs: str  # "", str, str2, num, num2, year, year2
    text: str  # description the action head sees


def _options(col: Column) -> list[Option]:
    a, c, k = col.alias, col.name, col.kind

    def o(op, needs, text):
        return Option(f"{c}|{op}", c, op, needs, text)

    blank = [o("blank", "", f"{a} is blank"), o("notblank", "", f"{a} is not blank")]
    if k == "like":
        return [
            o("contains", "str", f"{a} contains a given text"),
            o("starts", "str", f"{a} starts with a given text"),
            o("ends", "str", f"{a} ends with a given text"),
            *blank,
        ]
    if k in ("upper", "text"):
        return [
            o("is", "str", f"{a} is a given value"),
            o("isnot", "str", f"{a} is not a given value"),
            o("either", "str2", f"{a} is one of two given values"),
            *blank,
        ]
    if k == "yesno":
        return [o(f"{op}:{v}", "", f"{a} is {'not ' if op == 'isnot' else ''}{v.lower()}")
                for op in ("is", "isnot") for v in ("YES", "NO")] + blank  # fmt: skip
    if k == "code":
        labels = {**{s: s for s in col.samples}, **col.domain}
        return [o(f"{op}:{code}", "", f"{a} is {'not ' if op == 'isnot' else ''}{label}")
                for op in ("is", "isnot") for code, label in labels.items()]  # fmt: skip
    if k == "flag":
        return [o("is:1", "", f"{a} is yes (1)"), o("is:0", "", f"{a} is no (0)")]
    if k == "number":
        return [
            o("eq", "num", f"{a} equals a number"),
            o("lt", "num", f"{a} is less than a number"),
            o("gt", "num", f"{a} is greater than a number"),
            o("le", "num", f"{a} is at most a number"),
            o("ge", "num", f"{a} is at least a number"),
            o("between", "num2", f"{a} is between two numbers"),
        ]
    if k == "date":
        return [
            o("after", "year", f"{a} is after a year"),
            o("since", "year", f"{a} is in or after a year"),
            o("before", "year", f"{a} is before a year"),
            o("through", "year", f"{a} is in or before a year"),
            o("in", "year", f"{a} is in a given year"),
            o("between", "year2", f"{a} is between two years"),
        ]
    return []


def options(layer: Layer) -> dict[str, Option]:
    out = {NONE: Option(NONE, "", NONE, "", "no other attribute condition")}
    for col in layer.columns.values():
        out.update((opt.key, opt) for opt in _options(col))
    return out


def values(opt: Option, col: Column | None, found: list[Span], text: str) -> list[tuple[str, ...]]:
    """Literal tuples the request offers for this option, in text order, deduplicated."""
    if not opt.needs:
        return [()]
    if opt.needs.startswith("str"):
        pool = [s for s in found if s.kind == "str" and s.text]
        if col is not None and col.kind != "like":
            seen = {s.text.upper() for s in pool}
            pool += [s for s in sample_spans(text, col) if s.text.upper() not in seen]
            pool.sort(key=lambda s: s.start)
    elif opt.needs.startswith("year"):
        pool = [s for s in found if s.year]
    else:  # a measured value may carry a unit ("68.0 meters"), so distances qualify too
        pool = [s for s in found if s.kind in ("num", "dist")]
    texts = list(dict.fromkeys(s.text for s in pool))
    if opt.needs.endswith("2"):
        return [(a, b) for a, b in combinations(texts, 2)]
    return [(t,) for t in texts]


def _q(v: str) -> str:
    return "'" + v.replace("'", "''") + "'"


def render_condition(opt: Option, col: Column, vals: tuple[str, ...]) -> str:
    c, op = col.name, opt.op
    if col.kind == "upper":
        vals = tuple(v.upper() for v in vals)
    if op in ("blank", "notblank"):
        return f"{c} {'=' if op == 'blank' else '<>'} ''"
    if col.kind == "like":
        v = vals[0]
        pattern = {"contains": f"%{v}%", "starts": f"{v}%", "ends": f"%{v}"}[op]
        return f"{c} LIKE {_q(pattern)}"
    if col.kind in ("upper", "text"):
        if op == "either":
            return f"{c} = {_q(vals[0])} or {c} = {_q(vals[1])}"
        return f"{c} {'=' if op == 'is' else '<>'} {_q(vals[0])}"
    if col.kind in ("yesno", "code"):
        which, code = op.split(":", 1)
        return f"{c} {'=' if which == 'is' else '<>'} {_q(code)}"
    if col.kind == "flag":
        return f"{c} = cast({op.split(':')[1]} as {col.cast})"
    if col.kind == "number":
        if op == "between":
            return f"{c} BETWEEN cast({vals[0]} as {col.cast}) AND cast({vals[1]} as {col.cast})"
        sym = {"eq": "=", "lt": "<", "gt": ">", "le": "<=", "ge": ">="}[op]
        return f"{c} {sym} cast({vals[0]} as {col.cast})"
    if col.kind == "date":
        y = [int(v) for v in vals]
        ts = lambda year: f"timestamp '{year}-01-01'"  # noqa: E731
        lo, hi = {
            "after": (y[0] + 1, None), "since": (y[0], None), "before": (None, y[0]),
            "through": (None, y[0] + 1), "in": (y[0], y[0] + 1),
            "between": (y[0], y[-1] + 1),
        }[op]  # fmt: skip
        parts = ([f"{c} >= {ts(lo)}"] if lo else []) + ([f"{c} < {ts(hi)}"] if hi else [])
        return " AND ".join(parts)
    raise ValueError(f"cannot render {opt.key}")


def render_where(layer: Layer, code: str, opt: Option, vals: tuple[str, ...]) -> str:
    sub = ""
    if code != ANY:
        st = layer.columns[layer.subtype]
        sub = f"{st.name} = cast({code} as {st.cast})"
    cond = "" if opt.key == NONE else render_condition(opt, layer.columns[opt.col], vals)
    if sub and cond:
        return f"{sub} and ({cond})"
    return sub or cond


# --------------------------------------------------------------------------- questions
def structures(cat: Catalog) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Every (primary, related layers) choice, keyed 'P>A,B'."""
    names = list(cat.layers)
    out = {}
    for p in names:
        rest = [n for n in names if n != p]
        for r in range(len(rest) + 1):
            for secs in combinations(rest, r):
                out[f"{p}>{','.join(secs)}"] = (p, secs)
    return out


def _structure_text(cat: Catalog, p: str, secs: tuple[str, ...]) -> str:
    if not secs:
        return f"Return {cat[p].noun}, with no spatial relation to other layers."
    return f"Return {cat[p].noun} that are spatially related to {' and '.join(cat[s].noun for s in secs)}."


def _choice(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def _role(cat: Catalog, layer: str, primary: str) -> str:
    if layer == primary:
        return f"the returned {cat[layer].noun}"
    return f"the {cat[layer].noun} that the returned {cat[primary].noun} are related to"


def q_layers(cat: Catalog) -> dict:
    return _choice(
        "Which features does the request return, and which other layers must they be spatially related to?",
        {k: _structure_text(cat, p, s) for k, (p, s) in structures(cat).items()},
    )


def subtype_label(cat: Catalog, layer: str, code: str) -> str:
    ly = cat[layer]
    return "" if code == ANY else ly.columns[ly.subtype].domain[code].lower()


def mentions(cat: Catalog, text: str) -> dict[str, int]:
    """How often each subtype label (any layer) is named in the request, longest match first."""
    labels = {
        lab.lower() for ly in cat.layers.values() for lab in ly.columns[ly.subtype].domain.values()
    }
    alts = [
        re.escape(lab).replace("shows", "shows?") for lab in sorted(labels, key=len, reverse=True)
    ]
    counts: dict[str, int] = {}
    for m in re.finditer(rf"(?<![\w/])({'|'.join(alts)})(?![\w/])", normalize(text).lower()):
        lab = m.group(1) if m.group(1) in labels else m.group(1) + "s"
        counts[lab] = counts.get(lab, 0) + 1
    return counts


def subtype_codes(cat: Catalog, layer: str, named: dict[str, int]) -> list[str]:
    """The layer's subtype codes the request names, then "any"."""
    ly = cat[layer]
    return [c for c, lab in ly.columns[ly.subtype].domain.items() if lab.lower() in named] + [ANY]


def q_subtype(cat: Catalog, layer: str, primary: str, codes: list[str]) -> dict:
    ly = cat[layer]
    st = ly.columns[ly.subtype]

    def text(c: str) -> str:
        return (
            f"{ly.noun} of any {st.alias}"
            if c == ANY
            else f"{ly.noun} with {st.alias} {st.domain[c]}"
        )

    q = _choice(
        f"Which {st.alias} must {_role(cat, layer, primary)} have?", {c: text(c) for c in codes}
    )
    q["listing"] = [text(c) for c in [*st.domain, ANY]]  # the full domain, so a prefix never varies
    return q


def _named(cat: Catalog, layer: str, code: str) -> str:
    """'the oil pipelines' / 'the pipelines': a related layer as the request names it."""
    lab = subtype_label(cat, layer, code)
    return f"the {lab + ' ' if lab else ''}{cat[layer].noun}"


def q_relation(cat: Catalog, layer: str, primary: str, code: str) -> dict:
    return _choice(
        f"How must the returned {cat[primary].noun} be located relative to {_named(cat, layer, code)}?",
        dict(KINDS),
    )


def q_distance(cat: Catalog, layer: str, primary: str, code: str, found: list[Span]) -> dict:
    labels = list(dict.fromkeys(s.label for s in found if s.kind == "dist"))
    return _choice(
        f"Which distance applies between the returned {cat[primary].noun} and {_named(cat, layer, code)}?",
        {lab: lab for lab in labels},
    )


def q_column(cat: Catalog, layer: str, primary: str) -> dict:
    ly = cat[layer]
    crit = {NONE: "no other attribute condition"}
    for col in ly.columns.values():
        if _options(col):
            crit[col.name] = (
                f"{col.alias} ({col.about.rstrip('.').lower()})" if col.about else col.alias
            )
    alias = ly.columns[ly.subtype].alias
    return _choice(
        f"Besides its {alias}, which attribute of {_role(cat, layer, primary)} does the request constrain?",
        crit,
    )


def q_op(cat: Catalog, layer: str, primary: str, column: str) -> dict:
    col = cat[layer].columns[column]
    return _choice(
        f"Which condition on the {col.alias} must {_role(cat, layer, primary)} meet?",
        {o.key: o.text for o in _options(col)},
    )


def value_label(vals: tuple[str, ...]) -> str:
    return " and ".join(vals)


def q_value(
    cat: Catalog, layer: str, primary: str, opt: Option, cands: list[tuple[str, ...]]
) -> dict:
    role = _role(cat, layer, primary)
    return _choice(
        f"Condition on {role}: {opt.text}. Which value from the request is it?",
        {value_label(v): value_label(v) for v in cands},
    )


# --------------------------------------------------------------------------- framing
FORMATS = ("raw", "options", "ids", "prefix", "qprefix")
PREFIX_END = "\n\nRequest: "  # where a "prefix" turn's request-independent part ends
IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_ASSISTANT = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:"


def frame(fmt: str, text: str, qs: dict[str, dict]) -> tuple[str, dict[str, dict]]:
    """(state, questions) as the encoders see them; CLM joins them as state + blank line + question.

    ``options`` lists the candidates in a Qwen3 chat turn (thinking off) ending in "Answer:",
    so the pooled last token is the one that predicts the answer (CLM strips trailing
    whitespace, so the turn must end on a visible token). ``ids`` labels them with single-token letters and the
    candidates become those letters: the pooled state is the model's multiple-choice answer.
    ``prefix`` puts the option list first (subtypes list their full domain), then the request
    and question, so everything before ``PREFIX_END`` can be encoded once and reused;
    ``qprefix`` also states the question before the options (and again after the request).
    """
    if fmt == "raw":
        return text, qs
    if fmt in ("prefix", "qprefix"):  # request-free head first, so its KV state caches
        out = {}
        for qid, q in qs.items():
            listing = "\n".join(f"- {t}" for t in q.get("listing") or q["criteria"].values())
            ask = f"{q['instructions']}\n" if fmt == "qprefix" else ""
            ins = (f"<|im_start|>user\n{ask}Options:\n{listing}{PREFIX_END}{text}\n\n"
                   f"{q['instructions']}\nAnswer with one option.{_ASSISTANT}")  # fmt: skip
            out[qid] = {**q, "instructions": ins}
        return "", out
    out = {}
    for qid, q in qs.items():
        texts = list(q["criteria"].values())
        if fmt == "ids":
            if len(texts) > len(IDS):
                raise ValueError(f"{qid}: {len(texts)} options exceed {len(IDS)} letters")
            listing = "\n".join(f"{IDS[i]}. {t}" for i, t in enumerate(texts))
            ins = f"{q['instructions']}\nOptions:\n{listing}\nAnswer with the letter of the correct option."
            crit = {k: IDS[i] for i, k in enumerate(q["criteria"])}
        else:
            listing = "\n".join(f"- {t}" for t in texts)
            ins = f"{q['instructions']}\nOptions:\n{listing}\nAnswer with one option."
            crit = q["criteria"]
        out[qid] = {**q, "instructions": ins + _ASSISTANT, "criteria": crit}
    return f"<|im_start|>user\nRequest: {text}", out


# --------------------------------------------------------------------------- decisions
@dataclass
class Decisions:
    primary: str
    secondaries: tuple[str, ...]
    subtype: dict[str, str] = field(default_factory=dict)
    condition: dict[str, str] = field(default_factory=dict)
    value: dict[str, tuple[str, ...]] = field(default_factory=dict)
    relation: dict[str, str] = field(default_factory=dict)
    distance: dict[str, str] = field(default_factory=dict)  # span label, e.g. "25 kilometers"

    @property
    def structure(self) -> str:
        return f"{self.primary}>{','.join(self.secondaries)}"

    @property
    def layers(self) -> list[str]:
        return [self.primary, *self.secondaries]


def compose(cat: Catalog, d: Decisions) -> dict:
    """Decisions -> FELN meta (layers / where / relations)."""
    where = []
    for name in d.layers:
        ly = cat[name]
        opt = options(ly)[d.condition.get(name, NONE)]
        where.append(render_where(ly, d.subtype.get(name, ANY), opt, d.value.get(name, ())))
    rels = []
    for name in d.secondaries:
        kind = d.relation[name]
        rels.append(f"{kind} {d.distance[name]}" if kind in DISTANCE else kind)
    return {"layers": d.layers, "where": where, "relations": rels}


def _same_cond(a: str, b: str) -> bool:
    a, b = a.replace(" ILIKE ", " LIKE "), b.replace(" ILIKE ", " LIKE ")
    return a == b or identical(a, b)


def _split_where(ly: Layer, where: str) -> tuple[str, str]:
    st = ly.columns[ly.subtype]
    m = re.match(rf"^{re.escape(st.name)} = cast\((-?\d+) as [\w ]+?\)(.*)$", where.strip(), re.I)
    if not m:
        return ANY, where.strip()
    rest = m.group(2).strip()
    if rest.lower().startswith("and (") and rest.endswith(")"):
        return m.group(1), rest[5:-1]
    if rest:
        raise ValueError(f"unexpected where tail: {rest!r}")
    return m.group(1), ""


def decompose(cat: Catalog, text: str, meta: dict) -> Decisions:
    """Gold FELN -> decisions; raises ValueError when the grammar cannot reproduce it."""
    names = meta["layers"]
    order = list(cat.layers)
    primary, secs = names[0], tuple(sorted(names[1:], key=order.index))
    d = Decisions(primary, secs)
    found = spans(text)
    for name, where in zip(names, meta["where"]):
        ly = cat[name]
        code, cond = _split_where(ly, where)
        d.subtype[name] = code
        if not cond:
            d.condition[name] = NONE
            continue
        m = re.match(r"\(?\s*(\w+)", cond)
        ident = m.group(1).lower() if m else ""
        hit = None
        for opt in options(ly).values():
            if opt.col.lower() != ident:
                continue
            col = ly.columns[opt.col]
            hit = next((v for v in values(opt, col, found, text)
                        if _same_cond(render_condition(opt, col, v), cond)), None)  # fmt: skip
            if hit is not None:
                d.condition[name], d.value[name] = opt.key, hit
                break
        if hit is None:
            raise ValueError(f"{name}: no option renders {cond!r}")
    dists = [s for s in found if s.kind == "dist"]
    for name, rel in zip(names[1:], meta["relations"]):
        r = parse_relation(rel)
        d.relation[name] = "within" if r.kind == "inside" else r.kind
        if r.kind in DISTANCE:
            want = to_meters(r.distance, r.unit)
            span = next(
                (s for s in dists if abs(to_meters(float(s.text), s.unit) - want) < 0.01), None
            )
            if span is None:
                raise ValueError(f"{name}: distance {rel!r} not in text")
            d.distance[name] = span.label
    return d


def same(a: dict, b: dict) -> bool:
    return FELN(**a).same(FELN(**b))
