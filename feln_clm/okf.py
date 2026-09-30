"""Read an OKF folder (index.md + one markdown file per layer) into a typed catalog.

Only what text→FELN needs is kept: layer title/description, the subtype column and its
codes, and per column its alias, type, domain, sample values and SQL hints. Each column
gets a ``kind`` that fixes which conditions it can take and how they render as SQL.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

COMPARE = {"greater": "gt", "less": "lt"}  # 'Comparatives:' hint word -> option op
CASTS = {"SmallInteger": "SMALLINT", "Integer": "INTEGER", "Double": "DOUBLE PRECISION"}


@dataclass
class Column:
    name: str
    alias: str
    type: str
    samples: list[str] = field(default_factory=list)
    domain: dict[str, str] = field(default_factory=dict)  # code -> label
    hints: list[str] = field(default_factory=list)
    kind: str = ""  # subtype | code | yesno | flag | like | upper | text | number | date

    @property
    def cast(self) -> str:
        m = next(
            (re.search(r"CAST AS (\w+(?: PRECISION)?)", h) for h in self.hints if "CAST AS" in h),
            None,
        )
        return m.group(1) if m else CASTS.get(self.type, "")

    @property
    def synonyms(self) -> list[str]:
        """Other names requests use for the column: a hint 'Also called: a, b'."""
        return [w.strip() for h in self.hints if h.startswith("Also called:")
                for w in h.removeprefix("Also called:").split(",") if w.strip()]  # fmt: skip

    @property
    def comparatives(self) -> dict[str, str]:
        """Words that compare the column: a hint 'Comparatives: deeper = greater, shallower =
        less' gives {"deeper": "gt", "shallower": "lt"}."""
        pairs = [p.partition("=") for h in self.hints if h.startswith("Comparatives:")
                 for p in h.removeprefix("Comparatives:").split(",")]  # fmt: skip
        return {w.strip(): COMPARE[op.strip()] for w, _, op in pairs}


@dataclass
class Layer:
    name: str
    description: str
    subtype: str
    columns: dict[str, Column]
    resource: str = ""  # file:///…/Project.gdb/FeatureClass
    table: str = ""  # SQL table name

    @property
    def noun(self) -> str:
        return self.name.lower()


@dataclass
class Catalog:
    layers: dict[str, Layer]
    sha: str

    def __getitem__(self, name: str) -> Layer:
        return self.layers[name]


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _unquote(cell: str) -> str:
    return cell[1:-1] if cell.startswith("`") and cell.endswith("`") else cell


def _sections(body: str, level: str) -> dict[str, str]:
    """Split markdown on headings of exactly ``level`` (e.g. '# ' or '## ')."""
    parts = re.split(rf"^{level}(.+)$", body, flags=re.M)
    return {parts[i].strip().strip("`"): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _kind(col: Column, subtype: str) -> str:
    rules = " ".join(col.hints)
    samples = set(col.samples)
    if col.name == subtype:
        return "subtype"
    if col.type == "Date":
        return "date"
    if col.type == "Double":
        return "number"
    if col.type in ("SmallInteger", "Integer"):
        return "flag" if samples <= {"0", "1"} else "number"
    if col.domain:
        return "code"
    if "SQL LIKE" in rules and "NEVER use SQL LIKE" not in rules:
        return "like"
    if samples and samples <= {"YES", "NO"}:
        return "yesno"
    return "upper" if "uppercase" in rules else "text"


def parse_layer(text: str) -> Layer:
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        raise ValueError("OKF layer file has no front matter")
    front, body = m.groups()
    meta = dict(re.findall(r'^(\w+): "?(.*?)"?$', front, re.M))
    top = _sections(body, "# ")
    columns: dict[str, Column] = {}
    for line in top.get("Schema", "").splitlines():
        cells = _cells(line)
        if len(cells) < 4 or not cells[0].startswith("`"):
            continue
        samples = re.findall(r"`([^`]*)`", cells[3])
        columns[_unquote(cells[0])] = Column(_unquote(cells[0]), cells[1], cells[2], samples)
    for name, sec in _sections(top.get("Domains", ""), "## ").items():
        for line in sec.splitlines():
            cells = _cells(line)
            if len(cells) == 2 and cells[0].startswith("`"):
                columns[name].domain[_unquote(cells[0])] = cells[1]
    for name, sec in _sections(top.get("Query hints", ""), "## ").items():
        columns[name].hints = [h[2:].strip() for h in sec.splitlines() if h.startswith("- ")]
    subtype = meta.get("subtype", "")
    for col in columns.values():
        col.kind = _kind(col, subtype)
    return Layer(meta["title"], meta.get("description", ""), subtype, columns,
                 meta.get("resource", ""), meta.get("table_name", meta["title"]))  # fmt: skip


def load(folder: str | Path) -> Catalog:
    """Layers in the order index.md lists them."""
    folder = Path(folder)
    index = (folder / "index.md").read_text()
    files = re.findall(r"\]\(([^)]+\.md)\)", index)
    texts = [(folder / f).read_text() for f in files]
    sha = hashlib.sha256("\n".join([index, *texts]).encode()).hexdigest()[:16]
    layers = [parse_layer(t) for t in texts]
    return Catalog({ly.name: ly for ly in layers}, sha)
