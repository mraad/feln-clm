import json

import pytest

from feln_clm import okf
from feln_clm.rerank import group_key, partition, prepare


def test_grouping_keeps_equivalent_relation_orders_together():
    first = {
        "layers": ["Wells", "Pipelines", "Discoveries"],
        "where": ["", "", ""],
        "relations": ["withinDistance 5 kilometers", "intersects"],
    }
    reordered = {
        "layers": ["Wells", "Discoveries", "Pipelines"],
        "where": ["", "", ""],
        "relations": ["intersects", "withinDistance 5000 meters"],
    }
    assert group_key(first) == group_key(reordered)
    assert partition(group_key(first)) == partition(group_key(reordered))


def test_pool_rejects_dev_and_groups_paraphrases(tmp_path):
    cat = okf.Catalog(
        {
            "Wells": okf.Layer(
                "Wells",
                "",
                "content_type",
                {
                    "content_type": okf.Column(
                        "content_type",
                        "content type",
                        "SmallInteger",
                        domain={"2": "GAS"},
                        kind="subtype",
                    ),
                },
            )
        },
        "test",
    )
    gold = {"layers": ["Wells"], "where": ["content_type = cast(2 as SMALLINT)"], "relations": []}
    row = {
        "usage": "reranker_pool",
        "split": "fold0",
        "generator_holdout": "fold0",
        "generator_seed": 0,
        "id": 1,
        "text": "Find gas wells.",
        "tags": [],
        "gold": gold,
        "gold_rank": 1,
        "candidates": [{"meta": gold, "pieces": ["Wells [gas]"], "log_score": -0.1}],
    }
    pool = tmp_path / "pool.jsonl"
    pool.write_text("\n".join(json.dumps(r) for r in [row, {**row, "text": "List gas wells."}]))
    texts, selected, excluded = prepare(cat, [pool])
    assert len(selected) == 2 and len(texts) == 3
    assert len({r["partition"] for r in selected}) == 1
    assert not excluded
    _, blocked, counts = prepare(cat, [pool], {group_key(gold)})
    assert not blocked
    assert counts == {"generator_training_or_dev_gold_group": 2}
    for changed in ({"split": "dev"}, {"usage": "evaluation_only"}, {"generator_holdout": "fold1"}):
        pool.write_text(json.dumps({**row, **changed}))
        with pytest.raises(ValueError, match="held-out-fold"):
            prepare(cat, [pool])
