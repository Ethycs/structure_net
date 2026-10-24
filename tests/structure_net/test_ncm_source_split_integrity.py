from copy import deepcopy

import pytest

from experiments.structure_net import ncm_source_split_integrity as run


def row(identity, source_id, source):
    return {"id": identity, "source_id": source_id, "group_id": source_id,
        "source_sha256": identity + "-hash", "source": source, "spans": []}


def test_observed_session_variants_join_without_collapsing_ultrachat_ids():
    assert run.source_family("sharegpt_uD7CBng_2") == run.source_family("sharegpt_uD7CBng_4")
    assert run.source_family("b94200c5_2") == run.source_family("b94200c5_9")
    assert run.source_family("answer_ef84b994_abs_2") == run.source_family("answer_ef84b994_1")
    assert run.source_family("ultrachat_55863") != run.source_family("ultrachat_114211")


def test_whitespace_case_and_unicode_variants_cannot_evade_text_guard():
    a = row("a", "unrelated_a", "A café describes the ＦＡＣＴＳ.\n")
    b = row("b", "unrelated_b", " a CAFÉ describes the FACTS ")
    assert run.overlap(a, b)["normalized_text_equal"]
    with pytest.raises(ValueError, match="source family/text/turn overlap"):
        run.require_disjoint([a], [b])


def test_substantial_contained_passages_are_blocked_across_unrelated_ids():
    words = [f"term{i}" for i in range(100)]
    a = row("a", "different_a", " ".join(words))
    b = row("b", "different_b", " ".join(words[20:80]))
    value = run.overlap(a, b)
    assert value["near_text_overlap"] and value["shorter_text_containment"] == 1
    assert not value["normalized_text_equal"]
    b["source"] = " ".join(words[:15] + [f"fresh{i}" for i in range(80)])
    assert run.overlap(a, b) is None


def test_public_turn_provenance_blocks_nonoverlapping_fragments():
    a, b = row("a", "different_a", "first section"), row("b", "different_b", "other section")
    for r in (a, b):
        r["spans"] = [{"source_id": r["source_id"], "turn_text_sha256": "same-full-turn"}]
    assert run.overlap(a, b)["shared_public_turn"]


def test_cross_audit_retains_every_conflict_and_never_filters_by_model_quality():
    a = row("a", "sharegpt_x_0", "alpha")
    b = row("b", "sharegpt_x_1", "beta")
    c = row("c", "sharegpt_y_0", "gamma")
    values = run.cross_overlaps([a], [b, c])
    assert [(r["train_id"], r["heldout_id"]) for r in values] == [("a", "b")]
    run.require_disjoint([a], [c])
    changed = deepcopy(b)
    changed["passed"] = False
    assert run.cross_overlaps([a], [changed]) == values
