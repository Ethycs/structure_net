from copy import deepcopy

import pytest

from experiments.structure_net import ncm_dense_source_overlap_audit as run
from experiments.structure_net.ncm_memory_data import text_hash


def fixture():
    def row(identity, sid, text):
        return {"id": identity, "source_id": sid, "group_id": sid,
            "source_sha256": text_hash(text), "source": text, "spans": []}
    train = [{**row("t", "sharegpt_x_4", "\nA complete repeated passage.\n"), "target_text": "A summary."}]
    dev = {"long": [row("d", "sharegpt_x_2", "A complete repeated passage.")]}
    outputs = [{**dev["long"][0], "text": "A summary."}]
    scores = {"long": [{"id": "d", "text": "A summary.", "valid": True, "passed": True}]}
    return train, dev, outputs, scores


def test_audit_keeps_the_original_pass_and_denominator_while_exposing_overlap():
    result = run.analyze(*fixture())
    cohort = result["cohorts"]["long"]
    assert cohort["cases"] == cohort["valid"] == cohort["raw_semantic_passes"] == 1
    assert cohort["cases_with_source_overlap"] == 1
    assert cohort["passing_cases_with_source_overlap"] == ["d"]
    assert cohort["passing_exact_TRAIN_target_outputs"] == ["d"]
    assert result["TRAIN_development_conflicts"][0]["normalized_text_equal"]


def test_source_and_judged_output_binding_is_independent_of_pass_outcome():
    args = fixture()
    args[3]["long"][0]["passed"] = False
    result = run.analyze(*args)
    assert result["cohorts"]["long"]["cases_with_source_overlap"] == 1
    assert result["cohorts"]["long"]["raw_semantic_passes"] == 0
    changed = deepcopy(args)
    changed[2][0]["source"] = "other source"
    with pytest.raises(ValueError, match="prediction/source"):
        run.analyze(*changed)
    args[3]["long"][0]["text"] = "other prediction"
    with pytest.raises(ValueError, match="judged prediction"):
        run.analyze(*args)


def test_source_content_cannot_be_changed_without_rebinding_actual_hash():
    args = fixture()
    args[0][0]["source"] += "changed"
    with pytest.raises(ValueError, match="source bytes"):
        run.analyze(*args)
