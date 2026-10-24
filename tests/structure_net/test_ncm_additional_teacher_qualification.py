from copy import deepcopy

import pytest

from experiments.structure_net import ncm_additional_teacher_qualification as run


def labels():
    result = []
    for band, n in run.teacher.QUOTAS.items():
        for i in range(n):
            key = f"{band}-{i}"
            result.append({"id": key, "source_id": key, "group_id": key, "source_component": key,
                "source_sha256": key, "source": "source:" + key, "spans": [], "bin": band,
                "target_text": "original first:" + key, "source_cl100k_tokens": 200 + band * 400,
                "teacher_valid": True, "capacity_fit": True})
    return result


def test_four_validity_probes_do_not_change_or_filter_the_fixed64_candidates():
    source = labels()
    result = run.ordered_cases(source)
    assert len(result) == 64 and {r["id"] for r in result} == {r["id"] for r in source}
    assert [r["bin"] for r in result[:4]] == [0, 0, 1, 4]
    expected = {r["id"]: r for r in source}
    assert all(r["texts"]["labels"] == expected[r["id"]]["target_text"] and r["source"] == expected[r["id"]]["source"] for r in result)
    bad = deepcopy(source)
    bad[-1]["capacity_fit"] = False
    with pytest.raises(ValueError, match="all64"):
        run.ordered_cases(bad)


def test_teacher_semantic_metric_does_not_become_student_accuracy():
    score = {"id": "a", "text": "supported paraphrase", "tokens": 3, "valid": True, "passed": True,
        "evaluations": [{"faithful": True, "salient": True, "readable": True}]}
    value = run.metric([score])
    assert value["teacher_label_semantic_pass_rate"] == 1.
    assert value["task_accuracy"] is None and value["teacher_label_quality"]
    assert not value["oracle_target_codes"] and not value["source_prediction"]
    assert "oracle_semantic_pass_rate" not in value


def test_same_case_gate_rejects_two_different_61_of_64_marginals():
    a = [{"id": str(i), "valid": True, "passed": i >= 3} for i in range(64)]
    b = [{"id": str(i), "valid": True, "passed": i not in (3, 4, 5)} for i in range(64)]
    assert run.parent.joint_gate(a, a)["qualified"]
    assert run.parent.joint_gate(a, b)["joint_passed"] == 58
    assert not run.parent.joint_gate(a, b)["qualified"]
