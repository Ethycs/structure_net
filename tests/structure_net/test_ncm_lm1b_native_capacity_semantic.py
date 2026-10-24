import pytest

from experiments.structure_net import ncm_lm1b_native_capacity_semantic as run


def test_packets_are_single_anonymous_arm_with_full_source():
    rows = [{**{k: f"{k}-{i}" for k in run.IDENTITY}, "target_text": f"REFERENCE-{i}"} for i in range(16)]
    outputs = [{"id": r["id"], "source_sha256": r["source_sha256"], "text": f"MODEL-{i}"} for i, r in enumerate(rows)]
    cases = run.cases_for(rows, outputs)
    packets = run.packets_for(cases)
    for arm, other in (("reference", "MODEL"), ("native", "REFERENCE")):
        for i, packet in enumerate(packets[arm]):
            rendered = str(packet)
            assert rows[i]["source"] in rendered
            assert cases[i]["texts"][arm] in rendered
            assert other not in rendered
    with pytest.raises(ValueError):
        run.cases_for(rows, list(reversed(outputs)))


def test_joint_counts_same_case_intersection_and_requires_case_identity():
    a = [{"id": str(i), "valid": True, "passed": i < 8} for i in range(16)]
    b = [{"id": str(i), "valid": True, "passed": i >= 8} for i in range(16)]
    assert run.joint(a, b) == {"cases": 16, "both_valid": 16, "both_passed": 0}
    with pytest.raises(ValueError):
        run.joint(a, list(reversed(b)))
