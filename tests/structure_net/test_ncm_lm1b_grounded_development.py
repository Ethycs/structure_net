"""Guard the full population, source-only boundary and semantic joint decision."""
from copy import deepcopy

import pytest

from experiments.structure_net import ncm_lm1b_grounded_development as run


def test_complete_source_only_population_and_family_exclusions():
    cases = run.q.read(run.SOURCE / "cases.json")
    assert run.validate_sources(cases) is cases
    train = run.q.read(run.training.DATA / "cases.json")
    assert run.training.disjoint(train, cases)["source_id"]["overlap"] == 0
    bad = deepcopy(cases)
    bad[run.COHORTS[0]][0]["target_text"] = "target leak"
    with pytest.raises(ValueError, match="targets"):
        run.validate_sources(bad)
    bad = deepcopy(cases)
    bad[run.COHORTS[1]].pop()
    with pytest.raises(ValueError, match="complete"):
        run.validate_sources(bad)
    bad = deepcopy(cases)
    bad[run.COHORTS[1]][0]["source_component"] = bad[run.COHORTS[0]][0]["source_component"]
    with pytest.raises(ValueError, match="duplicate"):
        run.validate_sources(bad)


def scores(failing=()):
    return [{"id": str(i), "valid": True, "passed": i not in failing} for i in range(64)]


def test_same_case_joint_gate_inclusive_and_not_marginal():
    a = scores((0, 1, 2))
    assert run.joint_gate(a, a)["qualified"]
    b = scores((3, 4, 5))
    assert sum(r["passed"] for r in a) == sum(r["passed"] for r in b) == 61
    assert run.joint_gate(a, b) == {"cases": 64, "joint_passed": 58, "all_valid": True, "qualified": False}
    bad = deepcopy(a)
    bad[0].update(valid=False, passed=None)
    assert not run.joint_gate(a, bad)["qualified"]
    with pytest.raises(ValueError, match="complete64"):
        run.joint_gate(a[:-1], b[:-1])


def test_both_complete_cohorts_required_before_primary():
    metric = {"cases": 64, "valid": 64, "passed": 61}
    assert run.secondary_admitted({c: metric for c in run.COHORTS})
    assert not run.secondary_admitted({run.COHORTS[0]: metric})
    assert not run.secondary_admitted({run.COHORTS[0]: metric, run.COHORTS[1]: {**metric, "passed": 60}})
    assert not run.secondary_admitted({c: {**metric, "valid": 63} for c in run.COHORTS})


def test_native_renderer_preserves_raw_positions_and_exposes_unmapped_ids():
    tokenizer = run.native.tokenizer()
    words = [101, 30527, 1996, 101] + [0] * 124
    before = list(words)
    assert run.decode(words, tokenizer) == {"text": "the", "predicted_BOS": True,
        "predicted_EOS_position": 3, "missing_EOS": False, "active_unmapped_ids": 1}
    assert words == before
    assert run.decode([0] * 128, tokenizer)["text"] == ""
    assert run.decode([1996] * 128, tokenizer)["missing_EOS"]
    with pytest.raises(ValueError, match="all128"):
        run.decode(words[:-1], tokenizer)
    with pytest.raises(ValueError, match="full-vocabulary"):
        run.decode([30528] * 128, tokenizer)


def test_real_task_metric_keeps_invalids_and_semantic_failures():
    rows = [{"id": "a", "valid": True, "passed": True, "tokens": 3, "text": "a paraphrase",
        "evaluations": [{"faithful": True, "salient": True, "readable": True}]},
        {"id": "b", "valid": False, "passed": None, "tokens": 0, "text": "", "evaluations": None}]
    metric = run.task_metric(rows)
    assert metric["task_accuracy"] is None and metric["cases"] == 2 and metric["passed"] == 1
    assert metric["all_case_pass_lower_bound"] == .5 and metric["all_case_pass_upper_bound"] == 1.
    assert metric["source_prediction"] and not metric["oracle_target_codes"]
    assert "oracle_semantic_pass_rate" not in metric
    assert not run.q.passes({"faithful": True, "salient": True, "readable": True}, "", 0)


def test_prediction_rejects_targets_before_any_model_or_feature_access():
    for field in run.FORBIDDEN:
        with pytest.raises(ValueError, match="source metadata only"):
            run.predict(None, {field: []}, {}, None)


def test_pinned_latest_cache_and_source_identity():
    from experiments.structure_net.ncm_memory_data import file_hash
    assert file_hash(run.SOURCE / "cases.json") == run.SOURCE_SHA
    assert file_hash(run.FEATURES / "manifest.json") == run.FEATURE_SHA
    assert file_hash(run.OLD_CACHE) == run.CACHE_SHA
    assert len(run.q.read(run.OLD_CACHE)) == 2472
