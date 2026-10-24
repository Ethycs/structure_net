"""Guard full-native state loading and the unchanged source-only task boundary."""
from copy import deepcopy

import pytest

from experiments.structure_net import ncm_lm1b_native_development as run


def valid_metadata():
    state = {"status": "native655_training_completed_source_only_DEV_pending", "updates": 1024,
        "train_cases": 655, "trainable_parameters": 127732736, "strict_reload_passed": True,
        "root_queries_updated": True, "native16_or_pilot_weights_reused": False,
        "initial_trainable_sha256": run.training.capacity.INITIAL, "final_trainable_sha256": "updated",
        "protected_codec_hashes": {"encoder": run.training.lifecycle.ENCODER, "VQ": run.training.lifecycle.VQ},
        "peak_allocated_bytes": 6_000_000_000, "source_disjoint_task_accuracy": None, "speed_tested": False,
        "all_final_sources_closed": True, "producer_fingerprint": "fixed"}
    saved = {"step": 1024, "config": run.asdict(run.training.Config()), "producer_fingerprint": "fixed"}
    return state, saved


def test_full_native_checkpoint_budget_reset_and_geometry_required():
    state, saved = valid_metadata()
    run.checkpoint_metadata(state, saved)
    for patch in ({"updates": 2}, {"train_cases": 16}, {"trainable_parameters": 11796480},
            {"native16_or_pilot_weights_reused": True}, {"strict_reload_passed": False},
            {"root_queries_updated": False}, {"final_trainable_sha256": run.training.capacity.INITIAL},
            {"protected_codec_hashes": {"encoder": "changed", "VQ": run.training.lifecycle.VQ}}):
        with pytest.raises(ValueError):
            run.checkpoint_metadata({**state, **patch}, saved)
    with pytest.raises(ValueError):
        run.checkpoint_metadata(state, {**saved, "step": 512})


def test_actual128_source_boundary_and655_exclusions():
    cases = run.q.read(run.SOURCE / "cases.json")
    assert run.validate_sources(cases) is cases
    rows = run.training.choose(run.q.read(run.training.training.DATA / "cases.json"))
    assert run.training.split_guard(rows, cases)["group_id"]["overlap"] == 0
    bad = deepcopy(cases)
    bad[run.COHORTS[0]][0]["native_word_ids"] = [1]
    with pytest.raises(ValueError):
        run.validate_sources(bad)
    for field in run.FORBIDDEN:
        with pytest.raises(ValueError, match="source metadata only"):
            run.predict(None, {field: []}, {}, None)


def test_joint_gate_requires_the_same_cases_and_both_complete_cohorts():
    a = [{"id": str(i), "valid": True, "passed": i >= 3} for i in range(64)]
    b = [{"id": str(i), "valid": True, "passed": i not in (3, 4, 5)} for i in range(64)]
    assert run.joint_gate(a, a)["qualified"]
    assert run.joint_gate(a, b)["joint_passed"] == 58 and not run.joint_gate(a, b)["qualified"]
    metric = {"cases": 64, "valid": 64, "passed": 61}
    assert run.secondary_admitted({c: metric for c in run.COHORTS})
    assert not run.secondary_admitted({run.COHORTS[0]: metric})


def test_inference_loads_entire_native_state_before_freezing(monkeypatch):
    full = {"concepts.generator.weight": "trained", "concepts.codec.decoder.weight": "trained", "word_adapters.weight": "trained"}
    state = {"checkpoint": "fixed1024"}
    events = []

    class Model:
        def load_state_dict(self, value, strict):
            assert value is full and strict is True
            events.append("full_state_loaded")

        def eval(self):
            events.append("eval")
            return self

        def requires_grad_(self, value):
            assert value is False
            events.append("frozen")
            return self

    model = Model()
    monkeypatch.setattr(run, "checkpoint_contract", lambda: (state, {"model": full}))
    monkeypatch.setattr(run.training.capacity, "fresh", lambda: model)
    monkeypatch.setattr(run, "model_contract", lambda m, s: events.append("actual_full_hashes_checked") if m is model and s is state else pytest.fail())
    assert run.load_selected() == (model, state)
    assert events == ["full_state_loaded", "actual_full_hashes_checked", "eval", "frozen"]


def test_latest_canonical_cache_and_unchanged_source_files_are_pinned():
    assert run.file_hash(run.OLD_CACHE) == run.CACHE_SHA
    assert len(run.q.read(run.OLD_CACHE)) == run.CACHE_COUNT == 3094
    assert run.file_hash(run.SOURCE / "cases.json") == run.SOURCE_SHA
    assert run.file_hash(run.FEATURES / "manifest.json") == run.FEATURE_SHA
