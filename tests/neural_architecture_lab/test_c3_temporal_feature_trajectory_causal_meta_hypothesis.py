import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_temporal_feature_trajectory_causal_meta_hypothesis import (
    build_c3_temporal_feature_trajectory_causal_experiment_results,
    build_c3_temporal_feature_trajectory_causal_meta_hypothesis,
    store_c3_temporal_feature_trajectory_causal_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_temporal_feature_trajectory_causal/"
    "20260811_d6_preregistered/result.json"
)


def test_meta_preserves_registered_negative_decision() -> None:
    record = build_c3_temporal_feature_trajectory_causal_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_fixed_operator_available_but_frozen_continuation_"
        "cannot_use_projected_trajectory"
    )
    assert hypothesis["subclaims"][
        "all_frame_trajectory_repairs_frozen_continuation"
    ] == "not_supported_two_of_five"
    assert hypothesis["subclaims"]["fixed_all_frame_bypass_available"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["same_task_tinyllm_repair"] == "not_licensed"


def test_direct_evidence_preserves_all_frozen_checkpoint_controls() -> None:
    record = build_c3_temporal_feature_trajectory_causal_meta_hypothesis(RESULTS)
    cells = record["evidence"]["direct_tests"]
    assert len(cells) == 5
    assert sum(
        cell["arm_seed_gates"]["mean_consistent"]["true_both_shifts"]
        for cell in cells
    ) == 2
    for cell in cells:
        assert cell["valid"] is True
        assert cell["state_unchanged"] is True
        assert cell["fixed_mean_bypass_both_shifts"] is True
        assert cell["arm_seed_gates"]["mean_consistent"][
            "shuffled_both_shifts"
        ] is False
        assert cell["arm_seed_gates"]["early_deranged"][
            "true_both_shifts"
        ] is False


def test_experiment_results_preserve_five_independent_checkpoint_units() -> None:
    record = build_c3_temporal_feature_trajectory_causal_meta_hypothesis(RESULTS)
    experiments = build_c3_temporal_feature_trajectory_causal_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 5
    assert sum(item.primary_metric for item in experiments) == 2.0
    assert all(item.model_parameters == 29_951_232 for item in experiments)
    assert all(item.training_time == 0.0 for item in experiments)
    assert all(item.metrics["fixed_mean_bypass_both_shifts"] == 1.0 for item in experiments)
    assert min(
        item.metrics["extrapolation_early_argmax_change"] for item in experiments
    ) > 0.73


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["aggregates"]["mean_consistent_task_pass_count"] = 4
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="C3 feature-trajectory result"):
        build_c3_temporal_feature_trajectory_causal_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_feature_trajectory_causal_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_feature_trajectory_causal_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-temporal-feature-trajectory-causal-v1",
        "experiment_count": 5,
    }
