import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_temporal_fixed_operator_ceiling_meta_hypothesis import (
    build_c3_temporal_fixed_operator_ceiling_experiment_results,
    build_c3_temporal_fixed_operator_ceiling_meta_hypothesis,
    store_c3_temporal_fixed_operator_ceiling_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_temporal_fixed_operator_ceiling/"
    "20260811_preregistered/result.json"
)


def test_meta_preserves_fixed_operator_population_decision() -> None:
    record = build_c3_temporal_fixed_operator_ceiling_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_fixed_multistep_group_operator_dominance_five_of_five"
    )
    assert hypothesis["subclaims"]["fixed_all_increment_dominance"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["same_task_unrestricted_tinyllm_training"] == (
        "not_licensed"
    )
    assert hypothesis["subclaims"]["typed_residual_function_class_preflight"] == (
        "not_licensed"
    )


def test_direct_evidence_preserves_all_seedwise_comparisons() -> None:
    record = build_c3_temporal_fixed_operator_ceiling_meta_hypothesis(RESULTS)
    cells = record["evidence"]["direct_tests"]
    assert len(cells) == 5
    for pair in cells:
        for regime in ("composition", "extrapolation"):
            cell = pair[regime]
            assert cell["valid"] is True
            assert cell["comparison"]["material_dominance_pass"] is True
            assert cell["comparison"]["rmse_ratio"] <= 0.75
            assert cell["comparison"]["accuracy_delta"] > 0.0
            assert cell["comparison"]["cross_entropy_delta"] < 0.0
            assert cell["operators"]["mean_increment"]["task_pass"] is True


def test_experiment_results_preserve_five_independent_units() -> None:
    record = build_c3_temporal_fixed_operator_ceiling_meta_hypothesis(RESULTS)
    experiments = build_c3_temporal_fixed_operator_ceiling_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 5
    assert all(item.primary_metric == 1.0 for item in experiments)
    assert all(item.model_parameters == 0 for item in experiments)
    assert all(item.training_time == 0.0 for item in experiments)
    assert max(
        item.metrics["composition_rmse_ratio"] for item in experiments
    ) < 0.53
    assert max(
        item.metrics["extrapolation_rmse_ratio"] for item in experiments
    ) < 0.53
    assert min(
        item.metrics["extrapolation_mean_accuracy"] for item in experiments
    ) > 0.97


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["aggregates"]["material_dominance_seed_pass_count"] = 4
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="C3 temporal fixed-operator result"):
        build_c3_temporal_fixed_operator_ceiling_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_fixed_operator_ceiling_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_fixed_operator_ceiling_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-temporal-fixed-operator-ceiling-v1",
        "experiment_count": 5,
    }
