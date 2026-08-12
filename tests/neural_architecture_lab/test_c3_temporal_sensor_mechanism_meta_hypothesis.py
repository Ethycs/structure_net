import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_temporal_sensor_mechanism_meta_hypothesis import (
    build_c3_temporal_sensor_mechanism_experiment_results,
    build_c3_temporal_sensor_mechanism_meta_hypothesis,
    store_c3_temporal_sensor_mechanism_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_mechanism/"
    "20260811_preregistered/campaign_results.json"
)


def test_meta_preserves_affine_identity_population_result() -> None:
    record = build_c3_temporal_sensor_mechanism_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_affine_identity_character_causal_sufficiency_five_of_five"
    )
    assert hypothesis["subclaims"]["affine_identity_causal_sufficiency"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["nonlinear_response_necessity"] == (
        "rejected_zero_of_five"
    )
    assert hypothesis["subclaims"]["target_shuffle_specificity"] == (
        "supported_zero_of_five"
    )
    assert hypothesis["subclaims"]["tinyllm_utility"] == (
        "not_tested_tinyllm_absent"
    )


def test_direct_evidence_preserves_all_causal_response_gates() -> None:
    record = build_c3_temporal_sensor_mechanism_meta_hypothesis(RESULTS)
    cells = record["evidence"]["direct_tests"]
    assert len(cells) == 5
    for pair in cells:
        true = pair["true"]
        shuffled = pair["target_shuffled"]
        assert true["response_seed_gates"] == {
            "affine_only": True,
            "full_replay": True,
            "nonlinear_residual_only": False,
        }
        assert shuffled["response_seed_gates"]["affine_only"] is False
        assert true["full_replay_pass"] is True
        assert shuffled["full_replay_pass"] is True
        for regime in ("composition", "extrapolation"):
            responses = true["regimes"][regime]["responses"]
            assert responses["affine_only"]["joint_pass"] is True
            assert responses["nonlinear_residual_only"]["joint_pass"] is False
            assert responses["affine_only"]["task"]["exact_bin_accuracy"] >= 0.90
            assert true["regimes"][regime][
                "coefficient_reconstruction_maximum_error"
            ] <= 1e-6


def test_experiment_results_preserve_five_independent_seed_units() -> None:
    record = build_c3_temporal_sensor_mechanism_meta_hypothesis(RESULTS)
    experiments = build_c3_temporal_sensor_mechanism_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 5
    assert all(item.primary_metric == 1.0 for item in experiments)
    assert all(item.model_parameters == 184 for item in experiments)
    assert all(item.training_time == 0.0 for item in experiments)
    assert min(
        item.metrics["true_affine_extrapolation_accuracy"]
        for item in experiments
    ) >= 0.90
    assert max(
        item.metrics["true_residual_extrapolation_accuracy"]
        for item in experiments
    ) < 0.20
    assert max(
        item.metrics["shuffled_affine_extrapolation_accuracy"]
        for item in experiments
    ) < 0.10


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["aggregates"]["true_affine_only_pass_count"] = 4
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="C3 temporal sensor mechanism result"):
        build_c3_temporal_sensor_mechanism_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_sensor_mechanism_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_sensor_mechanism_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-temporal-sensor-mechanism-v1",
        "experiment_count": 5,
    }
