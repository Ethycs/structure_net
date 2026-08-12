import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_single_frame_corruption_fixed_estimator_meta_hypothesis import (
    build_c3_single_frame_corruption_fixed_estimator_experiment_results,
    build_c3_single_frame_corruption_fixed_estimator_meta_hypothesis,
    store_c3_single_frame_corruption_fixed_estimator_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_single_frame_corruption_fixed_estimator/"
    "20260811_preregistered/result.json"
)


def test_meta_preserves_robust_fixed_population_decision() -> None:
    record = build_c3_single_frame_corruption_fixed_estimator_meta_hypothesis(
        RESULTS
    )
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_fixed_robust_estimator_closes_single_frame_"
        "corruption_five_of_five"
    )
    assert hypothesis["subclaims"]["single_frame_corruption_materiality"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["robust_fixed_sufficiency"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["tinyllm_training"] == "not_licensed"


def test_direct_evidence_preserves_all_cells_and_controls() -> None:
    record = build_c3_single_frame_corruption_fixed_estimator_meta_hypothesis(
        RESULTS
    )
    pairs = record["evidence"]["direct_tests"]
    assert len(pairs) == 5
    for pair in pairs:
        for regime in ("composition", "extrapolation"):
            cell = pair[regime]
            assert cell["valid"] is True
            assert cell["corruption_material"] is True
            assert cell["operators"]["corrupted_all_frame_degree2"][
                "fixed_ceiling_pass"
            ] is False
            assert cell["operators"]["oracle_drop_one_quadratic"][
                "fixed_ceiling_pass"
            ] is True
            assert cell["operators"]["robust_drop_one_quadratic"][
                "fixed_ceiling_pass"
            ] is True
            assert cell["robust_repair"]["pass"] is True
            assert cell["oracle_fidelity"]["pass"] is True
            assert cell["quantized_corruption_index_recovery_rate"] >= 0.99


def test_experiment_results_preserve_five_independent_units() -> None:
    record = build_c3_single_frame_corruption_fixed_estimator_meta_hypothesis(
        RESULTS
    )
    experiments = (
        build_c3_single_frame_corruption_fixed_estimator_experiment_results(
            record, RESULTS
        )
    )
    assert len(experiments) == 5
    assert all(item.primary_metric == 1.0 for item in experiments)
    assert all(item.model_parameters == 0 for item in experiments)
    assert all(item.training_time == 0.0 for item in experiments)
    assert max(
        item.metrics["composition_robust_repair_rmse_ratio"]
        for item in experiments
    ) < 0.01
    assert min(
        item.metrics["extrapolation_corruption_index_recovery_rate"]
        for item in experiments
    ) > 0.99


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["aggregates"]["robust_repair_seed_pass_count"] = 4
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="C3 single-frame-corruption result"):
        build_c3_single_frame_corruption_fixed_estimator_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_single_frame_corruption_fixed_estimator_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 5


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_single_frame_corruption_fixed_estimator_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": (
            "tinyllm-c3-single-frame-corruption-fixed-estimator-v1"
        ),
        "experiment_count": 5,
    }
