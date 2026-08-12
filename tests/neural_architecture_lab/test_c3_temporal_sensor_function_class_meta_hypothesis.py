import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_temporal_sensor_function_class_meta_hypothesis import (
    build_c3_temporal_sensor_function_class_experiment_results,
    build_c3_temporal_sensor_function_class_meta_hypothesis,
    store_c3_temporal_sensor_function_class_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_temporal_sensor_function_class/"
    "20260811_preregistered/result.json"
)


def test_meta_preserves_constructive_capacity_and_scope_boundary() -> None:
    record = build_c3_temporal_sensor_function_class_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_function_class_capacity_and_local_gradient_only"
    )
    assert hypothesis["subclaims"]["analytic_carrier_in_function_class"] == (
        "supported_five_nonzero_parameter_witness"
    )
    assert hypothesis["subclaims"]["global_trainability"] == "not_tested"
    assert hypothesis["subclaims"]["tinyllm_utility"] == "not_tested"


def test_direct_evidence_preserves_zero_training_and_both_shifts() -> None:
    record = build_c3_temporal_sensor_function_class_meta_hypothesis(RESULTS)
    direct = record["evidence"]["direct_tests"]
    assert len(direct) == 1
    cell = direct[0]
    assert cell["gates"]["campaign_licensed"] is True
    assert cell["accounting"]["optimizer_steps"] == 0
    assert cell["accounting"]["tinyllm_models_instantiated"] == 0
    assert cell["witness"]["regimes"]["composition"]["pass"] is True
    assert cell["witness"]["regimes"]["extrapolation"]["pass"] is True
    assert cell["gradient_route"]["pass"] is True


def test_experiment_result_is_one_no_training_preflight() -> None:
    record = build_c3_temporal_sensor_function_class_meta_hypothesis(RESULTS)
    experiments = build_c3_temporal_sensor_function_class_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 1
    result = experiments[0]
    assert result.primary_metric == 1.0
    assert result.model_parameters == 184
    assert result.metrics["optimizer_steps"] == 0.0
    assert result.metrics["tinyllm_models_instantiated"] == 0.0
    assert result.metrics["composition_temporal_correlation"] >= 0.99
    assert result.metrics["extrapolation_temporal_correlation"] >= 0.99


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["gates"]["campaign_licensed"] = False
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="C3 sensor function-class preflight"):
        build_c3_temporal_sensor_function_class_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_sensor_function_class_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 1


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_temporal_sensor_function_class_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-temporal-sensor-function-class-v1",
        "experiment_count": 1,
    }
