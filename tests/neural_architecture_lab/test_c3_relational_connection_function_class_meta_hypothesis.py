import json
from pathlib import Path

import pytest

from neural_architecture_lab.c3_relational_connection_function_class_meta_hypothesis import (
    build_c3_relational_connection_function_class_experiment_results,
    build_c3_relational_connection_function_class_meta_hypothesis,
    store_c3_relational_connection_function_class_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_c3_relational_connection_function_class/"
    "20260811_preregistered/result.json"
)


def test_meta_preserves_exact_capacity_and_scope_boundary() -> None:
    record = build_c3_relational_connection_function_class_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "confirmed_exact_function_class_gradient_and_cuda_lifecycle_only"
    )
    assert hypothesis["subclaims"]["analytic_transport_in_function_class"] == (
        "supported_six_nonzero_parameter_witness"
    )
    assert hypothesis["subclaims"]["all_parameter_action_invariance"] == (
        "supported_by_architecture_and_three_state_audit"
    )
    assert hypothesis["subclaims"]["matched_sensor_readout_campaign"] == (
        "licensed_not_yet_tested"
    )
    assert hypothesis["subclaims"]["global_trainability"] == "not_tested"
    assert hypothesis["subclaims"]["unrestricted_tinyllm_training"] == (
        "not_licensed"
    )


def test_direct_evidence_preserves_gradient_controls_and_lifecycle() -> None:
    record = build_c3_relational_connection_function_class_meta_hypothesis(RESULTS)
    direct = record["evidence"]["direct_tests"]
    assert len(direct) == 1
    cell = direct[0]
    assert cell["aggregates"]["matched_sensor_readout_campaign_licensed"] is True
    assert cell["architecture_contract"]["total_parameters"] == 187
    assert cell["analytic_witness"]["nonzero_parameter_count"] == 6
    assert cell["gradient_and_invariance"]["pass"] is True
    assert cell["gradient_and_invariance"][
        "connection_shuffled_gradient"
    ]["pass"] is True
    assert cell["gradient_and_invariance"]["target_shuffled_gradient"][
        "pass"
    ] is True
    assert cell["checkpoint_device_lifecycle"]["cpu_pass"] is True
    assert cell["checkpoint_device_lifecycle"]["cuda"]["pass"] is True
    assert cell["accounting"]["optimizer_steps"] == 0
    assert cell["accounting"]["tinyllm_models_instantiated"] == 0


def test_experiment_result_is_one_no_training_preflight() -> None:
    record = build_c3_relational_connection_function_class_meta_hypothesis(RESULTS)
    experiments = build_c3_relational_connection_function_class_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 1
    result = experiments[0]
    assert result.primary_metric == 1.0
    assert result.model_parameters == 187
    assert result.metrics["optimizer_steps"] == 0.0
    assert result.metrics["tinyllm_models_instantiated"] == 0.0
    assert result.metrics["gradient_nonzero_fraction"] >= 0.90
    assert result.metrics["cuda_lifecycle_pass"] == 1.0


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["aggregates"]["cuda_lifecycle_pass"] = False
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="C3 relational function-class result"):
        build_c3_relational_connection_function_class_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_relational_connection_function_class_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 1


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_c3_relational_connection_function_class_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c3-relational-connection-function-class-v1",
        "experiment_count": 1,
    }
