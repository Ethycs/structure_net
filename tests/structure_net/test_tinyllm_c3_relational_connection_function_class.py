from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_relational_connection_function_class as study,
)


PRIMARY_RESULT = Path(
    "data/experiments/tinyllm_c3_relational_connection_function_class/"
    "20260811_preregistered/result.json"
)
PRIMARY_RESULT_SHA256 = (
    "2292e971bb655db246565675fece8dfd9e1546692b9b782e940a8bbef49de82c"
)


def _pilot() -> study.rel.RelationalDataset:
    return study.generate_diagnostic_dataset(seed=1_141_103, count=64)


def _pilot_action() -> torch.Tensor:
    return torch.randint(
        0,
        study.rel.CHANNELS,
        (64, study.rel.TIME_STEPS),
        generator=torch.Generator().manual_seed(1_141_109),
        dtype=torch.int64,
    )


def test_sources_and_predecessor_license_are_frozen() -> None:
    hashes, predecessor = study.validate_sources()
    assert hashes == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "relational_runner": study.RELATIONAL_RUNNER_SHA256,
        "relational_result": study.RELATIONAL_RESULT_SHA256,
        "relational_report": study.RELATIONAL_REPORT_SHA256,
        "sensor_family": study.SENSOR_FAMILY_SHA256,
        "interval_likelihood": study.INTERVAL_SHA256,
    }
    assert predecessor["aggregates"][
        "connection_conditioned_function_class_preflight_licensed"
    ] is True
    assert predecessor["aggregates"]["matched_training_directly_licensed"] is False


def test_closed_form_witness_has_six_nonzero_parameters_and_solves_pilot() -> None:
    module = study.construct_analytic_witness(
        study.ConnectionInvariantRelationalModule()
    )
    contract = study.module_contract(module)
    assert contract["pass"] is True
    assert contract["encoder_parameters"] == 184
    assert contract["head_parameters"] == 3
    assert contract["total_parameters"] == 187
    assert sum(int(torch.count_nonzero(value)) for value in module.parameters()) == 6

    dataset = _pilot()
    prediction = module(
        dataset.tokens, dataset.calibration, dataset.connection
    )
    metrics = study.rel._scalar_metrics(prediction, dataset.target)
    assert metrics["correlation"] >= 0.999
    assert metrics["rmse"] <= 0.01


def test_random_perturbed_and_witness_states_are_connection_invariant() -> None:
    dataset = _pilot()
    action = _pilot_action()
    torch.manual_seed(1_141_111)
    random_module = study.ConnectionInvariantRelationalModule()
    states = (
        random_module,
        study._parameter_perturbation(random_module),
        study.construct_analytic_witness(
            study.ConnectionInvariantRelationalModule()
        ),
    )
    for index, module in enumerate(states):
        result = study._state_invariance(
            f"state_{index}", module, dataset, action
        )
        assert result["pass"] is True
        assert result["charged_covariance_maximum_error"] <= 2e-5
        assert result["output_invariance_maximum_error"] <= 2e-5


def test_cpu_checkpoint_lifecycle_is_tensor_exact() -> None:
    dataset = _pilot()
    torch.manual_seed(1_141_113)
    module = study.ConnectionInvariantRelationalModule()
    lifecycle = study.checkpoint_device_lifecycle(
        module, dataset, require_cuda=False
    )
    assert lifecycle["cpu_pass"] is True
    assert lifecycle["cpu_state_digest_exact"] is True
    assert lifecycle["cpu_output_tensor_exact"] is True
    assert lifecycle["schema_and_parameter_count_exact"] is True


def test_classification_requires_every_lifecycle_gate() -> None:
    positive = study.classify(
        source_contract=True,
        valid=True,
        witness_pass=True,
        invariance_pass=True,
        gradient_pass=True,
        cpu_pass=True,
        cuda_pass=True,
    )
    assert positive["classification"] == study.CLASSIFICATION_PASS
    assert positive["matched_sensor_readout_campaign_licensed"] is True
    assert positive["unrestricted_tinyllm_training_licensed"] is False

    pending = study.classify(
        source_contract=True,
        valid=True,
        witness_pass=True,
        invariance_pass=True,
        gradient_pass=True,
        cpu_pass=True,
        cuda_pass=False,
    )
    assert pending["classification"] == (
        "connection_function_class_valid_cuda_lifecycle_pending"
    )
    assert pending["matched_sensor_readout_campaign_licensed"] is False

    assert study.classify(
        source_contract=True,
        valid=True,
        witness_pass=False,
        invariance_pass=True,
        gradient_pass=True,
        cpu_pass=True,
        cuda_pass=True,
    )["classification"] == "connection_function_class_missing_analytic_transport"
    assert study.classify(
        source_contract=True,
        valid=True,
        witness_pass=True,
        invariance_pass=True,
        gradient_pass=False,
        cpu_pass=True,
        cuda_pass=True,
    )["classification"] == "connection_function_class_gradient_route_insufficient"
    assert study.classify(
        source_contract=False,
        valid=True,
        witness_pass=True,
        invariance_pass=True,
        gradient_pass=True,
        cpu_pass=True,
        cuda_pass=True,
    )["classification"] == "invalid_connection_function_class_preflight"


def test_main_writes_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "analytic_witness_pass": True,
            "structural_invariance_pass": True,
            "gradient_route_pass": True,
            "cpu_lifecycle_pass": True,
            "cuda_lifecycle_pass": True,
            "matched_sensor_readout_campaign_licensed": True,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda **_: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output), "--allow-no-cuda"]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_authoritative_result_licenses_only_matched_acquisition() -> None:
    assert hashlib.sha256(PRIMARY_RESULT.read_bytes()).hexdigest() == (
        PRIMARY_RESULT_SHA256
    )
    result = json.loads(PRIMARY_RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["schema_version"] == study.SCHEMA_VERSION
    assert result["hypothesis_id"] == study.HYPOTHESIS_ID
    assert result["source_hashes"]["runner"] == (
        "7010794c3be5fda05a035e5a0b4a178aacd40c934dc1f5f7b36eb2eb03ea96b1"
    )
    assert result["aggregates"] == {
        "analytic_witness_pass": True,
        "classification": study.CLASSIFICATION_PASS,
        "cpu_lifecycle_pass": True,
        "cuda_lifecycle_pass": True,
        "gradient_route_pass": True,
        "matched_sensor_readout_campaign_licensed": True,
        "source_contract_pass": True,
        "structural_invariance_pass": True,
        "unrestricted_tinyllm_training_licensed": False,
        "valid": True,
    }
    assert result["architecture_contract"]["total_parameters"] == 187
    assert result["analytic_witness"]["nonzero_parameter_count"] == 6
    assert len(result["analytic_witness"]["cells"]) == 10
    assert all(cell["pass"] is True for cell in result["analytic_witness"]["cells"])
    assert result["gradient_and_invariance"]["pass"] is True
    assert result["gradient_and_invariance"]["nonzero_gradient_fraction"] >= 0.90
    assert len(result["gradient_and_invariance"]["state_invariance"]) == 3
    assert all(
        state["pass"] is True
        for state in result["gradient_and_invariance"]["state_invariance"]
    )
    assert result["checkpoint_device_lifecycle"]["cpu_output_tensor_exact"] is True
    assert result["checkpoint_device_lifecycle"]["cuda"]["pass"] is True
    assert result["accounting"] == {
        "closed_form_nonzero_witness_parameters": 6,
        "fresh_diagnostic_examples": 512,
        "historical_checkpoints_loaded": 0,
        "optimizer_steps": 0,
        "source_examples_reused": 40_960,
        "temporary_lifecycle_checkpoint_loads": 2,
        "temporary_lifecycle_checkpoints_created": 1,
        "tinyllm_models_instantiated": 0,
        "trained_parameters": 0,
    }
