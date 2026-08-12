from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_relational_connection_preflight as study,
)


PRIMARY_RESULT = Path(
    "data/experiments/tinyllm_c3_relational_connection_preflight/"
    "20260811_preregistered/result.json"
)
PRIMARY_RESULT_SHA256 = (
    "ae0321b3c3d5b7e8c80ebbd57bfaea10c0522c4b2241082daab007a54e55984e"
)


def _synthetic_dataset(count: int = 64) -> study.RelationalDataset:
    generator = torch.Generator(device="cpu").manual_seed(1193)
    theta_0 = 2.0 * math.pi * torch.rand(
        count, dtype=torch.float64, generator=generator
    )
    delta = math.pi * (
        2.0 * torch.rand(count, dtype=torch.float64, generator=generator) - 1.0
    )
    interior = 2.0 * math.pi * torch.rand(
        count,
        study.TIME_STEPS - 2,
        dtype=torch.float64,
        generator=generator,
    )
    phase = torch.cat(
        (theta_0[:, None], interior, (theta_0 + delta)[:, None]), dim=1
    )
    amplitude = torch.ones(count, dtype=torch.float64)
    offset = torch.zeros(count, dtype=torch.float64)
    drift = torch.zeros(count, dtype=torch.float64)
    gauge = torch.randint(
        0,
        study.CHANNELS,
        (count, study.TIME_STEPS),
        generator=generator,
        dtype=torch.int64,
    )
    continuous = study._continuous_observation(
        phase, amplitude, offset, drift
    )
    canonical = study.source.quantize(continuous)
    connection = study.edge_connection(gauge)
    return study.RelationalDataset(
        canonical_tokens=canonical,
        tokens=study.apply_local_action(canonical, gauge),
        calibration=torch.stack((amplitude, offset, drift), dim=-1),
        target=torch.cos(delta),
        phase=phase,
        delta=delta,
        gauge=gauge,
        connection=connection,
        saturation_count=0,
        dataset_seed=1193,
    )


def test_frozen_sources_and_predecessor_are_valid() -> None:
    hashes = study.validate_sources()
    assert hashes == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "generator": study.GENERATOR_SHA256,
        "interval_likelihood": study.INTERVAL_SHA256,
        "full_local_gauge_preregistration": (
            study.CLOSURE_PREREGISTRATION_SHA256
        ),
        "full_local_gauge_runner": study.CLOSURE_RUNNER_SHA256,
        "full_local_gauge_result": study.CLOSURE_RESULT_SHA256,
        "full_local_gauge_report": study.CLOSURE_REPORT_SHA256,
    }


def test_collision_removes_pointwise_identifiability_only() -> None:
    witness = study.collision_witness()
    assert witness["pass"] is True
    assert witness["token_error_count_without_connection"] == 0
    assert witness["pointwise_cube_maximum_error"] <= 2e-12
    assert witness["connection_difference_count"] == 1
    assert witness["target_absolute_separation"] == pytest.approx(1.5)


def test_observed_pair_action_is_exact_and_prediction_is_invariant() -> None:
    dataset = _synthetic_dataset()
    first = torch.randint(
        0,
        study.CHANNELS,
        dataset.gauge.shape,
        generator=torch.Generator().manual_seed(1201),
    )
    second = torch.randint(
        0,
        study.CHANNELS,
        dataset.gauge.shape,
        generator=torch.Generator().manual_seed(1207),
    )
    contracts = study.observed_pair_contracts(dataset, first, second)
    assert contracts["pass"] is True
    assert sum(
        value
        for name, value in contracts.items()
        if name.endswith("error_count")
    ) == 0

    character, _ = study.charged_character(
        dataset.tokens, dataset.calibration
    )
    prediction = study.connection_prediction(character, dataset.connection)
    transformed_tokens = study.apply_local_action(dataset.tokens, first)
    transformed_connection = study.transform_connection(
        dataset.connection, first
    )
    transformed_character, _ = study.charged_character(
        transformed_tokens, dataset.calibration
    )
    transformed_prediction = study.connection_prediction(
        transformed_character, transformed_connection
    )
    assert torch.max(torch.abs(transformed_prediction - prediction)) <= 2e-12
    assert torch.sqrt(torch.mean((prediction - dataset.target).square())) < 0.01


def _classification_cells(
    *,
    observed: int,
    specific: int,
    pointwise: int,
    shortcut: int,
) -> list[dict]:
    cells = []
    for index, seed in enumerate(study.SEEDS):
        for regime in study.REGIMES:
            cells.append(
                {
                    "seed": seed,
                    "regime": regime,
                    "observed_connection_pass": index < observed,
                    "connection_specificity_pass": index < specific,
                    "pointwise_insufficiency_pass": index < pointwise,
                    "connection_free_shortcut_pass": index < shortcut,
                }
            )
    return cells


def test_classification_licenses_only_function_class_preflight() -> None:
    positive = study.classify(
        _classification_cells(
            observed=5, specific=5, pointwise=5, shortcut=0
        ),
        True,
    )
    assert positive["classification"] == (
        "observed_edge_connection_identifies_nonpointwise_c3_relation"
    )
    assert positive["connection_conditioned_function_class_preflight_licensed"]
    assert positive["matched_training_directly_licensed"] is False
    assert positive["unrestricted_tinyllm_training_licensed"] is False

    shortcut = study.classify(
        _classification_cells(
            observed=5, specific=5, pointwise=5, shortcut=4
        ),
        True,
    )
    assert shortcut["classification"] == (
        "relational_scope_has_connection_free_shortcut"
    )
    assert shortcut["connection_conditioned_function_class_preflight_licensed"] is False

    assert study.classify(
        _classification_cells(
            observed=3, specific=5, pointwise=5, shortcut=0
        ),
        True,
    )["classification"] == "relational_connection_quantization_ceiling_failed"
    assert study.classify(
        _classification_cells(
            observed=5, specific=3, pointwise=5, shortcut=0
        ),
        True,
    )["classification"] == "relational_connection_not_causally_specific"
    assert study.classify(
        _classification_cells(
            observed=5, specific=5, pointwise=5, shortcut=0
        ),
        False,
    )["classification"] == "invalid_relational_connection_preflight"


def test_main_writes_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "observed_connection_seed_pass_count": 5,
            "connection_specificity_seed_pass_count": 5,
            "pointwise_insufficiency_seed_pass_count": 5,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_authoritative_result_establishes_connection_specific_relation() -> None:
    assert hashlib.sha256(PRIMARY_RESULT.read_bytes()).hexdigest() == (
        PRIMARY_RESULT_SHA256
    )
    result = json.loads(PRIMARY_RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["schema_version"] == study.SCHEMA_VERSION
    assert result["hypothesis_id"] == study.HYPOTHESIS_ID
    assert result["source_hashes"]["runner"] == (
        "2ab21b7885fa93c49427973f67e5c57913b7f13c154ad2e970ef9636b2b90214"
    )
    assert result["aggregates"] == {
        "classification": (
            "observed_edge_connection_identifies_nonpointwise_c3_relation"
        ),
        "connection_conditioned_function_class_preflight_licensed": True,
        "connection_free_shortcut_seed_pass_count": 0,
        "connection_specificity_seed_pass_count": 5,
        "matched_training_directly_licensed": False,
        "observed_connection_seed_pass_count": 5,
        "pointwise_insufficiency_seed_pass_count": 5,
        "required_seed_passes": 4,
        "unrestricted_tinyllm_training_licensed": False,
        "valid": True,
    }
    assert result["identifiability"]["collision_witness"]["pass"] is True
    assert result["accounting"] == {
        "checkpoints_loaded": 0,
        "local_action_counterfactual_evaluations": 40_960,
        "new_evaluation_examples": 40_960,
        "optimizer_steps": 0,
        "parameters_changed": 0,
        "reusable_parameter_fits": 0,
        "target_using_fits": 0,
        "tinyllm_models_instantiated": 0,
    }
    assert len(result["cells"]) == 10
    assert all(
        cell["valid"] is True
        and cell["observed_connection_pass"] is True
        and cell["connection_specificity_pass"] is True
        and cell["pointwise_insufficiency_pass"] is True
        and cell["connection_free_shortcut_pass"] is False
        for cell in result["cells"]
    )
    assert max(
        cell["arms"]["observed_connection"]["scalar"]["rmse"]
        for cell in result["cells"]
    ) <= 0.01
    assert max(
        cell["local_action_prediction_maximum_error"]
        for cell in result["cells"]
    ) <= 2e-12
    assert sum(
        value
        for cell in result["cells"]
        for name, value in cell["observed_pair_contracts"].items()
        if name.endswith("error_count")
    ) == 0
