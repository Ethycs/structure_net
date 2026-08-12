from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_constant_acceleration_fixed_operator as study,
)


PRIMARY_RESULT_SHA256 = (
    "b04a5574efc658ec1ed73f70fa494041ad16c0ae1342423cdde32925c1c7bc53"
)
RUNNER_SHA256 = (
    "6ea952f386b82b12355c3aa2e9552af6bf73e03e7cd47310fec764ce49d0d5e2"
)


def test_sources_and_preregistration_are_frozen() -> None:
    assert study.validate_sources() == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "generator": study.GENERATOR_SHA256,
        "interval_likelihood": study.INTERVAL_SHA256,
        "fixed_operator_runner": study.FIXED_OPERATOR_RUNNER_SHA256,
        "fixed_operator_result": study.FIXED_OPERATOR_RESULT_SHA256,
        "feature_trajectory_result": study.FEATURE_TRAJECTORY_RESULT_SHA256,
    }


def test_degree2_group_operators_are_exact_for_constant_acceleration() -> None:
    phase = torch.linspace(-math.pi, math.pi, 257, dtype=torch.float64)
    velocity = torch.linspace(-0.20, 0.20, 257, dtype=torch.float64)
    acceleration = torch.linspace(-0.05, 0.05, 257, dtype=torch.float64)
    time = torch.arange(9, dtype=torch.float64)[None]
    angle = 3.0 * (
        phase[:, None]
        + velocity[:, None] * time
        + 0.5 * acceleration[:, None] * time * (time - 1.0)
    )
    carrier = torch.polar(torch.ones_like(angle), angle).to(torch.complex128)
    predictions = study.operator_complex_predictions(carrier[:, :8])
    target = carrier[:, 8]
    assert float((predictions["recent_degree2"] - target).abs().max()) <= 1e-12
    assert float((predictions["all_frame_degree2"] - target).abs().max()) <= 1e-12
    assert float((predictions["constant_speed_mean"] - target).abs().max()) > 0.1


def test_generator_is_deterministic_and_preserves_exact_c3_action() -> None:
    first = study.generate_dataset(
        "composition", 991, sample_count=64, allow_pilot=True
    )
    second = study.generate_dataset(
        "composition", 991, sample_count=64, allow_pilot=True
    )
    assert study.dataset_hash(first) == study.dataset_hash(second)
    assert first.saturation_count == 0
    contracts = study._group_contracts(first)
    assert contracts["pass"] is True
    assert contracts["maximum_target_action_error"] <= study.ACTION_ERROR_MAXIMUM


def test_disjoint_pilot_cell_exercises_complete_analysis_path() -> None:
    cell = study.analyze_cell(
        "composition", 991, sample_count=64, allow_pilot=True
    )
    assert cell["dataset_deterministic"] is True
    assert cell["group_contracts"]["pass"] is True
    assert cell["saturation_count"] == 0
    assert cell["continuous_operator_maximum_errors"]["recent_degree2"] <= 1e-12
    assert cell["continuous_operator_maximum_errors"]["all_frame_degree2"] <= 1e-12
    assert all(operator["action_pass"] for operator in cell["operators"].values())


def test_all_frame_operator_uses_every_acceleration_and_increment() -> None:
    phase = torch.tensor([0.3], dtype=torch.float64)
    velocity = torch.tensor([0.08], dtype=torch.float64)
    acceleration = torch.tensor([0.02], dtype=torch.float64)
    time = torch.arange(8, dtype=torch.float64)[None]
    angle = 3.0 * (
        phase[:, None]
        + velocity[:, None] * time
        + 0.5 * acceleration[:, None] * time * (time - 1.0)
    )
    carrier = torch.polar(torch.ones_like(angle), angle).to(torch.complex128)
    baseline = study.operator_complex_predictions(carrier)["all_frame_degree2"]
    perturbed = carrier.clone()
    perturbed[:, 1] *= torch.polar(
        torch.ones(1, dtype=torch.float64),
        torch.tensor([0.01], dtype=torch.float64),
    )
    changed = study.operator_complex_predictions(perturbed)["all_frame_degree2"]
    assert not torch.equal(baseline, changed)


def test_task_and_fixed_ceiling_gates_are_joint_and_inclusive() -> None:
    composition = {
        "posterior_mean_correlation": 0.90,
        "exact_bin_accuracy": 0.90,
        "target_cross_entropy": 1.80,
        "predicted_bin_coverage": 14,
    }
    extrapolation = {
        "posterior_mean_correlation": 0.90,
        "exact_bin_accuracy": 0.90,
        "target_cross_entropy": 2.20,
        "predicted_bin_coverage": 12,
    }
    scalar = {"rmse": 0.020}
    assert study._task_gate(composition, "composition") is True
    assert study._fixed_ceiling_gate(scalar, composition, "composition") is True
    assert study._fixed_ceiling_gate(scalar, extrapolation, "extrapolation") is True
    scalar["rmse"] = 0.020001
    assert study._fixed_ceiling_gate(scalar, extrapolation, "extrapolation") is False


def test_population_classifications_are_locked() -> None:
    def cells(recent: int, all_frame: int, material: int) -> list[dict]:
        values = []
        for index, seed in enumerate(study.SEEDS):
            for regime in study.REGIMES:
                values.append(
                    {
                        "seed": seed,
                        "regime": regime,
                        "operators": {
                            "constant_speed_mean": {"fixed_ceiling_pass": False},
                            "recent_degree2": {
                                "fixed_ceiling_pass": index < recent
                            },
                            "all_frame_degree2": {
                                "fixed_ceiling_pass": index < all_frame
                            },
                        },
                        "all_frame_vs_recent": {
                            "material_dominance_pass": index < material
                        },
                    }
                )
        return values

    assert study.classify(cells(5, 5, 4), True)["classification"] == (
        "fixed_all_frame_degree2_closes_constant_acceleration"
    )
    assert study.classify(cells(4, 4, 3), True)["classification"] == (
        "fixed_degree2_closes_without_multistep_dominance"
    )
    assert study.classify(cells(3, 3, 5), True)["classification"] == (
        "registered_fixed_degree2_family_insufficient"
    )
    assert study.classify(cells(5, 5, 5), False)["classification"] == (
        "invalid_constant_acceleration_preflight"
    )


def test_authoritative_result_closes_constant_acceleration_training() -> None:
    result = json.loads(study.PRIMARY_RESULT_PATH.read_text(encoding="utf-8"))
    assert study._sha256(study.PRIMARY_RESULT_PATH) == PRIMARY_RESULT_SHA256
    assert result["status"] == "completed"
    assert result["schema_version"] == study.SCHEMA_VERSION
    assert result["source_hashes"]["runner"] == RUNNER_SHA256
    assert result["aggregates"] == {
        "all_frame_degree2_fixed_ceiling_seed_pass_count": 5,
        "all_frame_material_dominance_seed_pass_count": 5,
        "classification": "fixed_all_frame_degree2_closes_constant_acceleration",
        "constant_acceleration_tinyllm_training_licensed": False,
        "constant_speed_fixed_ceiling_seed_pass_count": 0,
        "recent_degree2_fixed_ceiling_seed_pass_count": 5,
        "required_seed_passes": 4,
        "robust_typed_estimator_preflight_licensed": False,
        "valid": True,
    }
    assert result["accounting"] == {
        "checkpoints_loaded": 0,
        "models_instantiated": 0,
        "new_evaluation_examples": 40_960,
        "optimizer_steps": 0,
        "parameters_changed": 0,
        "target_using_fits": 0,
    }
    assert len(result["cells"]) == 10
    for cell in result["cells"]:
        assert cell["valid"] is True
        assert cell["dataset_deterministic"] is True
        assert cell["saturation_count"] == 0
        assert cell["shuffle_fixed_points"] == 0
        assert cell["group_contracts"]["pass"] is True
        assert cell["operators"]["constant_speed_mean"][
            "fixed_ceiling_pass"
        ] is False
        for arm in ("recent_degree2", "all_frame_degree2"):
            assert cell["operators"][arm]["fixed_ceiling_pass"] is True
            assert cell["operators"][arm]["shuffle_pass"] is True
            assert cell["operators"][arm]["action_pass"] is True
            assert cell["continuous_operator_maximum_errors"][arm] <= 1e-12
        assert cell["all_frame_vs_recent"]["material_dominance_pass"] is True
        assert cell["all_frame_vs_recent"]["rmse_ratio"] < 0.50


def test_main_writes_strict_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "recent_degree2_fixed_ceiling_seed_pass_count": 4,
            "all_frame_degree2_fixed_ceiling_seed_pass_count": 5,
            "all_frame_material_dominance_seed_pass_count": 5,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
