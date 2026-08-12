from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from experiments.structure_net import (
    tinyllm_c3_temporal_fixed_operator_ceiling as study,
)


PRIMARY_RESULT_SHA256 = (
    "9b03a0cf19ef820586e2a031d80a694d498655de151f70f245c3c82b0abb853a"
)


def test_sources_and_preregistration_are_frozen() -> None:
    hashes = study.validate_sources()
    assert hashes == {
        "preregistration": study.PREREGISTRATION_SHA256,
        "generator": study.GENERATOR_SHA256,
        "interval_likelihood": study.INTERVAL_SHA256,
        "sensor_campaign": study.SENSOR_CAMPAIGN_SHA256,
        "sensor_mechanism_result": study.MECHANISM_RESULT_SHA256,
    }


def test_mean_increment_operator_is_exact_for_constant_group_speed() -> None:
    phase = torch.linspace(-math.pi, math.pi, 257, dtype=torch.float64)
    speed = torch.linspace(-0.20, 0.20, 257, dtype=torch.float64)
    time = torch.arange(8, dtype=torch.float64)[None]
    angle = 3.0 * (phase[:, None] + speed[:, None] * time)
    carrier = torch.polar(torch.ones_like(angle), angle).to(torch.complex128)
    target = torch.cos(3.0 * (phase + 8.0 * speed))
    baseline = study.source.temporal_prediction(carrier)
    candidate = study.mean_increment_prediction(carrier)
    assert float((baseline - target).abs().max()) <= 1e-12
    assert float((candidate - target).abs().max()) <= 1e-12


def test_sattolo_control_is_deterministic_permutation_without_fixed_points() -> None:
    first = study.sattolo_derangement(4096, 631_007)
    second = study.sattolo_derangement(4096, 631_007)
    assert torch.equal(first, second)
    assert torch.equal(torch.sort(first).values, torch.arange(4096))
    assert not torch.any(first == torch.arange(4096))


def test_task_gate_is_joint_and_threshold_inclusive() -> None:
    composition = {
        "posterior_mean_correlation": 0.90,
        "exact_bin_accuracy": 0.50,
        "target_cross_entropy": 1.80,
        "predicted_bin_coverage": 14,
    }
    extrapolation = {
        "posterior_mean_correlation": 0.90,
        "exact_bin_accuracy": 0.35,
        "target_cross_entropy": 2.20,
        "predicted_bin_coverage": 12,
    }
    assert study._task_gate(composition, "composition") is True
    assert study._task_gate(extrapolation, "extrapolation") is True
    extrapolation["target_cross_entropy"] = 2.200001
    assert study._task_gate(extrapolation, "extrapolation") is False


def test_population_classifications_are_locked() -> None:
    def cells(material: int, ceiling: int) -> list[dict]:
        values = []
        for index, seed in enumerate(study.SEEDS):
            for regime in study.REGIMES:
                values.append(
                    {
                        "seed": seed,
                        "regime": regime,
                        "comparison": {
                            "material_dominance_pass": index < material
                        },
                        "baseline_ceiling_pass": index < ceiling,
                    }
                )
        return values

    assert study.classify(cells(4, 5), True)["classification"] == (
        "fixed_multistep_group_operator_dominates_last_step"
    )
    assert study.classify(cells(3, 4), True)["classification"] == (
        "last_step_already_at_declared_fixed_operator_ceiling"
    )
    assert study.classify(cells(3, 3), True)["classification"] == (
        "fixed_operator_family_leaves_typed_residual_scope"
    )
    assert study.classify(cells(5, 5), False)["classification"] == (
        "invalid_fixed_operator_ceiling_contract"
    )


def test_primary_result_preserves_all_ten_registered_cells() -> None:
    result = json.loads(study.PRIMARY_RESULT_PATH.read_text(encoding="utf-8"))
    assert study._sha256(study.PRIMARY_RESULT_PATH) == PRIMARY_RESULT_SHA256
    assert result["status"] == "completed"
    assert result["source_hashes"]["runner"] == (
        "9471ffe7f319d3d53234fd795fc48ce39a7717970d0e191ae2882343ad6d3b37"
    )
    assert result["aggregates"] == {
        "baseline_ceiling_seed_pass_count": 5,
        "classification": "fixed_multistep_group_operator_dominates_last_step",
        "material_dominance_seed_pass_count": 5,
        "required_seed_passes": 4,
        "same_task_unrestricted_tinyllm_training_licensed": False,
        "typed_residual_function_class_preflight_licensed": False,
        "valid": True,
    }
    assert len(result["cells"]) == 10
    for cell in result["cells"]:
        assert cell["valid"] is True
        assert cell["dataset_deterministic"] is True
        assert cell["saturation_count"] == 0
        assert cell["shuffle_fixed_points"] == 0
        assert cell["comparison"]["material_dominance_pass"] is True
        assert cell["comparison"]["rmse_ratio"] <= 0.75
        for error in cell["continuous_positive_control_maximum_errors"].values():
            assert error <= 1e-12
        for operator in cell["operators"].values():
            assert operator["task_pass"] is True
            assert operator["action_pass"] is True
            assert operator["shuffle_pass"] is True


def test_main_writes_strict_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {
        "status": "completed",
        "aggregates": {
            "classification": "test_classification",
            "material_dominance_seed_pass_count": 4,
            "baseline_ceiling_seed_pass_count": 5,
        },
    }
    monkeypatch.setattr(study, "build_result", lambda: expected)
    output = tmp_path / "result.json"
    assert study.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
