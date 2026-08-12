from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.structure_net import tinyllm_observed_deck_twirl as twirl
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


def _underpowered_config(**changes: object) -> twirl.ObservedDeckTwirlConfig:
    values = {
        "conditions": ("analytic_calibrated",),
        "seeds": (7,),
        "required_seed_passes": 1,
        "maximum_control_seed_passes": 1,
        "device": "cpu",
        "allow_underpowered": True,
    }
    values.update(changes)
    return twirl.ObservedDeckTwirlConfig(**values)


def test_primary_configuration_and_preregistration_are_locked() -> None:
    config = twirl.ObservedDeckTwirlConfig()
    assert config.conditions == twirl.CONDITIONS
    assert config.seeds == twirl.SEEDS
    assert config.required_seed_passes == 4
    assert config.maximum_control_seed_passes == 1
    assert twirl._sha256(twirl.PREREGISTRATION_PATH) == (
        twirl.PREREGISTRATION_SHA256
    )
    with pytest.raises(ValueError, match="five checkpoints are fixed"):
        twirl.ObservedDeckTwirlConfig(seeds=(7, 17, 29, 41))
    with pytest.raises(ValueError, match="four of five"):
        twirl.ObservedDeckTwirlConfig(required_seed_passes=3)


def test_observed_action_is_involutive_and_uses_the_declared_axis() -> None:
    task = CircleTaskConfig()
    sensor = torch.zeros((2, task.sensor_steps, 3), dtype=torch.float32)
    sensor[:, :, 0] = torch.linspace(-0.8, 0.7, task.sensor_steps)
    sensor[:, :, 1] = torch.linspace(0.4, -0.6, task.sensor_steps)
    sensor[:, :, 2] = 0.37
    calibration = torch.zeros((2, 8), dtype=torch.float32)
    calibration[:, 0] = 1.0
    calibration[:, 2] = torch.tensor([0.35, -0.42])
    calibration[:, 3] = 1.0
    correct, correct_calibration = twirl.observed_deck_action(
        sensor, calibration, task
    )
    control, control_calibration = twirl.observed_deck_action(
        sensor, calibration, task, orthogonal_axis=True
    )
    assert torch.allclose(correct[..., 0], sensor[..., 0])
    assert torch.allclose(correct[..., 1], -sensor[..., 1])
    assert torch.allclose(control[..., 0], -sensor[..., 0])
    assert torch.allclose(control[..., 1], sensor[..., 1])
    assert torch.equal(correct[..., 2], sensor[..., 2])
    assert torch.equal(control[..., 2], sensor[..., 2])
    assert torch.equal(correct_calibration[:, 2], -calibration[:, 2])
    assert torch.equal(control_calibration[:, 2], -calibration[:, 2])
    restored, restored_calibration = twirl.observed_deck_action(
        correct, correct_calibration, task
    )
    assert torch.allclose(restored, sensor, atol=1e-6)
    assert torch.equal(restored_calibration, calibration)


def test_locked_cohorts_pass_the_observation_contract_without_model_loading() -> None:
    config = _underpowered_config()
    _, _, task, _, _, _, _ = twirl._load_sources(config)
    datasets = twirl.closure._datasets(task)
    contract = twirl.action_contract(datasets, task, config)
    assert contract["pass"]
    assert "fiber_id" in contract["forbidden_inputs"]
    assert "latent_phase" in contract["forbidden_inputs"]
    assert "independent_nuisance_draw" in contract["forbidden_inputs"]
    for regime in twirl.REGIMES:
        item = contract["regimes"][regime]
        assert item["target_cosine_maximum_error"] == 0.0
        assert item["oracle_mate_relative_rms"] >= 0.5
        assert item["analytic_feature_maximum_error"] <= 1e-6


@pytest.mark.parametrize(
    ("valid", "twirl_counts", "action_counts", "expected"),
    [
        (
            False,
            {condition: 5 for condition in twirl.CONDITIONS},
            {condition: 5 for condition in twirl.CONDITIONS},
            ("invalid", False),
        ),
        (
            True,
            {condition: 5 for condition in twirl.CONDITIONS},
            {condition: 5 for condition in twirl.CONDITIONS},
            ("observable_twirl_closed_action_invariant", True),
        ),
        (
            True,
            {condition: 5 for condition in twirl.CONDITIONS},
            {"analytic_calibrated": 5, "learned_calibrated_equivariant": 2},
            ("observable_twirl_closed_action_variant", True),
        ),
        (
            True,
            {"analytic_calibrated": 5, "learned_calibrated_equivariant": 2},
            {condition: 5 for condition in twirl.CONDITIONS},
            ("analytic_only_observable_twirl", False),
        ),
        (
            True,
            {"analytic_calibrated": 2, "learned_calibrated_equivariant": 5},
            {condition: 5 for condition in twirl.CONDITIONS},
            ("learned_only_observable_twirl", False),
        ),
        (
            True,
            {condition: 2 for condition in twirl.CONDITIONS},
            {condition: 2 for condition in twirl.CONDITIONS},
            ("observable_twirl_not_causally_sufficient", False),
        ),
    ],
)
def test_locked_classification_table(
    valid: bool,
    twirl_counts: dict[str, int],
    action_counts: dict[str, int],
    expected: tuple[str, bool],
) -> None:
    assert twirl.classify_campaign(
        valid=valid,
        twirl_pre_counts=twirl_counts,
        action_pre_counts=action_counts,
        config=twirl.ObservedDeckTwirlConfig(),
    ) == expected


def test_campaign_reuse_requires_every_expected_artifact(tmp_path: Path) -> None:
    config = _underpowered_config()
    result_path = tmp_path / "result.json"
    result_path.write_text("{}\n", encoding="utf-8")
    diagnostics_path = tmp_path / "diagnostics.npz"
    with diagnostics_path.open("wb") as handle:
        np.savez_compressed(handle, value=np.asarray([1]))
    implementation = "implementation"
    entry = {
        "condition": "analytic_calibrated",
        "seed": 7,
        "path": str(result_path),
        "result_sha256": twirl._sha256(result_path),
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": twirl._sha256(diagnostics_path),
    }
    campaign = {
        "status": "completed",
        "schema_version": twirl.SCHEMA_VERSION,
        "configuration": twirl._json_config(config),
        "implementation_sha256": implementation,
        "results": [entry],
    }
    assert twirl._campaign_reusable(campaign, config, implementation)
    campaign["results"] = []
    assert not twirl._campaign_reusable(campaign, config, implementation)
    campaign["results"] = [entry]
    result_path.write_text(json.dumps({"changed": True}), encoding="utf-8")
    assert not twirl._campaign_reusable(campaign, config, implementation)
