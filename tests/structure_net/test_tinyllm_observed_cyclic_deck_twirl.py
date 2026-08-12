from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import torch

import experiments.structure_net.tinyllm_observed_cyclic_deck_twirl as study


def _small_config(**changes: object) -> study.ObservedCyclicDeckConfig:
    values: dict[str, object] = {
        "degrees": (2, 3),
        "seeds": (7,),
        "evaluation_orbits": 8,
        "map_points": 24,
        "required_seed_passes": 1,
        "maximum_control_seed_passes": 1,
        "device": "cpu",
        "allow_underpowered": True,
    }
    values.update(changes)
    return study.ObservedCyclicDeckConfig(**values)


def test_primary_configuration_and_preregistration_are_locked() -> None:
    config = study.ObservedCyclicDeckConfig()
    assert config.degrees == (2, 3)
    assert config.seeds == (7, 17, 29, 41, 53)
    assert config.evaluation_orbits == 256
    assert config.map_points == 192
    assert config.required_seed_passes == 4
    assert config.maximum_control_seed_passes == 1
    assert config.front_cut_tolerance == 1
    assert study._sha256(study.PREREGISTRATION_PATH) == study.PREREGISTRATION_SHA256


def test_source_campaigns_match_locked_evidence() -> None:
    config = _small_config()
    task, ladder_campaign, deck_campaign = study._validate_source_campaigns(config)
    assert ladder_campaign["implementation_sha256"] == study.SOURCE_LADDER_IMPLEMENTATION_SHA256
    assert deck_campaign["implementation_sha256"] == study.SOURCE_DECK_IMPLEMENTATION_SHA256
    assert ladder_campaign["task_config"]["phase_bins"] == task.phase_bins


def test_observed_rotation_closes_and_preserves_calibrated_norm() -> None:
    task = study.CircleTaskConfig()
    sensor = torch.tensor(
        [[[0.4, -0.1, 0.7], [0.2, 0.5, -0.3]] + [[0.1, 0.2, 0.0]] * 6],
        dtype=torch.float32,
    )
    calibration = torch.tensor(
        [[1.0, 0.0, 1.0, 1.2, 0.1, -0.2, 0.03, -0.04]],
        dtype=torch.float32,
    )
    original = sensor.clone()
    value = sensor
    for _ in range(3):
        value = study.observed_cyclic_action(
            value, calibration, task, 2.0 * math.pi / 3.0
        )
    assert torch.max(torch.abs(value - original)).item() <= 3e-7
    rotated = study.observed_cyclic_action(sensor, calibration, task, math.pi / 3.0)
    before = study._corrected_planar(sensor, calibration, task).norm(dim=-1)
    after = study._corrected_planar(rotated, calibration, task).norm(dim=-1)
    assert torch.max(torch.abs(before - after)).item() <= 1e-7
    assert torch.equal(rotated[..., 2], sensor[..., 2])


def test_orbit_constructor_uses_anchor_not_generator_mates() -> None:
    task = study.CircleTaskConfig()
    exact = study.deck.generate_exact_orbits(
        task, k=3, orbit_count=5, seed=101, regime="composition"
    )
    first = study.construct_observed_orbit(exact, task, angle_offset=0.0)
    changed_sensor = exact.sensor.clone().reshape(5, 3, *exact.sensor.shape[1:])
    changed_sensor[:, 1:] += 100.0
    changed = replace(exact, sensor=changed_sensor.reshape_as(exact.sensor))
    second = study.construct_observed_orbit(changed, task, angle_offset=0.0)
    assert torch.equal(first.sensor, second.sensor)
    assert torch.equal(first.calibration, second.calibration)
    assert torch.equal(first.input_ids, second.input_ids)


def test_correct_and_half_turn_action_contracts_pass_for_c2_and_c3() -> None:
    task = study.CircleTaskConfig()
    config = _small_config()
    for k in (2, 3):
        cohorts, hashes, contract = study._generate_cohorts(task, config, k, 7)
        assert contract["pass"] is True
        assert len(hashes) == 8
        assert all(contract[regime][family]["pass"] for regime in study.REGIMES for family in ("task", "map"))
        correct = cohorts["composition"]["correct_task"]
        control = cohorts["composition"]["control_task"]
        expected = torch.remainder(correct.target_bins + task.phase_bins // 2, task.phase_bins)
        assert torch.equal(control.target_bins, expected)


def test_campaign_classification_requires_both_degrees() -> None:
    config = study.ObservedCyclicDeckConfig()
    passed = {
        "2": {"all_primary_gates_pass": True},
        "3": {"all_primary_gates_pass": True},
    }
    assert study._classify_campaign(valid=True, degrees=passed, config=config) == (
        "observed_cyclic_deck_front_reproduced",
        True,
    )
    c2_only = {
        "2": {"all_primary_gates_pass": True},
        "3": {
            "all_primary_gates_pass": False,
            "mature_observed_twirl_pass_count": 3,
        },
    }
    assert study._classify_campaign(valid=True, degrees=c2_only, config=config) == (
        "c2_only_observed_front",
        False,
    )


def test_strict_json_helpers_and_source_reference() -> None:
    config = _small_config()
    reference, path, digest = study._reference_result(config, 2, 7)
    assert reference["status"] == "completed"
    assert path.is_file()
    assert digest == study._sha256(path)
    assert json.loads(json.dumps(study._json_config(config), allow_nan=False))


def test_forbidden_inputs_are_explicit() -> None:
    task = study.CircleTaskConfig()
    config = _small_config(degrees=(2,))
    _, _, contract = study._generate_cohorts(task, config, 2, 7)
    assert {
        "latent_phase",
        "target_posterior",
        "target_bin",
        "branch",
        "quotient_phase",
        "fiber_id",
        "independently_generated_orbit_row",
    } == set(contract["forbidden_inputs"])
    assert Path(study.PREREGISTRATION_PATH).is_file()
