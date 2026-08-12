from __future__ import annotations

from dataclasses import replace
import math

import torch

import experiments.structure_net.tinyllm_observed_cyclic_action_semantics as study


def _small_config(**changes: object) -> study.ActionSemanticsConfig:
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
    return study.ActionSemanticsConfig(**values)


def test_primary_configuration_and_preregistration_are_locked() -> None:
    config = study.ActionSemanticsConfig()
    assert config.degrees == (2, 3)
    assert config.seeds == (7, 17, 29, 41, 53)
    assert config.minimum_distance_reduction == 0.50
    assert config.character_median_ceiling == 0.05
    assert config.character_p95_ceiling == 0.20
    assert config.forced_variant is None
    assert study._sha256(study.PREREGISTRATION_PATH) == study.PREREGISTRATION_SHA256


def test_locked_source_campaigns_validate() -> None:
    task, _, _, observed = study._validate_sources(_small_config())
    assert task.phase_bins == 16
    assert observed["aggregates"]["classification"] == "c2_only_observed_front"
    assert observed["aggregates"]["degrees"]["3"]["front_replication_pass_count"] == 2


def test_full_history_demodulation_recovers_noiseless_signal() -> None:
    task = study.CircleTaskConfig()
    phase = 0.73
    orientation_angle = -0.21
    omega = 0.37
    amplitude = 1.18
    orientation = torch.tensor(
        [[math.cos(orientation_angle), math.sin(orientation_angle)]],
        dtype=torch.float32,
    )
    calibration = torch.tensor(
        [[orientation[0, 0], orientation[0, 1], omega, amplitude, 0.1, -0.08, 0.03, -0.02]],
        dtype=torch.float32,
    )
    step = torch.arange(task.sensor_steps, dtype=torch.float32) - (task.sensor_steps - 1)
    angle = phase + omega * step
    perpendicular = torch.tensor(
        [[-orientation[0, 1], orientation[0, 0]]], dtype=torch.float32
    )
    physical = amplitude * (
        orientation[:, None] * torch.cos(angle)[None, :, None]
        + perpendicular[:, None] * torch.sin(angle)[None, :, None]
    )
    baseline = study._calibrated_baseline(calibration, task, physical)
    sensor = torch.zeros((1, task.sensor_steps, 3), dtype=torch.float32)
    sensor[..., :2] = baseline + physical
    clean, residual, current = study._observed_coherent_signal(
        sensor, calibration, task
    )
    assert torch.max(torch.abs(clean - physical)).item() <= 3e-6
    assert torch.max(torch.abs(residual)).item() <= 3e-6
    target = torch.tensor([[math.cos(phase), math.sin(phase)]])
    assert torch.max(torch.abs(current - target)).item() <= 3e-6


def test_residual_fixed_action_closes_and_composes_under_declared_metric() -> None:
    task = study.CircleTaskConfig()
    exact = study.base.deck.generate_exact_orbits(
        task, k=3, orbit_count=5, seed=211, regime="composition"
    )
    sensor = study._anchor(exact.sensor, exact)
    calibration = study._anchor(exact.calibration, exact)
    alpha = 2.0 * math.pi / 3.0
    once = study.residual_fixed_action(sensor, calibration, task, alpha)
    twice = study.residual_fixed_action(once, calibration, task, alpha)
    direct = study.residual_fixed_action(sensor, calibration, task, 2.0 * alpha)
    closed = study.residual_fixed_action(sensor, calibration, task, 2.0 * math.pi)
    composition = study._per_row_relative_rms(
        twice[..., :2], direct[..., :2]
    )
    assert composition.max() <= 0.02
    assert torch.max(torch.abs(closed - sensor)).item() <= 5e-7


def test_observable_variant_constructor_ignores_nonanchor_rows() -> None:
    task = study.CircleTaskConfig()
    exact = study.base.deck.generate_exact_orbits(
        task, k=3, orbit_count=5, seed=307, regime="extrapolation"
    )
    first = study.construct_variant_orbit(
        exact,
        task,
        variant="residual_fixed_continuous",
        angle_offset=0.0,
    )
    changed_sensor = exact.sensor.clone().reshape(5, 3, *exact.sensor.shape[1:])
    changed_sensor[:, 1:] += 100.0
    changed = replace(exact, sensor=changed_sensor.reshape_as(exact.sensor))
    second = study.construct_variant_orbit(
        changed,
        task,
        variant="residual_fixed_continuous",
        angle_offset=0.0,
    )
    assert torch.equal(first.sensor, second.sensor)
    assert torch.equal(first.input_ids, second.input_ids)


def test_oracle_variants_cannot_be_forced() -> None:
    for variant in study.ORACLE_VARIANTS:
        try:
            _small_config(forced_variant=variant)
        except ValueError as error:
            assert "not selectable" in str(error)
        else:
            raise AssertionError("oracle variant was accepted as selectable")


def test_requantization_is_idempotent_on_decoded_grid() -> None:
    task = study.CircleTaskConfig()
    exact = study.base.deck.generate_exact_orbits(
        task, k=2, orbit_count=4, seed=101, regime="composition"
    )
    once = study.requantize_sensor(exact.sensor, task)
    twice = study.requantize_sensor(once, task)
    assert torch.equal(once, twice)


def test_small_stage_a_is_strict_and_keeps_oracles_nonselectable() -> None:
    task = study.CircleTaskConfig()
    stage, arrays = study.run_stage_a(task, _small_config())
    assert stage["finite"] is True
    assert set(stage["variants"]) == set(study.ALL_VARIANTS)
    assert all(
        not stage["variants"][variant]["eligible"]
        for variant in study.ORACLE_VARIANTS
    )
    assert arrays
    assert "latent_phase" in stage["forbidden_inputs"]
