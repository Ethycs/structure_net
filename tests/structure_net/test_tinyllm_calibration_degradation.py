from __future__ import annotations

from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from experiments.structure_net import tinyllm_calibration_degradation as degradation


def _small_config() -> degradation.CalibrationDegradationConfig:
    return degradation.CalibrationDegradationConfig(
        seeds=(7,),
        noise_levels=(0.0, 0.5, 4.0),
        required_seed_passes=1,
        device="cpu",
        allow_underpowered=True,
    )


def test_primary_configuration_is_locked() -> None:
    config = degradation.CalibrationDegradationConfig(device="cpu")
    assert config.seeds == (7, 17, 29, 41, 53)
    assert config.conditions == degradation.CONDITIONS
    assert config.noise_levels == degradation.PRIMARY_LEVELS
    assert config.required_seed_passes == 4
    assert config.orientation_std_radians == pytest.approx(0.15)
    assert config.log_speed_std == pytest.approx(np.log(1.25))
    with pytest.raises(ValueError, match="primary seeds"):
        replace(config, seeds=(7, 17, 29, 41, 59))
    with pytest.raises(ValueError, match="primary noise levels"):
        replace(config, noise_levels=(0.0, 1.0, 4.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(
            config,
            noise_levels=(0.0, 0.5, 0.5),
            allow_underpowered=True,
        )


def _calibration(count: int = 128) -> torch.Tensor:
    angle = torch.linspace(-0.3, 0.3, count)
    return torch.stack(
        (
            angle.cos(),
            angle.sin(),
            torch.where(torch.arange(count) % 2 == 0, 0.35, -0.35),
            torch.linspace(0.75, 1.25, count),
            torch.linspace(-0.1, 0.1, count),
            torch.linspace(0.1, -0.1, count),
            torch.zeros(count),
            torch.zeros(count),
        ),
        dim=1,
    )


def test_zero_corruption_is_bitwise_identity() -> None:
    calibration = _calibration()
    noise = torch.randn(len(calibration), 7, generator=torch.Generator().manual_seed(9))
    result = degradation.corrupt_calibration(calibration, noise, 0.0, _small_config())
    assert torch.equal(result, calibration)
    assert result.data_ptr() != calibration.data_ptr()


def test_typed_corruption_preserves_declared_packet_constraints() -> None:
    calibration = _calibration()
    noise = torch.randn(
        len(calibration), 7, generator=torch.Generator().manual_seed(19)
    )
    result = degradation.corrupt_calibration(calibration, noise, 1.0, _small_config())
    assert torch.allclose(result[:, :2].norm(dim=1), torch.ones(len(result)), atol=1e-6)
    assert torch.equal(result[:, 2].sign(), calibration[:, 2].sign())
    assert bool((result[:, 3] > 0.0).all())
    assert not torch.equal(result, calibration)
    metrics = degradation.corruption_metrics(calibration, result)
    assert metrics["orientation_rms_radians"] > 0.0
    assert metrics["log_speed_rms"] > 0.0
    assert metrics["log_amplitude_rms"] > 0.0
    assert metrics["orientation_unit_norm_max_error"] < 2e-7


def test_nested_noise_curve_scales_the_same_error_field() -> None:
    calibration = _calibration(2_048)
    noise = torch.randn(
        len(calibration), 7, generator=torch.Generator().manual_seed(29)
    )
    first = degradation.corrupt_calibration(calibration, noise, 0.5, _small_config())
    second = degradation.corrupt_calibration(calibration, noise, 1.0, _small_config())
    first_metrics = degradation.corruption_metrics(calibration, first)
    second_metrics = degradation.corruption_metrics(calibration, second)
    for name in (
        "orientation_rms_radians",
        "log_speed_rms",
        "log_amplitude_rms",
        "offset_rms_sensor_units",
        "drift_rms_sensor_units",
    ):
        assert second_metrics[name] == pytest.approx(2.0 * first_metrics[name], rel=2e-5)


def test_breakpoint_summary_detects_bounded_reentry_and_censoring() -> None:
    levels = (0.0, 0.25, 0.5, 1.0)
    stable = degradation.breakpoint_summary((True, True, False, False), levels)
    assert stable["breakpoint_interval"] == [0.25, 0.5]
    assert stable["gate_reentry"] is False
    assert stable["right_censored"] is False
    reentry = degradation.breakpoint_summary((True, False, True, False), levels)
    assert reentry["gate_reentry"] is True
    censored = degradation.breakpoint_summary((True, True, True, True), levels)
    assert censored["right_censored"] is True
    assert censored["last_passing_level"] == 1.0


def _analysis(passed: bool, cosine: float | None = None) -> dict[str, object]:
    value = 0.95 if passed else 0.80
    if cosine is not None:
        value = cosine
    cuts = {}
    for cut in degradation.CUTS:
        evaluations = {}
        for regime in degradation.REGIMES:
            evaluations[regime] = {
                "cosine_pearson": value,
                "balanced_accuracy": 0.50,
                "conditional_log_loss_gain_over_cosine_only": 0.0,
            }
        cuts[cut] = {"probe": {"evaluations": evaluations}}
    task = {
        regime: {
            "exact_bin_accuracy": 0.5,
            "mean_circular_error_radians": 0.2,
            "mean_target_cross_entropy": 1.0,
        }
        for regime in degradation.REGIMES
    }
    return {"cuts": cuts, "task_metrics": task}


def test_joint_seed_gate_requires_every_primary_cut_and_shift() -> None:
    analysis = _analysis(True)
    assert degradation.joint_seed_pass(analysis)
    analysis["cuts"]["full"]["probe"]["evaluations"]["extrapolation"][
        "balanced_accuracy"
    ] = 0.551
    assert not degradation.joint_seed_pass(analysis)


def _result(
    condition: str,
    seed: int,
    passes: tuple[bool, bool, bool],
    shuffled_pass: bool = False,
) -> dict[str, object]:
    levels = (0.0, 0.5, 4.0)
    evaluations = {
        degradation._level_key(level): {
            "joint_seed_pass": passed,
            "analysis": _analysis(passed),
        }
        for level, passed in zip(levels, passes)
    }
    evaluations["shuffled"] = {
        "joint_seed_pass": shuffled_pass,
        "analysis": _analysis(shuffled_pass),
    }
    return {
        "condition": condition,
        "seed": seed,
        "evaluations": evaluations,
        "gates": {
            "source_provenance_contract": True,
            "source_checkpoint_state_contract": True,
            "zero_calibration_identity_contract": True,
            "zero_source_metric_replay_contract": True,
            "finite_numerical_contract": True,
        },
    }


def test_aggregate_requires_bounded_stable_breakpoint_in_both_arms() -> None:
    config = _small_config()
    results = [
        _result(condition, 7, (True, True, False))
        for condition in degradation.CONDITIONS
    ]
    aggregate = degradation.aggregate_results(results, config)
    assert aggregate["validity_contract"] is True
    assert aggregate["supported"] is False
    assert aggregate["conclusion"] == "systems_lifecycle_only_not_quality_evidence"
    for arm in aggregate["arms"].values():
        assert arm["bounded_stable_breakpoint"] is True
        assert arm["breakpoint"]["breakpoint_interval"] == [0.5, 4.0]


def test_source_campaign_and_cell_provenance_are_locked() -> None:
    config = _small_config()
    campaign, path, task, frontend_config = degradation._load_source_campaign(config)
    detail, provenance = degradation._source_provenance(
        config, "analytic_calibrated", 7
    )
    assert degradation._sha256(path) == degradation.SOURCE_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == degradation.SOURCE_IMPLEMENTATION_SHA256
    assert task.phase_bins == 16
    assert frontend_config.analysis_seed == 83
    assert detail["seed"] == 7
    assert len(provenance["model_checkpoint_sha256"]) == 64


def test_noise_fields_are_deterministic_and_permutations_are_bijections() -> None:
    _, _, task, _ = degradation._load_source_campaign(_small_config())
    datasets = degradation._datasets(task)
    first_noise, first_permutations = degradation._noise_fields(datasets, 123)
    second_noise, second_permutations = degradation._noise_fields(datasets, 123)
    for name, dataset in datasets.items():
        assert torch.equal(first_noise[name], second_noise[name])
        assert torch.equal(first_permutations[name], second_permutations[name])
        assert torch.equal(
            torch.sort(first_permutations[name]).values,
            torch.arange(len(dataset.calibration)),
        )
        assert not torch.equal(
            first_permutations[name], torch.arange(len(dataset.calibration))
        )


def test_result_resume_requires_exact_fingerprint() -> None:
    config = _small_config()
    record = {
        "schema_version": degradation.SCHEMA_VERSION,
        "hypothesis_id": degradation.HYPOTHESIS_ID,
        "status": "completed",
        "condition": "analytic_calibrated",
        "seed": 7,
        "configuration": json.loads(json.dumps(asdict(config))),
        "implementation_sha256": "implementation",
        "scientific_fingerprint": "fingerprint",
    }
    assert degradation._result_reusable(
        record,
        config,
        "analytic_calibrated",
        7,
        "implementation",
        "fingerprint",
    )
    assert not degradation._result_reusable(
        record,
        config,
        "analytic_calibrated",
        7,
        "implementation",
        "changed",
    )
