from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import tinyllm_calibration_orientation_noise as noise


def _underpowered() -> noise.OrientationNoiseConfig:
    return noise.OrientationNoiseConfig(
        seeds=(7, 17),
        sigmas=(0.0, 0.035, 0.087),
        required_seed_passes=2,
        device="cpu",
        allow_underpowered=True,
    )


def test_primary_protocol_is_locked() -> None:
    config = noise.OrientationNoiseConfig(device="cpu")
    assert config.seeds == (7, 17, 29, 41, 53)
    assert config.sigmas == (0.0, 0.035, 0.087, 0.175, 0.349, 0.524, 0.785)
    assert config.analysis_seed == 19_083
    assert config.required_seed_passes == 4
    with pytest.raises(ValueError, match="seeds are fixed"):
        replace(config, seeds=(7,))
    with pytest.raises(ValueError, match="grid is fixed"):
        replace(config, sigmas=(0.0, 0.1))


def test_pair_shared_corruption_is_unit_norm_and_scales_exactly() -> None:
    fiber_ids = torch.tensor([4, 4, 9, 9, 15, 15])
    dataset = SimpleNamespace(
        paired=SimpleNamespace(fiber=SimpleNamespace(fiber_id=fiber_ids))
    )
    base = noise.build_base_noise(dataset, 19_390)
    assert torch.equal(base[0::2], base[1::2])
    calibration = torch.zeros(6, 8)
    angles = torch.tensor([0.1, 0.1, -0.2, -0.2, 0.7, 0.7])
    calibration[:, 0] = torch.cos(angles)
    calibration[:, 1] = torch.sin(angles)
    calibration[:, 2] = 0.3
    calibration[:, 3] = 1.0
    clean, clean_audit = noise.corrupt_orientation(calibration, base, 0.0)
    assert torch.equal(clean, calibration)
    assert clean_audit["maximum_formula_replay_error"] == 0.0
    corrupted, audit = noise.corrupt_orientation(calibration, base, 0.175)
    assert audit["maximum_unit_norm_error"] <= 1e-6
    assert audit["maximum_formula_replay_error"] <= 1e-7
    assert noise.pair_shared_error(0.175 * base, fiber_ids) == 0.0
    observed = torch.atan2(corrupted[:, 1].double(), corrupted[:, 0].double())
    delta = torch.atan2(torch.sin(observed - angles), torch.cos(observed - angles))
    assert torch.allclose(delta, 0.175 * base, atol=1e-7, rtol=0.0)


def test_representation_gate_includes_conditional_log_loss() -> None:
    config = _underpowered()
    metrics = {
        "cosine_pearson": 0.95,
        "balanced_accuracy": 0.52,
        "conditional_log_loss_gain_over_cosine_only": 0.01,
    }
    assert noise.representation_pass(metrics, config)
    metrics["conditional_log_loss_gain_over_cosine_only"] = 0.021
    assert not noise.representation_pass(metrics, config)


def _result(condition: str, seed: int, pattern: tuple[bool, ...]) -> dict[str, object]:
    return {
        "condition": condition,
        "seed": seed,
        "levels": [{"complete_gate": value} for value in pattern],
    }


def test_curve_summary_does_not_smooth_nonmonotonicity() -> None:
    config = _underpowered()
    results = [
        _result("analytic_calibrated", 7, (True, False, True)),
        _result("analytic_calibrated", 17, (True, False, True)),
    ]
    summary = noise.curve_summary(results, "analytic_calibrated", config)
    assert summary["nonmonotone"] is True
    assert summary["robustness_radius_radians"] is None


@pytest.mark.parametrize(
    ("analytic", "learned", "expected"),
    [
        (0.175, 0.175, "learned_matches_analytic_radius"),
        (0.175, 0.087, "learned_brittle_relative_to_analytic"),
        (0.175, 0.0, "learned_clean_only_equivariance"),
        (0.0, 0.0, "reference_precision_critical"),
    ],
)
def test_campaign_classifications_are_locked(
    analytic: float, learned: float, expected: str
) -> None:
    arms = {
        "analytic_calibrated": {
            "nonmonotone": False,
            "robustness_radius_radians": analytic,
        },
        "learned_calibrated_equivariant": {
            "nonmonotone": False,
            "robustness_radius_radians": learned,
        },
    }
    assert noise.classify_campaign(arms, True) == expected
    assert noise.classify_campaign(arms, False) == "invalid"


def test_locked_source_campaign_replays() -> None:
    campaign, path, task, source_config = noise._load_source_campaign(
        noise.OrientationNoiseConfig(
            seeds=(7,),
            sigmas=(0.0,),
            required_seed_passes=1,
            device="cpu",
            allow_underpowered=True,
        )
    )
    assert noise._sha256(path) == noise.SOURCE_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == noise.SOURCE_IMPLEMENTATION_SHA256
    assert task.phase_bins == 16
    assert source_config.seeds == (7, 17, 29, 41, 53)
