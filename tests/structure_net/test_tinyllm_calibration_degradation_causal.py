from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.structure_net import tinyllm_calibration_degradation_causal as curve


def _config() -> curve.CalibrationDegradationConfig:
    return curve.CalibrationDegradationConfig(
        seeds=(7,),
        conditions=("analytic_calibrated",),
        required_seed_passes=1,
        device="cpu",
        allow_underpowered=True,
    )


def test_primary_configuration_is_locked() -> None:
    config = curve.CalibrationDegradationConfig(device="cpu")
    assert config.conditions == (
        "analytic_calibrated",
        "learned_calibrated_equivariant",
    )
    assert config.seeds == (7, 17, 29, 41, 53)
    assert config.noise_levels == (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)
    assert config.target_robust_radius == pytest.approx(0.20)
    assert config.required_seed_passes == 4
    assert config.task_accuracy_drop_ceiling == pytest.approx(0.03)
    with pytest.raises(ValueError, match="primary seeds"):
        replace(config, seeds=(7,))
    with pytest.raises(ValueError, match="noise grid"):
        replace(config, noise_levels=(0.0, 0.1))


def test_zero_noise_is_exact_and_nested_noise_is_typed() -> None:
    config = _config()
    calibration = torch.tensor(
        [
            [1.0, 0.0, 0.35, 1.0, 0.1, -0.1, 0.02, -0.02],
            [0.0, 1.0, -0.50, 1.5, -0.2, 0.2, -0.04, 0.04],
        ]
    )
    noise = torch.tensor(
        [
            [1.0, -1.0, 0.5, 1.0, -1.0, 0.5, -0.5],
            [-1.0, 1.0, -0.5, -1.0, 1.0, -0.5, 0.5],
        ]
    )
    zero = curve.perturb_calibration(calibration, noise, 0.0, config)
    torch.testing.assert_close(zero, calibration, atol=0, rtol=0)
    low = curve.perturb_calibration(calibration, noise, 0.1, config)
    high = curve.perturb_calibration(calibration, noise, 0.2, config)
    assert torch.allclose(low[:, :2].norm(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(high[:, :2].norm(dim=1), torch.ones(2), atol=1e-6)
    assert torch.all(high[:, 3] > 0)
    assert torch.linalg.vector_norm(high - calibration) > torch.linalg.vector_norm(
        low - calibration
    )


@pytest.mark.parametrize(
    ("name", "columns", "expected"),
    [
        ("orientation_default", (0, 1), (1.0, 0.0)),
        ("speed_default", (2,), (0.35,)),
        ("amplitude_default", (3,), (1.0,)),
        ("offset_default", (4, 5), (0.0, 0.0)),
        ("drift_default", (6, 7), (0.0, 0.0)),
    ],
)
def test_field_ablation_changes_only_declared_group(
    name: str, columns: tuple[int, ...], expected: tuple[float, ...]
) -> None:
    source = torch.tensor(
        [[0.8, 0.6, -0.4, 1.3, 0.2, -0.1, 0.05, -0.03]]
    )
    value = curve.ablate_calibration(source, name)
    for column, target in zip(columns, expected):
        assert float(value[0, column]) == pytest.approx(target)
    untouched = sorted(set(range(8)).difference(columns))
    torch.testing.assert_close(value[:, untouched], source[:, untouched])


def test_all_default_replaces_the_entire_packet() -> None:
    source = torch.randn(3, 8)
    value = curve.ablate_calibration(source, "all_default")
    expected = torch.tensor([1.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(value, expected.repeat(3, 1))


def test_joint_endpoint_checks_all_three_metrics() -> None:
    config = _config()
    metrics = {
        "cosine_pearson": 0.95,
        "balanced_accuracy": 0.52,
        "conditional_log_loss_gain_over_cosine_only": 0.01,
    }
    assert curve.endpoint_pass(metrics, config)
    for key, value in (
        ("cosine_pearson", 0.89),
        ("balanced_accuracy", 0.56),
        ("conditional_log_loss_gain_over_cosine_only", 0.021),
    ):
        changed = dict(metrics)
        changed[key] = value
        assert not curve.endpoint_pass(changed, config)


def test_activation_drift_identity_and_change() -> None:
    rng = np.random.default_rng(7)
    clean = rng.normal(size=(32, 6))
    identity = curve.activation_drift(clean, clean.copy(), 16)
    assert identity["mean_row_cosine_to_clean"] == pytest.approx(1.0)
    assert identity["normalized_rmse_from_clean"] == pytest.approx(0.0)
    assert identity["centered_linear_cka_to_clean"] == pytest.approx(1.0)
    changed = curve.activation_drift(clean, clean + 0.5 * rng.normal(size=clean.shape), 16)
    assert changed["normalized_rmse_from_clean"] > 0.0
    assert changed["centered_linear_cka_to_clean"] < 1.0


def test_robust_radius_requires_a_consecutive_prefix() -> None:
    levels = (0.0, 0.05, 0.10, 0.20)
    counts = {curve.noise_key(x): value for x, value in zip(levels, (5, 5, 3, 5))}
    assert curve.robust_radius(counts, levels, 4) == pytest.approx(0.05)


@pytest.mark.parametrize(
    (
        "valid",
        "analytic",
        "learned",
        "analytic_fail",
        "learned_fail",
        "expected",
        "passed",
    ),
    [
        (False, 0.4, 0.4, 5, 5, "invalid", False),
        (
            True,
            0.4,
            0.2,
            5,
            5,
            "learned_calibration_robustness_tracks_analytic_control",
            True,
        ),
        (True, 0.4, 0.1, 5, 5, "learned_calibration_brittle", False),
        (True, 0.1, 0.1, 5, 5, "exact_calibration_required", False),
        (
            True,
            0.05,
            0.4,
            5,
            5,
            "learned_more_robust_than_analytic_control",
            False,
        ),
        (True, 0.1, 0.2, 5, 5, "analytic_canonicalizer_brittle", False),
        (True, 0.4, 0.2, 3, 5, "mixed_calibration_robustness", False),
    ],
)
def test_campaign_classification_is_fixed(
    valid: bool,
    analytic: float,
    learned: float,
    analytic_fail: int,
    learned_fail: int,
    expected: str,
    passed: bool,
) -> None:
    actual = curve.campaign_classification(
        valid=valid,
        analytic_radius=analytic,
        learned_radius=learned,
        target_radius=0.2,
        analytic_default_fail_count=analytic_fail,
        learned_default_fail_count=learned_fail,
        required_default_failures=4,
        levels=curve.NOISE_LEVELS,
    )
    assert actual == (expected, passed)


def test_locked_source_manifest_is_authoritative() -> None:
    campaign, path, manifest, records = curve._load_source_campaign(_config())
    assert curve._sha256(path) == curve.SOURCE_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == curve.SOURCE_IMPLEMENTATION_SHA256
    assert manifest == curve.SOURCE_MANIFEST_SHA256
    assert len(records) == 30


def test_completed_result_resume_checks_activation_hash(tmp_path: Path) -> None:
    config = _config()
    result_path = tmp_path / "runs" / "analytic_calibrated" / "seed_7" / "result.json"
    arrays_path = result_path.with_name("activation_audit.npz")
    result_path.parent.mkdir(parents=True)
    arrays_path.write_bytes(b"fixed activations")
    fingerprint = curve._fingerprint(config, "analytic_calibrated", 7, "source")
    record = {
        "schema_version": curve.SCHEMA_VERSION,
        "hypothesis_id": curve.HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": curve._evidence_role(config),
        "implementation_sha256": "implementation",
        "configuration": json.loads(json.dumps(asdict(config))),
        "condition": "analytic_calibrated",
        "seed": 7,
        "scientific_fingerprint": fingerprint,
        "artifacts": {
            "result": str(result_path),
            "activations": str(arrays_path),
            "activations_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        },
    }
    result_path.write_text(json.dumps(record), encoding="utf-8")
    assert curve._reusable_result(
        result_path,
        arrays_path,
        config,
        "implementation",
        "analytic_calibrated",
        7,
        fingerprint,
    ) == record
    arrays_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="incompatible completed result"):
        curve._reusable_result(
            result_path,
            arrays_path,
            config,
            "implementation",
            "analytic_calibrated",
            7,
            fingerprint,
        )
