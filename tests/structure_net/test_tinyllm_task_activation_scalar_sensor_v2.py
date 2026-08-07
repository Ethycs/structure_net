from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import tinyllm_task_activation_scalar_sensor_v2 as sensor


def _config() -> sensor.TaskActivationScalarSensorConfig:
    return sensor.TaskActivationScalarSensorConfig(
        seeds=(7,), device="cpu", allow_underpowered=True
    )


def test_v2_supersedes_v1_and_uses_unseen_cohort_e() -> None:
    assert sensor.SUPERSEDES == (
        "tinyllm-c2-task-activation-scalar-sensor-v1",
        "tinyllm-c2-observable-scalar-residual-v1",
    )
    assert sensor.SCHEMA_VERSION.endswith(".v2")
    assert sensor.HYPOTHESIS_ID.endswith("-v2")


def test_primary_configuration_and_fresh_e_are_locked() -> None:
    config = sensor.TaskActivationScalarSensorConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    assert config.orbit_count == 64
    assert config.pca_rank == 8
    assert config.scalar_ridge == pytest.approx(1e-3)
    assert config.residual_mae_fraction_ceiling == pytest.approx(0.50)
    assert sensor.FRESH_COHORT_SEEDS == {
        "composition": 630_007,
        "extrapolation": 630_008,
    }
    old = {
        value
        for cohorts in sensor.fixed.COHORT_SEEDS.values()
        for value in cohorts.values()
    } | set(sensor.source.FRESH_COHORT_SEEDS.values()) | {530_007, 530_008}
    assert set(sensor.FRESH_COHORT_SEEDS.values()).isdisjoint(old)
    with pytest.raises(ValueError, match="64 exact orbits"):
        replace(config, orbit_count=32, allow_underpowered=True)
    with pytest.raises(ValueError, match="eight source PCA"):
        replace(config, pca_rank=4, allow_underpowered=True)


def test_underpowered_lifecycle_uses_disjoint_systems_cohort() -> None:
    pilot = _config()
    primary = sensor.TaskActivationScalarSensorConfig(device="cpu")
    assert sensor._evaluation_cohort(pilot) == "systems_f"
    assert sensor._evaluation_seeds(pilot) == {
        "composition": 730_007,
        "extrapolation": 730_008,
    }
    assert sensor._evaluation_cohort(primary) == "heldout_e"
    assert sensor._evaluation_seeds(primary) == sensor.FRESH_COHORT_SEEDS
    assert set(sensor.SYSTEMS_COHORT_SEEDS.values()).isdisjoint(
        sensor.FRESH_COHORT_SEEDS.values()
    )


def test_pca_is_orthonormal_and_source_transform_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(42)
    values = torch.randn(64, 12, generator=generator, dtype=torch.float64)
    first = sensor.fit_pca(values, 8)
    second = sensor.fit_pca(values, 8)
    torch.testing.assert_close(first["components"], second["components"])
    torch.testing.assert_close(
        first["components"] @ first["components"].T,
        torch.eye(8, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        sensor.apply_pca(values, first), sensor.apply_pca(values, second)
    )
    assert first["orthogonality_error"] < 1e-12
    assert 0.0 < first["cumulative_energy"] <= 1.0


def test_standardized_scalar_map_recovers_affine_target() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(128, 5, generator=generator, dtype=torch.float64)
    weights = torch.tensor([0.4, -1.2, 0.0, 2.0, 0.3], dtype=torch.float64)
    target = features @ weights + 0.75
    mapping = sensor.fit_scalar_map(features, target, ridge=1e-8)
    predicted = sensor.apply_scalar_map(features, mapping)
    torch.testing.assert_close(predicted, target, atol=1e-8, rtol=1e-8)


def test_tensor_digest_accepts_nonstandard_strides() -> None:
    value = torch.arange(18, dtype=torch.float64).reshape(2, 9).T
    assert value.stride(-1) != 1
    assert sensor._tensor_digest(value) == sensor._tensor_digest(value.contiguous())


def test_output_features_are_target_free_finite_and_have_declared_width() -> None:
    posterior = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], dtype=torch.float64
    )
    features = sensor.output_features(posterior)
    assert features.shape == (2, 9)
    assert torch.isfinite(features).all()
    torch.testing.assert_close(features[:, :3], posterior)


def test_feature_arms_keep_lookahead_out_of_primary() -> None:
    rows = 20
    raw = {
        "phase_only": torch.randn(rows, 9, dtype=torch.float64),
        "calibration": torch.randn(rows, 5, dtype=torch.float64),
        "prewrite_activation": torch.randn(rows, 12, dtype=torch.float64),
        "predicted_activation": torch.randn(rows, 12, dtype=torch.float64),
        "post_mlp_activation": torch.randn(rows, 12, dtype=torch.float64),
        "output": torch.randn(rows, 14, dtype=torch.float64),
    }
    pca = {
        name: sensor.fit_pca(raw[name], 8) for name in sensor.PCA_NAMES
    }
    arms = sensor.feature_arms(raw, pca)
    assert tuple(arms) == sensor.SCALAR_ARMS
    assert arms["causal_combined"].shape == (rows, 21)
    assert arms["post_mlp_lookahead"].shape == (rows, 29)
    assert arms["output_lookahead"].shape == (rows, 35)
    assert arms["full_lookahead"].shape == (rows, 43)


def test_prediction_gate_requires_all_three_metrics() -> None:
    config = _config()
    good = {"zero_referenced_r2": 0.6, "sign_agreement": 0.8, "relative_l2": 0.6}
    assert sensor.prediction_pass(good, config)
    for key, value in (
        ("zero_referenced_r2", 0.49),
        ("sign_agreement", 0.74),
        ("relative_l2", 0.72),
    ):
        failed = dict(good)
        failed[key] = value
        assert not sensor.prediction_pass(failed, config)


def _state(mean: float, passed: bool) -> dict[str, object]:
    return {
        "continuous": {
            "mean_moment_shift_bins": mean,
            "continuous_pass": passed,
        }
    }


def test_causal_gate_requires_primary_and_specific_controls() -> None:
    states = {
        "order4": _state(0.20, False),
        "local_oracle": _state(0.01, True),
        "source_covector_oracle_error": _state(0.01, True),
        **{name: _state(0.02, True) for name in sensor.SCALAR_ARMS},
        **{name: _state(0.10, False) for name in sensor.CONTROL_NAMES},
    }
    result = sensor.causal_gate_summary([{"states": states}], _config())
    assert result["primary_causal_scalar_gate"] is True
    states["causal_shuffled"] = _state(0.03, True)
    result = sensor.causal_gate_summary([{"states": states}], _config())
    assert result["primary_causal_scalar_gate"] is False


@pytest.mark.parametrize(
    ("valid", "predictive", "causal", "lookahead", "expected", "passed"),
    [
        (False, True, True, True, "invalid", False),
        (True, True, True, False, "causal_activation_scalar_supported", True),
        (
            True,
            True,
            False,
            False,
            "causal_activation_predictive_not_causal",
            False,
        ),
        (
            True,
            False,
            True,
            False,
            "causal_activation_causal_not_predictive",
            False,
        ),
        (True, False, False, True, "lookahead_scalar_only", False),
        (True, False, False, False, "observable_scalar_not_identified", False),
    ],
)
def test_classification_is_fixed(
    valid: bool,
    predictive: bool,
    causal: bool,
    lookahead: bool,
    expected: str,
    passed: bool,
) -> None:
    assert sensor.classify_checkpoint(
        valid=valid,
        predictive_pass=predictive,
        causal_pass=causal,
        lookahead_pass=lookahead,
    ) == (expected, passed)


def test_locked_source_campaign_is_authoritative() -> None:
    campaign, path, details = sensor._load_source_campaign(_config())
    assert sensor._sha256(path) == sensor.SOURCE_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == sensor.SOURCE_IMPLEMENTATION_SHA256
    assert set(details) == {7}


def test_fingerprint_changes_with_checkpoint_identity() -> None:
    config = _config()
    first = sensor._fingerprint(config, 7, "source", "checkpoint")
    assert first == sensor._fingerprint(config, 7, "source", "checkpoint")
    assert first != sensor._fingerprint(config, 7, "source", "different")


def test_completed_seed_resume_verifies_array_hash(tmp_path: Path) -> None:
    config = _config()
    result_path = tmp_path / "runs" / "seed_7" / "result.json"
    arrays_path = result_path.with_name("activation_sensor_arrays.npz")
    result_path.parent.mkdir(parents=True)
    arrays_path.write_bytes(b"arrays")
    fingerprint = sensor._fingerprint(config, 7, "source", "checkpoint")
    implementation = "implementation"
    record = {
        "schema_version": sensor.SCHEMA_VERSION,
        "hypothesis_id": sensor.HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": sensor._evidence_role(config),
        "implementation_sha256": implementation,
        "scientific_fingerprint": fingerprint,
        "configuration": json.loads(json.dumps(asdict(config))),
        "seed": 7,
        "experiment_id": "seed-seven",
        "artifacts": {
            "result": str(result_path),
            "arrays": str(arrays_path),
            "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        },
    }
    result_path.write_text(json.dumps(record), encoding="utf-8")
    assert sensor._reusable_seed_result(
        result_path,
        arrays_path,
        config,
        implementation,
        7,
        fingerprint,
    ) == record
    arrays_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="incompatible completed result"):
        sensor._reusable_seed_result(
            result_path,
            arrays_path,
            config,
            implementation,
            7,
            fingerprint,
        )
