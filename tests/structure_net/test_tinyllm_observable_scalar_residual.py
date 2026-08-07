from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path

import pytest
import torch

from experiments.structure_net import tinyllm_observable_scalar_residual as sensor


def _pilot_config() -> sensor.ObservableScalarResidualConfig:
    return sensor.ObservableScalarResidualConfig(
        seeds=(7,), device="cpu", allow_underpowered=True
    )


def test_primary_configuration_and_fresh_e_split_are_locked() -> None:
    config = sensor.ObservableScalarResidualConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    assert config.activation_context_rank == 2
    assert config.scalar_ridge == pytest.approx(1e-3)
    assert sensor.FRESH_COHORT_SEEDS == {
        "composition": 630007,
        "extrapolation": 630008,
    }
    earlier = {
        value
        for cohort in sensor.fixed.COHORT_SEEDS.values()
        for value in cohort.values()
    } | set(sensor.source.FRESH_COHORT_SEEDS.values())
    assert not earlier.intersection(sensor.FRESH_COHORT_SEEDS.values())
    with pytest.raises(ValueError, match="fixed"):
        sensor.ObservableScalarResidualConfig(seeds=(7,), device="cpu")
    pilot = sensor.ObservableScalarResidualConfig(
        seeds=(7,), device="cpu", allow_underpowered=True
    )
    assert sensor._evidence_role(pilot) == (
        "systems_lifecycle_only_not_quality_evidence"
    )


def test_completed_seed_resume_is_fingerprint_and_path_locked(tmp_path: Path) -> None:
    config = _pilot_config()
    result_path = tmp_path / "runs" / "seed_7" / "result.json"
    result_path.parent.mkdir(parents=True)
    fingerprint = sensor._fingerprint(
        config, 7, "campaign", "result", "checkpoint"
    )
    implementation = "implementation"
    record = {
        "schema_version": sensor.SCHEMA_VERSION,
        "hypothesis_id": sensor.HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": sensor._evidence_role(config),
        "implementation_sha256": implementation,
        "configuration": json.loads(json.dumps(asdict(config))),
        "scientific_fingerprint": fingerprint,
        "seed": 7,
        "experiment_id": "seed-seven",
        "artifacts": {"result": str(result_path)},
    }
    result_path.write_text(json.dumps(record), encoding="utf-8")
    assert sensor._reusable_seed_result(
        result_path, config, implementation, 7, fingerprint
    ) == record
    record["scientific_fingerprint"] = "changed"
    result_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible completed result"):
        sensor._reusable_seed_result(
            result_path, config, implementation, 7, fingerprint
        )


def test_source_only_shakedown_requires_one_underpowered_seed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="one underpowered seed"):
        sensor.run_source_only_shakedown(
            sensor.ObservableScalarResidualConfig(device="cpu"), tmp_path
        )


def test_fresh_e_campaign_is_execution_locked_after_source_failure(
    tmp_path: Path,
) -> None:
    config = sensor.ObservableScalarResidualConfig(device="cpu")
    assert sensor.SUPERSEDED_BY == "tinyllm-c2-task-activation-scalar-sensor-v2"
    with pytest.raises(RuntimeError, match="source-only CUDA shakedown failed"):
        sensor.run_campaign(config, tmp_path)


def test_posterior_statistics_are_intrinsic_and_finite() -> None:
    posterior = torch.tensor(
        [
            [0.60, 0.20, 0.10, 0.10],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=torch.float64,
    )
    statistics = sensor.posterior_statistics(posterior)
    assert statistics.shape == (2, 6)
    assert torch.isfinite(statistics).all()
    assert statistics[0, 0] < statistics[1, 0]
    assert statistics[0, 1] > statistics[1, 1]
    assert statistics[0, 2] > statistics[1, 2]
    assert torch.all((statistics[:, 5] >= 0.0) & (statistics[:, 5] <= 0.5))


def test_posterior_statistics_rotate_without_using_a_target_index() -> None:
    posterior = torch.tensor(
        [[0.55, 0.25, 0.10, 0.05, 0.03, 0.02]], dtype=torch.float64
    )
    rotated = posterior.roll(2, dims=1)
    left = sensor.posterior_statistics(posterior)
    right = sensor.posterior_statistics(rotated)
    assert torch.allclose(left, right, atol=1e-12)


def test_pca_context_is_source_fitted_and_rank_locked() -> None:
    generator = torch.Generator().manual_seed(17)
    latent = torch.randn((64, 2), generator=generator, dtype=torch.float64)
    mixing = torch.randn((2, 12), generator=generator, dtype=torch.float64)
    value = latent @ mixing
    model = sensor.fit_pca_context(value, rank=2)
    projected = sensor.apply_pca_context(value, model)
    assert projected.shape == (64, 2)
    assert model["basis"].shape == (12, 2)
    assert torch.allclose(
        model["basis"].T @ model["basis"],
        torch.eye(2, dtype=torch.float64),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="rank deficient"):
        sensor.fit_pca_context(torch.ones((16, 5), dtype=torch.float64), rank=2)


def test_standardizer_replays_source_statistics() -> None:
    value = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64
    )
    model = sensor.fit_standardizer(value, 1e-8)
    standardized = sensor.apply_standardizer(value, model)
    assert torch.allclose(standardized.mean(0), torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(
        standardized.std(0, unbiased=False), torch.ones(2, dtype=torch.float64)
    )


def test_candidate_feature_widths_are_exact_and_nested() -> None:
    rows = 32
    phase = torch.randn((rows, 9), dtype=torch.float64)
    posterior = torch.softmax(torch.randn((rows, 16), dtype=torch.float64), dim=1)
    calibration = torch.randn((rows, 8), dtype=torch.float64)
    activation = torch.randn((rows, 6), dtype=torch.float64)
    contexts = sensor.candidate_contexts(posterior, calibration, activation)
    standardizers = {
        name: sensor.fit_standardizer(value, 1e-8)
        for name, value in contexts.items()
    }
    features = sensor.candidate_features(phase, contexts, standardizers)
    assert {name: value.shape[1] for name, value in features.items()} == (
        sensor.EXPECTED_FEATURE_WIDTHS
    )
    assert contexts["posterior"].shape[1] == 6
    assert contexts["calibration"].shape[1] == 14
    assert contexts["activation"].shape[1] == 20


def test_scalar_maps_recover_linear_source_targets() -> None:
    generator = torch.Generator().manual_seed(91)
    features = {
        name: torch.randn((256, width), generator=generator, dtype=torch.float64)
        for name, width in sensor.EXPECTED_FEATURE_WIDTHS.items()
    }
    weights = torch.randn((9, 1), generator=generator, dtype=torch.float64)
    target = (features["phase_only"] @ weights).reshape(-1)
    maps = sensor.fit_scalar_maps(features, target, ridge=1e-12)
    predicted = sensor.transport.apply_affine(
        features["phase_only"], maps["phase_only"]
    ).reshape(-1)
    assert torch.allclose(predicted, target, atol=1e-9)


def test_random_signed_scalar_is_deterministic_and_norm_matched() -> None:
    value = torch.linspace(-2.0, 2.0, 64, dtype=torch.float64)
    first = sensor.random_signed_scalar(value, 311)
    second = sensor.random_signed_scalar(value, 311)
    different = sensor.random_signed_scalar(value, 312)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert torch.equal(first.abs(), value.abs())


def test_observed_carrier_gate_is_joint() -> None:
    config = sensor.ObservableScalarResidualConfig(device="cpu")
    audit = {
        "circular_alignment": 0.995,
        "mean_shift_bins": 0.10,
        "p95_shift_bins": 0.40,
        "maximum_sheet_difference": 0.005,
    }
    assert sensor.observed_carrier_pass(audit, config)
    audit["p95_shift_bins"] = 0.51
    assert not sensor.observed_carrier_pass(audit, config)


def _continuous(passed: bool, mean: float) -> dict[str, object]:
    return {"continuous_pass": passed, "mean_moment_shift_bins": mean}


def _cell(
    *,
    posterior_pass: bool = True,
    calibration_pass: bool = True,
    activation_pass: bool = True,
    control_pass: bool = False,
    control_mean: float = 0.50,
) -> dict[str, object]:
    endpoint = {
        "phase_only": False,
        "posterior": posterior_pass,
        "calibration": calibration_pass,
        "activation": activation_pass,
    }
    states = {
        name: {"continuous": _continuous(passed, 0.05 if passed else 0.30)}
        for name, passed in endpoint.items()
    }
    for name in sensor.CONTROLLED_CANDIDATES:
        for suffix in sensor.CONTROL_SUFFIXES:
            states[f"{name}_{suffix}"] = {
                "continuous": _continuous(control_pass, control_mean)
            }
    return {"states": states}


def test_arm_gate_requires_both_cells_and_each_specific_control() -> None:
    summary = sensor.arm_gate_summary([_cell(), _cell()], 0.125)
    assert summary["controlled_arms"]["posterior"]["complete_gate"] is True
    assert summary["controlled_arms"]["calibration"]["complete_gate"] is True
    assert summary["controlled_arms"]["activation"]["complete_gate"] is True

    low_margin = sensor.arm_gate_summary(
        [_cell(control_mean=0.10), _cell(control_mean=0.10)], 0.125
    )
    assert low_margin["controlled_arms"]["posterior"]["complete_gate"] is False
    cells = [_cell(), _cell()]
    cells[0]["states"]["posterior"]["continuous"] = _continuous(False, 0.30)
    failure = sensor.arm_gate_summary(cells, 0.125)
    assert failure["controlled_arms"]["posterior"][
        "endpoint_all_fresh_cells_pass"
    ] is False


def _arm_gates(
    *,
    posterior_endpoint: bool = True,
    posterior_specific: bool = True,
    calibration_endpoint: bool = True,
    calibration_specific: bool = True,
    activation_endpoint: bool = True,
    activation_specific: bool = True,
    phase_endpoint: bool = False,
) -> dict[str, object]:
    values = {
        "posterior": (posterior_endpoint, posterior_specific),
        "calibration": (calibration_endpoint, calibration_specific),
        "activation": (activation_endpoint, activation_specific),
    }
    return {
        "endpoint_all_fresh_cells_pass": {
            "phase_only": phase_endpoint,
            **{name: endpoint for name, (endpoint, _) in values.items()},
        },
        "controlled_arms": {
            name: {
                "endpoint_all_fresh_cells_pass": endpoint,
                "all_controls_specific": specific,
                "complete_gate": endpoint and specific,
            }
            for name, (endpoint, specific) in values.items()
        },
    }


@pytest.mark.parametrize(
    ("valid", "oracle", "gates", "expected", "passed"),
    [
        (False, True, _arm_gates(), "invalid", False),
        (
            True,
            False,
            _arm_gates(),
            "fresh_local_mechanism_not_replicated",
            False,
        ),
        (
            True,
            True,
            _arm_gates(),
            "posterior_scalar_sensor_sufficient",
            True,
        ),
        (
            True,
            True,
            _arm_gates(posterior_specific=False),
            "posterior_sensor_nonspecific",
            False,
        ),
        (
            True,
            True,
            _arm_gates(posterior_endpoint=False),
            "calibration_scalar_rescue",
            False,
        ),
        (
            True,
            True,
            _arm_gates(
                posterior_endpoint=False,
                calibration_endpoint=False,
            ),
            "activation_scalar_rescue",
            False,
        ),
        (
            True,
            True,
            _arm_gates(
                posterior_endpoint=False,
                calibration_specific=False,
                activation_specific=False,
            ),
            "secondary_sensor_nonspecific",
            False,
        ),
        (
            True,
            True,
            _arm_gates(
                posterior_endpoint=False,
                calibration_endpoint=False,
                activation_endpoint=False,
            ),
            "no_observable_scalar_sensor",
            False,
        ),
    ],
)
def test_checkpoint_classification_is_preregistered(
    valid: bool,
    oracle: bool,
    gates: dict[str, object],
    expected: str,
    passed: bool,
) -> None:
    assert sensor.classify_checkpoint(
        valid=valid, oracle_pass=oracle, arm_gates=gates
    ) == (expected, passed)


def test_campaign_decision_requires_three_primary_gates() -> None:
    supported = sensor._campaign_decision(
        ["posterior_scalar_sensor_sufficient"] * 3, 3, False
    )
    assert supported["supported"] is True
    partial = sensor._campaign_decision(
        ["posterior_scalar_sensor_sufficient"] * 2
        + ["calibration_scalar_rescue"],
        2,
        False,
    )
    assert partial["supported"] is False
    assert partial["conclusion"] == "checkpoint_stratified_scalar_sensor"
    systems = sensor._campaign_decision(
        ["posterior_scalar_sensor_sufficient"], 1, True
    )
    assert systems["conclusion"] == "systems_lifecycle_only_not_quality_evidence"


def test_source_covector_campaign_and_results_are_hash_locked() -> None:
    config = sensor.ObservableScalarResidualConfig(device="cpu")
    campaign, path, details = sensor._load_source_covectors(config)
    assert sensor._sha256(path) == sensor.SOURCE_COVECTOR_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == (
        sensor.SOURCE_COVECTOR_IMPLEMENTATION_SHA256
    )
    assert set(details) == {7, 29, 53}
    for detail, result_path in details.values():
        entry = next(
            item for item in campaign["results"] if item["seed"] == detail["seed"]
        )
        assert sensor._sha256(result_path) == entry["result_sha256"]
