from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.structure_net import tinyllm_analytic_observable_residual as analytic


def _config() -> analytic.AnalyticObservableResidualConfig:
    return analytic.AnalyticObservableResidualConfig(
        seeds=(7,), device="cpu", allow_underpowered=True
    )


def test_primary_configuration_and_thresholds_are_locked() -> None:
    config = analytic.AnalyticObservableResidualConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    assert config.orbit_count == 64
    assert config.phase_count == 16
    assert config.nuisance_replicates == 4
    assert config.carrier_rank == 3
    assert config.writer_order == 4
    assert config.covector_r2_floor == pytest.approx(0.90)
    assert config.covector_relative_l2_ceiling == pytest.approx(0.15)
    assert config.covector_cosine_floor == pytest.approx(0.99)
    assert config.specificity_margin_bins == pytest.approx(0.125)
    with pytest.raises(ValueError, match="fixed to 7,29,53"):
        replace(config, seeds=(7,))
    with pytest.raises(ValueError, match="64 exact orbits"):
        replace(config, orbit_count=32, allow_underpowered=True)


def test_primary_execution_is_forbidden_by_lifecycle_disposition(
    tmp_path: Path,
) -> None:
    config = analytic.AnalyticObservableResidualConfig(device="cpu")
    with pytest.raises(RuntimeError, match="primary analytic-observable execution"):
        analytic.run_campaign(config, tmp_path / "forbidden-primary")
    assert not (tmp_path / "forbidden-primary").exists()


def test_observed_carrier_gate_is_joint() -> None:
    config = _config()
    audit = {
        "circular_alignment": 0.995,
        "mean_shift_bins": 0.10,
        "p95_shift_bins": 0.40,
        "maximum_sheet_difference": 0.005,
    }
    assert analytic.observed_carrier_pass(audit, config)
    for key, value in (
        ("circular_alignment", 0.98),
        ("mean_shift_bins", 0.13),
        ("p95_shift_bins", 0.51),
        ("maximum_sheet_difference", 0.011),
    ):
        changed = dict(audit)
        changed[key] = value
        assert not analytic.observed_carrier_pass(changed, config)


def test_covector_replay_gate_is_joint() -> None:
    config = _config()
    metrics = {
        "zero_referenced_r2": 0.95,
        "relative_l2": 0.10,
        "mean_row_cosine": 0.995,
    }
    assert analytic.covector_replay_pass(metrics, config)
    metrics["mean_row_cosine"] = 0.98
    assert not analytic.covector_replay_pass(metrics, config)


def _continuous(passed: bool, mean: float) -> dict[str, object]:
    return {"continuous_pass": passed, "mean_moment_shift_bins": mean}


def _cell(
    *,
    analytic_pass: bool = True,
    analytic_mean: float = 0.04,
    control_pass: bool = False,
    control_mean: float = 0.30,
) -> dict[str, object]:
    states = {
        "order4": {"continuous": _continuous(False, 0.20)},
        "analytic_observable": {
            "continuous": _continuous(analytic_pass, analytic_mean)
        },
    }
    for name in analytic.CONTROL_NAMES:
        states[name] = {"continuous": _continuous(control_pass, control_mean)}
    return {"states": states}


def test_state_gate_requires_all_cells_and_each_specific_control() -> None:
    summary = analytic.state_gate_summary([_cell(), _cell()], 0.125)
    assert summary["analytic_all_cells_pass"] is True
    assert summary["all_controls_specific"] is True
    assert summary["complete_gate"] is True
    low_margin = analytic.state_gate_summary(
        [_cell(control_mean=0.10), _cell(control_mean=0.10)], 0.125
    )
    assert low_margin["all_controls_specific"] is False
    failed = analytic.state_gate_summary(
        [_cell(), _cell(analytic_pass=False)], 0.125
    )
    assert failed["analytic_all_cells_pass"] is False


@pytest.mark.parametrize(
    ("valid", "oracle", "gates", "expected", "passed"),
    [
        (False, True, analytic.state_gate_summary([_cell()], 0.125), "invalid", False),
        (
            True,
            False,
            analytic.state_gate_summary([_cell()], 0.125),
            "portable_covector_not_replicated",
            False,
        ),
        (
            True,
            True,
            analytic.state_gate_summary([_cell()], 0.125),
            "analytic_observable_residual_sufficient",
            True,
        ),
        (
            True,
            True,
            analytic.state_gate_summary([_cell(control_mean=0.10)], 0.125),
            "analytic_observable_residual_nonspecific",
            False,
        ),
        (
            True,
            True,
            analytic.state_gate_summary([_cell(analytic_pass=False)], 0.125),
            "observable_semantic_target_not_frozen_equivalent",
            False,
        ),
    ],
)
def test_classification_precedence_is_fixed(
    valid: bool,
    oracle: bool,
    gates: dict[str, object],
    expected: str,
    passed: bool,
) -> None:
    assert analytic.classify_checkpoint(
        valid=valid, oracle_pass=oracle, state_gates=gates
    ) == (expected, passed)


def test_locked_scalar_and_covector_sources_are_authoritative() -> None:
    _, scalar_path, scalar_details = analytic._load_scalar_law(_config())
    _, covector_path, covector_details = analytic._load_covectors(_config())
    assert analytic._sha256(scalar_path) == analytic.SCALAR_LAW_CAMPAIGN_SHA256
    assert analytic._sha256(covector_path) == analytic.COVECTOR_CAMPAIGN_SHA256
    assert set(scalar_details) == {7}
    assert set(covector_details) == {7}


def test_fingerprint_changes_with_source_identity() -> None:
    config = _config()
    first = analytic._fingerprint(config, 7, "scalar", "covector", "checkpoint")
    assert first == analytic._fingerprint(
        config, 7, "scalar", "covector", "checkpoint"
    )
    assert first != analytic._fingerprint(
        config, 7, "changed", "covector", "checkpoint"
    )


def test_completed_seed_resume_verifies_array_hash(tmp_path: Path) -> None:
    config = _config()
    result_path = tmp_path / "runs" / "seed_7" / "result.json"
    arrays_path = result_path.with_name("observable_residual_arrays.npz")
    result_path.parent.mkdir(parents=True)
    arrays_path.write_bytes(b"immutable-arrays")
    fingerprint = analytic._fingerprint(
        config, 7, "scalar", "covector", "checkpoint"
    )
    record = {
        "schema_version": analytic.SCHEMA_VERSION,
        "hypothesis_id": analytic.HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": analytic._evidence_role(config),
        "implementation_sha256": "implementation",
        "configuration": json.loads(json.dumps(asdict(config))),
        "seed": 7,
        "scientific_fingerprint": fingerprint,
        "artifacts": {
            "result": str(result_path),
            "arrays": str(arrays_path),
            "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        },
    }
    result_path.write_text(json.dumps(record), encoding="utf-8")
    assert analytic._reusable_seed_result(
        result_path,
        arrays_path,
        config,
        "implementation",
        7,
        fingerprint,
    ) == record
    arrays_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="incompatible completed result"):
        analytic._reusable_seed_result(
            result_path,
            arrays_path,
            config,
            "implementation",
            7,
            fingerprint,
        )


def test_phase_and_nuisance_controls_are_distinct_fixed_permutations() -> None:
    phase = analytic.defect.phase_shift_indices(16, 4)
    nuisance = analytic.group.phase_matched_shift_indices(16, 4).numpy()
    identity = np.arange(64)
    assert len(set(phase.tolist())) == 64
    assert len(set(nuisance.tolist())) == 64
    assert np.array_equal(phase % 4, identity % 4)
    assert np.array_equal(nuisance // 4, identity // 4)
    assert not np.array_equal(phase, nuisance)
