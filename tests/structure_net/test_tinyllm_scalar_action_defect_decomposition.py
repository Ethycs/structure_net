from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.structure_net import tinyllm_scalar_action_defect_decomposition as defect


def _config() -> defect.ScalarActionDefectConfig:
    return defect.ScalarActionDefectConfig(seeds=(7,), allow_underpowered=True)


def test_primary_configuration_and_thresholds_are_locked() -> None:
    config = defect.ScalarActionDefectConfig()
    assert config.seeds == (7, 29, 53)
    assert config.phase_count == 16
    assert config.nuisance_replicates == 4
    assert config.carrier_rank == 3
    assert config.prediction_r2_floor == pytest.approx(0.90)
    assert config.prediction_relative_l2_ceiling == pytest.approx(math.sqrt(0.10))
    assert config.prediction_sign_agreement_floor == pytest.approx(0.90)
    with pytest.raises(ValueError, match="fixed to 7,29,53"):
        replace(config, seeds=(7,))
    with pytest.raises(ValueError, match="16 phases x 4 nuisances"):
        replace(config, phase_count=8, allow_underpowered=True)


def test_wrap_bins_uses_declared_circular_period() -> None:
    value = np.array([-9.0, -8.0, -7.0, 7.0, 8.0, 9.0])
    assert defect.wrap_bins(value, 16).tolist() == [7.0, -8.0, -7.0, 7.0, -8.0, -7.0]


def test_phase_shift_changes_semantic_phase_not_replicate() -> None:
    indices = defect.phase_shift_indices(16, 4)
    assert indices[:8].tolist() == [4, 5, 6, 7, 8, 9, 10, 11]
    assert indices[-4:].tolist() == [0, 1, 2, 3]
    assert np.array_equal(indices % 4, np.arange(64) % 4)


def test_scalar_prediction_identity_passes() -> None:
    target = np.array([-2.0, -0.5, 0.25, 3.0])
    metrics = defect.scalar_prediction_metrics(target, target, 0.01)
    assert metrics["zero_referenced_r2"] == pytest.approx(1.0)
    assert metrics["relative_l2"] == pytest.approx(0.0)
    assert metrics["sign_agreement"] == pytest.approx(1.0)
    assert defect.prediction_pass(metrics, _config())


def test_scalar_prediction_sign_flip_fails() -> None:
    target = np.array([-2.0, -0.5, 0.25, 3.0])
    metrics = defect.scalar_prediction_metrics(-target, target, 0.01)
    assert metrics["zero_referenced_r2"] == pytest.approx(-3.0)
    assert metrics["relative_l2"] == pytest.approx(2.0)
    assert metrics["sign_agreement"] == pytest.approx(0.0)
    assert not defect.prediction_pass(metrics, _config())


def test_symmetric_decomposition_is_algebraically_exact() -> None:
    rng = np.random.default_rng(7)
    jr = rng.normal(size=(64, 3))
    jg = rng.normal(size=(64, 3))
    zr = rng.normal(size=(64, 3))
    zg = rng.normal(size=(64, 3))
    result = defect.symmetric_action_decomposition(jr, jg, zr, zg)
    assert result["algebraic_relative_error"] < 1e-14
    assert np.allclose(result["joint"], result["direct"])
    assert np.allclose(result["residual"] + result["covector"], result["joint"])


def test_symmetric_decomposition_separates_pure_residual_change() -> None:
    rng = np.random.default_rng(11)
    covector = rng.normal(size=(8, 3))
    zr = rng.normal(size=(8, 3))
    zg = rng.normal(size=(8, 3))
    result = defect.symmetric_action_decomposition(covector, covector, zr, zg)
    assert np.allclose(result["covector"], 0.0)
    assert np.allclose(result["residual"], result["joint"])


def test_symmetric_decomposition_separates_pure_covector_change() -> None:
    rng = np.random.default_rng(13)
    jr = rng.normal(size=(8, 3))
    jg = rng.normal(size=(8, 3))
    residual = rng.normal(size=(8, 3))
    result = defect.symmetric_action_decomposition(jr, jg, residual, residual)
    assert np.allclose(result["residual"], 0.0)
    assert np.allclose(result["covector"], result["joint"])


@pytest.mark.parametrize(
    ("joint", "residual", "covector", "expected"),
    [
        (False, True, True, "nonlinear_or_unresolved"),
        (True, True, False, "residual_coordinate_defect"),
        (True, False, True, "covector_transport_defect"),
        (True, True, True, "dual_sufficient"),
        (True, False, False, "coupled_action_defect"),
    ],
)
def test_cell_classification_is_fixed(
    joint: bool, residual: bool, covector: bool, expected: str
) -> None:
    assert defect.classify_cell(
        joint=joint, residual=residual, covector=covector
    ) == expected


def test_checkpoint_classification_requires_stable_specific_type() -> None:
    labels = ["residual_coordinate_defect"] * 8
    assert defect.classify_checkpoint(
        valid=True, cell_labels=labels, joint_specific=True
    ) == "residual_coordinate_defect"
    labels[-1] = "coupled_action_defect"
    assert defect.classify_checkpoint(
        valid=True, cell_labels=labels, joint_specific=True
    ) == "checkpoint_internal_mixture"
    assert defect.classify_checkpoint(
        valid=False, cell_labels=labels, joint_specific=True
    ) == "invalid"


def test_locked_source_campaign_is_authoritative() -> None:
    campaign, path, details = defect._load_source_campaign(_config())
    assert defect._sha256(path) == defect.SOURCE_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == defect.SOURCE_IMPLEMENTATION_SHA256
    assert set(details) == {7}


def test_fingerprint_changes_with_source_identity() -> None:
    config = _config()
    first = defect._fingerprint(config, 7, "result", "scalar", "group")
    assert first == defect._fingerprint(config, 7, "result", "scalar", "group")
    assert first != defect._fingerprint(config, 7, "changed", "scalar", "group")
    assert first != defect._fingerprint(config, 7, "result", "changed", "group")


def test_completed_seed_resume_verifies_array_hash(tmp_path: Path) -> None:
    config = _config()
    result_path = tmp_path / "runs" / "seed_7" / "result.json"
    arrays_path = result_path.with_name("decomposition_arrays.npz")
    result_path.parent.mkdir(parents=True)
    arrays_path.write_bytes(b"immutable-arrays")
    fingerprint = defect._fingerprint(config, 7, "result", "scalar", "group")
    implementation = "implementation"
    record = {
        "schema_version": defect.SCHEMA_VERSION,
        "hypothesis_id": defect.HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": defect._evidence_role(config),
        "implementation_sha256": implementation,
        "configuration": json.loads(json.dumps(asdict(config))),
        "seed": 7,
        "experiment_id": "seed-seven",
        "scientific_fingerprint": fingerprint,
        "artifacts": {
            "result": str(result_path),
            "arrays": str(arrays_path),
            "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        },
    }
    result_path.write_text(json.dumps(record), encoding="utf-8")
    assert defect._reusable_seed_result(
        result_path, arrays_path, config, implementation, 7, fingerprint
    ) == record
    arrays_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="incompatible completed result"):
        defect._reusable_seed_result(
            result_path, arrays_path, config, implementation, 7, fingerprint
        )
