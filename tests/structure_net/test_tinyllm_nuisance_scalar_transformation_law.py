from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from experiments.structure_net import tinyllm_nuisance_scalar_transformation_law as law
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


def _config() -> law.NuisanceScalarTransformationConfig:
    return law.NuisanceScalarTransformationConfig(
        seeds=(7,), device="cpu", allow_underpowered=True
    )


def test_primary_configuration_and_thresholds_are_locked() -> None:
    config = law.NuisanceScalarTransformationConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    assert config.orbit_count == 64
    assert config.phase_count == 16
    assert config.nuisance_replicates == 4
    assert config.invariance_r2_floor == pytest.approx(0.90)
    assert config.invariance_relative_l2_ceiling == pytest.approx(math.sqrt(0.10))
    assert config.invariance_sign_agreement_floor == pytest.approx(0.90)
    assert config.shuffled_r2_margin == pytest.approx(0.10)
    with pytest.raises(ValueError, match="64 exact orbits"):
        replace(config, orbit_count=32, allow_underpowered=True)
    with pytest.raises(ValueError, match="fixed to 7,29,53"):
        replace(config, seeds=(7,))


def test_scalar_pair_metrics_identity_passes_exactly() -> None:
    value = torch.tensor([-2.0, -0.5, 0.25, 3.0], dtype=torch.float64)
    metrics = law.scalar_pair_metrics(value, value.clone(), sign_floor=0.01)
    assert metrics["zero_referenced_r2"] == pytest.approx(1.0)
    assert metrics["relative_l2"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["action_difference_rms"] == pytest.approx(0.0)
    assert metrics["sign_agreement"] == pytest.approx(1.0)
    assert metrics["sign_evaluation_count"] == 4
    assert metrics["correlation"] == pytest.approx(1.0)
    assert law.invariance_pass(metrics, _config())


def test_scalar_pair_metrics_sign_flip_fails_invariance() -> None:
    transformed = torch.tensor([-2.0, -0.5, 0.25, 3.0], dtype=torch.float64)
    metrics = law.scalar_pair_metrics(-transformed, transformed, sign_floor=0.01)
    assert metrics["zero_referenced_r2"] == pytest.approx(-3.0)
    assert metrics["relative_l2"] == pytest.approx(2.0)
    assert metrics["sign_agreement"] == pytest.approx(0.0)
    assert metrics["correlation"] == pytest.approx(-1.0)
    assert not law.invariance_pass(metrics, _config())


def test_sign_gate_ignores_only_declared_small_targets() -> None:
    transformed = torch.tensor([0.001, -0.009, 0.02, -0.03], dtype=torch.float64)
    reference = torch.tensor([-0.001, 0.009, 0.02, -0.03], dtype=torch.float64)
    metrics = law.scalar_pair_metrics(reference, transformed, sign_floor=0.01)
    assert metrics["sign_evaluation_count"] == 2
    assert metrics["sign_agreement"] == pytest.approx(1.0)


def test_cell_gate_keeps_invariance_and_correspondence_separate() -> None:
    config = _config()
    paired = {
        "zero_referenced_r2": 0.95,
        "relative_l2": 0.20,
        "sign_agreement": 0.95,
    }
    nonspecific = {"zero_referenced_r2": 0.90}
    specific = {"zero_referenced_r2": 0.80}
    first = law.cell_gate(paired, nonspecific, config)
    second = law.cell_gate(paired, specific, config)
    assert first["invariance_pass"] is True
    assert first["correspondence_specific"] is False
    assert second["invariance_pass"] is True
    assert second["correspondence_specific"] is True


def test_basis_gauge_alignment_recovers_signed_axes_without_changing_subspace() -> None:
    basis = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    stored = torch.tensor(
        [[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5], [0.5, -1.0, 2.0], [3.0, 0.5, -1.0]],
        dtype=torch.float64,
    )
    transform = torch.diag(torch.tensor([-1.0, -1.0, 1.0], dtype=torch.float64))
    regenerated = stored @ transform
    aligned, summary = law.align_basis_to_stored_gauge(
        basis, regenerated, stored
    )
    torch.testing.assert_close(aligned, transform @ basis, atol=1e-12, rtol=0)
    assert summary["maximum_orthogonality_error"] < 1e-12
    assert summary["maximum_gauge_fit_error"] < 1e-12
    assert summary["aligned_basis_orthogonality_error"] < 1e-12


@pytest.mark.parametrize(
    ("valid", "invariant", "specific", "expected"),
    [
        (False, True, True, "invalid"),
        (
            True,
            True,
            True,
            "scalar_nuisance_invariant_and_correspondence_specific",
        ),
        (True, True, False, "scalar_nuisance_invariant_nonspecific"),
        (True, False, True, "scalar_action_dependent"),
        (True, False, False, "scalar_action_dependent"),
    ],
)
def test_classification_precedence_is_fixed(
    valid: bool, invariant: bool, specific: bool, expected: str
) -> None:
    assert law.classify_checkpoint(
        valid=valid,
        invariant=invariant,
        correspondence_specific=specific,
    ) == expected


def test_phase_matched_shuffle_preserves_phase_and_changes_replicate() -> None:
    indices = law.group.phase_matched_shift_indices(2, 4)
    assert indices.tolist() == [1, 2, 3, 0, 5, 6, 7, 4]
    phases = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    assert torch.equal(phases, phases[indices])
    assert not torch.equal(indices, torch.arange(8))


def test_regenerated_group_inputs_replay_locked_artifact() -> None:
    config = _config()
    _, _, details = law._load_group_campaign(config)
    detail = details[7][0]
    group_config = law._group_config(config)
    for regime in law.REGIMES:
        _, summary = law.group.generate_group_paired_orbits(
            CircleTaskConfig(),
            group_config,
            seed=law.group.FRESH_COHORT_SEEDS[regime],
            regime=regime,
        )
        assert summary == detail["fresh_inputs"][regime]


def test_locked_group_campaign_is_authoritative() -> None:
    campaign, path, details = law._load_group_campaign(_config())
    assert law._sha256(path) == law.GROUP_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == law.GROUP_IMPLEMENTATION_SHA256
    assert set(details) == {7}


def test_fingerprint_changes_with_source_identity() -> None:
    config = _config()
    first = law._fingerprint(config, 7, "group-result", "checkpoint")
    assert first == law._fingerprint(config, 7, "group-result", "checkpoint")
    assert first != law._fingerprint(config, 7, "different", "checkpoint")
    assert first != law._fingerprint(config, 7, "group-result", "different")


def test_completed_seed_resume_verifies_array_hash(tmp_path: Path) -> None:
    config = _config()
    result_path = tmp_path / "runs" / "seed_7" / "result.json"
    arrays_path = result_path.with_name("scalar_law_arrays.npz")
    result_path.parent.mkdir(parents=True)
    arrays_path.write_bytes(b"immutable-arrays")
    fingerprint = law._fingerprint(config, 7, "group-result", "checkpoint")
    implementation = "implementation"
    record = {
        "schema_version": law.SCHEMA_VERSION,
        "hypothesis_id": law.HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": law._evidence_role(config),
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
    assert law._reusable_seed_result(
        result_path,
        arrays_path,
        config,
        implementation,
        7,
        fingerprint,
    ) == record
    arrays_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="incompatible completed result"):
        law._reusable_seed_result(
            result_path,
            arrays_path,
            config,
            implementation,
            7,
            fingerprint,
        )
