from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.structure_net import tinyllm_local_metric_field_transport as audit
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


def _config() -> audit.LocalMetricFieldTransportConfig:
    return audit.LocalMetricFieldTransportConfig(
        seeds=(7,), device="cpu", allow_underpowered=True
    )


def test_primary_configuration_is_locked() -> None:
    config = audit.LocalMetricFieldTransportConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    assert config.orbit_count == 64
    assert config.phase_count == 16
    assert config.nuisance_replicates == 4
    assert config.group_arms == audit.GROUP_ARMS
    assert config.p95_projector_distance_ceiling == pytest.approx(
        math.sqrt(1.0 - 0.9**2)
    )
    with pytest.raises(ValueError, match="64 exact orbits"):
        replace(config, orbit_count=32, allow_underpowered=True)
    with pytest.raises(ValueError, match="group arms are fixed"):
        replace(config, group_arms=("amplitude",), allow_underpowered=True)


def test_phase_matched_shift_cycles_only_replicates() -> None:
    indices = audit.phase_matched_shift_indices(2, 4)
    assert indices.tolist() == [1, 2, 3, 0, 5, 6, 7, 4]
    phases = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    assert torch.equal(phases, phases[indices])
    assert not torch.equal(indices, torch.arange(8))


def test_observed_group_action_is_rotate_scale_translate() -> None:
    base = np.array([[[1.0, 0.0, 2.0]]])
    amplitude = np.array([[[2.0]]])
    orientation = np.array([[math.pi / 2.0]])
    offset = np.array([[[0.5, -0.5, 1.0]]])
    actual = audit._group_sensor(base, amplitude, orientation, offset)
    expected = np.array([[[0.5, 1.5, 5.0]]])
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize("regime", audit.REGIMES)
def test_fresh_group_pairs_preserve_target_and_change_tokens(regime: str) -> None:
    config = _config()
    datasets, summary = audit.generate_group_paired_orbits(
        CircleTaskConfig(),
        config,
        seed=audit.FRESH_COHORT_SEEDS[regime],
        regime=regime,
    )
    reference = datasets["reference"]
    assert summary["pair_contract"] is True
    assert reference.orbit_count == 64
    assert reference.k == 2
    for arm in audit.GROUP_ARMS:
        assert torch.equal(reference.target_posteriors, datasets[arm].target_posteriors)
        assert torch.equal(reference.target_bins, datasets[arm].target_bins)
        assert torch.equal(reference.quotient_phase, datasets[arm].quotient_phase)
        assert summary[arm]["nonfinite_count"] == 0
        assert summary[arm]["clipping_fraction"] <= config.clipping_fraction_ceiling
        assert summary[arm]["changed_token_fraction"] >= config.changed_token_fraction_floor


def _projector_from_kernel(kernel: torch.Tensor) -> torch.Tensor:
    identity = torch.eye(3, dtype=torch.float64).expand(len(kernel), -1, -1)
    return identity - kernel[:, :, None] * kernel[:, None, :]


def test_projector_geometry_detects_exact_pairing() -> None:
    kernels = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
    )
    projector = _projector_from_kernel(kernels)
    shift = audit.phase_matched_shift_indices(1, 4)
    result = audit.projector_geometry(projector, projector, shift)
    torch.testing.assert_close(
        result["paired_kernel_cosine"], torch.ones(4, dtype=torch.float64)
    )
    torch.testing.assert_close(
        result["projector_distance"], torch.zeros(4, dtype=torch.float64)
    )
    assert result["median_kernel_cosine"] == pytest.approx(1.0)
    assert result["shuffled_median_kernel_cosine"] == pytest.approx(0.0)
    assert result["paired_over_shuffled_median_margin"] == pytest.approx(1.0)


def test_random_projectors_are_deterministic_rank_two_and_idempotent() -> None:
    first = audit.random_rank_two_projectors(8, 123, torch.device("cpu"))
    second = audit.random_rank_two_projectors(8, 123, torch.device("cpu"))
    different = audit.random_rank_two_projectors(8, 124, torch.device("cpu"))
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert not torch.equal(first, different)
    torch.testing.assert_close(first @ first, first, atol=1e-12, rtol=0)
    assert torch.linalg.matrix_rank(first).tolist() == [2] * 8


@pytest.mark.parametrize(
    ("valid", "local", "geometry", "causal", "expected"),
    [
        (False, True, True, True, "invalid"),
        (True, True, True, True, "invariant_metric_field_transport_supported"),
        (True, True, False, True, "causal_transport_without_geometric_invariance"),
        (True, True, True, False, "geometric_invariance_without_causal_transport"),
        (True, False, False, False, "local_tangent_not_fresh_cohort_sufficient"),
        (True, True, False, False, "nuisance_specific_metric_field"),
    ],
)
def test_classification_precedence(
    valid: bool,
    local: bool,
    geometry: bool,
    causal: bool,
    expected: str,
) -> None:
    assert (
        audit.classify_checkpoint(
            valid=valid,
            local_tangent_pass=local,
            geometry_pass=geometry,
            causal_transport_pass=causal,
        )
        == expected
    )


def test_locked_source_campaigns_are_authoritative() -> None:
    writer_campaign, writer_path, details, corrective, corrective_path = (
        audit._load_sources(_config())
    )
    assert audit._sha256(writer_path) == audit.WRITER_CAMPAIGN_SHA256
    assert audit._sha256(corrective_path) == audit.CORRECTIVE_CAMPAIGN_SHA256
    assert writer_campaign["implementation_sha256"] == audit.WRITER_IMPLEMENTATION_SHA256
    assert corrective["implementation_sha256"] == audit.CORRECTIVE_IMPLEMENTATION_SHA256
    assert set(details) == {7, 29, 53}


def test_fingerprint_changes_with_checkpoint_identity() -> None:
    config = _config()
    first = audit._fingerprint(config, 7, "writer", "checkpoint")
    assert first == audit._fingerprint(config, 7, "writer", "checkpoint")
    assert first != audit._fingerprint(config, 7, "writer", "different")


def test_completed_seed_resume_preserves_fingerprint_matched_arrays(
    tmp_path: Path,
) -> None:
    config = _config()
    result_path = tmp_path / "runs" / "seed_7" / "result.json"
    arrays_path = result_path.with_name("metric_field_arrays.npz")
    result_path.parent.mkdir(parents=True)
    arrays_path.write_bytes(b"immutable-arrays")
    fingerprint = audit._fingerprint(config, 7, "writer", "checkpoint")
    implementation = "implementation"
    record = {
        "schema_version": audit.SCHEMA_VERSION,
        "hypothesis_id": audit.HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": audit._evidence_role(config),
        "implementation_sha256": implementation,
        "scientific_fingerprint": fingerprint,
        "configuration": audit._json_compatible(audit.asdict(config)),
        "seed": 7,
        "experiment_id": "seed-seven",
        "artifacts": {
            "result": str(result_path),
            "arrays": str(arrays_path),
            "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        },
    }
    result_path.write_text(json.dumps(record), encoding="utf-8")

    assert audit._reusable_seed_result(
        result_path,
        arrays_path,
        config,
        implementation,
        7,
        fingerprint,
    ) == record

    arrays_path.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="incompatible completed result"):
        audit._reusable_seed_result(
            result_path,
            arrays_path,
            config,
            implementation,
            7,
            fingerprint,
        )
