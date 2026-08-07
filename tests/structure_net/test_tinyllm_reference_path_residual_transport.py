from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

import experiments.structure_net.tinyllm_reference_path_residual_transport as transport


def test_primary_configuration_is_locked() -> None:
    config = transport.ReferencePathTransportConfig()
    assert config.fine_steps == 16
    assert config.step_counts == (1, 2, 4, 8, 16)
    assert config.path_sharing_tolerance == 1e-12
    assert config.required_seed_passes == 4
    with pytest.raises(ValueError, match="primary checkpoint seeds"):
        transport.ReferencePathTransportConfig(
            seeds=(7,), required_seed_passes=1
        )


def test_underpowered_configuration_is_explicit() -> None:
    config = transport.ReferencePathTransportConfig(
        seeds=(7,),
        step_counts=(1, 16),
        required_seed_passes=1,
        control_pass_ceiling=1,
        allow_underpowered=True,
        device="cpu",
    )
    assert config.allow_underpowered
    assert transport._evidence_role(config) == (
        "systems_lifecycle_only_not_quality_evidence"
    )
    assert transport._evidence_role(
        transport.ReferencePathTransportConfig()
    ) == "corrective_outcome_exposed_frozen_residual_transport_audit"


def test_implementation_digest_is_protocol_scoped() -> None:
    digests = transport._source_digests()
    assert digests["runner_protocol"] == transport._runner_protocol_digest()
    assert "runner" not in digests
    assert len(transport._implementation_digest(digests)) == 64


def test_shortest_reference_path_crosses_pi_not_zero() -> None:
    start_angle = torch.deg2rad(torch.tensor([170.0], dtype=torch.float64))
    end_angle = torch.deg2rad(torch.tensor([-170.0], dtype=torch.float64))
    start = torch.polar(torch.ones_like(start_angle), start_angle)
    endpoint = torch.polar(torch.ones_like(end_angle), end_angle)
    path = transport.shortest_reference_path(start, endpoint, 2)
    assert path.shape == (3, 1)
    assert torch.equal(path[0], start)
    assert torch.equal(path[-1], endpoint)
    assert abs(abs(float(torch.angle(path[1, 0]))) - torch.pi) < 1e-12


def test_shortest_reference_path_preserves_stored_endpoints_exactly() -> None:
    start = torch.complex(
        torch.tensor([0.8], dtype=torch.float64),
        torch.tensor([0.6], dtype=torch.float64),
    ) * (1.0 + 2.0e-16)
    endpoint = torch.complex(
        torch.tensor([0.6], dtype=torch.float64),
        torch.tensor([0.8], dtype=torch.float64),
    ) * (1.0 - 2.0e-16)
    path = transport.shortest_reference_path(start, endpoint, 16)
    assert torch.equal(path[0], start)
    assert torch.equal(path[-1], endpoint)
    assert torch.allclose(path[1:-1].abs(), torch.ones(15, 1, dtype=torch.float64))


def test_shared_angular_increments_allow_different_absolute_sheets() -> None:
    start = torch.polar(
        torch.ones(2, dtype=torch.float64),
        torch.tensor([0.1, 1.1], dtype=torch.float64),
    )
    endpoint = torch.polar(
        torch.ones(2, dtype=torch.float64),
        torch.tensor([0.3, 1.3], dtype=torch.float64),
    )
    path = transport.shortest_reference_path(start, endpoint, 16)
    increments = torch.angle(path * path[0:1].conj())
    assert not torch.allclose(path[:, 0], path[:, 1])
    assert torch.allclose(increments[:, 0], increments[:, 1], atol=1e-12)


def test_minimum_norm_delta_closes_a_linear_coordinate() -> None:
    current = torch.tensor([0.2, -0.4])
    target = torch.tensor([0.7, 0.1])
    gradient = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
    delta = transport.minimum_norm_task_delta(
        current, target, gradient, squared_floor=1e-8
    )
    predicted = current + (gradient * delta).sum(1)
    assert torch.allclose(predicted, target, atol=1e-6)


def test_nested_step_indices_are_exact_subsets() -> None:
    assert transport.nested_step_indices(16, 1) == (0, 16)
    assert transport.nested_step_indices(16, 4) == (0, 4, 8, 12, 16)
    assert transport.nested_step_indices(16, 16) == tuple(range(17))
    with pytest.raises(ValueError, match="divide"):
        transport.nested_step_indices(16, 3)


def test_residual_curve_geometry_identifies_a_straight_path() -> None:
    fractions = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    path = torch.stack(
        [torch.stack((fractions, 2.0 * fractions), dim=1)] * 3,
        dim=1,
    )
    geometry = transport.residual_curve_geometry(path)
    assert torch.allclose(
        geometry["arc_chord_ratio"],
        torch.ones_like(geometry["arc_chord_ratio"]),
    )
    assert torch.allclose(
        geometry["maximum_relative_chord_deviation"],
        torch.zeros_like(geometry["maximum_relative_chord_deviation"]),
    )


def test_fiber_block_permutation_preserves_sheet_identity() -> None:
    fiber = torch.tensor([0, 0, 1, 1, 2, 2])
    branch = torch.tensor([-1, 1, -1, 1, -1, 1])
    permutation = transport.fiber_block_permutation(fiber, branch, shift=0)
    assert torch.equal(branch[permutation], branch)
    assert not torch.equal(permutation, torch.arange(6))
    for target in range(3):
        donor_fibers = torch.unique(fiber[permutation[fiber == target]])
        assert len(donor_fibers) == 1
        assert int(donor_fibers[0]) != target


def test_jensen_shannon_is_zero_for_identity_and_symmetric() -> None:
    left = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
    right = torch.tensor([[0.4, 0.6], [0.1, 0.9]])
    identity = transport.jensen_shannon_divergence(left, left)
    assert torch.allclose(
        identity, torch.zeros_like(identity)
    )
    assert torch.allclose(
        transport.jensen_shannon_divergence(left, right),
        transport.jensen_shannon_divergence(right, left),
    )


def _counts(value: int) -> dict[str, int]:
    return {condition: value for condition in transport.CONDITIONS}


def test_classification_separates_schedule_outcomes() -> None:
    assert transport.classify_campaign(
        valid=True,
        true_cosine_pass=_counts(4),
        path_moment_pass=_counts(4),
        required=4,
    ) == "local_relinearization_sufficient"
    assert transport.classify_campaign(
        valid=True,
        true_cosine_pass=_counts(0),
        path_moment_pass=_counts(4),
        required=4,
    ) == "terminal_coordinate_mismatch"
    assert transport.classify_campaign(
        valid=True,
        true_cosine_pass=_counts(0),
        path_moment_pass=_counts(0),
        required=4,
    ) == "scalar_transport_insufficient"
    assert transport.classify_campaign(
        valid=False,
        true_cosine_pass=_counts(4),
        path_moment_pass=_counts(4),
        required=4,
    ) == "invalid"


def test_locked_source_campaign_replays() -> None:
    config = transport.ReferencePathTransportConfig(
        seeds=(7,),
        step_counts=(1, 16),
        required_seed_passes=1,
        control_pass_ceiling=1,
        allow_underpowered=True,
        device="cpu",
    )
    campaign, path, entries = transport._load_source_campaign(config)
    assert path.is_file()
    assert transport._sha256(path) == transport.SOURCE_CAMPAIGN_SHA256
    assert len(entries) == 10
    assert campaign["aggregates"]["target_oracle_pass_counts"] == {
        "analytic_calibrated": 0,
        "learned_calibrated_equivariant": 2,
    }


def test_exact_resume_binds_configuration_and_every_artifact(
    tmp_path: Path,
) -> None:
    config = transport.ReferencePathTransportConfig(
        seeds=(7,),
        step_counts=(1, 16),
        required_seed_passes=1,
        control_pass_ceiling=1,
        allow_underpowered=True,
        device="cpu",
    )
    implementation = "a" * 64
    entries = []
    result_paths = []
    for condition in transport.CONDITIONS:
        root = tmp_path / condition
        root.mkdir()
        result = root / "result.json"
        arrays = root / "arrays.npz"
        result.write_text(f"{condition}\n", encoding="utf-8")
        arrays.write_bytes(condition.encode())
        result_paths.append(result)
        entries.append({
            "condition": condition,
            "seed": 7,
            "path": str(result),
            "result_sha256": transport._sha256(result),
            "arrays_path": str(arrays),
            "arrays_sha256": transport._sha256(arrays),
        })
    campaign = {
        "schema_version": transport.SCHEMA_VERSION,
        "hypothesis_id": transport.HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": transport._evidence_role(config),
        "implementation_sha256": implementation,
        "configuration": asdict(config),
        "summary": {"requested": 2, "completed": 2},
        "results": entries,
        "result_manifest_sha256": transport._manifest_sha256(result_paths),
    }
    assert transport._campaign_reusable(campaign, config, implementation)
    changed = replace(config, analysis_seed=config.analysis_seed + 1)
    assert not transport._campaign_reusable(campaign, changed, implementation)
    Path(entries[0]["arrays_path"]).write_bytes(b"tampered")
    assert not transport._campaign_reusable(campaign, config, implementation)
