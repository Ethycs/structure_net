from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

import experiments.structure_net.tinyllm_posterior_coordinate_transport as coordinate


def test_primary_configuration_is_locked() -> None:
    config = coordinate.PosteriorCoordinateTransportConfig()
    assert config.seeds == (7, 17, 29, 41, 53)
    assert config.ranks == (1, 2, 4, 8, 15)
    assert config.step_counts == (4, 16)
    assert config.fine_steps == 16
    assert config.pinv_rcond == 1e-6
    assert config.posterior_floor == 1e-12
    assert config.required_seed_passes == 4
    with pytest.raises(ValueError, match="primary checkpoint seeds"):
        coordinate.PosteriorCoordinateTransportConfig(
            seeds=(7,), required_seed_passes=1
        )
    with pytest.raises(ValueError, match="rank ladder"):
        coordinate.PosteriorCoordinateTransportConfig(ranks=(1, 2, 15))
    with pytest.raises(ValueError, match="end at full"):
        coordinate.PosteriorCoordinateTransportConfig(
            ranks=(1, 2, 4), allow_underpowered=True
        )


def test_underpowered_configuration_is_explicit() -> None:
    config = coordinate.PosteriorCoordinateTransportConfig(
        seeds=(7,),
        required_seed_passes=1,
        control_pass_ceiling=1,
        allow_underpowered=True,
        device="cpu",
    )
    assert config.allow_underpowered
    assert coordinate._evidence_role(config) == (
        "systems_lifecycle_only_not_quality_evidence"
    )
    assert coordinate._evidence_role(
        coordinate.PosteriorCoordinateTransportConfig()
    ) == "preregistered_outcome_directed_posterior_coordinate_transport"


def test_implementation_digest_is_protocol_scoped() -> None:
    digests = coordinate._source_digests()
    assert digests["runner_protocol"] == coordinate._runner_protocol_digest()
    assert "runner" not in digests
    assert len(coordinate._implementation_digest(digests)) == 64


def test_dct_basis_is_orthonormal_nested_and_constant_free() -> None:
    basis = coordinate.dct_coordinate_basis(16)
    assert basis.shape == (15, 16)
    assert basis.dtype == torch.float64
    identity = basis @ basis.transpose(0, 1)
    assert float((identity - torch.eye(15, dtype=torch.float64)).abs().max()) < 1e-12
    assert float(basis.sum(1).abs().max()) < 1e-12
    for rank in (1, 2, 4, 8, 15):
        assert torch.equal(basis[:rank], coordinate.dct_coordinate_basis(16)[:rank])


def test_centered_log_posterior_is_zero_mean_and_shift_free() -> None:
    logits = torch.randn(5, 16, dtype=torch.float64)
    posterior = torch.softmax(logits, dim=-1)
    ell = coordinate.centered_log_posterior(posterior, 1e-12)
    assert float(ell.mean(dim=-1).abs().max()) < 1e-12
    shifted = torch.softmax(logits + 3.5, dim=-1)
    ell_shifted = coordinate.centered_log_posterior(shifted, 1e-12)
    assert float((ell - ell_shifted).abs().max()) < 1e-9


def test_coordinate_schedule_projects_path_posteriors() -> None:
    basis = coordinate.dct_coordinate_basis(16)
    posteriors = torch.softmax(torch.randn(3, 7, 16), dim=-1)
    schedule = coordinate.coordinate_schedule(posteriors, basis, 1e-12)
    assert schedule.shape == (3, 7, 15)
    assert schedule.dtype == torch.float64
    ell = coordinate.centered_log_posterior(posteriors.double(), 1e-12)
    assert torch.allclose(schedule, ell @ basis.transpose(0, 1))


def test_coordinate_jacobian_matches_analytic_linear_map() -> None:
    weight = torch.randn(3, 6, dtype=torch.float64)
    value = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
    coordinates = value @ weight.transpose(0, 1)
    jacobian = coordinate._coordinate_jacobian(value, coordinates)
    assert jacobian.shape == (4, 3, 6)
    expected = weight.unsqueeze(0).expand(4, -1, -1)
    assert torch.allclose(jacobian, expected)


def test_pseudoinverse_step_closes_a_full_rank_system() -> None:
    jacobian = torch.randn(5, 3, 8, dtype=torch.float64)
    error = torch.randn(5, 3, dtype=torch.float64)
    delta, effective_rank, condition = coordinate._pseudoinverse_step(
        jacobian, error, 1e-6
    )
    reached = (jacobian @ delta.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(reached, error, atol=1e-9)
    assert torch.equal(effective_rank, torch.full((5,), 3, dtype=torch.long))
    assert bool((condition >= 1.0).all())


def test_pseudoinverse_step_truncates_defective_directions() -> None:
    row = torch.randn(1, 8, dtype=torch.float64)
    jacobian = torch.cat([row, row * 1e-9], dim=0).unsqueeze(0)
    error = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    delta, effective_rank, condition = coordinate._pseudoinverse_step(
        jacobian, error, 1e-6
    )
    assert int(effective_rank[0]) == 1
    assert torch.isfinite(condition).all()
    reached = (jacobian @ delta.unsqueeze(-1)).squeeze(-1)
    assert abs(float(reached[0, 0]) - 1.0) < 1e-6


def test_rank_labels_expose_the_full_ladder() -> None:
    assert coordinate.rank_label(1) == "1"
    assert coordinate.rank_label(8) == "8"
    assert coordinate.rank_label(15) == "full"


def _gates(passing: tuple[int, ...]) -> dict[str, bool]:
    return {
        coordinate.rank_label(rank): rank in passing for rank in coordinate.RANKS
    }


def test_classification_follows_the_locked_ladder() -> None:
    assert coordinate.classify_campaign(
        valid=True, shuffled_valid=True, rank_gates=_gates((2, 4, 8, 15))
    ) == "compact_vector_chart"
    assert coordinate.classify_campaign(
        valid=True, shuffled_valid=True, rank_gates=_gates((8, 15))
    ) == "high_rank_answer_chart"
    assert coordinate.classify_campaign(
        valid=True, shuffled_valid=True, rank_gates=_gates((15,))
    ) == "high_rank_answer_chart"
    assert coordinate.classify_campaign(
        valid=True, shuffled_valid=True, rank_gates=_gates(())
    ) == "answer_coordinates_nonintegrable"
    assert coordinate.classify_campaign(
        valid=True, shuffled_valid=False, rank_gates=_gates((1,))
    ) == "invalid_control"
    assert coordinate.classify_campaign(
        valid=False, shuffled_valid=True, rank_gates=_gates((1,))
    ) == "invalid"


def test_shuffle_seed_is_condition_checkpoint_and_regime_specific() -> None:
    reference = coordinate._shuffle_seed("analytic_calibrated", 7, "composition", 41_141)
    assert reference == coordinate._shuffle_seed(
        "analytic_calibrated", 7, "composition", 41_141
    )
    assert reference != coordinate._shuffle_seed(
        "learned_calibrated_equivariant", 7, "composition", 41_141
    )
    assert reference != coordinate._shuffle_seed(
        "analytic_calibrated", 17, "composition", 41_141
    )
    assert reference != coordinate._shuffle_seed(
        "analytic_calibrated", 7, "extrapolation", 41_141
    )


def test_locked_source_campaign_replays() -> None:
    config = coordinate.PosteriorCoordinateTransportConfig(
        seeds=(7,),
        required_seed_passes=1,
        control_pass_ceiling=1,
        allow_underpowered=True,
        device="cpu",
    )
    campaign, path, entries = coordinate._load_source_campaign(config)
    assert path.is_file()
    assert coordinate._sha256(path) == coordinate.SOURCE_CAMPAIGN_SHA256
    assert len(entries) == 10
    assert campaign["aggregates"]["classification"] == "semantic_schedule_only"
    final_key = str(coordinate.FINE_STEPS)
    assert campaign["aggregates"]["rollout_pass_counts"]["path_moment"][final_key] == {
        "analytic_calibrated": 0,
        "learned_calibrated_equivariant": 0,
    }


def test_exact_resume_binds_configuration_and_every_artifact(
    tmp_path: Path,
) -> None:
    config = coordinate.PosteriorCoordinateTransportConfig(
        seeds=(7,),
        required_seed_passes=1,
        control_pass_ceiling=1,
        allow_underpowered=True,
        device="cpu",
    )
    implementation = "a" * 64
    entries = []
    result_paths = []
    for condition in coordinate.CONDITIONS:
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
            "result_sha256": coordinate._sha256(result),
            "arrays_path": str(arrays),
            "arrays_sha256": coordinate._sha256(arrays),
        })
    campaign = {
        "schema_version": coordinate.SCHEMA_VERSION,
        "hypothesis_id": coordinate.HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": coordinate._evidence_role(config),
        "implementation_sha256": implementation,
        "configuration": asdict(config),
        "summary": {"requested": 2, "completed": 2},
        "results": entries,
        "result_manifest_sha256": coordinate._manifest_sha256(result_paths),
    }
    assert coordinate._campaign_reusable(campaign, config, implementation)
    changed = replace(config, analysis_seed=config.analysis_seed + 1)
    assert not coordinate._campaign_reusable(campaign, changed, implementation)
    Path(entries[0]["arrays_path"]).write_bytes(b"tampered")
    assert not coordinate._campaign_reusable(campaign, config, implementation)
