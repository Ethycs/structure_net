from pathlib import Path

import json
import numpy as np
import pytest
import torch

import experiments.structure_net.tinyllm_repeated_reference_acquisition as acquisition


def test_primary_configuration_is_locked() -> None:
    config = acquisition.RepeatedReferenceConfig()
    assert config.replicate_counts == (1, 4, 16, 64, 256)
    assert config.measurement_sigma_radians == 0.175
    assert config.denoiser_steps == 1_200
    with pytest.raises(ValueError, match="primary arms and five seeds"):
        acquisition.RepeatedReferenceConfig(
            seeds=(7,), required_seed_passes=1
        )


def test_underpowered_configuration_is_explicit() -> None:
    config = acquisition.RepeatedReferenceConfig(
        seeds=(7,),
        replicate_counts=(1, 4),
        required_seed_passes=1,
        denoiser_steps=4,
        denoiser_validation_interval=2,
        denoiser_patience=1,
        allow_underpowered=True,
        device="cpu",
    )
    assert config.allow_underpowered


def test_full_campaign_is_labeled_corrective_not_confirmatory() -> None:
    assert acquisition._evidence_role(acquisition.RepeatedReferenceConfig()) == (
        "corrective_outcome_informed_frozen_system_acquisition_intervention"
    )


def test_implementation_digest_uses_scientific_protocol_not_cli_default() -> None:
    digests = acquisition._source_digests()
    assert digests["runner_protocol"] == acquisition._runner_protocol_digest()
    assert "runner" not in digests
    assert len(acquisition._implementation_digest(digests)) == 64


def test_circular_mean_m1_is_the_source_observation() -> None:
    true = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    inverse = torch.tensor([0, 0])
    errors = torch.tensor([[0.5], [-0.1]], dtype=torch.float64)
    value, audit = acquisition.circular_mean_orientation(
        true, inverse, errors, sigma=0.2, count=1
    )
    expected = torch.tensor([[np.cos(0.1), np.sin(0.1)]]).float().repeat(2, 1)
    assert torch.allclose(value, expected, atol=1e-6)
    assert audit["replicate_count"] == 1


def test_shared_error_preserves_distinct_sheet_base_orientations() -> None:
    base_angles = torch.tensor([0.2, -0.7], dtype=torch.float64)
    true = torch.stack((torch.cos(base_angles), torch.sin(base_angles)), dim=1)
    inverse = torch.tensor([0, 0])
    errors = torch.tensor([[0.5]], dtype=torch.float64)
    value, _ = acquisition.circular_mean_orientation(
        true, inverse, errors, sigma=0.2, count=1
    )
    value_angles = torch.atan2(value[:, 1].double(), value[:, 0].double())
    relative = torch.atan2(
        torch.sin(value_angles - base_angles),
        torch.cos(value_angles - base_angles),
    )
    assert torch.allclose(relative, torch.full_like(relative, 0.1), atol=1e-7)
    assert not torch.allclose(value[0], value[1])


def test_paired_fiber_shares_error_not_absolute_orientation() -> None:
    true = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    inverse = torch.tensor([0, 0])
    errors = torch.tensor([[0.5]], dtype=torch.float64)
    value, _ = acquisition.circular_mean_orientation(
        true, inverse, errors, sigma=0.2, count=1
    )
    value_angle = torch.atan2(value[:, 1].double(), value[:, 0].double())
    base_angle = torch.atan2(true[:, 1].double(), true[:, 0].double())
    angular_error = torch.atan2(
        torch.sin(value_angle - base_angle), torch.cos(value_angle - base_angle)
    )
    assert angular_error[0] == pytest.approx(0.1, abs=1e-6)
    assert angular_error[1] == pytest.approx(0.1, abs=1e-6)
    assert not torch.allclose(value[0], value[1])


def test_duplicate_observations_do_not_reduce_error() -> None:
    true = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    inverse = torch.tensor([0, 0])
    errors = torch.full((256, 1), 0.4, dtype=torch.float64)
    one, one_audit = acquisition.circular_mean_orientation(
        true, inverse, errors, 0.2, 1
    )
    many, many_audit = acquisition.circular_mean_orientation(
        true, inverse, errors, 0.2, 256
    )
    assert torch.allclose(one, many, atol=1e-12)
    assert many_audit["angular_rmse_radians"] == pytest.approx(
        one_audit["angular_rmse_radians"]
    )


def test_independent_references_contract_standard_error() -> None:
    generator = torch.Generator().manual_seed(17)
    errors = torch.randn((256, 4096), generator=generator, dtype=torch.float64)
    true = torch.tensor([[1.0, 0.0]]).repeat(4096, 1)
    inverse = torch.arange(4096)
    _, one = acquisition.circular_mean_orientation(true, inverse, errors, 0.175, 1)
    _, many = acquisition.circular_mean_orientation(
        true, inverse, errors, 0.175, 256
    )
    ratio = one["angular_rmse_radians"] / many["angular_rmse_radians"]
    assert 14.0 < ratio < 18.0


def test_equivariant_aggregator_has_exact_group_contract() -> None:
    torch.manual_seed(11)
    model = acquisition.EquivariantReliabilityAggregator(hidden=8)
    contracts = acquisition._denoiser_contracts(model, torch.device("cpu"))
    assert contracts["maximum_permutation_error"] <= 1e-6
    assert contracts["maximum_rotation_equivariance_error"] <= 1e-6
    assert contracts["maximum_unit_norm_error"] <= 1e-6


def test_task_recovery_gate_requires_both_shifts() -> None:
    clean = {
        "composition": {"exact_bin_accuracy": 0.80},
        "extrapolation": {"exact_bin_accuracy": 0.70},
    }
    metrics = {
        "composition": {"exact_bin_accuracy": 0.78},
        "extrapolation": {"exact_bin_accuracy": 0.68},
    }
    passed, summary = acquisition.task_recovery_pass(metrics, clean, 0.03)
    assert passed
    metrics["extrapolation"]["exact_bin_accuracy"] = 0.66
    passed, summary = acquisition.task_recovery_pass(metrics, clean, 0.03)
    assert not passed
    assert not summary["extrapolation"]["pass"]


def test_standard_error_gate_checks_ratio_and_slope() -> None:
    config = acquisition.RepeatedReferenceConfig()
    audits = {
        regime: {
            count: {
                "angular_rmse_radians": 0.175 / np.sqrt(count)
            }
            for count in acquisition.REPLICATE_COUNTS
        }
        for regime in acquisition.REGIMES
    }
    contract = acquisition.standard_error_contract(audits, config)
    assert contract["pass"]
    assert all(
        -0.51 < value["log_log_slope"] < -0.49
        for value in contract["splits"].values()
    )


def test_classification_requires_both_aggregators_and_controls() -> None:
    truth = {arm: True for arm in acquisition.ARMS}
    classification, primary = acquisition.classify_campaign(
        valid=True,
        analytic_primary_by_arm=truth,
        learned_primary_by_arm=truth,
        learned_match_by_arm=truth,
        rate_pass=True,
        controls_pass=True,
    )
    assert classification == "acquisition_variance_causally_sufficient"
    assert primary
    classification, primary = acquisition.classify_campaign(
        valid=True,
        analytic_primary_by_arm=truth,
        learned_primary_by_arm=truth,
        learned_match_by_arm=truth,
        rate_pass=True,
        controls_pass=False,
    )
    assert classification == "mixed_acquisition_result"
    assert not primary


def test_source_contract_and_first_replicate_replay(tmp_path: Path) -> None:
    source_path = Path(
        "data/experiments/tinyllm_calibration_orientation_noise/"
        "20260807_d8_preregistered/campaign_results.json"
    )
    if not source_path.is_file():
        pytest.skip("orientation source campaign is absent")
    config = acquisition.RepeatedReferenceConfig(
        seeds=(7,),
        replicate_counts=(1, 4),
        required_seed_passes=1,
        denoiser_steps=2,
        denoiser_validation_interval=1,
        denoiser_patience=1,
        allow_underpowered=True,
        device="cpu",
    )
    loaded = acquisition._load_source(config)
    campaign, campaign_path, entries, _, task, _ = loaded
    assert campaign_path == source_path
    assert len(entries) >= 2
    datasets = acquisition._datasets(task)
    noise, _, contracts = acquisition._ensure_noise_arrays(
        tmp_path / "noise.npz",
        datasets,
        Path(campaign["artifacts"]["noise_arrays"]),
        config,
    )
    assert contracts["first_replicate_source_identity"]
    with np.load(campaign["artifacts"]["noise_arrays"], allow_pickle=False) as source:
        for regime in acquisition.REGIMES:
            inverse = noise[f"{regime}__fiber_inverse"].numpy()
            errors = noise[f"{regime}__errors"].numpy()
            rows = source[f"{regime}__base_noise"]
            assert np.array_equal(errors[0][inverse], rows)


def test_denoiser_save_reload_lifecycle(tmp_path: Path) -> None:
    config = acquisition.RepeatedReferenceConfig(
        seeds=(7,),
        replicate_counts=(1, 4),
        required_seed_passes=1,
        denoiser_hidden=8,
        denoiser_batch_size=16,
        denoiser_steps=2,
        denoiser_validation_interval=1,
        denoiser_patience=1,
        allow_underpowered=True,
        device="cpu",
    )
    implementation = "0" * 64
    path = tmp_path / "denoiser.pt"
    first, record, digest = acquisition._fit_or_load_denoiser(
        path, config, implementation, torch.device("cpu")
    )
    second, reloaded, second_digest = acquisition._fit_or_load_denoiser(
        path, config, implementation, torch.device("cpu")
    )
    assert digest == second_digest
    assert record["state_sha256"] == reloaded["state_sha256"]
    assert acquisition._state_digest(first) == acquisition._state_digest(second)


def test_preregistration_quarantines_post_outcome_expansion() -> None:
    prereg = Path(
        "docs/07 - Status Reports/"
        "2026-08-07_tinyllm-repeated-reference-acquisition-preregistration.md"
    ).read_text(encoding="utf-8")
    assert "SUPERSEDED FOR CONFIRMATION" in prereg
    assert "corrective, outcome-informed" in prereg
    assert "quarantined from confirmatory use" in prereg
    assert "learned equivariant aggregator" in prereg.lower()
    assert "exact-reference oracle" in prereg.lower()
    assert "standard-error law" in prereg.lower()
    primary = Path(
        "data/experiments/tinyllm_repeated_reference_acquisition/"
        "20260807_d8_corrective_v2/campaign_results.json"
    )
    if primary.is_file():
        campaign = json.loads(primary.read_text(encoding="utf-8"))
        assert campaign["status"] == "completed"
