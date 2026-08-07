import math
from pathlib import Path

import pytest
import torch

import experiments.structure_net.tinyllm_calibration_readout_decomposition as readout


def test_primary_configuration_is_locked() -> None:
    config = readout.ReadoutDecompositionConfig()
    assert config.conditions == readout.CONDITIONS
    assert config.seeds == readout.SEEDS
    assert config.levels == readout.LEVELS
    with pytest.raises(ValueError, match="five primary seeds"):
        readout.ReadoutDecompositionConfig(seeds=(7,))


def test_underpowered_configuration_is_explicit() -> None:
    config = readout.ReadoutDecompositionConfig(
        seeds=(7,), required_seed_passes=1, allow_underpowered=True, device="cpu"
    )
    assert config.allow_underpowered
    assert config.required_seed_passes == 1


def test_interval_centers_and_posterior_are_normalized() -> None:
    centers = readout.interval_centers(16)
    posterior = readout.interval_posterior(torch.tensor([-1.0, 0.0, 1.0]), centers)
    assert centers[0] == -1.0
    assert centers[-1] == 1.0
    assert torch.allclose(posterior.sum(1), torch.ones(3), atol=1e-12)
    assert posterior[0].argmax() == 0
    assert posterior[-1].argmax() == 15


def test_coordinate_metrics_are_exact_on_centers() -> None:
    centers = readout.interval_centers(16)
    indices = torch.tensor([0, 5, 10, 15])
    coordinate = centers[indices]
    targets = torch.nn.functional.one_hot(indices, 16).float()
    metrics = readout.coordinate_metrics(coordinate, coordinate, targets, centers)
    assert metrics["exact_bin_accuracy"] == 1.0
    assert metrics["cosine_mae"] == 0.0
    assert metrics["cosine_rmse"] == 0.0
    assert math.isclose(metrics["cosine_pearson"], 1.0, abs_tol=1e-12)


def test_posterior_mean_can_differ_from_argmax() -> None:
    centers = readout.interval_centers(4)
    posterior = torch.tensor([[0.39, 0.00, 0.30, 0.31]])
    target_cosine = centers[[2]]
    targets = torch.nn.functional.one_hot(torch.tensor([2]), 4).float()
    argmax = readout.posterior_metrics(
        posterior, target_cosine, targets, centers, use_argmax=True
    )
    mean = readout.posterior_metrics(
        posterior, target_cosine, targets, centers, use_argmax=False
    )
    assert argmax["exact_bin_accuracy"] == 0.0
    assert mean["exact_bin_accuracy"] == 1.0


def test_utility_gate_uses_clean_adequacy_and_own_drop() -> None:
    passed, detail = readout.utility_pass(0.70, 0.68, 0.72, 0.03)
    assert passed
    assert detail["clean_adequacy_gap"] == pytest.approx(0.02)
    failed, _ = readout.utility_pass(0.60, 0.60, 0.72, 0.03)
    assert not failed
    failed, _ = readout.utility_pass(0.70, 0.66, 0.72, 0.03)
    assert not failed


def test_patch_gate_uses_unchanged_clean_model_baseline() -> None:
    passed, detail = readout.patch_utility_pass(0.85, 0.80, 0.75, 0.03)
    assert passed
    assert detail["accuracy_drop_from_clean_model"] == pytest.approx(-0.05)
    failed, _ = readout.patch_utility_pass(0.85, 0.70, 0.75, 0.03)
    assert not failed


def _counts(value: int) -> dict[str, int]:
    return {condition: value for condition in readout.CONDITIONS}


def test_classification_order_is_locked() -> None:
    assert readout.classify_campaign(
        valid=False,
        argmax_pass=_counts(0),
        posterior_pass=_counts(5), observer_pass=_counts(5),
        patch_pass=_counts(5), target_oracle_pass=_counts(5),
        controls_specific=True, required=4,
    ) == "invalid"
    assert readout.classify_campaign(
        valid=True,
        argmax_pass=_counts(0),
        posterior_pass=_counts(5), observer_pass=_counts(0),
        patch_pass=_counts(0), target_oracle_pass=_counts(0),
        controls_specific=False, required=4,
    ) == "posterior_shape_calibration_limited"
    assert readout.classify_campaign(
        valid=True,
        argmax_pass=_counts(0),
        posterior_pass=_counts(0), observer_pass=_counts(5),
        patch_pass=_counts(5), target_oracle_pass=_counts(5),
        controls_specific=True, required=4,
    ) == "decoder_relation_limited"


def test_classification_preserves_one_arm_failure() -> None:
    observer = {
        "analytic_calibrated": 2,
        "learned_calibrated_equivariant": 5,
    }
    assert readout.classify_campaign(
        valid=True,
        argmax_pass=_counts(0),
        posterior_pass=_counts(0), observer_pass=observer,
        patch_pass=_counts(0), target_oracle_pass=_counts(0),
        controls_specific=False, required=4,
    ) == "analytic_reference_precision_limited"


def test_source_campaign_contract_replays() -> None:
    source = Path(
        "data/experiments/tinyllm_calibration_degradation_causal/"
        "20260807_d8_existing_checkpoints/campaign_results.json"
    )
    if not source.is_file():
        pytest.skip("retained source campaign is not present")
    loaded = readout._load_sources(readout.ReadoutDecompositionConfig(device="cpu"))
    _, _, _, _, _, datasets, noise_sha256 = loaded
    assert noise_sha256 == readout.SOURCE_NOISE_SHA256
    assert all(readout.evaluation_key(regime, level) in datasets
               for regime in readout.REGIMES for level in readout.LEVELS)


def test_branch_labels_are_transient_fit_metadata() -> None:
    cosine = torch.tensor([0.1, 0.1])
    labels = torch.tensor([0.0, 1.0])
    returned = readout._attach_branch_labels(cosine, labels)
    assert returned is cosine
    assert torch.equal(getattr(returned, "_branch_labels"), labels)
