import numpy as np
import torch

from experiments.structure_net.tinyllm_internal_quotient_probe import (
    ProbeCampaignConfig,
    fit_probe_suite,
    generate_matched_fiber_dataset,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


def _task() -> CircleTaskConfig:
    return CircleTaskConfig(
        phase_bins=8,
        sensor_steps=4,
        value_bins=8,
        vocab_size=512,
        answer_token_start=400,
        train_samples=32,
        evaluation_samples=16,
        phase_grid_points=12,
        nuisance_phase_count=2,
        nuisance_replicates=4,
    )


def test_matched_fibers_balance_branch_at_identical_cosine():
    dataset = generate_matched_fiber_dataset(
        _task(), sample_count=32, seed=5, family="training_support"
    )
    cosine = dataset.cosine.numpy()
    branch = dataset.branch.numpy()

    assert branch.mean() == 0.5
    for value in np.unique(cosine):
        assert sorted(branch[cosine == value].tolist()) == [0.0, 1.0]


def test_disjoint_nuisance_family_changes_serialized_inputs():
    task = _task()
    base = generate_matched_fiber_dataset(
        task, sample_count=32, seed=7, family="training_support"
    )
    shifted = generate_matched_fiber_dataset(
        task, sample_count=32, seed=7, family="disjoint_third_harmonic_drift"
    )

    assert torch.equal(base.cosine, shifted.cosine)
    assert torch.equal(base.branch, shifted.branch)
    assert not torch.equal(base.input_ids, shifted.input_ids)


def test_nonlinear_probe_recovers_known_synthetic_branch_and_cosine():
    task = _task()
    train = generate_matched_fiber_dataset(
        task, sample_count=128, seed=11, family="training_support"
    )
    validation = generate_matched_fiber_dataset(
        task, sample_count=64, seed=13, family="training_support"
    )
    test = generate_matched_fiber_dataset(
        task, sample_count=64, seed=17, family="training_support"
    )

    def features(dataset):
        phase = dataset.phase.numpy()
        return np.column_stack(
            (np.cos(phase), np.sin(phase), np.cos(2.0 * phase))
        ).astype(np.float32)

    campaign = ProbeCampaignConfig(
        train_samples=128,
        validation_samples=64,
        test_samples=64,
        probe_batch_size=32,
        probe_steps=100,
        probe_width=16,
        validation_interval=10,
        early_stopping_patience=5,
        device="cpu",
    )
    result = fit_probe_suite(
        features(train),
        features(validation),
        {"in_distribution": features(test)},
        train,
        validation,
        {"in_distribution": test},
        campaign,
        torch.device("cpu"),
        seed=19,
    )["evaluations"]["in_distribution"]

    assert result["conditional_branch_accuracy"] > 0.9
    assert result["cosine_pearson"] > 0.95
    assert result["phase_alignment"] > 0.9
