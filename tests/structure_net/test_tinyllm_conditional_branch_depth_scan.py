import numpy as np
import torch

from experiments.structure_net.tinyllm_conditional_branch_depth_scan import (
    BranchDepthCampaignConfig,
    depths_for_arm,
    extract_depth_representations,
    fiber_component_record,
    fit_conditional_probe,
    generate_fiber_component_dataset,
)
from experiments.structure_net.tinyllm_internal_quotient_probe import (
    generate_matched_fiber_dataset,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from structure_net.components.models import TinyLLMConfig, TinyLLMModel


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


def _campaign(**changes) -> BranchDepthCampaignConfig:
    values = dict(
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
    values.update(changes)
    return BranchDepthCampaignConfig(**values)


def test_standard_depth_grid_adds_fine_front_points_only_to_standard_arm():
    campaign = _campaign()

    standard = depths_for_arm("standard_final", campaign, 8)
    joint = depths_for_arm("continuous_gate", campaign, 8)

    assert 1.85 in standard
    assert 1.85 not in joint
    assert 0.005 in standard and 0.005 in joint


def test_depth_extractor_matches_public_real_depth_residual():
    task = _task()
    model = TinyLLMModel(
        TinyLLMConfig(
            block_size=task.sequence_length,
            vocab_size=task.vocab_size,
            n_layer=2,
            n_head=2,
            n_embd=16,
            initialization_seed=3,
        )
    ).eval()
    dataset = generate_matched_fiber_dataset(
        task, sample_count=8, seed=5, family="training_support"
    )
    depths = (0.0, 0.1, 1.0, 1.75, 2.0)

    captured = extract_depth_representations(
        model, dataset.input_ids, depths, torch.device("cpu"), batch_size=4
    )

    for depth in depths:
        expected = model.residual_at_depth(dataset.input_ids, depth)[:, -1, :]
        assert np.allclose(captured[depth], expected.detach().numpy(), atol=1e-6)


def test_conditional_probe_recovers_branch_beyond_exact_matched_cosine():
    task = _task()
    datasets = {
        "train": generate_matched_fiber_dataset(
            task, sample_count=128, seed=11, family="training_support"
        ),
        "validation": generate_matched_fiber_dataset(
            task, sample_count=64, seed=13, family="training_support"
        ),
        "test": generate_matched_fiber_dataset(
            task, sample_count=64, seed=17, family="training_support"
        ),
    }

    def features(dataset):
        phase = dataset.phase.numpy()
        return np.column_stack(
            (np.cos(phase), np.sin(phase), np.cos(2.0 * phase))
        ).astype(np.float32)

    result = fit_conditional_probe(
        features(datasets["train"]),
        features(datasets["validation"]),
        {"test": features(datasets["test"])},
        datasets["train"],
        datasets["validation"],
        {"test": datasets["test"]},
        _campaign(),
        torch.device("cpu"),
        kind="nonlinear",
        seed=19,
    )["evaluations"]["test"]

    assert result["balanced_accuracy"] > 0.9
    assert result["cosine_pearson"] > 0.95


def test_fiber_component_proxy_distinguishes_separated_and_collapsed_branches():
    dataset = generate_fiber_component_dataset(
        _task(),
        cosine_values=(-0.5, 0.0, 0.5),
        replicates=6,
        family="training_support",
        seed=23,
    )
    generator = np.random.default_rng(29)
    nuisance = generator.normal(0.0, 0.03, (len(dataset.branch), 2))
    separated = nuisance.copy()
    separated[:, 0] += 3.0 * dataset.branch.numpy()
    collapsed = nuisance.copy()

    retained = fiber_component_record(separated.astype(np.float32), dataset)
    quotient = fiber_component_record(collapsed.astype(np.float32), dataset)

    assert retained["majority_component_count_proxy"] == 2
    assert quotient["majority_component_count_proxy"] == 1
    assert retained["mean_branch_silhouette"] > quotient["mean_branch_silhouette"]
