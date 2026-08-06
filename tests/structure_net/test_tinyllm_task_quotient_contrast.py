from pathlib import Path

import numpy as np
import torch

from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    generate_circle_dataset,
)
from experiments.structure_net.tinyllm_task_quotient_contrast import (
    QuotientContrastConfig,
    aggregate_contrast,
    run_quotient_contrast,
    semantic_coordinate,
    target_class_baselines,
    with_quotient_targets,
)


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


def test_interval_quotient_changes_only_targets_and_identifies_opposite_phases():
    task = _task()
    future = np.array([0.4, 2.0 * np.pi - 0.4])
    phases = future - task.future_delta
    base = generate_circle_dataset(
        task,
        sample_count=2,
        seed=5,
        phases=phases,
        directions=np.ones(2),
    )
    interval = with_quotient_targets(base, task, "cosine_interval")

    assert torch.equal(interval.input_ids, base.input_ids)
    assert np.allclose(np.cos(semantic_coordinate(interval, task)), np.cos(future))
    assert torch.allclose(interval.target_posteriors[0], interval.target_posteriors[1])


def test_interval_baselines_account_for_nonuniform_exact_bin_priors():
    task = _task()
    campaign = QuotientContrastConfig(
        presets=("tiny",),
        seeds=(3,),
        quotients=("cosine_interval",),
        conditions=("trained",),
        training_steps=1,
        checkpoints=(0, 1),
        device="cpu",
    )

    baseline = target_class_baselines(task, campaign)["cosine_interval"]

    assert baseline["mean_uniform_random_accuracy"] == 1.0 / task.phase_bins
    assert baseline["mean_empirical_majority_class_accuracy"] >= (
        baseline["mean_empirical_prior_random_accuracy"]
    )
    assert sum(baseline["per_seed"][0]["counts"]) == task.evaluation_samples


def test_tiny_quotient_contrast_runs_both_task_topologies(tmp_path: Path):
    task = _task()
    campaign = QuotientContrastConfig(
        presets=("tiny",),
        seeds=(3,),
        quotients=("phase_circle", "cosine_interval"),
        conditions=("trained",),
        training_steps=1,
        checkpoints=(0, 1),
        batch_size=4,
        bootstrap_repeats=2,
        fisher_neighbors=3,
        device="cpu",
        save_primary_checkpoints=False,
    )
    bundle = run_quotient_contrast(task, campaign, tmp_path)

    assert bundle["status"] == "completed"
    assert len(bundle["runs"]) == 2
    assert {run["quotient"] for run in bundle["runs"]} == {
        "phase_circle",
        "cosine_interval",
    }
    for run in bundle["runs"]:
        final = run["checkpoints"][-1]
        assert "semantic_map" in final["quotient_analysis"]
        assert "nuisance_collapse" in final
        assert "final_residual_pullback_fisher" in final["quotient_analysis"]


def test_aggregate_contrast_requires_map_aware_task_separation():
    def run(quotient, condition, metrics):
        return {
            "preset": "d6",
            "quotient": quotient,
            "condition": condition,
            "primary_metrics": metrics,
        }

    common = {
        "in_domain_exact_bin_accuracy": 0.7,
        "posterior_normalized_h1_lifetime": 0.0,
    }
    runs = [
        run(
            "phase_circle",
            "trained",
            {**common, "posterior_normalized_h1_lifetime": 0.8, "map_alignment": 0.95, "map_winding_degree": 1.0},
        ),
        run(
            "cosine_interval",
            "trained",
            {**common, "map_pearson_correlation": 0.96},
        ),
        run(
            "phase_circle",
            "label_shuffled",
            {**common, "in_domain_exact_bin_accuracy": 0.1, "map_alignment": 0.1, "map_winding_degree": 0.0},
        ),
        run(
            "cosine_interval",
            "label_shuffled",
            {**common, "in_domain_exact_bin_accuracy": 0.1, "map_pearson_correlation": 0.0},
        ),
    ]
    aggregate = aggregate_contrast(runs)["d6"]

    assert aggregate["criteria_passed"] is True
