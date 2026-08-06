from pathlib import Path

import numpy as np
import torch

from experiments.structure_net.tinyllm_predictive_circle import (
    CircleCampaignConfig,
    CircleTaskConfig,
    generate_circle_dataset,
    run_campaign,
    shuffle_time_steps,
)


def test_circle_generator_is_reproducible_and_preserves_task_contract():
    config = CircleTaskConfig(
        phase_bins=8,
        sensor_steps=4,
        value_bins=8,
        train_samples=32,
        evaluation_samples=16,
        phase_grid_points=12,
        nuisance_phase_count=2,
        nuisance_replicates=4,
    )
    phases = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    first = generate_circle_dataset(
        config,
        sample_count=12,
        seed=3,
        phases=phases,
        directions=np.ones(12),
    )
    second = generate_circle_dataset(
        config,
        sample_count=12,
        seed=3,
        phases=phases,
        directions=np.ones(12),
    )

    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.target_posteriors, second.target_posteriors)
    assert first.input_ids.shape == (12, config.sequence_length)
    assert torch.allclose(first.target_posteriors.sum(1), torch.ones(12))
    assert first.target_bins.unique().numel() == config.phase_bins


def test_time_shuffle_moves_whole_three_channel_blocks():
    config = CircleTaskConfig(sensor_steps=5)
    dataset = generate_circle_dataset(config, sample_count=3, seed=5)
    shuffled = shuffle_time_steps(dataset.input_ids, config, seed=7)

    original_blocks = dataset.input_ids[:, 1:-1].reshape(3, config.sensor_steps, 3)
    shuffled_blocks = shuffled[:, 1:-1].reshape(3, config.sensor_steps, 3)
    assert torch.equal(dataset.input_ids[:, [0, -1]], shuffled[:, [0, -1]])
    for row in range(3):
        assert sorted(map(tuple, original_blocks[row].tolist())) == sorted(
            map(tuple, shuffled_blocks[row].tolist())
        )


def test_tiny_campaign_runs_all_analysis_stages(tmp_path: Path):
    task = CircleTaskConfig(
        phase_bins=8,
        sensor_steps=4,
        value_bins=8,
        train_samples=32,
        evaluation_samples=16,
        phase_grid_points=12,
        nuisance_phase_count=2,
        nuisance_replicates=4,
    )
    campaign = CircleCampaignConfig(
        presets=("tiny",),
        seeds=(3,),
        conditions=("trained",),
        training_steps=1,
        checkpoints=(0, 1),
        batch_size=4,
        bootstrap_repeats=2,
        bootstrap_fraction=0.75,
        fisher_neighbors=3,
        device="cpu",
        save_primary_checkpoints=False,
    )

    bundle = run_campaign(task, campaign, tmp_path)
    run = bundle["runs"][0]
    final = run["checkpoints"][-1]

    assert bundle["status"] == "completed"
    assert bundle["ground_truth_semantic_quotient"]["space"] == "S1"
    assert len(run["checkpoint_diagram_tracking"]) == 1
    assert "final_residual_pullback_fisher" in final["phase_topology"]
    assert "decision_margin_sublevel_h0" in final["phase_topology"]
    assert "nuisance_fibers" in final
    assert (tmp_path / "results.json").is_file()
