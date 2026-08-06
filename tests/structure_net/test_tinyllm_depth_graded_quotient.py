import numpy as np
import torch

from experiments.structure_net.tinyllm_depth_graded_quotient import (
    DepthGradedCampaignConfig,
    _cosine_map_record,
    analyze_depth_diagram,
    depth_grid,
    soft_depth_outputs,
    training_depth_schedule,
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
        phase_grid_points=32,
        nuisance_phase_count=2,
        nuisance_replicates=4,
    )


def _model(task: CircleTaskConfig) -> TinyLLMModel:
    return TinyLLMModel(
        TinyLLMConfig(
            block_size=task.sequence_length,
            vocab_size=task.vocab_size,
            n_layer=2,
            n_head=2,
            n_embd=16,
            initialization_seed=3,
        )
    )


def test_depth_grid_includes_exact_integer_and_fractional_depths():
    values = depth_grid(2, 0.25)

    assert np.array_equal(values, np.arange(0.0, 2.25, 0.25))


def test_depth_training_schedules_are_deterministic_and_matched():
    discrete = training_depth_schedule(
        "discrete_multi_exit",
        layer_count=8,
        steps=10,
        depths_per_batch=4,
        seed=9,
    )
    continuous = training_depth_schedule(
        "continuous_gate",
        layer_count=8,
        steps=10,
        depths_per_batch=4,
        seed=9,
    )

    assert np.array_equal(
        discrete,
        training_depth_schedule(
            "discrete_multi_exit",
            layer_count=8,
            steps=10,
            depths_per_batch=4,
            seed=9,
        ),
    )
    assert np.all(discrete[:, 0] == 8.0)
    assert np.all(discrete[:, 1] == 1.0)
    assert np.all(discrete == np.floor(discrete))
    assert np.all(continuous[:, 0] == 8.0)
    assert np.all(continuous[:, 1] == 1.0)
    assert np.any(continuous[:, 2:] != np.floor(continuous[:, 2:]))


def test_soft_depth_output_varies_continuously_across_gate():
    task = _task()
    model = _model(task).eval()
    phases = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    below, below_residual = soft_depth_outputs(
        model, task, phases, 0.4999, torch.device("cpu")
    )
    above, above_residual = soft_depth_outputs(
        model, task, phases, 0.5001, torch.device("cpu")
    )

    assert np.max(np.abs(above - below)) < 1e-3
    assert np.max(np.abs(above_residual - below_residual)) < 1e-2


def test_phase_depth_diagram_obeys_discrete_charge_identity():
    task = _task()
    model = _model(task).eval()
    campaign = DepthGradedCampaignConfig(
        preset="tiny",
        training_steps=1,
        checkpoints=(0, 1),
        depth_step=0.5,
        phase_points=32,
        max_phase_points=64,
        save_checkpoints=False,
        device="cpu",
    )
    diagram, arrays = analyze_depth_diagram(
        model,
        task,
        depth_grid(2, 0.5),
        "phase_circle",
        campaign,
        torch.device("cpu"),
        final_checkpoint=False,
    )

    assert len(diagram["depth_cells"]) == 5
    assert diagram["depth_defect_charge"]["charge_identity_holds"] is True
    assert arrays["posteriors"].shape[:2] == (5, diagram["phase_grid_points"])


def test_cosine_map_handles_constant_depth_zero_posterior():
    task = _task()
    phases = (np.arange(32) + 0.5) * 2.0 * np.pi / 32
    posterior = np.full((32, task.phase_bins), 1.0 / task.phase_bins)
    record = _cosine_map_record(posterior, phases, task)

    assert record["pearson_correlation"] == 0.0
    assert record["spearman_correlation"] == 0.0
    assert np.isfinite(list(record.values())).all()
