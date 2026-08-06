import numpy as np
import torch

from experiments.structure_net.tinyllm_degree_defect_cobordism import (
    fixed_nuisance_sensor_values,
    moment_record,
    soft_posterior_moment,
    soft_sensor_token_embeddings,
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


def test_fixed_nuisance_slice_is_periodic_and_smooth():
    task = _task()
    phases = np.array([0.0, 2.0 * np.pi])
    values = fixed_nuisance_sensor_values(task, phases)

    assert np.allclose(values[0], values[1])
    assert np.isfinite(values).all()


def test_soft_token_embedding_lift_is_continuous_at_hard_bin_boundary():
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
    )
    boundary = -task.quantization_limit + (
        2.0 * task.quantization_limit / task.value_bins
    )
    below = np.full((1, task.sensor_steps, 3), boundary - 1e-7)
    above = np.full((1, task.sensor_steps, 3), boundary + 1e-7)
    first = soft_sensor_token_embeddings(
        model, task, below, torch.device("cpu")
    )
    second = soft_sensor_token_embeddings(
        model, task, above, torch.device("cpu")
    )

    assert torch.max(torch.abs(first - second)) < 1e-5


def test_moment_record_recovers_degree_one():
    phases = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    record = moment_record(np.exp(1j * phases), phases)

    assert record["rounded_degree"] == 1
    assert record["phase_alignment"] > 0.999
    assert record["sampling_resolved"] is True


def test_soft_posterior_moment_chunks_large_phase_grid():
    task = _task()
    model = TinyLLMModel(
        TinyLLMConfig(
            block_size=task.sequence_length,
            vocab_size=task.vocab_size,
            n_layer=1,
            n_head=2,
            n_embd=8,
            initialization_seed=4,
        )
    )
    phases = np.linspace(0.0, 2.0 * np.pi, 1_025, endpoint=False)
    moments = soft_posterior_moment(model, task, phases, torch.device("cpu"))

    assert moments.shape == (1_025,)
    assert np.isfinite(moments).all()
