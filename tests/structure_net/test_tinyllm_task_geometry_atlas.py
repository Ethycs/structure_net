import numpy as np
import torch
from pathlib import Path

from experiments.structure_net.tinyllm_internal_quotient_probe import (
    ProbeCampaignConfig,
    generate_matched_fiber_dataset,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from experiments.structure_net.tinyllm_task_geometry_atlas import (
    _reference_distances,
    analyze_checkpoint,
    extract_sublayer_representations,
    stage_labels,
)
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


def test_stage_labels_split_attention_and_mlp():
    assert stage_labels(2) == [
        "query_embedding",
        "block_1_post_attention",
        "block_1_post_mlp",
        "block_2_post_attention",
        "block_2_post_mlp",
    ]


def test_manual_sublayer_capture_matches_normal_forward_block_outputs():
    task = _task()
    dataset = generate_matched_fiber_dataset(
        task, sample_count=8, seed=5, family="training_support"
    )
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
    captured = extract_sublayer_representations(
        model, dataset.input_ids, torch.device("cpu"), batch_size=4
    )
    output = model(
        dataset.input_ids,
        output_hidden_states=True,
        return_dict=True,
    )

    assert len(captured) == 5
    assert output.hidden_states is not None
    assert np.allclose(captured[2], output.hidden_states[0][:, -1, :].detach().numpy())
    assert np.allclose(captured[4], output.hidden_states[1][:, -1, :].detach().numpy())


def test_reference_distances_distinguish_circle_and_cosine_quotient():
    phase = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    circle = _reference_distances(phase, "phase_circle")
    interval = _reference_distances(phase, "cosine_interval")

    assert circle[1, -1] > 0.0
    assert interval[1, -1] == 0.0


def test_tiny_checkpoint_produces_every_sublayer_atlas(tmp_path: Path):
    task = _task()
    checkpoint = tmp_path / "tiny.pt"
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
    model.save_checkpoint(
        checkpoint,
        metadata={"quotient": "phase_circle", "seed": 3},
    )
    campaign = ProbeCampaignConfig(
        train_samples=16,
        validation_samples=8,
        test_samples=8,
        activation_batch_size=8,
        probe_batch_size=8,
        probe_steps=1,
        probe_width=8,
        validation_interval=1,
        early_stopping_patience=1,
        phase_grid_points=12,
        device="cpu",
    )
    datasets = {
        "train": generate_matched_fiber_dataset(
            task, sample_count=16, seed=1, family="training_support"
        ),
        "validation": generate_matched_fiber_dataset(
            task, sample_count=8, seed=2, family="training_support"
        ),
        "in_distribution": generate_matched_fiber_dataset(
            task, sample_count=8, seed=3, family="training_support"
        ),
        "disjoint_nuisance": generate_matched_fiber_dataset(
            task,
            sample_count=8,
            seed=4,
            family="disjoint_third_harmonic_drift",
        ),
    }

    result = analyze_checkpoint(
        checkpoint, task, campaign, torch.device("cpu"), datasets
    )

    assert result["model_frozen"] is True
    assert len(result["stagewise_atlas"]) == 5
    assert result["stagewise_atlas"][-1]["stage"] == "block_2_post_mlp"
