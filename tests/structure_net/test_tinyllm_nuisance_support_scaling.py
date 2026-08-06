import numpy as np
import pytest
import torch

from experiments.structure_net.tinyllm_nuisance_support_scaling import (
    NuisanceScalingConfig,
    SUPPORTS,
    TRAINING_ARMS,
    _experiments,
    _nuisance_values,
    _quotient_contrastive_loss,
    fiber_geometry_record,
    generate_geometry_dataset,
    generate_paired_dataset,
    nuisance_axes,
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


def _config(**changes) -> NuisanceScalingConfig:
    values = {
        "preset": "tiny",
        "seeds": (7,),
        "supports": ("N0",),
        "training_arms": ("standard_final",),
        "training_steps": 2,
        "train_samples": 32,
        "batch_size": 8,
        "probe_train_samples": 64,
        "probe_validation_samples": 32,
        "probe_test_samples": 32,
        "probe_steps": 20,
        "probe_width": 16,
        "probe_batch_size": 16,
        "fiber_replicates": 3,
        "allow_underpowered": True,
    }
    values.update(changes)
    return NuisanceScalingConfig(**values)


def test_support_axes_are_strictly_nested():
    axes = [set(nuisance_axes(support)) for support in SUPPORTS]

    assert axes[0] < axes[1] < axes[2] < axes[3]
    assert axes[3] == {
        "amplitude",
        "offset",
        "orientation",
        "angular_speed",
        "harmonic",
        "direction",
        "noise",
    }


def test_composition_split_reverses_the_trained_amplitude_orientation_relation():
    interpolation = _nuisance_values(
        "N3", "interpolation", 256, np.random.default_rng(11)
    )
    composition = _nuisance_values(
        "N3", "composition", 256, np.random.default_rng(11)
    )
    interpolation_product = (
        (interpolation["amplitude"][:, 0, 0] - 1.0)
        * interpolation["orientation"][:, 0]
    )
    composition_product = (
        (composition["amplitude"][:, 0, 0] - 1.0)
        * composition["orientation"][:, 0]
    )

    assert np.all(interpolation_product < 0.0)
    assert np.all(composition_product > 0.0)


def test_paired_dataset_has_exact_cosine_fibers_and_fixed_sample_count():
    dataset = generate_paired_dataset(
        _task(), sample_count=32, seed=13, support="N3", shuffle=False
    )

    assert len(dataset.circle.input_ids) == 32
    assert dataset.circle.input_ids.shape[1] == _task().sequence_length
    assert torch.equal(dataset.fiber.fiber_id[0::2], dataset.fiber.fiber_id[1::2])
    assert torch.allclose(dataset.fiber.cosine[0::2], dataset.fiber.cosine[1::2])
    assert torch.allclose(
        torch.cos(dataset.fiber.phase), dataset.fiber.cosine, atol=1e-6
    )


def test_quotient_contrastive_loss_rewards_collapsed_matched_branches():
    collapsed = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )[:, None, :]
    separated = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )[:, None, :]

    collapsed_loss, collapsed_positive, collapsed_negative = (
        _quotient_contrastive_loss(collapsed, 0.5)
    )
    separated_loss, separated_positive, _ = _quotient_contrastive_loss(
        separated, 0.5
    )

    assert collapsed_positive < separated_positive
    assert collapsed_negative > 0.5
    assert collapsed_loss < separated_loss


def test_fiber_gamma_detects_hidden_branch_sheet_separation():
    dataset = generate_geometry_dataset(
        _task(),
        cosine_values=(-0.5, 0.0, 0.5),
        replicates=6,
        seed=17,
        support="N1",
        regime="interpolation",
    )
    generator = np.random.default_rng(19)
    nuisance = generator.normal(0.0, 0.05, (len(dataset.branch), 3))
    collapsed = nuisance.astype(np.float32)
    separated = collapsed.copy()
    separated[:, 0] += 3.0 * dataset.branch.numpy()
    posteriors = np.tile(np.array([[0.2, 0.3, 0.5]]), (len(dataset.branch), 1))

    collapsed_record = fiber_geometry_record(collapsed, posteriors, dataset)
    separated_record = fiber_geometry_record(separated, posteriors, dataset)

    assert collapsed_record["euclidean"]["median_gamma"] < 1.25
    assert separated_record["euclidean"]["median_gamma"] > 1.25
    assert collapsed_record["fisher_rao"]["median_gamma"] == pytest.approx(1.0)


def test_preregistered_configuration_requires_five_seeds():
    with pytest.raises(ValueError, match="at least five seeds"):
        NuisanceScalingConfig(seeds=(7,))


def test_full_factorial_contains_eighty_independent_cells(tmp_path):
    config = NuisanceScalingConfig(device_ids=(-1,))
    experiments = _experiments(config, CircleTaskConfig(), tmp_path)

    assert len(experiments) == len(SUPPORTS) * len(TRAINING_ARMS) * 5 == 80
    assert len({experiment.id for experiment in experiments}) == 80


def test_primary_design_contains_only_coverage_curve_and_n3_objectives(tmp_path):
    config = NuisanceScalingConfig(device_ids=(-1,), design="primary35")
    experiments = _experiments(config, CircleTaskConfig(), tmp_path)
    cells = {
        (
            experiment.parameters["support"],
            experiment.parameters["training_arm"],
        )
        for experiment in experiments
    }

    assert len(experiments) == 35
    assert cells == {
        ("N0", "standard_final"),
        ("N1", "standard_final"),
        ("N2", "standard_final"),
        ("N3", "standard_final"),
        ("N3", "discrete_multi_exit"),
        ("N3", "continuous_gate"),
        ("N3", "quotient_contrastive"),
    }
