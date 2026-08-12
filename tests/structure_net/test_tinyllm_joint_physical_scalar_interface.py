from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
from experiments.structure_net.tinyllm_joint_physical_scalar_interface import (
    ARMS,
    CONDITIONS,
    PRESETS,
    SEEDS,
    JointPhysicalInterfaceConfig,
    PhysicalScalarInterface,
    _trainable_parameters,
    aggregate_details,
    endpoint_pass,
    interval_posterior_unclipped,
    pair_preserving_target_permutation,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from structure_net.components.models import TinyLLMModel


def test_primary_configuration_is_locked() -> None:
    config = JointPhysicalInterfaceConfig()
    assert config.presets == PRESETS
    assert config.conditions == CONDITIONS
    assert config.seeds == SEEDS
    assert config.training_steps == 600
    assert config.required_seed_passes == 4
    with pytest.raises(ValueError, match="primary joint-interface configuration changed"):
        JointPhysicalInterfaceConfig(training_steps=599)


def test_pair_shuffle_preserves_sheets_and_is_deterministic() -> None:
    first = pair_preserving_target_permutation(
        20, "d6", "learned_calibrated_equivariant", 7, 20_260_811
    )
    second = pair_preserving_target_permutation(
        20, "d6", "learned_calibrated_equivariant", 7, 20_260_811
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, torch.arange(20))
    paired = first.reshape(-1, 2)
    assert torch.equal(paired[:, 1], paired[:, 0] + 1)
    assert torch.equal(paired[:, 0] % 2, torch.zeros(10, dtype=torch.long))
    assert sorted(first.tolist()) == list(range(20))


def test_unclipped_interval_decoder_is_exact_on_physical_domain() -> None:
    scalar = torch.linspace(-1.0, 1.0, 257, dtype=torch.float64)
    observed = interval_posterior_unclipped(scalar, 16)
    centers = torch.linspace(-1.0, 1.0, 16, dtype=torch.float64)
    expected = torch.softmax(
        -0.5 * ((centers[None] - scalar[:, None]) / (2.0 / 15.0)).square(),
        dim=-1,
    )
    assert torch.equal(observed, expected)
    assert observed.argmax(1)[0].item() == 0
    assert observed.argmax(1)[-1].item() == 15


def test_endpoint_thresholds_are_joint_and_inclusive() -> None:
    config = JointPhysicalInterfaceConfig(
        allow_underpowered=True, seeds=(7,), required_seed_passes=1
    )
    record = {
        "scalar_metrics": {"cosine_pearson": 0.90},
        "conditional_branch": {
            "balanced_accuracy": 0.55,
            "conditional_log_loss_gain_over_cosine_only": 0.02,
        },
        "task_gate": True,
    }
    assert endpoint_pass(record, config)
    for key, value in (
        ("cosine_pearson", 0.899),
        ("balanced_accuracy", 0.551),
        ("conditional_log_loss_gain_over_cosine_only", 0.021),
    ):
        failed = {
            "scalar_metrics": dict(record["scalar_metrics"]),
            "conditional_branch": dict(record["conditional_branch"]),
            "task_gate": True,
        }
        destination = (
            failed["scalar_metrics"]
            if key == "cosine_pearson"
            else failed["conditional_branch"]
        )
        destination[key] = value
        assert not endpoint_pass(failed, config)
    failed_task = dict(record)
    failed_task["task_gate"] = False
    assert not endpoint_pass(failed_task, config)


def test_gradient_routing_excludes_the_tinyllm_backbone() -> None:
    task = CircleTaskConfig(sensor_steps=4)
    config = calibrated.CalibratedFrontendConfig(
        preset="d6", seeds=(7,), vector_channels=4, allow_underpowered=True
    )
    model = TinyLLMModel(calibrated._model_config("d6", task, 7))
    system = calibrated.CalibratedTinyLLM(
        model, "learned_calibrated_equivariant", task, config
    )
    for parameter in system.parameters():
        parameter.requires_grad_(False)
    assert system.scalar_embedding is not None
    assert system.encoder is not None
    system.scalar_embedding.requires_grad_(True)
    system.encoder.requires_grad_(True)
    interface = PhysicalScalarInterface(system)
    with torch.no_grad():
        interface.final_scalar.weight.fill_(0.1)
    selected, count = _trainable_parameters(interface)
    names = [name for name, _ in selected]
    assert count > 0
    assert names
    assert all(
        name.startswith(
            ("system.scalar_embedding.", "system.encoder.", "final_scalar.")
        )
        for name in names
    )
    assert all(not parameter.requires_grad for parameter in system.model.parameters())

    dataset = calibrated.generate_calibrated_dataset(
        task, sample_count=8, seed=1008, shuffle=False
    )
    sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    values = interface.forward_scalars(
        dataset.paired.circle.input_ids, sensor, dataset.calibration
    )
    target = dataset.paired.fiber.cosine[:, None]
    loss = (values["frontend"] - target).square().mean()
    loss = loss + (values["full"] - target).square().mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in system.model.parameters())
    assert any(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for _, parameter in selected
    )


def _details(
    learned_passes: int,
    analytic_passes: int = 5,
    shuffled_passes: int = 0,
) -> list[dict]:
    values = []
    for preset in PRESETS:
        for condition in CONDITIONS:
            true_count = (
                analytic_passes
                if condition == "analytic_calibrated"
                else learned_passes
            )
            for index, seed in enumerate(SEEDS):
                values.append(
                    {
                        "preset": preset,
                        "condition": condition,
                        "seed": seed,
                        "gates": {
                            "physical_true_joint_seed_pass": index < true_count,
                            "pair_shuffled_joint_seed_pass": index < shuffled_passes,
                            "validity": True,
                        },
                    }
                )
    return values


def test_aggregate_requires_four_joint_seeds_in_every_learned_stratum() -> None:
    config = JointPhysicalInterfaceConfig()
    passing = aggregate_details(_details(learned_passes=4), config)
    assert passing["primary_hypothesis_pass"]
    assert passing["stop_before_transformer_finetuning"]
    assert not passing["full_interface_extension_licensed"]
    assert (
        passing["classification"]
        == "frozen_backbone_joint_physical_interface_architecture_stable"
    )

    failing = aggregate_details(_details(learned_passes=3), config)
    assert not failing["primary_hypothesis_pass"]
    assert failing["full_interface_extension_licensed"]
    assert failing["classification"] == "frozen_backbone_joint_interface_insufficient"


def test_aggregate_rejects_failed_positive_or_specificity_controls() -> None:
    config = JointPhysicalInterfaceConfig()
    analytic_failure = aggregate_details(
        _details(learned_passes=5, analytic_passes=3), config
    )
    assert analytic_failure["classification"] == "analytic_positive_control_failed"
    assert not analytic_failure["full_interface_extension_licensed"]

    shuffled_failure = aggregate_details(
        _details(learned_passes=5, shuffled_passes=2), config
    )
    assert shuffled_failure["classification"] == "specificity_control_failed"
    assert not shuffled_failure["primary_hypothesis_pass"]


def test_all_declared_arms_are_covered() -> None:
    assert ARMS == ("physical_true", "pair_shuffled")
    assert asdict(JointPhysicalInterfaceConfig())["sensor_loss_weight"] == 1.0
