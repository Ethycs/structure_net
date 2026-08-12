from __future__ import annotations

import pytest
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
from experiments.structure_net.tinyllm_joint_interface_gradient_attribution import (
    CONDITIONS,
    HYPOTHESIS_ID,
    PRESETS,
    SCHEMA_VERSION,
    SEEDS,
    GradientAttributionConfig,
    aggregate_details,
    block_gradient_record,
    gradient_cosine,
    gradient_snapshot,
)
from experiments.structure_net.tinyllm_joint_physical_scalar_interface import (
    PhysicalScalarInterface,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig
from structure_net.components.models import TinyLLMModel


def test_primary_configuration_is_locked() -> None:
    config = GradientAttributionConfig()
    assert SCHEMA_VERSION.endswith(".v2")
    assert HYPOTHESIS_ID.endswith("-v2")
    assert config.schedule_steps == (0, 599)
    assert config.gradient_clip == 1.0
    assert config.global_clip_maximum == 0.10
    assert config.additivity_tolerance == 2e-5
    assert config.relative_additivity_tolerance == 1e-6
    with pytest.raises(ValueError, match="first and last"):
        GradientAttributionConfig(schedule_steps=(0, 598), allow_underpowered=True)
    with pytest.raises(ValueError, match="configuration changed"):
        GradientAttributionConfig(additivity_tolerance=1e-5)
    with pytest.raises(ValueError, match="configuration changed"):
        GradientAttributionConfig(relative_additivity_tolerance=2e-6)


def test_block_record_detects_conflict_and_cross_block_suppression() -> None:
    vectors = {
        "sensor": torch.tensor([1.0, 0.0]),
        "final": torch.tensor([-2.0, 0.0]),
        "task": torch.tensor([0.0, 0.0]),
    }
    record = block_gradient_record(vectors, 1.0, 0.05)
    assert record["block_clip_coefficient"] == 1.0
    assert record["cross_block_suppression"] == pytest.approx(0.05)
    assert record["sensor_descent_ratio"] == pytest.approx(-1.0)
    assert record["objective_cosines"]["sensor_downstream"] == pytest.approx(-1.0)
    assert gradient_cosine(torch.zeros(2), torch.ones(2)) is None


def test_gradient_snapshot_is_additive_on_real_interface() -> None:
    task = CircleTaskConfig(sensor_steps=4)
    config = calibrated.CalibratedFrontendConfig(
        preset="d6", seeds=(7,), vector_channels=4, allow_underpowered=True
    )
    model = TinyLLMModel(calibrated._model_config("d6", task, 7))
    system = calibrated.CalibratedTinyLLM(
        model, "learned_calibrated_equivariant", task, config
    )
    for parameter in system.model.parameters():
        parameter.requires_grad_(False)
    interface = PhysicalScalarInterface(system)
    with torch.no_grad():
        interface.final_scalar.weight.fill_(0.01)
    dataset = calibrated.generate_calibrated_dataset(
        task, sample_count=8, seed=1008, shuffle=False
    )
    sensor = calibrated.decode_sensor_tokens(dataset.paired.circle.input_ids, task)
    record, arrays = gradient_snapshot(
        interface,
        dataset.paired.circle.input_ids,
        sensor,
        dataset.calibration,
        dataset.paired.fiber.cosine,
        task,
        1.0,
    )
    assert record["gradient_additivity_maximum_absolute_error"] <= 2e-5
    assert record["gradient_additivity_maximum_relative_error"] <= 1e-6
    assert record["blocks"]["encoder"]["objective_norms"]["sensor"] > 0
    assert record["blocks"]["final_scalar"]["objective_norms"]["sensor"] == 0
    assert arrays["gradient__encoder__sensor"].size > 0


def _details(initial_count: int, persistent_count: int) -> list[dict]:
    details = []
    for preset in PRESETS:
        for condition in CONDITIONS:
            learned = condition == "learned_calibrated_equivariant"
            for index, seed in enumerate(SEEDS):
                details.append(
                    {
                        "preset": preset,
                        "condition": condition,
                        "seed": seed,
                        "gates": {
                            "validity": True,
                            "initial_cross_block_starvation": (
                                index < initial_count if learned else None
                            ),
                            "persistent_learned_state_conflict": (
                                index < persistent_count if learned else None
                            ),
                        },
                    }
                )
    return details


@pytest.mark.parametrize(
    ("initial", "persistent", "classification"),
    [
        (4, 4, "global_clip_starvation_and_persistent_conflict"),
        (4, 3, "initial_cross_block_clip_starvation_only"),
        (3, 4, "persistent_objective_conflict_without_initial_starvation"),
        (3, 3, "no_registered_gradient_failure_mechanism"),
    ],
)
def test_aggregate_classification_is_joint_across_presets(
    initial: int, persistent: int, classification: str
) -> None:
    aggregate = aggregate_details(
        _details(initial, persistent), GradientAttributionConfig()
    )
    assert aggregate["classification"] == classification
    assert aggregate["valid"]


def test_analytic_cells_do_not_enter_learned_population_gate() -> None:
    aggregate = aggregate_details(_details(4, 4), GradientAttributionConfig())
    for preset in PRESETS:
        control = aggregate["strata"][f"{preset}/analytic_calibrated"]
        assert control["initial_cross_block_starvation_count"] is None
        assert control["persistent_learned_state_conflict_gate"] is None
