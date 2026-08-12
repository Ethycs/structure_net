from __future__ import annotations

from pathlib import Path

import pytest
import torch

import experiments.structure_net.tinyllm_joint_full_interface as full


def test_primary_configuration_is_locked() -> None:
    config = full.FullInterfaceConfig()
    assert full.SCHEMA_VERSION.endswith(".v1")
    assert full.HYPOTHESIS_ID.endswith("-v1")
    assert config.conditions == ("learned_calibrated_equivariant",)
    assert config.training_steps == 600
    assert config.train_samples == 4_096
    assert config.gradient_clip == 1.0
    assert config.learning_rate == 3e-4
    with pytest.raises(ValueError, match="configuration changed"):
        full.FullInterfaceConfig(gradient_clip=0.5)
    with pytest.raises(ValueError, match="4096"):
        full.FullInterfaceConfig(train_samples=128, allow_underpowered=True)


def test_state_digest_matches_sealed_model_digest_implementation() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.LayerNorm(4),
    )
    assert full._model_state_digest(model) == full.joint.calibrated._state_digest(model)


def test_parameter_signature_tracks_tied_parameters() -> None:
    class Tied(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 3)
            self.head = torch.nn.Linear(3, 4, bias=False)
            self.head.weight = self.embedding.weight

    signature = full.model_parameter_signature(Tied())
    by_name = {record["name"]: record for record in signature["parameters"]}
    assert by_name["embedding.weight"]["alias_group"] == by_name["head.weight"][
        "alias_group"
    ]


def _details(d6_true: int, d10_true: int, shuffled: int = 0) -> list[dict]:
    records = []
    for preset, count in (("d6", d6_true), ("d10", d10_true)):
        for index, seed in enumerate(full.SEEDS):
            records.append(
                {
                    "preset": preset,
                    "seed": seed,
                    "gates": {
                        "validity": True,
                        "physical_true_joint_seed_pass": index < count,
                        "pair_shuffled_joint_seed_pass": index < shuffled,
                    },
                }
            )
    return records


@pytest.mark.parametrize(
    ("d6", "d10", "classification", "primary"),
    [
        (4, 4, "full_interface_physical_typing_architecture_stable", True),
        (4, 3, "architecture_conditional_full_interface_repair", False),
        (3, 3, "flexible_full_interface_physical_typing_insufficient", False),
    ],
)
def test_population_classification(
    d6: int, d10: int, classification: str, primary: bool
) -> None:
    aggregate = full.aggregate_details(_details(d6, d10), full.FullInterfaceConfig())
    assert aggregate["valid"]
    assert aggregate["classification"] == classification
    assert aggregate["primary_hypothesis_pass"] is primary


def test_shuffled_control_invalidates_causal_classification() -> None:
    aggregate = full.aggregate_details(
        _details(5, 5, shuffled=2), full.FullInterfaceConfig()
    )
    assert aggregate["classification"] == "specificity_control_failed"
    assert aggregate["primary_hypothesis_pass"] is False


def test_parent_campaigns_and_full_parameter_route_replay() -> None:
    config = full.FullInterfaceConfig(
        presets=("d6",),
        seeds=(7,),
        required_seed_passes=1,
        allow_underpowered=True,
    )
    stage_a, stage_a_details = full._stage_a_population(config)
    block = full._block_comparator(config)
    assert stage_a["aggregates"]["classification"] == (
        "frozen_backbone_joint_interface_insufficient"
    )
    assert block["aggregates"]["classification"] == (
        "parameter_block_clipping_insufficient"
    )
    _, task, source_details = full.joint._source_population(full._joint_config(config))
    detail = source_details[("d6", full.CONDITION, 7)]
    interface = full._load_full_interface(
        detail,
        task,
        config,
        "d6",
        full.CONDITION,
        torch.device("cpu"),
    )
    selected, route = full.trainable_parameters(interface)
    assert selected
    assert all(parameter.requires_grad for _name, parameter in selected)
    assert all(value > 0 for value in route["group_parameter_counts"].values())
    assert full._model_state_digest(interface.system.model) == detail["training"][
        "final_model_state_sha256"
    ]
    initial = full.joint._interface_state_digest(interface)
    expected = stage_a_details[("d6", full.CONDITION, 7)]["arms"]["physical_true"][
        "training"
    ]["initial_interface_state_sha256"]
    assert initial == expected
    signature = full.model_parameter_signature(interface.system.model)
    aliases = {record["name"]: record["alias_group"] for record in signature["parameters"]}
    assert aliases["transformer.wte.weight"] == aliases["lm_head.weight"]
    assert signature["feedback_topology"] == []
    assert Path(detail["artifacts"]["checkpoint"]).is_file()
