from __future__ import annotations

import pytest
import torch

from experiments.structure_net.tinyllm_joint_interface_block_clipping import (
    HYPOTHESIS_ID,
    SCHEMA_VERSION,
    BlockClippingConfig,
    _stage_a_population,
    aggregate_details,
    clip_parameter_blocks,
)


def test_primary_configuration_is_locked() -> None:
    config = BlockClippingConfig()
    assert SCHEMA_VERSION.endswith(".v1")
    assert HYPOTHESIS_ID.endswith("-v1")
    assert config.conditions == ("learned_calibrated_equivariant",)
    assert config.training_steps == 600
    assert config.train_samples == 4_096
    assert config.block_gradient_clip == 1.0
    with pytest.raises(ValueError, match="configuration changed"):
        BlockClippingConfig(block_gradient_clip=0.5)
    with pytest.raises(ValueError, match="4096"):
        BlockClippingConfig(train_samples=128, allow_underpowered=True)


def test_block_clipping_removes_cross_block_coupling() -> None:
    first = torch.nn.Parameter(torch.zeros(1))
    second = torch.nn.Parameter(torch.zeros(1))
    third = torch.nn.Parameter(torch.zeros(1))
    first.grad = torch.tensor([10.0])
    second.grad = torch.tensor([0.5])
    third.grad = torch.tensor([0.0])
    record = clip_parameter_blocks(
        {
            "encoder": [("encoder", first)],
            "scalar_embedding": [("embedding", second)],
            "final_scalar": [("head", third)],
        },
        1.0,
    )
    assert record["preclip_norms"] == {
        "encoder": 10.0,
        "scalar_embedding": 0.5,
        "final_scalar": 0.0,
    }
    assert record["clip_coefficients"]["encoder"] == pytest.approx(0.1)
    assert record["clip_coefficients"]["scalar_embedding"] == 1.0
    assert record["global_equivalent_clip_coefficient"] < 0.1
    assert float(first.grad) == pytest.approx(1.0)
    assert float(second.grad) == pytest.approx(0.5)


def _details(d6_true: int, d10_true: int, shuffled: int = 0) -> list[dict]:
    records = []
    for preset, count in (("d6", d6_true), ("d10", d10_true)):
        for index, seed in enumerate((7, 17, 29, 41, 53)):
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
    ("d6", "d10", "classification", "extension"),
    [
        (4, 4, "parameter_block_clipping_repairs_physical_interface", False),
        (4, 3, "architecture_conditional_block_clipping_repair", True),
        (3, 3, "parameter_block_clipping_insufficient", True),
    ],
)
def test_population_classification(
    d6: int, d10: int, classification: str, extension: bool
) -> None:
    aggregate = aggregate_details(_details(d6, d10), BlockClippingConfig())
    assert aggregate["valid"]
    assert aggregate["classification"] == classification
    assert aggregate["full_interface_extension_licensed"] is extension


def test_shuffled_control_invalidates_causal_classification() -> None:
    aggregate = aggregate_details(_details(5, 5, shuffled=2), BlockClippingConfig())
    assert aggregate["classification"] == "specificity_control_failed"
    assert aggregate["primary_hypothesis_pass"] is False


def test_stage_a_comparator_is_exact_and_learned_cells_failed() -> None:
    _campaign, details = _stage_a_population(BlockClippingConfig())
    assert len(details) == 10
    assert all(
        detail["gates"]["physical_true_joint_seed_pass"] is False
        and detail["gates"]["pair_shuffled_joint_seed_pass"] is False
        for detail in details.values()
    )

