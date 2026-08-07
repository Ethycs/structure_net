from __future__ import annotations

import pytest
import torch

from experiments.structure_net.tinyllm_c2_attention_head_decomposition import (
    HeadDecompositionConfig,
    _heldout_summary,
    _source_selection,
    exact_head_defects,
    projected_attention_heads,
    shapley_values,
    subset_heads,
    subset_states,
)
from structure_net.components.models.tinyllm_model import (
    TinyLLMBlock,
    TinyLLMConfig,
)


def test_config_freezes_primary_cohort_and_orders_thresholds() -> None:
    with pytest.raises(ValueError, match="five seeds"):
        HeadDecompositionConfig(seeds=(7,), primary_seeds=(7,))
    with pytest.raises(ValueError, match="fixed"):
        HeadDecompositionConfig(primary_seeds=(7, 17, 29))
    with pytest.raises(ValueError, match="order"):
        HeadDecompositionConfig(
            complement_fisher_ceiling=0.95,
            sufficient_fisher_threshold=0.90,
        )


def test_projected_heads_exactly_decompose_attention_reynolds_defect() -> None:
    torch.manual_seed(7)
    config = TinyLLMConfig(
        block_size=8,
        vocab_size=32,
        n_layer=1,
        n_head=3,
        n_embd=12,
    )
    block = TinyLLMBlock(config, index=0).eval()
    sheets = torch.randn(5, 2, 4, 12)
    propagated, defects, exact, error = exact_head_defects(block, sheets)
    assert projected_attention_heads(block, sheets[:, 0]).shape == (5, 3, 4, 12)
    assert defects.shape == (5, 3, 4, 12)
    assert error < 1e-6
    assert torch.allclose(propagated + defects.sum(1), exact, atol=1e-6)
    states = subset_states(propagated, defects)
    assert torch.equal(states[0], propagated)
    assert torch.allclose(states[7], exact, atol=1e-6)


def test_subset_heads_and_shapley_are_exact_for_additive_game() -> None:
    weights = (0.15, 0.35, 0.50)
    values = {
        mask: sum(weights[head] for head in subset_heads(mask, 3))
        for mask in range(8)
    }
    assert subset_heads(5, 3) == (0, 2)
    assert shapley_values(values, 3) == pytest.approx(weights)


def _record(preservation: float, causal_pass: bool = True) -> dict:
    return {
        "causal_pass": causal_pass,
        "effect_preservation": {
            "degenerate": False,
            "preserved_fraction": preservation,
        },
    }


def _cell(values: dict[int, float], *, endpoint: bool = True) -> dict:
    return {
        "head_count": 3,
        "endpoint_replication": endpoint,
        "subsets": {str(mask): _record(value) for mask, value in values.items()},
    }


def test_source_selection_is_minimal_then_maximin() -> None:
    config = HeadDecompositionConfig(
        seeds=(7,), primary_seeds=(7,), allow_underpowered=True
    )
    composition = {mask: 0.1 for mask in range(8)}
    extrapolation = {mask: 0.1 for mask in range(8)}
    composition.update({1: 0.91, 2: 0.95, 3: 0.99, 7: 1.0})
    extrapolation.update({1: 0.92, 2: 0.93, 3: 0.99, 7: 1.0})
    cohorts = {
        "source_selection": {
            "composition": _cell(composition),
            "extrapolation": _cell(extrapolation),
        }
    }
    selected = _source_selection(cohorts, config)
    assert selected["selected_mask"] == 2
    assert selected["selected_heads"] == [1]
    assert selected["cardinality"] == 1


def test_heldout_summary_requires_fixed_subset_and_weak_complement() -> None:
    config = HeadDecompositionConfig(
        seeds=(7,), primary_seeds=(7,), allow_underpowered=True,
        specificity_margin=0.2,
    )
    values = {mask: 0.1 for mask in range(8)}
    values.update({1: 0.95, 2: 0.2, 4: 0.3, 6: 0.4, 7: 1.0})
    cell = _cell(values)
    cohorts = {
        "source_selection": {"composition": cell, "extrapolation": cell},
        "heldout_a": {"composition": cell, "extrapolation": cell},
        "heldout_b": {"composition": cell, "extrapolation": cell},
    }
    summary = _heldout_summary(
        cohorts,
        {"selected_mask": 1, "selected_heads": [0], "cardinality": 1},
        config,
    )
    assert summary["endpoint_replication"] is True
    assert summary["fixed_subset_sufficiency"] is True
    assert summary["complement_dominance"] is True
    assert summary["subset_specificity"] is True

