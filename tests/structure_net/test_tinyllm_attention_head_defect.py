import torch

from experiments.structure_net.tinyllm_attention_head_defect import (
    AttentionHeadDefectConfig,
    aggregate,
    attention_head_defects,
    intervention_subsets,
    projected_head_outputs,
    seed_gates,
)
from structure_net.components.models.tinyllm_model import TinyLLMBlock, TinyLLMConfig


def test_projected_heads_reconstruct_attention_and_reynolds_defect() -> None:
    torch.manual_seed(4)
    config = TinyLLMConfig(
        block_size=8, vocab_size=31, n_layer=1, n_head=3, n_embd=12
    )
    block = TinyLLMBlock(config, 0).eval()
    values = torch.randn(7, 4, 12)
    normalized = block.ln_1(values)
    heads = projected_head_outputs(block.attn, normalized)
    reconstructed = heads.sum(1) + block.attn.c_proj.bias
    torch.testing.assert_close(reconstructed, block.attn(normalized), rtol=1e-5, atol=1e-6)

    sheets = torch.randn(5, 2, 4, 12)
    detail = attention_head_defects(block, sheets)
    assert detail["ordinary_attention_relative_error"] < 1e-6
    assert detail["defect_relative_error"] < 1e-6
    torch.testing.assert_close(
        detail["head_defects"].sum(1), detail["full_defect"], rtol=1e-5, atol=1e-6
    )


def test_intervention_family_is_the_preregistered_29_subsets() -> None:
    subsets = intervention_subsets(6)
    assert len(subsets) == 29
    assert len(set(subsets)) == 29
    assert () in subsets and tuple(range(6)) in subsets
    assert sum(len(value) == 1 for value in subsets) == 6
    assert sum(len(value) == 2 for value in subsets) == 15
    assert sum(len(value) == 5 for value in subsets) == 6


def test_seed_gates_require_the_same_sparse_subset_and_necessary_head() -> None:
    regimes = {
        "composition": {
            "contract_pass": True,
            "endpoint_pass": True,
            "sufficient_sparse_subsets": ["h0_h2", "h1_h2"],
            "necessary_heads": [2, 4],
        },
        "extrapolation": {
            "contract_pass": True,
            "endpoint_pass": True,
            "sufficient_sparse_subsets": ["h0_h2"],
            "necessary_heads": [2],
        },
    }
    gates = seed_gates(regimes)
    assert gates["exact_contract_and_endpoint"]
    assert gates["sparse_sufficiency"]
    assert gates["shift_stable_sparse_circuit"]
    assert gates["individual_necessity"]
    assert gates["common_sufficient_subsets"] == ["h0_h2"]
    assert gates["common_necessary_heads"] == [2]

    regimes["extrapolation"]["sufficient_sparse_subsets"] = ["h3_h4"]
    assert not seed_gates(regimes)["shift_stable_sparse_circuit"]


def test_aggregate_applies_four_of_five_to_joint_seed_gates() -> None:
    config = AttentionHeadDefectConfig()
    runs = []
    for index, seed in enumerate(config.seeds):
        passed = index < 4
        runs.append(
            {
                "seed": seed,
                "gates": {
                    "exact_contract_and_endpoint": True,
                    "sparse_sufficiency": passed,
                    "shift_stable_sparse_circuit": passed,
                    "individual_necessity": passed,
                },
                "regimes": {
                    regime: {
                        "target_cut": "block_0_post_attention",
                        "sufficient_sparse_subsets": ["h0_h1"] if passed else [],
                        "necessary_heads": [0] if passed else [],
                    }
                    for regime in ("composition", "extrapolation")
                },
            }
        )
    result = aggregate(runs, config)
    assert result["confirmed"]
    assert result["gate_counts"]["sparse_sufficiency"] == 4
