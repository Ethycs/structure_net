import numpy as np
import pytest

from experiments.structure_net.tinyllm_dynamic_ttno_rank import (
    DynamicTTNORankConfig,
    aggregate,
    analyze_qk,
    balanced_tree_nodes,
    causal_attention,
    gaussian_causal_attention,
    hss_boundary_rank_profile,
    numerical_rank_from_singular_values,
    paired_operator_tensor,
    paired_tensor_to_operator,
    qk_pca_order,
    qttno_rank_profile,
    topk_sparse,
)
from experiments.structure_net import tinyllm_dynamic_ttno_rank as ttno


def test_config_rejects_non_power_of_two_lengths() -> None:
    with pytest.raises(ValueError, match="power of two"):
        DynamicTTNORankConfig(lengths=(32, 48))


def test_numerical_rank_uses_relative_frobenius_tail() -> None:
    singular_values = np.array([4.0, 3.0, 0.1])
    assert numerical_rank_from_singular_values(singular_values, 0.1) == 2
    assert numerical_rank_from_singular_values(singular_values, 0.01) == 3
    assert numerical_rank_from_singular_values(np.zeros(3), 0.01) == 0


def test_gram_spectrum_matches_direct_svd_at_declared_tolerances() -> None:
    generator = np.random.default_rng(13)
    matrix = generator.standard_normal((48, 80))
    direct = ttno._singular_values(matrix, stable_svd=True)
    gram = ttno._singular_values(matrix, stable_svd=False)
    for epsilon in (1e-2, 1e-3):
        assert numerical_rank_from_singular_values(
            gram, epsilon
        ) == numerical_rank_from_singular_values(direct, epsilon)


def test_paired_operator_tensor_round_trip() -> None:
    operator = np.arange(64, dtype=np.float64).reshape(8, 8)
    tensor = paired_operator_tensor(operator)
    assert tensor.shape == (4, 4, 4)
    np.testing.assert_array_equal(paired_tensor_to_operator(tensor), operator)


def test_identity_has_exact_quantized_ttno_rank_one() -> None:
    profile = qttno_rank_profile(np.eye(16), (1e-12,))
    assert profile["max_ranks"]["1e-12"] == 1


def test_rank_one_dense_operator_has_hss_boundary_rank_one() -> None:
    left = np.arange(1.0, 17.0)
    right = np.arange(2.0, 18.0)
    profile = hss_boundary_rank_profile(np.outer(left, right), (1e-12,))
    assert profile["max_ranks"]["1e-12"] == 1


def test_gaussian_factorization_matches_causal_attention() -> None:
    generator = np.random.default_rng(7)
    query = generator.standard_normal((16, 6))
    key = generator.standard_normal((16, 6))
    ordinary = causal_attention(query, key)
    gaussian = gaussian_causal_attention(query, key)
    np.testing.assert_allclose(gaussian, ordinary, rtol=1e-13, atol=1e-13)


def test_pca_order_is_deterministic_permutation() -> None:
    generator = np.random.default_rng(11)
    query = generator.standard_normal((16, 4))
    key = generator.standard_normal((16, 4))
    first = qk_pca_order(query, key)
    second = qk_pca_order(query, key)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(16))


def test_topk_sparse_retains_declared_entries_and_mass() -> None:
    query = np.zeros((8, 4))
    attention = causal_attention(query, query)
    sparse, mass = topk_sparse(attention, 2)
    assert np.all(np.count_nonzero(sparse, axis=1) <= 2)
    assert 0.0 < mass <= 1.0
    assert np.all(sparse <= attention)


def test_balanced_tree_excludes_root_and_covers_leaves() -> None:
    nodes = balanced_tree_nodes(tuple(range(8)))
    assert tuple(range(8)) not in nodes
    assert {(index,) for index in range(8)}.issubset(set(nodes))


def test_tiny_aggregate_emits_controlled_classification() -> None:
    config = DynamicTTNORankConfig(
        validation_tokens=32,
        evaluation_seeds=(1,),
        lengths=(4, 8),
        layers=(0,),
        heads=(0,),
        epsilons=(1e-2, 1e-3),
        shakedown=True,
    )
    query = np.zeros((8, 4))
    natural = {
        str(length): analyze_qk(
            query[:length],
            query[:length],
            config,
            include_sparse=length == 8,
        )
        for length in config.lengths
    }
    controls = {
        condition: {
            str(length): analyze_qk(
                query[:length],
                query[:length],
                config,
                include_sparse=False,
            )
            for length in config.lengths
        }
        for condition in ("causal_uniform", "smooth_fourier", "iid_qk")
    }
    result = aggregate(
        [{"natural_cells": [{"layer": 0, "head": 0, "lengths": natural}], "controls": controls}],
        config,
    )
    assert result["classification"] in {
        "polylog_compatible_pilot",
        "mixed_rank_pilot",
        "bond_growth_observed_pilot",
    }
    assert result["gaussian_identity"]["pass"] is True
    assert result["counts"]["natural_cells"] == 1
