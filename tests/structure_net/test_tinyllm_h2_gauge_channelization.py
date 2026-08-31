from __future__ import annotations

import numpy as np
import torch

from experiments.structure_net import tinyllm_causal_h2_attention as a7
from experiments.structure_net.tinyllm_h2_gauge_channelization import (
    GaugeChannelizationConfig,
    assemble_effective_dense,
    block_mask,
    gauge_transform,
    offblock_objective,
    optimize_all_gauges,
    optimize_gauges,
    structured_accounting,
)


def _synthetic_representation() -> a7.H2Representation:
    dtype = torch.float64
    angle_query = 0.47
    angle_key = -0.31
    query_gauge = torch.tensor(
        [
            [np.cos(angle_query), -np.sin(angle_query)],
            [np.sin(angle_query), np.cos(angle_query)],
        ],
        dtype=dtype,
    )
    key_gauge = torch.tensor(
        [
            [np.cos(angle_key), -np.sin(angle_key)],
            [np.sin(angle_key), np.cos(angle_key)],
        ],
        dtype=dtype,
    )
    coupling = query_gauge @ torch.diag(torch.tensor([3.0, 1.0], dtype=dtype)) @ key_gauge.T
    return a7.H2Representation(
        operator_name="synthetic",
        length=2,
        clusters={},
        children={},
        root="",
        blocks=[],
        query_bases={"q": torch.eye(2, dtype=dtype)},
        key_bases={"k": torch.eye(2, dtype=dtype)},
        query_transfers={},
        key_transfers={},
        couplings={("q", "k"): coupling},
        dense_operator=torch.zeros((2, 2), dtype=dtype),
        construction_records={},
    )


def _small_a7_representation() -> a7.H2Representation:
    generator = np.random.Generator(np.random.PCG64(17))
    query = generator.normal(size=(64, 8))
    key = generator.normal(size=(64, 8))
    kernel, _, attention = a7.stabilized_kernel(query, key)
    config = a7.CausalH2Config(
        evaluation_seeds=(101,),
        lengths=(64,),
        primary_lengths=(64,),
        layers=(0,),
        heads=(0,),
        device="cpu",
        shakedown=True,
    )
    representation, _, _ = a7.build_h2_representations(kernel, attention, config)
    return representation


def test_block_mask_has_fixed_rectangular_blocks() -> None:
    mask = block_mask(4, 5, 2, device=torch.device("cpu"))
    assert int(mask.sum().item()) == 8
    assert mask.tolist() == [
        [True, True, False, False, False],
        [True, True, False, False, False],
        [False, False, True, True, False],
        [False, False, True, True, False],
    ]


def test_spectral_restart_recovers_jointly_diagonalizable_coupling() -> None:
    representation = _synthetic_representation()
    config = GaugeChannelizationConfig(
        evaluation_seeds=(101,),
        lengths=(64,),
        primary_length=64,
        layers=(0,),
        heads=(0,),
        optimizer_updates=2,
        device="cpu",
        shakedown=True,
    )
    identity_query = {"q": torch.eye(2, dtype=torch.float64)}
    identity_key = {"k": torch.eye(2, dtype=torch.float64)}
    before = float(
        offblock_objective(
            representation, identity_query, identity_key, 1
        ).item()
    )
    query_gauges, key_gauges, record = optimize_gauges(
        representation, 1, config
    )
    after = float(
        offblock_objective(
            representation, query_gauges, key_gauges, 1
        ).item()
    )
    assert after < before * 1e-10
    assert record["orthogonality_maximum"] < 1e-12


def test_batched_optimizer_preserves_candidate_semantics() -> None:
    representation = _synthetic_representation()
    config = GaugeChannelizationConfig(
        evaluation_seeds=(101,),
        lengths=(64,),
        primary_length=64,
        layers=(0,),
        heads=(0,),
        optimizer_updates=2,
        device="cpu",
        shakedown=True,
    )
    optimized = optimize_all_gauges(representation, config)
    assert set(optimized) == {1, 2, 4}
    query_gauges, key_gauges, record = optimized[1]
    objective = float(
        offblock_objective(
            representation, query_gauges, key_gauges, 1
        ).item()
    )
    assert objective < 1e-10
    assert record["execution"] == "six_candidate_batched_gpu"


def test_unpruned_gauge_is_exact_and_accounting_is_unchanged() -> None:
    representation = _small_a7_representation()
    query_gauges = {}
    key_gauges = {}
    for family, output in (
        (representation.query_bases, query_gauges),
        (representation.key_bases, key_gauges),
    ):
        for node, basis in family.items():
            rank = basis.shape[1]
            if rank == 0:
                output[node] = torch.empty((0, 0), dtype=torch.float64)
            else:
                matrix = torch.arange(
                    1, rank * rank + 1, dtype=torch.float64
                ).reshape(rank, rank)
                output[node] = torch.linalg.qr(matrix).Q
    transformed = gauge_transform(
        representation, query_gauges, key_gauges, None
    )
    original_dense = assemble_effective_dense(representation)
    transformed_dense = assemble_effective_dense(transformed)
    assert torch.allclose(original_dense, transformed_dense, atol=1e-10, rtol=1e-10)
    original_accounting = structured_accounting(representation, 9, None)
    transformed_accounting = structured_accounting(transformed, 9, None)
    assert transformed_accounting == original_accounting


def test_fixed_blocks_reduce_structural_counts() -> None:
    representation = _small_a7_representation()
    full = structured_accounting(representation, 9, None)
    diagonal = structured_accounting(representation, 9, 1)
    block4 = structured_accounting(representation, 9, 4)
    assert diagonal["storage"]["total"] < block4["storage"]["total"]
    assert block4["storage"]["total"] < full["storage"]["total"]
    assert diagonal["multiply_adds"]["total"] < full["multiply_adds"]["total"]


def test_pruned_effective_assembly_matches_h2_contraction() -> None:
    representation = _small_a7_representation()
    query_gauges = {
        node: torch.eye(basis.shape[1], dtype=torch.float64)
        for node, basis in representation.query_bases.items()
    }
    key_gauges = {
        node: torch.eye(basis.shape[1], dtype=torch.float64)
        for node, basis in representation.key_bases.items()
    }
    pruned = gauge_transform(representation, query_gauges, key_gauges, 2)
    explicit = assemble_effective_dense(pruned)
    contracted = a7.h2_matvec(pruned, torch.eye(64, dtype=torch.float64))
    assert torch.allclose(explicit, contracted, atol=1e-10, rtol=1e-10)
