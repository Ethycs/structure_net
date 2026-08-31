import numpy as np
import torch

from experiments.structure_net.tinyllm_causal_h2_attention import (
    CausalH2Config,
    PARTITION_COUNTS,
    PARTITION_FINGERPRINTS,
    aggregate,
    assemble_dense,
    build_h2_representations,
    h2_matvec,
    partition_integrity,
    stabilized_kernel,
)


def _cpu_config(lengths=(32, 64), primary=(64,)) -> CausalH2Config:
    return CausalH2Config(
        evaluation_seeds=(1,),
        lengths=lengths,
        primary_lengths=primary,
        layers=(0,),
        heads=(0,),
        device="cpu",
        shakedown=True,
    )


def test_partition_counts_and_hashes_match_preregistration() -> None:
    for length in (32, 64, 128, 256, 512):
        integrity = partition_integrity(length)
        assert integrity["counts"] == PARTITION_COUNTS[length]
        assert integrity["sha256"] == PARTITION_FINGERPRINTS[length]


def test_length32_is_exact_near_field_and_contraction_matches() -> None:
    config = _cpu_config(lengths=(32,), primary=(32,))
    generator = np.random.default_rng(4)
    query = generator.standard_normal((32, 4))
    key = generator.standard_normal((32, 4))
    kernel, _, attention = stabilized_kernel(query, key)
    primary, _, records = build_h2_representations(kernel, attention, config)
    assembled = assemble_dense(primary)
    np.testing.assert_allclose(assembled.numpy(), kernel, rtol=0.0, atol=0.0)
    values = torch.tensor(generator.standard_normal((32, 5)), dtype=torch.float64)
    np.testing.assert_allclose(
        h2_matvec(primary, values).numpy(),
        (assembled @ values).numpy(),
        rtol=1e-13,
        atol=1e-13,
    )
    assert records["partition"]["counts"]["ADMISSIBLE"] == 0


def test_rank_one_far_field_is_nested_and_exact() -> None:
    config = _cpu_config()
    left = np.linspace(1.0, 2.0, 64)
    right = np.linspace(2.0, 3.0, 64)
    kernel = np.tril(np.outer(left, right))
    attention = kernel.copy()
    attention /= attention.sum(axis=1, keepdims=True)
    primary, _, records = build_h2_representations(kernel, attention, config)
    assembled = assemble_dense(primary)
    np.testing.assert_allclose(assembled.numpy(), kernel, rtol=1e-12, atol=1e-12)
    assert max(records["orthogonality"].values()) <= 1e-12
    assert max(records["nestedness"].values()) <= 1e-12


def _fake_record(passed: bool, oracle: bool, storage: float = 2.0) -> dict:
    return {
        "validity": {"pass": True},
        "cell_pass": passed,
        "oracle_pass": oracle,
        "primary_metrics": {"kernel_row_relative_maximum": 0.0 if passed else 1.0},
        "construction": {"rank_cap_hits": {"query": 0, "key": 0}},
        "compression": {
            "storage": {"ratio": storage},
            "multiply_adds": {"ratio": storage},
        },
    }


def test_aggregate_distinguishes_normalization_and_compression() -> None:
    config = _cpu_config()
    normalization = aggregate(
        [
            {
                "cells": [
                    {
                        "layer": 0,
                        "head": 0,
                        "lengths": {
                            "32": _fake_record(False, True),
                            "64": _fake_record(False, True),
                        },
                    }
                ]
            }
        ],
        config,
    )
    assert normalization["classification"] == "h2_normalization_path_failed"
    no_compression = aggregate(
        [
            {
                "cells": [
                    {
                        "layer": 0,
                        "head": 0,
                        "lengths": {
                            "32": _fake_record(True, True),
                            "64": _fake_record(True, True),
                        },
                    }
                ]
            }
        ],
        config,
    )
    assert (
        no_compression["classification"]
        == "h2_representation_pass_no_finite_size_compression"
    )
