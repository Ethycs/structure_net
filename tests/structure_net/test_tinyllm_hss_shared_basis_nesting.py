import numpy as np

from experiments.structure_net.tinyllm_dynamic_ttno_rank import (
    causal_attention,
    hss_boundary_rank_profile,
)
from experiments.structure_net.tinyllm_hss_shared_basis_nesting import (
    SharedBasisNestingConfig,
    aggregate,
    complement_peers,
    diagnose_operator,
    dyadic_cluster_tree,
)


def _cpu_config() -> SharedBasisNestingConfig:
    return SharedBasisNestingConfig(
        evaluation_seeds=(1,),
        lengths=(8, 16),
        layers=(0,),
        heads=(0,),
        device="cpu",
        shakedown=True,
    )


def test_peer_partition_is_exact_complement() -> None:
    cluster_list, children, root = dyadic_cluster_tree(16)
    clusters = {cluster.id: cluster for cluster in cluster_list}
    for cluster in cluster_list:
        if cluster.id == root:
            continue
        peers = complement_peers(cluster.id, clusters, children)
        peer_tokens = sorted(
            token
            for peer in peers
            for token in range(clusters[peer].start, clusters[peer].stop)
        )
        complement = [
            token
            for token in range(16)
            if not cluster.start <= token < cluster.stop
        ]
        assert peer_tokens == complement


def test_shared_maximum_reproduces_hss_boundary_rank() -> None:
    config = _cpu_config()
    generator = np.random.default_rng(11)
    query = generator.standard_normal((16, 4))
    key = generator.standard_normal((16, 4))
    operator = causal_attention(query, key)
    diagnostic = diagnose_operator(operator, config)
    expected = hss_boundary_rank_profile(operator, config.epsilons)["max_ranks"]
    assert diagnostic["summary"]["max_shared_ranks"] == expected
    assert diagnostic["summary"]["concatenation_exact"] is True


def test_exact_rank_one_has_shared_nested_rank_one() -> None:
    config = _cpu_config()
    operator = np.outer(np.arange(1.0, 17.0), np.arange(2.0, 18.0))
    summary = diagnose_operator(operator, config, retain_records=False)["summary"]
    assert summary["max_shared_ranks"]["0.01"] == 1
    assert summary["stable_nesting_defect"]["maximum"] <= 1e-10
    assert summary["stable_augmented_rank_ratio"]["maximum"] == 1.0


def _fake_cell(
    sharing_median: float,
    sharing_p90: float,
    defect: float,
    ratio: float,
    stable_fraction: float = 1.0,
    valid: bool = True,
) -> dict:
    stable = int(100 * stable_fraction)
    summary = {
        "sharing_inflation": {"median": sharing_median, "p90": sharing_p90},
        "stable_nesting_defect": {"median": defect, "p90": defect},
        "stable_augmented_rank_ratio": {"median": ratio, "p90": ratio},
        "stable_cuts": stable,
        "nesting_cuts": 100,
        "max_shared_ranks": {"0.01": 1, "0.001": 1},
    }
    return {
        "validity": {"pass": valid},
        "lengths": {str(length): {"summary": summary} for length in (8, 16, 256)},
    }


def _rank_one_control() -> dict:
    return {
        "max_shared_ranks": {"0.01": 1},
        "stable_nesting_defect": {"maximum": 0.0},
        "stable_augmented_rank_ratio": {"maximum": 1.0},
    }


def test_aggregate_applies_frozen_classifications() -> None:
    config = _cpu_config()
    supported = aggregate(
        [
            {
                "cells": [_fake_cell(1.2, 1.5, 0.01, 1.1)],
                "controls": {"exact_rank_one": _rank_one_control()},
            }
        ],
        config,
    )
    assert supported["classification"] == "shared_and_nested_hierarchy_supported"
    nesting = aggregate(
        [
            {
                "cells": [_fake_cell(1.2, 1.5, 0.5, 2.5)],
                "controls": {"exact_rank_one": _rank_one_control()},
            }
        ],
        config,
    )
    assert nesting["classification"] == "nesting_bottleneck"
    indeterminate = aggregate(
        [
            {
                "cells": [_fake_cell(1.2, 1.5, 0.01, 1.1, 0.1)],
                "controls": {"exact_rank_one": _rank_one_control()},
            }
        ],
        config,
    )
    assert (
        indeterminate["classification"]
        == "shared_basis_result_nesting_indeterminate"
    )
