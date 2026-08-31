import numpy as np

from experiments.structure_net.tinyllm_ttno_cut_localization_parity import (
    CutLocalizationConfig,
    aggregate,
    duplicate_operator,
    gray_code_order,
    mode_order,
    ordered_qttno_rank_profile,
    zero_pad_operator,
)


def test_frozen_mode_orders_and_gray_permutation() -> None:
    assert mode_order(8, "msb_balanced") == tuple(range(8))
    assert mode_order(8, "lsb_balanced") == tuple(range(7, -1, -1))
    assert mode_order(8, "odd_even_modes") == (1, 3, 5, 7, 0, 2, 4, 6)
    gray = gray_code_order(16)
    np.testing.assert_array_equal(np.sort(gray), np.arange(16))
    assert all((int(gray[index]) ^ int(gray[index + 1])).bit_count() == 1 for index in range(15))


def test_zero_padding_and_duplication_are_exact() -> None:
    operator = np.arange(16, dtype=np.float64).reshape(4, 4)
    zero = zero_pad_operator(operator)
    duplicate = duplicate_operator(operator)
    np.testing.assert_array_equal(zero[:4, :4], operator)
    assert np.count_nonzero(zero[4:]) == 0
    np.testing.assert_array_equal(duplicate[:4, :4], operator)
    np.testing.assert_array_equal(duplicate[4:, 4:], operator)
    assert np.count_nonzero(duplicate[:4, 4:]) == 0
    assert np.count_nonzero(duplicate[4:, :4]) == 0


def test_ordered_profile_matches_a4_shape_for_msb() -> None:
    operator = np.eye(16, dtype=np.float64)
    profile = ordered_qttno_rank_profile(operator, (1e-2, 1e-3), "msb_balanced")
    assert profile["max_ranks"] == {"0.01": 1, "0.001": 1}
    assert profile["mode_order"] == [0, 1, 2, 3]


def _cell(artifact: float, topology: float, valid: bool = True) -> dict:
    return {
        "validity": {"pass": valid},
        "primary": {
            "maximum_artifact_fraction": artifact,
            "topology_reduction": topology,
            "new_root_cliff_fraction": artifact,
            "winning_alternative_topology": "lsb_balanced",
            "r128": 2,
            "r256": 4,
            "rzero": 3,
            "rduplicate": 3,
            "ralt": 3,
            "cliff": 2,
            "new_root_split_rank": 4,
        },
    }


def test_aggregate_applies_preregistered_classification_order() -> None:
    config = CutLocalizationConfig(
        evaluation_seeds=(1,), layers=(0,), heads=(0,), shakedown=True
    )
    parity = aggregate([{"cells": [_cell(0.8, 0.5)]}], config)
    assert parity["classification"] == "new_cut_artifact_dominant"
    topology = aggregate([{"cells": [_cell(0.5, 0.5)]}], config)
    assert topology["classification"] == "bit_topology_sensitive"
    intrinsic = aggregate([{"cells": [_cell(0.1, 0.0)]}], config)
    assert intrinsic["classification"] == "intrinsic_operator_rank_growth"
    invalid = aggregate([{"cells": [_cell(0.8, 0.5, False)]}], config)
    assert invalid["classification"] == "invalid_parent_or_tensor_contract"
