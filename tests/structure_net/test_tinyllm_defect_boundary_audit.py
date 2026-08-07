from __future__ import annotations

import pytest
import torch

from experiments.structure_net.tinyllm_defect_boundary_audit import (
    DefectBoundaryAuditConfig,
    _path_key,
    aggregate,
    classify_boundary_mechanism,
    predecessor_replication_contract,
    prediction_boundary_metrics,
)


def test_config_freezes_cells_and_orders_thresholds() -> None:
    with pytest.raises(ValueError, match="fixed"):
        DefectBoundaryAuditConfig(seeds=(29,))
    with pytest.raises(ValueError, match="zero and one"):
        DefectBoundaryAuditConfig(path_coefficients=(0.125, 1.0))
    with pytest.raises(ValueError, match="ordered"):
        DefectBoundaryAuditConfig(
            alignment_loss_ceiling=0.02,
            distortion_alignment_loss=0.01,
        )


def test_path_keys_are_stable() -> None:
    assert _path_key(0.0) == "path_0"
    assert _path_key(0.125) == "path_0p125"
    assert _path_key(1.0) == "path_1"


def test_boundary_metrics_count_adjacent_introduced_errors() -> None:
    exact = torch.tensor([
        [0.51, 0.49, 0.00, 0.00],
        [0.00, 0.52, 0.48, 0.00],
        [0.00, 0.00, 0.55, 0.45],
        [0.49, 0.00, 0.00, 0.51],
    ])
    near = torch.tensor([
        [0.49, 0.51, 0.00, 0.00],
        [0.00, 0.51, 0.49, 0.00],
        [0.00, 0.00, 0.45, 0.55],
        [0.51, 0.00, 0.00, 0.49],
    ])
    metrics = prediction_boundary_metrics(near, exact, torch.tensor([0, 1, 2, 3]))
    assert metrics["disagreement_count"] == 3
    assert metrics["introduced_error_count"] == 3
    assert metrics["rescued_error_count"] == 0
    assert metrics["adjacent_disagreement_fraction"] == 1.0


def test_mechanism_classification_respects_preregistered_regions() -> None:
    config = DefectBoundaryAuditConfig()
    exact = {"circular_alignment": 0.995}
    near = {"circular_alignment": 0.992}
    local = {
        "disagreement_rate": 0.05,
        "adjacent_disagreement_fraction": 1.0,
        "mean_moment_shift_bins": 0.08,
    }
    assert classify_boundary_mechanism(local, near, exact, config) == "boundary_only"
    distorted = {**local, "mean_moment_shift_bins": 0.30}
    assert classify_boundary_mechanism(distorted, near, exact, config) == "continuous_map_distortion"
    distributed = {**local, "disagreement_rate": 0.20, "mean_moment_shift_bins": 0.15}
    assert classify_boundary_mechanism(distributed, near, exact, config) == "distributed_boundary_sensitivity"


def _endpoint(preservation: float, causal: bool = True) -> dict:
    return {
        "causal_pass": causal,
        "effect_preservation": {"preserved_fraction": preservation},
        "diagnostics": {
            "circular_alignment": 0.99,
            "exact_bin_accuracy": 0.75,
            "maximum_angular_increment": 0.4,
            "minimum_moment_magnitude": 0.9,
            "winding_degree": 2.0,
        },
    }


def test_predecessor_contract_checks_all_three_endpoints() -> None:
    current = {
        "cohort": "heldout_a",
        "regime": "composition",
        "near_rank": 2,
        "sufficient_rank": 4,
        "interventions": {
            "path_0": _endpoint(0.96, causal=False),
            "sufficient_rank": _endpoint(1.0),
            "path_1": _endpoint(1.0),
        },
    }
    previous = {
        "heldout_cells": [
            {
                "cohort": "heldout_a",
                "regime": "composition",
                "interventions": {
                    "rank_2": _endpoint(0.96, causal=False),
                    "rank_4": _endpoint(1.0),
                    "exact": _endpoint(1.0),
                },
            }
        ]
    }
    contract = predecessor_replication_contract([current], previous, 1e-8)
    assert contract["passed"] is True
    assert contract["comparison_count"] == 3
    previous["heldout_cells"][0]["interventions"]["rank_2"][
        "effect_preservation"
    ]["preserved_fraction"] = 0.90
    assert predecessor_replication_contract([current], previous, 1e-8)["passed"] is False


def test_aggregate_cannot_confirm_an_underpowered_classification() -> None:
    run = {
        "seed": 29,
        "all_primary_gates_passed": True,
        "gates": {
            "predecessor_identity": True,
            "predecessor_endpoint_replication": True,
            "exact_decomposition": True,
            "primary_endpoint_controls": True,
            "refined_minimum_sufficient_rank": 5,
        },
        "cells": [
            {
                "cohort": "heldout_b",
                "regime": "extrapolation",
                "primary_failure_cell": True,
                "mechanism_classification": "boundary_only",
                "interventions": {
                    "path_0": {"boundary_to_exact": {}},
                },
                "crossing_summary": {},
            }
        ],
    }
    summary = aggregate([run])
    assert summary["classification_counts"]["boundary_only"] == 1
    assert summary["expected_primary_seeds_present"] is False
    assert summary["confirmed"] is False
