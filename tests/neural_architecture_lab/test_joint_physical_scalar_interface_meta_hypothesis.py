import json
from pathlib import Path

import pytest

from neural_architecture_lab.joint_physical_scalar_interface_meta_hypothesis import (
    build_joint_physical_scalar_interface_experiment_results,
    build_joint_physical_scalar_interface_meta_hypothesis,
    store_joint_physical_scalar_interface_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_joint_physical_scalar_interface/"
    "20260811_d6_d10_preregistered/campaign_results.json"
)


def test_meta_rejects_joint_physical_interface_claim() -> None:
    record = build_joint_physical_scalar_interface_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_learned_frontend_zero_of_five_both_presets"
    )
    assert hypothesis["subclaims"]["learned_frontend_physical_typing"] == (
        "contradicted_zero_of_ten"
    )


def test_meta_preserves_positive_and_specificity_controls() -> None:
    record = build_joint_physical_scalar_interface_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    strata = record["result"]["descriptive_metrics"]["strata"]
    assert hypothesis["subclaims"]["analytic_positive_control"] == (
        "supported_d6_five_of_five_d10_four_of_five"
    )
    assert hypothesis["subclaims"]["semantic_specificity"] == (
        "supported_zero_of_twenty_pair_shuffled"
    )
    assert strata["d6/analytic_calibrated"]["physical_true_pass_count"] == 5
    assert strata["d10/analytic_calibrated"]["physical_true_pass_count"] == 4


def test_experiment_results_preserve_twenty_seed_units() -> None:
    record = build_joint_physical_scalar_interface_meta_hypothesis(RESULTS)
    experiments = build_joint_physical_scalar_interface_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 20
    assert sum(item.primary_metric for item in experiments) == 9
    assert all(item.metrics["validity"] == 1 for item in experiments)
    assert all(item.metrics["pair_shuffled_joint_seed_pass"] == 0 for item in experiments)


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["classification"] = (
        "frozen_backbone_joint_physical_interface_architecture_stable"
    )
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="joint physical-interface campaign"):
        build_joint_physical_scalar_interface_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_joint_physical_scalar_interface_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 20


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_joint_physical_scalar_interface_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-joint-physical-scalar-interface-v1",
        "experiment_count": 20,
    }
