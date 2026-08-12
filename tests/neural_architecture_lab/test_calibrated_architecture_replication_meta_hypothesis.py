import json
from pathlib import Path

import pytest

from neural_architecture_lab.calibrated_architecture_replication_meta_hypothesis import (
    build_calibrated_architecture_replication_experiment_results,
    build_calibrated_architecture_replication_meta_hypothesis,
    store_calibrated_architecture_replication_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_calibrated_architecture_replication/"
    "20260810_d6_d10_preregistered/campaign_results.json"
)


def test_meta_records_negative_joint_result_and_positive_subclaims() -> None:
    record = build_calibrated_architecture_replication_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_task_adequacy_not_architecture_stable"
    )
    assert hypothesis["direct_experiment_count"] == 30
    assert hypothesis["subclaims"]["structured_causal_closure_family_stable"] == (
        "supported_all_twenty_seeds_all_cuts"
    )
    assert hypothesis["subclaims"]["natural_task_adequacy_family_stable"] == (
        "contradicted"
    )


def test_meta_preserves_locked_population_counts() -> None:
    record = build_calibrated_architecture_replication_meta_hypothesis(RESULTS)
    arms = record["result"]["descriptive_metrics"]["campaign"]["arms"]
    assert arms["d6"]["analytic_calibrated"]["joint_pass_count"] == 5
    assert arms["d10"]["analytic_calibrated"]["joint_pass_count"] == 3
    assert arms["d6"]["learned_calibrated_equivariant"]["joint_pass_count"] == 1
    assert arms["d10"]["learned_calibrated_equivariant"]["joint_pass_count"] == 1
    for preset in ("d6", "d10"):
        for condition in (
            "analytic_calibrated",
            "learned_calibrated_equivariant",
        ):
            assert arms[preset][condition]["representation_pass_count"] == 5
            assert arms[preset][condition]["causal_all_cuts_pass_count"] == 5


def test_experiment_results_preserve_all_thirty_replication_units() -> None:
    record = build_calibrated_architecture_replication_meta_hypothesis(RESULTS)
    experiments = build_calibrated_architecture_replication_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 30
    assert sum(item.primary_metric for item in experiments) == 10
    structured = [
        item
        for item in experiments
        if "Condition: raw_calibrated." not in item.observations
    ]
    assert len(structured) == 20
    assert all(item.metrics["representation_pass"] == 1 for item in structured)
    assert all(item.metrics["causal_all_cuts_pass"] == 1 for item in structured)


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["primary_hypothesis_pass"] = True
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="calibrated architecture campaign"):
        build_calibrated_architecture_replication_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_calibrated_architecture_replication_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 30


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_calibrated_architecture_replication_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-calibrated-architecture-replication-v1",
        "experiment_count": 30,
    }
