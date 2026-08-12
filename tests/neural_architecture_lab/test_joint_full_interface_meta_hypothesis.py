import json
from pathlib import Path

import pytest

from neural_architecture_lab.joint_full_interface_meta_hypothesis import (
    build_joint_full_interface_experiment_results,
    build_joint_full_interface_meta_hypothesis,
    store_joint_full_interface_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_joint_full_interface/"
    "20260811_d6_d10_preregistered/campaign_results.json"
)


def test_meta_rejects_full_interface_repair() -> None:
    record = build_joint_full_interface_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_physical_true_zero_of_five_both_presets"
    )
    assert hypothesis["subclaims"]["architecture_family_full_interface_repair"] == (
        "contradicted_zero_of_ten"
    )
    assert hypothesis["subclaims"]["flexible_joint_supervision_branch"] == (
        "closed_require_fixed_chart_by_construction"
    )


def test_meta_preserves_population_and_control_counts() -> None:
    record = build_joint_full_interface_meta_hypothesis(RESULTS)
    strata = record["result"]["descriptive_metrics"]["strata"]
    for preset in ("d6", "d10"):
        assert strata[preset]["physical_true_pass_count"] == 0
        assert strata[preset]["pair_shuffled_pass_count"] == 0
        assert strata[preset]["valid_count"] == 5


def test_direct_evidence_preserves_changed_full_continuations() -> None:
    record = build_joint_full_interface_meta_hypothesis(RESULTS)
    cells = record["evidence"]["direct_tests"]
    assert len(cells) == 10
    assert all(cell["model_changed"] for cell in cells)
    assert all(cell["model_topology_unchanged"] for cell in cells)
    d10 = [cell for cell in cells if cell["preset"] == "d10"]
    assert all(
        cell["cuts"]["frontend"]["composition"]["cosine_correlation"] < -0.99
        for cell in d10
    )
    assert all(
        cell["cuts"]["full"]["composition"]["cosine_correlation"] > 0.99
        for cell in d10
    )


def test_experiment_results_preserve_ten_seed_units() -> None:
    record = build_joint_full_interface_meta_hypothesis(RESULTS)
    experiments = build_joint_full_interface_experiment_results(record, RESULTS)
    assert len(experiments) == 10
    assert sum(item.primary_metric for item in experiments) == 0
    assert all(item.metrics["validity"] == 1 for item in experiments)
    assert all(item.metrics["continuation_changed"] == 1 for item in experiments)
    assert all(item.metrics["pair_shuffled_joint_seed_pass"] == 0 for item in experiments)


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["classification"] = (
        "full_interface_physical_typing_architecture_stable"
    )
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="full-interface campaign"):
        build_joint_full_interface_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_joint_full_interface_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_joint_full_interface_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-joint-full-interface-physical-typing-v1",
        "experiment_count": 10,
    }
