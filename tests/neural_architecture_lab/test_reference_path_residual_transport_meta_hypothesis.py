import json
from pathlib import Path

import pytest

from neural_architecture_lab.reference_path_residual_transport_meta_hypothesis import (
    build_reference_path_residual_transport_experiment_results,
    build_reference_path_residual_transport_meta_hypothesis,
    store_reference_path_residual_transport_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_reference_path_residual_transport/"
    "20260807_d8_corrected_v4/campaign_results.json"
)


def test_meta_preserves_semantic_schedule_only_falsification() -> None:
    record = build_reference_path_residual_transport_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "semantic_schedule_only_finite_radius_hypothesis_rejected"
    )
    assert hypothesis["subclaims"]["k16_true_cosine_transport"] == (
        "supported_five_of_five_analytic_four_of_five_learned"
    )
    assert hypothesis["subclaims"]["k16_path_moment_transport"] == (
        "rejected_zero_of_five_both_arms"
    )
    assert hypothesis["subclaims"]["finite_radius_only_explanation"] == "rejected"


def test_experiment_records_keep_joint_gate_separate_from_semantic_subresult() -> None:
    record = build_reference_path_residual_transport_meta_hypothesis(RESULTS)
    experiments = build_reference_path_residual_transport_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 10
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert sum(item.metrics["k16_true_cosine_pass"] for item in experiments) == 9.0
    assert sum(item.metrics["k16_path_moment_pass"] for item in experiments) == 0.0
    assert sum(
        item.metrics["k16_shuffled_path_moment_pass"] for item in experiments
    ) == 0.0
    assert all(item.metrics["campaign_valid"] == 1.0 for item in experiments)


def test_campaign_cannot_rewrite_path_failure_as_success(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["classification"] = "finite_radius_repair"
    campaign["aggregates"]["rollout_pass_counts"]["path_moment"]["16"] = {
        "analytic_calibrated": 5,
        "learned_calibrated_equivariant": 5,
    }
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(
        ValueError, match="reference-path residual-transport campaign"
    ):
        build_reference_path_residual_transport_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_reference_path_residual_transport_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_reference_path_residual_transport_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-reference-path-residual-transport-v1",
        "experiment_count": 10,
    }
