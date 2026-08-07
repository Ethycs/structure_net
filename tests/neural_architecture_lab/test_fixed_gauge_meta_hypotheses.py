import json
from pathlib import Path

import pytest

from neural_architecture_lab.fixed_gauge_meta_hypotheses import (
    build_fixed_gauge_error_decomposition_experiment_results,
    build_fixed_gauge_error_decomposition_meta_hypothesis,
    build_fixed_semantic_gauge_writer_experiment_results,
    build_fixed_semantic_gauge_writer_meta_hypothesis,
    build_fixed_gauge_writer_capacity_experiment_results,
    build_fixed_gauge_writer_capacity_meta_hypothesis,
    store_fixed_gauge_error_decomposition_meta_hypothesis,
    store_fixed_semantic_gauge_writer_meta_hypothesis,
    store_fixed_gauge_writer_capacity_meta_hypothesis,
)


WRITER = Path(
    "data/experiments/tinyllm_fixed_semantic_gauge_writer/"
    "20260806_d6_preregistered_diagnostic/campaign_results.json"
)
DECOMPOSITION = Path(
    "data/experiments/tinyllm_fixed_gauge_error_decomposition/"
    "20260806_d6_preregistered_diagnostic/campaign_results.json"
)
CAPACITY = Path(
    "data/experiments/tinyllm_fixed_gauge_writer_capacity/"
    "20260806_d6_preregistered_diagnostic/campaign_results.json"
)


def test_writer_meta_preserves_failed_joint_gate_and_positive_controls() -> None:
    record = build_fixed_semantic_gauge_writer_meta_hypothesis(WRITER)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_observation_and_causal_writer_failed_three_of_three"
    )
    assert hypothesis["subclaims"]["fixed_portable_semantic_gauge"] == (
        "rejected_zero_of_three"
    )
    assert hypothesis["subclaims"]["shuffled_specificity"] == (
        "supported_three_of_three"
    )
    experiments = build_fixed_semantic_gauge_writer_experiment_results(
        record, WRITER
    )
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(item.metrics["shuffled_specificity"] == 1.0 for item in experiments)


def test_decomposition_meta_preserves_three_writer_limited_classes() -> None:
    record = build_fixed_gauge_error_decomposition_meta_hypothesis(DECOMPOSITION)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "diagnostic_writer_or_carrier_limited_three_of_three"
    )
    assert hypothesis["subclaims"]["oracle_refit_linear_writer"] == (
        "rejected_zero_of_three"
    )
    assert hypothesis["subclaims"]["writer_or_carrier_limitation"] == (
        "supported_three_of_three"
    )
    experiments = build_fixed_gauge_error_decomposition_experiment_results(
        record, DECOMPOSITION
    )
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(item.metrics["oracle_carrier_contract"] == 1.0 for item in experiments)


def test_capacity_meta_preserves_three_unresolved_classes() -> None:
    record = build_fixed_gauge_writer_capacity_meta_hypothesis(CAPACITY)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "diagnostic_unresolved_writer_limited_three_of_three"
    )
    assert hypothesis["subclaims"]["low_order_curvature_sufficiency"] == (
        "rejected_zero_of_three"
    )
    assert hypothesis["subclaims"]["shuffled_specificity"] == (
        "supported_twelve_of_twelve_candidates"
    )
    experiments = build_fixed_gauge_writer_capacity_experiment_results(
        record, CAPACITY
    )
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(item.metrics["quotient_order4_specific_causal_pass"] == 0.0 for item in experiments)


def test_writer_campaign_cannot_be_silently_promoted(tmp_path: Path) -> None:
    campaign = json.loads(WRITER.read_text())
    campaign["aggregates"]["gate_counts"]["fixed_gauge_causal_writer"] = 3
    campaign["aggregates"]["confirmed"] = True
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="fixed-gauge campaign"):
        build_fixed_semantic_gauge_writer_meta_hypothesis(path)


def test_decomposition_classification_cannot_be_rewritten(tmp_path: Path) -> None:
    campaign = json.loads(DECOMPOSITION.read_text())
    campaign["aggregates"]["classification_by_seed"]["53"] = "sensor_limited"
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="fixed-gauge campaign|decomposition classes"):
        build_fixed_gauge_error_decomposition_meta_hypothesis(path)


def test_capacity_classification_cannot_be_rewritten(tmp_path: Path) -> None:
    campaign = json.loads(CAPACITY.read_text())
    campaign["aggregates"]["candidate_specific_pass_counts"]["quotient_order4"] = 3
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="writer-capacity campaign"):
        build_fixed_gauge_writer_capacity_meta_hypothesis(path)


def test_json_storage_round_trip_for_both_records(tmp_path: Path) -> None:
    writer = store_fixed_semantic_gauge_writer_meta_hypothesis(
        WRITER, tmp_path / "writer.json", chromadb_path=None
    )
    decomposition = store_fixed_gauge_error_decomposition_meta_hypothesis(
        DECOMPOSITION, tmp_path / "decomposition.json", chromadb_path=None
    )
    capacity = store_fixed_gauge_writer_capacity_meta_hypothesis(
        CAPACITY, tmp_path / "capacity.json", chromadb_path=None
    )
    assert writer["storage"]["experiment_ids"] == [
        "tinyllm-c2-fixed-semantic-gauge-writer-seed7",
        "tinyllm-c2-fixed-semantic-gauge-writer-seed29",
        "tinyllm-c2-fixed-semantic-gauge-writer-seed53",
    ]
    assert len(decomposition["storage"]["experiment_ids"]) == 3
    assert len(capacity["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip_for_both_records(tmp_path: Path) -> None:
    chroma = tmp_path / "chroma"
    writer = store_fixed_semantic_gauge_writer_meta_hypothesis(
        WRITER, tmp_path / "writer.json", chromadb_path=chroma
    )
    decomposition = store_fixed_gauge_error_decomposition_meta_hypothesis(
        DECOMPOSITION, tmp_path / "decomposition.json", chromadb_path=chroma
    )
    capacity = store_fixed_gauge_writer_capacity_meta_hypothesis(
        CAPACITY, tmp_path / "capacity.json", chromadb_path=chroma
    )
    assert writer["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-fixed-semantic-gauge-writer-v1",
        "experiment_count": 3,
    }
    assert decomposition["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-fixed-gauge-error-decomposition-v1",
        "experiment_count": 3,
    }
    assert capacity["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-fixed-gauge-writer-capacity-v1",
        "experiment_count": 3,
    }
