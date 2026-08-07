import json
from pathlib import Path

import pytest

from neural_architecture_lab.source_task_covector_portability_meta_hypothesis import (
    build_source_task_covector_portability_experiment_results,
    build_source_task_covector_portability_meta_hypothesis,
    store_source_task_covector_portability_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_source_task_covector_portability/"
    "20260807_d6_preregistered_fresh_cohort/campaign_results.json"
)


def test_meta_preserves_covector_and_scalar_component_split() -> None:
    record = build_source_task_covector_portability_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_source_covector_portable_scalar_amplitude_failed_three_of_three"
    )
    assert hypothesis["subclaims"]["source_covector_causal_sufficiency_with_fresh_error"] == (
        "supported_three_of_three_six_of_six_cells"
    )
    assert hypothesis["subclaims"]["source_signed_error_prediction"] == (
        "not_supported_three_of_three"
    )
    assert hypothesis["subclaims"]["full_preregistered_gate"] == (
        "not_confirmed_zero_of_three"
    )


def test_experiment_records_preserve_zero_full_gates_and_three_covector_passes() -> None:
    record = build_source_task_covector_portability_meta_hypothesis(RESULTS)
    experiments = build_source_task_covector_portability_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(
        item.metrics["source_covector_oracle_error_all_cells_pass"] == 1.0
        for item in experiments
    )
    assert all(
        item.metrics["source_predicted_all_cells_pass"] == 0.0
        for item in experiments
    )
    assert min(
        item.metrics["fresh_covector_zero_referenced_r2"] for item in experiments
    ) > 0.98


def test_campaign_cannot_rewrite_component_failure_as_success(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["source_covector_portability_pass_count"] = 3
    campaign["aggregates"]["supported"] = True
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="source task-covector portability campaign"):
        build_source_task_covector_portability_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_source_task_covector_portability_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_source_task_covector_portability_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c2-source-task-covector-portability-v1",
        "experiment_count": 3,
    }
