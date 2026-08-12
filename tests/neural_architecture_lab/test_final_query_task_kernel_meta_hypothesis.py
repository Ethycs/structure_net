import json
from pathlib import Path

import pytest

from neural_architecture_lab.final_query_task_kernel_meta_hypothesis import (
    build_final_query_task_kernel_experiment_results,
    build_final_query_task_kernel_meta_hypothesis,
    store_final_query_task_kernel_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_final_query_task_kernel/"
    "20260810_d8_seed7_preregistered/campaign_results.json"
)


def test_meta_preserves_kernel_success_and_contraction_failure() -> None:
    record = build_final_query_task_kernel_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_valid_kernel_preservation_but_contraction_failed"
    )
    assert hypothesis["evidence_count"] == 1
    assert hypothesis["direct_quality_experiment_count"] == 1
    assert hypothesis["fixed_checkpoint_count"] == 1
    assert hypothesis["subclaims"] == {
        "local_task_kernel_preserves_output": "yes_both_regimes",
        "local_task_kernel_contains_most_fiber_chord": "no",
        "final_query_task_rowspace_contains_material_fiber_chord": "yes",
        "task_component_explains_full_output_change": "yes_both_regimes",
        "random_rank15_specificity": "yes_both_regimes",
        "same_scope_retraining": "no",
    }


def test_experiment_record_preserves_decisive_metrics() -> None:
    record = build_final_query_task_kernel_meta_hypothesis(RESULTS)
    experiments = build_final_query_task_kernel_experiment_results(record, RESULTS)
    assert len(experiments) == 1
    result = experiments[0]
    assert result.primary_metric == 0.0
    assert result.model_architecture == [8, 512]
    assert result.metrics["training_support_kernel_task_sufficient"] == 1.0
    assert result.metrics["outside_range_kernel_task_sufficient"] == 1.0
    assert result.metrics["training_support_remaining_pair_ratio"] > 0.80
    assert result.metrics["outside_range_remaining_pair_ratio"] > 0.84
    assert result.metrics["training_support_kernel_posterior_js"] < 0.001
    assert result.metrics["outside_range_kernel_posterior_js"] < 0.02
    assert result.metrics["outside_range_full_posterior_js"] > 0.20
    assert result.metrics["training_support_random_js_disadvantage"] > 0.01
    assert result.metrics["outside_range_random_js_disadvantage"] > 0.17


def test_campaign_cannot_rewrite_contraction_failure_as_success(
    tmp_path: Path,
) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["gates"]["primary_hypothesis_pass"] = True
    campaign["regimes"]["training_support"]["gates"][
        "kernel_pair_contraction"
    ] = True
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="final-query task/kernel campaign"):
        build_final_query_task_kernel_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_final_query_task_kernel_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 1


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_final_query_task_kernel_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-final-query-task-kernel-barycenter-v1",
        "experiment_count": 1,
    }
