import json
from pathlib import Path

import pytest

from neural_architecture_lab.task_relative_activation_barycenter_meta_hypothesis import (
    build_task_relative_activation_barycenter_experiment_results,
    build_task_relative_activation_barycenter_meta_hypothesis,
    store_task_relative_activation_barycenter_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_task_relative_activation_barycenter/"
    "20260810_seed7_preregistered/campaign_results.json"
)


def test_meta_preserves_valid_d8_null_and_d6_quarantine() -> None:
    record = build_task_relative_activation_barycenter_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_d8_valid_no_causal_front_d6_baseline_invalid"
    )
    assert hypothesis["direct_quality_experiment_count"] == 1
    assert hypothesis["validity_only_experiment_count"] == 1
    assert hypothesis["subclaims"]["d8_mature_causal_barycenter_front"] == (
        "not_supported_no_passing_cut"
    )
    assert hypothesis["subclaims"]["d6_mature_causal_barycenter_front"] == (
        "not_interpretable_baseline_invalid"
    )
    assert hypothesis["subclaims"]["query_only_barycenter_mature_front"] == (
        "impossible_by_exact_final_cut_equivalence"
    )
    assert record["provenance"]["deductive_corollaries"] == {
        "final_post_mlp_full_equals_query_only": True,
        "final_post_mlp_context_only_is_inert": True,
        "query_only_mature_front_possible": False,
    }


def test_experiment_records_preserve_evidence_pedigree() -> None:
    record = build_task_relative_activation_barycenter_meta_hypothesis(RESULTS)
    experiments = build_task_relative_activation_barycenter_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 2
    by_preset = {item.model_architecture[0]: item for item in experiments}
    assert by_preset[6].metrics["validity"] == 0.0
    assert by_preset[8].metrics["validity"] == 1.0
    assert by_preset[8].metrics["mature_causal_front_exists"] == 0.0
    assert by_preset[8].metrics["outside_range_final_posterior_js"] > 0.20
    assert by_preset[8].metrics["training_support_final_accuracy_gain"] > 0.13


def test_campaign_cannot_promote_invalid_d6_or_invent_front(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["valid"] = True
    campaign["aggregates"]["mature_causal_fronts"]["d8"] = (
        "block_3_post_attention"
    )
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="task-relative barycenter campaign"):
        build_task_relative_activation_barycenter_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_task_relative_activation_barycenter_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 2


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_task_relative_activation_barycenter_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-task-relative-activation-barycenter-v1",
        "experiment_count": 2,
    }
