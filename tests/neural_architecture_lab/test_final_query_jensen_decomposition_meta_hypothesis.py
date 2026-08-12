import json
from pathlib import Path

import pytest

from neural_architecture_lab.final_query_jensen_decomposition_meta_hypothesis import (
    build_final_query_jensen_decomposition_experiment_results,
    build_final_query_jensen_decomposition_meta_hypothesis,
    store_final_query_jensen_decomposition_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_final_query_jensen_decomposition/"
    "20260810_d8_seed7_registered_v2/campaign_results.json"
)


def test_meta_preserves_post_outcome_evidence_pedigree() -> None:
    record = build_final_query_jensen_decomposition_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "supported_registered_post_outcome_diagnostic_one_checkpoint"
    )
    assert hypothesis["direct_quality_experiment_count"] == 0
    assert hypothesis["diagnostic_experiment_count"] == 1
    assert hypothesis["subclaims"] == {
        "generic_jensen_sufficient_support": "yes",
        "generic_jensen_sufficient_outside_range": "yes",
        "near_complete_jensen_support": "yes",
        "near_complete_jensen_outside_range": "no_adverse_ln_modulation",
        "jensen_is_quotient_specific": "no",
        "parent_posterior_preservation_rehabilitated": "no",
        "same_scope_retraining": "no",
    }


def test_experiment_record_preserves_decisive_accounting() -> None:
    record = build_final_query_jensen_decomposition_meta_hypothesis(RESULTS)
    experiments = build_final_query_jensen_decomposition_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 1
    result = experiments[0]
    assert result.primary_metric > 1.0
    assert result.model_architecture == [8, 512]
    metrics = result.metrics
    assert metrics["training_support_jensen_over_observed_gain"] > 1.0
    assert metrics["outside_range_jensen_over_observed_gain"] > 1.36
    assert metrics["training_support_nonlinear_over_jensen"] < 0.004
    assert 0.26 < metrics["outside_range_nonlinear_over_jensen"] < 0.27
    assert metrics["training_support_control_observed_activation_gain"] < 0.0
    assert metrics["outside_range_control_observed_activation_gain"] > 0.0


def test_campaign_cannot_rewrite_post_outcome_result_as_fresh_confirmation(
    tmp_path: Path,
) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["evidence_role"] = "preregistered_confirmatory"
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="final-query Jensen campaign"):
        build_final_query_jensen_decomposition_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_final_query_jensen_decomposition_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 1


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_final_query_jensen_decomposition_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-final-query-jensen-decomposition-v1",
        "experiment_count": 1,
    }

