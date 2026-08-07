import json
from pathlib import Path

import pytest

from neural_architecture_lab.repeated_reference_acquisition_meta_hypothesis import (
    build_repeated_reference_acquisition_experiment_results,
    build_repeated_reference_acquisition_meta_hypothesis,
    store_repeated_reference_acquisition_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_repeated_reference_acquisition/"
    "20260807_d8_corrective_expanded_v8/campaign_results.json"
)


def test_meta_preserves_corrective_not_confirmatory_pedigree() -> None:
    record = build_repeated_reference_acquisition_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "corrective_outcome_informed_acquisition_repair_supported_not_confirmatory"
    )
    assert hypothesis["subclaims"]["confirmatory_status"] == (
        "not_confirmatory_post_outcome_expansion"
    )


def test_meta_preserves_population_threshold_and_controls() -> None:
    record = build_repeated_reference_acquisition_meta_hypothesis(RESULTS)
    subclaims = record["hypothesis"]["subclaims"]
    assert subclaims["analytic_arm_m64"] == "supported_five_of_five"
    assert subclaims["learned_arm_m64"] == "supported_four_of_five"
    assert subclaims["all_checkpoint_repair_at_m256"] == "supported_ten_of_ten"
    assert subclaims["fiber_shuffle_specificity"] == (
        "supported_zero_of_five_both_arms"
    )
    rate = record["result"]["descriptive_metrics"]["standard_error_law"]
    assert rate["pass"] is True
    assert rate["splits"]["composition"]["log_log_slope"] == pytest.approx(
        -0.47660607260782173
    )


def test_experiment_results_keep_seed53_boundary_visible() -> None:
    record = build_repeated_reference_acquisition_meta_hypothesis(RESULTS)
    experiments = build_repeated_reference_acquisition_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 10
    assert sum(item.primary_metric for item in experiments) == 9.0
    assert all(item.metrics["analytic_m256_pass"] == 1.0 for item in experiments)
    anomalies = [item for item in experiments if item.anomalies]
    assert len(anomalies) == 1
    assert "256" in anomalies[0].anomalies[0]


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["classification"] = "mixed_acquisition_result"
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="corrective repeated-reference campaign"):
        build_repeated_reference_acquisition_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_repeated_reference_acquisition_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_repeated_reference_acquisition_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-repeated-reference-acquisition-v1",
        "experiment_count": 10,
    }
