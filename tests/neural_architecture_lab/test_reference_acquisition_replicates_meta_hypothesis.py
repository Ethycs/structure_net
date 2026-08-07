import json
from pathlib import Path

import pytest

from neural_architecture_lab.reference_acquisition_replicates_meta_hypothesis import (
    build_reference_acquisition_replicates_experiment_results,
    build_reference_acquisition_replicates_meta_hypothesis,
    store_reference_acquisition_replicates_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_reference_acquisition_replicates/"
    "20260807_d8_preregistered/campaign_results.json"
)


def test_meta_preserves_invalid_primary_and_supported_subresult() -> None:
    record = build_reference_acquisition_replicates_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "invalid_causal_ceiling_nonportable_acquisition_subresult_supported"
    )
    assert hypothesis["subclaims"]["complete_preregistered_hypothesis"] == (
        "invalid"
    )
    assert hypothesis["subclaims"]["analytic_acquisition_repair_at_m64"] == (
        "supported_five_of_five_both_arms"
    )
    assert hypothesis["subclaims"]["one_step_true_coordinate_ceiling"] == (
        "rejected_zero_of_five_analytic_two_of_five_learned"
    )


def test_meta_preserves_scaling_controls_and_no_learned_advantage() -> None:
    record = build_reference_acquisition_replicates_meta_hypothesis(RESULTS)
    metrics = record["result"]["descriptive_metrics"]
    assert metrics["scaling"]["contract"] is True
    assert metrics["scaling"]["slopes"]["composition"] == pytest.approx(
        -0.48368462595852607
    )
    assert metrics["campaign"]["misgrouped_control_pass_counts"] == {
        "analytic_calibrated": 0,
        "learned_calibrated_equivariant": 0,
    }
    assert metrics["learned_analytic_comparison"]["contract"] is True


def test_experiment_results_keep_complete_gate_separate_from_acquisition() -> None:
    record = build_reference_acquisition_replicates_meta_hypothesis(RESULTS)
    experiments = build_reference_acquisition_replicates_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 10
    assert all(item.metrics["campaign_valid"] == 0.0 for item in experiments)
    assert all(item.metrics["analytic_m64_task_gate"] == 1.0 for item in experiments)
    assert all(item.metrics["learned_m64_task_gate"] == 1.0 for item in experiments)
    assert sum(item.primary_metric for item in experiments) == 2.0
    assert sum(item.metrics["target_oracle_pass"] for item in experiments) == 2.0
    assert all(item.metrics["misgrouped_control_pass"] == 0.0 for item in experiments)


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["classification"] = "acquisition_precision_sufficient"
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="repeated-reference acquisition campaign"):
        build_reference_acquisition_replicates_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_reference_acquisition_replicates_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_reference_acquisition_replicates_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-reference-acquisition-replicates-v1",
        "experiment_count": 10,
    }
    assert (tmp_path / "experiment_queue").is_dir()
