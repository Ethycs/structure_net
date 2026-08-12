import json
from pathlib import Path

import pytest

from neural_architecture_lab.acquisition_draw_stability_meta_hypothesis import (
    build_acquisition_draw_stability_experiment_results,
    build_acquisition_draw_stability_meta_hypothesis,
    store_acquisition_draw_stability_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_acquisition_draw_stability/"
    "20260810_d16_preregistered/campaign_results.json"
)


def test_meta_confirms_only_the_preregistered_m256_scope() -> None:
    record = build_acquisition_draw_stability_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "preregistered_m256_stability_confirmed_m64_checkpoint_variable"
    )
    assert hypothesis["independent_acquisition_draw_count"] == 16
    assert hypothesis["subclaims"]["independent_gaussian_sample_count_branch"] == (
        "closed"
    )


def test_meta_preserves_joint_draw_counts_and_controls() -> None:
    record = build_acquisition_draw_stability_meta_hypothesis(RESULTS)
    campaign = record["result"]["descriptive_metrics"]["campaign"]
    assert campaign["counts"]["m64"]["complete_draw_passes"] == 14
    assert campaign["counts"]["m256"]["complete_draw_passes"] == 16
    assert campaign["counts"]["m64"]["arm_population_draw_passes"] == {
        "analytic_calibrated": 14,
        "learned_calibrated_equivariant": 16,
    }
    assert campaign["controls_pass"] is True
    assert record["hypothesis"]["subclaims"]["fiber_shuffle_specificity"] == (
        "supported_zero_of_five_both_arms"
    )


def test_checkpoint_summaries_keep_draw_variability_visible() -> None:
    record = build_acquisition_draw_stability_meta_hypothesis(RESULTS)
    experiments = build_acquisition_draw_stability_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 10
    assert all(item.primary_metric == 1.0 for item in experiments)
    by_id = {item.experiment_id: item for item in experiments}
    seed53 = by_id[
        "tinyllm-acquisition-draw-learned_calibrated_equivariant-seed53"
    ]
    assert seed53.metrics["m64_draw_passes"] == 10.0
    assert seed53.metrics["m256_draw_passes"] == 16.0
    assert "10/16" in seed53.anomalies[0]


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["counts"]["m256"]["complete_draw_passes"] = 15
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="acquisition-draw stability campaign"):
        build_acquisition_draw_stability_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_acquisition_draw_stability_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_acquisition_draw_stability_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-acquisition-draw-stability-v1",
        "experiment_count": 10,
    }
