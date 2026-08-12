import json
from pathlib import Path

import pytest

from neural_architecture_lab.noisy_bias_pilot_acquisition_meta_hypothesis import (
    build_noisy_bias_pilot_acquisition_experiment_results,
    build_noisy_bias_pilot_acquisition_meta_hypothesis,
    store_noisy_bias_pilot_acquisition_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_noisy_bias_pilot_acquisition/"
    "20260810_d16_preregistered/campaign_results.json"
)


def test_meta_preserves_finite_noisy_pilot_result() -> None:
    record = build_noisy_bias_pilot_acquisition_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is True
    assert hypothesis["confirmation_status"] == (
        "preregistered_finite_noisy_pilot_m4_population_repair_confirmed"
    )
    assert hypothesis["evidence_count"] == 16
    assert hypothesis["stored_checkpoint_summary_count"] == 10
    assert hypothesis["subclaims"] == {
        "m256_complete_draw_repair": "supported_sixteen_of_sixteen",
        "m256_checkpoint_cells": "supported_160_of_160",
        "smallest_reliable_population_count": "four",
        "m4_complete_draw_repair": "supported_sixteen_of_sixteen",
        "m4_system_level_ceiling": "not_supported_two_checkpoint_failures",
        "m1_complete_draw_repair": "insufficient_twelve_of_sixteen",
        "wrong_sign_specificity": (
            "supported_analytic_one_of_five_learned_zero_of_five"
        ),
        "exact_pilot_source": "supported_five_of_five_both_arms",
        "new_randomness_or_fitting": "none",
        "exact_gaussian_pilot_count_branch": "closed",
        "same_scope_retraining": "not_licensed",
    }


def test_experiment_records_preserve_count_ladder_and_controls() -> None:
    record = build_noisy_bias_pilot_acquisition_meta_hypothesis(RESULTS)
    experiments = build_noisy_bias_pilot_acquisition_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 10
    assert all(item.metrics["validity"] == 1.0 for item in experiments)
    assert all(item.metrics["m256_draw_passes"] == 16.0 for item in experiments)
    assert sum(item.metrics["m4_draw_passes"] for item in experiments) == 155.0
    assert sum(item.metrics["m1_draw_passes"] for item in experiments) == 136.0
    assert sum(item.metrics["source_exact_pilot_seed_pass"] for item in experiments) == 10.0
    assert sum(item.metrics["source_full_plus_seed_pass"] for item in experiments) == 4.0
    assert sum(item.metrics["wrong_sign_seed_pass"] for item in experiments) == 1.0
    assert sum(item.primary_metric for item in experiments) == 155.0 / 16.0


def test_campaign_cannot_rewrite_count_boundary(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["smallest_reliable_count"] = 1
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="noisy-bias pilot campaign"):
        build_noisy_bias_pilot_acquisition_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_noisy_bias_pilot_acquisition_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is True
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_noisy_bias_pilot_acquisition_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-noisy-bias-pilot-acquisition-v1",
        "experiment_count": 10,
    }
