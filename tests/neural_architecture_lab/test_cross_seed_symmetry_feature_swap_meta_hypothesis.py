import json
from pathlib import Path

import pytest

from neural_architecture_lab.cross_seed_symmetry_feature_swap_meta_hypothesis import (
    build_cross_seed_symmetry_feature_swap_experiment_results,
    build_cross_seed_symmetry_feature_swap_meta_hypothesis,
    store_cross_seed_symmetry_feature_swap_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_cross_seed_symmetry_feature_swap/"
    "20260806_d8_preregistered/campaign_results.json"
)


def test_meta_record_keeps_exact_symmetry_separate_from_portability() -> None:
    record = build_cross_seed_symmetry_feature_swap_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_exact_equivariance_checkpoint_local_gauges"
    )
    assert hypothesis["subclaims"]["exact_acquisition_group_contract"] == (
        "supported_five_of_five"
    )
    assert hypothesis["subclaims"]["portable_three_channel_feature_gauge"] == (
        "rejected_zero_of_twenty_pairs"
    )
    assert hypothesis["subclaims"]["extrapolation_cell_portability"] == (
        "rejected_zero_of_forty_cells"
    )


def test_experiment_records_preserve_pair_and_cell_failures() -> None:
    record = build_cross_seed_symmetry_feature_swap_meta_hypothesis(RESULTS)
    experiments = build_cross_seed_symmetry_feature_swap_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 20
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert sum(item.metrics["feature_cell_pass_count"] for item in experiments) == 8.0
    assert sum(item.metrics["scalar_cell_pass_count"] for item in experiments) == 0.0
    assert all(item.metrics["pair_group_contract"] == 1.0 for item in experiments)


def test_campaign_cannot_promote_symmetry_contract_to_portability(
    tmp_path: Path,
) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["confirmed"] = True
    campaign["aggregates"]["pair_pass_count"] = 20
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="symmetry-feature swap campaign"):
        build_cross_seed_symmetry_feature_swap_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_cross_seed_symmetry_feature_swap_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 20


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_cross_seed_symmetry_feature_swap_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-cross-seed-symmetry-feature-gauge-v1",
        "experiment_count": 20,
    }
