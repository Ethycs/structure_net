import json
from pathlib import Path

import pytest

from neural_architecture_lab.affine_gauge_transport_meta_hypothesis import (
    build_affine_gauge_transport_experiment_results,
    build_affine_gauge_transport_meta_hypothesis,
    store_affine_gauge_transport_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_affine_gauge_transport/"
    "20260811_d6_d10_registered/campaign_results.json"
)


def test_meta_rejects_pure_affine_gauge_sufficiency() -> None:
    record = build_affine_gauge_transport_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_training_affine_chart_fails_population_transport"
    )
    assert hypothesis["subclaims"]["pure_affine_gauge_sufficiency"] == (
        "contradicted_d6_two_of_five_d10_one_of_five"
    )
    assert hypothesis["subclaims"]["post_hoc_scalar_map_branch"] == (
        "closed_require_fixed_chart_by_construction"
    )


def test_meta_preserves_registered_population_and_control_counts() -> None:
    record = build_affine_gauge_transport_meta_hypothesis(RESULTS)
    strata = record["result"]["descriptive_metrics"]["strata"]
    assert strata["d6"]["physical_true_canonical_front_pass_count"] == 2
    assert strata["d10"]["physical_true_canonical_front_pass_count"] == 2
    assert strata["d6"]["physical_true_joint_pass_count"] == 2
    assert strata["d10"]["physical_true_joint_pass_count"] == 1
    assert all(
        strata[preset]["pair_shuffled_joint_pass_count"] == 0
        for preset in ("d6", "d10")
    )


def test_direct_evidence_preserves_near_affinity_and_transport_failure() -> None:
    record = build_affine_gauge_transport_meta_hypothesis(RESULTS)
    cells = record["evidence"]["direct_tests"]
    assert len(cells) == 10
    assert all(cell["training_r_squared"] > 0.99 for cell in cells)
    assert all(cell["maximum_float32_transport_error"] <= 2e-6 for cell in cells)
    assert sum(cell["physical_true_joint_seed_pass"] for cell in cells) == 3
    assert not any(cell["pair_shuffled_joint_seed_pass"] for cell in cells)
    assert all(
        cell["regimes"]["composition"]["endpoint_pass"] for cell in cells
    )
    assert sum(
        cell["regimes"]["extrapolation"]["endpoint_pass"] for cell in cells
    ) == 4


def test_direct_evidence_preserves_orientation_classes() -> None:
    record = build_affine_gauge_transport_meta_hypothesis(RESULTS)
    cells = record["evidence"]["direct_tests"]
    d6 = [cell for cell in cells if cell["preset"] == "d6"]
    d10 = [cell for cell in cells if cell["preset"] == "d10"]
    assert sum(cell["orientation"] > 0 for cell in d6) == 3
    assert all(cell["orientation"] < 0 for cell in d10)


def test_experiment_results_preserve_ten_seed_units() -> None:
    record = build_affine_gauge_transport_meta_hypothesis(RESULTS)
    experiments = build_affine_gauge_transport_experiment_results(record, RESULTS)
    assert len(experiments) == 10
    assert sum(item.primary_metric for item in experiments) == 3
    assert all(item.metrics["validity"] == 1 for item in experiments)
    assert all(item.metrics["optimizer_steps"] == 0 for item in experiments)
    assert all(item.metrics["trained_parameters"] == 0 for item in experiments)
    assert not any(
        item.metrics["pair_shuffled_joint_seed_pass"] for item in experiments
    )


def test_campaign_tampering_is_rejected(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text(encoding="utf-8"))
    campaign["aggregates"]["classification"] = "affine_gauge_transport_sufficient"
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(ValueError, match="affine-gauge campaign"):
        build_affine_gauge_transport_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_affine_gauge_transport_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 10


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_affine_gauge_transport_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-affine-gauge-transport-v1",
        "experiment_count": 10,
    }
