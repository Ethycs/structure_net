import json
from pathlib import Path

import pytest

from neural_architecture_lab.cyclic_writer_intertwiner_meta_hypothesis import (
    build_cyclic_writer_intertwiner_experiment_results,
    build_cyclic_writer_intertwiner_meta_hypothesis,
    store_cyclic_writer_intertwiner_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_cyclic_writer_intertwiner/"
    "20260807_preregistered_diagnostic/campaign_results.json"
)


def test_meta_preserves_three_of_three_intertwiner_rejection() -> None:
    record = build_cyclic_writer_intertwiner_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "rejected_harmonic_mixed_writer_image_three_of_three"
    )
    assert hypothesis["subclaims"]["writer_image_c16_invariance"] == (
        "rejected_zero_of_three"
    )
    assert hypothesis["subclaims"]["first_harmonic_dominance"] == (
        "supported_three_of_three"
    )


def test_experiment_records_preserve_absolute_failure_and_specificity() -> None:
    record = build_cyclic_writer_intertwiner_meta_hypothesis(RESULTS)
    experiments = build_cyclic_writer_intertwiner_experiment_results(record, RESULTS)
    assert len(experiments) == 3
    assert sum(item.primary_metric for item in experiments) == 0.0
    assert all(item.metrics["orbit_obstruction_pass"] == 0.0 for item in experiments)
    assert all(item.metrics["random_specificity"] == 1.0 for item in experiments)
    assert all(
        item.metrics["maximum_orbit_obstruction"]
        < item.metrics["random_fifth_percentile"]
        for item in experiments
    )
    assert all(
        item.metrics["first_harmonic_overlap"] >= 0.98 for item in experiments
    )


def test_campaign_cannot_rewrite_mixed_image_as_success(tmp_path: Path) -> None:
    campaign = json.loads(RESULTS.read_text())
    campaign["aggregates"]["writer_intertwiner_pass_count"] = 3
    path = tmp_path / "campaign_results.json"
    path.write_text(json.dumps(campaign))
    with pytest.raises(ValueError, match="cyclic writer-intertwiner campaign"):
        build_cyclic_writer_intertwiner_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_cyclic_writer_intertwiner_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text())
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 3


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_cyclic_writer_intertwiner_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-c16-writer-intertwiner-v1",
        "experiment_count": 3,
    }
