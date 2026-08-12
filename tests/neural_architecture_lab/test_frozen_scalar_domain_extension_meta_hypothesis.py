import json
from pathlib import Path

import pytest

from neural_architecture_lab.frozen_scalar_domain_extension_meta_hypothesis import (
    build_frozen_scalar_domain_extension_experiment_results,
    build_frozen_scalar_domain_extension_meta_hypothesis,
    store_frozen_scalar_domain_extension_meta_hypothesis,
)


RESULTS = Path(
    "data/experiments/tinyllm_frozen_scalar_domain_extension/"
    "20260811_d10_learned_seed29_registered/result.json"
)


def test_meta_rejects_bounded_encoder_explanation() -> None:
    record = build_frozen_scalar_domain_extension_meta_hypothesis(RESULTS)
    hypothesis = record["hypothesis"]
    assert hypothesis["confirmed"] is False
    assert hypothesis["confirmation_status"] == (
        "not_confirmed_no_missing_bin_appears_through_radius_8"
    )
    assert hypothesis["direct_experiment_count"] == 1
    assert hypothesis["subclaims"]["bounded_encoder_range_explains_hole"] == (
        "contradicted_zero_of_four_bins"
    )


def test_meta_preserves_every_radius_and_shift() -> None:
    record = build_frozen_scalar_domain_extension_meta_hypothesis(RESULTS)
    direct = record["evidence"]["direct_tests"][0]
    assert direct["discovered_external_bins"] == {
        "composition": [],
        "extrapolation": [],
    }
    for regime in ("composition", "extrapolation"):
        for radius in ("1", "2", "4", "8"):
            assert 0 < direct[f"{regime}_radius_{radius}_reachability"] < 1


def test_experiment_result_is_one_no_fit_frozen_unit() -> None:
    record = build_frozen_scalar_domain_extension_meta_hypothesis(RESULTS)
    experiments = build_frozen_scalar_domain_extension_experiment_results(
        record, RESULTS
    )
    assert len(experiments) == 1
    assert experiments[0].primary_metric == 0
    assert experiments[0].metrics["validity"] == 1
    assert experiments[0].metrics["common_external_discovered_bin_fraction"] == 0


def test_result_tampering_is_rejected(tmp_path: Path) -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    result["summary"]["common_external_discovered_bin_count"] = 4
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen scalar-domain result"):
        build_frozen_scalar_domain_extension_meta_hypothesis(path)


def test_json_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_frozen_scalar_domain_extension_meta_hypothesis(
        RESULTS, output, chromadb_path=None
    )
    readback = json.loads(output.read_text(encoding="utf-8"))
    assert readback["hypothesis"]["id"] == stored["hypothesis"]["id"]
    assert readback["hypothesis"]["confirmed"] is False
    assert len(readback["storage"]["experiment_ids"]) == 1


def test_chromadb_storage_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    stored = store_frozen_scalar_domain_extension_meta_hypothesis(
        RESULTS, output, chromadb_path=tmp_path / "chroma"
    )
    assert stored["storage"]["readback"] == {
        "verified": True,
        "hypothesis_id": "tinyllm-frozen-scalar-domain-extension-v1",
        "experiment_count": 1,
    }
