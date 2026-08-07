from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from experiments.structure_net import tinyllm_scalar_groupoid_defect as defect


def _config() -> defect.ScalarGroupoidDefectConfig:
    return defect.ScalarGroupoidDefectConfig(
        seeds=(7,), device="cpu", allow_underpowered=True
    )


def test_primary_configuration_and_gates_are_locked() -> None:
    config = defect.ScalarGroupoidDefectConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    assert config.orbit_count == 64
    assert config.groupoid_max_error_ceiling_bins == pytest.approx(1e-6)
    assert config.direct_to_total_rms_ceiling == pytest.approx(0.10)
    assert config.direct_p95_ceiling_bins == pytest.approx(0.05)
    assert config.prediction_r2_floor == pytest.approx(0.90)
    assert config.prediction_relative_l2_ceiling == pytest.approx(math.sqrt(0.10))
    with pytest.raises(ValueError, match="64 orbits"):
        replace(config, orbit_count=32, allow_underpowered=True)
    with pytest.raises(ValueError, match="fixed to 7,29,53"):
        replace(config, seeds=(7,))


def test_wrap_bin_values_uses_task_period() -> None:
    value = torch.tensor([-17.0, -8.0, 0.0, 8.0, 17.0], dtype=torch.float64)
    actual = defect.wrap_bin_values(value, bins=16)
    expected = torch.tensor([-1.0, -8.0, 0.0, 8.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=0)


def test_exact_groupoid_identity_holds_across_wrap_boundaries() -> None:
    reference_direct = torch.tensor([3.05, -3.10, 0.2], dtype=torch.float64)
    transformed_direct = torch.tensor([-3.04, 3.00, -0.3], dtype=torch.float64)
    reference_writer = torch.tensor([-3.00, 2.90, -0.5], dtype=torch.float64)
    transformed_writer = torch.tensor([3.02, -3.05, 0.7], dtype=torch.float64)
    values = defect.groupoid_components(
        reference_direct=reference_direct,
        transformed_direct=transformed_direct,
        reference_writer=reference_writer,
        transformed_writer=transformed_writer,
        bins=16,
    )
    assert float(values["reconstruction_error"].abs().max()) < 1e-12


def test_writer_only_metrics_pass_when_direct_term_is_zero_and_specific() -> None:
    target = torch.tensor(
        [0.10, 0.24, -0.14, -0.28, 0.35, -0.18, 0.16, -0.32],
        dtype=torch.float64,
    )
    components = {
        "delta_scalar": target,
        "delta_direct": torch.zeros_like(target),
        "delta_writer": -target,
        "reconstruction_error": torch.zeros_like(target),
    }
    shuffled = defect.law.group.phase_matched_shift_indices(2, 4)
    metrics = defect.groupoid_cell_metrics(components, shuffled, _config())
    assert metrics["gates"]["action_defect_nondegenerate"] is True
    assert metrics["gates"]["exact_groupoid_identity"] is True
    assert metrics["gates"]["direct_term_negligible"] is True
    assert metrics["gates"]["writer_only_prediction"] is True
    assert metrics["gates"]["writer_only_specificity"] is True
    assert metrics["gates"]["primary_cell_pass"] is True


def test_material_direct_term_selects_two_term_classification() -> None:
    target = torch.tensor([0.2, -0.3, 0.4, -0.5], dtype=torch.float64)
    components = {
        "delta_scalar": target,
        "delta_direct": 0.5 * target,
        "delta_writer": -0.5 * target,
        "reconstruction_error": torch.zeros_like(target),
    }
    metrics = defect.groupoid_cell_metrics(
        components, torch.tensor([1, 0, 3, 2]), _config()
    )
    assert metrics["gates"]["exact_groupoid_identity"] is True
    assert metrics["gates"]["direct_term_negligible"] is False
    assert defect.classify_checkpoint(
        valid=True,
        direct_negligible_all=False,
        writer_prediction_all=False,
        specificity_all=False,
    ) == "two_term_groupoid_defect"


@pytest.mark.parametrize(
    ("valid", "direct", "writer", "specific", "expected"),
    [
        (False, True, True, True, "invalid"),
        (True, True, True, True, "writer_symmetry_defect_dominant"),
        (True, True, True, False, "writer_defect_descriptive_not_specific"),
        (True, False, True, True, "two_term_groupoid_defect"),
        (True, True, False, False, "writer_only_reduction_failed"),
    ],
)
def test_classification_precedence(
    valid: bool,
    direct: bool,
    writer: bool,
    specific: bool,
    expected: str,
) -> None:
    assert defect.classify_checkpoint(
        valid=valid,
        direct_negligible_all=direct,
        writer_prediction_all=writer,
        specificity_all=specific,
    ) == expected


def test_locked_transformation_campaign_is_authoritative() -> None:
    campaign, path, details = defect._load_transformation_campaign(_config())
    assert defect._sha256(path) == defect.TRANSFORMATION_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == (
        defect.TRANSFORMATION_IMPLEMENTATION_SHA256
    )
    assert set(details) == {7}


def test_fingerprint_changes_with_source_identity() -> None:
    config = _config()
    first = defect._fingerprint(config, 7, "source", "checkpoint")
    assert first == defect._fingerprint(config, 7, "source", "checkpoint")
    assert first != defect._fingerprint(config, 7, "different", "checkpoint")


def test_completed_seed_resume_verifies_array_hash(tmp_path: Path) -> None:
    config = _config()
    result_path = tmp_path / "runs" / "seed_7" / "result.json"
    arrays_path = result_path.with_name("groupoid_arrays.npz")
    result_path.parent.mkdir(parents=True)
    arrays_path.write_bytes(b"immutable-arrays")
    fingerprint = defect._fingerprint(config, 7, "source", "checkpoint")
    implementation = "implementation"
    record = {
        "schema_version": defect.SCHEMA_VERSION,
        "hypothesis_id": defect.HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": defect._evidence_role(config),
        "implementation_sha256": implementation,
        "configuration": json.loads(json.dumps(asdict(config))),
        "seed": 7,
        "experiment_id": "seed-seven",
        "scientific_fingerprint": fingerprint,
        "artifacts": {
            "result": str(result_path),
            "arrays": str(arrays_path),
            "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
        },
    }
    result_path.write_text(json.dumps(record), encoding="utf-8")
    assert defect._reusable_seed_result(
        result_path,
        arrays_path,
        config,
        implementation,
        7,
        fingerprint,
    ) == record
    arrays_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="incompatible completed result"):
        defect._reusable_seed_result(
            result_path,
            arrays_path,
            config,
            implementation,
            7,
            fingerprint,
        )
