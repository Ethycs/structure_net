from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net.tinyllm_carrier_jacobian_axis_audit import (
    CarrierJacobianAxisAuditConfig,
    HYPOTHESIS_ID,
    SCHEMA_VERSION,
    _campaign_decision,
    _campaign_is_reusable,
    _implementation_digest,
    _load_predecessor,
    canonical_alignment_frame,
    canonical_axis_components,
    classify_axis_dominance,
    linearization_metrics,
)


def test_config_freezes_predecessor_geometry_and_primary_seeds() -> None:
    with pytest.raises(ValueError, match="fixed"):
        CarrierJacobianAxisAuditConfig(seeds=(7, 29))
    with pytest.raises(ValueError, match="64 exact orbits"):
        CarrierJacobianAxisAuditConfig(orbit_count=8, allow_underpowered=True)
    with pytest.raises(ValueError, match="rank is fixed"):
        CarrierJacobianAxisAuditConfig(carrier_rank=2)
    with pytest.raises(ValueError, match="positive and ordered"):
        CarrierJacobianAxisAuditConfig(fine_step_std=0.1)


def test_canonical_frame_reproduces_affine_map_and_axis_decomposition() -> None:
    generator = torch.Generator().manual_seed(17)
    source = torch.randn(256, 3, generator=generator, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )
    target = source @ rotation @ torch.diag(
        torch.tensor([0.5, 1.4, 2.0], dtype=torch.float64)
    ) + torch.tensor([0.4, -0.2, 0.7], dtype=torch.float64)
    frame = canonical_alignment_frame(source, target, 1e-8)
    predicted = source @ frame["linear"] + frame["intercept"]
    assert torch.linalg.vector_norm(predicted - target) < 1e-5

    error = torch.randn(32, 3, generator=generator, dtype=torch.float64)
    canonical, components, relative_error = canonical_axis_components(error, frame)
    assert canonical.shape == (32, 3)
    assert components.shape == (32, 3, 3)
    assert relative_error < 1e-12
    assert torch.allclose(components.sum(1), error, atol=1e-12, rtol=1e-12)


def test_linearization_gate_uses_zero_referenced_prediction() -> None:
    config = CarrierJacobianAxisAuditConfig()
    fine = torch.tensor(
        [[1.0, 0.5, -0.2], [0.7, -0.3, 0.4]], dtype=torch.float64
    )
    coarse = fine * 1.01
    observed = torch.tensor([0.2, -0.3], dtype=torch.float64)
    metrics = linearization_metrics(fine, coarse, observed, observed, config)
    assert metrics["adequate"]
    assert metrics["signed_error_zero_referenced_r2"] == pytest.approx(1.0)
    assert metrics["prediction_residual_mae_fraction"] == pytest.approx(0.0)
    failed = linearization_metrics(fine, coarse, -observed, observed, config)
    assert not failed["adequate"]
    assert not failed["gates"]["signed_error_zero_r2"]


def test_axis_dominance_rules_are_exact_reciprocals() -> None:
    config = CarrierJacobianAxisAuditConfig()
    paired = torch.ones(16)
    third = classify_axis_dominance(
        True,
        torch.full((16,), 0.1),
        torch.ones(16),
        paired,
        torch.full((16,), 0.1),
        torch.ones(16),
        config,
    )
    assert third["classification"] == "axis_3_dominant"
    shared = classify_axis_dominance(
        True,
        torch.ones(16),
        torch.full((16,), 0.1),
        paired,
        torch.ones(16),
        torch.full((16,), 0.1),
        config,
    )
    assert shared["classification"] == "axes_1_2_dominant"
    unresolved = classify_axis_dominance(
        False, paired, paired, paired, paired, paired, config
    )
    assert unresolved["classification"] == "nonlinear_or_unresolved"


def test_campaign_decision_requires_all_six_adequate_and_five_dominant() -> None:
    supported = _campaign_decision(["axis_3_dominant"] * 5 + ["mixed"], 6)
    assert supported["universal_2d_base_plus_local_scalar_supported"]
    assert not supported["shared_axes_causally_misoriented_supported"]
    not_supported = _campaign_decision(["axis_3_dominant"] * 6, 5)
    assert not not_supported["universal_2d_base_plus_local_scalar_supported"]
    assert not_supported["conclusion"] == "unresolved_local_linearization_inadequate"


def test_authoritative_transport_predecessor_has_six_immutable_pairs() -> None:
    campaign, pairs = _load_predecessor(CarrierJacobianAxisAuditConfig())
    assert campaign["aggregates"]["gate_counts"]["paired_causal_transport"] == 0
    assert len(pairs) == 6
    assert set(pairs) == {
        (7, 29),
        (7, 53),
        (29, 7),
        (29, 53),
        (53, 7),
        (53, 29),
    }


def test_completed_campaign_validation_checks_pair_hashes(tmp_path: Path) -> None:
    config = CarrierJacobianAxisAuditConfig(
        seeds=(7, 29), allow_underpowered=True, device="cpu"
    )
    implementation = _implementation_digest()
    results = []
    for source, target in ((7, 29), (29, 7)):
        detail = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "status": "completed",
            "evidence_role": "systems_lifecycle_only_not_quality_evidence",
            "implementation_sha256": implementation,
            "scientific_fingerprint": f"fingerprint-{source}-{target}",
        }
        path = tmp_path / f"{source}-{target}.json"
        path.write_text(json.dumps(detail))
        results.append(
            {
                "source_seed": source,
                "target_seed": target,
                "scientific_fingerprint": detail["scientific_fingerprint"],
                "path": str(path),
                "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": "systems_lifecycle_only_not_quality_evidence",
        "implementation_sha256": implementation,
        "configuration": json.loads(json.dumps(asdict(config))),
        "summary": {"completed": 2},
        "results": results,
    }
    assert _campaign_is_reusable(campaign, config, implementation)
    Path(results[0]["path"]).write_text("{}")
    assert not _campaign_is_reusable(campaign, config, implementation)
