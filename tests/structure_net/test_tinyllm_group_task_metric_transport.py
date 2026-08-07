from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net.tinyllm_group_task_metric_transport import (
    GroupTaskMetricTransportConfig,
    HYPOTHESIS_ID,
    SCHEMA_VERSION,
    _campaign_is_reusable,
    _implementation_digest,
    _load_predecessor_campaign,
    fisher_metric_from_logits_jacobian,
    fit_task_metric_affine,
    metric_stability,
    pair_gates,
)
from experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport import (
    apply_affine,
)


def test_config_freezes_primary_design() -> None:
    with pytest.raises(ValueError, match="fixed"):
        GroupTaskMetricTransportConfig(seeds=(7, 29))
    with pytest.raises(ValueError, match="fixed"):
        GroupTaskMetricTransportConfig(carrier_rank=2)
    with pytest.raises(ValueError, match="step fraction"):
        GroupTaskMetricTransportConfig(metric_step_fraction=0.2)
    with pytest.raises(ValueError, match="dominance"):
        GroupTaskMetricTransportConfig(baseline_dominance_ratio=1.0)


def test_pullback_fisher_metric_is_psd_and_trace_normalized() -> None:
    config = GroupTaskMetricTransportConfig()
    generator = torch.Generator().manual_seed(13)
    logits = torch.randn(11, 16, generator=generator, dtype=torch.float64)
    jacobian = torch.randn(11, 16, 3, generator=generator, dtype=torch.float64)
    metrics, traces, raw_eigenvalues = fisher_metric_from_logits_jacobian(
        logits, jacobian, config
    )
    assert metrics.shape == (11, 3, 3)
    assert torch.all(traces > 0)
    assert float(raw_eigenvalues.min()) >= -1e-10
    expected_trace = 1.0 + config.metric_isotropic_floor
    assert torch.allclose(
        torch.diagonal(metrics, dim1=-2, dim2=-1).sum(-1),
        torch.full((11,), expected_trace, dtype=torch.float64),
        atol=1e-10,
        rtol=1e-10,
    )


def test_task_metric_affine_recovers_exact_affine_change() -> None:
    config = GroupTaskMetricTransportConfig()
    generator = torch.Generator().manual_seed(17)
    source = torch.randn(128, 3, generator=generator, dtype=torch.float64)
    linear = torch.tensor(
        [[0.8, -0.2, 0.4], [0.1, 1.2, -0.3], [-0.5, 0.2, 0.9]],
        dtype=torch.float64,
    )
    intercept = torch.tensor([0.4, -0.6, 0.2], dtype=torch.float64)
    target = source @ linear + intercept
    factors = torch.randn(128, 3, 3, generator=generator, dtype=torch.float64)
    metrics = factors.transpose(-1, -2) @ factors + 0.01 * torch.eye(3)
    mapping = fit_task_metric_affine(source, target, metrics, config.metric_ridge)
    predicted = apply_affine(source, mapping)
    assert torch.allclose(predicted, target, atol=1e-6, rtol=1e-6)


def test_metric_stability_is_zero_for_identical_metrics() -> None:
    metrics = torch.eye(3, dtype=torch.float64).repeat(9, 1, 1)
    summary = metric_stability(metrics, metrics.clone())
    assert summary == {
        "median_relative_error": 0.0,
        "maximum_relative_error": 0.0,
        "mean_relative_error": 0.0,
    }


def test_pair_gates_require_continuous_rescue_dominance_and_specificity() -> None:
    config = GroupTaskMetricTransportConfig()

    def state(passed: bool, shift: float) -> dict:
        return {
            "continuous": {
                "continuous_pass": passed,
                "mean_moment_shift_bins": shift,
            }
        }

    def cell() -> dict:
        return {
            "decomposition_relative_error": 0.0,
            "coordinate_metrics": {
                "task_metric": {"variance_explained": 0.9},
                "task_metric_shuffled": {"variance_explained": -0.2},
            },
            "states": {
                "zero": state(False, 4.0),
                "exact": state(True, 0.0),
                "direct_rank3": state(True, 0.03),
                "paired": state(False, 0.24),
                "affine_ridge": state(False, 0.20),
                "task_metric": state(True, 0.10),
                "task_metric_shuffled": state(False, 2.0),
            },
        }

    gates = pair_gates([cell() for _ in range(4)], True, 0.0, config)
    required = (
        "metric_contract",
        "continuous_target_control_contract",
        "task_metric_coordinate_transport",
        "task_metric_causal_transport",
        "task_metric_dominates_euclidean_baselines",
        "task_metric_shuffled_specificity",
        "predecessor_replay_contract",
    )
    assert all(gates[name] for name in required)
    failed = [cell() for _ in range(4)]
    failed[0]["states"]["task_metric"]["continuous"]["continuous_pass"] = False
    assert not pair_gates(failed, True, 0.0, config)[
        "task_metric_causal_transport"
    ]


def test_predecessor_campaign_is_frozen_by_hash() -> None:
    campaign, path = _load_predecessor_campaign(GroupTaskMetricTransportConfig())
    assert campaign["aggregates"]["confirmed"] is False
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "44707fd4bcd810e63614671aa491095fae735ee52359d464ab25abb10a2bc228"
    )


def test_completed_campaign_validation_checks_result_hashes(tmp_path: Path) -> None:
    config = GroupTaskMetricTransportConfig(
        seeds=(7, 29), orbit_count=8, allow_underpowered=True, device="cpu"
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
