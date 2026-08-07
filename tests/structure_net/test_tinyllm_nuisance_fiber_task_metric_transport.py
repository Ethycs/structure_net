from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import (
    tinyllm_nuisance_fiber_task_metric_transport as audit,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


def test_primary_configuration_and_fresh_seeds_are_locked() -> None:
    config = audit.NuisanceFiberMetricTransportConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    assert config.orbit_count == 64
    assert config.random_controls == 8
    assert config.fourier_order == 4
    assert audit.FRESH_COHORT_SEEDS["fresh_1"]["composition"] == {
        "quotient": 910101,
        "source": 910111,
        "target": 910121,
    }
    with pytest.raises(ValueError, match="fixed"):
        audit.NuisanceFiberMetricTransportConfig(seeds=(7,), device="cpu")
    with pytest.raises(ValueError, match="eight"):
        audit.NuisanceFiberMetricTransportConfig(random_controls=4, device="cpu")
    pilot = audit.NuisanceFiberMetricTransportConfig(
        seeds=(7,),
        orbit_count=8,
        random_controls=2,
        device="cpu",
        allow_underpowered=True,
    )
    assert audit._evidence_role(pilot) == (
        "systems_lifecycle_only_not_quality_evidence"
    )


def test_locked_calibration_action_preserves_analytic_phase_carrier() -> None:
    task = CircleTaskConfig()
    batch = 9
    history = torch.arange(task.sensor_steps, dtype=torch.float32)
    history = history / max(1.0, float(task.sensor_steps - 1)) - 1.0
    phase = torch.linspace(-2.0, 2.0, batch)
    orientation_angle = torch.linspace(-0.4, 0.4, batch)
    orientation = torch.stack(
        (orientation_angle.cos(), orientation_angle.sin()), dim=1
    )
    amplitude = torch.linspace(0.7, 1.5, batch)[:, None]
    offset = torch.stack((phase * 0.07, phase * -0.03), dim=1)
    drift = torch.stack((phase * 0.01, phase * 0.02), dim=1)
    speed = torch.where(
        torch.arange(batch) % 2 == 0,
        torch.full((batch,), 0.35),
        torch.full((batch,), -0.35),
    )
    angle = phase[:, None] + speed[:, None] * history[None, :]
    lab = torch.stack((angle.cos(), angle.sin()), dim=-1)
    rotation = torch.stack(
        (
            torch.stack((orientation[:, 0], -orientation[:, 1]), dim=1),
            torch.stack((orientation[:, 1], orientation[:, 0]), dim=1),
        ),
        dim=1,
    )
    sensor = amplitude[:, None, :] * torch.einsum("bij,btj->bti", rotation, lab)
    sensor += offset[:, None, :] + drift[:, None, :] * history[None, :, None]
    sensor = torch.cat((sensor, torch.randn(batch, task.sensor_steps, 1)), dim=-1)
    calibration = torch.cat((orientation, speed[:, None], amplitude, offset, drift), dim=1)
    dataset = SimpleNamespace(sensor=sensor, calibration=calibration)
    # replace() needs a dataclass in production; use the actual OrbitDataset shell here.
    shell = audit.local.transport.rank.deck.OrbitDataset(
        input_ids=torch.zeros(batch, task.sequence_length, dtype=torch.long),
        sensor=sensor,
        calibration=calibration,
        phase=torch.zeros(batch),
        quotient_phase=torch.zeros(batch),
        branch=torch.zeros(batch, dtype=torch.long),
        target_posteriors=torch.zeros(batch, task.phase_bins),
        target_bins=torch.zeros(batch, dtype=torch.long),
        orbit_count=batch,
        k=1,
    )
    transformed = audit.apply_calibration_action(shell, task)
    carrier = audit.local.fixed.ladder.AnalyticPhaseCarrier(task)
    assert torch.allclose(
        carrier(dataset.sensor, dataset.calibration),
        carrier(transformed.sensor, transformed.calibration),
        atol=2e-6,
        rtol=2e-6,
    )


def test_kernel_line_cosines_are_sign_invariant() -> None:
    source = torch.tensor(
        [[[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]], dtype=torch.float64
    )
    target = torch.tensor(
        [[[-2.0, 0.0, 0.0], [0.0, -4.0, 0.0]]], dtype=torch.float64
    )
    assert audit.kernel_line_cosines(source, target).item() == pytest.approx(1.0)


def test_random_projector_controls_are_deterministic_and_norm_matched() -> None:
    residual = torch.tensor(
        [[1.0, -2.0, 3.0], [0.5, 0.25, -0.75]], dtype=torch.float64
    )
    reference = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]], dtype=torch.float64
    )
    first = audit.random_projector_tangents(residual, reference, 8, 123)
    second = audit.random_projector_tangents(residual, reference, 8, 123)
    assert torch.equal(first, second)
    assert torch.allclose(
        torch.linalg.vector_norm(first, dim=-1),
        torch.linalg.vector_norm(reference, dim=-1)[None, :],
        atol=1e-12,
    )


def _state(mean: float, passed: bool, movement_mean: float = 0.0, movement_p95: float = 0.0):
    return {
        "continuous": {
            "continuous_pass": passed,
            "mean_moment_shift_bins": mean,
        },
        "movement_from_predicted": {
            "mean_bins": movement_mean,
            "p95_bins": movement_p95,
        },
    }


def _passing_cell(random_controls: int) -> dict:
    states = {
        "zero": _state(2.0, False),
        "predicted": _state(1.0, False),
        "exact": _state(0.0, True),
        "full": _state(0.0, True),
        "local_tangent": _state(0.02, True),
        "transported_tangent": _state(0.03, True),
        "local_kernel": _state(1.0, False, 0.01, 0.02),
        "shuffled_tangent": _state(0.30, False),
    }
    for index in range(random_controls):
        states[f"random_tangent_{index:02d}"] = _state(0.40, False)
    return {
        "numerical": {
            "source_rank_two": True,
            "target_rank_two": True,
            "jacobians_finite": True,
            "maximum_decomposition_error": 1e-12,
        },
        "geometry": {
            "median_kernel_line_absolute_cosine": 0.95,
            "p10_kernel_line_absolute_cosine": 0.85,
        },
        "states": states,
    }


def test_checkpoint_gates_require_geometry_causality_and_specificity() -> None:
    config = audit.NuisanceFiberMetricTransportConfig(device="cpu")
    cells = [_passing_cell(config.random_controls) for _ in range(4)]
    gates = audit.checkpoint_gates(cells, True, config)
    assert all(
        gates[name]
        for name in (
            "exact_calibration_action_contract",
            "numerical_target_control_contract",
            "geometric_transport",
            "causal_transport",
            "specificity",
        )
    )
    assert audit.classify_checkpoint(gates) == (
        "nuisance_fiber_task_metric_transport_confirmed"
    )
    cells[0]["geometry"]["p10_kernel_line_absolute_cosine"] = 0.5
    gates = audit.checkpoint_gates(cells, True, config)
    assert not gates["geometric_transport"]
    assert audit.classify_checkpoint(gates) == (
        "causal_equivalence_without_projector_equality"
    )


def test_predecessor_campaigns_are_frozen_by_hash() -> None:
    config = audit.NuisanceFiberMetricTransportConfig(device="cpu")
    writer = Path(config.writer_root) / "campaign_results.json"
    method = Path(config.local_method_root) / "campaign_results.json"
    assert audit._sha256(writer) == audit.WRITER_CAMPAIGN_SHA256
    assert audit._sha256(method) == audit.LOCAL_METHOD_CAMPAIGN_SHA256


def test_completed_campaign_validation_checks_artifact_hashes(tmp_path: Path) -> None:
    config = audit.NuisanceFiberMetricTransportConfig(
        seeds=(7,),
        orbit_count=8,
        random_controls=2,
        device="cpu",
        allow_underpowered=True,
    )
    implementation = "digest"
    result = tmp_path / "result.json"
    arrays = tmp_path / "arrays.npz"
    result.write_text("{}\n")
    arrays.write_bytes(b"npz")
    campaign = {
        "schema_version": audit.SCHEMA_VERSION,
        "hypothesis_id": audit.HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": audit._evidence_role(config),
        "implementation_sha256": implementation,
        "configuration": audit._json_compatible(config.__dict__),
        "summary": {"completed": 1},
        "results": [
            {
                "path": str(result),
                "result_sha256": audit._sha256(result),
                "arrays": str(arrays),
                "arrays_sha256": audit._sha256(arrays),
            }
        ],
    }
    assert audit._campaign_is_reusable(campaign, config, implementation)
    arrays.write_bytes(b"changed")
    assert not audit._campaign_is_reusable(campaign, config, implementation)
