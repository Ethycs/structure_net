from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net import tinyllm_fixed_semantic_gauge_writer as writer


def test_primary_configuration_and_evidence_role_are_fixed() -> None:
    config = writer.FixedSemanticGaugeWriterConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    assert config.carrier_rank == 3
    with pytest.raises(ValueError, match="fixed"):
        writer.FixedSemanticGaugeWriterConfig(seeds=(7,), device="cpu")
    pilot = writer.FixedSemanticGaugeWriterConfig(
        seeds=(7,), orbit_count=8, device="cpu", allow_underpowered=True
    )
    assert writer._evidence_role(pilot) == (
        "systems_lifecycle_only_not_quality_evidence"
    )


def test_neutral_c2_carrier_is_exactly_swap_invariant() -> None:
    vector = torch.tensor(
        [[1.0, 0.0], [0.6, 0.8], [-0.3, 0.4]], dtype=torch.float64
    )
    fused = writer.neutral_c2_carrier(vector)
    swapped = writer.neutral_c2_carrier(-vector)
    assert torch.equal(fused, swapped)
    assert torch.allclose(fused[:, 2], vector.square().sum(-1))
    angle = torch.atan2(vector[:, 1], vector[:, 0])
    assert torch.allclose(fused[:, 0], torch.cos(2.0 * angle) * fused[:, 2])
    assert torch.allclose(fused[:, 1], torch.sin(2.0 * angle) * fused[:, 2])


def test_no_intercept_writer_recovers_exact_three_channel_map() -> None:
    generator = torch.Generator().manual_seed(23)
    carrier = torch.randn((128, 3), generator=generator, dtype=torch.float64)
    linear = torch.tensor(
        [[0.7, -0.2, 0.3], [0.1, 1.2, -0.4], [0.5, 0.3, 0.9]],
        dtype=torch.float64,
    )
    coordinates = carrier @ linear
    fitted = writer.fit_linear_writer(carrier, coordinates, 1e-10)
    predicted = carrier @ fitted["linear"] + fitted["intercept"]
    assert torch.allclose(predicted, coordinates, atol=1e-9, rtol=1e-9)
    assert torch.equal(fitted["intercept"], torch.zeros(3, dtype=torch.float64))


def test_writer_shuffle_preserves_regime_blocks_and_is_deterministic() -> None:
    first = writer.regime_preserving_writer_permutation(64, 29)
    second = writer.regime_preserving_writer_permutation(64, 29)
    assert torch.equal(first, second)
    assert set(first[:64].tolist()) == set(range(64))
    assert set(first[64:].tolist()) == set(range(64, 128))
    assert not torch.equal(first, torch.arange(128))


def test_checkpoint_gate_requires_observation_causality_controls_and_specificity() -> None:
    config = writer.FixedSemanticGaugeWriterConfig(device="cpu")

    def state(passed: bool, shift: float) -> dict:
        return {
            "continuous": {
                "continuous_pass": passed,
                "mean_moment_shift_bins": shift,
            }
        }

    cell = {
        "decomposition_relative_error": 0.0,
        "coordinate_metrics": {"fixed_gauge": {"variance_explained": 0.5}},
        "states": {
            "zero": state(False, 4.0),
            "exact": state(True, 0.0),
            "direct_rank3": state(True, 0.02),
            "fixed_gauge": state(True, 0.10),
            "fixed_gauge_shuffled": state(False, 2.0),
        },
    }
    observation = {
        "circular_alignment": 0.999,
        "mean_shift_bins": 0.01,
        "p95_shift_bins": 0.02,
    }
    gates = writer.checkpoint_gates(
        [cell for _ in range(4)], [observation for _ in range(4)], config
    )
    assert gates["observation_contract"]
    assert gates["continuous_target_control_contract"]
    assert gates["fixed_gauge_causal_writer"]
    assert gates["shuffled_specificity"]
    failed = dict(observation)
    failed["circular_alignment"] = 0.98
    assert not writer.observation_contract([failed], config)


def test_authoritative_predecessor_is_hash_locked() -> None:
    campaign, path, identities = writer._load_predecessor(
        writer.FixedSemanticGaugeWriterConfig(device="cpu")
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        writer.PREDECESSOR_CAMPAIGN_SHA256
    )
    assert campaign["implementation_sha256"] == (
        writer.PREDECESSOR_IMPLEMENTATION_SHA256
    )
    assert set(identities) == {7, 29, 53}


def test_completed_campaign_validation_checks_result_hashes(tmp_path: Path) -> None:
    config = writer.FixedSemanticGaugeWriterConfig(
        seeds=(7,), orbit_count=8, device="cpu", allow_underpowered=True
    )
    implementation = writer._implementation_digest()
    detail = tmp_path / "result.json"
    detail.write_text("{}")
    campaign = {
        "schema_version": writer.SCHEMA_VERSION,
        "hypothesis_id": writer.HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": "systems_lifecycle_only_not_quality_evidence",
        "implementation_sha256": implementation,
        "configuration": json.loads(json.dumps(asdict(config))),
        "summary": {"completed": 1},
        "results": [
            {
                "path": str(detail),
                "result_sha256": hashlib.sha256(detail.read_bytes()).hexdigest(),
            }
        ],
    }
    assert writer._campaign_is_reusable(campaign, config, implementation)
    detail.write_text('{"changed": true}')
    assert not writer._campaign_is_reusable(campaign, config, implementation)
