from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport import (
    CrossSeedCarrierTransportConfig,
    HYPOTHESIS_ID,
    SCHEMA_VERSION,
    _campaign_is_reusable,
    _implementation_digest,
    _load_readout_predecessor,
    _pair_gates,
    apply_affine,
    coordinate_metrics,
    fit_whitened_orthogonal,
    regime_preserving_permutation,
)


def test_config_freezes_primary_seeds_rank_and_thresholds() -> None:
    with pytest.raises(ValueError, match="fixed"):
        CrossSeedCarrierTransportConfig(seeds=(7, 29))
    with pytest.raises(ValueError, match="rank is fixed"):
        CrossSeedCarrierTransportConfig(carrier_rank=2)
    with pytest.raises(ValueError, match="R2 floor"):
        CrossSeedCarrierTransportConfig(coordinate_r2_floor=1.0)


def test_whitened_orthogonal_recovers_affine_coordinate_change() -> None:
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(256, 3, generator=generator, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )
    scales = torch.diag(torch.tensor([0.5, 1.5, 2.0], dtype=torch.float64))
    target = source @ rotation @ scales + torch.tensor(
        [0.4, -0.2, 0.7], dtype=torch.float64
    )
    mapping = fit_whitened_orthogonal(source, target, 1e-8)
    metrics = coordinate_metrics(apply_affine(source, mapping), target)
    assert metrics["variance_explained"] > 0.999999
    assert metrics["normalized_rmse"] < 1e-3


def test_regime_preserving_permutation_never_crosses_regimes() -> None:
    permutation = regime_preserving_permutation(64, 7, 29)
    assert sorted(permutation[:64].tolist()) == list(range(64))
    assert sorted(permutation[64:].tolist()) == list(range(64, 128))
    assert torch.equal(permutation, regime_preserving_permutation(64, 7, 29))


def test_pair_gates_require_coordinate_causal_control_and_specificity() -> None:
    config = CrossSeedCarrierTransportConfig()

    def cell(shuffled_r2: float = 0.1) -> dict:
        passed = {"joint_pass": True}
        return {
            "decomposition_relative_error": 0.0,
            "coordinate_metrics": {
                "paired": {"variance_explained": 0.9},
                "shuffled": {"variance_explained": shuffled_r2},
            },
            "states": {
                "zero": {"joint_pass": False},
                "exact": passed,
                "direct_rank3": passed,
                "paired": passed,
                "shuffled": {"joint_pass": shuffled_r2 >= 0.8},
            },
        }

    passed = _pair_gates([cell() for _ in range(4)], config)
    assert all(
        passed[name]
        for name in (
            "target_control_contract",
            "paired_coordinate_transport",
            "paired_causal_transport",
            "shuffled_specificity",
        )
    )
    assert not _pair_gates([cell(0.85) for _ in range(4)], config)[
        "shuffled_specificity"
    ]


def test_readout_predecessor_is_authoritative_corrective_result() -> None:
    config = CrossSeedCarrierTransportConfig()
    checkpoint = (
        Path(config.source_root) / "runs" / "k2" / "seed_7" / "model.pt"
    )
    import hashlib

    predecessor = _load_readout_predecessor(
        config, 7, hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert predecessor["campaign"].endswith(
        "20260806_d6_corrective_v3_6a88480c/campaign_results.json"
    )
    assert predecessor["rotation_bins"] == pytest.approx(0.157958984375)


def test_completed_campaign_validation_checks_result_hashes(tmp_path: Path) -> None:
    config = CrossSeedCarrierTransportConfig(
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
        import hashlib

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
