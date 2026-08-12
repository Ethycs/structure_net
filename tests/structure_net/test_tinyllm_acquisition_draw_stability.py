from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
from experiments.structure_net import tinyllm_acquisition_draw_stability as study


def _underpowered_config(**changes: object) -> study.AcquisitionDrawStabilityConfig:
    values = {
        "arms": ("analytic_calibrated",),
        "seeds": (7,),
        "replicate_counts": (2, 4),
        "draw_count": 3,
        "required_seed_passes": 1,
        "required_primary_draw_passes": 2,
        "maximum_inter_draw_correlation": 0.999,
        "device": "cpu",
        "allow_underpowered": True,
    }
    values.update(changes)
    return study.AcquisitionDrawStabilityConfig(**values)


def _datasets() -> dict[str, calibrated.CalibratedDataset]:
    fibers = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    branch = torch.tensor([-1, 1, -1, 1], dtype=torch.long)
    paired = SimpleNamespace(
        fiber=SimpleNamespace(fiber_id=fibers, branch=branch)
    )
    calibration = torch.zeros((4, 8), dtype=torch.float32)
    angles = torch.tensor([0.0, 0.6, 1.4, 2.0], dtype=torch.float32)
    calibration[:, 0] = torch.cos(angles)
    calibration[:, 1] = torch.sin(angles)
    dataset = calibrated.CalibratedDataset(
        paired=paired, calibration=calibration
    )
    return {regime: dataset for regime in study.REGIMES}


def test_primary_configuration_is_locked() -> None:
    config = study.AcquisitionDrawStabilityConfig()
    assert config.draw_count == 16
    assert config.replicate_counts == (64, 256)
    assert config.required_primary_draw_passes == 15
    with pytest.raises(ValueError, match="draw count is fixed"):
        study.AcquisitionDrawStabilityConfig(draw_count=15)
    with pytest.raises(ValueError, match="checkpoint gate"):
        study.AcquisitionDrawStabilityConfig(required_seed_passes=3)


def test_preregistration_digest_is_locked() -> None:
    assert study._sha256(study.PREREGISTRATION_PATH) == (
        study.PREREGISTRATION_SHA256
    )


@pytest.mark.parametrize(
    ("valid", "m64", "m256", "classification", "primary"),
    [
        (False, 16, 16, "invalid", False),
        (True, 16, 14, "m256_not_stable", False),
        (True, 15, 15, "m64_stable_population_ceiling", True),
        (True, 12, 15, "m64_broadly_stable_checkpoint_variable", True),
        (True, 4, 15, "m64_draw_sensitive_m256_stable", True),
        (True, 3, 15, "m64_unreliable_m256_stable", True),
    ],
)
def test_locked_classification_table(
    valid: bool,
    m64: int,
    m256: int,
    classification: str,
    primary: bool,
) -> None:
    assert study.classify_campaign(
        valid=valid,
        m64_complete_draw_passes=m64,
        m256_complete_draw_passes=m256,
        config=study.AcquisitionDrawStabilityConfig(),
    ) == (classification, primary)


def test_wilson_interval_contains_observed_frequency() -> None:
    for successes in (0, 4, 12, 16):
        low, high = study.wilson_interval(successes, 16)
        assert 0.0 <= low <= successes / 16 <= high <= 1.0
    with pytest.raises(ValueError, match="invalid binomial"):
        study.wilson_interval(17, 16)


def test_draw_arrays_are_deterministic_independent_and_exactly_reusable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "draws.npz"
    config = _underpowered_config()
    first, first_sha, first_contracts = study._ensure_draw_arrays(
        path, _datasets(), config
    )
    second, second_sha, second_contracts = study._ensure_draw_arrays(
        path, _datasets(), config
    )
    assert first_sha == second_sha
    assert first_contracts == second_contracts
    assert first_contracts["pass"]
    assert first_contracts["all_draws_distinct"]
    assert first_contracts["maximum_pair_shared_error"] == 0.0
    assert first["composition__errors"].shape == (3, 4, 2)
    assert torch.equal(
        first["composition__errors"], second["composition__errors"]
    )
    with pytest.raises(ValueError, match="incompatible acquisition draw arrays"):
        study._ensure_draw_arrays(
            path,
            _datasets(),
            _underpowered_config(draw_seed_root=config.draw_seed_root + 1),
        )


def test_interventions_preserve_unit_norm_and_fiber_pairing(
    tmp_path: Path,
) -> None:
    config = _underpowered_config()
    noise, _, _ = study._ensure_draw_arrays(
        tmp_path / "draws.npz", _datasets(), config
    )
    interventions, audits, shuffled, contracts = study._build_interventions(
        _datasets(), noise, config
    )
    assert contracts["pass"]
    assert contracts["maximum_pair_shared_error"] <= config.pair_tolerance
    for draw_index in range(config.draw_count):
        for count in config.replicate_counts:
            for regime in study.REGIMES:
                packet = interventions[draw_index][count][regime].calibration[:, :2]
                assert torch.allclose(packet.norm(dim=1), torch.ones(4), atol=1e-6)
                base = _datasets()[regime].calibration[:, :2]
                base_angle = torch.atan2(base[:, 1], base[:, 0])
                packet_angle = torch.atan2(packet[:, 1], packet[:, 0])
                error = torch.atan2(
                    torch.sin(packet_angle - base_angle),
                    torch.cos(packet_angle - base_angle),
                )
                assert torch.allclose(error[0], error[1], atol=1e-6)
                assert torch.allclose(error[2], error[3], atol=1e-6)
                assert audits[draw_index][count][regime][
                    "maximum_pair_shared_error"
                ] <= config.pair_tolerance
    for value in shuffled.values():
        assert torch.allclose(
            value.calibration[:, :2].norm(dim=1), torch.ones(4), atol=1e-6
        )


def test_aggregate_uses_draw_as_replication_unit() -> None:
    config = _underpowered_config(
        replicate_counts=(64, 256),
        draw_count=3,
        required_primary_draw_passes=2,
    )
    results = []
    for seed_gate in ((True, True, False),):
        draws = {}
        for index, passed in enumerate(seed_gate):
            draws[study.draw_key(index)] = {
                "m64": {"seed_gate": passed},
                "m256": {"seed_gate": True},
            }
        results.append(
            {
                "condition": "analytic_calibrated",
                "seed": 7,
                "draws": draws,
                "inherited_single_observation": {"seed_gate": False},
                "controls": {
                    "fiber_shuffled_draw0_m256": {"seed_gate": False}
                },
            }
        )
    aggregate = study._aggregate_results(results, config)
    assert aggregate["counts"]["m64"]["complete_draw_passes"] == 2
    assert aggregate["counts"]["m256"]["complete_draw_passes"] == 3
    assert aggregate["arms"]["analytic_calibrated"][
        "checkpoint_frequencies"
    ]["7"]["m64"]["passes"] == 2


def test_campaign_reuse_checks_every_artifact(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text("{}\n", encoding="utf-8")
    arrays_path = tmp_path / "arrays.npz"
    with arrays_path.open("wb") as handle:
        np.savez_compressed(handle, value=np.asarray([1]))
    config = _underpowered_config()
    implementation = "implementation"
    campaign = {
        "status": "completed",
        "schema_version": study.SCHEMA_VERSION,
        "implementation_sha256": implementation,
        "configuration": study._json_config(config),
        "artifacts": {
            "draw_arrays": str(arrays_path),
            "draw_arrays_sha256": study._sha256(arrays_path),
        },
        "results": [
            {
                "path": str(result_path),
                "result_sha256": study._sha256(result_path),
            }
        ],
    }
    assert study._campaign_reusable(campaign, config, implementation)
    result_path.write_text(json.dumps({"changed": True}), encoding="utf-8")
    assert not study._campaign_reusable(campaign, config, implementation)
