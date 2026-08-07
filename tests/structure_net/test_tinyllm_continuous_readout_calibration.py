from __future__ import annotations

import math
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

import experiments.structure_net.tinyllm_defect_subspace_rank as defect_rank
from experiments.structure_net.tinyllm_continuous_readout_calibration import (
    ContinuousReadoutCalibrationConfig,
    HYPOTHESIS_ID,
    SCHEMA_VERSION,
    _fingerprint,
    _implementation_digest,
    _load_predecessors,
    _rank_config,
    _sha256,
    continuous_metrics,
    fit_boundary_rotation,
    posterior_moment,
    quantize_moment_angles,
    run_campaign,
    select_source_rank,
)
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


def test_config_freezes_rank_grid_and_odd_calibration_grid() -> None:
    with pytest.raises(ValueError, match="fixed"):
        ContinuousReadoutCalibrationConfig(seeds=(7,))
    with pytest.raises(ValueError, match="start at one"):
        ContinuousReadoutCalibrationConfig(ranks=(2, 3, 4, 5))
    with pytest.raises(ValueError, match="odd"):
        ContinuousReadoutCalibrationConfig(calibration_grid_points=100)
    with pytest.raises(ValueError, match="exclusive"):
        ContinuousReadoutCalibrationConfig(
            allow_underpowered=True,
            post_outcome_corrective_replication=True,
        )
    assert ContinuousReadoutCalibrationConfig.boundary_root.endswith(
        "20260806_d6_preregistered_v3"
    )


def test_posterior_moment_and_quantization_follow_bin_centers() -> None:
    posterior = torch.eye(8, dtype=torch.float64)
    angle, magnitude = posterior_moment(posterior)
    assert torch.allclose(magnitude, torch.ones(8, dtype=torch.float64))
    assert quantize_moment_angles(angle, 8, 0.0).tolist() == list(range(8))


def test_boundary_rotation_recovers_fixed_offset() -> None:
    bins = 16
    targets = torch.arange(bins).repeat(4)
    width = 2.0 * math.pi / bins
    angles = (targets.double() - 0.25) * width
    fitted = fit_boundary_rotation(angles, targets, bins, 4097)
    assert fitted["source_accuracy"] == 1.0
    assert fitted["rotation_bins"] == 0.0
    predicted = quantize_moment_angles(angles, bins, fitted["rotation_bins"])
    assert torch.equal(predicted, targets)


def test_continuous_endpoint_detects_small_and_large_shifts() -> None:
    config = ContinuousReadoutCalibrationConfig()
    exact = torch.eye(16, dtype=torch.float64)
    near = 0.99 * exact + 0.01 * exact.roll(1, dims=1)
    diagnostics = {"circular_alignment": 0.995, "sampling_resolved": True, "winding_degree": 2.0}
    exact_diagnostics = {"circular_alignment": 0.996}
    assert continuous_metrics(near, exact, diagnostics, exact_diagnostics, config)["continuous_pass"]
    shifted = exact.roll(1, dims=1)
    assert not continuous_metrics(shifted, exact, diagnostics, exact_diagnostics, config)["continuous_pass"]


def test_source_rank_selection_is_minimal_and_conjunctive() -> None:
    def cell(rank_two_pass: bool) -> dict:
        return {
            "ranks": {
                "1": {"continuous": {"continuous_pass": False}},
                "2": {"continuous": {"continuous_pass": rank_two_pass}},
                "3": {"continuous": {"continuous_pass": True}},
            }
        }
    assert select_source_rank([cell(True), cell(True)], (1, 2, 3)) == 2
    assert select_source_rank([cell(True), cell(False)], (1, 2, 3)) == 3


def test_completed_aggregate_resume_is_byte_immutable(tmp_path: Path) -> None:
    config = ContinuousReadoutCalibrationConfig(device="cuda")
    task = CircleTaskConfig()
    implementation = _implementation_digest()
    evidence_role = "preregistered_underpowered_mechanistic_evidence"
    runs = []
    for seed in config.seeds:
        checkpoint = (
            Path(config.source_root) / "runs" / "k2" / f"seed_{seed}" / "model.pt"
        )
        checkpoint_sha256 = _sha256(checkpoint)
        _, character_path = defect_rank._load_character_source(
            _rank_config(config),
            seed,
            checkpoint_sha256,
        )
        predecessors = _load_predecessors(config, seed, checkpoint_sha256)
        fingerprint = _fingerprint(
            config,
            task,
            seed,
            checkpoint_sha256,
            _sha256(character_path),
            predecessors,
        )
        path = tmp_path / "runs" / f"seed_{seed}" / "result.json"
        run = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-continuous-readout-calibration-seed{seed}",
            "status": "completed",
            "evidence_role": evidence_role,
            "seed": seed,
            "scientific_fingerprint": fingerprint,
            "implementation_sha256": implementation,
            "provenance": {
                "character_result_sha256": _sha256(character_path),
            },
            "artifacts": {"result": str(path)},
            "analysis_seconds": 0.0,
            "gates": {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run, sort_keys=True))
        runs.append(run)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "configuration": json.loads(json.dumps(asdict(config))),
        "implementation_sha256": implementation,
        "results": [
            {
                "experiment_id": run["experiment_id"],
                "scientific_fingerprint": run["scientific_fingerprint"],
                "result_sha256": _sha256(Path(run["artifacts"]["result"])),
            }
            for run in runs
        ],
        "sentinel": "must remain byte-identical",
    }
    campaign_path = tmp_path / "campaign_results.json"
    campaign_path.write_text(json.dumps(campaign, indent=2) + "\n")
    before = campaign_path.read_bytes()
    resumed = run_campaign(config, tmp_path)
    assert resumed["sentinel"] == "must remain byte-identical"
    assert campaign_path.read_bytes() == before
