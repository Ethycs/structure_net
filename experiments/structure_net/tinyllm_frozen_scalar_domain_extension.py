#!/usr/bin/env python3
"""Scan the frozen d10 learned scalar continuation outside its encoder image."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_calibrated_frontend_causal as calibrated
import experiments.structure_net.tinyllm_calibrated_frontend_causal_closure as closure
import experiments.structure_net.tinyllm_frozen_scalar_interface_decomposition as source


SCHEMA_VERSION = "nal.tinyllm-frozen-scalar-domain-extension.v1"
HYPOTHESIS_ID = "tinyllm-frozen-scalar-domain-extension-v1"
EVIDENCE_ROLE = "registered_outcome_informed_frozen_scalar_domain_diagnostic"
SOURCE_CAMPAIGN_SHA256 = (
    "f71b85bc51f9694346fa23482bc63e1631e0918d55f5cae4f3a0d5ce09a47ba2"
)
SOURCE_RESULT_MANIFEST_SHA256 = (
    "201f6bb780e5a92938df1596e61f9f75164bb0e3bfcd0554f6ad03ba74f177b4"
)
SOURCE_RESULT_SHA256 = (
    "16deb857f4449c66a22eac347c2c7e2b2cf23fbd5a8a1ad0daaf39660bf910f0"
)
SOURCE_DIAGNOSTICS_SHA256 = (
    "3bfbf496e1cb85d79548849b533b4139c0737e7079a53e4e5bb2f974de7f9476"
)
SOURCE_RUNNER_SHA256 = source.RUNNER_SHA256 if hasattr(source, "RUNNER_SHA256") else (
    "0228836047e99c8af865e377e8cae329ddb56c70a4de22207087b8d10f2da128"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-frozen-scalar-domain-extension-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "1eec9eec7832124d5e4fd8d66632e2bede641cba1f4559f2ae677d909670a96e"
)
SOURCE_ROOT = Path(
    "data/experiments/tinyllm_frozen_scalar_interface_decomposition/"
    "20260811_d6_d10_preregistered"
)
SOURCE_RESULT_PATH = (
    SOURCE_ROOT
    / "runs/d10/learned_calibrated_equivariant/seed_29/result.json"
)
SOURCE_DIAGNOSTICS_PATH = SOURCE_RESULT_PATH.with_name("diagnostics.npz")
SOURCE_MISSING_BINS = (0, 1, 6, 15)
SOURCE_REACHABLE_BINS = (2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14)
REGIMES = ("composition", "extrapolation")


@dataclass(frozen=True)
class DomainExtensionConfig:
    source_root: str = str(SOURCE_ROOT)
    preset: str = "d10"
    condition: str = "learned_calibrated_equivariant"
    seed: int = 29
    regimes: tuple[str, ...] = REGIMES
    samples_per_regime: int = 1_024
    radii: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    grid_step_denominator: int = 2_048
    batch_size: int = 512
    replay_tolerance: float = 2e-6
    device: str = "cuda:0"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if (self.preset, self.condition, self.seed) != (
            "d10",
            "learned_calibrated_equivariant",
            29,
        ):
            raise ValueError("only the registered d10 learned seed-29 cell is allowed")
        if self.regimes != REGIMES:
            raise ValueError("both registered shifts are required")
        if not self.radii or tuple(sorted(set(self.radii))) != self.radii:
            raise ValueError("radii must be distinct and increasing")
        if self.radii[0] != 1.0 or self.radii[-1] <= 1.0:
            raise ValueError("radii must begin at the source boundary and extend it")
        if self.grid_step_denominator < 2 or self.batch_size < 1:
            raise ValueError("invalid grid or batch size")
        if self.samples_per_regime < 2:
            raise ValueError("at least two examples per shift are required")
        if not self.allow_underpowered:
            expected = {
                "source_root": str(SOURCE_ROOT),
                "regimes": REGIMES,
                "samples_per_regime": 1_024,
                "radii": (1.0, 2.0, 4.0, 8.0),
                "grid_step_denominator": 2_048,
                "batch_size": 512,
                "replay_tolerance": 2e-6,
            }
            actual = {
                "source_root": self.source_root,
                "regimes": self.regimes,
                "samples_per_regime": self.samples_per_regime,
                "radii": self.radii,
                "grid_step_denominator": self.grid_step_denominator,
                "batch_size": self.batch_size,
                "replay_tolerance": self.replay_tolerance,
            }
            if actual != expected:
                raise ValueError("primary domain-extension configuration changed")

    @property
    def grid_step(self) -> float:
        return 1.0 / self.grid_step_denominator

    @property
    def fixed_grid_points(self) -> int:
        return int(round(2.0 * self.radii[-1] * self.grid_step_denominator)) + 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=64)
def _sha256_cached(path: Path, size: int, modified_ns: int) -> str:
    del size, modified_ns
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(path, stat.st_size, stat.st_mtime_ns)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_config(config: DomainExtensionConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _finite(value: Any) -> bool:
    return closure._finite(value)


def _implementation_sources() -> dict[str, str]:
    values = {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "scalar_interface_runner": _sha256(Path(source.__file__)),
        "calibrated_frontend": _sha256(Path(calibrated.__file__)),
        "causal_closure": _sha256(Path(closure.__file__)),
    }
    if values["preregistration"] != PREREGISTRATION_SHA256:
        raise RuntimeError("domain-extension preregistration changed")
    if values["scalar_interface_runner"] != SOURCE_RUNNER_SHA256:
        raise RuntimeError("scalar-interface source runner changed")
    return values


def _implementation_digest(values: Mapping[str, str]) -> str:
    return _json_hash(dict(values))


def _source_bundle(
    config: DomainExtensionConfig,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Any,
    dict[str, calibrated.CalibratedDataset],
    dict[str, str],
]:
    campaign_path = Path(config.source_root) / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    detail = json.loads(SOURCE_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        _sha256(campaign_path) != SOURCE_CAMPAIGN_SHA256
        or campaign.get("schema_version") != source.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != source.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("result_manifest_sha256") != SOURCE_RESULT_MANIFEST_SHA256
        or _sha256(SOURCE_RESULT_PATH) != SOURCE_RESULT_SHA256
        or _sha256(SOURCE_DIAGNOSTICS_PATH) != SOURCE_DIAGNOSTICS_SHA256
        or detail.get("preset") != config.preset
        or detail.get("condition") != config.condition
        or int(detail.get("seed", -1)) != config.seed
        or detail.get("classification") != "continuation_or_answer_row_failure"
        or detail.get("gates", {}).get("validity") is not True
    ):
        raise ValueError("invalid frozen scalar-interface source")

    source_config = source.ScalarInterfaceConfig(
        samples_per_regime=config.samples_per_regime,
        batch_size=config.batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    _, task, architecture_details = source._source_details(source_config)
    architecture_detail = architecture_details[(config.preset, config.condition, config.seed)]
    if (
        architecture_detail["_result_sha256"] != detail["source"]["result_sha256"]
        or architecture_detail["artifacts"]["checkpoint_sha256"]
        != detail["source"]["model_checkpoint_sha256"]
        or architecture_detail["artifacts"]["frontend_checkpoint_sha256"]
        != detail["source"]["frontend_checkpoint_sha256"]
    ):
        raise ValueError("architecture source identity changed")
    datasets = source._datasets(task, source_config)
    dataset_hashes = source._dataset_hashes(datasets)
    if not config.allow_underpowered and dataset_hashes != campaign["dataset_hashes"]:
        raise ValueError("primary dataset hashes changed")
    return campaign, detail, task, datasets, dataset_hashes


def classify_external_bins(discovered_by_regime: Mapping[str, Sequence[int]]) -> str:
    expected = set(SOURCE_MISSING_BINS)
    common = set.intersection(
        *(set(discovered_by_regime[regime]) for regime in REGIMES)
    )
    union = set.union(*(set(discovered_by_regime[regime]) for regime in REGIMES))
    if common == expected:
        return "bounded_encoder_range_hole"
    if not union.intersection(expected):
        return "continuation_answer_curve_hole_persists_to_radius_8"
    return "mixed_range_and_answer_curve_hole"


def _bin_regions(
    candidates: torch.Tensor, posterior: torch.Tensor, phase_bins: int
) -> tuple[list[int], dict[str, Any]]:
    winners = posterior.argmax(1)
    reachable = sorted(set(int(item) for item in winners.tolist()))
    regions: dict[str, Any] = {}
    for target_bin in range(phase_bins):
        indices = torch.nonzero(winners == target_bin, as_tuple=False).reshape(-1)
        if not len(indices):
            regions[str(target_bin)] = None
            continue
        values = candidates[indices]
        probabilities = posterior[indices, target_bin]
        nearest = indices[values.abs().argmin()]
        peak = indices[probabilities.argmax()]
        regions[str(target_bin)] = {
            "minimum_scalar": float(values.min()),
            "maximum_scalar": float(values.max()),
            "nearest_to_zero_scalar": float(candidates[nearest]),
            "peak_probability": float(posterior[peak, target_bin]),
            "peak_probability_scalar": float(candidates[peak]),
            "resolved_point_count": int(len(indices)),
        }
    return reachable, regions


def _oracle_summary(
    candidates: torch.Tensor,
    posterior: torch.Tensor,
    target: torch.Tensor,
    floor: float,
) -> tuple[dict[str, Any], np.ndarray]:
    target_bins = target.argmax(1)
    curve_bins = posterior.argmax(1)
    reachable_bins = sorted(set(int(item) for item in curve_bins.tolist()))
    reachable = torch.tensor(
        [int(item) in reachable_bins for item in target_bins.tolist()],
        dtype=torch.bool,
    )
    cross_entropy = -torch.matmul(
        target.double(), posterior.clamp_min(1e-12).log().double().T
    )
    selected = cross_entropy.argmin(1)
    selected_posterior = posterior[selected]
    metrics = source.task_metrics(selected_posterior, target)
    record = {
        "candidate_count": int(len(candidates)),
        "reachable_target_bins": reachable_bins,
        "missing_target_bins": sorted(set(range(target.shape[1])) - set(reachable_bins)),
        "exact_bin_reachability": float(reachable.float().mean()),
        "unreachable_example_count": int((~reachable).sum()),
        "minimum_cross_entropy_selection_metrics": metrics,
        "minimum_cross_entropy_selection_pass": source.task_adequacy_pass(metrics, floor),
        "selected_scalar": source._safe_distribution_summary(candidates[selected]),
    }
    return record, selected.numpy()


def _fingerprint(
    config: DomainExtensionConfig,
    implementation: str,
    dataset_hashes: Mapping[str, str],
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "implementation_sha256": implementation,
            "source_campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_diagnostics_sha256": SOURCE_DIAGNOSTICS_SHA256,
            "dataset_hashes": dict(dataset_hashes),
        }
    )


def run(config: DomainExtensionConfig, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    implementation_sources = _implementation_sources()
    implementation = _implementation_digest(implementation_sources)
    campaign, source_detail, task, datasets, dataset_hashes = _source_bundle(config)
    fingerprint = _fingerprint(config, implementation, dataset_hashes)
    result_path = output_dir / "result.json"
    diagnostics_path = output_dir / "diagnostics.npz"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "completed"
            and existing.get("scientific_fingerprint") == fingerprint
            and existing.get("implementation_sha256") == implementation
            and existing.get("gates", {}).get("validity") is True
            and diagnostics_path.is_file()
            and _sha256(diagnostics_path)
            == existing.get("artifacts", {}).get("diagnostics_sha256")
        ):
            print("result already complete; leaving bytes unchanged", flush=True)
            return existing

    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    source_config = source.ScalarInterfaceConfig(
        samples_per_regime=config.samples_per_regime,
        batch_size=config.batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    _, _, architecture_details = source._source_details(source_config)
    architecture_detail = architecture_details[(config.preset, config.condition, config.seed)]
    system = source._load_system(
        architecture_detail, task, config.preset, config.condition, device
    )
    state_before = {
        "model": calibrated._state_digest(system.model),
        "system": calibrated._module_digest(system),
    }
    fixed_grid = torch.linspace(
        -config.radii[-1],
        config.radii[-1],
        config.fixed_grid_points,
        dtype=torch.float32,
    )
    source_arrays = np.load(SOURCE_DIAGNOSTICS_PATH, allow_pickle=False)
    regime_records: dict[str, Any] = {}
    diagnostic_arrays: dict[str, np.ndarray] = {"fixed_grid": fixed_grid.numpy()}
    discovered_by_regime: dict[str, list[int]] = {}
    replay_errors = []
    for regime in config.regimes:
        dataset = datasets[regime]
        input_ids = dataset.paired.circle.input_ids
        target = dataset.paired.circle.target_posteriors.float()
        source_candidates = torch.from_numpy(
            source_arrays[f"{regime}__context_0_candidates"]
        ).float()
        stored_curve = torch.from_numpy(
            source_arrays[f"{regime}__context_0_posterior_curve"]
        ).float()
        candidates = torch.unique(torch.cat((fixed_grid, source_candidates)), sorted=True)
        representative = input_ids[:1]
        curve = source._scalar_posteriors(
            system,
            representative.repeat(len(candidates), 1),
            candidates,
            task,
            source_config,
            device,
        )
        source_indices = torch.searchsorted(candidates, source_candidates)
        source_replay_error = float((curve[source_indices] - stored_curve).abs().max())
        replay_errors.append(source_replay_error)
        floor = float(source_detail["regimes"][regime]["task_accuracy_floor"])
        ladders: dict[str, Any] = {}
        selected_arrays: dict[str, np.ndarray] = {}
        for radius in config.radii:
            mask = candidates.abs() <= radius + config.grid_step / 4.0
            local_candidates = candidates[mask]
            local_curve = curve[mask]
            reachable_bins, regions = _bin_regions(
                local_candidates, local_curve, task.phase_bins
            )
            oracle, selected = _oracle_summary(
                local_candidates, local_curve, target, floor
            )
            radius_key = f"{radius:g}"
            ladders[radius_key] = {
                "radius": radius,
                "reachable_target_bins": reachable_bins,
                "missing_target_bins": sorted(
                    set(range(task.phase_bins)) - set(reachable_bins)
                ),
                "bin_regions": regions,
                "oracle": oracle,
                "negative_boundary_winner": int(local_curve[0].argmax()),
                "positive_boundary_winner": int(local_curve[-1].argmax()),
                "negative_boundary_max_probability": float(local_curve[0].max()),
                "positive_boundary_max_probability": float(local_curve[-1].max()),
            }
            selected_arrays[f"{regime}__radius_{radius_key}__selected_indices"] = selected
        source_bins_replayed = tuple(ladders["1"]["reachable_target_bins"])
        external_mask = candidates.abs() > 1.0 + config.grid_step / 4.0
        external_bins = sorted(
            set(int(item) for item in curve[external_mask].argmax(1).tolist())
            .intersection(SOURCE_MISSING_BINS)
        )
        discovered_by_regime[regime] = external_bins
        regime_records[regime] = {
            "dataset_sha256": dataset_hashes[regime],
            "task_accuracy_floor": floor,
            "source_curve_replay_maximum_absolute_error": source_replay_error,
            "source_reachable_bins_replayed": list(source_bins_replayed),
            "source_missing_bins": list(SOURCE_MISSING_BINS),
            "source_missing_bins_discovered_outside_encoder_image": external_bins,
            "ladders": ladders,
        }
        diagnostic_arrays.update(
            {
                f"{regime}__candidates": candidates.numpy(),
                f"{regime}__posterior_curve": curve.numpy(),
                f"{regime}__target_posterior": target.numpy(),
                **selected_arrays,
            }
        )

    classification = classify_external_bins(discovered_by_regime)
    state_after = {
        "model": calibrated._state_digest(system.model),
        "system": calibrated._module_digest(system),
    }
    state_unchanged = state_before == state_after == {
        "model": architecture_detail["training"]["final_model_state_sha256"],
        "system": architecture_detail["training"]["final_system_state_sha256"],
    }
    source_curve_replay = max(replay_errors) <= config.replay_tolerance
    source_bin_replay = all(
        tuple(regime_records[regime]["source_reachable_bins_replayed"])
        == SOURCE_REACHABLE_BINS
        for regime in config.regimes
    )
    finite = _finite(regime_records) and all(
        np.isfinite(value).all() for value in diagnostic_arrays.values()
    )
    cuda_valid = device.type == "cuda" and torch.cuda.is_available()
    validity = bool(
        source_curve_replay
        and source_bin_replay
        and state_unchanged
        and finite
        and (cuda_valid or config.allow_underpowered)
    )
    if not validity:
        classification = "invalid"
    closure._write_npz(diagnostics_path, diagnostic_arrays)
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": "tinyllm-frozen-scalar-domain-d10-learned-seed29",
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_scientific_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "completed_at": _utc_now(),
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
        "implementation_sha256": implementation,
        "implementation_sources": implementation_sources,
        "scientific_fingerprint": fingerprint,
        "source": {
            "campaign": str(Path(config.source_root) / "campaign_results.json"),
            "campaign_sha256": SOURCE_CAMPAIGN_SHA256,
            "result": str(SOURCE_RESULT_PATH),
            "result_sha256": SOURCE_RESULT_SHA256,
            "diagnostics": str(SOURCE_DIAGNOSTICS_PATH),
            "diagnostics_sha256": SOURCE_DIAGNOSTICS_SHA256,
            "architecture_campaign_sha256": campaign["source"]["campaign_sha256"],
            "model_checkpoint": source_detail["source"]["model_checkpoint"],
            "model_checkpoint_sha256": source_detail["source"][
                "model_checkpoint_sha256"
            ],
            "frontend_checkpoint": source_detail["source"]["frontend_checkpoint"],
            "frontend_checkpoint_sha256": source_detail["source"][
                "frontend_checkpoint_sha256"
            ],
        },
        "dataset_hashes": dataset_hashes,
        "regimes": regime_records,
        "classification": classification,
        "gates": {
            "source_curve_replay": source_curve_replay,
            "source_bin_replay": source_bin_replay,
            "state_unchanged": state_unchanged,
            "finite": finite,
            "cuda_valid": cuda_valid,
            "validity": validity,
        },
        "state_record": {"before": state_before, "after": state_after},
        "summary": {
            "trained_models": 0,
            "trained_frontends": 0,
            "trained_probes": 0,
            "fitted_parameters": 0,
            "source_missing_bin_count": len(SOURCE_MISSING_BINS),
            "common_external_discovered_bin_count": len(
                set.intersection(
                    *(set(discovered_by_regime[regime]) for regime in REGIMES)
                )
            ),
        },
        "method_boundaries": [
            "This is outcome-informed localization from one frozen checkpoint.",
            "Injected scalars outside [-1,1] are not producible by the retained tanh encoder.",
            "Finite-grid absence is limited to radius 8 at resolution 1/2048.",
            "Minimum-cross-entropy selection uses hidden target information and is not a repair.",
        ],
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_gb": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
        "artifacts": {
            "result": str(result_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
        },
    }
    _write_json(result_path, result)
    del system
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_frozen_scalar_domain_extension/"
            "20260811_d10_learned_seed29_registered"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.shakedown:
        config = DomainExtensionConfig(
            samples_per_regime=min(args.samples, 64),
            radii=(1.0, 2.0),
            grid_step_denominator=128,
            batch_size=args.batch_size,
            device=args.device,
            allow_underpowered=True,
        )
    else:
        config = DomainExtensionConfig(
            samples_per_regime=args.samples,
            batch_size=args.batch_size,
            device=args.device,
        )
    result = run(config, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence_role": result["evidence_role"],
                "classification": result["classification"],
                "validity": result["gates"]["validity"],
                "summary": result["summary"],
                "wall_seconds": result["wall_seconds"],
                "peak_cuda_allocated_gb": result["peak_cuda_allocated_gb"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
