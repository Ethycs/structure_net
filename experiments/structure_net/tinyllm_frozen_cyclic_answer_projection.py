#!/usr/bin/env python3
"""Apply fixed cyclic Fourier projections to a stored TinyLLM answer curve."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_frozen_scalar_interface_decomposition as scalar


SCHEMA_VERSION = "nal.tinyllm-frozen-cyclic-answer-projection.v1"
HYPOTHESIS_ID = "tinyllm-frozen-cyclic-answer-projection-v1"
EVIDENCE_ROLE = "registered_outcome_informed_artifact_only_answer_head_diagnostic"
SOURCE_RESULT_SHA256 = (
    "e0bbb6120272627de59acf639e886d434ab53ec7ed0fc11874d77864a4dd1312"
)
SOURCE_DIAGNOSTICS_SHA256 = (
    "95334b9475bd73111f20f07747ec716fe0bf3c5ff44c14c2a91b78c825218d69"
)
INTERFACE_RESULT_SHA256 = (
    "16deb857f4449c66a22eac347c2c7e2b2cf23fbd5a8a1ad0daaf39660bf910f0"
)
INTERFACE_DIAGNOSTICS_SHA256 = (
    "3bfbf496e1cb85d79548849b533b4139c0737e7079a53e4e5bb2f974de7f9476"
)
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-frozen-cyclic-answer-projection-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "40a24744e956664b1c67950067021f45c54738f7d7dc0ae27e945e48f01ea05d"
)
SOURCE_ROOT = Path(
    "data/experiments/tinyllm_frozen_scalar_domain_extension/"
    "20260811_d10_learned_seed29_registered"
)
SOURCE_RESULT_PATH = SOURCE_ROOT / "result.json"
SOURCE_DIAGNOSTICS_PATH = SOURCE_ROOT / "diagnostics.npz"
INTERFACE_ROOT = Path(
    "data/experiments/tinyllm_frozen_scalar_interface_decomposition/"
    "20260811_d6_d10_preregistered/runs/d10/"
    "learned_calibrated_equivariant/seed_29"
)
INTERFACE_RESULT_PATH = INTERFACE_ROOT / "result.json"
INTERFACE_DIAGNOSTICS_PATH = INTERFACE_ROOT / "diagnostics.npz"
REGIMES = ("composition", "extrapolation")
STRICT_ORDERS = (1, 2, 4)
ALL_ORDERS = (0, 1, 2, 4, 8)
SOURCE_MISSING_BINS = (0, 1, 6, 15)
SOURCE_REACHABLE_BINS = (2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14)


@dataclass(frozen=True)
class CyclicProjectionConfig:
    source_root: str = str(SOURCE_ROOT)
    interface_root: str = str(INTERFACE_ROOT)
    regimes: tuple[str, ...] = REGIMES
    orders: tuple[int, ...] = ALL_ORDERS
    strict_orders: tuple[int, ...] = STRICT_ORDERS
    radii: tuple[float, ...] = (1.0, 8.0)
    semantic_shift_bins: int = 4
    replay_tolerance: float = 2e-6
    candidate_stride: int = 1
    example_limit: int = 1_024
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.orders or tuple(sorted(set(self.orders))) != self.orders:
            raise ValueError("orders must be distinct and increasing")
        if min(self.orders) < 0 or max(self.orders) > 8:
            raise ValueError("real C16 Fourier orders must lie in [0,8]")
        if set(self.strict_orders).difference(self.orders):
            raise ValueError("strict orders must be evaluated orders")
        if self.radii != (1.0, 8.0):
            raise ValueError("the registered radius pair is fixed")
        if self.semantic_shift_bins % 16 == 0:
            raise ValueError("semantic control must change the target")
        if self.candidate_stride < 1 or self.example_limit < 2:
            raise ValueError("invalid artifact subsampling")
        if not self.allow_underpowered:
            expected = {
                "source_root": str(SOURCE_ROOT),
                "interface_root": str(INTERFACE_ROOT),
                "regimes": REGIMES,
                "orders": ALL_ORDERS,
                "strict_orders": STRICT_ORDERS,
                "radii": (1.0, 8.0),
                "semantic_shift_bins": 4,
                "replay_tolerance": 2e-6,
                "candidate_stride": 1,
                "example_limit": 1_024,
            }
            actual = {
                "source_root": self.source_root,
                "interface_root": self.interface_root,
                "regimes": self.regimes,
                "orders": self.orders,
                "strict_orders": self.strict_orders,
                "radii": self.radii,
                "semantic_shift_bins": self.semantic_shift_bins,
                "replay_tolerance": self.replay_tolerance,
                "candidate_stride": self.candidate_stride,
                "example_limit": self.example_limit,
            }
            if actual != expected:
                raise ValueError("primary cyclic projection configuration changed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=32)
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


def _json_config(config: CyclicProjectionConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, Iterable):
        return all(_finite(item) for item in value)
    return True


def _implementation_sources() -> dict[str, str]:
    values = {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "task_metrics": _sha256(Path(scalar.__file__)),
    }
    if values["preregistration"] != PREREGISTRATION_SHA256:
        raise RuntimeError("cyclic answer-projection preregistration changed")
    return values


def harmonic_project_logits(
    posterior: torch.Tensor, order: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if posterior.ndim != 2 or posterior.shape[1] != 16:
        raise ValueError("posterior must have shape [n,16]")
    if not 0 <= order <= 8:
        raise ValueError("order must lie in [0,8]")
    logits = posterior.clamp_min(1e-12).log()
    logits = logits - logits.mean(1, keepdim=True)
    spectrum = torch.fft.rfft(logits.double(), dim=1)
    spectrum[:, order + 1 :] = 0
    projected_logits = torch.fft.irfft(spectrum, n=16, dim=1).float()
    return projected_logits, torch.softmax(projected_logits, dim=1)


def _oracle_summary(
    posterior_curve: torch.Tensor,
    target: torch.Tensor,
    floor: float,
) -> tuple[dict[str, Any], np.ndarray]:
    curve_bins = posterior_curve.argmax(1)
    target_bins = target.argmax(1)
    reachable_bins = sorted(set(int(item) for item in curve_bins.tolist()))
    reachable = torch.tensor(
        [int(item) in reachable_bins for item in target_bins.tolist()],
        dtype=torch.bool,
    )
    cross_entropy = -torch.matmul(
        target.double(), posterior_curve.clamp_min(1e-12).log().double().T
    )
    selected = cross_entropy.argmin(1)
    metrics = scalar.task_metrics(posterior_curve[selected], target)
    return (
        {
            "reachable_bins": reachable_bins,
            "missing_bins": sorted(set(range(16)) - set(reachable_bins)),
            "exact_bin_reachability": float(reachable.float().mean()),
            "minimum_cross_entropy_selection_metrics": metrics,
            "minimum_cross_entropy_selection_pass": scalar.task_adequacy_pass(
                metrics, floor
            ),
        },
        selected.numpy(),
    )


def classify_projection(
    orders: Mapping[int, Mapping[str, Any]],
    strict_orders: Sequence[int] = STRICT_ORDERS,
) -> str:
    specificity_candidates = []
    for order in strict_orders:
        record = orders[order]
        complete = all(
            record[regime]["radius_1"]["reachable_bins"] == list(range(16))
            and record[regime]["radius_1"]["minimum_cross_entropy_selection_pass"]
            and record[regime]["natural_metrics_pass"]
            for regime in REGIMES
        )
        chart = all(
            record[regime]["radius_1"]["reachable_bins"] == list(range(16))
            and record[regime]["radius_1"]["minimum_cross_entropy_selection_pass"]
            for regime in REGIMES
        )
        radius_eight = all(
            record[regime]["radius_8"]["reachable_bins"] == list(range(16))
            for regime in REGIMES
        )
        specificity = all(
            not record[regime]["shifted_natural_metrics_pass"] for regime in REGIMES
        )
        if complete or chart or radius_eight:
            specificity_candidates.append(specificity)
        if complete and specificity:
            return "cyclic_answer_projection_repairs_deployable_chart"
    if specificity_candidates and not all(specificity_candidates):
        return "specificity_failed"
    if any(
        all(
            orders[order][regime]["radius_1"]["reachable_bins"] == list(range(16))
            and orders[order][regime]["radius_1"][
                "minimum_cross_entropy_selection_pass"
            ]
            for regime in REGIMES
        )
        for order in strict_orders
    ):
        return "cyclic_answer_projection_repairs_chart_not_natural_calibration"
    if any(
        all(
            orders[order][regime]["radius_8"]["reachable_bins"] == list(range(16))
            for regime in REGIMES
        )
        for order in strict_orders
    ):
        return "cyclic_answer_projection_repairs_only_out_of_range"
    recovered = set()
    for order in strict_orders:
        for regime in REGIMES:
            recovered.update(
                set(orders[order][regime]["radius_8"]["reachable_bins"])
                .intersection(SOURCE_MISSING_BINS)
            )
    if recovered:
        return "partial_cyclic_answer_projection_repair"
    return "cyclic_answer_projection_does_not_repair_coverage"


def _source() -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(SOURCE_RESULT_PATH.read_text(encoding="utf-8"))
    interface = json.loads(INTERFACE_RESULT_PATH.read_text(encoding="utf-8"))
    if (
        _sha256(SOURCE_RESULT_PATH) != SOURCE_RESULT_SHA256
        or _sha256(SOURCE_DIAGNOSTICS_PATH) != SOURCE_DIAGNOSTICS_SHA256
        or _sha256(INTERFACE_RESULT_PATH) != INTERFACE_RESULT_SHA256
        or _sha256(INTERFACE_DIAGNOSTICS_PATH) != INTERFACE_DIAGNOSTICS_SHA256
        or result.get("classification")
        != "continuation_answer_curve_hole_persists_to_radius_8"
        or result.get("gates", {}).get("validity") is not True
        or interface.get("classification") != "continuation_or_answer_row_failure"
        or interface.get("gates", {}).get("validity") is not True
    ):
        raise ValueError("invalid cyclic answer-projection source")
    return result, interface


def _fingerprint(
    config: CyclicProjectionConfig, implementation: str
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "implementation_sha256": implementation,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_diagnostics_sha256": SOURCE_DIAGNOSTICS_SHA256,
            "interface_result_sha256": INTERFACE_RESULT_SHA256,
            "interface_diagnostics_sha256": INTERFACE_DIAGNOSTICS_SHA256,
        }
    )


def run(config: CyclicProjectionConfig, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _implementation_sources()
    implementation = _json_hash(sources)
    source_result, interface_result = _source()
    fingerprint = _fingerprint(config, implementation)
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

    started = time.perf_counter()
    domain_arrays = np.load(SOURCE_DIAGNOSTICS_PATH, allow_pickle=False)
    interface_arrays = np.load(INTERFACE_DIAGNOSTICS_PATH, allow_pickle=False)
    order_records: dict[int, dict[str, Any]] = {
        order: {} for order in config.orders
    }
    diagnostic_arrays: dict[str, np.ndarray] = {}
    replay_errors = []
    uniform_errors = []
    natural_matches = []
    source_bin_replays = []
    for regime in config.regimes:
        candidates = torch.from_numpy(domain_arrays[f"{regime}__candidates"]).float()
        original_curve = torch.from_numpy(
            domain_arrays[f"{regime}__posterior_curve"]
        ).float()
        target = torch.from_numpy(domain_arrays[f"{regime}__target_posterior"]).float()
        natural_scalar = torch.from_numpy(
            interface_arrays[f"{regime}__natural_scalar"]
        ).float()
        if config.example_limit < len(target):
            target = target[: config.example_limit]
            natural_scalar = natural_scalar[: config.example_limit]
        if config.candidate_stride > 1:
            keep = torch.arange(0, len(candidates), config.candidate_stride)
            endpoints = torch.tensor([0, len(candidates) - 1])
            natural_indices_full = torch.searchsorted(candidates, natural_scalar)
            keep = torch.unique(torch.cat((keep, endpoints, natural_indices_full)), sorted=True)
            candidates = candidates[keep]
            original_curve = original_curve[keep]
        natural_indices = torch.searchsorted(candidates, natural_scalar)
        natural_match = bool(
            natural_indices.max() < len(candidates)
            and torch.equal(candidates[natural_indices], natural_scalar)
        )
        natural_matches.append(natural_match)
        floor = float(interface_result["regimes"][regime]["task_accuracy_floor"])
        target_normalization_error = float((target.sum(1) - 1.0).abs().max())
        if target_normalization_error > 2e-6:
            raise ValueError("stored target posterior is not normalized")
        for order in config.orders:
            projected_logits, projected_curve = harmonic_project_logits(
                original_curve, order
            )
            shifted_curve = torch.softmax(
                torch.roll(
                    projected_logits, shifts=config.semantic_shift_bins, dims=1
                ),
                dim=1,
            )
            natural_posterior = projected_curve[natural_indices]
            shifted_natural_posterior = shifted_curve[natural_indices]
            natural_metrics = scalar.task_metrics(natural_posterior, target)
            shifted_natural_metrics = scalar.task_metrics(
                shifted_natural_posterior, target
            )
            record: dict[str, Any] = {
                "natural_metrics": natural_metrics,
                "natural_metrics_pass": scalar.task_adequacy_pass(
                    natural_metrics, floor
                ),
                "shifted_natural_metrics": shifted_natural_metrics,
                "shifted_natural_metrics_pass": scalar.task_adequacy_pass(
                    shifted_natural_metrics, floor
                ),
                "task_accuracy_floor": floor,
            }
            for radius in config.radii:
                mask = candidates.abs() <= radius + 1e-7
                oracle, selected = _oracle_summary(
                    projected_curve[mask], target, floor
                )
                record[f"radius_{radius:g}"] = oracle
                diagnostic_arrays[
                    f"order_{order}__{regime}__radius_{radius:g}__selected_indices"
                ] = selected
            order_records[order][regime] = record
            diagnostic_arrays[f"order_{order}__{regime}__posterior_curve"] = (
                projected_curve.numpy()
            )
            diagnostic_arrays[f"order_{order}__{regime}__shifted_curve"] = (
                shifted_curve.numpy()
            )
            if order == 8:
                replay_errors.append(float((projected_curve - original_curve).abs().max()))
            if order == 0:
                uniform_errors.append(
                    float((projected_curve - 1.0 / 16.0).abs().max())
                )
        source_bin_replays.append(
            order_records[8][regime]["radius_1"]["reachable_bins"]
            == list(SOURCE_REACHABLE_BINS)
            and order_records[8][regime]["radius_8"]["reachable_bins"]
            == list(SOURCE_REACHABLE_BINS)
        )
        diagnostic_arrays[f"{regime}__candidates"] = candidates.numpy()
        diagnostic_arrays[f"{regime}__natural_indices"] = natural_indices.numpy()
        diagnostic_arrays[f"{regime}__target_posterior"] = target.numpy()

    identity_replay = max(replay_errors) <= config.replay_tolerance
    uniform_control = max(uniform_errors) <= config.replay_tolerance
    natural_index_replay = all(natural_matches)
    source_bin_replay = all(source_bin_replays)
    finite = _finite(order_records) and all(
        np.isfinite(value).all() for value in diagnostic_arrays.values()
    )
    validity = bool(
        identity_replay
        and uniform_control
        and natural_index_replay
        and source_bin_replay
        and finite
    )
    classification = (
        classify_projection(order_records, config.strict_orders)
        if validity
        else "invalid"
    )
    complete_orders = [
        order
        for order in config.strict_orders
        if all(
            order_records[order][regime]["radius_1"]["reachable_bins"]
            == list(range(16))
            and order_records[order][regime]["radius_1"][
                "minimum_cross_entropy_selection_pass"
            ]
            and order_records[order][regime]["natural_metrics_pass"]
            and not order_records[order][regime]["shifted_natural_metrics_pass"]
            for regime in REGIMES
        )
    ]
    closure_write = getattr(scalar.closure, "_write_npz")
    closure_write(diagnostics_path, diagnostic_arrays)
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": "tinyllm-cyclic-answer-projection-d10-learned-seed29",
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_scientific_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "completed_at": _utc_now(),
        "configuration": _json_config(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "checkpoint_load_count": 0,
            "device": "cpu_artifact_only",
        },
        "implementation_sha256": implementation,
        "implementation_sources": sources,
        "scientific_fingerprint": fingerprint,
        "source": {
            "result": str(SOURCE_RESULT_PATH),
            "result_sha256": SOURCE_RESULT_SHA256,
            "diagnostics": str(SOURCE_DIAGNOSTICS_PATH),
            "diagnostics_sha256": SOURCE_DIAGNOSTICS_SHA256,
            "interface_result": str(INTERFACE_RESULT_PATH),
            "interface_result_sha256": INTERFACE_RESULT_SHA256,
            "interface_diagnostics": str(INTERFACE_DIAGNOSTICS_PATH),
            "interface_diagnostics_sha256": INTERFACE_DIAGNOSTICS_SHA256,
            "dataset_hashes": source_result["dataset_hashes"],
        },
        "orders": {str(key): value for key, value in order_records.items()},
        "classification": classification,
        "gates": {
            "identity_replay": identity_replay,
            "uniform_control": uniform_control,
            "natural_index_replay": natural_index_replay,
            "source_bin_replay": source_bin_replay,
            "finite": finite,
            "validity": validity,
        },
        "summary": {
            "smallest_complete_order": min(complete_orders) if complete_orders else None,
            "complete_order_count": len(complete_orders),
            "maximum_identity_replay_error": max(replay_errors),
            "maximum_uniform_control_error": max(uniform_errors),
            "trained_models": 0,
            "trained_frontends": 0,
            "trained_heads": 0,
            "trained_probes": 0,
            "fitted_parameters": 0,
            "checkpoint_load_count": 0,
        },
        "method_boundaries": [
            "This is an outcome-informed artifact-only diagnostic from one checkpoint.",
            "The fixed logit projection is not a deployable tied-embedding model.",
            "A successful counterfactual would license, not replace, prospective multi-seed testing.",
            "No checkpoint was loaded and no parameter was trained or fit.",
        ],
        "wall_seconds": time.perf_counter() - started,
        "artifacts": {
            "result": str(result_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
        },
    }
    _write_json(result_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_frozen_cyclic_answer_projection/"
            "20260811_d10_learned_seed29_registered"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = (
        CyclicProjectionConfig(
            orders=(0, 1, 8),
            strict_orders=(1,),
            candidate_stride=1,
            example_limit=64,
            allow_underpowered=True,
        )
        if args.shakedown
        else CyclicProjectionConfig()
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
