#!/usr/bin/env python3
"""Audit affine sensor gauges and inverse scalar-embedding transport without training."""

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
import torch.nn.functional as F

import experiments.structure_net.tinyllm_joint_full_interface as full
import experiments.structure_net.tinyllm_joint_physical_scalar_interface as joint
import neural_architecture_lab.joint_full_interface_meta_hypothesis as parent
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-affine-gauge-transport.v1"
HYPOTHESIS_ID = "tinyllm-affine-gauge-transport-v1"
EVIDENCE_ROLE = "registered_post_outcome_artifact_only_gauge_audit"
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-affine-gauge-transport-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "aa7624e98859822592552e0d2a645c8e90a2946fdd5bcf64a72ef37f938a4813"
)
PARENT_ROOT = Path(
    "data/experiments/tinyllm_joint_full_interface/"
    "20260811_d6_d10_preregistered"
)
PARENT_CAMPAIGN_SHA256 = parent.CAMPAIGN_SHA256
PRESETS = parent.PRESETS
CONDITION = parent.CONDITION
SEEDS = parent.SEEDS
ARMS = parent.ARMS
REGIMES = parent.REGIMES


@dataclass(frozen=True)
class AffineGaugeConfig:
    parent_root: str = str(PARENT_ROOT)
    presets: tuple[str, ...] = PRESETS
    seeds: tuple[int, ...] = SEEDS
    encoder_batch_size: int = 512
    slope_absolute_minimum: float = 1e-4
    transport_tolerance: float = 2e-6
    cosine_minimum: float = 0.90
    branch_accuracy_maximum: float = 0.55
    conditional_log_loss_gain_maximum: float = 0.02
    required_seed_passes: int = 4
    shuffled_seed_pass_ceiling: int = 1
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.presets or set(self.presets).difference(PRESETS):
            raise ValueError("unknown or empty preset selection")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if min(
            self.encoder_batch_size,
            self.slope_absolute_minimum,
            self.transport_tolerance,
        ) <= 0:
            raise ValueError("batch size, slope floor, and tolerance must be positive")
        if not 1 <= self.required_seed_passes <= len(self.seeds):
            raise ValueError("required seed passes are outside the population")
        if not self.allow_underpowered:
            expected = {
                "parent_root": str(PARENT_ROOT),
                "presets": PRESETS,
                "seeds": SEEDS,
                "encoder_batch_size": 512,
                "slope_absolute_minimum": 1e-4,
                "transport_tolerance": 2e-6,
                "cosine_minimum": 0.90,
                "branch_accuracy_maximum": 0.55,
                "conditional_log_loss_gain_maximum": 0.02,
                "required_seed_passes": 4,
                "shuffled_seed_pass_ceiling": 1,
            }
            actual = {key: getattr(self, key) for key in expected}
            if actual != expected:
                raise ValueError("primary affine-gauge configuration changed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@lru_cache(maxsize=256)
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


def _manifest_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=str):
        digest.update(f"{_sha256(path)}  {path}\n".encode("utf-8"))
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _json_config(config: AffineGaugeConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def _implementation_sources() -> dict[str, str]:
    values = {
        "runner": _sha256(Path(__file__)),
        "preregistration": _sha256(PREREGISTRATION_PATH),
        "parent_validator": _sha256(Path(parent.__file__)),
        "full_interface_runner": _sha256(Path(full.__file__)),
        "joint_interface_runner": _sha256(Path(joint.__file__)),
        "calibrated_frontend": _sha256(Path(joint.calibrated.__file__)),
        "interval_metrics": _sha256(Path(joint.interval.__file__)),
    }
    if values["preregistration"] != PREREGISTRATION_SHA256:
        raise RuntimeError("affine-gauge preregistration changed")
    if values["full_interface_runner"] != parent.RUNNER_SHA256:
        raise RuntimeError("full-interface source runner changed")
    return values


def _implementation_digest(values: Mapping[str, str] | None = None) -> str:
    return _json_hash(dict(values or _implementation_sources()))


def fit_affine_chart(
    scalar: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    x = target.reshape(-1).double()
    y = scalar.reshape(-1).double()
    design = torch.stack((x, torch.ones_like(x)), dim=1)
    solution = torch.linalg.lstsq(design, y[:, None]).solution[:, 0]
    alpha, beta = float(solution[0]), float(solution[1])
    prediction = alpha * x + beta
    residual = float((y - prediction).square().sum())
    total = float((y - y.mean()).square().sum())
    return {
        "alpha": alpha,
        "beta": beta,
        "orientation": 1.0 if alpha >= 0.0 else -1.0,
        "r_squared": 1.0 - residual / max(total, 1e-30),
        "scalar_rmse": math.sqrt(float((y - prediction).square().mean())),
    }


def canonicalize(
    scalar: torch.Tensor, alpha: float, beta: float
) -> torch.Tensor:
    return (scalar - scalar.new_tensor(beta)) / scalar.new_tensor(alpha)


def embedding_transport_errors(
    scalars: Iterable[torch.Tensor],
    weight: torch.Tensor,
    bias: torch.Tensor,
    alpha: float,
    beta: float,
) -> dict[str, float]:
    maxima = {"float32": 0.0, "float64": 0.0}
    for dtype, key in ((torch.float32, "float32"), (torch.float64, "float64")):
        w = weight.to(dtype=dtype)
        b = bias.to(dtype=dtype)
        a = torch.tensor(alpha, dtype=dtype)
        c = torch.tensor(beta, dtype=dtype)
        transported_weight = a * w
        transported_bias = b + c * w[:, 0]
        for scalar in scalars:
            value = scalar.reshape(-1, 1).to(dtype=dtype)
            chart = (value - c) / a
            original = F.linear(value, w, b)
            transported = F.linear(chart, transported_weight, transported_bias)
            maxima[key] = max(
                maxima[key], float((original - transported).abs().max())
            )
    return {
        "maximum_float32_absolute_error": maxima["float32"],
        "maximum_float64_absolute_error": maxima["float64"],
    }


@torch.inference_mode()
def extract_training_scalar(
    checkpoint: Mapping[str, Any],
    dataset: joint.calibrated.CalibratedDataset,
    task: CircleTaskConfig,
    batch_size: int,
) -> torch.Tensor:
    encoder = joint.calibrated.CalibratedEquivariantEncoder(
        task.sensor_steps, vector_channels=16
    )
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    sensor = joint.calibrated.decode_sensor_tokens(
        dataset.paired.circle.input_ids, task
    )
    values = []
    for start in range(0, len(sensor), batch_size):
        stop = min(len(sensor), start + batch_size)
        values.append(encoder(sensor[start:stop], dataset.calibration[start:stop]))
    return torch.cat(values).reshape(-1).float()


def _best_shift_fit(scalar: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return fit_affine_chart(scalar, target)


def _canonical_front_record(
    scalar: torch.Tensor,
    exact_cosine: torch.Tensor,
    fiber_id: torch.Tensor,
    target_posterior: torch.Tensor,
    parent_record: Mapping[str, Any],
    task: CircleTaskConfig,
    config: AffineGaugeConfig,
    alpha: float,
    beta: float,
) -> tuple[dict[str, Any], torch.Tensor]:
    value = canonicalize(scalar, alpha, beta).float()
    scalar_metrics = joint.interval.scalar_metrics(value, exact_cosine, fiber_id)
    posterior = joint.interval_posterior_unclipped(value, task.phase_bins).cpu()
    task_metrics = joint.interval.task_metrics(posterior, target_posterior)
    branch = dict(parent_record["conditional_branch"])
    task_floor = float(parent_record["task_accuracy_floor"])
    task_gate = float(task_metrics["exact_bin_accuracy"]) >= task_floor
    endpoint_pass = bool(
        float(scalar_metrics["cosine_pearson"]) >= config.cosine_minimum
        and float(branch["balanced_accuracy"])
        <= config.branch_accuracy_maximum
        and float(branch["conditional_log_loss_gain_over_cosine_only"])
        <= config.conditional_log_loss_gain_maximum
        and task_gate
    )
    return {
        "scalar_metrics": scalar_metrics,
        "task_metrics": task_metrics,
        "task_accuracy_floor": task_floor,
        "task_gate": task_gate,
        "conditional_branch": branch,
        "conditional_branch_reuse_reason": (
            "invertible_affine_chart_preserves_represented_information"
        ),
        "endpoint_pass": endpoint_pass,
    }, value


def _cell_directory(root: Path, preset: str, seed: int) -> Path:
    return root / "runs" / preset / CONDITION / f"seed_{seed}"


def _fingerprint(
    config: AffineGaugeConfig,
    implementation: str,
    detail: Mapping[str, Any],
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "implementation_sha256": implementation,
            "parent_result_sha256": _sha256(Path(detail["artifacts"]["result"])),
            "parent_diagnostics_sha256": detail["artifacts"][
                "diagnostics_sha256"
            ],
            "parent_checkpoint_sha256": {
                arm: detail["artifacts"]["full_interfaces"][arm]["sha256"]
                for arm in ARMS
            },
        }
    )


def analyze_cell(
    detail: Mapping[str, Any],
    task: CircleTaskConfig,
    config: AffineGaugeConfig,
    output_root: Path,
    implementation: str,
    sources: Mapping[str, str],
) -> dict[str, Any]:
    preset = str(detail["preset"])
    seed = int(detail["seed"])
    cell = _cell_directory(output_root, preset, seed)
    cell.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    protocol = joint._sealed_training_protocol_config(preset)
    training, sealed_batches, training_hash, batch_hash = (
        joint.calibrated._protocol_material(task, protocol, seed)
    )
    if (
        training_hash != detail["source"]["training_data_sha256"]
        or batch_hash != detail["source"]["minibatch_schedule_sha256"]
    ):
        raise RuntimeError("sealed training material changed")
    diagnostics_path = Path(detail["artifacts"]["diagnostics"])
    if _sha256(diagnostics_path) != detail["artifacts"]["diagnostics_sha256"]:
        raise RuntimeError("parent diagnostics changed")
    with np.load(diagnostics_path, allow_pickle=False) as loaded:
        parent_arrays = {name: loaded[name].copy() for name in loaded.files}
    permutation = torch.from_numpy(
        parent_arrays["pair_shuffled_target_permutation"]
    ).long()
    if (
        joint.calibrated._tensor_digest(permutation)
        != detail["target_permutation_sha256"]
    ):
        raise RuntimeError("parent target permutation changed")

    arrays: dict[str, np.ndarray] = {
        "training_pair_batches": sealed_batches.numpy(),
        "pair_shuffled_target_permutation": permutation.numpy(),
        "training_exact_cosine": training.paired.fiber.cosine.numpy(),
    }
    arms: dict[str, Any] = {}
    for arm in ARMS:
        checkpoint_record = detail["artifacts"]["full_interfaces"][arm]
        checkpoint_path = Path(checkpoint_record["path"])
        if _sha256(checkpoint_path) != checkpoint_record["sha256"]:
            raise RuntimeError("parent full checkpoint changed")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if (
            checkpoint.get("schema_version") != full.SCHEMA_VERSION
            or checkpoint.get("hypothesis_id") != full.HYPOTHESIS_ID
            or checkpoint.get("preset") != preset
            or int(checkpoint.get("seed", -1)) != seed
            or checkpoint.get("arm") != arm
            or checkpoint.get("model_state_sha256")
            != detail["arms"][arm]["training"]["final_model_state_sha256"]
            or checkpoint.get("interface_state_sha256")
            != detail["arms"][arm]["training"]["final_interface_state_sha256"]
        ):
            raise RuntimeError("parent checkpoint metadata changed")
        training_scalar = extract_training_scalar(
            checkpoint, training, task, config.encoder_batch_size
        )
        physical_target = training.paired.fiber.cosine.reshape(-1).float()
        fit_target = (
            physical_target
            if arm == "physical_true"
            else physical_target[permutation]
        )
        fit = fit_affine_chart(training_scalar, fit_target)
        slope_valid = abs(fit["alpha"]) >= config.slope_absolute_minimum
        if not slope_valid:
            raise RuntimeError("affine chart slope below registered floor")

        heldout_scalars = {
            regime: torch.from_numpy(
                parent_arrays[f"{arm}__{regime}__frontend__scalar"]
            ).float()
            for regime in REGIMES
        }
        weight = checkpoint["scalar_embedding"]["weight"].float()
        bias = checkpoint["scalar_embedding"]["bias"].float()
        transport = embedding_transport_errors(
            [training_scalar, *heldout_scalars.values()],
            weight,
            bias,
            fit["alpha"],
            fit["beta"],
        )
        transport_pass = (
            transport["maximum_float32_absolute_error"]
            <= config.transport_tolerance
            and transport["maximum_float64_absolute_error"]
            <= config.transport_tolerance
        )
        cuts: dict[str, Any] = {}
        full_parent: dict[str, Any] = {}
        shift_laws: dict[str, Any] = {}
        for regime in REGIMES:
            exact = torch.from_numpy(
                parent_arrays[f"{arm}__{regime}__exact_cosine"]
            ).float()
            fiber_id = torch.from_numpy(
                parent_arrays[f"{arm}__{regime}__fiber_id"]
            ).long()
            target_posterior = torch.from_numpy(
                parent_arrays[f"{arm}__{regime}__target_posterior"]
            ).float()
            parent_front = detail["arms"][arm]["analysis"]["cuts"][
                "frontend"
            ][regime]
            record, canonical = _canonical_front_record(
                heldout_scalars[regime],
                exact,
                fiber_id,
                target_posterior,
                parent_front,
                task,
                config,
                fit["alpha"],
                fit["beta"],
            )
            cuts[regime] = record
            full_parent[regime] = detail["arms"][arm]["analysis"]["cuts"][
                "full"
            ][regime]
            shift_fit = _best_shift_fit(heldout_scalars[regime], exact)
            shift_fit["alpha_delta_from_training"] = (
                shift_fit["alpha"] - fit["alpha"]
            )
            shift_fit["beta_delta_from_training"] = (
                shift_fit["beta"] - fit["beta"]
            )
            shift_laws[regime] = shift_fit
            arrays[f"{arm}__{regime}__canonical_scalar"] = canonical.numpy()
            arrays[f"{arm}__{regime}__posterior"] = (
                joint.interval_posterior_unclipped(canonical, task.phase_bins)
                .cpu()
                .numpy()
            )
        front_joint = all(cuts[regime]["endpoint_pass"] for regime in REGIMES)
        full_joint = all(
            full_parent[regime]["endpoint_pass"] for regime in REGIMES
        )
        joint_pass = bool(front_joint and full_joint)
        arrays[f"{arm}__training_scalar"] = training_scalar.numpy()
        arrays[f"{arm}__training_fit_target"] = fit_target.numpy()
        arms[arm] = {
            "training_fit": fit,
            "slope_valid": slope_valid,
            "transport": transport,
            "transport_pass": transport_pass,
            "canonical_front": cuts,
            "shift_best_affine_laws": shift_laws,
            "parent_full_depth": full_parent,
            "canonical_front_joint_pass": front_joint,
            "parent_full_depth_joint_pass": full_joint,
            "joint_seed_pass": joint_pass,
            "checkpoint": checkpoint_record,
        }
        del checkpoint
        gc.collect()

    derived_path = cell / "derived_diagnostics.npz"
    _write_npz(derived_path, arrays)
    with np.load(derived_path, allow_pickle=False) as reloaded:
        derived_reload = bool(
            set(reloaded.files) == set(arrays)
            and all(np.array_equal(reloaded[name], arrays[name]) for name in arrays)
        )
    finite = full.joint.source._finite(arms) and all(
        np.isfinite(value).all() for value in arrays.values()
    )
    validity = bool(
        detail["gates"]["validity"]
        and all(record["slope_valid"] for record in arms.values())
        and all(record["transport_pass"] for record in arms.values())
        and derived_reload
        and finite
    )
    result_path = cell / "result.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-affine-gauge-{preset}-seed{seed}",
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_scientific_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "completed_at": _utc_now(),
        "preset": preset,
        "condition": CONDITION,
        "seed": seed,
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "implementation_sha256": implementation,
        "implementation_sources": dict(sources),
        "scientific_fingerprint": _fingerprint(config, implementation, detail),
        "parent": {
            "campaign": str(Path(config.parent_root) / "campaign_results.json"),
            "campaign_sha256": PARENT_CAMPAIGN_SHA256,
            "result": detail["artifacts"]["result"],
            "result_sha256": _sha256(Path(detail["artifacts"]["result"])),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": detail["artifacts"]["diagnostics_sha256"],
            "training_data_sha256": training_hash,
            "minibatch_schedule_sha256": batch_hash,
            "target_permutation_sha256": detail["target_permutation_sha256"],
        },
        "optimizer_steps": 0,
        "trained_parameters": 0,
        "arms": arms,
        "gates": {
            "physical_true_canonical_front_joint_pass": arms["physical_true"][
                "canonical_front_joint_pass"
            ],
            "physical_true_joint_seed_pass": arms["physical_true"][
                "joint_seed_pass"
            ],
            "pair_shuffled_canonical_front_joint_pass": arms["pair_shuffled"][
                "canonical_front_joint_pass"
            ],
            "pair_shuffled_joint_seed_pass": arms["pair_shuffled"][
                "joint_seed_pass"
            ],
            "transport_identity": all(
                record["transport_pass"] for record in arms.values()
            ),
            "derived_diagnostics_reload": derived_reload,
            "finite": finite,
            "validity": validity,
        },
        "wall_seconds": time.perf_counter() - started,
        "artifacts": {
            "result": str(result_path),
            "derived_diagnostics": str(derived_path),
            "derived_diagnostics_sha256": _sha256(derived_path),
        },
    }
    _write_json(result_path, result)
    return result


def classify_aggregates(
    strata: Mapping[str, Mapping[str, Any]], valid: bool
) -> tuple[str, bool]:
    if not valid:
        return "invalid", False
    if not all(record["shuffled_specificity_gate"] for record in strata.values()):
        return "specificity_control_failed", False
    joint_passed = [preset for preset in PRESETS if strata[preset]["family_gate"]]
    if len(joint_passed) == len(PRESETS):
        return "affine_gauge_transport_sufficient", True
    if joint_passed:
        return "architecture_conditional_affine_gauge_transport", False
    if all(record["canonical_front_family_gate"] for record in strata.values()):
        return "front_gauge_repaired_continuation_insufficient", False
    return "support_relative_affine_gauge_insufficient", False


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: AffineGaugeConfig
) -> dict[str, Any]:
    strata: dict[str, Any] = {}
    for preset in config.presets:
        selected = sorted(
            (detail for detail in details if detail["preset"] == preset),
            key=lambda detail: detail["seed"],
        )
        if len(selected) != len(config.seeds):
            raise ValueError(f"incomplete stratum {preset}")
        physical_front = sum(
            bool(detail["gates"]["physical_true_canonical_front_joint_pass"])
            for detail in selected
        )
        physical_joint = sum(
            bool(detail["gates"]["physical_true_joint_seed_pass"])
            for detail in selected
        )
        shuffled_joint = sum(
            bool(detail["gates"]["pair_shuffled_joint_seed_pass"])
            for detail in selected
        )
        specificity = shuffled_joint <= config.shuffled_seed_pass_ceiling
        strata[preset] = {
            "valid_count": sum(
                bool(detail["gates"]["validity"]) for detail in selected
            ),
            "physical_true_canonical_front_pass_count": physical_front,
            "physical_true_joint_pass_count": physical_joint,
            "physical_true_joint_pass_by_seed": {
                str(detail["seed"]): bool(
                    detail["gates"]["physical_true_joint_seed_pass"]
                )
                for detail in selected
            },
            "pair_shuffled_joint_pass_count": shuffled_joint,
            "pair_shuffled_joint_pass_by_seed": {
                str(detail["seed"]): bool(
                    detail["gates"]["pair_shuffled_joint_seed_pass"]
                )
                for detail in selected
            },
            "canonical_front_family_gate": (
                physical_front >= config.required_seed_passes
            ),
            "shuffled_specificity_gate": specificity,
            "family_gate": bool(
                physical_joint >= config.required_seed_passes and specificity
            ),
        }
    complete = bool(
        not config.allow_underpowered
        and len(details) == len(config.presets) * len(config.seeds)
    )
    valid = bool(
        all(detail["gates"]["validity"] for detail in details)
        and (complete or config.allow_underpowered)
    )
    if config.allow_underpowered:
        classification = "systems_lifecycle_only_not_scientific_evidence"
        primary = False
    else:
        classification, primary = classify_aggregates(strata, valid)
    return {
        "valid": valid,
        "complete_primary_population": complete,
        "classification": classification,
        "primary_hypothesis_pass": primary,
        "strata": strata,
        "preregistered_gate": {
            "required_seed_passes": config.required_seed_passes,
            "shuffled_seed_pass_ceiling": config.shuffled_seed_pass_ceiling,
            "cosine_minimum": config.cosine_minimum,
            "branch_accuracy_maximum": config.branch_accuracy_maximum,
            "conditional_log_loss_gain_maximum": (
                config.conditional_log_loss_gain_maximum
            ),
            "transport_tolerance": config.transport_tolerance,
            "regimes": list(REGIMES),
        },
    }


def _existing_detail(
    parent_detail: Mapping[str, Any],
    config: AffineGaugeConfig,
    output_root: Path,
    implementation: str,
) -> dict[str, Any] | None:
    path = _cell_directory(
        output_root, str(parent_detail["preset"]), int(parent_detail["seed"])
    ) / "result.json"
    if not path.is_file():
        return None
    try:
        detail = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    diagnostics = Path(
        str(detail.get("artifacts", {}).get("derived_diagnostics", ""))
    )
    if (
        detail.get("schema_version") != SCHEMA_VERSION
        or detail.get("status") != "completed"
        or detail.get("scientific_fingerprint")
        != _fingerprint(config, implementation, parent_detail)
        or detail.get("gates", {}).get("validity") is not True
        or not diagnostics.is_file()
        or _sha256(diagnostics)
        != detail.get("artifacts", {}).get("derived_diagnostics_sha256")
    ):
        return None
    return detail


def _campaign_fingerprint(
    config: AffineGaugeConfig,
    task: CircleTaskConfig,
    implementation: str,
    parent_result_manifest: str,
) -> str:
    return _json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "configuration": _json_config(config),
            "task_config": asdict(task),
            "implementation_sha256": implementation,
            "parent_campaign_sha256": PARENT_CAMPAIGN_SHA256,
            "parent_selected_result_manifest_sha256": parent_result_manifest,
        }
    )


def run_campaign(config: AffineGaugeConfig, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    sources = _implementation_sources()
    implementation = _implementation_digest(sources)
    parent_path = Path(config.parent_root) / "campaign_results.json"
    parent_campaign, all_parent_details = parent._campaign(parent_path)
    task = CircleTaskConfig(**parent_campaign["task_config"])
    selected_parent = [
        detail
        for detail in all_parent_details
        if detail["preset"] in config.presets and detail["seed"] in config.seeds
    ]
    parent_result_paths = [Path(detail["artifacts"]["result"]) for detail in selected_parent]
    parent_manifest = _manifest_sha256(parent_result_paths)
    fingerprint = _campaign_fingerprint(
        config, task, implementation, parent_manifest
    )
    campaign_path = output_root / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "completed"
            and existing.get("campaign_fingerprint") == fingerprint
            and all(
                _existing_detail(detail, config, output_root, implementation)
                is not None
                for detail in selected_parent
            )
        ):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing

    details = []
    reused = 0
    for parent_detail in selected_parent:
        existing = _existing_detail(
            parent_detail, config, output_root, implementation
        )
        if existing is not None:
            details.append(existing)
            reused += 1
            continue
        details.append(
            analyze_cell(
                parent_detail,
                task,
                config,
                output_root,
                implementation,
                sources,
            )
        )
    complete = len(details) == len(selected_parent)
    result_paths = [Path(detail["artifacts"]["result"]) for detail in details]
    derived_paths = [
        Path(detail["artifacts"]["derived_diagnostics"]) for detail in details
    ]
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if complete else "partial",
        "completed_at": _utc_now(),
        "evidence_role": (
            "systems_lifecycle_only_not_scientific_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "configuration": _json_config(config),
        "task_config": asdict(task),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "implementation_sha256": implementation,
        "implementation_sources": sources,
        "campaign_fingerprint": fingerprint,
        "parent": {
            "campaign": str(parent_path),
            "campaign_sha256": PARENT_CAMPAIGN_SHA256,
            "campaign_fingerprint": parent_campaign["campaign_fingerprint"],
            "selected_result_manifest_sha256": parent_manifest,
            "classification": parent_campaign["aggregates"]["classification"],
        },
        "summary": {
            "requested_cells": len(selected_parent),
            "completed_cells": len(details),
            "failed_cells": len(selected_parent) - len(details),
            "reused_cells": reused,
            "analyzed_arms": 2 * len(details),
            "optimizer_steps": 0,
            "trained_parameters": 0,
        },
        "aggregates": aggregate_details(details, config) if complete else {},
        "result_manifest_sha256": (
            _manifest_sha256(result_paths) if result_paths else None
        ),
        "derived_diagnostics_manifest_sha256": (
            _manifest_sha256(derived_paths) if derived_paths else None
        ),
        "method_boundaries": [
            "The full-interface outcomes and an exploratory composition self-fit were known before registration.",
            "Only the exact sealed training cohort fits alpha and beta.",
            "No optimizer step, model gradient, parameter update, checkpoint selection, or held-out refit occurs.",
            "The inverse scalar-embedding transport is an algebraic diagnostic, not a repaired model artifact.",
            "Parent full-depth endpoints remain byte-identical and cannot be relabeled.",
        ],
    }
    _write_json(campaign_path, bundle)
    return bundle


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _comma_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", default=",".join(PRESETS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--encoder-batch-size", type=int, default=512)
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_affine_gauge_transport/"
            "20260811_d6_d10_registered"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shakedown:
        args.presets = "d6"
        args.seeds = "7"
        args.allow_underpowered = True
    config = AffineGaugeConfig(
        presets=_comma_strings(args.presets),
        seeds=_comma_ints(args.seeds),
        encoder_batch_size=args.encoder_batch_size,
        required_seed_passes=1 if args.shakedown else 4,
        allow_underpowered=args.allow_underpowered,
    )
    result = run_campaign(config, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "classification": result.get("aggregates", {}).get(
                    "classification", "partial"
                ),
                "summary": result["summary"],
                "output": str(args.output / "campaign_results.json"),
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
