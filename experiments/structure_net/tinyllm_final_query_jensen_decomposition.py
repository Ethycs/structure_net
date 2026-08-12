#!/usr/bin/env python3
"""Separate generic Jensen gain from final-LN geometry at a raw final query."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_final_query_task_kernel as posterior
import experiments.structure_net.tinyllm_task_relative_activation_barycenter as parent
from experiments.structure_net.tinyllm_predictive_circle import _resolve_device


SCHEMA_VERSION = "nal.tinyllm-final-query-jensen-decomposition.v1"
HYPOTHESIS_ID = "tinyllm-final-query-jensen-decomposition-v1"
EVIDENCE_ROLE = "registered_post_outcome_frozen_final_query_diagnostic"
REGIMES = posterior.REGIMES
DATASET_SEEDS = posterior.DATASET_SEEDS
FINAL_CUT = posterior.FINAL_CUT
PREREGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-10_tinyllm-final-query-jensen-decomposition-preregistration.md"
)
PREREGISTRATION_SHA256 = (
    "0c8e4d871637007405e8e57868a24a02962f52b04a2e1c87af72de0f71b5df69"
)
POSTERIOR_RUNNER_PATH = Path(
    "experiments/structure_net/tinyllm_final_query_task_kernel.py"
)
POSTERIOR_RUNNER_SHA256 = (
    "52b59f96e53e76c1794e9d2d251760a76eb2300f8e64e70dbe165090a5899895"
)
POSTERIOR_CAMPAIGN_PATH = Path(
    "data/experiments/tinyllm_final_query_task_kernel/"
    "20260810_d8_seed7_preregistered/campaign_results.json"
)
POSTERIOR_CAMPAIGN_SHA256 = (
    "93d9e22d766aa56943f0bd0c41b31ed25dc592e97ae0667863f3963588c38cde"
)
POSTERIOR_DIAGNOSTICS_PATH = Path(
    "data/experiments/tinyllm_final_query_task_kernel/"
    "20260810_d8_seed7_preregistered/diagnostics.npz"
)
POSTERIOR_DIAGNOSTICS_SHA256 = (
    "ad1798b960375503381cb59725dfb84db9d823f488ab8073fea178d0399e0974"
)
CHECKPOINT_PATH = posterior.CHECKPOINT_PATH
CHECKPOINT_SHA256 = posterior.CHECKPOINT_SHA256


@dataclass(frozen=True)
class FinalQueryJensenConfig:
    regimes: tuple[str, ...] = REGIMES
    fiber_pairs: int = 512
    batch_size: int = 64
    target_pair_tolerance: float = 1e-12
    replay_tolerance: float = 2e-6
    accounting_tolerance: float = 1e-10
    jensen_tolerance: float = 1e-10
    material_gain_floor: float = 1e-4
    jensen_coverage_floor: float = 0.90
    near_complete_remainder_ratio: float = 0.10
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.regimes or set(self.regimes).difference(REGIMES):
            raise ValueError(f"regimes must be drawn from {REGIMES}")
        if self.fiber_pairs < 2 or self.fiber_pairs % 2:
            raise ValueError("fiber_pairs must be positive and even")
        if self.batch_size < 4 or self.batch_size % 4:
            raise ValueError("batch_size must be divisible by four")
        if any(
            value <= 0.0
            for value in (
                self.target_pair_tolerance,
                self.replay_tolerance,
                self.accounting_tolerance,
                self.jensen_tolerance,
                self.material_gain_floor,
            )
        ):
            raise ValueError("numerical tolerances must be positive")
        if not 0.0 < self.jensen_coverage_floor <= 1.0:
            raise ValueError("jensen_coverage_floor must be in (0, 1]")
        if not 0.0 < self.near_complete_remainder_ratio <= 1.0:
            raise ValueError("near_complete_remainder_ratio must be in (0, 1]")
        if not self.allow_underpowered and (
            self.regimes != REGIMES
            or self.fiber_pairs != 512
            or self.batch_size != 64
        ):
            raise ValueError("primary campaign axes are frozen")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _json_config(config: FinalQueryJensenConfig) -> dict[str, Any]:
    value = asdict(config)
    value["regimes"] = list(config.regimes)
    return value


def _source_digests() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "preregistration": PREREGISTRATION_PATH,
        "posterior_runner": POSTERIOR_RUNNER_PATH,
        "parent_runner": posterior.PARENT_RUNNER_PATH,
        "tinyllm_model": Path("src/structure_net/components/models/tinyllm_model.py"),
        "circle_source": Path("experiments/structure_net/tinyllm_predictive_circle.py"),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _implementation_digest(source_digests: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(source_digests), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _validate_sources(
    config: FinalQueryJensenConfig,
) -> tuple[parent.CircleTaskConfig, dict[str, str], dict[str, Any]]:
    required = {
        PREREGISTRATION_PATH: PREREGISTRATION_SHA256,
        POSTERIOR_RUNNER_PATH: POSTERIOR_RUNNER_SHA256,
        POSTERIOR_CAMPAIGN_PATH: POSTERIOR_CAMPAIGN_SHA256,
        POSTERIOR_DIAGNOSTICS_PATH: POSTERIOR_DIAGNOSTICS_SHA256,
        posterior.PARENT_CAMPAIGN_PATH: posterior.PARENT_CAMPAIGN_SHA256,
        posterior.PARENT_D8_RESULT_PATH: posterior.PARENT_D8_RESULT_SHA256,
        posterior.PARENT_D8_DIAGNOSTICS_PATH: posterior.PARENT_D8_DIAGNOSTICS_SHA256,
        CHECKPOINT_PATH: CHECKPOINT_SHA256,
    }
    for path, expected in required.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"source artifact changed: {path}")
    prior = json.loads(POSTERIOR_CAMPAIGN_PATH.read_text(encoding="utf-8"))
    if (
        prior.get("schema_version") != posterior.SCHEMA_VERSION
        or prior.get("hypothesis_id") != posterior.HYPOTHESIS_ID
        or prior.get("status") != "completed"
        or prior.get("gates", {}).get("validity") is not True
        or prior.get("classification")
        != "task_rowspace_contains_material_fiber_chord"
        or prior.get("artifacts", {}).get("diagnostics_sha256")
        != POSTERIOR_DIAGNOSTICS_SHA256
    ):
        raise ValueError("final-query task-kernel source contract changed")
    source_config = posterior.FinalQueryKernelConfig(
        regimes=config.regimes,
        fiber_pairs=config.fiber_pairs,
        batch_size=config.batch_size,
        jacobian_batch_pairs=min(32, config.fiber_pairs),
        finite_difference_pairs=min(32, config.fiber_pairs),
        device=config.device,
        allow_underpowered=True,
    )
    task, hashes, detail = posterior._validate_parent(source_config)
    hashes.update(
        {
            "posterior_runner": POSTERIOR_RUNNER_SHA256,
            "posterior_campaign": POSTERIOR_CAMPAIGN_SHA256,
            "posterior_diagnostics": POSTERIOR_DIAGNOSTICS_SHA256,
        }
    )
    return task, hashes, detail


def _soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets.double() * torch.log_softmax(logits.double(), -1)).sum(-1)


def _probability_cross_entropy(
    probabilities: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return -(
        targets.double() * probabilities.double().clamp_min(1e-300).log()
    ).sum(-1)


def decompose_pair_logits(
    endpoint_logits: torch.Tensor,
    activation_midpoint_logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the exact pairwise CE accounting terms for fixed targets."""
    if endpoint_logits.ndim != 3 or endpoint_logits.shape[1] != 2:
        raise ValueError("endpoint_logits must have shape [pairs, 2, classes]")
    if activation_midpoint_logits.shape != endpoint_logits[:, 0].shape:
        raise ValueError("activation midpoint shape mismatch")
    if targets.shape != activation_midpoint_logits.shape:
        raise ValueError("target shape mismatch")
    logit_midpoint = endpoint_logits.double().mean(1)
    endpoint_probabilities = torch.softmax(endpoint_logits.double(), -1)
    probability_midpoint = endpoint_probabilities.mean(1)
    endpoint_ce = _soft_cross_entropy(
        endpoint_logits.double(), targets[:, None, :].expand_as(endpoint_logits)
    ).mean(1)
    logit_midpoint_ce = _soft_cross_entropy(logit_midpoint, targets)
    activation_midpoint_ce = _soft_cross_entropy(
        activation_midpoint_logits, targets
    )
    probability_midpoint_ce = _probability_cross_entropy(
        probability_midpoint, targets
    )
    jensen_gain = endpoint_ce - logit_midpoint_ce
    nonlinear_remainder = activation_midpoint_ce - logit_midpoint_ce
    observed_gain = endpoint_ce - activation_midpoint_ce
    return {
        "endpoint_ce": endpoint_ce,
        "logit_midpoint_ce": logit_midpoint_ce,
        "activation_midpoint_ce": activation_midpoint_ce,
        "probability_midpoint_ce": probability_midpoint_ce,
        "jensen_gain": jensen_gain,
        "nonlinear_remainder": nonlinear_remainder,
        "observed_gain": observed_gain,
        "accounting_residual": observed_gain - (jensen_gain - nonlinear_remainder),
        "logit_midpoint": logit_midpoint,
        "probability_midpoint": probability_midpoint,
    }


def _quantile(value: torch.Tensor, q: float) -> float:
    return float(torch.quantile(value.double(), q))


def _aggregate(parts: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    endpoint = float(parts["endpoint_ce"].mean())
    logit = float(parts["logit_midpoint_ce"].mean())
    activation = float(parts["activation_midpoint_ce"].mean())
    probability = float(parts["probability_midpoint_ce"].mean())
    jensen = endpoint - logit
    nonlinear = activation - logit
    observed = endpoint - activation
    return {
        "endpoint_cross_entropy": endpoint,
        "logit_midpoint_cross_entropy": logit,
        "activation_midpoint_cross_entropy": activation,
        "probability_midpoint_cross_entropy": probability,
        "jensen_gain": jensen,
        "nonlinear_remainder": nonlinear,
        "observed_activation_gain": observed,
        "probability_jensen_gain": endpoint - probability,
        "jensen_over_observed_gain": (
            jensen / observed if observed > 0.0 else None
        ),
        "nonlinear_over_jensen": nonlinear / max(abs(jensen), 1e-300),
        "positive_observed_gain_pair_fraction": float(
            (parts["observed_gain"] > 0.0).double().mean()
        ),
        "activation_beats_logit_pair_fraction": float(
            (parts["nonlinear_remainder"] < 0.0).double().mean()
        ),
        "minimum_pair_jensen_gain": float(parts["jensen_gain"].min()),
        "maximum_absolute_accounting_residual": float(
            parts["accounting_residual"].abs().max()
        ),
    }


@torch.no_grad()
def analyze_regime(
    model: parent.TinyLLMModel,
    task: parent.CircleTaskConfig,
    regime: str,
    config: FinalQueryJensenConfig,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cohort = parent.generate_exact_task_fibers(
        task,
        pair_count=config.fiber_pairs,
        seed=DATASET_SEEDS[regime],
        regime=regime,
    )
    if cohort.contract.get("pass") is not True:
        raise ValueError(f"cohort contract failed for {regime}")
    answer_ids = torch.tensor(
        task.answer_token_ids, dtype=torch.long, device=device
    )
    query, captured_baseline, capture_error = posterior._capture_final_queries(
        model, cohort, answer_ids, config.batch_size
    )
    shaped_query = query.reshape(config.fiber_pairs, 2, -1)
    target_rows = cohort.cosine_targets.to(device).double().reshape(
        config.fiber_pairs, 2, -1
    )
    target_pair_error = float(
        torch.max(torch.abs(target_rows[:, 0] - target_rows[:, 1]))
    )
    targets = target_rows[:, 0]

    endpoint_posterior, endpoint_logits = posterior._final_outputs(
        model, query, answer_ids
    )
    endpoint_logits = endpoint_logits.reshape(config.fiber_pairs, 2, -1)
    endpoint_posterior = endpoint_posterior.reshape(config.fiber_pairs, 2, -1)
    activation_midpoint = shaped_query.mean(1)
    activation_posterior, activation_logits = posterior._final_outputs(
        model, activation_midpoint.to(query.dtype), answer_ids
    )
    natural = decompose_pair_logits(endpoint_logits, activation_logits, targets)

    partner = torch.arange(config.fiber_pairs, device=device) ^ 1
    control_endpoint_logits = endpoint_logits.index_select(0, partner)
    control_midpoint = activation_midpoint.index_select(0, partner)
    control_posterior, control_activation_logits = posterior._final_outputs(
        model, control_midpoint.to(query.dtype), answer_ids
    )
    control = decompose_pair_logits(
        control_endpoint_logits, control_activation_logits, targets
    )

    natural_logit_posterior = torch.softmax(natural["logit_midpoint"], -1)
    posterior_js = parent._jensen_shannon_rows(
        activation_posterior, natural_logit_posterior
    )
    logit_remainder_norm = torch.linalg.vector_norm(
        activation_logits.double() - natural["logit_midpoint"], dim=1
    )
    endpoint_chord_norm = torch.linalg.vector_norm(
        endpoint_logits[:, 0].double() - endpoint_logits[:, 1].double(), dim=1
    )
    remainder_ratio = logit_remainder_norm / endpoint_chord_norm.clamp_min(1e-12)

    parent_replay_required = config.fiber_pairs == 512
    baseline_replay_error: Optional[float] = None
    capture_replay_error: Optional[float] = None
    correct_replay_error: Optional[float] = None
    semantic_replay_error: Optional[float] = None
    maximum_replay_error = capture_error
    if parent_replay_required:
        with np.load(posterior.PARENT_D8_DIAGNOSTICS_PATH) as source_arrays:
            prefix = f"cosine_interval__{regime}"
            stored_baseline = torch.from_numpy(
                source_arrays[f"{prefix}__baseline_posterior"]
            ).double()
            stored_correct = torch.from_numpy(
                source_arrays[f"{prefix}__{FINAL_CUT}__correct_posterior"]
            ).double()
            stored_semantic = torch.from_numpy(
                source_arrays[f"{prefix}__{FINAL_CUT}__semantic_posterior"]
            ).double()
        baseline_replay_error = float(
            torch.max(torch.abs(endpoint_posterior.reshape_as(stored_baseline).cpu() - stored_baseline))
        )
        capture_replay_error = float(
            torch.max(torch.abs(captured_baseline.cpu() - stored_baseline))
        )
        correct_replay_error = float(
            torch.max(
                torch.abs(
                    activation_posterior.repeat_interleave(2, dim=0).cpu()
                    - stored_correct
                )
            )
        )
        semantic_replay_error = float(
            torch.max(
                torch.abs(
                    control_posterior.repeat_interleave(2, dim=0).cpu()
                    - stored_semantic
                )
            )
        )
        maximum_replay_error = max(
            capture_error,
            baseline_replay_error,
            capture_replay_error,
            correct_replay_error,
            semantic_replay_error,
        )

    aggregate = _aggregate(natural)
    control_aggregate = _aggregate(control)
    aggregate.update(
        {
            "mean_midpoint_posterior_js": float(posterior_js.mean()),
            "p95_midpoint_posterior_js": _quantile(posterior_js, 0.95),
            "mean_logit_remainder_over_endpoint_chord": float(
                remainder_ratio.mean()
            ),
            "p95_logit_remainder_over_endpoint_chord": _quantile(
                remainder_ratio, 0.95
            ),
        }
    )
    target_contract = target_pair_error <= config.target_pair_tolerance
    replay_contract = maximum_replay_error <= config.replay_tolerance
    accounting_contract = bool(
        aggregate["maximum_absolute_accounting_residual"]
        <= config.accounting_tolerance
        and abs(
            aggregate["observed_activation_gain"]
            - (
                aggregate["jensen_gain"]
                - aggregate["nonlinear_remainder"]
            )
        )
        <= config.accounting_tolerance
    )
    jensen_contract = bool(
        aggregate["minimum_pair_jensen_gain"] >= -config.jensen_tolerance
    )
    control_jensen_contract = bool(
        control_aggregate["minimum_pair_jensen_gain"]
        >= -config.jensen_tolerance
    )
    finite_contract = bool(
        all(torch.isfinite(value).all() for value in natural.values())
        and all(torch.isfinite(value).all() for value in control.values())
        and torch.isfinite(posterior_js).all()
        and torch.isfinite(remainder_ratio).all()
    )
    validity = bool(
        cohort.contract.get("pass") is True
        and target_contract
        and replay_contract
        and accounting_contract
        and jensen_contract
        and control_jensen_contract
        and finite_contract
    )
    material_gain = bool(
        aggregate["observed_activation_gain"] >= config.material_gain_floor
    )
    jensen_sufficient = bool(
        aggregate["jensen_gain"]
        >= config.jensen_coverage_floor
        * aggregate["observed_activation_gain"]
    )
    near_complete = bool(
        abs(aggregate["nonlinear_remainder"])
        <= config.near_complete_remainder_ratio
        * max(abs(aggregate["jensen_gain"]), 1e-300)
    )
    primary = bool(validity and material_gain and jensen_sufficient)
    result = {
        "regime": regime,
        "cohort_contract": dict(cohort.contract),
        "natural": aggregate,
        "semantic_reassignment_control": control_aggregate,
        "replay": {
            "parent_replay_required": parent_replay_required,
            "maximum_capture_error": capture_error,
            "maximum_baseline_replay_error": baseline_replay_error,
            "maximum_capture_posterior_replay_error": capture_replay_error,
            "maximum_correct_barycenter_replay_error": correct_replay_error,
            "maximum_semantic_reassignment_replay_error": semantic_replay_error,
            "maximum_replay_error": maximum_replay_error,
        },
        "contracts": {
            "maximum_target_pair_error": target_pair_error,
            "target_pair_identity": target_contract,
            "replay": replay_contract,
            "accounting_identity": accounting_contract,
            "pairwise_logit_jensen": jensen_contract,
            "control_pairwise_logit_jensen": control_jensen_contract,
            "finite": finite_contract,
        },
        "gates": {
            "validity": validity,
            "material_activation_gain": material_gain,
            "generic_jensen_sufficient": jensen_sufficient,
            "near_complete_jensen": near_complete,
            "primary_regime_gate": primary,
        },
    }
    arrays: dict[str, np.ndarray] = {}
    for prefix, parts in (("natural", natural), ("control", control)):
        for name in (
            "endpoint_ce",
            "logit_midpoint_ce",
            "activation_midpoint_ce",
            "probability_midpoint_ce",
            "jensen_gain",
            "nonlinear_remainder",
            "observed_gain",
            "accounting_residual",
        ):
            arrays[f"{prefix}__{name}"] = parts[name].detach().cpu().numpy()
    arrays["midpoint_posterior_js"] = posterior_js.detach().cpu().numpy()
    arrays["logit_remainder_over_endpoint_chord"] = (
        remainder_ratio.detach().cpu().numpy()
    )
    return result, arrays


def _fingerprint(
    config: FinalQueryJensenConfig,
    implementation: str,
    source_hashes: Mapping[str, str],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": _json_config(config),
        "implementation_sha256": implementation,
        "source_hashes": dict(source_hashes),
        "dataset_seeds": DATASET_SEEDS,
        "final_cut": FINAL_CUT,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _classification(
    config: FinalQueryJensenConfig,
    valid: bool,
    primary: bool,
    regimes: Mapping[str, Mapping[str, Any]],
) -> str:
    if config.allow_underpowered:
        return "systems_lifecycle_only_not_quality_evidence"
    if not valid:
        return "invalid_final_query_jensen_decomposition"
    if any(
        not item["gates"]["material_activation_gain"]
        for item in regimes.values()
    ):
        return "no_material_activation_barycenter_gain"
    if primary:
        if all(
            item["gates"]["near_complete_jensen"]
            for item in regimes.values()
        ):
            return "generic_jensen_near_complete"
        return "generic_jensen_sufficient_with_layernorm_modulation"
    return "activation_geometry_materially_assists_jensen"


def run_campaign(
    config: FinalQueryJensenConfig, output: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    source_digests = _source_digests()
    if source_digests["preregistration"] != PREREGISTRATION_SHA256:
        raise ValueError("preregistration changed")
    if source_digests["posterior_runner"] != POSTERIOR_RUNNER_SHA256:
        raise ValueError("posterior runner changed")
    implementation = _implementation_digest(source_digests)
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "completed"
            and existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("configuration") == _json_config(config)
            and existing.get("implementation_sha256") == implementation
            and existing.get("scientific_fingerprint")
            == _fingerprint(config, implementation, existing.get("source_hashes", {}))
            and Path(existing["artifacts"]["diagnostics"]).is_file()
            and _sha256(Path(existing["artifacts"]["diagnostics"]))
            == existing["artifacts"]["diagnostics_sha256"]
        ):
            print("campaign already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed campaign {campaign_path}")

    task, source_hashes, parent_detail = _validate_sources(config)
    fingerprint = _fingerprint(config, implementation, source_hashes)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model, initial_system_state = parent._load_model(
        CHECKPOINT_PATH, "d8", "cosine_interval", device
    )
    initial_model_state = parent._state_digest(model)
    results: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for regime in config.regimes:
        result, regime_arrays = analyze_regime(model, task, regime, config, device)
        results[regime] = result
        arrays.update(
            {f"{regime}__{name}": value for name, value in regime_arrays.items()}
        )
    final_model_state = parent._state_digest(model)
    final_system_state = parent._model_system_state(model)
    state_unchanged = bool(
        initial_model_state == final_model_state
        and initial_system_state == final_system_state
    )
    valid = bool(
        state_unchanged
        and all(item["gates"]["validity"] for item in results.values())
    )
    primary = bool(
        not config.allow_underpowered
        and set(config.regimes) == set(REGIMES)
        and config.fiber_pairs == 512
        and valid
        and all(item["gates"]["primary_regime_gate"] for item in results.values())
    )
    classification = _classification(config, valid, primary, results)
    diagnostics_path = output / "diagnostics.npz"
    _write_npz(diagnostics_path, arrays)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "completed_at": _utc_now(),
        "configuration": _json_config(config),
        "scientific_fingerprint": fingerprint,
        "implementation_sha256": implementation,
        "source_digests": source_digests,
        "source_hashes": source_hashes,
        "task_config": asdict(task),
        "parent_summary": {
            "classification": parent_detail.get("classification"),
            "valid_d8_null": parent_detail["gates"]["validity"],
            "mature_causal_front": parent_detail["summary"]["mature_causal_front"],
        },
        "regimes": results,
        "state_record": {
            "initial_model_state_sha256": initial_model_state,
            "final_model_state_sha256": final_model_state,
            "initial_system_state_sha256": initial_system_state,
            "final_system_state_sha256": final_system_state,
            "unchanged": state_unchanged,
        },
        "gates": {
            "state_unchanged": state_unchanged,
            "validity": valid,
            "primary_hypothesis_pass": primary,
        },
        "classification": classification,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "summary": {
            "requested": 1,
            "scheduled": 1,
            "completed": 1,
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": 0,
            "trained_models": 0,
            "trained_probes": 0,
            "fitted_carriers": 0,
            "fitted_decoders": 0,
            "analyzed_pairs": len(config.regimes) * config.fiber_pairs,
        },
        "artifacts": {
            "campaign": str(campaign_path),
            "diagnostics": str(diagnostics_path),
            "diagnostics_sha256": _sha256(diagnostics_path),
        },
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "This is a registered post-outcome diagnostic, not fresh confirmation.",
            "One retained raw d8 seed-7 checkpoint is evaluated; population prevalence is not tested.",
            "Only the final post-MLP query and native final-LN/answer-head map are decomposed.",
            "The Jensen term is a generic convexity effect and is not quotient evidence.",
            "No model, head, probe, observer, carrier, threshold, or decoder is fit or trained.",
        ],
    }
    if not _finite(campaign):
        raise ValueError("non-finite final-query Jensen campaign")
    _write_json(campaign_path, campaign)
    return campaign


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_final_query_jensen_decomposition/"
            "20260810_d8_seed7_registered"
        ),
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--regimes", type=_strings, default=REGIMES)
    parser.add_argument("--fiber-pairs", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args(argv)
    config = FinalQueryJensenConfig(
        regimes=args.regimes,
        fiber_pairs=args.fiber_pairs,
        batch_size=args.batch_size,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(
        json.dumps(
            {
                "classification": campaign["classification"],
                "valid": campaign["gates"]["validity"],
                "primary_hypothesis_pass": campaign["gates"][
                    "primary_hypothesis_pass"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
