#!/usr/bin/env python3
"""Test an analytic observable scalar residual on frozen TinyLLM continuations."""

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
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import experiments.structure_net.tinyllm_fixed_gauge_writer_capacity as capacity
import experiments.structure_net.tinyllm_fixed_semantic_gauge_writer as fixed
import experiments.structure_net.tinyllm_local_metric_field_transport as group
import experiments.structure_net.tinyllm_local_task_tangent as local
import experiments.structure_net.tinyllm_nuisance_scalar_transformation_law as law
import experiments.structure_net.tinyllm_scalar_action_defect_decomposition as defect
import experiments.structure_net.tinyllm_source_task_covector_portability as source
import experiments.structure_net.tinyllm_cross_seed_causal_carrier_transport as transport
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c2-analytic-observable-residual.v1"
HYPOTHESIS_ID = "tinyllm-c2-analytic-observable-residual-v1"
SCALAR_LAW_SCHEMA = "nal.tinyllm-c2-nuisance-scalar-transformation-law.v1"
SCALAR_LAW_HYPOTHESIS = "tinyllm-c2-nuisance-scalar-transformation-law-v1"
SCALAR_LAW_CAMPAIGN_SHA256 = (
    "1e1795c931335c739a38550b85def7c4b442be39afcea76af8cd8491b1ff6589"
)
SCALAR_LAW_IMPLEMENTATION_SHA256 = (
    "e92c3894b3e8820c0242edeba29e1f893f321e7f1dc20d75583c898ce7b1526f"
)
COVECTOR_CAMPAIGN_SHA256 = (
    "fe153abd0aa1749a862095e577e6e9cf8a0f7054bade2ef3c6833bf32d7f4ac5"
)
COVECTOR_IMPLEMENTATION_SHA256 = (
    "6716b909d0c245059a1ed1310f20f4d9e56deb8c49a7d3a031972542fccb3046"
)
COVECTOR_SCHEMA = "nal.tinyllm-c2-source-task-covector-portability.v1"
COVECTOR_HYPOTHESIS = "tinyllm-c2-source-task-covector-portability-v1"
REGIMES = group.REGIMES
ARMS = ("reference", *group.GROUP_ARMS)
CONTROL_NAMES = (
    "analytic_phase_shift",
    "analytic_nuisance_shift",
    "analytic_flipped",
)


@dataclass(frozen=True)
class AnalyticObservableResidualConfig:
    scalar_law_root: str = (
        "data/experiments/tinyllm_nuisance_scalar_transformation_law/"
        "20260807_d6_existing_group_gauge_replay"
    )
    covector_root: str = (
        "data/experiments/tinyllm_source_task_covector_portability/"
        "20260807_d6_preregistered_fresh_cohort"
    )
    group_root: str = (
        "data/experiments/tinyllm_local_metric_field_transport/"
        "20260807_d6_fresh_cohort"
    )
    seeds: tuple[int, ...] = fixed.PRIMARY_SEEDS
    orbit_count: int = 64
    phase_count: int = 16
    nuisance_replicates: int = 4
    carrier_rank: int = 3
    writer_order: int = 4
    fine_step_std: float = 0.025
    coarse_step_std: float = 0.05
    coordinate_scale_floor: float = 1e-8
    gradient_denominator_floor: float = 1e-12
    replay_tolerance: float = 1e-6
    basis_gauge_replay_tolerance: float = 1e-5
    scalar_replay_tolerance: float = 1e-5
    derivative_cosine_floor: float = 0.98
    derivative_relative_l2_ceiling: float = 0.15
    signed_error_r2_floor: float = 0.50
    residual_mae_fraction_ceiling: float = 0.50
    sign_agreement_floor: float = 0.75
    sign_magnitude_floor_bins: float = 0.01
    observed_alignment_floor: float = 0.99
    observed_mean_shift_ceiling_bins: float = 0.125
    observed_p95_shift_ceiling_bins: float = 0.50
    observed_sheet_difference_ceiling: float = 0.01
    covector_r2_floor: float = 0.90
    covector_relative_l2_ceiling: float = 0.15
    covector_cosine_floor: float = 0.99
    specificity_margin_bins: float = 0.125
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary analytic-residual seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.orbit_count != 64:
            raise ValueError("the existing group cohort fixes 64 exact orbits")
        if self.phase_count != 16 or self.nuisance_replicates != 4:
            raise ValueError("the existing cohort fixes 16 phases x 4 nuisances")
        if self.carrier_rank != 3 or self.writer_order != 4:
            raise ValueError("carrier rank and writer order are fixed to three and four")
        if not 0.0 < self.fine_step_std < self.coarse_step_std:
            raise ValueError("finite-difference steps must be positive and ordered")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


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


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(law.__file__),
        Path(defect.__file__),
        Path(source.__file__),
        Path(group.__file__),
        Path(fixed.__file__),
        Path(local.__file__),
        Path(transport.__file__),
        Path(capacity.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: AnalyticObservableResidualConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_existing_artifact_underpowered_mechanistic_positive_control"
    )


def observed_carrier_pass(
    audit: Mapping[str, float], config: AnalyticObservableResidualConfig
) -> bool:
    return bool(
        audit["circular_alignment"] >= config.observed_alignment_floor
        and audit["mean_shift_bins"] <= config.observed_mean_shift_ceiling_bins
        and audit["p95_shift_bins"] <= config.observed_p95_shift_ceiling_bins
        and audit["maximum_sheet_difference"]
        <= config.observed_sheet_difference_ceiling
    )


def covector_replay_pass(
    metrics: Mapping[str, float], config: AnalyticObservableResidualConfig
) -> bool:
    return bool(
        metrics["zero_referenced_r2"] >= config.covector_r2_floor
        and metrics["relative_l2"] <= config.covector_relative_l2_ceiling
        and metrics["mean_row_cosine"] >= config.covector_cosine_floor
    )


def state_gate_summary(
    cells: Sequence[Mapping[str, Any]], specificity_margin: float
) -> dict[str, Any]:
    names = ("order4", "analytic_observable", *CONTROL_NAMES)
    means = {
        name: sum(
            float(cell["states"][name]["continuous"]["mean_moment_shift_bins"])
            for cell in cells
        )
        / len(cells)
        for name in names
    }
    analytic_pass = all(
        cell["states"]["analytic_observable"]["continuous"]["continuous_pass"]
        for cell in cells
    )
    controls = {}
    for name in CONTROL_NAMES:
        any_failure = any(
            not cell["states"][name]["continuous"]["continuous_pass"]
            for cell in cells
        )
        margin = means[name] - means["analytic_observable"]
        controls[name] = {
            "any_failure": any_failure,
            "aggregate_mean_shift_bins": means[name],
            "margin_over_analytic_bins": margin,
            "specific": bool(any_failure and margin >= specificity_margin),
        }
    return {
        "aggregate_mean_shift_bins": means,
        "analytic_all_cells_pass": analytic_pass,
        "controls": controls,
        "all_controls_specific": all(item["specific"] for item in controls.values()),
        "complete_gate": bool(
            analytic_pass and all(item["specific"] for item in controls.values())
        ),
    }


def classify_checkpoint(
    *, valid: bool, oracle_pass: bool, state_gates: Mapping[str, Any]
) -> tuple[str, bool]:
    if not valid:
        return "invalid", False
    if not oracle_pass:
        return "portable_covector_not_replicated", False
    if state_gates["complete_gate"]:
        return "analytic_observable_residual_sufficient", True
    if state_gates["analytic_all_cells_pass"]:
        return "analytic_observable_residual_nonspecific", False
    return "observable_semantic_target_not_frozen_equivalent", False


def _load_scalar_law(
    config: AnalyticObservableResidualConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path, Path]]]:
    path = Path(config.scalar_law_root) / "campaign_results.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    expected_counts = campaign.get("aggregates", {}).get("gate_counts", {})
    if (
        _sha256(path) != SCALAR_LAW_CAMPAIGN_SHA256
        or campaign.get("schema_version") != SCALAR_LAW_SCHEMA
        or campaign.get("hypothesis_id") != SCALAR_LAW_HYPOTHESIS
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != SCALAR_LAW_IMPLEMENTATION_SHA256
        or int(expected_counts.get("basis_gauge_replay_contract", -1)) != 3
        or int(expected_counts.get("local_linearization_contract", -1)) != 3
        or int(expected_counts.get("target_control_contract", -1)) != 3
        or int(campaign.get("summary", {}).get("trained_models", -1)) != 0
    ):
        raise ValueError(f"invalid nuisance-scalar source campaign {path}")
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if not set(config.seeds).issubset(entries):
        raise ValueError("nuisance-scalar campaign lacks a requested checkpoint")
    details = {}
    for seed in config.seeds:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        arrays_path = Path(entry["arrays"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        if (
            _sha256(detail_path) != entry.get("result_sha256")
            or _sha256(arrays_path) != entry.get("arrays_sha256")
            or detail.get("implementation_sha256") != SCALAR_LAW_IMPLEMENTATION_SHA256
            or detail.get("classification") != "scalar_action_dependent"
            or detail.get("gates", {}).get("basis_gauge_replay_contract") is not True
            or detail.get("gates", {}).get("local_linearization_contract") is not True
            or detail.get("gates", {}).get("target_control_contract") is not True
        ):
            raise ValueError(f"invalid nuisance-scalar source result {detail_path}")
        details[seed] = (detail, detail_path, arrays_path)
    return campaign, path, details


def _load_covectors(
    config: AnalyticObservableResidualConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path]]]:
    path = Path(config.covector_root) / "campaign_results.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if (
        _sha256(path) != COVECTOR_CAMPAIGN_SHA256
        or campaign.get("schema_version") != COVECTOR_SCHEMA
        or campaign.get("hypothesis_id") != COVECTOR_HYPOTHESIS
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256") != COVECTOR_IMPLEMENTATION_SHA256
        or int(campaign.get("summary", {}).get("completed", -1)) != 3
    ):
        raise ValueError(f"invalid source-covector campaign {path}")
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if not set(config.seeds).issubset(entries):
        raise ValueError("source-covector campaign lacks a requested checkpoint")
    details = {}
    for seed in config.seeds:
        entry = entries[seed]
        detail_path = Path(entry["path"])
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        covector = detail.get("source_maps", {}).get("covector", {})
        if (
            _sha256(detail_path) != entry.get("result_sha256")
            or detail.get("schema_version") != COVECTOR_SCHEMA
            or detail.get("hypothesis_id") != COVECTOR_HYPOTHESIS
            or detail.get("implementation_sha256") != COVECTOR_IMPLEMENTATION_SHA256
            or detail.get("gates", {}).get(
                "source_covector_oracle_error_all_fresh_cells_pass"
            )
            is not True
            or len(detail.get("coordinate_scale", [])) != config.carrier_rank
            or len(covector.get("linear", [])) != 2 * config.writer_order + 1
        ):
            raise ValueError(f"invalid source-covector result {detail_path}")
        details[seed] = (detail, detail_path)
    return campaign, path, details


def _mapping(record: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "linear": torch.tensor(record["linear"], dtype=torch.float64, device=device),
        "intercept": torch.tensor(
            record["intercept"], dtype=torch.float64, device=device
        ),
    }


def _fingerprint(
    config: AnalyticObservableResidualConfig,
    seed: int,
    scalar_result_sha256: str,
    covector_result_sha256: str,
    checkpoint_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "scalar_law_campaign_sha256": SCALAR_LAW_CAMPAIGN_SHA256,
        "covector_campaign_sha256": COVECTOR_CAMPAIGN_SHA256,
        "group_cohort_seeds": group.FRESH_COHORT_SEEDS,
        "seed": seed,
        "scalar_result_sha256": scalar_result_sha256,
        "covector_result_sha256": covector_result_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reusable_seed_result(
    result_path: Path,
    arrays_path: Path,
    config: AnalyticObservableResidualConfig,
    implementation: str,
    seed: int,
    fingerprint: str,
) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if not (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("evidence_role") == _evidence_role(config)
        and value.get("implementation_sha256") == implementation
        and value.get("configuration") == _json_compatible(asdict(config))
        and int(value.get("seed", -1)) == seed
        and value.get("scientific_fingerprint") == fingerprint
        and value.get("artifacts", {}).get("result") == str(result_path)
        and value.get("artifacts", {}).get("arrays") == str(arrays_path)
        and arrays_path.is_file()
        and _sha256(arrays_path) == value.get("artifacts", {}).get("arrays_sha256")
    ):
        raise ValueError(f"incompatible completed result {result_path}; use a new root")
    return value


def _campaign_is_reusable(
    value: Mapping[str, Any],
    config: AnalyticObservableResidualConfig,
    implementation: str,
) -> bool:
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("evidence_role") == _evidence_role(config)
        and value.get("implementation_sha256") == implementation
        and value.get("configuration") == _json_compatible(asdict(config))
        and int(value.get("summary", {}).get("completed", -1)) == len(config.seeds)
        and len(value.get("results", [])) == len(config.seeds)
        and all(
            Path(item.get("path", "")).is_file()
            and Path(item.get("arrays", "")).is_file()
            and _sha256(Path(item["path"])) == item.get("result_sha256")
            and _sha256(Path(item["arrays"])) == item.get("arrays_sha256")
            for item in value.get("results", [])
        )
    )


@torch.no_grad()
def _order4_angle(
    system: Any,
    task: CircleTaskConfig,
    config: AnalyticObservableResidualConfig,
    cell: Mapping[str, Any],
    basis: torch.Tensor,
    predicted: torch.Tensor,
) -> torch.Tensor:
    state = local._state_from_coordinates(cell, basis, predicted)
    posterior = local._continue_states(
        system, task, config, cell["target_cut"], {"order4": state}
    )["order4"]
    return local.readout.posterior_moment(posterior)[0]


@torch.no_grad()
def run_campaign(
    config: AnalyticObservableResidualConfig, output: Path
) -> dict[str, Any]:
    if not config.allow_underpowered:
        raise RuntimeError(
            "primary analytic-observable execution is forbidden by the locked "
            "source-only lifecycle disposition"
        )
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    implementation = _implementation_digest()
    campaign_path = output / "campaign_results.json"
    if campaign_path.is_file():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if _campaign_is_reusable(existing, config, implementation):
            print("aggregate already complete; leaving bytes unchanged", flush=True)
            return existing
        raise ValueError(f"incompatible completed aggregate {campaign_path}")

    device = torch.device(config.device)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    task = CircleTaskConfig()
    scalar_campaign, scalar_path, scalar_details = _load_scalar_law(config)
    covector_campaign, covector_path, covector_details = _load_covectors(config)
    law_config = law.NuisanceScalarTransformationConfig(
        group_root=config.group_root,
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        phase_count=config.phase_count,
        nuisance_replicates=config.nuisance_replicates,
        carrier_rank=config.carrier_rank,
        writer_order=config.writer_order,
        fine_step_std=config.fine_step_std,
        coarse_step_std=config.coarse_step_std,
        coordinate_scale_floor=config.coordinate_scale_floor,
        derivative_cosine_floor=config.derivative_cosine_floor,
        derivative_relative_l2_ceiling=config.derivative_relative_l2_ceiling,
        signed_error_r2_floor=config.signed_error_r2_floor,
        residual_mae_fraction_ceiling=config.residual_mae_fraction_ceiling,
        sign_agreement_floor=config.sign_agreement_floor,
        sign_magnitude_floor_bins=config.sign_magnitude_floor_bins,
        replay_tolerance=config.replay_tolerance,
        basis_gauge_replay_tolerance=config.basis_gauge_replay_tolerance,
        continuation_batch_size=config.continuation_batch_size,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    _, group_path, group_details = law._load_group_campaign(law_config)
    group_config = law._group_config(law_config)
    _, writer_path, writer_details, _, corrective_path = group._load_sources(group_config)
    paired = {
        regime: group.generate_group_paired_orbits(
            task,
            group_config,
            seed=group.FRESH_COHORT_SEEDS[regime],
            regime=regime,
        )
        for regime in REGIMES
    }
    nuisance_shift = group.phase_matched_shift_indices(
        config.phase_count, config.nuisance_replicates
    ).to(device)
    phase_shift = torch.from_numpy(
        defect.phase_shift_indices(config.phase_count, config.nuisance_replicates)
    ).to(device)
    fixed_config = fixed.FixedSemanticGaugeWriterConfig(
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    transport_config = fixed._transport_config(fixed_config)
    rank_config = transport._rank_config(transport_config)
    bridge = transport.rank._bridge_config(rank_config)

    results = []
    reused = 0
    for seed in config.seeds:
        seed_started = time.perf_counter()
        scalar_detail, scalar_result_path, scalar_arrays_path = scalar_details[seed]
        covector_detail, covector_result_path = covector_details[seed]
        group_detail, group_result_path, group_arrays_path = group_details[seed]
        prior, prior_path = writer_details[seed]
        with np.load(group_arrays_path, allow_pickle=False) as archive:
            stored_group = {name: archive[name].copy() for name in archive.files}
        with np.load(scalar_arrays_path, allow_pickle=False) as archive:
            stored_scalar = {name: archive[name].copy() for name in archive.files}
        fingerprint = _fingerprint(
            config,
            seed,
            _sha256(scalar_result_path),
            _sha256(covector_result_path),
            group_detail["provenance"]["checkpoint_sha256"],
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        arrays_path = output / "runs" / f"seed_{seed}" / "observable_residual_arrays.npz"
        existing = _reusable_seed_result(
            result_path, arrays_path, config, implementation, seed, fingerprint
        )
        if existing is not None:
            results.append(existing)
            reused += 1
            print(f"resuming {existing['experiment_id']}", flush=True)
            continue

        system, provenance = transport.rank.deck.load_source(task, bridge, 2, seed, device)
        if provenance["checkpoint_sha256"] != group_detail["provenance"]["checkpoint_sha256"]:
            raise ValueError(f"checkpoint mismatch for seed {seed}")
        frozen, frozen_path = transport.rank._load_character_source(
            rank_config, seed, provenance["checkpoint_sha256"]
        )
        basis, basis_summary = transport._fit_seed_basis(
            system, task, transport_config, rank_config, bridge, frozen, seed, device
        )
        if source.decomposition._numeric_max_difference(
            basis_summary, group_detail["basis"]
        ) > config.replay_tolerance:
            raise ValueError(f"basis replay mismatch for seed {seed}")
        gauge_cell = transport._extract_cell(
            system,
            task,
            transport_config,
            bridge,
            paired["composition"][0]["reference"],
            "heldout_a",
            "composition",
            device,
        )
        basis, gauge_replay = law.align_basis_to_stored_gauge(
            basis,
            transport._coordinates(gauge_cell, basis),
            torch.tensor(
                stored_group["composition__reference__target"], dtype=torch.float64
            ),
        )
        gauge_contract = bool(
            gauge_replay["maximum_orthogonality_error"]
            <= config.basis_gauge_replay_tolerance
            and gauge_replay["maximum_gauge_fit_error"]
            <= config.basis_gauge_replay_tolerance
            and gauge_replay["aligned_basis_orthogonality_error"]
            <= config.basis_gauge_replay_tolerance
        )
        writer_mapping = _mapping(
            prior["alignment_fit"]["mappings"]["fourier_m04"], device
        )
        covector_mapping = _mapping(
            covector_detail["source_maps"]["covector"], device
        )
        coordinate_scale = torch.tensor(
            covector_detail["coordinate_scale"], dtype=torch.float64, device=device
        )
        if (
            not bool(torch.isfinite(coordinate_scale).all())
            or float(coordinate_scale.min()) <= config.coordinate_scale_floor
        ):
            raise ValueError(f"invalid coordinate scale for seed {seed}")

        records = []
        arrays: dict[str, np.ndarray] = {}
        gradients = []
        predicted_gradients = []
        input_pair_pass = True
        input_replay_pass = True
        numerical_pass = True
        target_control_pass = True
        observation_pass = True
        scalar_replay_pass = True
        target_replay_pass = True
        maximum_scalar_replay_error = 0.0
        maximum_target_replay_error = 0.0
        for regime in REGIMES:
            datasets, input_summary = paired[regime]
            input_pair_pass = bool(input_pair_pass and input_summary["pair_contract"])
            input_replay_pass = bool(
                input_replay_pass
                and input_summary == group_detail["fresh_inputs"][regime]
            )
            latent_theta = (
                datasets["reference"]
                .quotient_phase.to(device)
                .reshape(config.orbit_count, 2)[:, 0]
                .double()
            )
            writer_features = group.writer.fourier_features(
                latent_theta, config.writer_order
            )
            predicted = transport.apply_affine(writer_features, writer_mapping)
            for arm, dataset in datasets.items():
                cell = transport._extract_cell(
                    system,
                    task,
                    transport_config,
                    bridge,
                    dataset,
                    "heldout_a",
                    regime,
                    device,
                )
                target = transport._coordinates(cell, basis)
                stored_target = torch.tensor(
                    stored_group[f"{regime}__{arm}__target"],
                    dtype=torch.float64,
                    device=device,
                )
                target_error = float((target - stored_target).abs().max())
                maximum_target_replay_error = max(maximum_target_replay_error, target_error)
                target_replay_pass = bool(
                    target_replay_pass
                    and target_error <= config.basis_gauge_replay_tolerance
                )
                derivative = local.finite_difference_cell(
                    system,
                    task,
                    config,
                    cell,
                    basis,
                    predicted,
                    target,
                    coordinate_scale,
                )
                linearization = source.jacobian.linearization_metrics(
                    derivative["fine_gradient"],
                    derivative["coarse_gradient"],
                    derivative["predicted_delta"],
                    derivative["observed_delta"],
                    config,
                )
                stored_signed = torch.tensor(
                    stored_scalar[f"{regime}__{arm}__signed_scalar"],
                    dtype=torch.float64,
                    device=device,
                )
                scalar_error = float(
                    (derivative["observed_delta"] - stored_signed).abs().max()
                )
                maximum_scalar_replay_error = max(maximum_scalar_replay_error, scalar_error)
                scalar_replay_pass = bool(
                    scalar_replay_pass
                    and scalar_error <= config.scalar_replay_tolerance
                )
                carrier, carrier_audit = fixed.semantic_orbit_carrier(
                    dataset, task, config.orbit_count, device
                )
                observation_pass = bool(
                    observation_pass and observed_carrier_pass(carrier_audit, config)
                )
                covector_features = capacity.fourier_features(
                    carrier, config.writer_order
                )
                predicted_gradient = transport.apply_affine(
                    covector_features, covector_mapping
                )
                writer_angle = _order4_angle(
                    system, task, config, cell, basis, predicted
                )
                semantic_angle = torch.atan2(carrier[:, 1], carrier[:, 0])
                analytic_scalar = source.jacobian._wrapped_bins(
                    semantic_angle, writer_angle, len(task.answer_token_ids)
                )
                analytic_std = source.task_inverse_correction(
                    predicted_gradient,
                    analytic_scalar,
                    config.gradient_denominator_floor,
                )
                oracle_std = source.task_inverse_correction(
                    predicted_gradient,
                    derivative["observed_delta"],
                    config.gradient_denominator_floor,
                )
                phase_std = source.task_inverse_correction(
                    predicted_gradient,
                    analytic_scalar[phase_shift],
                    config.gradient_denominator_floor,
                )
                nuisance_std = source.task_inverse_correction(
                    predicted_gradient,
                    analytic_scalar[nuisance_shift],
                    config.gradient_denominator_floor,
                )
                scale = coordinate_scale.unsqueeze(0)
                coordinates = {
                    "direct_rank3": target,
                    "order4": predicted,
                    "frozen_covector_oracle_error": predicted + oracle_std * scale,
                    "analytic_observable": predicted + analytic_std * scale,
                    "analytic_phase_shift": predicted + phase_std * scale,
                    "analytic_nuisance_shift": predicted + nuisance_std * scale,
                    "analytic_flipped": predicted - analytic_std * scale,
                }
                states = local.evaluate_science_states(
                    system, task, config, cell, basis, coordinates
                )
                control = bool(
                    not states["zero"]["continuous"]["continuous_pass"]
                    and states["exact"]["continuous"]["continuous_pass"]
                    and states["direct_rank3"]["continuous"]["continuous_pass"]
                )
                finite = all(
                    bool(torch.isfinite(value).all())
                    for value in (
                        *derivative.values(),
                        carrier,
                        predicted_gradient,
                        writer_angle,
                        analytic_scalar,
                        analytic_std,
                        oracle_std,
                    )
                )
                numerical_pass = bool(
                    numerical_pass and finite and linearization["adequate"]
                )
                target_control_pass = bool(target_control_pass and control)
                gradients.append(derivative["fine_gradient"])
                predicted_gradients.append(predicted_gradient)
                scalar_metrics = source.regression_metrics(
                    analytic_scalar,
                    derivative["observed_delta"],
                    config.sign_magnitude_floor_bins,
                )
                records.append(
                    {
                        "regime": regime,
                        "arm": arm,
                        "evaluation_seed": group.FRESH_COHORT_SEEDS[regime],
                        "observed_carrier_audit": carrier_audit,
                        "local_linearization": linearization,
                        "scalar_prediction": scalar_metrics,
                        "states": states,
                        "finite": finite,
                        "target_control_pass": control,
                    }
                )
                prefix = f"{regime}__{arm}"
                arrays[f"{prefix}__analytic_scalar"] = (
                    analytic_scalar.detach().cpu().numpy()
                )
                arrays[f"{prefix}__exact_scalar"] = (
                    derivative["observed_delta"].detach().cpu().numpy()
                )
                arrays[f"{prefix}__predicted_covector"] = (
                    predicted_gradient.detach().cpu().numpy()
                )

        covector_replay = source.regression_metrics(
            torch.cat(predicted_gradients), torch.cat(gradients)
        )
        covector_pass = covector_replay_pass(covector_replay, config)
        oracle_pass = all(
            cell["states"]["frozen_covector_oracle_error"]["continuous"][
                "continuous_pass"
            ]
            for cell in records
        )
        gates_summary = state_gate_summary(records, config.specificity_margin_bins)
        valid = bool(
            input_pair_pass
            and input_replay_pass
            and gauge_contract
            and target_replay_pass
            and scalar_replay_pass
            and numerical_pass
            and target_control_pass
            and observation_pass
            and covector_pass
        )
        classification, primary_gate = classify_checkpoint(
            valid=valid, oracle_pass=oracle_pass, state_gates=gates_summary
        )
        _write_npz(arrays_path, arrays)
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-analytic-observable-residual-seed{seed}",
            "status": "completed",
            "evidence_role": _evidence_role(config),
            "completed_at": _utc_now(),
            "seed": seed,
            "configuration": asdict(config),
            "scientific_fingerprint": fingerprint,
            "implementation_sha256": implementation,
            "provenance": {
                "scalar_law_campaign": str(scalar_path),
                "scalar_law_campaign_sha256": _sha256(scalar_path),
                "scalar_law_result": str(scalar_result_path),
                "scalar_law_result_sha256": _sha256(scalar_result_path),
                "scalar_law_arrays": str(scalar_arrays_path),
                "scalar_law_arrays_sha256": _sha256(scalar_arrays_path),
                "covector_campaign": str(covector_path),
                "covector_campaign_sha256": _sha256(covector_path),
                "covector_result": str(covector_result_path),
                "covector_result_sha256": _sha256(covector_result_path),
                "group_campaign": str(group_path),
                "group_result": str(group_result_path),
                "group_arrays": str(group_arrays_path),
                "writer_campaign": str(writer_path),
                "writer_result": str(prior_path),
                "corrective_campaign": str(corrective_path),
                "checkpoint": provenance["checkpoint"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "character_result": str(frozen_path),
                "character_result_sha256": _sha256(frozen_path),
            },
            "basis": basis_summary,
            "basis_gauge_replay": {
                **gauge_replay,
                "maximum_all_cell_target_replay_error": maximum_target_replay_error,
            },
            "maximum_scalar_source_replay_error": maximum_scalar_replay_error,
            "coordinate_scale": coordinate_scale.detach().cpu().tolist(),
            "covector_replay": covector_replay,
            "state_gates": gates_summary,
            "cells": records,
            "gates": {
                "provenance_contract": True,
                "input_and_pair_contract": input_pair_pass,
                "input_identity_replay_contract": input_replay_pass,
                "basis_gauge_replay_contract": bool(gauge_contract and target_replay_pass),
                "scalar_source_replay_contract": scalar_replay_pass,
                "observed_carrier_contract": observation_pass,
                "local_linearization_contract": numerical_pass,
                "target_control_contract": target_control_pass,
                "source_covector_replay_contract": covector_pass,
                "frozen_covector_oracle_all_cells_pass": oracle_pass,
                "analytic_observable_all_cells_pass": gates_summary[
                    "analytic_all_cells_pass"
                ],
                "all_controls_specific": gates_summary["all_controls_specific"],
                "analytic_observable_residual_gate": primary_gate,
            },
            "classification": classification,
            "primary_metric": float(primary_gate),
            "analysis_seconds": time.perf_counter() - seed_started,
            "artifacts": {
                "result": str(result_path),
                "arrays": str(arrays_path),
                "arrays_sha256": _sha256(arrays_path),
            },
        }
        _write_json(result_path, result)
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("analytic-observable implementation changed during campaign")

    classifications = [item["classification"] for item in results]
    pass_count = sum(item["gates"]["analytic_observable_residual_gate"] for item in results)
    if any(name == "invalid" for name in classifications):
        conclusion = "invalid_campaign"
    elif len(config.seeds) == 3 and pass_count == 3:
        conclusion = "supported_analytic_observable_residual_three_of_three"
    elif len(set(classifications)) == 1:
        conclusion = classifications[0]
    else:
        conclusion = "checkpoint_stratified_analytic_observable_residual"
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "group_cohort_seeds": group.FRESH_COHORT_SEEDS,
        "implementation_sha256": implementation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
        "provenance": {
            "scalar_law_campaign": str(scalar_path),
            "scalar_law_campaign_sha256": _sha256(scalar_path),
            "scalar_law_implementation_sha256": scalar_campaign[
                "implementation_sha256"
            ],
            "covector_campaign": str(covector_path),
            "covector_campaign_sha256": _sha256(covector_path),
            "covector_implementation_sha256": covector_campaign[
                "implementation_sha256"
            ],
        },
        "summary": {
            "requested": len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": reused,
            "trained_models": 0,
            "fitted_writers": 0,
            "fitted_covectors": 0,
            "fitted_observers": 0,
            "causal_cells": len(results) * len(REGIMES) * len(ARMS),
        },
        "aggregates": {
            "supported": bool(len(config.seeds) == 3 and pass_count == 3),
            "analytic_observable_residual_pass_count": pass_count,
            "required_checkpoint_count": 3,
            "conclusion": conclusion,
            "classification_counts": {
                name: classifications.count(name) for name in sorted(set(classifications))
            },
            "classification_by_seed": {
                str(item["seed"]): item["classification"] for item in results
            },
            "gate_counts": {
                name: sum(bool(item["gates"][name]) for item in results)
                for name in results[0]["gates"]
            },
        },
        "results": [
            {
                "experiment_id": item["experiment_id"],
                "seed": item["seed"],
                "scientific_fingerprint": item["scientific_fingerprint"],
                "classification": item["classification"],
                "gates": item["gates"],
                "path": item["artifacts"]["result"],
                "result_sha256": _sha256(Path(item["artifacts"]["result"])),
                "arrays": item["artifacts"]["arrays"],
                "arrays_sha256": item["artifacts"]["arrays_sha256"],
            }
            for item in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The analytic semantic estimate uses the exact synthetic calibration reference.",
            "The order-four baseline is a diagnostic internal writer state, not a deployed single-example state.",
            "The covector and checkpoints were selected after prior outcomes.",
            "The primary endpoint is frozen-continuation equivalence, not task-label accuracy.",
            "Three selected checkpoints do not establish population prevalence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "seed_*" / "result.json"),
            "arrays": str(output / "runs" / "seed_*" / "observable_residual_arrays.npz"),
        },
    }
    _write_json(campaign_path, campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_analytic_observable_residual/"
            "20260807_d6_existing_group"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=fixed.PRIMARY_SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = AnalyticObservableResidualConfig(
        seeds=args.seeds,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
