#!/usr/bin/env python3
"""Decompose TinyLLM's action-dependent correction into exact task-angle defects."""

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
from typing import Any, Mapping

import numpy as np
import torch

import experiments.structure_net.tinyllm_nuisance_scalar_transformation_law as law
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


SCHEMA_VERSION = "nal.tinyllm-c2-scalar-groupoid-defect.v1"
HYPOTHESIS_ID = "tinyllm-c2-scalar-groupoid-defect-v1"
TRANSFORMATION_CAMPAIGN_SHA256 = (
    "1e1795c931335c739a38550b85def7c4b442be39afcea76af8cd8491b1ff6589"
)
TRANSFORMATION_IMPLEMENTATION_SHA256 = (
    "e92c3894b3e8820c0242edeba29e1f893f321e7f1dc20d75583c898ce7b1526f"
)
REGIMES = law.REGIMES
GROUP_ARMS = law.GROUP_ARMS


@dataclass(frozen=True)
class ScalarGroupoidDefectConfig:
    transformation_root: str = (
        "data/experiments/tinyllm_nuisance_scalar_transformation_law/"
        "20260807_d6_existing_group_gauge_replay"
    )
    seeds: tuple[int, ...] = law.fixed.PRIMARY_SEEDS
    orbit_count: int = 64
    phase_count: int = 16
    nuisance_replicates: int = 4
    carrier_rank: int = 3
    writer_order: int = 4
    sign_magnitude_floor_bins: float = 0.01
    groupoid_max_error_ceiling_bins: float = 1e-6
    groupoid_relative_l2_ceiling: float = 1e-6
    action_defect_rms_floor_bins: float = 0.02
    direct_to_total_rms_ceiling: float = 0.10
    direct_p95_ceiling_bins: float = 0.05
    prediction_r2_floor: float = 0.90
    prediction_relative_l2_ceiling: float = math.sqrt(0.10)
    prediction_sign_agreement_floor: float = 0.90
    specificity_r2_margin: float = 0.10
    basis_gauge_replay_tolerance: float = 1e-5
    signed_scalar_replay_tolerance_bins: float = 1e-5
    continuation_batch_size: int = 256
    device: str = "cuda"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != law.fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary scalar-groupoid seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.orbit_count != 64:
            raise ValueError("the exact group cohort fixes 64 orbits")
        if self.phase_count != 16 or self.nuisance_replicates != 4:
            raise ValueError("the exact group cohort fixes 16 phases x 4 nuisances")
        if self.phase_count * self.nuisance_replicates != self.orbit_count:
            raise ValueError("phase and nuisance counts must tile the cohort")
        if self.carrier_rank != 3 or self.writer_order != 4:
            raise ValueError("carrier rank and writer order are fixed to three and four")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(law.__file__),
        Path(law.group.__file__),
        Path(law.local.__file__),
        Path(law.source.__file__),
        Path(law.fixed.__file__),
        Path(law.transport.__file__),
        Path(law.transport.rank.__file__),
        Path(law.transport.coupling.__file__),
    ):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: ScalarGroupoidDefectConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_existing_artifact_post_outcome_underpowered_mechanistic_evidence"
    )


def wrap_bin_values(value: torch.Tensor, bins: int) -> torch.Tensor:
    width = 2.0 * math.pi / bins
    radians = value.double() * width
    return torch.atan2(torch.sin(radians), torch.cos(radians)) / width


def groupoid_components(
    *,
    reference_direct: torch.Tensor,
    transformed_direct: torch.Tensor,
    reference_writer: torch.Tensor,
    transformed_writer: torch.Tensor,
    bins: int,
) -> dict[str, torch.Tensor]:
    delta_direct = law.source.jacobian._wrapped_bins(
        transformed_direct, reference_direct, bins
    )
    delta_writer = law.source.jacobian._wrapped_bins(
        transformed_writer, reference_writer, bins
    )
    reference_scalar = law.source.jacobian._wrapped_bins(
        reference_direct, reference_writer, bins
    )
    transformed_scalar = law.source.jacobian._wrapped_bins(
        transformed_direct, transformed_writer, bins
    )
    delta_scalar = wrap_bin_values(transformed_scalar - reference_scalar, bins)
    reconstructed = wrap_bin_values(delta_direct - delta_writer, bins)
    reconstruction_error = wrap_bin_values(reconstructed - delta_scalar, bins)
    return {
        "reference_scalar": reference_scalar,
        "transformed_scalar": transformed_scalar,
        "delta_direct": delta_direct,
        "delta_writer": delta_writer,
        "delta_scalar": delta_scalar,
        "reconstructed_delta_scalar": reconstructed,
        "reconstruction_error": reconstruction_error,
    }


def prediction_pass(
    metrics: Mapping[str, Any], config: ScalarGroupoidDefectConfig
) -> bool:
    return bool(
        metrics["zero_referenced_r2"] >= config.prediction_r2_floor
        and metrics["relative_l2"] <= config.prediction_relative_l2_ceiling
        and metrics["sign_agreement"] >= config.prediction_sign_agreement_floor
    )


def groupoid_cell_metrics(
    components: Mapping[str, torch.Tensor],
    shuffled_indices: torch.Tensor,
    config: ScalarGroupoidDefectConfig,
) -> dict[str, Any]:
    target = components["delta_scalar"]
    direct = components["delta_direct"]
    writer_prediction = -components["delta_writer"]
    paired = law.scalar_pair_metrics(
        writer_prediction, target, config.sign_magnitude_floor_bins
    )
    sign_flipped = law.scalar_pair_metrics(
        -writer_prediction, target, config.sign_magnitude_floor_bins
    )
    shuffled = law.scalar_pair_metrics(
        writer_prediction[shuffled_indices.to(writer_prediction.device)],
        target,
        config.sign_magnitude_floor_bins,
    )
    target_rms = float(torch.sqrt(target.double().square().mean()))
    direct_rms = float(torch.sqrt(direct.double().square().mean()))
    direct_p95 = float(torch.quantile(direct.double().abs(), 0.95))
    error = components["reconstruction_error"].double()
    reconstruction_relative_l2 = float(
        torch.linalg.vector_norm(error)
        / torch.linalg.vector_norm(target.double()).clamp_min(1e-24)
    )
    identity_pass = bool(
        float(error.abs().max()) <= config.groupoid_max_error_ceiling_bins
        and reconstruction_relative_l2 <= config.groupoid_relative_l2_ceiling
    )
    nondegenerate = target_rms >= config.action_defect_rms_floor_bins
    direct_negligible = bool(
        direct_rms / max(target_rms, 1e-24) <= config.direct_to_total_rms_ceiling
        and direct_p95 <= config.direct_p95_ceiling_bins
    )
    writer_pass = prediction_pass(paired, config)
    specificity = bool(
        writer_pass
        and not prediction_pass(sign_flipped, config)
        and not prediction_pass(shuffled, config)
        and paired["zero_referenced_r2"] - sign_flipped["zero_referenced_r2"]
        >= config.specificity_r2_margin
        and paired["zero_referenced_r2"] - shuffled["zero_referenced_r2"]
        >= config.specificity_r2_margin
    )
    return {
        "delta_scalar_rms_bins": target_rms,
        "delta_scalar_p95_abs_bins": float(torch.quantile(target.double().abs(), 0.95)),
        "delta_direct_rms_bins": direct_rms,
        "delta_direct_p95_abs_bins": direct_p95,
        "delta_writer_rms_bins": float(
            torch.sqrt(components["delta_writer"].double().square().mean())
        ),
        "delta_writer_p95_abs_bins": float(
            torch.quantile(components["delta_writer"].double().abs(), 0.95)
        ),
        "direct_to_total_rms_ratio": direct_rms / max(target_rms, 1e-24),
        "groupoid_max_abs_error_bins": float(error.abs().max()),
        "groupoid_relative_l2": reconstruction_relative_l2,
        "writer_only": paired,
        "sign_flipped_control": sign_flipped,
        "phase_matched_shuffled_control": shuffled,
        "gates": {
            "action_defect_nondegenerate": nondegenerate,
            "exact_groupoid_identity": identity_pass,
            "direct_term_negligible": direct_negligible,
            "writer_only_prediction": writer_pass,
            "writer_only_specificity": specificity,
            "primary_cell_pass": bool(
                nondegenerate
                and identity_pass
                and direct_negligible
                and writer_pass
                and specificity
            ),
        },
    }


def classify_checkpoint(
    *,
    valid: bool,
    direct_negligible_all: bool,
    writer_prediction_all: bool,
    specificity_all: bool,
) -> str:
    if not valid:
        return "invalid"
    if direct_negligible_all and writer_prediction_all and specificity_all:
        return "writer_symmetry_defect_dominant"
    if direct_negligible_all and writer_prediction_all:
        return "writer_defect_descriptive_not_specific"
    if not direct_negligible_all:
        return "two_term_groupoid_defect"
    return "writer_only_reduction_failed"


def _law_config(config: ScalarGroupoidDefectConfig) -> law.NuisanceScalarTransformationConfig:
    return law.NuisanceScalarTransformationConfig(
        seeds=config.seeds,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )


def _load_transformation_campaign(
    config: ScalarGroupoidDefectConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path, Path]]]:
    path = Path(config.transformation_root) / "campaign_results.json"
    campaign = json.loads(path.read_text(encoding="utf-8"))
    required_true = (
        "provenance_contract",
        "input_and_pair_contract",
        "input_identity_replay_contract",
        "basis_gauge_replay_contract",
        "local_linearization_contract",
        "target_control_contract",
    )
    gates = campaign.get("aggregates", {}).get("gate_counts", {})
    if (
        _sha256(path) != TRANSFORMATION_CAMPAIGN_SHA256
        or campaign.get("schema_version") != law.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != law.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256")
        != TRANSFORMATION_IMPLEMENTATION_SHA256
        or any(int(gates.get(name, -1)) != 3 for name in required_true)
        or int(campaign.get("summary", {}).get("trained_models", -1)) != 0
        or campaign.get("aggregates", {}).get("conclusion")
        != "scalar_action_dependent"
    ):
        raise ValueError(f"invalid transformation-law campaign {path}")
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if not set(config.seeds).issubset(entries):
        raise ValueError("transformation-law campaign lacks a requested checkpoint")
    details = {}
    for seed in config.seeds:
        entry = entries[seed]
        result_path = Path(entry["path"])
        arrays_path = Path(entry["arrays"])
        detail = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            _sha256(result_path) != entry.get("result_sha256")
            or _sha256(arrays_path) != entry.get("arrays_sha256")
            or detail.get("implementation_sha256")
            != TRANSFORMATION_IMPLEMENTATION_SHA256
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("classification") != "scalar_action_dependent"
            or any(detail.get("gates", {}).get(name) is not True for name in required_true)
        ):
            raise ValueError(f"invalid transformation-law result {result_path}")
        details[seed] = (detail, result_path, arrays_path)
    return campaign, path, details


def _fingerprint(
    config: ScalarGroupoidDefectConfig,
    seed: int,
    transformation_result_sha256: str,
    checkpoint_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "source_campaign_sha256": TRANSFORMATION_CAMPAIGN_SHA256,
        "seed": seed,
        "source_result_sha256": transformation_result_sha256,
        "checkpoint_sha256": checkpoint_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _reusable_seed_result(
    result_path: Path,
    arrays_path: Path,
    config: ScalarGroupoidDefectConfig,
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
    config: ScalarGroupoidDefectConfig,
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
def run_campaign(
    config: ScalarGroupoidDefectConfig, output: Path
) -> dict[str, Any]:
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
    task = CircleTaskConfig()
    transformation, transformation_path, transformation_details = (
        _load_transformation_campaign(config)
    )
    law_config = _law_config(config)
    _, group_path, group_details = law._load_group_campaign(law_config)
    group_config = law._group_config(law_config)
    _, writer_path, writer_details, _, corrective_path = law.group._load_sources(
        group_config
    )
    paired = {
        regime: law.group.generate_group_paired_orbits(
            task,
            group_config,
            seed=law.group.FRESH_COHORT_SEEDS[regime],
            regime=regime,
        )
        for regime in REGIMES
    }
    shuffled_indices = law.group.phase_matched_shift_indices(
        config.phase_count, config.nuisance_replicates
    )
    fixed_config = law.fixed.FixedSemanticGaugeWriterConfig(
        seeds=config.seeds,
        orbit_count=config.orbit_count,
        device=config.device,
        allow_underpowered=config.allow_underpowered,
    )
    transport_config = law.fixed._transport_config(fixed_config)
    rank_config = law.transport._rank_config(transport_config)
    bridge = law.transport.rank._bridge_config(rank_config)

    results: list[dict[str, Any]] = []
    reused = 0
    for seed in config.seeds:
        seed_started = time.perf_counter()
        transformation_detail, transformation_result_path, transformation_arrays_path = (
            transformation_details[seed]
        )
        group_detail, group_result_path, group_arrays_path = group_details[seed]
        prior, prior_path = writer_details[seed]
        fingerprint = _fingerprint(
            config,
            seed,
            _sha256(transformation_result_path),
            transformation_detail["provenance"]["checkpoint_sha256"],
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        arrays_path = output / "runs" / f"seed_{seed}" / "groupoid_arrays.npz"
        existing = _reusable_seed_result(
            result_path, arrays_path, config, implementation, seed, fingerprint
        )
        if existing is not None:
            results.append(existing)
            reused += 1
            print(f"resuming {existing['experiment_id']}", flush=True)
            continue

        with np.load(group_arrays_path) as archive:
            stored_group = {name: archive[name].copy() for name in archive.files}
        with np.load(transformation_arrays_path) as archive:
            stored_scalars = {name: archive[name].copy() for name in archive.files}

        system, provenance = law.transport.rank.deck.load_source(
            task, bridge, 2, seed, device
        )
        if provenance["checkpoint_sha256"] != transformation_detail["provenance"][
            "checkpoint_sha256"
        ]:
            raise ValueError(f"checkpoint mismatch for seed {seed}")
        frozen, frozen_path = law.transport.rank._load_character_source(
            rank_config, seed, provenance["checkpoint_sha256"]
        )
        basis, basis_summary = law.transport._fit_seed_basis(
            system,
            task,
            transport_config,
            rank_config,
            bridge,
            frozen,
            seed,
            device,
        )
        gauge_dataset = paired["composition"][0]["reference"]
        gauge_cell = law.transport._extract_cell(
            system,
            task,
            transport_config,
            bridge,
            gauge_dataset,
            "heldout_a",
            "composition",
            device,
        )
        regenerated_gauge = law.transport._coordinates(gauge_cell, basis)
        stored_gauge = torch.tensor(
            stored_group["composition__reference__target"], dtype=torch.float64
        )
        basis, basis_gauge = law.align_basis_to_stored_gauge(
            basis, regenerated_gauge, stored_gauge
        )
        if max(
            basis_gauge["maximum_orthogonality_error"],
            basis_gauge["maximum_gauge_fit_error"],
            basis_gauge["aligned_basis_orthogonality_error"],
        ) > config.basis_gauge_replay_tolerance:
            raise ValueError(f"basis gauge replay mismatch for seed {seed}")

        mapping_record = prior["alignment_fit"]["mappings"]["fourier_m04"]
        mapping = {
            "linear": torch.tensor(
                mapping_record["linear"], dtype=torch.float64, device=device
            ),
            "intercept": torch.tensor(
                mapping_record["intercept"], dtype=torch.float64, device=device
            ),
        }
        if law.group._tensor_digest(
            mapping["linear"], mapping["intercept"]
        ) != group_detail["writer_mapping_sha256"]:
            raise ValueError(f"writer mapping mismatch for seed {seed}")

        cell_angles: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        arrays: dict[str, np.ndarray] = {}
        input_replay_pass = True
        target_replay_pass = True
        scalar_replay_pass = True
        finite_pass = True
        max_target_replay_error = 0.0
        max_scalar_replay_error = 0.0
        for regime in REGIMES:
            datasets, input_summary = paired[regime]
            input_replay_pass = bool(
                input_replay_pass
                and input_summary == group_detail["fresh_inputs"][regime]
            )
            theta = (
                datasets["reference"]
                .quotient_phase.to(device)
                .reshape(config.orbit_count, 2)[:, 0]
                .double()
            )
            features = law.group.writer.fourier_features(theta, config.writer_order)
            predicted = law.transport.apply_affine(features, mapping)
            cell_angles[regime] = {}
            for arm, dataset in datasets.items():
                cell = law.transport._extract_cell(
                    system,
                    task,
                    transport_config,
                    bridge,
                    dataset,
                    "heldout_a",
                    regime,
                    device,
                )
                target = law.transport._coordinates(cell, basis)
                stored_target = torch.tensor(
                    stored_group[f"{regime}__{arm}__target"],
                    dtype=torch.float64,
                    device=device,
                )
                target_error = float((target - stored_target).abs().max())
                max_target_replay_error = max(max_target_replay_error, target_error)
                target_replay_pass = bool(
                    target_replay_pass
                    and target_error <= config.basis_gauge_replay_tolerance
                )
                posteriors = law.group.local._posteriors_for_coordinates(
                    system,
                    task,
                    cell["target_cut"],
                    cell["propagated"],
                    basis,
                    {"writer": predicted, "direct": target},
                    config.continuation_batch_size,
                )
                writer_angle = law.local.readout.posterior_moment(posteriors["writer"])[0]
                direct_angle = law.local.readout.posterior_moment(posteriors["direct"])[0]
                scalar = law.source.jacobian._wrapped_bins(
                    direct_angle, writer_angle, len(task.answer_token_ids)
                )
                stored_scalar = torch.tensor(
                    stored_scalars[f"{regime}__{arm}__signed_scalar"],
                    dtype=torch.float64,
                    device=device,
                )
                scalar_error = float(
                    wrap_bin_values(
                        scalar - stored_scalar, len(task.answer_token_ids)
                    )
                    .abs()
                    .max()
                )
                max_scalar_replay_error = max(max_scalar_replay_error, scalar_error)
                scalar_replay_pass = bool(
                    scalar_replay_pass
                    and scalar_error <= config.signed_scalar_replay_tolerance_bins
                )
                finite_pass = bool(
                    finite_pass
                    and torch.isfinite(writer_angle).all()
                    and torch.isfinite(direct_angle).all()
                    and torch.isfinite(scalar).all()
                )
                cell_angles[regime][arm] = {
                    "writer": writer_angle,
                    "direct": direct_angle,
                }
                prefix = f"{regime}__{arm}"
                arrays[f"{prefix}__writer_angle"] = writer_angle.cpu().numpy()
                arrays[f"{prefix}__direct_angle"] = direct_angle.cpu().numpy()
                arrays[f"{prefix}__signed_scalar"] = scalar.cpu().numpy()

        cells: list[dict[str, Any]] = []
        groupoid_identity_pass = True
        for regime in REGIMES:
            reference = cell_angles[regime]["reference"]
            for arm in GROUP_ARMS:
                transformed = cell_angles[regime][arm]
                components = groupoid_components(
                    reference_direct=reference["direct"],
                    transformed_direct=transformed["direct"],
                    reference_writer=reference["writer"],
                    transformed_writer=transformed["writer"],
                    bins=len(task.answer_token_ids),
                )
                metrics = groupoid_cell_metrics(
                    components, shuffled_indices, config
                )
                groupoid_identity_pass = bool(
                    groupoid_identity_pass
                    and metrics["gates"]["exact_groupoid_identity"]
                )
                prefix = f"{regime}__{arm}"
                for name in (
                    "delta_scalar",
                    "delta_direct",
                    "delta_writer",
                    "reconstructed_delta_scalar",
                    "reconstruction_error",
                ):
                    arrays[f"{prefix}__{name}"] = components[name].cpu().numpy()
                cells.append({"regime": regime, "arm": arm, **metrics})

        inherited_contract = all(
            transformation_detail["gates"][name]
            for name in (
                "provenance_contract",
                "input_and_pair_contract",
                "input_identity_replay_contract",
                "basis_gauge_replay_contract",
                "local_linearization_contract",
                "target_control_contract",
            )
        )
        valid = bool(
            inherited_contract
            and input_replay_pass
            and target_replay_pass
            and scalar_replay_pass
            and finite_pass
            and groupoid_identity_pass
        )
        direct_all = all(cell["gates"]["direct_term_negligible"] for cell in cells)
        writer_all = all(cell["gates"]["writer_only_prediction"] for cell in cells)
        specificity_all = all(
            cell["gates"]["writer_only_specificity"] for cell in cells
        )
        primary = bool(
            valid and all(cell["gates"]["primary_cell_pass"] for cell in cells)
        )
        classification = classify_checkpoint(
            valid=valid,
            direct_negligible_all=direct_all,
            writer_prediction_all=writer_all,
            specificity_all=specificity_all,
        )
        _write_npz(arrays_path, arrays)
        arrays_sha256 = _sha256(arrays_path)
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c2-scalar-groupoid-defect-seed{seed}",
            "status": "completed",
            "evidence_role": _evidence_role(config),
            "completed_at": _utc_now(),
            "seed": seed,
            "configuration": asdict(config),
            "scientific_fingerprint": fingerprint,
            "implementation_sha256": implementation,
            "provenance": {
                "transformation_campaign": str(transformation_path),
                "transformation_campaign_sha256": _sha256(transformation_path),
                "transformation_result": str(transformation_result_path),
                "transformation_result_sha256": _sha256(transformation_result_path),
                "transformation_arrays": str(transformation_arrays_path),
                "transformation_arrays_sha256": _sha256(transformation_arrays_path),
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
                **basis_gauge,
                "maximum_all_cell_target_replay_error": max_target_replay_error,
            },
            "maximum_signed_scalar_replay_error_bins": max_scalar_replay_error,
            "cells": cells,
            "gates": {
                "provenance_and_inherited_contract": inherited_contract,
                "input_identity_replay_contract": input_replay_pass,
                "basis_gauge_and_target_replay_contract": target_replay_pass,
                "signed_scalar_replay_contract": scalar_replay_pass,
                "finite_angle_contract": finite_pass,
                "exact_groupoid_identity_all_cells": groupoid_identity_pass,
                "direct_term_negligible_all_cells": direct_all,
                "writer_only_prediction_all_cells": writer_all,
                "writer_only_specificity_all_cells": specificity_all,
                "writer_symmetry_defect_dominance_gate": primary,
            },
            "classification": classification,
            "primary_metric": float(primary),
            "analysis_seconds": time.perf_counter() - seed_started,
            "artifacts": {
                "result": str(result_path),
                "arrays": str(arrays_path),
                "arrays_sha256": arrays_sha256,
            },
        }
        _write_json(result_path, result)
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("scalar-groupoid implementation changed during campaign")

    classifications = [result["classification"] for result in results]
    pass_count = sum(
        result["gates"]["writer_symmetry_defect_dominance_gate"]
        for result in results
    )
    if any(name == "invalid" for name in classifications):
        conclusion = "invalid_campaign"
    elif len(config.seeds) == 3 and pass_count == 3:
        conclusion = "supported_writer_symmetry_defect_dominant_three_of_three"
    elif len(set(classifications)) == 1:
        conclusion = classifications[0]
    else:
        conclusion = "checkpoint_stratified_groupoid_defect"
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": _evidence_role(config),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
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
            "transformation_campaign": str(transformation_path),
            "transformation_campaign_sha256": _sha256(transformation_path),
            "transformation_implementation_sha256": transformation[
                "implementation_sha256"
            ],
        },
        "summary": {
            "requested": len(config.seeds),
            "scheduled": len(config.seeds) - reused,
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "retries": 0,
            "reused": reused,
            "trained_models": 0,
            "fitted_writers": 0,
            "fitted_encoders": 0,
            "fitted_observers": 0,
            "groupoid_cells": len(config.seeds) * len(REGIMES) * len(GROUP_ARMS),
        },
        "aggregates": {
            "conclusion": conclusion,
            "primary_checkpoint_pass_count": pass_count,
            "required_checkpoint_pass_count": 3,
            "classification_counts": {
                name: classifications.count(name) for name in sorted(set(classifications))
            },
            "classification_by_seed": {
                str(result["seed"]): result["classification"] for result in results
            },
            "gate_counts": {
                name: sum(bool(result["gates"][name]) for result in results)
                for name in results[0]["gates"]
            },
        },
        "results": [
            {
                "experiment_id": result["experiment_id"],
                "seed": result["seed"],
                "scientific_fingerprint": result["scientific_fingerprint"],
                "classification": result["classification"],
                "gates": result["gates"],
                "path": result["artifacts"]["result"],
                "result_sha256": _sha256(Path(result["artifacts"]["result"])),
                "arrays": result["artifacts"]["arrays"],
                "arrays_sha256": _sha256(Path(result["artifacts"]["arrays"])),
            }
            for result in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "The exact groupoid identity is algebraic; only term dominance and specificity are scientific.",
            "Direct-rank-three and order-four states are frozen diagnostic patches, not native deployable interfaces.",
            "The observed group covers positive scale, planar rotation, constant offset, and their composition only.",
            "This reuses an analyzed cohort and is not independent fresh confirmation.",
            "Three selected checkpoints do not establish population prevalence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "seed_*" / "result.json"),
            "arrays": str(output / "runs" / "seed_*" / "groupoid_arrays.npz"),
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
            "data/experiments/tinyllm_scalar_groupoid_defect/"
            "20260807_d6_existing_group"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=_ints, default=law.fixed.PRIMARY_SEEDS)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = ScalarGroupoidDefectConfig(
        seeds=args.seeds,
        device=args.device,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
