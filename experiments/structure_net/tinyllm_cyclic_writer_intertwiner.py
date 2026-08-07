#!/usr/bin/env python3
"""Test whether the fixed TinyLLM writer image carries the exact C16 action."""

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

import torch

import experiments.structure_net.tinyllm_fixed_gauge_writer_capacity as capacity
import experiments.structure_net.tinyllm_fixed_semantic_gauge_writer as fixed


SCHEMA_VERSION = "nal.tinyllm-c16-writer-intertwiner.v1"
HYPOTHESIS_ID = "tinyllm-c16-writer-intertwiner-v1"
PREDECESSOR_CAMPAIGN_SHA256 = (
    "c5592ee53b96e2d064a992070be6c3e699b24d88dbb658283e8dc2f00c267078"
)
PREDECESSOR_IMPLEMENTATION_SHA256 = (
    "7c284e35b5afc225eea45309262ab83c5f6d276736a557ebf20675ed3ccbfe7b"
)


@dataclass(frozen=True)
class CyclicWriterIntertwinerConfig:
    source_campaign: str = (
        "data/experiments/tinyllm_fixed_gauge_writer_capacity/"
        "20260806_d6_preregistered_diagnostic"
    )
    seeds: tuple[int, ...] = fixed.PRIMARY_SEEDS
    phase_bins: int = 16
    writer_order: int = 4
    writer_rank: int = 3
    numerical_rank_rtol: float = 1e-10
    analytic_tolerance: float = 1e-12
    obstruction_ceiling: float = 0.05
    closure_ceiling: float = 0.05
    random_controls: int = 256
    random_quantile: float = 0.05
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if tuple(self.seeds) != fixed.PRIMARY_SEEDS and not self.allow_underpowered:
            raise ValueError("primary writer-intertwiner seeds are fixed to 7,29,53")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.phase_bins != 16 or self.writer_order != 4 or self.writer_rank != 3:
            raise ValueError("the C16 order-four rank-three interface is fixed")
        if self.random_controls != 256 and not self.allow_underpowered:
            raise ValueError("the primary random-control count is fixed to 256")
        if self.random_controls < 8:
            raise ValueError("at least eight random controls are required")
        if not 0.0 < self.random_quantile < 0.5:
            raise ValueError("random quantile must lie between zero and one half")
        for name in (
            "numerical_rank_rtol",
            "analytic_tolerance",
            "obstruction_ceiling",
            "closure_ceiling",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(capacity.__file__), Path(fixed.__file__)):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _evidence_role(config: CyclicWriterIntertwinerConfig) -> str:
    return (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else "preregistered_post_outcome_underpowered_algebraic_diagnostic"
    )


def cyclic_feature_action(
    order: int, phase_bins: int, *, steps: int = 1
) -> torch.Tensor:
    """Return the exact row-feature action for one cyclic phase rotation."""
    alpha = 2.0 * math.pi * steps / phase_bins
    action = torch.zeros((2 * order + 1, 2 * order + 1), dtype=torch.float64)
    for harmonic in range(1, order + 1):
        angle = harmonic * alpha
        cosine = math.cos(angle)
        sine = math.sin(angle)
        start = 2 * (harmonic - 1)
        action[start : start + 2, start : start + 2] = torch.tensor(
            [[cosine, sine], [-sine, cosine]], dtype=torch.float64
        )
    action[-1, -1] = 1.0
    return action


def canonical_harmonic_basis(order: int, harmonic: int) -> torch.Tensor:
    if not 1 <= harmonic <= order:
        raise ValueError("harmonic is outside the declared feature ladder")
    basis = torch.zeros((2 * order + 1, 3), dtype=torch.float64)
    start = 2 * (harmonic - 1)
    basis[start, 0] = 1.0
    basis[start + 1, 1] = 1.0
    basis[-1, 2] = 1.0
    return basis


def _orthonormal_image(
    writer: torch.Tensor, numerical_rank_rtol: float
) -> tuple[torch.Tensor, torch.Tensor, int]:
    writer = writer.double()
    singular = torch.linalg.svdvals(writer)
    threshold = float(singular.max()) * numerical_rank_rtol
    rank = int((singular > threshold).sum())
    left, _, _ = torch.linalg.svd(writer, full_matrices=False)
    return left[:, :rank], singular, rank


def writer_intertwiner_diagnostics(
    writer: torch.Tensor,
    action: torch.Tensor,
    phase_bins: int,
    numerical_rank_rtol: float,
) -> dict[str, Any]:
    """Measure whether a writer image is invariant under a finite group action."""
    writer = writer.double()
    action = action.double()
    basis, singular, rank = _orthonormal_image(writer, numerical_rank_rtol)
    projector = basis @ basis.T
    identity = torch.eye(writer.shape[0], dtype=torch.float64)
    obstructions = []
    current = identity
    for power in range(phase_bins):
        moved = current @ writer
        residual = (identity - projector) @ moved
        denominator = torch.linalg.matrix_norm(moved).clamp_min(1e-24)
        obstructions.append(float(torch.linalg.matrix_norm(residual) / denominator))
        current = current @ action
    induced = torch.linalg.pinv(writer, rtol=numerical_rank_rtol) @ action @ writer
    closure = float(
        torch.linalg.matrix_norm(
            torch.linalg.matrix_power(induced, phase_bins)
            - torch.eye(writer.shape[1], dtype=torch.float64)
        )
        / math.sqrt(writer.shape[1])
    )
    nonidentity = torch.tensor(obstructions[1:], dtype=torch.float64)
    condition = (
        float(singular[0] / singular[rank - 1]) if rank == writer.shape[1] else None
    )
    return {
        "writer_rank": rank,
        "writer_singular_values": singular.tolist(),
        "writer_condition_number": condition,
        "identity_obstruction": obstructions[0],
        "orbit_obstructions": obstructions,
        "maximum_nonidentity_obstruction": float(nonidentity.max()),
        "rms_nonidentity_obstruction": float(torch.sqrt(nonidentity.square().mean())),
        "induced_generator": induced.tolist(),
        "induced_closure_error": closure,
        "projector": projector,
    }


def canonical_overlaps(
    writer_projector: torch.Tensor, order: int
) -> dict[str, float]:
    output = {}
    for harmonic in range(1, order + 1):
        basis = canonical_harmonic_basis(order, harmonic)
        canonical_projector = basis @ basis.T
        output[str(harmonic)] = float(
            torch.trace(writer_projector.double() @ canonical_projector) / 3.0
        )
    return output


def random_subspace_obstructions(
    *,
    action: torch.Tensor,
    phase_bins: int,
    feature_width: int,
    rank: int,
    count: int,
    seed: int,
    numerical_rank_rtol: float,
) -> list[float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = []
    for _ in range(count):
        random = torch.randn(
            feature_width, rank, dtype=torch.float64, generator=generator
        )
        basis, _ = torch.linalg.qr(random, mode="reduced")
        diagnostics = writer_intertwiner_diagnostics(
            basis, action, phase_bins, numerical_rank_rtol
        )
        values.append(diagnostics["maximum_nonidentity_obstruction"])
    return values


def analytic_controls(
    config: CyclicWriterIntertwinerConfig, action: torch.Tensor
) -> dict[str, Any]:
    width = 2 * config.writer_order + 1
    closure = float(
        torch.linalg.matrix_norm(
            torch.linalg.matrix_power(action, config.phase_bins)
            - torch.eye(width, dtype=torch.float64)
        )
        / math.sqrt(width)
    )
    canonical = {}
    for harmonic in range(1, config.writer_order + 1):
        diagnostics = writer_intertwiner_diagnostics(
            canonical_harmonic_basis(config.writer_order, harmonic),
            action,
            config.phase_bins,
            config.numerical_rank_rtol,
        )
        canonical[str(harmonic)] = {
            "maximum_nonidentity_obstruction": diagnostics[
                "maximum_nonidentity_obstruction"
            ],
            "induced_closure_error": diagnostics["induced_closure_error"],
        }
    passed = bool(
        closure <= config.analytic_tolerance
        and all(
            item["maximum_nonidentity_obstruction"] <= config.analytic_tolerance
            and item["induced_closure_error"] <= config.analytic_tolerance
            for item in canonical.values()
        )
    )
    return {
        "feature_action_closure_error": closure,
        "canonical_subspaces": canonical,
        "passed": passed,
    }


def classify_checkpoint(
    *,
    contracts_valid: bool,
    writer_rank: int,
    expected_rank: int,
    obstruction_pass: bool,
    closure_pass: bool,
    specificity_pass: bool,
) -> tuple[str, bool]:
    if not contracts_valid:
        return "invalid", False
    if writer_rank < expected_rank:
        return "degenerate_writer_image", False
    if obstruction_pass and closure_pass and specificity_pass:
        return "approximate_c16_writer_representation", True
    if obstruction_pass and not closure_pass:
        return "nonclosing_induced_action", False
    if not obstruction_pass:
        return "harmonic_mixed_writer_image", False
    return "nonspecific_writer_subspace", False


def _load_predecessor(
    config: CyclicWriterIntertwinerConfig,
) -> tuple[dict[str, Any], Path, dict[int, tuple[dict[str, Any], Path]]]:
    campaign_path = Path(config.source_campaign) / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    expected_classes = {
        str(seed): "unresolved_writer_limited" for seed in fixed.PRIMARY_SEEDS
    }
    if (
        _sha256(campaign_path) != PREDECESSOR_CAMPAIGN_SHA256
        or campaign.get("schema_version") != capacity.SCHEMA_VERSION
        or campaign.get("hypothesis_id") != capacity.HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("implementation_sha256")
        != PREDECESSOR_IMPLEMENTATION_SHA256
        or campaign.get("aggregates", {}).get("classification_by_seed")
        != expected_classes
        or int(campaign.get("summary", {}).get("trained_models", -1)) != 0
    ):
        raise ValueError(f"invalid writer-capacity predecessor {campaign_path}")
    entries = {int(item["seed"]): item for item in campaign.get("results", [])}
    if not set(config.seeds).issubset(entries):
        raise ValueError("writer-capacity predecessor lacks a requested checkpoint")
    details = {}
    for seed in config.seeds:
        entry = entries[seed]
        path = Path(entry["path"])
        detail = json.loads(path.read_text(encoding="utf-8"))
        writer = detail.get("alignment_fit", {}).get("writers", {}).get(
            "quotient_order4", {}
        )
        if (
            _sha256(path) != entry.get("result_sha256")
            or detail.get("schema_version") != capacity.SCHEMA_VERSION
            or detail.get("hypothesis_id") != capacity.HYPOTHESIS_ID
            or detail.get("implementation_sha256")
            != PREDECESSOR_IMPLEMENTATION_SHA256
            or detail.get("scientific_fingerprint")
            != entry.get("scientific_fingerprint")
            or detail.get("classification") != "unresolved_writer_limited"
            or not isinstance(writer.get("linear"), list)
            or not isinstance(writer.get("intercept"), list)
        ):
            raise ValueError(f"invalid writer-capacity result {path}")
        details[seed] = (detail, path)
    return campaign, campaign_path, details


def _fingerprint(
    config: CyclicWriterIntertwinerConfig,
    seed: int,
    predecessor_result_sha256: str,
    writer_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "seed": seed,
        "predecessor_campaign_sha256": PREDECESSOR_CAMPAIGN_SHA256,
        "predecessor_result_sha256": predecessor_result_sha256,
        "writer_sha256": writer_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _campaign_decision(
    classifications: Sequence[str], pass_count: int, allow_underpowered: bool
) -> dict[str, Any]:
    if allow_underpowered:
        conclusion = "systems_lifecycle_only_not_quality_evidence"
        supported = False
    elif len(classifications) == 3 and pass_count == 3:
        conclusion = "supported_c16_writer_representation_three_of_three"
        supported = True
    elif len(set(classifications)) == 1:
        conclusion = classifications[0]
        supported = False
    else:
        conclusion = "checkpoint_stratified_writer_representation"
        supported = False
    return {
        "supported": supported,
        "writer_intertwiner_pass_count": pass_count,
        "required_checkpoint_count": 3,
        "conclusion": conclusion,
    }


def _campaign_is_reusable(
    value: Mapping[str, Any],
    config: CyclicWriterIntertwinerConfig,
    implementation: str,
) -> bool:
    return bool(
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("hypothesis_id") == HYPOTHESIS_ID
        and value.get("status") == "completed"
        and value.get("evidence_role") == _evidence_role(config)
        and value.get("implementation_sha256") == implementation
        and value.get("configuration") == _strict_json(asdict(config))
        and int(value.get("summary", {}).get("completed", -1))
        == len(config.seeds)
        and len(value.get("results", [])) == len(config.seeds)
        and all(
            Path(item.get("path", "")).is_file()
            and _sha256(Path(item["path"])) == item.get("result_sha256")
            for item in value.get("results", [])
        )
    )


def run_campaign(
    config: CyclicWriterIntertwinerConfig, output: Path
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

    predecessor, predecessor_path, details = _load_predecessor(config)
    action = cyclic_feature_action(config.writer_order, config.phase_bins)
    controls = analytic_controls(config, action)
    results = []
    for seed in config.seeds:
        prior, prior_path = details[seed]
        writer_summary = prior["alignment_fit"]["writers"]["quotient_order4"]
        writer = torch.tensor(writer_summary["linear"], dtype=torch.float64)
        intercept = torch.tensor(writer_summary["intercept"], dtype=torch.float64)
        writer_payload = json.dumps(
            writer_summary, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        writer_sha = hashlib.sha256(writer_payload).hexdigest()
        diagnostics = writer_intertwiner_diagnostics(
            writer, action, config.phase_bins, config.numerical_rank_rtol
        )
        overlaps = canonical_overlaps(
            diagnostics.pop("projector"), config.writer_order
        )
        random_values = random_subspace_obstructions(
            action=action,
            phase_bins=config.phase_bins,
            feature_width=writer.shape[0],
            rank=config.writer_rank,
            count=config.random_controls,
            seed=16_000_003 + 100_003 * seed,
            numerical_rank_rtol=config.numerical_rank_rtol,
        )
        random_tensor = torch.tensor(random_values, dtype=torch.float64)
        random_threshold = float(
            torch.quantile(random_tensor, config.random_quantile)
        )
        specificity = bool(
            diagnostics["maximum_nonidentity_obstruction"] < random_threshold
        )
        finite = bool(
            torch.isfinite(writer).all()
            and torch.isfinite(intercept).all()
            and all(
                math.isfinite(float(value))
                for value in (
                    diagnostics["maximum_nonidentity_obstruction"],
                    diagnostics["rms_nonidentity_obstruction"],
                    diagnostics["induced_closure_error"],
                )
            )
        )
        contracts = bool(
            controls["passed"]
            and finite
            and diagnostics["identity_obstruction"] <= config.analytic_tolerance
            and float(intercept.abs().max()) <= config.analytic_tolerance
        )
        obstruction_pass = bool(
            diagnostics["maximum_nonidentity_obstruction"]
            <= config.obstruction_ceiling
        )
        closure_pass = bool(
            diagnostics["induced_closure_error"] <= config.closure_ceiling
        )
        classification, primary = classify_checkpoint(
            contracts_valid=contracts,
            writer_rank=diagnostics["writer_rank"],
            expected_rank=config.writer_rank,
            obstruction_pass=obstruction_pass,
            closure_pass=closure_pass,
            specificity_pass=specificity,
        )
        fingerprint = _fingerprint(
            config, seed, _sha256(prior_path), writer_sha
        )
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-c16-writer-intertwiner-seed{seed}",
            "status": "completed",
            "evidence_role": _evidence_role(config),
            "completed_at": _utc_now(),
            "seed": seed,
            "configuration": asdict(config),
            "scientific_fingerprint": fingerprint,
            "implementation_sha256": implementation,
            "provenance": {
                "predecessor_campaign": str(predecessor_path),
                "predecessor_campaign_sha256": _sha256(predecessor_path),
                "predecessor_result": str(prior_path),
                "predecessor_result_sha256": _sha256(prior_path),
                "checkpoint": prior["provenance"]["checkpoint"],
                "checkpoint_sha256": prior["provenance"]["checkpoint_sha256"],
                "writer_sha256": writer_sha,
            },
            "analytic_controls": controls,
            "writer": {
                "feature_width": int(writer.shape[0]),
                "coordinate_width": int(writer.shape[1]),
                "intercept_maximum_absolute": float(intercept.abs().max()),
                "canonical_harmonic_overlaps": overlaps,
                **diagnostics,
            },
            "random_controls": {
                "seed": 16_000_003 + 100_003 * seed,
                "count": config.random_controls,
                "maximum_obstructions": random_values,
                "minimum": float(random_tensor.min()),
                "median": float(random_tensor.median()),
                "fifth_percentile": random_threshold,
                "maximum": float(random_tensor.max()),
            },
            "gates": {
                "provenance_and_analytic_contract": contracts,
                "writer_rank_three": diagnostics["writer_rank"]
                == config.writer_rank,
                "orbit_obstruction": obstruction_pass,
                "induced_closure": closure_pass,
                "random_subspace_specificity": specificity,
                "writer_intertwiner_gate": primary,
            },
            "classification": classification,
            "primary_metric": float(primary),
            "analysis_seconds": time.perf_counter() - started,
            "artifacts": {"result": str(result_path)},
        }
        _write_json(result_path, result)
        results.append(result)
        print(f"seed {seed}: {classification}", flush=True)
        if _implementation_digest() != implementation:
            raise RuntimeError("writer-intertwiner implementation changed during run")

    classifications = [item["classification"] for item in results]
    pass_count = sum(item["gates"]["writer_intertwiner_gate"] for item in results)
    decision = _campaign_decision(
        classifications, pass_count, config.allow_underpowered
    )
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
            "device": "cpu",
        },
        "provenance": {
            "predecessor_campaign": str(predecessor_path),
            "predecessor_campaign_sha256": _sha256(predecessor_path),
            "predecessor_implementation_sha256": predecessor[
                "implementation_sha256"
            ],
        },
        "analytic_controls": controls,
        "summary": {
            "requested": len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "excluded": 0,
            "trained_models": 0,
            "fitted_writers": 0,
            "fitted_predictive_observers": 0,
            "random_subspaces": len(config.seeds) * config.random_controls,
        },
        "aggregates": {
            **decision,
            "classification_counts": {
                name: classifications.count(name)
                for name in sorted(set(classifications))
            },
            "classification_by_seed": {
                str(item["seed"]): item["classification"] for item in results
            },
            "gate_counts": {
                name: sum(bool(item["gates"][name]) for item in results)
                for name in (
                    "provenance_and_analytic_contract",
                    "writer_rank_three",
                    "orbit_obstruction",
                    "induced_closure",
                    "random_subspace_specificity",
                    "writer_intertwiner_gate",
                )
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
            }
            for item in results
        ],
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "C16 is the synthetic task-output phase group, not the full acquisition nuisance group.",
            "The test concerns the selected order-four writer image, not every possible carrier.",
            "Random subspaces are specificity controls rather than a trained-model null distribution.",
            "Three selected checkpoints do not establish population prevalence.",
        ],
        "artifacts": {
            "campaign": str(campaign_path),
            "runs": str(output / "runs" / "seed_*" / "result.json"),
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
            "data/experiments/tinyllm_cyclic_writer_intertwiner/"
            "20260807_preregistered_diagnostic"
        ),
    )
    parser.add_argument("--seeds", type=_ints, default=fixed.PRIMARY_SEEDS)
    parser.add_argument("--random-controls", type=int, default=256)
    parser.add_argument("--allow-underpowered", action="store_true")
    args = parser.parse_args()
    config = CyclicWriterIntertwinerConfig(
        seeds=args.seeds,
        random_controls=args.random_controls,
        allow_underpowered=args.allow_underpowered,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
