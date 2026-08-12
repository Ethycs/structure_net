"""Build and store the TinyLLM observed cyclic action-semantics null."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from neural_architecture_lab.core import ExperimentResult


META_SCHEMA = "nal.meta-hypothesis.v1"
EXPERIMENT_SCHEMA = "nal.tinyllm-observed-cyclic-action-semantics-front.v1"
STAGE_A_SCHEMA = f"{EXPERIMENT_SCHEMA}.stage-a"
HYPOTHESIS_ID = "tinyllm-observed-cyclic-action-semantics-front-v1"
EVIDENCE_ROLE = "preregistered_staged_frozen_action_semantics_intervention"
CAMPAIGN_SHA256 = (
    "022fa4256c37555dd267f2be94d6a3eec2f50dc03e2f1b1e35e166b3d64e1815"
)
IMPLEMENTATION_SHA256 = (
    "dd7bbf10594cf5129acb65977f48108916304087acf7201527e5bee681679f4b"
)
RUNNER_SHA256 = (
    "f8342f897b7e151949c9e7f682f4125644bb05af276ff517022310616e7574e8"
)
STAGE_A_SHA256 = (
    "652bf9d082cfe8b8cd997724f4b1d36b82e46aead2283354fc6e376c981c138e"
)
STAGE_A_ARRAYS_SHA256 = (
    "7f0997be3741a58feebb003d453f0aa81b9e2800627aa47209307ee087128a6d"
)
RESULT_MANIFEST_SHA256 = (
    "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
)
PREREGISTRATION_SHA256 = (
    "8c649c67500fecd3bf672e7a879f83c8af316f838237ca051db7539e2d25977d"
)
SOURCE_LADDER_SHA256 = (
    "cf12b76691da41b7bc15e47570bce324f6aaefc7c9f670ef68db1fa4d9421046"
)
SOURCE_DECK_SHA256 = (
    "a3c14ce7022b7301344beaca876e0d454445c972a57de69c9cd4cd89098036b3"
)
SOURCE_OBSERVED_SHA256 = (
    "7a1b099495f7ecb6c3eeea7c9b836411a5baee709eb6b51ab8103f88927e8a86"
)
DEGREES = (2, 3)
SEEDS = (7, 17, 29, 41, 53)
REGIMES = ("composition", "extrapolation")
SELECTABLE_VARIANTS = (
    "residual_fixed_continuous",
    "residual_fixed_requantized",
)
ORACLE_VARIANTS = (
    "oracle_residual_fixed_continuous",
    "oracle_residual_fixed_requantized",
)
BASELINE_VARIANTS = ("rotate_all_continuous", "rotate_all_requantized")
EXPECTED_REDUCTIONS = {
    "residual_fixed_continuous": {
        "k2_composition": -7.37350066673242,
        "k2_extrapolation": 0.4515139507849143,
        "k3_composition": 0.08790610017583744,
        "k3_extrapolation": 0.4759472619077195,
    },
    "residual_fixed_requantized": {
        "k2_composition": 1.0,
        "k2_extrapolation": 1.0,
        "k3_composition": -0.049412803466609034,
        "k3_extrapolation": 0.37986174112462656,
    },
    "oracle_residual_fixed_continuous": {
        "k2_composition": -6.485987179036617,
        "k2_extrapolation": 0.5042858936567461,
        "k3_composition": 0.09107608950708623,
        "k3_extrapolation": 0.521610927068636,
    },
    "oracle_residual_fixed_requantized": {
        "k2_composition": 1.0,
        "k2_extrapolation": 1.0,
        "k3_composition": -0.05454450985745707,
        "k3_extrapolation": 0.42114504545949083,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _source_campaign(config: Mapping[str, Any], key: str) -> Path:
    return Path(config[key]) / "campaign_results.json"


def _load_campaign(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = json.loads(path.read_text(encoding="utf-8"))
    config = campaign.get("configuration", {})
    aggregate = campaign.get("aggregates", {})
    selection = campaign.get("stage_a", {}).get("selection", {})
    artifacts = campaign.get("artifacts", {})
    provenance = campaign.get("provenance", {})
    summary = campaign.get("summary", {})
    stage_path = Path(artifacts.get("stage_a", ""))
    arrays_path = Path(artifacts.get("stage_a_arrays", ""))
    preregistration = Path(provenance.get("preregistration", ""))
    runner = Path(
        "experiments/structure_net/"
        "tinyllm_observed_cyclic_action_semantics.py"
    )
    source_specs = (
        ("source_ladder_root", "source_ladder_campaign_sha256", SOURCE_LADDER_SHA256),
        ("source_deck_root", "source_deck_campaign_sha256", SOURCE_DECK_SHA256),
        ("source_observed_root", "source_observed_campaign_sha256", SOURCE_OBSERVED_SHA256),
    )
    expected_summary = {
        "requested": 0,
        "scheduled": 0,
        "completed": 0,
        "failed": 0,
        "excluded": 0,
        "retries": 0,
        "reused": 0,
        "trained_models": 0,
        "trained_frontends": 0,
        "trained_task_heads": 0,
        "fitted_probes": 0,
        "fitted_observers": 0,
        "fitted_action_parameters": 0,
    }
    if (
        _sha256(path) != CAMPAIGN_SHA256
        or campaign.get("schema_version") != EXPERIMENT_SCHEMA
        or campaign.get("hypothesis_id") != HYPOTHESIS_ID
        or campaign.get("status") != "completed"
        or campaign.get("evidence_role") != EVIDENCE_ROLE
        or campaign.get("implementation_sha256") != IMPLEMENTATION_SHA256
        or campaign.get("stage_a_sha256") != STAGE_A_SHA256
        or campaign.get("result_manifest_sha256") != RESULT_MANIFEST_SHA256
        or tuple(config.get("degrees", ())) != DEGREES
        or tuple(config.get("seeds", ())) != SEEDS
        or int(config.get("evaluation_orbits", -1)) != 256
        or int(config.get("map_points", -1)) != 192
        or config.get("allow_underpowered") is not False
        or config.get("stage_a_only") is not False
        or config.get("forced_variant") is not None
        or summary != expected_summary
        or aggregate.get("classification")
        != "no_observable_action_semantics_candidate"
        or aggregate.get("primary_hypothesis_pass") is not False
        or aggregate.get("valid") is not True
        or aggregate.get("selected_variant") is not None
        or aggregate.get("causal_stage_authorized") is not False
        or campaign.get("results") != []
        or selection.get("selected_variant") is not None
        or selection.get("eligible_variants") != []
        or selection.get("causal_stage_authorized") is not False
        or selection.get("requantization_alone_eligible") is not False
        or not stage_path.is_file()
        or _sha256(stage_path) != STAGE_A_SHA256
        or not arrays_path.is_file()
        or _sha256(arrays_path) != STAGE_A_ARRAYS_SHA256
        or artifacts.get("stage_a_arrays_sha256") != STAGE_A_ARRAYS_SHA256
        or not preregistration.is_file()
        or _sha256(preregistration) != PREREGISTRATION_SHA256
        or not runner.is_file()
        or _sha256(runner) != RUNNER_SHA256
        or not _finite(campaign)
    ):
        raise ValueError(f"invalid observed action-semantics campaign {path}")
    for root_key, provenance_key, expected_hash in source_specs:
        source_path = _source_campaign(config, root_key)
        if (
            provenance.get(provenance_key) != expected_hash
            or not source_path.is_file()
            or _sha256(source_path) != expected_hash
        ):
            raise ValueError(f"invalid action-semantics source {source_path}")

    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    variants = stage.get("variants", {})
    expected_variants = set(SELECTABLE_VARIANTS + ORACLE_VARIANTS + BASELINE_VARIANTS)
    if (
        stage.get("schema_version") != STAGE_A_SCHEMA
        or stage.get("status") != "completed"
        or stage.get("finite") is not True
        or stage.get("valid") is not False
        or len(stage.get("cells", {})) != 20
        or set(variants) != expected_variants
        or any(item.get("eligible") is not False for item in variants.values())
        or stage.get("selection") != selection
        or not _finite(stage)
    ):
        raise ValueError(f"invalid observed action-semantics Stage A {stage_path}")
    for variant, cells in EXPECTED_REDUCTIONS.items():
        observed = variants[variant].get("by_degree_shift", {})
        if set(observed) != set(cells):
            raise ValueError(f"invalid action-semantics cells for {variant}")
        for cell, expected in cells.items():
            actual = float(observed[cell]["distance_reduction_from_rotate_all"])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"invalid action-semantics reduction {variant}/{cell}")
    return campaign, stage


def _cell_summary(
    stage: Mapping[str, Any], k: int, seed: int, regime: str
) -> dict[str, float]:
    cell = stage["cells"][f"k{k}_seed{seed}_{regime}"]
    metrics: dict[str, float] = {}
    baseline = float(
        cell["rotate_all_continuous"]["generator_orbit_planar_relative_rms"][
            "median"
        ]
    )
    metrics["rotate_all_continuous_distance_median"] = baseline
    for variant in SELECTABLE_VARIANTS + ORACLE_VARIANTS + ("rotate_all_requantized",):
        item = cell[variant]
        distance = float(item["generator_orbit_planar_relative_rms"]["median"])
        metrics[f"{variant}_distance_median"] = distance
        metrics[f"{variant}_distance_reduction"] = float(
            1.0 - distance / max(baseline, 1e-12)
        )
        metrics[f"{variant}_character_median"] = float(
            item["task_character_angular_error"]["median"]
        )
        metrics[f"{variant}_character_p95"] = float(
            item["task_character_angular_error"]["p95"]
        )
        metrics[f"{variant}_composition_maximum"] = float(
            item["composition_planar_relative_rms"]["maximum"]
        )
        metrics[f"{variant}_norm_p95"] = float(
            item["corrected_norm_relative_error"]["p95"]
        )
    metrics["observable_candidate_selected"] = 0.0
    metrics["causal_model_evaluation_run"] = 0.0
    return metrics


def build_observed_cyclic_action_semantics_meta_hypothesis(
    results_path: Path,
    report_path: Path = Path(
        "docs/08 - Analysis/"
        "2026-08-10_tinyllm-observed-cyclic-action-semantics.md"
    ),
) -> dict[str, Any]:
    campaign, stage = _load_campaign(results_path)
    cells = [
        {
            "experiment_id": f"tinyllm_action_semantics_k{k}_seed{seed}_{regime}",
            "k": k,
            "seed": seed,
            "regime": regime,
            "evidence_role": EVIDENCE_ROLE,
            **_cell_summary(stage, k, seed, regime),
        }
        for k in DEGREES
        for seed in SEEDS
        for regime in REGIMES
    ]
    return {
        "schema_version": META_SCHEMA,
        "hypothesis": {
            "id": HYPOTHESIS_ID,
            "name": "TinyLLM observed cyclic action-semantics front test",
            "category": "mechanistic_interpretability",
            "description": (
                "An input-only staged diagnostic tests whether observable residual "
                "transport or requantization can recover the generator-defined "
                "C2/C3 orbit semantics before any frozen-model evaluation."
            ),
            "question": (
                "Can a no-fit action constructed from one decoded observation "
                "match the separately quantized, shared-noise generator orbit well "
                "enough to justify causal-front replication?"
            ),
            "prediction": (
                "At least one observable residual-fixed action passes every locked "
                "input gate for both degrees and shifts, restoring authorization "
                "for the frozen causal stage."
            ),
            "created_at": campaign["completed_at"],
            "tested": True,
            "confirmed": False,
            "confirmation_status": (
                "preregistered_stage_a_null_no_observable_candidate_"
                "causal_stage_not_run"
            ),
            "evidence_count": 20,
            "direct_experiment_count": 20,
            "loaded_checkpoint_count": 0,
            "referenced_checkpoint_cohort_count": 10,
            "evidence_role": EVIDENCE_ROLE,
            "power_profile": "two_degree_five_cohort_two_shift_input_only_stop_rule",
            "tested_scope": (
                "twenty N3 generator cohorts from retained d6 C2/C3 seeds, six "
                "fixed action constructors, no model load or fitted parameter"
            ),
            "subclaims": {
                "observable_residual_fixed_action_eligible": "no_zero_of_two",
                "requantization_alone_explains_gap": "no",
                "phase_estimation_alone_explains_gap": "no_oracle_also_fails",
                "generator_front_observable_from_one_decoded_sheet": (
                    "not_established_under_registered_constructors"
                ),
                "causal_stage_authorized": "no",
                "new_training_licensed": "no",
                "front_should_be_action_semantics_qualified": "yes",
            },
            "tags": [
                "tinyllm",
                "observed-action",
                "C2",
                "C3",
                "quantization",
                "noise-transport",
                "identifiability",
                "input-only",
                "preregistered-stop",
                "falsification",
            ],
            "explicitly_not_tested": campaign["method_boundaries"],
        },
        "result": {
            "confirmed": False,
            "confirmation_reason": (
                "Neither observable residual-fixed constructor passes the locked "
                "four-cell input gate. Requantization alone is ineligible, and "
                "latent-phase oracle residual transport also fails, so Stage B "
                "correctly stops before model loading."
            ),
            "confidence": 0.99,
            "confidence_assessment": (
                "strong_preregistered_input_only_null_with_oracle_attribution"
            ),
            "num_direct_experiments": 20,
            "descriptive_metrics": {
                "campaign": campaign["aggregates"],
                "selection": stage["selection"],
                "variants": stage["variants"],
            },
            "key_insights": [
                "Observable residual fixing improves extrapolation distances but does not clear all C2/C3 composition and geometry gates.",
                "Requantization exactly matches some C2 medians yet worsens or barely changes C3, so it is not the general explanation.",
                "The latent-phase oracle also fails character and norm gates, ruling out demodulation error as the sole cause.",
                "A decoded anchor cannot recover the pre-quantization noise realization needed to reproduce the historical generator orbit exactly.",
                "Mature observable quotient sufficiency remains valid; only equivalence to the older generator-defined front is unsupported.",
            ],
            "suggested_hypotheses": [
                "Causal-front locations should be indexed by an explicitly declared observable action rather than by an unobservable generator coupling.",
                "Front intervals across several valid observable actions will be more stable than one exact generator-defined cut.",
                "A prospective generator that emits the pre-quantized observation or its noise reference can test generator-front recovery without hidden coupling ambiguity.",
            ],
            "completed_at": campaign["completed_at"],
        },
        "evidence": {"input_cells": cells},
        "source_artifacts": [str(results_path), str(report_path)],
        "provenance": {
            "campaign_sha256": CAMPAIGN_SHA256,
            "implementation_sha256": IMPLEMENTATION_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "stage_a_sha256": STAGE_A_SHA256,
            "stage_a_arrays_sha256": STAGE_A_ARRAYS_SHA256,
            "result_manifest_sha256": RESULT_MANIFEST_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "source_ladder_campaign_sha256": SOURCE_LADDER_SHA256,
            "source_deck_campaign_sha256": SOURCE_DECK_SHA256,
            "source_observed_campaign_sha256": SOURCE_OBSERVED_SHA256,
        },
    }


def build_observed_cyclic_action_semantics_experiment_results(
    record: Mapping[str, Any], results_path: Path
) -> list[ExperimentResult]:
    del record
    campaign, stage = _load_campaign(results_path)
    completed = datetime.fromisoformat(
        campaign["completed_at"].replace("Z", "+00:00")
    )
    return [
        ExperimentResult(
            experiment_id=f"tinyllm_action_semantics_k{k}_seed{seed}_{regime}",
            hypothesis_id=HYPOTHESIS_ID,
            metrics=_cell_summary(stage, k, seed, regime),
            primary_metric=0.0,
            model_architecture=[],
            model_parameters=0,
            training_time=float(campaign["analysis_seconds"]) / 20.0,
            model_checkpoint=None,
            observations=[
                f"Input-only C{k} {regime} action-semantic cell, seed {seed}.",
                "No TinyLLM checkpoint, activation, output, probe, or fitted action was loaded.",
                "The locked Stage-A stop rule forbade causal evaluation.",
                f"Implementation SHA-256: {IMPLEMENTATION_SHA256}.",
            ],
            anomalies=[
                "No observable residual-fixed constructor passed every registered input gate."
            ],
            timestamp=completed,
        )
        for k in DEGREES
        for seed in SEEDS
        for regime in REGIMES
    ]


def store_observed_cyclic_action_semantics_meta_hypothesis(
    results_path: Path,
    output_path: Path,
    *,
    chromadb_path: Path | None = None,
) -> dict[str, Any]:
    record = build_observed_cyclic_action_semantics_meta_hypothesis(results_path)
    experiments = build_observed_cyclic_action_semantics_experiment_results(
        record, results_path
    )
    storage: dict[str, Any] = {
        "aggregate_path": str(output_path),
        "experiment_ids": [item.experiment_id for item in experiments],
    }
    if chromadb_path is not None:
        from structure_net.logging.standardized_logging import (
            LoggingConfig,
            StandardizedLogger,
        )

        root = chromadb_path.parent
        logger = StandardizedLogger(
            LoggingConfig(
                project_name="structure_net_meta_hypotheses",
                queue_dir=str(root / "experiment_queue"),
                sent_dir=str(root / "experiment_sent"),
                rejected_dir=str(root / "experiment_rejected"),
                enable_wandb=False,
                auto_upload=False,
                enable_chromadb=True,
                chromadb_path=str(chromadb_path),
            )
        )
        logger.log_hypothesis(record["hypothesis"])
        storage["result_hashes"] = [
            logger.log_experiment_result(item) for item in experiments
        ]
        storage["chromadb_path"] = str(chromadb_path)
        hypothesis = logger.hypotheses_collection.get(
            ids=[HYPOTHESIS_ID], include=["metadatas"]
        )
        results = logger.experiments_collection.get(
            ids=storage["result_hashes"], include=["metadatas"]
        )
        if (
            hypothesis.get("ids") != [HYPOTHESIS_ID]
            or len(results.get("ids", [])) != len(experiments)
            or {item.get("hypothesis_id") for item in results.get("metadatas", [])}
            != {HYPOTHESIS_ID}
        ):
            raise RuntimeError("observed action-semantics ChromaDB read-back failed")
        storage["readback"] = {
            "verified": True,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_count": len(results["ids"]),
        }
    record["storage"] = storage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


__all__ = [
    "build_observed_cyclic_action_semantics_experiment_results",
    "build_observed_cyclic_action_semantics_meta_hypothesis",
    "store_observed_cyclic_action_semantics_meta_hypothesis",
]
