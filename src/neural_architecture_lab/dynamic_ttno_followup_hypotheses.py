"""Pending NAL registrations for the A5--A7 post-A4 studies."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core import Hypothesis, HypothesisCategory


REGISTRY_SCHEMA = "nal.pending-hypothesis-registry.v1"
REGISTERED_AT = datetime(2026, 8, 30, tzinfo=timezone.utc)

A5_HYPOTHESIS_ID = "tinyllm-ttno-cut-localization-parity-v1"
A6_HYPOTHESIS_ID = "tinyllm-hss-shared-basis-nesting-v1"
A7_HYPOTHESIS_ID = "tinyllm-causal-h2-attention-v1"
PARENT_HYPOTHESIS_ID = "tinyllm-dynamic-ttno-rank-pilot-v1"

A4_AGGREGATE_PATH = (
    "data/experiments/tinyllm_dynamic_ttno_rank/"
    "20260829_d8_babylm_pilot/campaign_results.json"
)
A4_AGGREGATE_SHA256 = (
    "9d3fa8ec7332860785b3d62dff5805e4ec23c9f6fedc5178c6809b15cd05feba"
)
A4_IMPLEMENTATION_SHA256 = (
    "ffdf9bb77449a4dcad6c67f111b70a3543eae42495ab067044835d13bf65c8fb"
)
CHECKPOINT_SHA256 = (
    "5a7491b4231c30feaaabda7babf0a831dd4d72fbb4e7f3e757114cadf098df09"
)
TOKEN_STREAM_SHA256 = (
    "f339655453b970ae6cc1cbdc7c78f8a5234c42437bc7bba8fb31a1dee5c9d765"
)

A5_PREREGISTRATION = (
    "docs/07 - Status Reports/"
    "2026-08-30_tinyllm-ttno-cut-localization-parity-preregistration.md"
)
A6_PREREGISTRATION = (
    "docs/07 - Status Reports/"
    "2026-08-30_tinyllm-hss-shared-basis-nesting-preregistration.md"
)
A7_PREREGISTRATION = (
    "docs/07 - Status Reports/"
    "2026-08-30_tinyllm-causal-h2-attention-preregistration.md"
)
A5_PREREGISTRATION_SHA256 = (
    "9d059361d84d07fa27da35a0833253a04bc3757534158b3d99b0ae2380fc6d24"
)
A6_PREREGISTRATION_SHA256 = (
    "dd3abe846f066b9188150903b6bafbc7d78d77b29d87683bbbb858dabeffdb35"
)
A7_PREREGISTRATION_SHA256 = (
    "f222389b56280ac09cd31168967b4c60e03e9179128cee83e5d36b4d72d90954"
)

FROZEN_EVALUATION_SEEDS = [101, 211, 307, 401, 503]
FROZEN_LAYERS = list(range(8))
FROZEN_HEADS = [0, 3, 7]
FROZEN_LENGTHS = [32, 64, 128, 256]
FROZEN_EPSILONS = [1e-2, 1e-3]


def _pending(study: str) -> RuntimeError:
    return RuntimeError(
        f"{study} is registered but has not been implemented or run; "
        "execute only against its frozen preregistration."
    )


def run_a5_pending(_config: dict[str, Any]) -> None:
    """Prevent an incomplete A5 registration from being run accidentally."""

    raise _pending("A5")


def run_a6_pending(_config: dict[str, Any]) -> None:
    """Prevent an incomplete A6 registration from being run accidentally."""

    raise _pending("A6")


def run_a7_pending(_config: dict[str, Any]) -> None:
    """Prevent an incomplete A7 registration from being run accidentally."""

    raise _pending("A7")


def _parent_controls() -> dict[str, Any]:
    return {
        "parent_hypothesis_id": PARENT_HYPOTHESIS_ID,
        "parent_aggregate_path": A4_AGGREGATE_PATH,
        "parent_aggregate_sha256": A4_AGGREGATE_SHA256,
        "parent_implementation_sha256": A4_IMPLEMENTATION_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "token_stream_sha256": TOKEN_STREAM_SHA256,
        "evaluation_seed_semantics": "input-selection replicates, not model seeds",
        "evaluation_seeds": FROZEN_EVALUATION_SEEDS,
        "layers": FROZEN_LAYERS,
        "heads": FROZEN_HEADS,
        "lengths": FROZEN_LENGTHS,
        "rank_relative_frobenius_epsilons": FROZEN_EPSILONS,
        "primary_rank_relative_frobenius_epsilon": 1e-2,
        "frozen_cell_count": 120,
        "status": "registered_not_run",
    }


def build_a5_hypothesis() -> Hypothesis:
    """Build the frozen A5 cut-localization/parity-control registration."""

    controls = _parent_controls()
    controls.update(
        {
            "menu_id": "A5",
            "planned_schema": "nal.tinyllm-ttno-cut-localization-parity.v1",
            "preregistration": A5_PREREGISTRATION,
            "preregistration_sha256": A5_PREREGISTRATION_SHA256,
            "topology_arms": [
                "msb_balanced",
                "lsb_balanced",
                "odd_even_modes",
                "gray_token_order",
                "zero_pad_128_to_256",
                "duplicate_128_to_256",
                "natural_256",
            ],
            "primary_endpoints": [
                "zero_artifact_fraction",
                "duplicate_artifact_fraction",
                "topology_reduction",
            ],
            "decision_order": [
                "invalid_parent_or_tensor_contract",
                "new_cut_artifact_dominant",
                "bit_topology_sensitive",
                "intrinsic_operator_rank_growth",
                "mixed_cut_and_operator_effect",
            ],
            "natural_512_extension": "conditional_new_checkpoint_only",
        }
    )
    return Hypothesis(
        id=A5_HYPOTHESIS_ID,
        name="TinyLLM TTNO cut localization and parity controls",
        description=(
            "Post-A4 diagnostic separating intrinsic attention-operator rank "
            "growth from the new 256-token paired-bit cut and fixed topology."
        ),
        category=HypothesisCategory.ARCHITECTURE,
        question=(
            "Is the A4 128-to-256 paired-bit TTNO rank cliff intrinsic, a "
            "new-cut artifact, or sensitive to a fixed bit-mode topology?"
        ),
        prediction=(
            "Frozen parity controls and alternative bit-mode topologies will "
            "localize the observed cliff under the preregistered decision order."
        ),
        test_function=run_a5_pending,
        parameter_space={"evaluation_seed": FROZEN_EVALUATION_SEEDS},
        control_parameters=controls,
        success_metrics={
            "valid_parent_and_tensor_contract": 1.0,
            "preregistered_cause_classified": 1.0,
        },
        created_at=REGISTERED_AT,
        tags=[
            "A5",
            "TinyLLM",
            "TTNO",
            "cut-localization",
            "parity-control",
            "pending",
        ],
        references=[A4_AGGREGATE_PATH, A5_PREREGISTRATION],
        tested=False,
    )


def build_a6_hypothesis() -> Hypothesis:
    """Build the frozen A6 shared-basis/nesting diagnostic registration."""

    controls = _parent_controls()
    controls.update(
        {
            "menu_id": "A6",
            "planned_schema": "nal.tinyllm-hss-shared-basis-nesting.v1",
            "preregistration": A6_PREREGISTRATION,
            "preregistration_sha256": A6_PREREGISTRATION_SHA256,
            "tree": "chronological_dyadic",
            "peer_partition": "siblings_on_cluster_to_root_path",
            "orientations": ["query", "key"],
            "primary_endpoints": [
                "sharing_inflation",
                "nesting_defect",
                "augmented_shared_rank_ratio",
            ],
            "spectrally_stable_cut_gap_minimum": 2.0,
            "nesting_minimum_stable_cut_fraction": 0.25,
            "controls": [
                "causal_uniform",
                "smooth_fourier",
                "iid_qk",
                "seeded_simultaneous_token_permutation",
                "exact_rank_one",
            ],
            "decision_order": [
                "invalid_parent_or_boundary_contract",
                "shared_and_nested_hierarchy_supported",
                "shared_basis_bottleneck",
                "nesting_bottleneck",
                "combined_sharing_and_nesting_bottleneck",
                "shared_basis_result_nesting_indeterminate",
            ],
        }
    )
    return Hypothesis(
        id=A6_HYPOTHESIS_ID,
        name="TinyLLM HSS shared-basis and nesting diagnostic",
        description=(
            "Post-A4 diagnostic testing whether low chronological HSS boundary "
            "ranks admit compact shared cluster bases nested across the tree."
        ),
        category=HypothesisCategory.ARCHITECTURE,
        question=(
            "Do A4's individually low-rank dyadic interactions share one "
            "compact basis per cluster, and are parent-child bases nested?"
        ),
        prediction=(
            "Shared-boundary ranks and stable-cut nesting defects will identify "
            "whether sharing, nesting, or both obstruct a constructive H2 model."
        ),
        test_function=run_a6_pending,
        parameter_space={"evaluation_seed": FROZEN_EVALUATION_SEEDS},
        control_parameters=controls,
        success_metrics={
            "valid_parent_and_boundary_contract": 1.0,
            "shared_basis_compactness": 1.0,
            "nesting_fidelity": 1.0,
            "nested_rank_compactness": 1.0,
        },
        created_at=REGISTERED_AT,
        tags=[
            "A6",
            "TinyLLM",
            "HSS",
            "H2",
            "shared-basis",
            "nesting",
            "pending",
        ],
        references=[A4_AGGREGATE_PATH, A6_PREREGISTRATION],
        tested=False,
    )


def build_a7_hypothesis() -> Hypothesis:
    """Build the frozen A7 constructive causal H2 registration."""

    controls = _parent_controls()
    controls.pop("rank_relative_frobenius_epsilons")
    controls.pop("primary_rank_relative_frobenius_epsilon")
    controls.update(
        {
            "menu_id": "A7",
            "planned_schema": "nal.tinyllm-causal-h2-attention.v1",
            "preregistration": A7_PREREGISTRATION,
            "preregistration_sha256": A7_PREREGISTRATION_SHA256,
            "diagnostic_predecessor_id": A6_HYPOTHESIS_ID,
            "diagnostic_predecessor_can_tune_primary": False,
            "operator": {
                "target": "causal_stabilized_unnormalized_kernel",
                "dtype": "float64",
                "normalization": "recomputed_from_h2_kernel",
                "augmented_value_channel": True,
                "posthoc_clipping": False,
                "posthoc_row_sum_repair": False,
                "sparse_correction": False,
            },
            "tree": {
                "ordering": "chronological",
                "topology": "balanced_binary",
                "leaf_size": 16,
            },
            "admissibility": {
                "future_blocks": "exact_zero",
                "fully_past_required": True,
                "gap_rule": "gap_tokens >= max(query_size, key_size)",
                "mixed_or_near_blocks": "exact_dense",
                "numerical_rank_based": False,
            },
            "partition_integrity": {
                "32": {
                    "admissible": 0,
                    "dense": 3,
                    "zero": 1,
                    "sha256": (
                        "8e753f6334cd7d928b78b01fa4acc4310181ca053a81d42b72cd71c67b49d7e5"
                    ),
                },
                "64": {
                    "admissible": 3,
                    "dense": 7,
                    "zero": 3,
                    "sha256": (
                        "5a5a87fb6313c2dda8d7bdb1ce818ed8f81b931558cb9328f4b3aa3c83b77a76"
                    ),
                },
                "128": {
                    "admissible": 12,
                    "dense": 15,
                    "zero": 7,
                    "sha256": (
                        "97ac50d907bdfe6518af2e6d313bb62c6aebd284e2e8c7819955a8a763939600"
                    ),
                },
                "256": {
                    "admissible": 33,
                    "dense": 31,
                    "zero": 15,
                    "sha256": (
                        "1535f675b18b8493d5da1c4e1048eefb73c884e700953451de7f9b8b380688f2"
                    ),
                },
                "512": {
                    "admissible": 78,
                    "dense": 63,
                    "zero": 31,
                    "sha256": (
                        "79d9f9d12ff42883224ae0d0629933718503b800567203e04712f8765c3924e3"
                    ),
                },
            },
            "construction": {
                "method": "deterministic_nested_svd",
                "dtype": "float64",
                "basis_weighting": "row_mass_relative",
                "build_tolerance": 0.0025,
                "local_squared_error_budget": (
                    "build_tolerance^2 / (2 * tree_levels)"
                ),
                "rank_cap": "ceil(log2(sequence_length)^2)",
                "per_cell_rank_retuning": False,
                "global_error_retuning": False,
            },
            "probe": {
                "seed": 1707,
                "bit_generator": "numpy.random.PCG64",
                "columns": 32,
                "distribution": "2 * integers(0, 2) - 1",
                "scope": "reinitialize_per_length_and_reuse_across_cells",
            },
            "integrity_length": 32,
            "primary_lengths": [64, 128, 256],
            "extension_length": 512,
            "extension_requires_new_fingerprinted_checkpoint": True,
            "primary_gates": {
                "kernel_row_relative_max": 0.01,
                "denominator_relative_max": 0.01,
                "positive_approximate_denominator": True,
                "attention_row_l1_max": 0.025,
                "probe_relative_frobenius_max": 0.02,
                "value_output_relative_frobenius_max": 0.02,
                "token_output_p99_global_rms_normalized_max": 0.05,
            },
            "campaign_gates": {
                "required_cell_pass_fraction_each_primary_length": 0.80,
                "required_min_layer_pass_fraction_at_max_length": 0.50,
            },
            "compression_gates": {
                "storage_ratio_median_max": 0.75,
                "storage_ratio_p90_max": 1.0,
                "multiply_add_ratio_median_max": 0.75,
                "multiply_add_ratio_p90_max": 1.0,
            },
            "decision_order": [
                "invalid_h2_construction_contract",
                "h2_normalization_path_failed",
                "h2_representation_failed",
                "h2_layer_selective_only",
                "h2_representation_pass",
            ],
            "compression_labels": [
                "h2_constructive_compression_pass",
                "h2_representation_pass_no_finite_size_compression",
            ],
            "oracle_arm": "direct_h2_of_normalized_attention_diagnostic_only",
            "sensitivity_can_rescue_primary": False,
        }
    )
    return Hypothesis(
        id=A7_HYPOTHESIS_ID,
        name="TinyLLM constructive causal H2 attention",
        description=(
            "Construct one simultaneous strong-admissibility H2 approximation "
            "of each frozen causal attention kernel with shared nested bases."
        ),
        category=HypothesisCategory.ARCHITECTURE,
        question=(
            "Do well-separated chronological causal interactions admit a "
            "single accurate shared and recursively nested H2 representation?"
        ),
        prediction=(
            "The fixed kernel-normalized H2 construction will determine whether "
            "the 120 frozen cells pass both representation and finite-size gates."
        ),
        test_function=run_a7_pending,
        parameter_space={"evaluation_seed": FROZEN_EVALUATION_SEEDS},
        control_parameters=controls,
        success_metrics={
            "valid_h2_construction_contract": 1.0,
            "primary_length_pass_fraction": 0.80,
            "minimum_layer_pass_fraction": 0.50,
            "representation_pass": 1.0,
        },
        created_at=REGISTERED_AT,
        tags=[
            "A7",
            "TinyLLM",
            "H2",
            "causal-attention",
            "strong-admissibility",
            "constructive",
            "pending",
        ],
        references=[
            A4_AGGREGATE_PATH,
            A7_PREREGISTRATION,
            "https://arxiv.org/html/2506.16759",
            "https://arxiv.org/html/2310.11960v3",
        ],
        tested=False,
    )


def get_dynamic_ttno_followup_hypotheses() -> list[Hypothesis]:
    """Return A5 through A7 in their fixed registration order."""

    return [build_a5_hypothesis(), build_a6_hypothesis(), build_a7_hypothesis()]


def _registry_hypothesis_record(hypothesis: Hypothesis) -> dict[str, Any]:
    preregistration = hypothesis.control_parameters["preregistration"]
    return {
        "id": hypothesis.id,
        "menu_id": hypothesis.control_parameters["menu_id"],
        "name": hypothesis.name,
        "description": hypothesis.description,
        "category": hypothesis.category.value,
        "question": hypothesis.question,
        "prediction": hypothesis.prediction,
        "test_function": (
            f"{hypothesis.test_function.__module__}."
            f"{hypothesis.test_function.__name__}"
        ),
        "parameter_space": hypothesis.parameter_space,
        "control_parameters": hypothesis.control_parameters,
        "success_metrics": hypothesis.success_metrics,
        "statistical_significance": hypothesis.statistical_significance,
        "created_at": hypothesis.created_at.isoformat(),
        "tags": hypothesis.tags,
        "references": hypothesis.references,
        "preregistration": preregistration,
        "status": "registered_not_run",
        "tested": hypothesis.tested,
        "result_count": len(hypothesis.results),
    }


def register_dynamic_ttno_followups(
    lab: Any,
    output_path: Path,
) -> dict[str, Any]:
    """Register all follow-ups, verify ChromaDB read-back, and write a ledger."""

    hypotheses = get_dynamic_ttno_followup_hypotheses()
    lab.register_hypothesis_batch(hypotheses)

    expected_ids = [item.id for item in hypotheses]
    if any(item_id not in lab.hypotheses for item_id in expected_ids) or any(
        item_id not in lab.pending_hypotheses for item_id in expected_ids
    ):
        raise RuntimeError("A5--A7 in-memory pending registration read-back failed")

    collection = getattr(lab.logger, "hypotheses_collection", None)
    if collection is None:
        raise RuntimeError("A5--A7 registration requires ChromaDB persistence")
    readback = collection.get(ids=expected_ids, include=["metadatas"])
    readback_ids = readback.get("ids", [])
    metadata_by_id = {
        item["id"]: item for item in readback.get("metadatas", []) if "id" in item
    }
    if set(readback_ids) != set(expected_ids) or any(
        metadata_by_id.get(item_id, {}).get("tested") is not False
        for item_id in expected_ids
    ):
        raise RuntimeError("A5--A7 ChromaDB registration read-back failed")

    record = {
        "schema": REGISTRY_SCHEMA,
        "registered_at": REGISTERED_AT.isoformat(),
        "status": "registered_not_run",
        "parent_hypothesis_id": PARENT_HYPOTHESIS_ID,
        "hypotheses": [
            _registry_hypothesis_record(hypothesis) for hypothesis in hypotheses
        ],
        "storage": {
            "registry_path": str(output_path),
            "chromadb_path": str(lab.results_dir / "chroma_db"),
            "collection": "hypotheses",
            "readback": {
                "verified": True,
                "hypothesis_ids": expected_ids,
                "tested": False,
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return record


__all__ = [
    "A5_HYPOTHESIS_ID",
    "A6_HYPOTHESIS_ID",
    "A7_HYPOTHESIS_ID",
    "REGISTRY_SCHEMA",
    "build_a5_hypothesis",
    "build_a6_hypothesis",
    "build_a7_hypothesis",
    "get_dynamic_ttno_followup_hypotheses",
    "register_dynamic_ttno_followups",
]
