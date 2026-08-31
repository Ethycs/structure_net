#!/usr/bin/env python3
"""Execute preregistered A6 HSS shared-basis and nesting diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

try:
    from experiments.structure_net.tinyllm_dynamic_ttno_rank import (
        _load_model,
        _sha256,
        _validation_segment,
        causal_attention,
        control_qk,
        extract_selected_qk,
        numerical_rank_from_singular_values,
    )
    from experiments.structure_net import tinyllm_dynamic_ttno_rank as a4
except ModuleNotFoundError:  # Direct script execution.
    from tinyllm_dynamic_ttno_rank import (  # type: ignore[no-redef]
        _load_model,
        _sha256,
        _validation_segment,
        causal_attention,
        control_qk,
        extract_selected_qk,
        numerical_rank_from_singular_values,
    )
    import tinyllm_dynamic_ttno_rank as a4  # type: ignore[no-redef]


SCHEMA_VERSION = "nal.tinyllm-hss-shared-basis-nesting.v1"
HYPOTHESIS_ID = "tinyllm-hss-shared-basis-nesting-v1"
PARENT_HYPOTHESIS_ID = "tinyllm-dynamic-ttno-rank-pilot-v1"
PARENT_AGGREGATE_SHA256 = (
    "9d3fa8ec7332860785b3d62dff5805e4ec23c9f6fedc5178c6809b15cd05feba"
)
PARENT_IMPLEMENTATION_SHA256 = (
    "ffdf9bb77449a4dcad6c67f111b70a3543eae42495ab067044835d13bf65c8fb"
)
CHECKPOINT_SHA256 = (
    "5a7491b4231c30feaaabda7babf0a831dd4d72fbb4e7f3e757114cadf098df09"
)
TOKEN_STREAM_SHA256 = (
    "f339655453b970ae6cc1cbdc7c78f8a5234c42437bc7bba8fb31a1dee5c9d765"
)
PREREGISTRATION_SHA256 = (
    "dd3abe846f066b9188150903b6bafbc7d78d77b29d87683bbbb858dabeffdb35"
)


@dataclass(frozen=True)
class SharedBasisNestingConfig:
    checkpoint: str = (
        "data/experiments/tinyllm_babylm_pretrain/"
        "20260812_d8_seed7/checkpoint_step12000.pt"
    )
    token_cache: str = "data/corpora/babylm_10M_bpe16k.tokens.npy"
    parent_aggregate: str = (
        "data/experiments/tinyllm_dynamic_ttno_rank/"
        "20260829_d8_babylm_pilot/campaign_results.json"
    )
    validation_tokens: int = 262_144
    evaluation_seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    lengths: tuple[int, ...] = (32, 64, 128, 256)
    layers: tuple[int, ...] = tuple(range(8))
    heads: tuple[int, ...] = (0, 3, 7)
    epsilons: tuple[float, ...] = (1e-2, 1e-3)
    primary_epsilon: float = 1e-2
    spectral_gap_minimum: float = 2.0
    stable_cut_fraction_minimum: float = 0.25
    sharing_median_maximum: float = 2.0
    sharing_p90_maximum: float = 3.0
    nesting_defect_median_maximum: float = 0.10
    nesting_defect_p90_maximum: float = 0.25
    augmented_ratio_median_maximum: float = 1.5
    augmented_ratio_p90_maximum: float = 2.0
    permutation_seed_salt: int = 61_703
    device: str = "cuda:0"
    shakedown: bool = False

    def __post_init__(self) -> None:
        if not self.evaluation_seeds or not self.layers or not self.heads:
            raise ValueError("evaluation seeds, layers, and heads must be non-empty")
        if tuple(sorted(set(self.lengths))) != self.lengths:
            raise ValueError("lengths must be unique and increasing")
        if any(length < 4 or length & (length - 1) for length in self.lengths):
            raise ValueError("lengths must be powers of two")
        if self.primary_epsilon not in self.epsilons:
            raise ValueError("primary epsilon must be one of epsilons")


@dataclass(frozen=True)
class Cluster:
    start: int
    stop: int
    parent: str | None
    depth: int

    @property
    def id(self) -> str:
        return f"{self.start}:{self.stop}"

    @property
    def size(self) -> int:
        return self.stop - self.start


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    for path in (Path(__file__), Path(a4.__file__)):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fingerprint(
    config: SharedBasisNestingConfig,
    seed: int,
    implementation_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "seed": seed,
        "implementation_sha256": implementation_sha256,
        "parent_aggregate_sha256": PARENT_AGGREGATE_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def dyadic_cluster_tree(
    length: int,
) -> tuple[list[Cluster], dict[str, tuple[str, str]], str]:
    """Return preorder clusters, child mapping, and root identifier."""

    if length < 2 or length & (length - 1):
        raise ValueError("length must be a power of two")
    clusters: list[Cluster] = []
    children: dict[str, tuple[str, str]] = {}

    def visit(start: int, stop: int, parent: str | None, depth: int) -> str:
        cluster = Cluster(start, stop, parent, depth)
        clusters.append(cluster)
        if stop - start > 1:
            middle = (start + stop) // 2
            left = visit(start, middle, cluster.id, depth + 1)
            right = visit(middle, stop, cluster.id, depth + 1)
            children[cluster.id] = (left, right)
        return cluster.id

    root = visit(0, length, None, 0)
    return clusters, children, root


def complement_peers(
    cluster_id: str,
    clusters: Mapping[str, Cluster],
    children: Mapping[str, tuple[str, str]],
) -> tuple[str, ...]:
    """Siblings encountered from a cluster to the root."""

    peers: list[str] = []
    current = clusters[cluster_id]
    while current.parent is not None:
        left, right = children[current.parent]
        peers.append(right if current.id == left else left)
        current = clusters[current.parent]
    return tuple(peers)


def _rank(singular_values: np.ndarray, epsilon: float) -> int:
    return numerical_rank_from_singular_values(singular_values, epsilon)


def _svd(
    matrix: torch.Tensor, epsilons: Sequence[float]
) -> tuple[torch.Tensor, np.ndarray, dict[str, int]]:
    if matrix.ndim != 2:
        raise ValueError("SVD input must be a matrix")
    if matrix.shape[1] == 0:
        values = np.empty(0, dtype=np.float64)
        return matrix[:, :0], values, {f"{epsilon:.12g}": 0 for epsilon in epsilons}
    left, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    values = singular.detach().cpu().numpy()
    ranks = {
        f"{epsilon:.12g}": _rank(values, epsilon) for epsilon in epsilons
    }
    return left, values, ranks


def _svd_ranks(
    matrix: torch.Tensor, epsilons: Sequence[float]
) -> dict[str, int]:
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return {f"{epsilon:.12g}": 0 for epsilon in epsilons}
    values = torch.linalg.svdvals(matrix).detach().cpu().numpy()
    return {f"{epsilon:.12g}": _rank(values, epsilon) for epsilon in epsilons}


def _spectral_cut(
    singular_values: np.ndarray,
    rank: int,
    minimum_gap: float,
) -> dict[str, Any]:
    if rank <= 0 or singular_values.size == 0:
        return {
            "rank": rank,
            "retained": 0.0,
            "discarded": 0.0,
            "gap": 0.0,
            "stable": False,
        }
    retained = float(singular_values[rank - 1])
    floor = 1e-12 * float(singular_values[0])
    discarded = float(singular_values[rank]) if rank < len(singular_values) else 0.0
    gap = retained / max(discarded, floor)
    return {
        "rank": rank,
        "retained": retained,
        "discarded": discarded,
        "gap": gap,
        "stable": gap >= minimum_gap,
    }


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "maximum": float(np.max(array)),
    }


def diagnose_operator(
    operator: np.ndarray,
    config: SharedBasisNestingConfig,
    *,
    retain_records: bool = True,
) -> dict[str, Any]:
    """Measure independent/shared ranks and parent-child nesting for one operator."""

    value = np.asarray(operator, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("operator must be square")
    length = len(value)
    cluster_list, children, root = dyadic_cluster_tree(length)
    clusters = {cluster.id: cluster for cluster in cluster_list}
    device = torch.device(config.device)
    tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
    epsilon_key = f"{config.primary_epsilon:.12g}"
    bases: dict[str, dict[str, torch.Tensor]] = {"query": {}, "key": {}}
    cuts: dict[str, dict[str, dict[str, Any]]] = {"query": {}, "key": {}}
    nodes: list[dict[str, Any]] = []
    concatenation_exact = True
    maxima = {key: 0 for key in (f"{epsilon:.12g}" for epsilon in config.epsilons)}

    for cluster in cluster_list:
        if cluster.id == root:
            continue
        peer_ids = complement_peers(cluster.id, clusters, children)
        inside = torch.arange(cluster.start, cluster.stop, device=device)
        peer_indices = [
            torch.arange(clusters[item].start, clusters[item].stop, device=device)
            for item in peer_ids
        ]
        outside = torch.cat(peer_indices)
        query_blocks = [tensor[inside][:, peer] for peer in peer_indices]
        key_blocks = [tensor[peer][:, inside].T for peer in peer_indices]
        query_shared = torch.cat(query_blocks, dim=1)
        key_shared = torch.cat(key_blocks, dim=1)
        query_direct = tensor[inside][:, outside]
        key_direct = tensor[outside][:, inside].T
        exact = torch.equal(query_shared, query_direct) and torch.equal(
            key_shared, key_direct
        )
        concatenation_exact = concatenation_exact and exact

        orientations: dict[str, Any] = {}
        for orientation, blocks, shared in (
            ("query", query_blocks, query_shared),
            ("key", key_blocks, key_shared),
        ):
            independent = [_svd_ranks(block, config.epsilons) for block in blocks]
            left, singular, shared_ranks = _svd(shared, config.epsilons)
            shared_rank = int(shared_ranks[epsilon_key])
            independent_rank = max(
                (int(item[epsilon_key]) for item in independent), default=0
            )
            bases[orientation][cluster.id] = left[:, :shared_rank]
            cuts[orientation][cluster.id] = _spectral_cut(
                singular, shared_rank, config.spectral_gap_minimum
            )
            for key, rank in shared_ranks.items():
                maxima[key] = max(maxima[key], int(rank))
            orientations[orientation] = {
                "independent_peer_ranks": [
                    {
                        "peer": peer_id,
                        "ranks": rank_record,
                    }
                    for peer_id, rank_record in zip(peer_ids, independent)
                ],
                "independent_max_rank": independent_rank,
                "shared_ranks": shared_ranks,
                "sharing_inflation": float(
                    shared_rank / max(independent_rank, 1)
                ),
                "spectral_cut": cuts[orientation][cluster.id],
            }
        nodes.append(
            {
                "cluster": cluster.id,
                "start": cluster.start,
                "stop": cluster.stop,
                "size": cluster.size,
                "depth": cluster.depth,
                "parent": cluster.parent,
                "peers": list(peer_ids),
                "concatenation_exact": exact,
                "orientations": orientations,
            }
        )

    nesting: list[dict[str, Any]] = []
    for cluster in cluster_list:
        if cluster.id == root or cluster.parent == root:
            continue
        if cluster.parent is None:
            continue
        parent = clusters[cluster.parent]
        offset_start = cluster.start - parent.start
        offset_stop = cluster.stop - parent.start
        for orientation in ("query", "key"):
            child_basis = bases[orientation][cluster.id]
            restricted = bases[orientation][parent.id][offset_start:offset_stop]
            restricted_norm = torch.linalg.norm(restricted)
            if restricted.shape[1] == 0:
                defect = 0.0
            elif child_basis.shape[1] == 0:
                defect = 1.0
            else:
                residual = restricted - child_basis @ (child_basis.T @ restricted)
                defect = float(
                    (
                        torch.linalg.norm(residual)
                        / torch.clamp(restricted_norm, min=1e-12)
                    ).item()
                )
            augmented = torch.cat((child_basis, restricted), dim=1)
            augmented_rank = int(
                _svd_ranks(augmented, (config.primary_epsilon,))[epsilon_key]
            )
            child_rank = child_basis.shape[1]
            stable = bool(
                cuts[orientation][cluster.id]["stable"]
                and cuts[orientation][parent.id]["stable"]
            )
            nesting.append(
                {
                    "child": cluster.id,
                    "parent": parent.id,
                    "orientation": orientation,
                    "defect": defect,
                    "child_shared_rank": child_rank,
                    "augmented_rank": augmented_rank,
                    "augmented_shared_rank_ratio": float(
                        augmented_rank / max(child_rank, 1)
                    ),
                    "spectrally_stable": stable,
                }
            )

    sharing_values = [
        float(record["orientations"][orientation]["sharing_inflation"])
        for record in nodes
        for orientation in ("query", "key")
    ]
    stable_nesting = [item for item in nesting if item["spectrally_stable"]]
    defect_values = [float(item["defect"]) for item in stable_nesting]
    ratio_values = [
        float(item["augmented_shared_rank_ratio"]) for item in stable_nesting
    ]
    summary = {
        "sharing_inflation": _summary(sharing_values),
        "stable_nesting_defect": _summary(defect_values),
        "stable_augmented_rank_ratio": _summary(ratio_values),
        "stable_cut_fraction": float(len(stable_nesting) / max(len(nesting), 1)),
        "stable_cuts": len(stable_nesting),
        "nesting_cuts": len(nesting),
        "max_shared_ranks": maxima,
        "concatenation_exact": concatenation_exact,
    }
    output: dict[str, Any] = {"summary": summary}
    if retain_records:
        output["nodes"] = nodes
        output["nesting"] = nesting
    return output


def _parent_cells(parent_run: Mapping[str, Any]) -> dict[tuple[int, int], Any]:
    return {
        (int(cell["layer"]), int(cell["head"])): cell
        for cell in parent_run["natural_cells"]
    }


def _expected_hss_maximum(
    parent_cell: Mapping[str, Any], length: int
) -> dict[str, int]:
    return {
        key: int(value)
        for key, value in parent_cell["lengths"][str(length)]["orders"]
        ["chronological"]["hss_boundary"]["max_ranks"].items()
    }


def _permutation(length: int, seed: int, salt: int) -> np.ndarray:
    return np.random.default_rng((seed, salt, length)).permutation(length)


def analyze_seed(
    config: SharedBasisNestingConfig,
    seed: int,
    model: Any,
    tokens: np.ndarray,
    parent_run: Mapping[str, Any],
    output: Path,
    implementation_sha256: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    device = next(model.parameters()).device
    segment, validation_start = _validation_segment(
        tokens, config.validation_tokens, 256, seed
    )
    if validation_start != int(parent_run["provenance"]["validation_start"]):
        raise RuntimeError("validation-prefix replay differs from A4")
    input_ids = torch.from_numpy(segment.copy()).unsqueeze(0).to(device)
    selected = extract_selected_qk(model, input_ids, config.layers, config.heads)
    parents = _parent_cells(parent_run)
    cells = []
    maximum_length = max(config.lengths)
    permutation = _permutation(
        maximum_length, seed, config.permutation_seed_salt
    )
    for layer in config.layers:
        for head in config.heads:
            query, key = selected[(layer, head)]
            parent_cell = parents[(layer, head)]
            lengths: dict[str, Any] = {}
            parent_reproduced = True
            concatenation_exact = True
            for length in config.lengths:
                attention = causal_attention(query[:length], key[:length])
                diagnostic = diagnose_operator(attention, config)
                expected = _expected_hss_maximum(parent_cell, length)
                reproduced = diagnostic["summary"]["max_shared_ranks"] == expected
                parent_reproduced = parent_reproduced and reproduced
                concatenation_exact = (
                    concatenation_exact
                    and bool(diagnostic["summary"]["concatenation_exact"])
                )
                lengths[str(length)] = {
                    "parent_hss_max_ranks": expected,
                    "parent_hss_reproduced": reproduced,
                    **diagnostic,
                }
            attention_maximum = causal_attention(
                query[:maximum_length], key[:maximum_length]
            )
            permuted = attention_maximum[np.ix_(permutation, permutation)]
            random_control = diagnose_operator(
                permuted, config, retain_records=False
            )["summary"]
            cells.append(
                {
                    "layer": layer,
                    "head": head,
                    "validity": {
                        "parent_hss_reproduced": parent_reproduced,
                        "complement_concatenations_exact": concatenation_exact,
                        "pass": parent_reproduced and concatenation_exact,
                    },
                    "lengths": lengths,
                    f"random_permutation_control_{maximum_length}": random_control,
                }
            )

    head_width = model.config.n_embd // model.config.n_head
    controls: dict[str, Any] = {}
    for condition in ("causal_uniform", "smooth_fourier", "iid_qk"):
        controls[condition] = {}
        for length in config.lengths:
            query, key = control_qk(condition, length, head_width, seed)
            controls[condition][str(length)] = diagnose_operator(
                causal_attention(query, key), config, retain_records=False
            )["summary"]
    left = np.linspace(1.0, 2.0, maximum_length, dtype=np.float64)
    right = np.linspace(2.0, 3.0, maximum_length, dtype=np.float64)
    controls["exact_rank_one"] = diagnose_operator(
        np.outer(left, right), config, retain_records=False
    )["summary"]

    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "parent_hypothesis_id": PARENT_HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-hss-shared-basis-nesting-seed{seed}",
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.shakedown
            else "preregistered_primary"
        ),
        "completed_at": _utc_now(),
        "seed": seed,
        "configuration": asdict(config),
        "scientific_fingerprint": _fingerprint(
            config, seed, implementation_sha256
        ),
        "implementation_sha256": implementation_sha256,
        "provenance": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "token_cache_sha256": TOKEN_STREAM_SHA256,
            "parent_aggregate_sha256": PARENT_AGGREGATE_SHA256,
            "parent_experiment_id": parent_run["experiment_id"],
            "validation_start": validation_start,
            "validation_stop": validation_start + 256,
            "random_permutation": permutation.tolist(),
        },
        "cells": cells,
        "controls": controls,
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(output / "runs" / f"seed_{seed}" / "result.json", result)
    return result


def _cell_metric(
    cells: Sequence[Mapping[str, Any]],
    length: int,
    section: str,
    key: str,
) -> list[float]:
    values = []
    for cell in cells:
        value = cell["lengths"][str(length)]["summary"][section][key]
        if value is not None:
            values.append(float(value))
    return values


def aggregate(
    runs: Sequence[Mapping[str, Any]], config: SharedBasisNestingConfig
) -> dict[str, Any]:
    cells = [cell for run in runs for cell in run["cells"]]
    valid = [bool(cell["validity"]["pass"]) for cell in cells]
    primary_length = 256 if 256 in config.lengths else max(config.lengths)

    sharing_cell_medians = _cell_metric(
        cells, primary_length, "sharing_inflation", "median"
    )
    sharing_cell_p90s = _cell_metric(
        cells, primary_length, "sharing_inflation", "p90"
    )
    defect_cell_medians = _cell_metric(
        cells, primary_length, "stable_nesting_defect", "median"
    )
    defect_cell_p90s = _cell_metric(
        cells, primary_length, "stable_nesting_defect", "p90"
    )
    ratio_cell_medians = _cell_metric(
        cells, primary_length, "stable_augmented_rank_ratio", "median"
    )
    ratio_cell_p90s = _cell_metric(
        cells, primary_length, "stable_augmented_rank_ratio", "p90"
    )

    stable_cuts = sum(
        int(cell["lengths"][str(primary_length)]["summary"]["stable_cuts"])
        for cell in cells
    )
    nesting_cuts = sum(
        int(cell["lengths"][str(primary_length)]["summary"]["nesting_cuts"])
        for cell in cells
    )
    stable_fraction = stable_cuts / max(nesting_cuts, 1)
    stable_sufficient = stable_fraction >= config.stable_cut_fraction_minimum

    sharing_median = float(np.median(sharing_cell_medians))
    sharing_p90 = float(np.percentile(sharing_cell_p90s, 90))
    sharing_pass = (
        sharing_median <= config.sharing_median_maximum
        and sharing_p90 <= config.sharing_p90_maximum
    )
    defect_median = (
        float(np.median(defect_cell_medians)) if defect_cell_medians else None
    )
    defect_p90 = (
        float(np.percentile(defect_cell_p90s, 90)) if defect_cell_p90s else None
    )
    defect_pass = bool(
        stable_sufficient
        and defect_median is not None
        and defect_p90 is not None
        and defect_median <= config.nesting_defect_median_maximum
        and defect_p90 <= config.nesting_defect_p90_maximum
    )
    ratio_median = (
        float(np.median(ratio_cell_medians)) if ratio_cell_medians else None
    )
    ratio_p90 = (
        float(np.percentile(ratio_cell_p90s, 90)) if ratio_cell_p90s else None
    )
    ratio_pass = bool(
        stable_sufficient
        and ratio_median is not None
        and ratio_p90 is not None
        and ratio_median <= config.augmented_ratio_median_maximum
        and ratio_p90 <= config.augmented_ratio_p90_maximum
    )
    rank_one = [run["controls"]["exact_rank_one"] for run in runs]
    rank_one_pass = all(
        record["max_shared_ranks"]["0.01"] == 1
        and record["stable_nesting_defect"]["maximum"] is not None
        and record["stable_nesting_defect"]["maximum"] <= 1e-10
        and record["stable_augmented_rank_ratio"]["maximum"] == 1.0
        for record in rank_one
    )
    validity_pass = bool(valid) and all(valid) and rank_one_pass
    nesting_pass = defect_pass and ratio_pass
    if not validity_pass:
        classification = "invalid_parent_or_boundary_contract"
    elif not stable_sufficient:
        classification = "shared_basis_result_nesting_indeterminate"
    elif sharing_pass and nesting_pass:
        classification = "shared_and_nested_hierarchy_supported"
    elif not sharing_pass and nesting_pass:
        classification = "shared_basis_bottleneck"
    elif sharing_pass and not nesting_pass:
        classification = "nesting_bottleneck"
    else:
        classification = "combined_sharing_and_nesting_bottleneck"

    scaling: dict[str, Any] = {}
    for length in config.lengths:
        records = [cell["lengths"][str(length)]["summary"] for cell in cells]
        scaling[str(length)] = {
            "shared_rank_median": float(
                np.median(
                    [record["max_shared_ranks"]["0.01"] for record in records]
                )
            ),
            "sharing_inflation_cell_median": float(
                np.median(
                    [record["sharing_inflation"]["median"] for record in records]
                )
            ),
        }

    return {
        "classification": classification,
        "validity": {
            "passing_cells": sum(valid),
            "required_cells": len(cells),
            "exact_rank_one_control_pass": rank_one_pass,
            "pass": validity_pass,
        },
        "shared_basis_compactness": {
            "median_of_cell_medians": sharing_median,
            "p90_of_cell_p90s": sharing_p90,
            "median_maximum": config.sharing_median_maximum,
            "p90_maximum": config.sharing_p90_maximum,
            "pass": sharing_pass,
        },
        "nesting_stability": {
            "stable_cuts": stable_cuts,
            "total_cuts": nesting_cuts,
            "stable_fraction": stable_fraction,
            "minimum_fraction": config.stable_cut_fraction_minimum,
            "sufficient": stable_sufficient,
        },
        "nesting_fidelity": {
            "median_of_cell_medians": defect_median,
            "p90_of_cell_p90s": defect_p90,
            "median_maximum": config.nesting_defect_median_maximum,
            "p90_maximum": config.nesting_defect_p90_maximum,
            "pass": defect_pass,
        },
        "nested_rank_compactness": {
            "median_of_cell_medians": ratio_median,
            "p90_of_cell_p90s": ratio_p90,
            "median_maximum": config.augmented_ratio_median_maximum,
            "p90_maximum": config.augmented_ratio_p90_maximum,
            "pass": ratio_pass,
        },
        "scaling": scaling,
        "counts": {
            "evaluation_seeds": len(runs),
            "cells": len(cells),
            "operator_lengths": len(cells) * len(config.lengths),
        },
    }


def _load_parent(config: SharedBasisNestingConfig) -> dict[int, Any]:
    path = Path(config.parent_aggregate)
    if _sha256(path) != PARENT_AGGREGATE_SHA256:
        raise RuntimeError("A4 aggregate hash differs from the preregistration")
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("implementation_sha256") != PARENT_IMPLEMENTATION_SHA256:
        raise RuntimeError("unexpected A4 implementation hash")
    runs = {}
    for item in campaign["results"]:
        run = json.loads(Path(item["result"]).read_text(encoding="utf-8"))
        runs[int(run["seed"])] = run
    return runs


def run_campaign(config: SharedBasisNestingConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(config.checkpoint)
    token_cache = Path(config.token_cache)
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash differs from the preregistration")
    if _sha256(token_cache) != TOKEN_STREAM_SHA256:
        raise RuntimeError("token stream hash differs from the preregistration")
    parent_runs = _load_parent(config)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(True)
    model = _load_model(checkpoint, device)
    tokens = np.load(token_cache, mmap_mode="r")
    implementation_sha256 = _implementation_digest()

    runs = []
    reused = 0
    for seed in config.evaluation_seeds:
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        expected = _fingerprint(config, seed, implementation_sha256)
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "completed"
                and existing.get("scientific_fingerprint") == expected
                and existing.get("implementation_sha256") == implementation_sha256
            ):
                runs.append(existing)
                reused += 1
                print(f"resuming {existing['experiment_id']}", flush=True)
                continue
        result = analyze_seed(
            config,
            seed,
            model,
            tokens,
            parent_runs[seed],
            output,
            implementation_sha256,
        )
        runs.append(result)
        print(
            result["experiment_id"], f"{result['analysis_seconds']:.1f}s", flush=True
        )

    aggregates = aggregate(runs, config)
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "parent_hypothesis_id": PARENT_HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.shakedown
            else "preregistered_primary"
        ),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation_sha256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "provenance": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "token_cache_sha256": TOKEN_STREAM_SHA256,
            "parent_aggregate_sha256": PARENT_AGGREGATE_SHA256,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        },
        "summary": {
            "requested": len(config.evaluation_seeds),
            "completed": len(runs),
            "failed": 0,
            "reused": reused,
            "trained_models": 0,
        },
        "aggregates": aggregates,
        "results": [
            {
                "experiment_id": run["experiment_id"],
                "seed": run["seed"],
                "analysis_seconds": run["analysis_seconds"],
                "result": str(
                    output / "runs" / f"seed_{run['seed']}" / "result.json"
                ),
            }
            for run in runs
        ],
        "method_boundaries": [
            "A6 diagnoses necessary shared/nested subspace structure and does not construct A7's simultaneous H2 operator.",
            "Stable-cut gates require both child and non-root parent SVD cutoffs to meet the fixed gap.",
            "SVD bases are diagnostic and do not imply an implicit subquadratic core generator.",
            "One checkpoint does not establish cross-model repeatability.",
        ],
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=SharedBasisNestingConfig.checkpoint)
    parser.add_argument("--token-cache", default=SharedBasisNestingConfig.token_cache)
    parser.add_argument(
        "--parent-aggregate", default=SharedBasisNestingConfig.parent_aggregate
    )
    parser.add_argument("--evaluation-seeds", default="101,211,307,401,503")
    parser.add_argument("--lengths", default="32,64,128,256")
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--heads", default="0,3,7")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_hss_shared_basis_nesting/"
            "20260830_registered"
        ),
    )
    args = parser.parse_args(argv)
    seeds = _ints(args.evaluation_seeds)
    lengths = _ints(args.lengths)
    layers = _ints(args.layers)
    heads = _ints(args.heads)
    if args.shakedown:
        seeds = seeds[:1]
        lengths = tuple(length for length in lengths if length <= 64)
        layers = layers[:1]
        heads = heads[:1]
    config = SharedBasisNestingConfig(
        checkpoint=args.checkpoint,
        token_cache=args.token_cache,
        parent_aggregate=args.parent_aggregate,
        evaluation_seeds=seeds,
        lengths=lengths,
        layers=layers,
        heads=heads,
        device=args.device,
        shakedown=args.shakedown,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    print(args.output / "campaign_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
