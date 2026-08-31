#!/usr/bin/env python3
"""Execute preregistered A7 constructive causal H2 attention."""

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

try:
    from experiments.structure_net.tinyllm_dynamic_ttno_rank import (
        _load_model,
        _sha256,
        _validation_segment,
        causal_attention,
    )
    from experiments.structure_net import tinyllm_dynamic_ttno_rank as a4
except ModuleNotFoundError:  # Direct script execution.
    from tinyllm_dynamic_ttno_rank import (  # type: ignore[no-redef]
        _load_model,
        _sha256,
        _validation_segment,
        causal_attention,
    )
    import tinyllm_dynamic_ttno_rank as a4  # type: ignore[no-redef]


SCHEMA_VERSION = "nal.tinyllm-causal-h2-attention.v1"
HYPOTHESIS_ID = "tinyllm-causal-h2-attention-v1"
PARENT_HYPOTHESIS_ID = "tinyllm-dynamic-ttno-rank-pilot-v1"
PREDECESSOR_HYPOTHESIS_ID = "tinyllm-hss-shared-basis-nesting-v1"
PARENT_AGGREGATE_SHA256 = (
    "9d3fa8ec7332860785b3d62dff5805e4ec23c9f6fedc5178c6809b15cd05feba"
)
A6_AGGREGATE_SHA256 = (
    "6f42a59b3a723eb4b80742e8fab8278be9b21d1db35131cc4ea81bd702e03c01"
)
CHECKPOINT_SHA256 = (
    "5a7491b4231c30feaaabda7babf0a831dd4d72fbb4e7f3e757114cadf098df09"
)
TOKEN_STREAM_SHA256 = (
    "f339655453b970ae6cc1cbdc7c78f8a5234c42437bc7bba8fb31a1dee5c9d765"
)
PREREGISTRATION_SHA256 = (
    "f222389b56280ac09cd31168967b4c60e03e9179128cee83e5d36b4d72d90954"
)
PARTITION_FINGERPRINTS = {
    32: "8e753f6334cd7d928b78b01fa4acc4310181ca053a81d42b72cd71c67b49d7e5",
    64: "5a5a87fb6313c2dda8d7bdb1ce818ed8f81b931558cb9328f4b3aa3c83b77a76",
    128: "97ac50d907bdfe6518af2e6d313bb62c6aebd284e2e8c7819955a8a763939600",
    256: "1535f675b18b8493d5da1c4e1048eefb73c884e700953451de7f9b8b380688f2",
    512: "79d9f9d12ff42883224ae0d0629933718503b800567203e04712f8765c3924e3",
}
PARTITION_COUNTS = {
    32: {"ADMISSIBLE": 0, "DENSE": 3, "ZERO": 1},
    64: {"ADMISSIBLE": 3, "DENSE": 7, "ZERO": 3},
    128: {"ADMISSIBLE": 12, "DENSE": 15, "ZERO": 7},
    256: {"ADMISSIBLE": 33, "DENSE": 31, "ZERO": 15},
    512: {"ADMISSIBLE": 78, "DENSE": 63, "ZERO": 31},
}


@dataclass(frozen=True)
class CausalH2Config:
    checkpoint: str = (
        "data/experiments/tinyllm_babylm_pretrain/"
        "20260812_d8_seed7/checkpoint_step12000.pt"
    )
    token_cache: str = "data/corpora/babylm_10M_bpe16k.tokens.npy"
    parent_aggregate: str = (
        "data/experiments/tinyllm_dynamic_ttno_rank/"
        "20260829_d8_babylm_pilot/campaign_results.json"
    )
    predecessor_aggregate: str = (
        "data/experiments/tinyllm_hss_shared_basis_nesting/"
        "20260830_registered/campaign_results.json"
    )
    validation_tokens: int = 262_144
    evaluation_seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    lengths: tuple[int, ...] = (32, 64, 128, 256)
    primary_lengths: tuple[int, ...] = (64, 128, 256)
    layers: tuple[int, ...] = tuple(range(8))
    heads: tuple[int, ...] = (0, 3, 7)
    leaf_size: int = 16
    separation_ratio: float = 1.0
    build_tolerance: float = 0.0025
    rank_multiplier: float = 1.0
    probe_seed: int = 1707
    probe_columns: int = 32
    kernel_row_relative_maximum: float = 0.01
    denominator_relative_maximum: float = 0.01
    attention_row_l1_maximum: float = 0.025
    probe_relative_frobenius_maximum: float = 0.02
    value_relative_frobenius_maximum: float = 0.02
    token_tail_maximum: float = 0.05
    cell_pass_fraction_minimum: float = 0.80
    layer_pass_fraction_minimum: float = 0.50
    storage_ratio_median_maximum: float = 0.75
    storage_ratio_p90_maximum: float = 1.0
    operation_ratio_median_maximum: float = 0.75
    operation_ratio_p90_maximum: float = 1.0
    device: str = "cuda:0"
    shakedown: bool = False

    def __post_init__(self) -> None:
        if not self.evaluation_seeds or not self.layers or not self.heads:
            raise ValueError("evaluation seeds, layers, and heads must be non-empty")
        if tuple(sorted(set(self.lengths))) != self.lengths:
            raise ValueError("lengths must be unique and increasing")
        if any(length < self.leaf_size or length & (length - 1) for length in self.lengths):
            raise ValueError("lengths must be powers of two at least leaf_size")
        if any(length not in self.lengths for length in self.primary_lengths):
            raise ValueError("primary lengths must be included in lengths")
        if self.leaf_size <= 0 or self.leaf_size & (self.leaf_size - 1):
            raise ValueError("leaf size must be a power of two")
        if self.separation_ratio <= 0.0 or self.rank_multiplier <= 0.0:
            raise ValueError("separation ratio and rank multiplier must be positive")


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


@dataclass(frozen=True)
class Block:
    kind: str
    query: str
    key: str


@dataclass
class H2Representation:
    operator_name: str
    length: int
    clusters: dict[str, Cluster]
    children: dict[str, tuple[str, str]]
    root: str
    blocks: list[Block]
    query_bases: dict[str, torch.Tensor]
    key_bases: dict[str, torch.Tensor]
    query_transfers: dict[tuple[str, str], torch.Tensor]
    key_transfers: dict[tuple[str, str], torch.Tensor]
    couplings: dict[tuple[str, str], torch.Tensor]
    dense_operator: torch.Tensor
    construction_records: dict[str, Any]


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
    config: CausalH2Config,
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
        "predecessor_aggregate_sha256": A6_AGGREGATE_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def cluster_tree(
    length: int, leaf_size: int
) -> tuple[list[Cluster], dict[str, tuple[str, str]], str]:
    clusters: list[Cluster] = []
    children: dict[str, tuple[str, str]] = {}

    def visit(start: int, stop: int, parent: str | None, depth: int) -> str:
        node = Cluster(start, stop, parent, depth)
        clusters.append(node)
        if node.size > leaf_size:
            middle = (start + stop) // 2
            left = visit(start, middle, node.id, depth + 1)
            right = visit(middle, stop, node.id, depth + 1)
            children[node.id] = (left, right)
        return node.id

    root = visit(0, length, None, 0)
    return clusters, children, root


def partition_blocks(
    length: int,
    leaf_size: int = 16,
    separation_ratio: float = 1.0,
) -> tuple[list[Block], dict[str, Cluster], dict[str, tuple[str, str]], str]:
    """Dual-tree strong-admissibility partition in canonical traversal order."""

    cluster_list, children, root = cluster_tree(length, leaf_size)
    clusters = {cluster.id: cluster for cluster in cluster_list}
    blocks: list[Block] = []

    def visit(query_id: str, key_id: str) -> None:
        query = clusters[query_id]
        key = clusters[key_id]
        if key.start >= query.stop:
            blocks.append(Block("ZERO", query_id, key_id))
            return
        if key.stop <= query.start:
            gap = query.start - key.stop
            required = separation_ratio * max(query.size, key.size)
            if gap >= required:
                blocks.append(Block("ADMISSIBLE", query_id, key_id))
                return
        query_children = children.get(query_id, (query_id,))
        key_children = children.get(key_id, (key_id,))
        if query_children == (query_id,) and key_children == (key_id,):
            blocks.append(Block("DENSE", query_id, key_id))
            return
        for query_child in query_children:
            for key_child in key_children:
                visit(query_child, key_child)

    visit(root, root)
    return blocks, clusters, children, root


def canonical_partition(blocks: Sequence[Block], clusters: Mapping[str, Cluster]) -> str:
    lines = []
    for block in blocks:
        query = clusters[block.query]
        key = clusters[block.key]
        lines.append(
            f"{block.kind}:{query.start}-{query.stop - 1}:"
            f"{key.start}-{key.stop - 1}"
        )
    return "\n".join(lines)


def partition_integrity(
    length: int,
    leaf_size: int = 16,
    separation_ratio: float = 1.0,
) -> dict[str, Any]:
    blocks, clusters, _, _ = partition_blocks(
        length, leaf_size, separation_ratio
    )
    counts = {
        kind: sum(block.kind == kind for block in blocks)
        for kind in ("ADMISSIBLE", "DENSE", "ZERO")
    }
    packed = canonical_partition(blocks, clusters).encode("utf-8")
    return {"counts": counts, "sha256": hashlib.sha256(packed).hexdigest()}


def rank_cap(length: int, multiplier: float = 1.0) -> int:
    return max(1, int(math.ceil(multiplier * math.log2(length) ** 2)))


def _is_descendant(node: Cluster, ancestor: Cluster) -> bool:
    return ancestor.start <= node.start and node.stop <= ancestor.stop


def _rank_from_values(values: np.ndarray, epsilon: float) -> int:
    energy = np.square(np.asarray(values, dtype=np.float64))
    total = float(energy.sum())
    if total == 0.0:
        return 0
    tail = np.concatenate((np.cumsum(energy[::-1])[::-1], np.zeros(1)))
    candidates = np.flatnonzero(tail <= epsilon**2 * total)
    return int(candidates[0]) if candidates.size else len(values)


def _select_nested_rank(
    singular_values: np.ndarray,
    total_energy: float,
    maximum_rank: int,
    squared_error_budget: float,
) -> tuple[int, bool, float]:
    if total_energy <= 0.0 or maximum_rank <= 0:
        return 0, False, 0.0
    values = np.square(np.asarray(singular_values, dtype=np.float64))
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    maximum_rank = min(maximum_rank, len(values))
    for rank in range(maximum_rank + 1):
        residual = max(total_energy - float(cumulative[rank]), 0.0)
        relative = residual / total_energy
        if relative <= squared_error_budget:
            return rank, False, relative
    residual = max(total_energy - float(cumulative[maximum_rank]), 0.0)
    return maximum_rank, True, residual / total_energy


def _empty_basis(rows: int, device: torch.device) -> torch.Tensor:
    return torch.empty((rows, 0), dtype=torch.float64, device=device)


def _block_diagonal(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    rows = left.shape[0] + right.shape[0]
    columns = left.shape[1] + right.shape[1]
    output = torch.zeros(
        (rows, columns), dtype=left.dtype, device=left.device
    )
    output[: left.shape[0], : left.shape[1]] = left
    output[left.shape[0] :, left.shape[1] :] = right
    return output


def _sample_matrix(
    normalized: torch.Tensor,
    node: Cluster,
    orientation: str,
    admissible: Sequence[Block],
    clusters: Mapping[str, Cluster],
) -> torch.Tensor:
    samples: list[torch.Tensor] = []
    for block in admissible:
        query = clusters[block.query]
        key = clusters[block.key]
        if orientation == "query" and _is_descendant(node, query):
            samples.append(
                normalized[node.start : node.stop, key.start : key.stop]
            )
        elif orientation == "key" and _is_descendant(node, key):
            samples.append(
                normalized[query.start : query.stop, node.start : node.stop].T
            )
    if not samples:
        return torch.empty(
            (node.size, 0), dtype=normalized.dtype, device=normalized.device
        )
    return torch.cat(samples, dim=1)


def _nested_bases(
    normalized: torch.Tensor,
    clusters: Mapping[str, Cluster],
    children: Mapping[str, tuple[str, str]],
    admissible: Sequence[Block],
    config: CausalH2Config,
    orientation: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    bases: dict[str, torch.Tensor] = {}
    records: dict[str, Any] = {}
    levels = max(1, int(math.ceil(math.log2(len(normalized) / config.leaf_size))))
    budget = config.build_tolerance**2 / (2 * levels)
    cap = rank_cap(len(normalized), config.rank_multiplier)
    ordered = sorted(clusters.values(), key=lambda node: (-node.depth, node.start))
    for node in ordered:
        sample = _sample_matrix(
            normalized, node, orientation, admissible, clusters
        )
        total_energy = float(torch.sum(sample * sample).item())
        if node.id in children:
            left_id, right_id = children[node.id]
            candidate = _block_diagonal(bases[left_id], bases[right_id])
            maximum = min(node.size, cap, candidate.shape[1])
            if candidate.shape[1] == 0 or sample.shape[1] == 0:
                basis = _empty_basis(node.size, normalized.device)
                selected_rank = 0
                cap_hit = total_energy > 0.0
                residual = 1.0 if total_energy > 0.0 else 0.0
            else:
                coordinates = candidate.T @ sample
                left, singular, _ = torch.linalg.svd(
                    coordinates, full_matrices=False
                )
                singular_values = singular.detach().cpu().numpy()
                selected_rank, cap_hit, residual = _select_nested_rank(
                    singular_values, total_energy, maximum, budget
                )
                basis = candidate @ left[:, :selected_rank]
        else:
            maximum = min(node.size, cap)
            if sample.shape[1] == 0:
                basis = _empty_basis(node.size, normalized.device)
                selected_rank = 0
                cap_hit = False
                residual = 0.0
            else:
                left, singular, _ = torch.linalg.svd(sample, full_matrices=False)
                singular_values = singular.detach().cpu().numpy()
                selected_rank, cap_hit, residual = _select_nested_rank(
                    singular_values, total_energy, maximum, budget
                )
                basis = left[:, :selected_rank]
        bases[node.id] = basis
        records[node.id] = {
            "start": node.start,
            "stop": node.stop,
            "depth": node.depth,
            "sample_columns": sample.shape[1],
            "rank": selected_rank,
            "maximum_permissible_rank": maximum,
            "rank_cap_hit": cap_hit,
            "relative_local_squared_residual": residual,
            "squared_error_budget": budget,
        }
    return bases, records


def _row_rescaled_orthonormal_bases(
    normalized_bases: Mapping[str, torch.Tensor],
    row_mass: torch.Tensor,
    clusters: Mapping[str, Cluster],
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for node_id, basis in normalized_bases.items():
        node = clusters[node_id]
        if basis.shape[1] == 0:
            output[node_id] = basis
            continue
        raw = row_mass[node.start : node.stop, None] * basis
        orthonormal, _ = torch.linalg.qr(raw, mode="reduced")
        output[node_id] = orthonormal[:, : basis.shape[1]]
    return output


def _transfers(
    bases: Mapping[str, torch.Tensor],
    clusters: Mapping[str, Cluster],
    children: Mapping[str, tuple[str, str]],
) -> tuple[dict[tuple[str, str], torch.Tensor], float]:
    transfers: dict[tuple[str, str], torch.Tensor] = {}
    maximum_residual = 0.0
    for parent_id, child_ids in children.items():
        parent_basis = bases[parent_id]
        parent = clusters[parent_id]
        for child_id in child_ids:
            child = clusters[child_id]
            child_basis = bases[child_id]
            restricted = parent_basis[
                child.start - parent.start : child.stop - parent.start
            ]
            transfer = child_basis.T @ restricted
            reconstructed = child_basis @ transfer
            denominator = max(float(torch.linalg.norm(restricted).item()), 1e-12)
            residual = float(
                (torch.linalg.norm(restricted - reconstructed).item() / denominator)
            )
            maximum_residual = max(maximum_residual, residual)
            transfers[(child_id, parent_id)] = transfer
    return transfers, maximum_residual


def _orthogonality_maximum(bases: Mapping[str, torch.Tensor]) -> float:
    maximum = 0.0
    for basis in bases.values():
        if basis.shape[1] == 0:
            continue
        identity = torch.eye(
            basis.shape[1], dtype=basis.dtype, device=basis.device
        )
        residual = torch.linalg.matrix_norm(
            basis.T @ basis - identity, ord=2
        )
        maximum = max(maximum, float(residual.item()))
    return maximum


def build_h2_representations(
    kernel: np.ndarray,
    attention: np.ndarray,
    config: CausalH2Config,
) -> tuple[H2Representation, H2Representation, dict[str, Any]]:
    """Build primary H2(K) and diagnostic H2(A) under identical rank choices."""

    device = torch.device(config.device)
    kernel_tensor = torch.as_tensor(kernel, dtype=torch.float64, device=device)
    attention_tensor = torch.as_tensor(attention, dtype=torch.float64, device=device)
    length = len(kernel)
    blocks, clusters, children, root = partition_blocks(
        length, config.leaf_size, config.separation_ratio
    )
    admissible = [block for block in blocks if block.kind == "ADMISSIBLE"]
    query_a, query_records = _nested_bases(
        attention_tensor,
        clusters,
        children,
        admissible,
        config,
        "query",
    )
    key, key_records = _nested_bases(
        attention_tensor,
        clusters,
        children,
        admissible,
        config,
        "key",
    )
    row_mass = torch.sum(kernel_tensor, dim=1)
    query_k = _row_rescaled_orthonormal_bases(
        query_a, row_mass, clusters
    )

    query_a_transfers, query_a_nestedness = _transfers(
        query_a, clusters, children
    )
    query_k_transfers, query_k_nestedness = _transfers(
        query_k, clusters, children
    )
    key_transfers, key_nestedness = _transfers(key, clusters, children)

    couplings_k: dict[tuple[str, str], torch.Tensor] = {}
    couplings_a: dict[tuple[str, str], torch.Tensor] = {}
    block_records = []
    for block in admissible:
        query_node = clusters[block.query]
        key_node = clusters[block.key]
        kernel_block = kernel_tensor[
            query_node.start : query_node.stop,
            key_node.start : key_node.stop,
        ]
        attention_block = attention_tensor[
            query_node.start : query_node.stop,
            key_node.start : key_node.stop,
        ]
        coupling_k = query_k[block.query].T @ kernel_block @ key[block.key]
        coupling_a = query_a[block.query].T @ attention_block @ key[block.key]
        couplings_k[(block.query, block.key)] = coupling_k
        couplings_a[(block.query, block.key)] = coupling_a
        block_records.append(
            {
                "query": block.query,
                "key": block.key,
                "query_rank": query_k[block.query].shape[1],
                "key_rank": key[block.key].shape[1],
                "coupling_scalars": coupling_k.numel(),
            }
        )

    primary = H2Representation(
        operator_name="kernel",
        length=length,
        clusters=clusters,
        children=children,
        root=root,
        blocks=blocks,
        query_bases=query_k,
        key_bases=key,
        query_transfers=query_k_transfers,
        key_transfers=key_transfers,
        couplings=couplings_k,
        dense_operator=kernel_tensor,
        construction_records={},
    )
    oracle = H2Representation(
        operator_name="attention_oracle",
        length=length,
        clusters=clusters,
        children=children,
        root=root,
        blocks=blocks,
        query_bases=query_a,
        key_bases=key,
        query_transfers=query_a_transfers,
        key_transfers=key_transfers,
        couplings=couplings_a,
        dense_operator=attention_tensor,
        construction_records={},
    )
    integrity = partition_integrity(
        length, config.leaf_size, config.separation_ratio
    )
    records = {
        "partition": integrity,
        "rank_cap": rank_cap(length, config.rank_multiplier),
        "query_nodes": query_records,
        "key_nodes": key_records,
        "admissible_blocks": block_records,
        "rank_cap_hits": {
            "query": sum(item["rank_cap_hit"] for item in query_records.values()),
            "key": sum(item["rank_cap_hit"] for item in key_records.values()),
        },
        "orthogonality": {
            "kernel_query_maximum": _orthogonality_maximum(query_k),
            "oracle_query_maximum": _orthogonality_maximum(query_a),
            "key_maximum": _orthogonality_maximum(key),
        },
        "nestedness": {
            "kernel_query_maximum": query_k_nestedness,
            "oracle_query_maximum": query_a_nestedness,
            "key_maximum": key_nestedness,
        },
    }
    primary.construction_records = records
    oracle.construction_records = records
    return primary, oracle, records


def assemble_dense(representation: H2Representation) -> torch.Tensor:
    output = torch.zeros_like(representation.dense_operator)
    for block in representation.blocks:
        query = representation.clusters[block.query]
        key = representation.clusters[block.key]
        query_slice = slice(query.start, query.stop)
        key_slice = slice(key.start, key.stop)
        if block.kind == "DENSE":
            output[query_slice, key_slice] = representation.dense_operator[
                query_slice, key_slice
            ]
        elif block.kind == "ADMISSIBLE":
            output[query_slice, key_slice] = (
                representation.query_bases[block.query]
                @ representation.couplings[(block.query, block.key)]
                @ representation.key_bases[block.key].T
            )
    return output


def h2_matvec(
    representation: H2Representation, values: torch.Tensor
) -> torch.Tensor:
    """Prescribed upward, interaction, and downward H2 application."""

    if values.ndim == 1:
        values = values[:, None]
    channels = values.shape[1]
    clusters = representation.clusters
    leaves = [node for node in clusters.values() if node.id not in representation.children]
    coefficients: dict[str, torch.Tensor] = {}
    for leaf in leaves:
        coefficients[leaf.id] = (
            representation.key_bases[leaf.id].T
            @ values[leaf.start : leaf.stop]
        )
    internal = sorted(
        (clusters[node_id] for node_id in representation.children),
        key=lambda node: -node.depth,
    )
    for node in internal:
        rank = representation.key_bases[node.id].shape[1]
        coefficient = torch.zeros(
            (rank, channels), dtype=values.dtype, device=values.device
        )
        for child_id in representation.children[node.id]:
            transfer = representation.key_transfers[(child_id, node.id)]
            coefficient += transfer.T @ coefficients[child_id]
        coefficients[node.id] = coefficient

    local = {
        node_id: torch.zeros(
            (basis.shape[1], channels),
            dtype=values.dtype,
            device=values.device,
        )
        for node_id, basis in representation.query_bases.items()
    }
    for block in representation.blocks:
        if block.kind == "ADMISSIBLE":
            local[block.query] += (
                representation.couplings[(block.query, block.key)]
                @ coefficients[block.key]
            )

    preorder = sorted(clusters.values(), key=lambda node: (node.depth, node.start))
    for node in preorder:
        if node.id not in representation.children:
            continue
        for child_id in representation.children[node.id]:
            transfer = representation.query_transfers[(child_id, node.id)]
            local[child_id] += transfer @ local[node.id]

    output = torch.zeros(
        (representation.length, channels),
        dtype=values.dtype,
        device=values.device,
    )
    for leaf in leaves:
        output[leaf.start : leaf.stop] += (
            representation.query_bases[leaf.id] @ local[leaf.id]
        )
    for block in representation.blocks:
        if block.kind != "DENSE":
            continue
        query = clusters[block.query]
        key = clusters[block.key]
        output[query.start : query.stop] += (
            representation.dense_operator[
                query.start : query.stop, key.start : key.stop
            ]
            @ values[key.start : key.stop]
        )
    return output


def _relative_frobenius(error: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.norm(error)
    denominator = torch.clamp(torch.linalg.norm(reference), min=1e-12)
    return float((numerator / denominator).item())


def _fixed_probes(length: int, columns: int, seed: int) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(seed))
    return (
        2 * generator.integers(0, 2, size=(length, columns), dtype=np.int8) - 1
    ).astype(np.float64)


def _storage_and_operations(
    representation: H2Representation, channels: int
) -> dict[str, Any]:
    clusters = representation.clusters
    leaves = [node for node in clusters.values() if node.id not in representation.children]
    near_storage = 0
    near_nonzero = 0
    coupling = 0
    for block in representation.blocks:
        query = clusters[block.query]
        key = clusters[block.key]
        if block.kind == "DENSE":
            near_storage += query.size * key.size
            near_nonzero += int(
                torch.count_nonzero(
                    representation.dense_operator[
                        query.start : query.stop, key.start : key.stop
                    ]
                ).item()
            )
        elif block.kind == "ADMISSIBLE":
            coupling += representation.couplings[(block.query, block.key)].numel()
    leaf_storage = sum(
        leaf.size
        * (
            representation.query_bases[leaf.id].shape[1]
            + representation.key_bases[leaf.id].shape[1]
        )
        for leaf in leaves
    )
    transfer_storage = sum(
        value.numel() for value in representation.query_transfers.values()
    ) + sum(value.numel() for value in representation.key_transfers.values())
    total_storage = near_storage + leaf_storage + transfer_storage + coupling
    dense_storage = representation.length * (representation.length + 1) // 2

    leaf_operations = channels * leaf_storage
    transfer_operations = channels * transfer_storage
    coupling_operations = channels * coupling
    near_operations = channels * near_nonzero
    total_operations = (
        leaf_operations
        + transfer_operations
        + coupling_operations
        + near_operations
    )
    dense_operations = channels * dense_storage
    return {
        "channels": channels,
        "storage": {
            "near": near_storage,
            "leaf_bases": leaf_storage,
            "transfers": transfer_storage,
            "couplings": coupling,
            "total": total_storage,
            "causal_dense": dense_storage,
            "ratio": float(total_storage / dense_storage),
        },
        "multiply_adds": {
            "near": near_operations,
            "leaf_passes": leaf_operations,
            "transfer_passes": transfer_operations,
            "interactions": coupling_operations,
            "total": total_operations,
            "causal_dense": dense_operations,
            "ratio": float(total_operations / dense_operations),
        },
    }


def stabilized_kernel(
    query: np.ndarray, key: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    logits = query @ key.T / math.sqrt(query.shape[1])
    logits[np.triu_indices(len(logits), 1)] = -np.inf
    maxima = np.max(logits, axis=1, keepdims=True)
    kernel = np.exp(logits - maxima)
    kernel[np.triu_indices(len(logits), 1)] = 0.0
    denominator = kernel.sum(axis=1)
    attention = kernel / denominator[:, None]
    return kernel, denominator, attention


def evaluate_cell_length(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    config: CausalH2Config,
) -> dict[str, Any]:
    kernel, denominator, attention = stabilized_kernel(query, key)
    softmax_attention = causal_attention(query, key)
    softmax_error = float(np.max(np.abs(attention - softmax_attention)))
    primary, oracle, construction = build_h2_representations(
        kernel, attention, config
    )
    kernel_tilde = assemble_dense(primary)
    attention_oracle_tilde = assemble_dense(oracle)
    device = kernel_tilde.device
    kernel_tensor = torch.as_tensor(kernel, dtype=torch.float64, device=device)
    attention_tensor = torch.as_tensor(attention, dtype=torch.float64, device=device)
    denominator_tensor = torch.as_tensor(
        denominator, dtype=torch.float64, device=device
    )
    value_tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
    probes = torch.as_tensor(
        _fixed_probes(len(kernel), config.probe_columns, config.probe_seed),
        dtype=torch.float64,
        device=device,
    )
    contraction_input = torch.cat(
        (
            value_tensor,
            torch.ones((len(kernel), 1), dtype=torch.float64, device=device),
            probes,
        ),
        dim=1,
    )
    explicit = kernel_tilde @ contraction_input
    contracted = h2_matvec(primary, contraction_input)
    contraction_error = _relative_frobenius(contracted - explicit, explicit)

    denominator_tilde = torch.sum(kernel_tilde, dim=1)
    positive_denominator = bool(torch.all(denominator_tilde > 0.0).item())
    kernel_row_error = torch.sum(torch.abs(kernel_tensor - kernel_tilde), dim=1)
    delta = float(torch.max(kernel_row_error / denominator_tensor).item())
    denominator_error = float(
        torch.max(
            torch.abs(denominator_tensor - denominator_tilde) / denominator_tensor
        ).item()
    )
    normalized_available = positive_denominator and bool(
        torch.all(torch.isfinite(denominator_tilde)).item()
    )
    if normalized_available:
        attention_tilde = kernel_tilde / denominator_tilde[:, None]
        attention_row = float(
            torch.max(torch.sum(torch.abs(attention_tensor - attention_tilde), dim=1)).item()
        )
        probe_error = _relative_frobenius(
            (attention_tilde - attention_tensor) @ probes,
            attention_tensor @ probes,
        )
        output = attention_tensor @ value_tensor
        output_error_matrix = (attention_tilde - attention_tensor) @ value_tensor
        value_error = _relative_frobenius(output_error_matrix, output)
        global_rms = torch.sqrt(torch.mean(torch.sum(output * output, dim=1)))
        token_errors = torch.linalg.norm(output_error_matrix, dim=1) / torch.clamp(
            global_rms, min=1e-12
        )
        token_tail = float(torch.quantile(token_errors, 0.99).item())
        negativity = float(
            torch.max(torch.sum(torch.clamp(-attention_tilde, min=0.0), dim=1)).item()
        )
    else:
        attention_tilde = torch.zeros_like(attention_tensor)
        attention_row = None
        probe_error = None
        value_error = None
        token_tail = None
        negativity = None

    oracle_row = float(
        torch.max(
            torch.sum(torch.abs(attention_tensor - attention_oracle_tilde), dim=1)
        ).item()
    )
    oracle_probe = _relative_frobenius(
        (attention_oracle_tilde - attention_tensor) @ probes,
        attention_tensor @ probes,
    )
    oracle_output = attention_tensor @ value_tensor
    oracle_error_matrix = (attention_oracle_tilde - attention_tensor) @ value_tensor
    oracle_value = _relative_frobenius(oracle_error_matrix, oracle_output)
    oracle_global_rms = torch.sqrt(
        torch.mean(torch.sum(oracle_output * oracle_output, dim=1))
    )
    oracle_tail = float(
        torch.quantile(
            torch.linalg.norm(oracle_error_matrix, dim=1)
            / torch.clamp(oracle_global_rms, min=1e-12),
            0.99,
        ).item()
    )

    future_leakage = float(
        torch.max(torch.abs(torch.triu(kernel_tilde, diagonal=1))).item()
    )
    integrity_expected = (
        PARTITION_COUNTS.get(len(kernel))
        if config.leaf_size == 16 and config.separation_ratio == 1.0
        else None
    )
    fingerprint_expected = (
        PARTITION_FINGERPRINTS.get(len(kernel))
        if config.leaf_size == 16 and config.separation_ratio == 1.0
        else None
    )
    partition_pass = bool(
        integrity_expected is None
        or (
            construction["partition"]["counts"] == integrity_expected
            and construction["partition"]["sha256"] == fingerprint_expected
        )
    )
    orthogonality_max = max(construction["orthogonality"].values())
    nestedness_max = max(construction["nestedness"].values())
    finite = all(
        math.isfinite(item)
        for item in (
            softmax_error,
            contraction_error,
            future_leakage,
            orthogonality_max,
            nestedness_max,
            delta,
            denominator_error,
            oracle_row,
            oracle_probe,
            oracle_value,
            oracle_tail,
        )
    )
    validity = {
        "softmax_max_absolute_error": softmax_error,
        "softmax_pass": softmax_error <= 1e-12,
        "partition_pass": partition_pass,
        "query_key_orthogonality_maximum": orthogonality_max,
        "orthogonality_pass": orthogonality_max <= 1e-10,
        "query_key_nestedness_maximum": nestedness_max,
        "nestedness_pass": nestedness_max <= 1e-10,
        "explicit_vs_contraction_relative_error": contraction_error,
        "contraction_pass": contraction_error <= 1e-10,
        "future_token_leakage": future_leakage,
        "future_leakage_pass": future_leakage < 1e-15,
        "finite": finite,
    }
    validity["pass"] = all(
        validity[key]
        for key in (
            "softmax_pass",
            "partition_pass",
            "orthogonality_pass",
            "nestedness_pass",
            "contraction_pass",
            "future_leakage_pass",
            "finite",
        )
    )
    gates = {
        "kernel_row_relative": delta <= config.kernel_row_relative_maximum,
        "denominator": (
            denominator_error <= config.denominator_relative_maximum
            and positive_denominator
        ),
        "attention_row_l1": bool(
            attention_row is not None
            and attention_row <= config.attention_row_l1_maximum
        ),
        "probe": bool(
            probe_error is not None
            and probe_error <= config.probe_relative_frobenius_maximum
        ),
        "value_output": bool(
            value_error is not None
            and value_error <= config.value_relative_frobenius_maximum
        ),
        "token_tail": bool(
            token_tail is not None and token_tail <= config.token_tail_maximum
        ),
    }
    cell_pass = bool(validity["pass"] and all(gates.values()))
    oracle_gates = {
        "row_relative": oracle_row <= config.kernel_row_relative_maximum,
        "attention_row_l1": oracle_row <= config.attention_row_l1_maximum,
        "probe": oracle_probe <= config.probe_relative_frobenius_maximum,
        "value_output": oracle_value <= config.value_relative_frobenius_maximum,
        "token_tail": oracle_tail <= config.token_tail_maximum,
    }
    compression = _storage_and_operations(primary, value.shape[1] + 1)
    return {
        "validity": validity,
        "primary_metrics": {
            "kernel_row_relative_maximum": delta,
            "denominator_relative_maximum": denominator_error,
            "positive_approximate_denominator": positive_denominator,
            "attention_row_l1_maximum": attention_row,
            "probe_relative_frobenius": probe_error,
            "value_output_relative_frobenius": value_error,
            "token_output_p99_global_rms_normalized": token_tail,
            "negative_attention_mass_maximum": negativity,
            "nearly_positivity_preserving": bool(
                negativity is not None and negativity <= 1e-3
            ),
        },
        "primary_gates": gates,
        "cell_pass": cell_pass,
        "oracle_metrics": {
            "attention_row_l1_maximum": oracle_row,
            "probe_relative_frobenius": oracle_probe,
            "value_output_relative_frobenius": oracle_value,
            "token_output_p99_global_rms_normalized": oracle_tail,
        },
        "oracle_gates": oracle_gates,
        "oracle_pass": bool(validity["pass"] and all(oracle_gates.values())),
        "compression": compression,
        "construction": construction,
    }


@torch.no_grad()
def extract_selected_qkv(
    model: Any,
    input_ids: torch.Tensor,
    layers: Sequence[int],
    heads: Sequence[int],
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    hidden = model.transformer["wte"](input_ids) + model.transformer["wpe"](
        positions
    )
    selected: dict[
        tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for layer_index, block in enumerate(model.transformer["h"]):
        normalized = block.ln_1(hidden)
        if layer_index in layers:
            query, key, value = block.attn.c_attn(normalized).split(
                block.attn.n_embd, dim=-1
            )
            head_width = model.config.n_embd // model.config.n_head
            query = query.view(1, input_ids.shape[1], model.config.n_head, head_width)
            key = key.view(1, input_ids.shape[1], model.config.n_head, head_width)
            value = value.view(1, input_ids.shape[1], model.config.n_head, head_width)
            for head in heads:
                selected[(layer_index, head)] = (
                    query[0, :, head].double().cpu().numpy(),
                    key[0, :, head].double().cpu().numpy(),
                    value[0, :, head].double().cpu().numpy(),
                )
        hidden = block(hidden)
    return selected


def analyze_seed(
    config: CausalH2Config,
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
    selected = extract_selected_qkv(model, input_ids, config.layers, config.heads)
    cells = []
    for layer in config.layers:
        for head in config.heads:
            query, key, value = selected[(layer, head)]
            lengths = {}
            for length in config.lengths:
                lengths[str(length)] = evaluate_cell_length(
                    query[:length], key[:length], value[:length], config
                )
            cells.append({"layer": layer, "head": head, "lengths": lengths})
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "parent_hypothesis_id": PARENT_HYPOTHESIS_ID,
        "predecessor_hypothesis_id": PREDECESSOR_HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-causal-h2-attention-seed{seed}",
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
            "predecessor_aggregate_sha256": A6_AGGREGATE_SHA256,
            "validation_start": validation_start,
            "validation_stop": validation_start + 256,
        },
        "cells": cells,
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(output / "runs" / f"seed_{seed}" / "result.json", result)
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def aggregate(
    runs: Sequence[Mapping[str, Any]], config: CausalH2Config
) -> dict[str, Any]:
    cells = [cell for run in runs for cell in run["cells"]]
    validity_pass = all(
        cell["lengths"][str(length)]["validity"]["pass"]
        for cell in cells
        for length in config.lengths
    )
    length_summary: dict[str, Any] = {}
    primary_campaign_pass = True
    oracle_campaign_pass = True
    for length in config.lengths:
        records = [cell["lengths"][str(length)] for cell in cells]
        passes = [bool(record["cell_pass"]) for record in records]
        oracle_passes = [bool(record["oracle_pass"]) for record in records]
        pass_fraction = float(np.mean(passes))
        oracle_fraction = float(np.mean(oracle_passes))
        if length in config.primary_lengths:
            primary_campaign_pass = (
                primary_campaign_pass
                and pass_fraction >= config.cell_pass_fraction_minimum
            )
            oracle_campaign_pass = (
                oracle_campaign_pass
                and oracle_fraction >= config.cell_pass_fraction_minimum
            )
        length_summary[str(length)] = {
            "passing_cells": sum(passes),
            "cell_count": len(records),
            "pass_fraction": pass_fraction,
            "required_fraction": (
                config.cell_pass_fraction_minimum
                if length in config.primary_lengths
                else None
            ),
            "oracle_pass_fraction": oracle_fraction,
            "kernel_row_relative_median": float(
                np.median(
                    [
                        record["primary_metrics"]["kernel_row_relative_maximum"]
                        for record in records
                    ]
                )
            ),
            "kernel_row_relative_p90": _percentile(
                [
                    record["primary_metrics"]["kernel_row_relative_maximum"]
                    for record in records
                ],
                90,
            ),
            "rank_cap_hit_cell_fraction": float(
                np.mean(
                    [
                        (
                            record["construction"]["rank_cap_hits"]["query"]
                            + record["construction"]["rank_cap_hits"]["key"]
                        )
                        > 0
                        for record in records
                    ]
                )
            ),
        }

    maximum_length = max(config.primary_lengths)
    layer_pass_fractions = {}
    for layer in config.layers:
        layer_records = [
            cell["lengths"][str(maximum_length)]
            for cell in cells
            if int(cell["layer"]) == layer
        ]
        layer_pass_fractions[str(layer)] = float(
            np.mean([record["cell_pass"] for record in layer_records])
        )
    minimum_layer_fraction = min(layer_pass_fractions.values())
    layer_gate = minimum_layer_fraction >= config.layer_pass_fraction_minimum

    longest_records = [
        cell["lengths"][str(maximum_length)] for cell in cells
    ]
    storage_ratios = [
        float(record["compression"]["storage"]["ratio"])
        for record in longest_records
    ]
    operation_ratios = [
        float(record["compression"]["multiply_adds"]["ratio"])
        for record in longest_records
    ]
    compression = {
        "length": maximum_length,
        "storage_ratio_median": float(np.median(storage_ratios)),
        "storage_ratio_p90": _percentile(storage_ratios, 90),
        "operation_ratio_median": float(np.median(operation_ratios)),
        "operation_ratio_p90": _percentile(operation_ratios, 90),
    }
    compression["pass"] = bool(
        compression["storage_ratio_median"]
        <= config.storage_ratio_median_maximum
        and compression["storage_ratio_p90"]
        <= config.storage_ratio_p90_maximum
        and compression["operation_ratio_median"]
        <= config.operation_ratio_median_maximum
        and compression["operation_ratio_p90"]
        <= config.operation_ratio_p90_maximum
    )

    if not validity_pass:
        classification = "invalid_h2_construction_contract"
    elif not primary_campaign_pass and oracle_campaign_pass:
        classification = "h2_normalization_path_failed"
    elif not primary_campaign_pass:
        classification = "h2_representation_failed"
    elif not layer_gate:
        classification = "h2_layer_selective_only"
    elif compression["pass"]:
        classification = "h2_constructive_compression_pass"
    else:
        classification = "h2_representation_pass_no_finite_size_compression"
    return {
        "classification": classification,
        "representation_pass": bool(
            validity_pass and primary_campaign_pass and layer_gate
        ),
        "validity": {
            "passing_cell_lengths": sum(
                cell["lengths"][str(length)]["validity"]["pass"]
                for cell in cells
                for length in config.lengths
            ),
            "required_cell_lengths": len(cells) * len(config.lengths),
            "pass": validity_pass,
        },
        "lengths": length_summary,
        "layer_gate_at_maximum_length": {
            "layer_pass_fractions": layer_pass_fractions,
            "minimum": minimum_layer_fraction,
            "required_minimum": config.layer_pass_fraction_minimum,
            "pass": layer_gate,
        },
        "oracle_campaign_pass": oracle_campaign_pass,
        "compression": compression,
        "counts": {
            "evaluation_seeds": len(runs),
            "cells": len(cells),
            "operator_lengths": len(cells) * len(config.lengths),
        },
    }


def _load_evidence(config: CausalH2Config) -> dict[int, Any]:
    parent_path = Path(config.parent_aggregate)
    predecessor_path = Path(config.predecessor_aggregate)
    if _sha256(parent_path) != PARENT_AGGREGATE_SHA256:
        raise RuntimeError("A4 aggregate hash differs from the preregistration")
    if _sha256(predecessor_path) != A6_AGGREGATE_SHA256:
        raise RuntimeError("A6 aggregate hash differs from the frozen predecessor")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor["aggregates"]["classification"] != (
        "shared_and_nested_hierarchy_supported"
    ):
        raise RuntimeError("unexpected A6 predecessor classification")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    runs = {}
    for item in parent["results"]:
        run = json.loads(Path(item["result"]).read_text(encoding="utf-8"))
        runs[int(run["seed"])] = run
    return runs


def run_campaign(config: CausalH2Config, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(config.checkpoint)
    token_cache = Path(config.token_cache)
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash differs from the preregistration")
    if _sha256(token_cache) != TOKEN_STREAM_SHA256:
        raise RuntimeError("token stream hash differs from the preregistration")
    parent_runs = _load_evidence(config)
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
        "predecessor_hypothesis_id": PREDECESSOR_HYPOTHESIS_ID,
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
            "predecessor_aggregate_sha256": A6_AGGREGATE_SHA256,
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
            "Dense operators are materialized; A7 tests representation existence, not A9 implicit construction.",
            "The direct H2(A) arm is diagnostic and cannot rescue the kernel-normalized primary verdict.",
            "Theoretical scalar and multiply-add counts are not wall-clock benchmarks.",
            "The frozen checkpoint limits the primary campaign to 256 tokens.",
            "One checkpoint does not establish cross-model repeatability.",
        ],
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CausalH2Config.checkpoint)
    parser.add_argument("--token-cache", default=CausalH2Config.token_cache)
    parser.add_argument("--parent-aggregate", default=CausalH2Config.parent_aggregate)
    parser.add_argument(
        "--predecessor-aggregate", default=CausalH2Config.predecessor_aggregate
    )
    parser.add_argument("--evaluation-seeds", default="101,211,307,401,503")
    parser.add_argument("--lengths", default="32,64,128,256")
    parser.add_argument("--primary-lengths", default="64,128,256")
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--heads", default="0,3,7")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_causal_h2_attention/20260830_registered"
        ),
    )
    args = parser.parse_args(argv)
    seeds = _ints(args.evaluation_seeds)
    lengths = _ints(args.lengths)
    primary_lengths = _ints(args.primary_lengths)
    layers = _ints(args.layers)
    heads = _ints(args.heads)
    if args.shakedown:
        seeds = seeds[:1]
        lengths = tuple(length for length in lengths if length <= 64)
        primary_lengths = (max(lengths),)
        layers = layers[:1]
        heads = heads[:1]
    config = CausalH2Config(
        checkpoint=args.checkpoint,
        token_cache=args.token_cache,
        parent_aggregate=args.parent_aggregate,
        predecessor_aggregate=args.predecessor_aggregate,
        evaluation_seeds=seeds,
        lengths=lengths,
        primary_lengths=primary_lengths,
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
