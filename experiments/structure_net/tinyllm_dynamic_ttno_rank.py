#!/usr/bin/env python3
"""Measure token-tree and quantized-TTNO ranks of frozen TinyLLM attention."""

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
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from structure_net.components.models import TinyLLMModel
from structure_net.components.models.tinyllm_model import TinyLLMConfig


SCHEMA_VERSION = "nal.tinyllm-dynamic-ttno-rank-pilot.v1"
HYPOTHESIS_ID = "tinyllm-dynamic-ttno-rank-pilot-v1"
TREE_ORDERS = ("chronological", "qk_pca")


@dataclass(frozen=True)
class DynamicTTNORankConfig:
    checkpoint: str = (
        "data/experiments/tinyllm_babylm_pretrain/"
        "20260812_d8_seed7/checkpoint_step12000.pt"
    )
    token_cache: str = "data/corpora/babylm_10M_bpe16k.tokens.npy"
    validation_tokens: int = 262_144
    evaluation_seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    lengths: tuple[int, ...] = (32, 64, 128, 256)
    layers: tuple[int, ...] = tuple(range(8))
    heads: tuple[int, ...] = (0, 3, 7)
    epsilons: tuple[float, ...] = (1e-2, 1e-3)
    primary_epsilon: float = 1e-2
    gaussian_tolerance: float = 1e-10
    sparse_k: int = 2
    envelope_fraction: float = 0.80
    random_rank_ratio: float = 0.75
    device: str = "cpu"
    shakedown: bool = False

    def __post_init__(self) -> None:
        if not self.evaluation_seeds:
            raise ValueError("at least one evaluation seed is required")
        if tuple(sorted(set(self.lengths))) != self.lengths:
            raise ValueError("lengths must be unique and strictly increasing")
        if any(length < 4 or length & (length - 1) for length in self.lengths):
            raise ValueError("every context length must be a power of two >= 4")
        if not self.layers or min(self.layers) < 0:
            raise ValueError("layers must be non-empty and nonnegative")
        if not self.heads or min(self.heads) < 0:
            raise ValueError("heads must be non-empty and nonnegative")
        if not self.epsilons or any(not 0.0 < value < 1.0 for value in self.epsilons):
            raise ValueError("epsilons must lie in (0, 1)")
        if self.primary_epsilon not in self.epsilons:
            raise ValueError("primary_epsilon must be included in epsilons")
        if self.validation_tokens <= max(self.lengths):
            raise ValueError("validation suffix must exceed the maximum context length")
        if self.gaussian_tolerance <= 0.0:
            raise ValueError("gaussian_tolerance must be positive")
        if self.sparse_k < 1:
            raise ValueError("sparse_k must be positive")
        if not 0.0 < self.envelope_fraction <= 1.0:
            raise ValueError("envelope_fraction must lie in (0, 1]")
        if not 0.0 < self.random_rank_ratio <= 1.0:
            raise ValueError("random_rank_ratio must lie in (0, 1]")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    dependencies = (
        Path(__file__),
        Path(TinyLLMModel.__module__.replace(".", "/") + ".py"),
    )
    # The installed package path, rather than the relative module spelling, is
    # the outcome-relevant dependency when the working directory differs.
    model_source = Path(__import__(TinyLLMModel.__module__, fromlist=["x"]).__file__)
    for path in (dependencies[0], model_source):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fingerprint(
    config: DynamicTTNORankConfig,
    seed: int,
    checkpoint_sha256: str,
    token_cache_sha256: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "token_cache_sha256": token_cache_sha256,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def numerical_rank_from_singular_values(
    singular_values: np.ndarray, epsilon: float
) -> int:
    """Smallest rank whose discarded Frobenius norm is within ``epsilon``."""
    values = np.asarray(singular_values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("singular_values must be one-dimensional")
    energy = np.square(values)
    total = float(energy.sum())
    if total == 0.0:
        return 0
    tail = np.concatenate((np.cumsum(energy[::-1])[::-1], np.zeros(1)))
    threshold = float(epsilon) ** 2 * total
    candidates = np.flatnonzero(tail <= threshold)
    return int(candidates[0]) if candidates.size else len(values)


def _singular_values(
    matrix: np.ndarray, *, stable_svd: bool = False
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if not matrix.size or min(matrix.shape) == 0:
        return np.empty(0, dtype=np.float64)
    if stable_svd:
        return np.linalg.svd(matrix, compute_uv=False)
    # The eigenvalues of the smaller Gram matrix are exactly the squared
    # singular values.  This is much faster than this environment's LAPACK SVD
    # for the 128--256 square matricizations.  It is sufficiently stable for
    # the declared 1e-2/1e-3 relative-energy ranks; the 1e-12 sparse diagnostic
    # deliberately retains the direct SVD path above.
    gram = matrix @ matrix.T if matrix.shape[0] <= matrix.shape[1] else matrix.T @ matrix
    eigenvalues = np.linalg.eigvalsh(gram)
    return np.sqrt(np.maximum(eigenvalues, 0.0))[::-1]


def paired_operator_tensor(operator: np.ndarray) -> np.ndarray:
    """Pair row/column bits into order-log2(n) modes of dimension four."""
    value = np.asarray(operator, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("operator must be square")
    length = int(value.shape[0])
    if length < 2 or length & (length - 1):
        raise ValueError("operator size must be a power of two")
    bit_count = int(math.log2(length))
    binary = value.reshape((2,) * (2 * bit_count))
    interleaved = tuple(
        index
        for bit in range(bit_count)
        for index in (bit, bit_count + bit)
    )
    return binary.transpose(interleaved).reshape((4,) * bit_count)


def paired_tensor_to_operator(tensor: np.ndarray) -> np.ndarray:
    """Inverse of :func:`paired_operator_tensor`, used by contract tests."""
    value = np.asarray(tensor, dtype=np.float64)
    if value.ndim < 1 or any(size != 4 for size in value.shape):
        raise ValueError("paired tensor modes must all have dimension four")
    bit_count = value.ndim
    binary = value.reshape((2, 2) * bit_count)
    row_axes = tuple(2 * bit for bit in range(bit_count))
    column_axes = tuple(2 * bit + 1 for bit in range(bit_count))
    length = 1 << bit_count
    return binary.transpose(row_axes + column_axes).reshape(length, length)


def balanced_tree_nodes(items: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Return every non-root node of a deterministic balanced binary tree."""
    root = tuple(items)
    if len(root) < 2:
        raise ValueError("a tree needs at least two items")
    nodes: list[tuple[int, ...]] = []

    def visit(node: tuple[int, ...], *, is_root: bool = False) -> None:
        if not is_root:
            nodes.append(node)
        if len(node) > 1:
            middle = len(node) // 2
            visit(node[:middle])
            visit(node[middle:])

    visit(root, is_root=True)
    return tuple(nodes)


def _rank_record(
    singular_values: np.ndarray, epsilons: Iterable[float]
) -> dict[str, int]:
    return {
        f"{epsilon:.12g}": numerical_rank_from_singular_values(
            singular_values, epsilon
        )
        for epsilon in epsilons
    }


def qttno_rank_profile(
    operator: np.ndarray, epsilons: Sequence[float]
) -> dict[str, Any]:
    """Numerical matricization ranks on a balanced paired-bit dimension tree."""
    tensor = paired_operator_tensor(operator)
    mode_count = tensor.ndim
    edges = []
    for node in balanced_tree_nodes(tuple(range(mode_count))):
        complement = tuple(index for index in range(mode_count) if index not in node)
        permutation = node + complement
        rows = 4 ** len(node)
        matrix = tensor.transpose(permutation).reshape(rows, -1)
        singular_values = _singular_values(
            matrix, stable_svd=min(epsilons) < 1e-8
        )
        edges.append(
            {
                "modes": list(node),
                "shape": list(matrix.shape),
                "ranks": _rank_record(singular_values, epsilons),
            }
        )
    maxima = {
        f"{epsilon:.12g}": max(
            edge["ranks"][f"{epsilon:.12g}"] for edge in edges
        )
        for epsilon in epsilons
    }
    return {"max_ranks": maxima, "edges": edges}


def hss_boundary_rank_profile(
    operator: np.ndarray, epsilons: Sequence[float]
) -> dict[str, Any]:
    """HSS-style incoming/outgoing boundary ranks on a token cluster tree."""
    value = np.asarray(operator, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("operator must be square")
    length = value.shape[0]
    universe = np.arange(length)
    edges = []
    for node in balanced_tree_nodes(tuple(range(length))):
        inside = np.fromiter(node, dtype=np.int64)
        mask = np.ones(length, dtype=bool)
        mask[inside] = False
        outside = universe[mask]
        outgoing = value[np.ix_(inside, outside)]
        incoming = value[np.ix_(outside, inside)]
        stable_svd = min(epsilons) < 1e-8
        outgoing_ranks = _rank_record(
            _singular_values(outgoing, stable_svd=stable_svd), epsilons
        )
        incoming_ranks = _rank_record(
            _singular_values(incoming, stable_svd=stable_svd), epsilons
        )
        edges.append(
            {
                "start": int(node[0]),
                "stop": int(node[-1] + 1),
                "cluster_size": len(node),
                "outgoing_ranks": outgoing_ranks,
                "incoming_ranks": incoming_ranks,
            }
        )
    maxima = {
        f"{epsilon:.12g}": max(
            max(
                edge["outgoing_ranks"][f"{epsilon:.12g}"],
                edge["incoming_ranks"][f"{epsilon:.12g}"],
            )
            for edge in edges
        )
        for epsilon in epsilons
    }
    return {"max_ranks": maxima, "edges": edges}


def causal_attention(query: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Float64 causal scaled-dot-product attention matrix."""
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    if query.shape != key.shape or query.ndim != 2:
        raise ValueError("query and key must have the same [token, channel] shape")
    logits = query @ key.T / math.sqrt(query.shape[1])
    logits[np.triu_indices(len(logits), 1)] = -np.inf
    maximum = np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits - maximum)
    weights[np.triu_indices(len(logits), 1)] = 0.0
    return weights / weights.sum(axis=1, keepdims=True)


def gaussian_causal_attention(query: np.ndarray, key: np.ndarray) -> np.ndarray:
    """The same attention operator through the normalized Gaussian identity."""
    query = np.asarray(query, dtype=np.float64)
    key = np.asarray(key, dtype=np.float64)
    if query.shape != key.shape or query.ndim != 2:
        raise ValueError("query and key must have the same [token, channel] shape")
    scale = math.sqrt(query.shape[1])
    squared_distance = (
        np.square(query).sum(1, keepdims=True)
        + np.square(key).sum(1)[None, :]
        - 2.0 * (query @ key.T)
    )
    log_gaussian = -np.maximum(squared_distance, 0.0) / (2.0 * scale)
    log_column_scaling = np.square(key).sum(1)[None, :] / (2.0 * scale)
    logits = log_gaussian + log_column_scaling
    logits[np.triu_indices(len(logits), 1)] = -np.inf
    maximum = np.max(logits, axis=1, keepdims=True)
    weights = np.exp(logits - maximum)
    weights[np.triu_indices(len(logits), 1)] = 0.0
    return weights / weights.sum(axis=1, keepdims=True)


def qk_pca_order(query: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Deterministic Q/K-only leaf order declared by the preregistration."""
    features = np.concatenate((query, key), axis=1).astype(np.float64, copy=False)
    centered = features - features.mean(axis=0, keepdims=True)
    if not np.any(centered):
        return np.arange(len(features), dtype=np.int64)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    component = right[0]
    anchor = int(np.argmax(np.abs(component)))
    if component[anchor] < 0.0:
        component = -component
    scores = centered @ component
    return np.argsort(scores, kind="stable").astype(np.int64)


def tree_orders(query: np.ndarray, key: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "chronological": np.arange(len(query), dtype=np.int64),
        "qk_pca": qk_pca_order(query, key),
    }


def topk_sparse(operator: np.ndarray, count: int) -> tuple[np.ndarray, float]:
    """Retain the largest ``count`` nonzero entries in every causal row."""
    value = np.asarray(operator, dtype=np.float64)
    sparse = np.zeros_like(value)
    for row in range(len(value)):
        available = np.flatnonzero(value[row] > 0.0)
        if not available.size:
            continue
        keep_count = min(count, len(available))
        local = np.argpartition(value[row, available], -keep_count)[-keep_count:]
        keep = available[local]
        sparse[row, keep] = value[row, keep]
    mass = float(sparse.sum() / value.sum()) if value.sum() else 0.0
    return sparse, mass


def _analyze_ordered_operator(
    operator: np.ndarray,
    order: np.ndarray,
    epsilons: Sequence[float],
) -> dict[str, Any]:
    ordered = operator[np.ix_(order, order)]
    return {
        "qttno": qttno_rank_profile(ordered, epsilons),
        "hss_boundary": hss_boundary_rank_profile(ordered, epsilons),
    }


def analyze_qk(
    query: np.ndarray,
    key: np.ndarray,
    config: DynamicTTNORankConfig,
    *,
    include_sparse: bool,
) -> dict[str, Any]:
    attention = causal_attention(query, key)
    gaussian = gaussian_causal_attention(query, key)
    error = float(np.max(np.abs(attention - gaussian)))
    order_records: dict[str, Any] = {}
    orders = tree_orders(query, key)
    for name in TREE_ORDERS:
        order = orders[name]
        record = _analyze_ordered_operator(attention, order, config.epsilons)
        if include_sparse:
            sparse, mass = topk_sparse(attention, config.sparse_k)
            remainder = attention - sparse
            ordered_sparse = sparse[np.ix_(order, order)]
            ordered_remainder = remainder[np.ix_(order, order)]
            exact_key = "1e-12"
            sparse_profile = qttno_rank_profile(ordered_sparse, (1e-12,))
            remainder_profile = qttno_rank_profile(
                ordered_remainder, (config.primary_epsilon,)
            )
            record["sparse_exception"] = {
                "k_per_row": config.sparse_k,
                "mass_fraction": mass,
                "sparse_exact_max_rank": sparse_profile["max_ranks"][exact_key],
                "remainder_max_rank": remainder_profile["max_ranks"][
                    f"{config.primary_epsilon:.12g}"
                ],
            }
        order_records[name] = record
    epsilon_key = f"{config.primary_epsilon:.12g}"
    return {
        "gaussian_max_absolute_error": error,
        "orders": order_records,
        "best_declared_tree": {
            "qttno_max_rank": min(
                record["qttno"]["max_ranks"][epsilon_key]
                for record in order_records.values()
            ),
            "hss_boundary_max_rank": min(
                record["hss_boundary"]["max_ranks"][epsilon_key]
                for record in order_records.values()
            ),
        },
    }


def _smooth_fourier(length: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(length, dtype=np.float64) / max(1, length - 1)
    frequencies = np.arange(1, width // 2 + 1, dtype=np.float64)
    phase = 2.0 * math.pi * positions[:, None] * frequencies[None, :]
    features = np.concatenate((np.sin(phase), np.cos(phase)), axis=1)
    if features.shape[1] < width:
        features = np.pad(features, ((0, 0), (0, width - features.shape[1])))
    return features[:, :width], features[:, :width].copy()


def control_qk(
    condition: str, length: int, width: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "causal_uniform":
        return np.zeros((length, width)), np.zeros((length, width))
    if condition == "smooth_fourier":
        return _smooth_fourier(length, width)
    if condition == "iid_qk":
        generator = np.random.default_rng((seed, length, width, 91_337))
        return (
            generator.standard_normal((length, width)),
            generator.standard_normal((length, width)),
        )
    raise ValueError(f"unknown control condition {condition!r}")


def _load_model(checkpoint: Path, device: torch.device) -> TinyLLMModel:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("pretraining checkpoint must contain a mapping")
    if "model_config" not in payload or "model_state" not in payload:
        raise ValueError("pretraining checkpoint lacks model_config/model_state")
    model = TinyLLMModel(TinyLLMConfig(**payload["model_config"]), name="DynamicTTNORank")
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def extract_selected_qk(
    model: TinyLLMModel,
    input_ids: torch.Tensor,
    layers: Sequence[int],
    heads: Sequence[int],
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """Capture selected head Q/K arrays while executing the ordinary model."""
    selected_layers = set(layers)
    selected_heads = set(heads)
    if max(selected_layers) >= model.config.n_layer:
        raise ValueError("selected layer exceeds model depth")
    if max(selected_heads) >= model.config.n_head:
        raise ValueError("selected head exceeds model head count")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    value = model.transformer["wte"](input_ids) + model.transformer["wpe"](positions)
    output: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for layer_index, block in enumerate(model.transformer["h"]):
        normalized = block.ln_1(value)
        if layer_index in selected_layers:
            query, key, _ = block.attn.c_attn(normalized).split(
                block.attn.n_embd, dim=-1
            )
            head_width = model.config.n_embd // model.config.n_head
            query = query.view(1, input_ids.shape[1], model.config.n_head, head_width)
            key = key.view(1, input_ids.shape[1], model.config.n_head, head_width)
            for head_index in heads:
                output[(layer_index, head_index)] = (
                    query[0, :, head_index].double().cpu().numpy(),
                    key[0, :, head_index].double().cpu().numpy(),
                )
        value = block(value)
    return output


def _validation_segment(
    tokens: np.ndarray,
    validation_tokens: int,
    maximum_length: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    if len(tokens) < validation_tokens:
        raise ValueError("token cache is shorter than the declared validation suffix")
    validation = tokens[-validation_tokens:]
    generator = np.random.default_rng((seed, 73_001))
    start = int(generator.integers(0, len(validation) - maximum_length + 1))
    return np.asarray(validation[start : start + maximum_length], dtype=np.int64), start


def analyze_seed(
    config: DynamicTTNORankConfig,
    seed: int,
    model: TinyLLMModel,
    tokens: np.ndarray,
    output: Path,
    checkpoint_sha256: str,
    token_cache_sha256: str,
    implementation_sha256: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    device = next(model.parameters()).device
    segment, validation_start = _validation_segment(
        tokens, config.validation_tokens, max(config.lengths), seed
    )
    input_ids = torch.from_numpy(segment.copy()).unsqueeze(0).to(device)
    selected = extract_selected_qk(model, input_ids, config.layers, config.heads)
    natural_cells = []
    for layer in config.layers:
        for head in config.heads:
            query, key = selected[(layer, head)]
            lengths = {}
            for length in config.lengths:
                lengths[str(length)] = analyze_qk(
                    query[:length],
                    key[:length],
                    config,
                    include_sparse=length == max(config.lengths),
                )
            natural_cells.append({"layer": layer, "head": head, "lengths": lengths})

    head_width = model.config.n_embd // model.config.n_head
    controls: dict[str, Any] = {}
    for condition in ("causal_uniform", "smooth_fourier", "iid_qk"):
        controls[condition] = {}
        for length in config.lengths:
            query, key = control_qk(condition, length, head_width, seed)
            controls[condition][str(length)] = analyze_qk(
                query, key, config, include_sparse=False
            )

    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-dynamic-ttno-rank-seed{seed}",
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.shakedown
            else "preregistered_exploratory_pilot"
        ),
        "completed_at": _utc_now(),
        "seed": seed,
        "configuration": asdict(config),
        "scientific_fingerprint": _fingerprint(
            config, seed, checkpoint_sha256, token_cache_sha256
        ),
        "implementation_sha256": implementation_sha256,
        "provenance": {
            "checkpoint": config.checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "token_cache": config.token_cache,
            "token_cache_sha256": token_cache_sha256,
            "validation_start": validation_start,
            "validation_stop": validation_start + max(config.lengths),
        },
        "natural_cells": natural_cells,
        "controls": controls,
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(output / "runs" / f"seed_{seed}" / "result.json", result)
    return result


def _median(values: Sequence[float | int]) -> float:
    if not values:
        raise ValueError("cannot take a median of an empty sequence")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _rank_key(config: DynamicTTNORankConfig) -> str:
    return f"{config.primary_epsilon:.12g}"


def _best_rank(record: Mapping[str, Any], family: str, epsilon_key: str) -> int:
    return min(
        int(order[family]["max_ranks"][epsilon_key])
        for order in record["orders"].values()
    )


def aggregate(
    runs: Sequence[Mapping[str, Any]], config: DynamicTTNORankConfig
) -> dict[str, Any]:
    epsilon_key = _rank_key(config)
    natural_by_length: dict[int, list[int]] = {length: [] for length in config.lengths}
    hss_by_length: dict[int, list[int]] = {length: [] for length in config.lengths}
    sensitivity_by_length: dict[int, list[int]] = {
        length: [] for length in config.lengths
    }
    gaussian_errors: list[float] = []
    cell_envelope_passes: list[bool] = []
    sparse_full: list[int] = []
    sparse_remainders: list[int] = []
    sparse_exact: list[int] = []
    sparse_mass: list[float] = []
    sparse_improvements: list[bool] = []

    for run in runs:
        for cell in run["natural_cells"]:
            envelope_pass = True
            for length in config.lengths:
                record = cell["lengths"][str(length)]
                gaussian_errors.append(float(record["gaussian_max_absolute_error"]))
                rank = _best_rank(record, "qttno", epsilon_key)
                natural_by_length[length].append(rank)
                hss_by_length[length].append(
                    _best_rank(record, "hss_boundary", epsilon_key)
                )
                sensitivity_key = f"{config.epsilons[-1]:.12g}"
                sensitivity_by_length[length].append(
                    _best_rank(record, "qttno", sensitivity_key)
                )
                envelope = int(math.ceil(math.log2(length) ** 2))
                envelope_pass = envelope_pass and rank <= envelope
            cell_envelope_passes.append(envelope_pass)

            final = cell["lengths"][str(max(config.lengths))]
            for order in final["orders"].values():
                sparse = order["sparse_exception"]
                full_rank = int(order["qttno"]["max_ranks"][epsilon_key])
                remainder_rank = int(sparse["remainder_max_rank"])
                sparse_full.append(full_rank)
                sparse_remainders.append(remainder_rank)
                sparse_exact.append(int(sparse["sparse_exact_max_rank"]))
                sparse_mass.append(float(sparse["mass_fraction"]))
                sparse_improvements.append(remainder_rank < full_rank)

    control_ranks: dict[str, dict[int, list[int]]] = {
        condition: {length: [] for length in config.lengths}
        for condition in ("causal_uniform", "smooth_fourier", "iid_qk")
    }
    for run in runs:
        for condition, length_records in run["controls"].items():
            for length in config.lengths:
                control_ranks[condition][length].append(
                    _best_rank(length_records[str(length)], "qttno", epsilon_key)
                )

    medians = {
        str(length): {
            "qttno": _median(natural_by_length[length]),
            "hss_boundary": _median(hss_by_length[length]),
            "qttno_sensitivity": _median(sensitivity_by_length[length]),
            "polylog_envelope": int(math.ceil(math.log2(length) ** 2)),
            "qttno_rank_range": [
                min(natural_by_length[length]),
                max(natural_by_length[length]),
            ],
        }
        for length in config.lengths
    }
    envelope_pass_fraction = float(np.mean(cell_envelope_passes))
    envelope_pass = envelope_pass_fraction >= config.envelope_fraction
    length64 = 64 if 64 in config.lengths else config.lengths[0]
    length256 = 256 if 256 in config.lengths else config.lengths[-1]
    normalized_start = _median(natural_by_length[length64]) / math.log2(length64) ** 2
    normalized_end = _median(natural_by_length[length256]) / math.log2(length256) ** 2
    normalized_growth_pass = normalized_end <= normalized_start
    gaussian_max = max(gaussian_errors)
    gaussian_pass = gaussian_max <= config.gaussian_tolerance
    natural_final = _median(natural_by_length[length256])
    iid_final = _median(control_ranks["iid_qk"][length256])
    random_ratio = natural_final / max(iid_final, 1e-12)
    random_separation_pass = random_ratio <= config.random_rank_ratio

    if not gaussian_pass:
        classification = "invalid_arithmetic_contract"
    elif envelope_pass and normalized_growth_pass:
        classification = "polylog_compatible_pilot"
    elif envelope_pass or normalized_growth_pass:
        classification = "mixed_rank_pilot"
    else:
        classification = "bond_growth_observed_pilot"

    return {
        "classification": classification,
        "gaussian_identity": {
            "max_absolute_error": gaussian_max,
            "tolerance": config.gaussian_tolerance,
            "pass": gaussian_pass,
        },
        "polylog_envelope": {
            "passing_cell_fraction": envelope_pass_fraction,
            "required_fraction": config.envelope_fraction,
            "pass": envelope_pass,
        },
        "normalized_growth": {
            "start_length": length64,
            "end_length": length256,
            "start_median_rank_over_log2_squared": normalized_start,
            "end_median_rank_over_log2_squared": normalized_end,
            "pass": normalized_growth_pass,
        },
        "random_separation": {
            "natural_median_rank": natural_final,
            "iid_qk_median_rank": iid_final,
            "ratio": random_ratio,
            "maximum_ratio": config.random_rank_ratio,
            "pass": random_separation_pass,
            "role": "secondary_diagnostic",
        },
        "natural_rank_summary": medians,
        "control_rank_summary": {
            condition: {
                str(length): {
                    "median": _median(values[length]),
                    "range": [min(values[length]), max(values[length])],
                }
                for length in config.lengths
            }
            for condition, values in control_ranks.items()
        },
        "sparse_exception": {
            "k_per_row": config.sparse_k,
            "median_mass_fraction": _median(sparse_mass),
            "median_full_qttno_rank": _median(sparse_full),
            "median_remainder_qttno_rank": _median(sparse_remainders),
            "median_sparse_exact_qttno_rank": _median(sparse_exact),
            "remainder_rank_improvement_fraction": float(
                np.mean(sparse_improvements)
            ),
            "role": "secondary_diagnostic",
        },
        "counts": {
            "evaluation_seeds": len(runs),
            "natural_cells": sum(len(run["natural_cells"]) for run in runs),
            "natural_operator_lengths": sum(
                len(run["natural_cells"]) * len(config.lengths) for run in runs
            ),
        },
    }


def run_campaign(config: DynamicTTNORankConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(config.checkpoint)
    token_cache = Path(config.token_cache)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not token_cache.is_file():
        raise FileNotFoundError(token_cache)
    checkpoint_sha256 = _sha256(checkpoint)
    token_cache_sha256 = _sha256(token_cache)
    implementation_sha256 = _implementation_digest()
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(True)
    model = _load_model(checkpoint, device)
    if max(config.lengths) > model.config.block_size:
        raise ValueError("context length exceeds the checkpoint block size")
    tokens = np.load(token_cache, mmap_mode="r")

    runs = []
    reused = 0
    for seed in config.evaluation_seeds:
        result_path = output / "runs" / f"seed_{seed}" / "result.json"
        expected = _fingerprint(
            config, seed, checkpoint_sha256, token_cache_sha256
        )
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "completed"
                and existing.get("schema_version") == SCHEMA_VERSION
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
            output,
            checkpoint_sha256,
            token_cache_sha256,
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
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.shakedown
            else "preregistered_exploratory_pilot"
        ),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation_sha256,
        "provenance": {
            "checkpoint_sha256": checkpoint_sha256,
            "token_cache_sha256": token_cache_sha256,
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
                "result": str(output / "runs" / f"seed_{run['seed']}" / "result.json"),
            }
            for run in runs
        ],
        "method_boundaries": [
            "One pretrained checkpoint does not test variation across learned models.",
            "Evaluation seeds select validation prefixes; they are not model-training seeds.",
            "Lengths 32--256 cannot establish an asymptotic complexity law.",
            "Dense attention is materialized, so this is not a subquadratic implementation benchmark.",
            "The QTT/TTNO tree has log2(n) paired bit modes, not one tensor-product site per token.",
            "HSS boundary ranks and QTT/TTNO ranks are separate diagnostics and are not interchangeable.",
            "Per-cut numerical ranks do not construct one simultaneous epsilon-accurate TTNO.",
            "The top-k split is a diagnostic, not a learned sparse router or an H2 near-field partition.",
        ],
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DynamicTTNORankConfig.checkpoint)
    parser.add_argument("--token-cache", default=DynamicTTNORankConfig.token_cache)
    parser.add_argument("--evaluation-seeds", default="101,211,307,401,503")
    parser.add_argument("--lengths", default="32,64,128,256")
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--heads", default="0,3,7")
    parser.add_argument("--epsilons", default="0.01,0.001")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_dynamic_ttno_rank/"
            "20260829_d8_babylm_pilot"
        ),
    )
    args = parser.parse_args(argv)
    if args.shakedown:
        seeds = (_ints(args.evaluation_seeds) or (101,))[:1]
        lengths = tuple(length for length in _ints(args.lengths) if length <= 64)
        layers = (_ints(args.layers) or (0,))[:1]
        heads = (_ints(args.heads) or (0,))[:1]
    else:
        seeds = _ints(args.evaluation_seeds)
        lengths = _ints(args.lengths)
        layers = _ints(args.layers)
        heads = _ints(args.heads)
    config = DynamicTTNORankConfig(
        checkpoint=args.checkpoint,
        token_cache=args.token_cache,
        evaluation_seeds=seeds,
        lengths=lengths,
        layers=layers,
        heads=heads,
        epsilons=_floats(args.epsilons),
        primary_epsilon=_floats(args.epsilons)[0],
        device=args.device,
        shakedown=args.shakedown,
    )
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    print(args.output / "campaign_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
