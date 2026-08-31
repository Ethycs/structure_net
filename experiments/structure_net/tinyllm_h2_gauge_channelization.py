#!/usr/bin/env python3
"""Execute preregistered A8 internal-gauge H2 channelization."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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

try:
    from experiments.structure_net import tinyllm_causal_h2_attention as a7
except ModuleNotFoundError:  # Direct script execution.
    import tinyllm_causal_h2_attention as a7  # type: ignore[no-redef]


SCHEMA_VERSION = "nal.tinyllm-h2-gauge-channelization.v1"
HYPOTHESIS_ID = "tinyllm-h2-gauge-channelization-v1"
PARENT_HYPOTHESIS_ID = "tinyllm-causal-h2-attention-v1"
PARENT_AGGREGATE_SHA256 = (
    "4885696dd746a52cb015b51d34733901c2acd50baccfc59f2f76cfe176eeb9b2"
)
PARENT_IMPLEMENTATION_SHA256 = (
    "9e61430463ecc11ff5d99560625a9f85a6e5b189d39a43fdbcf26bb06e804dbc"
)
A6_AGGREGATE_SHA256 = (
    "6f42a59b3a723eb4b80742e8fab8278be9b21d1db35131cc4ea81bd702e03c01"
)
PREREGISTRATION_SHA256 = (
    "59451a7cc0e61de73335f5c455954b355ebb9611c5c5cb4133ec1d27acd4a53b"
)


@dataclass(frozen=True)
class GaugeChannelizationConfig:
    checkpoint: str = a7.CausalH2Config.checkpoint
    token_cache: str = a7.CausalH2Config.token_cache
    parent_aggregate: str = (
        "data/experiments/tinyllm_causal_h2_attention/"
        "20260830_registered/campaign_results.json"
    )
    preregistration: str = (
        "docs/07 - Status Reports/"
        "2026-08-30_tinyllm-h2-gauge-channelization-preregistration.md"
    )
    validation_tokens: int = 262_144
    evaluation_seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    lengths: tuple[int, ...] = (64, 128, 256)
    primary_length: int = 256
    layers: tuple[int, ...] = tuple(range(8))
    heads: tuple[int, ...] = (0, 3, 7)
    block_widths: tuple[int, ...] = (1, 2, 4)
    optimizer_updates: int = 96
    optimizer_learning_rate: float = 0.03
    optimizer_restarts: tuple[str, ...] = ("identity", "spectral_covariance")
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
    gauge_orthogonality_maximum: float = 1e-10
    exact_equivalence_maximum: float = 1e-10
    device: str = "cuda:0"
    shakedown: bool = False

    def __post_init__(self) -> None:
        if not self.evaluation_seeds or not self.layers or not self.heads:
            raise ValueError("evaluation seeds, layers, and heads must be non-empty")
        if tuple(sorted(set(self.lengths))) != self.lengths:
            raise ValueError("lengths must be unique and increasing")
        if self.primary_length not in self.lengths:
            raise ValueError("primary length must be included in lengths")
        if self.block_widths != (1, 2, 4):
            raise ValueError("the preregistered block widths are exactly 1, 2, and 4")
        if self.optimizer_updates <= 0 or self.optimizer_learning_rate <= 0.0:
            raise ValueError("optimizer settings must be positive")
        if self.optimizer_restarts != ("identity", "spectral_covariance"):
            raise ValueError("the two preregistered restarts are frozen")
        if not self.shakedown:
            frozen = {
                "evaluation_seeds": (self.evaluation_seeds, (101, 211, 307, 401, 503)),
                "lengths": (self.lengths, (64, 128, 256)),
                "primary_length": (self.primary_length, 256),
                "layers": (self.layers, tuple(range(8))),
                "heads": (self.heads, (0, 3, 7)),
                "optimizer_updates": (self.optimizer_updates, 96),
            }
            changed = [name for name, (actual, expected) in frozen.items() if actual != expected]
            if changed:
                raise ValueError(
                    "primary A8 settings are frozen; changed: " + ", ".join(changed)
                )
            if not self.device.startswith("cuda"):
                raise ValueError("the preregistered primary A8 campaign requires CUDA")

    def a7_config(self) -> a7.CausalH2Config:
        return a7.CausalH2Config(
            checkpoint=self.checkpoint,
            token_cache=self.token_cache,
            validation_tokens=self.validation_tokens,
            evaluation_seeds=self.evaluation_seeds,
            lengths=self.lengths,
            primary_lengths=self.lengths,
            layers=self.layers,
            heads=self.heads,
            leaf_size=self.leaf_size,
            separation_ratio=self.separation_ratio,
            build_tolerance=self.build_tolerance,
            rank_multiplier=self.rank_multiplier,
            probe_seed=self.probe_seed,
            probe_columns=self.probe_columns,
            kernel_row_relative_maximum=self.kernel_row_relative_maximum,
            denominator_relative_maximum=self.denominator_relative_maximum,
            attention_row_l1_maximum=self.attention_row_l1_maximum,
            probe_relative_frobenius_maximum=(
                self.probe_relative_frobenius_maximum
            ),
            value_relative_frobenius_maximum=(
                self.value_relative_frobenius_maximum
            ),
            token_tail_maximum=self.token_tail_maximum,
            cell_pass_fraction_minimum=self.cell_pass_fraction_minimum,
            layer_pass_fraction_minimum=self.layer_pass_fraction_minimum,
            storage_ratio_median_maximum=self.storage_ratio_median_maximum,
            storage_ratio_p90_maximum=self.storage_ratio_p90_maximum,
            operation_ratio_median_maximum=self.operation_ratio_median_maximum,
            operation_ratio_p90_maximum=self.operation_ratio_p90_maximum,
            device=self.device,
            shakedown=self.shakedown,
        )


@dataclass(frozen=True)
class Factor:
    kind: str
    key: tuple[str, str]
    matrix: torch.Tensor
    left_family: str
    left_node: str
    right_family: str
    right_node: str


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


def _sha256(path: str | Path) -> str:
    return a7._sha256(Path(path))


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(a7.__file__)):
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fingerprint(
    config: GaugeChannelizationConfig,
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
        "parent_implementation_sha256": PARENT_IMPLEMENTATION_SHA256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def block_mask(
    rows: int,
    columns: int,
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return the public fixed-width rectangular channel mask."""

    if width <= 0:
        raise ValueError("block width must be positive")
    row_blocks = torch.arange(rows, device=device) // width
    column_blocks = torch.arange(columns, device=device) // width
    return row_blocks[:, None] == column_blocks[None, :]


def _factors(representation: a7.H2Representation) -> list[Factor]:
    factors: list[Factor] = []
    for key, matrix in representation.query_transfers.items():
        child, parent = key
        factors.append(
            Factor("query_transfer", key, matrix, "query", child, "query", parent)
        )
    for key, matrix in representation.key_transfers.items():
        child, parent = key
        factors.append(
            Factor("key_transfer", key, matrix, "key", child, "key", parent)
        )
    for key, matrix in representation.couplings.items():
        query, key_node = key
        factors.append(
            Factor("coupling", key, matrix, "query", query, "key", key_node)
        )
    return factors


def _ranks(
    representation: a7.H2Representation,
) -> tuple[dict[str, int], dict[str, int]]:
    return (
        {node: int(basis.shape[1]) for node, basis in representation.query_bases.items()},
        {node: int(basis.shape[1]) for node, basis in representation.key_bases.items()},
    )


def _identity_gauges(
    representation: a7.H2Representation,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    query = {
        node: torch.eye(
            basis.shape[1], dtype=basis.dtype, device=basis.device
        )
        for node, basis in representation.query_bases.items()
    }
    key = {
        node: torch.eye(
            basis.shape[1], dtype=basis.dtype, device=basis.device
        )
        for node, basis in representation.key_bases.items()
    }
    return query, key


def _canonical_eigenbasis(covariance: torch.Tensor) -> torch.Tensor:
    rank = covariance.shape[0]
    if rank == 0:
        return covariance.clone()
    if float(torch.linalg.norm(covariance).item()) == 0.0:
        return torch.eye(rank, dtype=covariance.dtype, device=covariance.device)
    _, vectors = torch.linalg.eigh(covariance)
    vectors = torch.flip(vectors, dims=(1,)).clone()
    for column in range(rank):
        pivot = int(torch.argmax(torch.abs(vectors[:, column])).item())
        if float(vectors[pivot, column].item()) < 0.0:
            vectors[:, column] *= -1.0
    return vectors


def _spectral_gauges(
    representation: a7.H2Representation,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    query_ranks, key_ranks = _ranks(representation)
    query_covariances = {
        node: torch.zeros(
            (rank, rank),
            dtype=representation.dense_operator.dtype,
            device=representation.dense_operator.device,
        )
        for node, rank in query_ranks.items()
    }
    key_covariances = {
        node: torch.zeros(
            (rank, rank),
            dtype=representation.dense_operator.dtype,
            device=representation.dense_operator.device,
        )
        for node, rank in key_ranks.items()
    }
    covariances = {"query": query_covariances, "key": key_covariances}
    for factor in _factors(representation):
        energy = torch.sum(factor.matrix * factor.matrix)
        if float(energy.item()) == 0.0:
            continue
        covariances[factor.left_family][factor.left_node] += (
            factor.matrix @ factor.matrix.T
        ) / energy
        covariances[factor.right_family][factor.right_node] += (
            factor.matrix.T @ factor.matrix
        ) / energy
    return (
        {node: _canonical_eigenbasis(value) for node, value in query_covariances.items()},
        {node: _canonical_eigenbasis(value) for node, value in key_covariances.items()},
    )


def _cayley(raw: torch.Tensor, initializer: torch.Tensor) -> torch.Tensor:
    if raw.shape[0] == 0:
        return initializer
    skew = 0.5 * (raw - raw.T)
    identity = torch.eye(raw.shape[0], dtype=raw.dtype, device=raw.device)
    delta = torch.linalg.solve(identity + skew, identity - skew)
    return initializer @ delta


class _GaugeParameters(torch.nn.Module):
    def __init__(
        self,
        query_initial: Mapping[str, torch.Tensor],
        key_initial: Mapping[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.query_nodes = tuple(sorted(query_initial))
        self.key_nodes = tuple(sorted(key_initial))
        self.query_initial = {
            node: value.detach().clone() for node, value in query_initial.items()
        }
        self.key_initial = {
            node: value.detach().clone() for node, value in key_initial.items()
        }
        self.query_raw = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros_like(query_initial[node])) for node in self.query_nodes]
        )
        self.key_raw = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros_like(key_initial[node])) for node in self.key_nodes]
        )

    def gauges(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        query = {
            node: _cayley(raw, self.query_initial[node])
            for node, raw in zip(self.query_nodes, self.query_raw)
        }
        key = {
            node: _cayley(raw, self.key_initial[node])
            for node, raw in zip(self.key_nodes, self.key_raw)
        }
        return query, key


def _batched_cayley(raw: torch.Tensor, initializer: torch.Tensor) -> torch.Tensor:
    if raw.shape[-1] == 0:
        return initializer
    skew = 0.5 * (raw - raw.transpose(-1, -2))
    identity = torch.eye(raw.shape[-1], dtype=raw.dtype, device=raw.device)
    identity = identity.expand(raw.shape[0], -1, -1)
    delta = torch.linalg.solve(identity + skew, identity - skew)
    return initializer @ delta


class _BatchedGaugeParameters(torch.nn.Module):
    def __init__(
        self,
        query_initial: Mapping[str, torch.Tensor],
        key_initial: Mapping[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.query_nodes = tuple(sorted(query_initial))
        self.key_nodes = tuple(sorted(key_initial))
        self.query_initial = {
            node: value.detach().clone() for node, value in query_initial.items()
        }
        self.key_initial = {
            node: value.detach().clone() for node, value in key_initial.items()
        }
        self.query_raw = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros_like(query_initial[node])) for node in self.query_nodes]
        )
        self.key_raw = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros_like(key_initial[node])) for node in self.key_nodes]
        )

    def gauges(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        query = {
            node: _batched_cayley(raw, self.query_initial[node])
            for node, raw in zip(self.query_nodes, self.query_raw)
        }
        key = {
            node: _batched_cayley(raw, self.key_initial[node])
            for node, raw in zip(self.key_nodes, self.key_raw)
        }
        return query, key


def _transformed_factor(
    factor: Factor,
    query_gauges: Mapping[str, torch.Tensor],
    key_gauges: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    gauges = {"query": query_gauges, "key": key_gauges}
    return (
        gauges[factor.left_family][factor.left_node].T
        @ factor.matrix
        @ gauges[factor.right_family][factor.right_node]
    )


def offblock_objective(
    representation: a7.H2Representation,
    query_gauges: Mapping[str, torch.Tensor],
    key_gauges: Mapping[str, torch.Tensor],
    block_width: int,
) -> torch.Tensor:
    numerator = torch.zeros(
        (),
        dtype=representation.dense_operator.dtype,
        device=representation.dense_operator.device,
    )
    denominator = torch.zeros_like(numerator)
    for factor in _factors(representation):
        transformed = _transformed_factor(factor, query_gauges, key_gauges)
        mask = block_mask(
            transformed.shape[0],
            transformed.shape[1],
            block_width,
            device=transformed.device,
        )
        numerator = numerator + torch.sum(transformed[~mask] ** 2)
        denominator = denominator + torch.sum(factor.matrix * factor.matrix)
    return numerator / torch.clamp(denominator, min=1e-30)


def _batched_offblock_objective(
    representation: a7.H2Representation,
    query_gauges: Mapping[str, torch.Tensor],
    key_gauges: Mapping[str, torch.Tensor],
    candidate_widths: Sequence[int],
) -> torch.Tensor:
    candidate_count = len(candidate_widths)
    numerator = torch.zeros(
        candidate_count,
        dtype=representation.dense_operator.dtype,
        device=representation.dense_operator.device,
    )
    denominator = torch.zeros(
        (),
        dtype=representation.dense_operator.dtype,
        device=representation.dense_operator.device,
    )
    gauges = {"query": query_gauges, "key": key_gauges}
    for factor in _factors(representation):
        left = gauges[factor.left_family][factor.left_node]
        right = gauges[factor.right_family][factor.right_node]
        transformed = (
            left.transpose(-1, -2)
            @ factor.matrix.unsqueeze(0)
            @ right
        )
        masks = torch.stack(
            [
                block_mask(
                    transformed.shape[1],
                    transformed.shape[2],
                    width,
                    device=transformed.device,
                )
                for width in candidate_widths
            ],
            dim=0,
        )
        numerator = numerator + torch.sum(
            transformed.square() * (~masks).to(transformed.dtype),
            dim=(1, 2),
        )
        denominator = denominator + torch.sum(factor.matrix * factor.matrix)
    return numerator / torch.clamp(denominator, min=1e-30)


def _orthogonality_maximum(
    query_gauges: Mapping[str, torch.Tensor],
    key_gauges: Mapping[str, torch.Tensor],
) -> float:
    maximum = 0.0
    for gauge in (*query_gauges.values(), *key_gauges.values()):
        if gauge.shape[0] == 0:
            continue
        identity = torch.eye(gauge.shape[0], dtype=gauge.dtype, device=gauge.device)
        residual = torch.linalg.matrix_norm(gauge.T @ gauge - identity, ord=2)
        maximum = max(maximum, float(residual.item()))
    return maximum


def optimize_gauges(
    representation: a7.H2Representation,
    block_width: int,
    config: GaugeChannelizationConfig,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Optimize the preregistered factor-only objective."""

    candidates: list[tuple[float, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]] = []
    initializers = {
        "identity": _identity_gauges(representation),
        "spectral_covariance": _spectral_gauges(representation),
    }
    for restart in config.optimizer_restarts:
        query_initial, key_initial = initializers[restart]
        parameters = _GaugeParameters(query_initial, key_initial)
        trainable = [item for item in parameters.parameters() if item.numel() > 0]
        with torch.no_grad():
            initial_loss = float(
                offblock_objective(
                    representation, query_initial, key_initial, block_width
                ).item()
            )
        if trainable:
            optimizer = torch.optim.Adam(
                trainable,
                lr=config.optimizer_learning_rate,
                weight_decay=0.0,
                eps=1e-4,
            )
            for _ in range(config.optimizer_updates):
                optimizer.zero_grad(set_to_none=True)
                query_gauges, key_gauges = parameters.gauges()
                loss = offblock_objective(
                    representation, query_gauges, key_gauges, block_width
                )
                loss.backward()
                optimizer.step()
        query_gauges, key_gauges = parameters.gauges()
        query_final = {node: value.detach() for node, value in query_gauges.items()}
        key_final = {node: value.detach() for node, value in key_gauges.items()}
        final_loss = float(
            offblock_objective(
                representation, query_final, key_final, block_width
            ).item()
        )
        candidates.append(
            (
                final_loss,
                query_final,
                key_final,
                {
                    "restart": restart,
                    "initial_objective": initial_loss,
                    "final_objective": final_loss,
                    "updates": config.optimizer_updates,
                },
            )
        )
    candidates.sort(key=lambda item: (item[0], item[3]["restart"]))
    _, query_best, key_best, selected = candidates[0]
    return query_best, key_best, {
        "selected_restart": selected["restart"],
        "selected_initial_objective": selected["initial_objective"],
        "selected_final_objective": selected["final_objective"],
        "restart_results": [item[3] for item in candidates],
        "orthogonality_maximum": _orthogonality_maximum(query_best, key_best),
    }


def optimize_all_gauges(
    representation: a7.H2Representation,
    config: GaugeChannelizationConfig,
) -> dict[int, tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]]:
    """Run all preregistered block/restart candidates as one GPU batch."""

    identity_query, identity_key = _identity_gauges(representation)
    spectral_query, spectral_key = _spectral_gauges(representation)
    initializers = {
        "identity": (identity_query, identity_key),
        "spectral_covariance": (spectral_query, spectral_key),
    }
    candidates = [
        (width, restart)
        for width in config.block_widths
        for restart in config.optimizer_restarts
    ]
    candidate_widths = [width for width, _ in candidates]
    query_initial = {
        node: torch.stack(
            [initializers[restart][0][node] for _, restart in candidates], dim=0
        )
        for node in identity_query
    }
    key_initial = {
        node: torch.stack(
            [initializers[restart][1][node] for _, restart in candidates], dim=0
        )
        for node in identity_key
    }
    parameters = _BatchedGaugeParameters(query_initial, key_initial)
    trainable = [item for item in parameters.parameters() if item.numel() > 0]
    with torch.no_grad():
        initial_objectives = _batched_offblock_objective(
            representation, query_initial, key_initial, candidate_widths
        ).detach()
    if trainable:
        optimizer = torch.optim.Adam(
            trainable,
            lr=config.optimizer_learning_rate,
            weight_decay=0.0,
            eps=1e-4,
        )
        for _ in range(config.optimizer_updates):
            optimizer.zero_grad(set_to_none=True)
            query_gauges, key_gauges = parameters.gauges()
            objectives = _batched_offblock_objective(
                representation,
                query_gauges,
                key_gauges,
                candidate_widths,
            )
            torch.sum(objectives).backward()
            optimizer.step()
    query_final_batch, key_final_batch = parameters.gauges()
    query_final_batch = {
        node: value.detach() for node, value in query_final_batch.items()
    }
    key_final_batch = {
        node: value.detach() for node, value in key_final_batch.items()
    }
    final_objectives = _batched_offblock_objective(
        representation,
        query_final_batch,
        key_final_batch,
        candidate_widths,
    ).detach()
    output: dict[
        int,
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]],
    ] = {}
    for width in config.block_widths:
        indices = [index for index, item in enumerate(candidates) if item[0] == width]
        selected_index = min(
            indices,
            key=lambda index: (
                float(final_objectives[index].item()),
                candidates[index][1],
            ),
        )
        query_selected = {
            node: value[selected_index] for node, value in query_final_batch.items()
        }
        key_selected = {
            node: value[selected_index] for node, value in key_final_batch.items()
        }
        restart_results = [
            {
                "restart": candidates[index][1],
                "initial_objective": float(initial_objectives[index].item()),
                "final_objective": float(final_objectives[index].item()),
                "updates": config.optimizer_updates,
            }
            for index in indices
        ]
        restart_results.sort(key=lambda item: (item["final_objective"], item["restart"]))
        output[width] = (
            query_selected,
            key_selected,
            {
                "selected_restart": candidates[selected_index][1],
                "selected_initial_objective": float(
                    initial_objectives[selected_index].item()
                ),
                "selected_final_objective": float(
                    final_objectives[selected_index].item()
                ),
                "restart_results": restart_results,
                "orthogonality_maximum": _orthogonality_maximum(
                    query_selected, key_selected
                ),
                "execution": "six_candidate_batched_gpu",
            },
        )
    return output


def gauge_transform(
    representation: a7.H2Representation,
    query_gauges: Mapping[str, torch.Tensor],
    key_gauges: Mapping[str, torch.Tensor],
    block_width: int | None,
) -> a7.H2Representation:
    """Apply an exact gauge and optionally the fixed structural projection."""

    query_bases = {
        node: basis @ query_gauges[node]
        for node, basis in representation.query_bases.items()
    }
    key_bases = {
        node: basis @ key_gauges[node]
        for node, basis in representation.key_bases.items()
    }

    def transform(
        matrix: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        output = left.T @ matrix @ right
        if block_width is None:
            return output
        mask = block_mask(
            output.shape[0],
            output.shape[1],
            block_width,
            device=output.device,
        )
        return output * mask.to(output.dtype)

    query_transfers = {
        (child, parent): transform(
            matrix, query_gauges[child], query_gauges[parent]
        )
        for (child, parent), matrix in representation.query_transfers.items()
    }
    key_transfers = {
        (child, parent): transform(
            matrix, key_gauges[child], key_gauges[parent]
        )
        for (child, parent), matrix in representation.key_transfers.items()
    }
    couplings = {
        (query, key): transform(
            matrix, query_gauges[query], key_gauges[key]
        )
        for (query, key), matrix in representation.couplings.items()
    }
    return replace(
        representation,
        query_bases=query_bases,
        key_bases=key_bases,
        query_transfers=query_transfers,
        key_transfers=key_transfers,
        couplings=couplings,
    )


def _structural_count(matrix: torch.Tensor, block_width: int | None) -> int:
    if block_width is None:
        return matrix.numel()
    return int(
        block_mask(
            matrix.shape[0],
            matrix.shape[1],
            block_width,
            device=matrix.device,
        ).sum().item()
    )


def structured_accounting(
    representation: a7.H2Representation,
    channels: int,
    block_width: int | None,
) -> dict[str, Any]:
    clusters = representation.clusters
    leaves = [node for node in clusters.values() if node.id not in representation.children]
    near_storage = 0
    near_nonzero = 0
    coupling_storage = 0
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
            coupling_storage += _structural_count(
                representation.couplings[(block.query, block.key)], block_width
            )
    leaf_storage = sum(
        leaf.size
        * (
            representation.query_bases[leaf.id].shape[1]
            + representation.key_bases[leaf.id].shape[1]
        )
        for leaf in leaves
    )
    transfer_storage = sum(
        _structural_count(value, block_width)
        for value in representation.query_transfers.values()
    ) + sum(
        _structural_count(value, block_width)
        for value in representation.key_transfers.values()
    )
    total_storage = near_storage + leaf_storage + transfer_storage + coupling_storage
    dense_storage = representation.length * (representation.length + 1) // 2
    leaf_operations = channels * leaf_storage
    transfer_operations = channels * transfer_storage
    coupling_operations = channels * coupling_storage
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
        "block_width": block_width,
        "storage": {
            "near": near_storage,
            "leaf_bases": leaf_storage,
            "transfers": transfer_storage,
            "couplings": coupling_storage,
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


def _effective_bases(
    representation: a7.H2Representation,
    family: str,
) -> dict[str, torch.Tensor]:
    """Expand leaf bases through the stored (possibly pruned) transfers."""

    if family == "query":
        source = representation.query_bases
        transfers = representation.query_transfers
    elif family == "key":
        source = representation.key_bases
        transfers = representation.key_transfers
    else:
        raise ValueError(f"unknown basis family: {family}")
    effective: dict[str, torch.Tensor] = {}
    ordered = sorted(
        representation.clusters.values(),
        key=lambda node: (-node.depth, node.start),
    )
    for node in ordered:
        if node.id not in representation.children:
            effective[node.id] = source[node.id]
            continue
        pieces = [
            effective[child] @ transfers[(child, node.id)]
            for child in representation.children[node.id]
        ]
        effective[node.id] = torch.cat(pieces, dim=0)
    return effective


def assemble_effective_dense(
    representation: a7.H2Representation,
) -> torch.Tensor:
    """Explicitly assemble the operator realized by its transfer paths."""

    query_bases = _effective_bases(representation, "query")
    key_bases = _effective_bases(representation, "key")
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
                query_bases[block.query]
                @ representation.couplings[(block.query, block.key)]
                @ key_bases[block.key].T
            )
    return output


def _relative_frobenius(error: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        (
            torch.linalg.norm(error)
            / torch.clamp(torch.linalg.norm(reference), min=1e-12)
        ).item()
    )


def evaluate_representation(
    representation: a7.H2Representation,
    kernel: np.ndarray,
    attention: np.ndarray,
    value: np.ndarray,
    denominator: np.ndarray,
    config: GaugeChannelizationConfig,
    block_width: int | None,
) -> dict[str, Any]:
    kernel_tilde = assemble_effective_dense(representation)
    device = kernel_tilde.device
    kernel_tensor = torch.as_tensor(kernel, dtype=torch.float64, device=device)
    attention_tensor = torch.as_tensor(attention, dtype=torch.float64, device=device)
    denominator_tensor = torch.as_tensor(denominator, dtype=torch.float64, device=device)
    value_tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
    probes = torch.as_tensor(
        a7._fixed_probes(len(kernel), config.probe_columns, config.probe_seed),
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
    contracted = a7.h2_matvec(representation, contraction_input)
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
            torch.max(
                torch.sum(torch.abs(attention_tensor - attention_tilde), dim=1)
            ).item()
        )
        probe_error = _relative_frobenius(
            (attention_tilde - attention_tensor) @ probes,
            attention_tensor @ probes,
        )
        output = attention_tensor @ value_tensor
        output_error = (attention_tilde - attention_tensor) @ value_tensor
        value_error = _relative_frobenius(output_error, output)
        global_rms = torch.sqrt(torch.mean(torch.sum(output * output, dim=1)))
        token_tail = float(
            torch.quantile(
                torch.linalg.norm(output_error, dim=1)
                / torch.clamp(global_rms, min=1e-12),
                0.99,
            ).item()
        )
        negativity = float(
            torch.max(
                torch.sum(torch.clamp(-attention_tilde, min=0.0), dim=1)
            ).item()
        )
    else:
        attention_row = None
        probe_error = None
        value_error = None
        token_tail = None
        negativity = None
    future_leakage = float(
        torch.max(torch.abs(torch.triu(kernel_tilde, diagonal=1))).item()
    )
    finite_values: Iterable[float | None] = (
        contraction_error,
        future_leakage,
        delta,
        denominator_error,
        attention_row,
        probe_error,
        value_error,
        token_tail,
        negativity,
    )
    finite = all(item is None or math.isfinite(item) for item in finite_values)
    validity = {
        "explicit_vs_contraction_relative_error": contraction_error,
        "contraction_pass": contraction_error <= config.exact_equivalence_maximum,
        "future_token_leakage": future_leakage,
        "future_leakage_pass": future_leakage < 1e-15,
        "finite": finite,
    }
    validity["pass"] = all(
        validity[key]
        for key in ("contraction_pass", "future_leakage_pass", "finite")
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
    return {
        "validity": validity,
        "metrics": {
            "kernel_row_relative_maximum": delta,
            "denominator_relative_maximum": denominator_error,
            "positive_approximate_denominator": positive_denominator,
            "attention_row_l1_maximum": attention_row,
            "probe_relative_frobenius": probe_error,
            "value_output_relative_frobenius": value_error,
            "token_output_p99_global_rms_normalized": token_tail,
            "negative_attention_mass_maximum": negativity,
        },
        "gates": gates,
        "cell_pass": bool(validity["pass"] and all(gates.values())),
        "compression": structured_accounting(
            representation, value.shape[1] + 1, block_width
        ),
    }


def evaluate_cell_length(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    config: GaugeChannelizationConfig,
) -> dict[str, Any]:
    kernel, denominator, attention = a7.stabilized_kernel(query, key)
    base, _, construction = a7.build_h2_representations(
        kernel, attention, config.a7_config()
    )
    expected_partition = a7.PARTITION_COUNTS[len(kernel)]
    expected_fingerprint = a7.PARTITION_FINGERPRINTS[len(kernel)]
    construction_validity = {
        "partition_counts_pass": (
            construction["partition"]["counts"] == expected_partition
        ),
        "partition_fingerprint_pass": (
            construction["partition"]["sha256"] == expected_fingerprint
        ),
        "orthogonality_maximum": max(construction["orthogonality"].values()),
        "nestedness_maximum": max(construction["nestedness"].values()),
    }
    construction_validity["orthogonality_pass"] = (
        construction_validity["orthogonality_maximum"] <= 1e-10
    )
    construction_validity["nestedness_pass"] = (
        construction_validity["nestedness_maximum"] <= 1e-10
    )
    construction_validity["pass"] = all(
        construction_validity[key]
        for key in (
            "partition_counts_pass",
            "partition_fingerprint_pass",
            "orthogonality_pass",
            "nestedness_pass",
        )
    )
    baseline_dense = assemble_effective_dense(base)
    identity_query, identity_key = _identity_gauges(base)
    baseline_accounting = structured_accounting(
        base, value.shape[1] + 1, None
    )
    arms: dict[str, Any] = {}
    optimization: dict[str, Any] = {}
    exact_controls: dict[str, Any] = {}
    optimized_gauges = optimize_all_gauges(base, config)
    for width in config.block_widths:
        label = "diagonal" if width == 1 else f"block{width}"
        identity_projected = gauge_transform(
            base, identity_query, identity_key, width
        )
        arms[f"identity_{label}"] = evaluate_representation(
            identity_projected,
            kernel,
            attention,
            value,
            denominator,
            config,
            width,
        )
        query_gauges, key_gauges, optimizer_record = optimized_gauges[width]
        exact = gauge_transform(base, query_gauges, key_gauges, None)
        exact_dense = assemble_effective_dense(exact)
        exact_accounting = structured_accounting(
            exact, value.shape[1] + 1, None
        )
        exact_evaluation = evaluate_representation(
            exact,
            kernel,
            attention,
            value,
            denominator,
            config,
            None,
        )
        exact_equivalence = _relative_frobenius(
            exact_dense - baseline_dense, baseline_dense
        )
        exact_controls[label] = {
            "assembly_relative_error": exact_equivalence,
            "assembly_pass": exact_equivalence <= config.exact_equivalence_maximum,
            "storage_unchanged": (
                exact_accounting["storage"] == baseline_accounting["storage"]
            ),
            "operations_unchanged": (
                exact_accounting["multiply_adds"]
                == baseline_accounting["multiply_adds"]
            ),
            "contraction_relative_error": exact_evaluation["validity"][
                "explicit_vs_contraction_relative_error"
            ],
            "contraction_pass": exact_evaluation["validity"]["contraction_pass"],
            "future_leakage_pass": exact_evaluation["validity"][
                "future_leakage_pass"
            ],
        }
        exact_controls[label]["pass"] = all(
            (
                exact_controls[label]["assembly_pass"],
                exact_controls[label]["storage_unchanged"],
                exact_controls[label]["operations_unchanged"],
                exact_controls[label]["contraction_pass"],
                exact_controls[label]["future_leakage_pass"],
                optimizer_record["orthogonality_maximum"]
                <= config.gauge_orthogonality_maximum,
            )
        )
        projected = gauge_transform(base, query_gauges, key_gauges, width)
        arms[f"optimized_{label}"] = evaluate_representation(
            projected,
            kernel,
            attention,
            value,
            denominator,
            config,
            width,
        )
        identity_objective = float(
            offblock_objective(base, identity_query, identity_key, width).item()
        )
        optimizer_record["identity_objective"] = identity_objective
        optimizer_record["relative_objective_reduction"] = float(
            (identity_objective - optimizer_record["selected_final_objective"])
            / max(identity_objective, 1e-30)
        )
        optimization[label] = optimizer_record
    return {
        "construction": construction,
        "construction_validity": construction_validity,
        "baseline_accounting": baseline_accounting,
        "exact_controls": exact_controls,
        "optimization": optimization,
        "arms": arms,
    }


def analyze_seed(
    config: GaugeChannelizationConfig,
    seed: int,
    model: Any,
    tokens: np.ndarray,
    parent_run: Mapping[str, Any],
    output: Path,
    implementation_sha256: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    device = next(model.parameters()).device
    segment, validation_start = a7._validation_segment(
        tokens, config.validation_tokens, 256, seed
    )
    if validation_start != int(parent_run["provenance"]["validation_start"]):
        raise RuntimeError("validation-prefix replay differs from A7")
    input_ids = torch.from_numpy(segment.copy()).unsqueeze(0).to(device)
    selected = a7.extract_selected_qkv(model, input_ids, config.layers, config.heads)
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
            print(
                json.dumps(
                    {
                        "event": "cell_completed",
                        "seed": seed,
                        "layer": layer,
                        "head": head,
                        "cells_completed": len(cells),
                        "cells_total": len(config.layers) * len(config.heads),
                        "elapsed_seconds": time.perf_counter() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "parent_hypothesis_id": PARENT_HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-h2-gauge-channelization-seed{seed}",
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_scientific_evidence"
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
            "checkpoint_sha256": a7.CHECKPOINT_SHA256,
            "token_cache_sha256": a7.TOKEN_STREAM_SHA256,
            "parent_aggregate_sha256": PARENT_AGGREGATE_SHA256,
            "parent_implementation_sha256": PARENT_IMPLEMENTATION_SHA256,
            "a6_aggregate_sha256": A6_AGGREGATE_SHA256,
            "preregistration_sha256": PREREGISTRATION_SHA256,
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


def _arm_summary(
    cells: Sequence[Mapping[str, Any]],
    arm: str,
    config: GaugeChannelizationConfig,
) -> dict[str, Any]:
    records = [cell["lengths"][str(config.primary_length)]["arms"][arm] for cell in cells]
    passes = [bool(record["cell_pass"]) for record in records]
    layer_pass_fractions: dict[str, float] = {}
    for layer in config.layers:
        selected = [
            record
            for cell, record in zip(cells, records)
            if int(cell["layer"]) == layer
        ]
        layer_pass_fractions[str(layer)] = float(
            np.mean([record["cell_pass"] for record in selected])
        )
    storage = [float(record["compression"]["storage"]["ratio"]) for record in records]
    operations = [
        float(record["compression"]["multiply_adds"]["ratio"])
        for record in records
    ]
    representation_pass = bool(
        float(np.mean(passes)) >= config.cell_pass_fraction_minimum
        and min(layer_pass_fractions.values()) >= config.layer_pass_fraction_minimum
    )
    compression_pass = bool(
        float(np.median(storage)) <= config.storage_ratio_median_maximum
        and _percentile(storage, 90) <= config.storage_ratio_p90_maximum
        and float(np.median(operations)) <= config.operation_ratio_median_maximum
        and _percentile(operations, 90) <= config.operation_ratio_p90_maximum
    )
    return {
        "cell_count": len(records),
        "passing_cells": sum(passes),
        "pass_fraction": float(np.mean(passes)),
        "layer_pass_fractions": layer_pass_fractions,
        "minimum_layer_pass_fraction": min(layer_pass_fractions.values()),
        "representation_pass": representation_pass,
        "storage_ratio_median": float(np.median(storage)),
        "storage_ratio_p90": _percentile(storage, 90),
        "operation_ratio_median": float(np.median(operations)),
        "operation_ratio_p90": _percentile(operations, 90),
        "compression_pass": compression_pass,
        "kernel_row_relative_median": float(
            np.median(
                [record["metrics"]["kernel_row_relative_maximum"] for record in records]
            )
        ),
        "kernel_row_relative_p90": _percentile(
            [record["metrics"]["kernel_row_relative_maximum"] for record in records],
            90,
        ),
    }


def aggregate(
    runs: Sequence[Mapping[str, Any]],
    config: GaugeChannelizationConfig,
) -> dict[str, Any]:
    cells = [cell for run in runs for cell in run["cells"]]
    labels = ["diagonal", "block2", "block4"]
    exact_pass = all(
        cell["lengths"][str(length)]["exact_controls"][label]["pass"]
        for cell in cells
        for length in config.lengths
        for label in labels
    )
    contraction_pass = all(
        arm["validity"]["pass"]
        for cell in cells
        for length in config.lengths
        for arm in cell["lengths"][str(length)]["arms"].values()
    )
    construction_pass = all(
        cell["lengths"][str(length)]["construction_validity"]["pass"]
        for cell in cells
        for length in config.lengths
    )
    parent_valid = (
        _sha256(config.parent_aggregate) == PARENT_AGGREGATE_SHA256
        and _sha256(config.preregistration) == PREREGISTRATION_SHA256
    )
    arm_summaries = {
        f"identity_{label}": _arm_summary(cells, f"identity_{label}", config)
        for label in labels
    }
    arm_summaries.update(
        {
            f"optimized_{label}": _arm_summary(
                cells, f"optimized_{label}", config
            )
            for label in labels
        }
    )
    optimized = [f"optimized_{label}" for label in labels]
    fully_passing = [
        arm
        for arm in optimized
        if arm_summaries[arm]["representation_pass"]
        and arm_summaries[arm]["compression_pass"]
    ]
    representation_passing = [
        arm for arm in optimized if arm_summaries[arm]["representation_pass"]
    ]
    compression_passing = [
        arm for arm in optimized if arm_summaries[arm]["compression_pass"]
    ]
    valid = bool(parent_valid and construction_pass and exact_pass and contraction_pass)
    if not valid:
        classification = "invalid_parent_or_gauge_contract"
    elif fully_passing:
        classification = "gauge_channelization_compression_pass"
    elif representation_passing:
        classification = "gauge_channelization_representation_only"
    elif compression_passing:
        classification = "gauge_channelization_sparsity_accuracy_tradeoff"
    else:
        classification = "gauge_channelization_no_structural_gain"

    width_order = {"optimized_diagonal": 1, "optimized_block2": 2, "optimized_block4": 4}
    if fully_passing:
        preferred = min(fully_passing, key=lambda arm: width_order[arm])
    elif representation_passing:
        preferred = min(
            representation_passing,
            key=lambda arm: (
                arm_summaries[arm]["storage_ratio_median"],
                arm_summaries[arm]["kernel_row_relative_median"],
            ),
        )
    else:
        preferred = min(
            optimized,
            key=lambda arm: (
                arm_summaries[arm]["storage_ratio_median"],
                arm_summaries[arm]["kernel_row_relative_median"],
            ),
        )

    diagnostic_lengths: dict[str, Any] = {}
    for length in config.lengths:
        diagnostic_lengths[str(length)] = {}
        for label in labels:
            objectives = [
                cell["lengths"][str(length)]["optimization"][label]
                for cell in cells
            ]
            diagnostic_lengths[str(length)][label] = {
                "final_objective_median": float(
                    np.median([item["selected_final_objective"] for item in objectives])
                ),
                "relative_objective_reduction_median": float(
                    np.median(
                        [item["relative_objective_reduction"] for item in objectives]
                    )
                ),
                "spectral_restart_fraction": float(
                    np.mean(
                        [
                            item["selected_restart"] == "spectral_covariance"
                            for item in objectives
                        ]
                    )
                ),
            }
    return {
        "classification": classification,
        "validity": {
            "parent_and_preregistration_hashes": parent_valid,
            "a7_construction_replay": construction_pass,
            "exact_unpruned_gauge_controls": exact_pass,
            "all_contractions_and_causality": contraction_pass,
            "pass": valid,
        },
        "primary_length": config.primary_length,
        "arms": arm_summaries,
        "fully_passing_optimized_arms": fully_passing,
        "preferred_arm": preferred,
        "diagnostic_lengths": diagnostic_lengths,
        "counts": {
            "evaluation_seeds": len(runs),
            "cells": len(cells),
            "cell_lengths": len(cells) * len(config.lengths),
            "evaluated_arms_per_cell_length": 6,
        },
    }


def _load_parent(config: GaugeChannelizationConfig) -> dict[int, Any]:
    if _sha256(config.parent_aggregate) != PARENT_AGGREGATE_SHA256:
        raise RuntimeError("A7 parent aggregate hash mismatch")
    parent = json.loads(Path(config.parent_aggregate).read_text(encoding="utf-8"))
    if parent.get("status") != "completed" or parent.get("summary", {}).get("failed") != 0:
        raise RuntimeError("A7 parent campaign is incomplete")
    output: dict[int, Any] = {}
    for item in parent["results"]:
        path = Path(item["result"])
        run = json.loads(path.read_text(encoding="utf-8"))
        output[int(item["seed"])] = run
    return output


def _validate_frozen_inputs(config: GaugeChannelizationConfig) -> None:
    checks = {
        config.checkpoint: a7.CHECKPOINT_SHA256,
        config.token_cache: a7.TOKEN_STREAM_SHA256,
        config.parent_aggregate: PARENT_AGGREGATE_SHA256,
        config.preregistration: PREREGISTRATION_SHA256,
    }
    for path, expected in checks.items():
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen SHA-256 mismatch for {path}: {actual}")
    if _sha256(a7.__file__) != PARENT_IMPLEMENTATION_SHA256:
        raise RuntimeError("A7 parent implementation hash mismatch")


def run_campaign(
    config: GaugeChannelizationConfig,
    output: Path,
) -> dict[str, Any]:
    _validate_frozen_inputs(config)
    implementation_sha256 = _implementation_digest()
    parent = _load_parent(config)
    missing = set(config.evaluation_seeds) - set(parent)
    if missing:
        raise RuntimeError(f"A7 parent is missing seeds: {sorted(missing)}")
    output.mkdir(parents=True, exist_ok=True)
    tokens = np.load(config.token_cache, mmap_mode="r")
    model = a7._load_model(Path(config.checkpoint), torch.device(config.device))
    runs = []
    reused = 0
    failures: list[dict[str, Any]] = []
    for seed in config.evaluation_seeds:
        path = output / "runs" / f"seed_{seed}" / "result.json"
        expected_fingerprint = _fingerprint(config, seed, implementation_sha256)
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "completed"
                and existing.get("scientific_fingerprint") == expected_fingerprint
            ):
                runs.append(existing)
                reused += 1
                continue
        try:
            print(
                json.dumps(
                    {
                        "event": "seed_started",
                        "seed": seed,
                        "device": config.device,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            runs.append(
                analyze_seed(
                    config,
                    seed,
                    model,
                    tokens,
                    parent[seed],
                    output,
                    implementation_sha256,
                )
            )
            print(
                json.dumps(
                    {
                        "event": "seed_completed",
                        "seed": seed,
                        "analysis_seconds": runs[-1]["analysis_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as error:  # Persist campaign-level failures.
            failures.append({"seed": seed, "error": repr(error)})
            _write_json(
                output / "runs" / f"seed_{seed}" / "failure.json",
                {
                    "status": "failed",
                    "seed": seed,
                    "error": repr(error),
                    "completed_at": _utc_now(),
                },
            )
            break
    status = "completed" if len(runs) == len(config.evaluation_seeds) and not failures else "failed"
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "parent_hypothesis_id": PARENT_HYPOTHESIS_ID,
        "status": status,
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation_sha256,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "provenance": {
            "checkpoint_sha256": a7.CHECKPOINT_SHA256,
            "token_cache_sha256": a7.TOKEN_STREAM_SHA256,
            "parent_aggregate_sha256": PARENT_AGGREGATE_SHA256,
            "parent_implementation_sha256": PARENT_IMPLEMENTATION_SHA256,
            "a6_aggregate_sha256": A6_AGGREGATE_SHA256,
        },
        "summary": {
            "requested": len(config.evaluation_seeds),
            "completed": len(runs),
            "failed": len(failures),
            "reused": reused,
        },
        "failures": failures,
        "results": [
            {
                "seed": run["seed"],
                "experiment_id": run["experiment_id"],
                "result": str(output / "runs" / f"seed_{run['seed']}" / "result.json"),
                "analysis_seconds": run["analysis_seconds"],
            }
            for run in runs
        ],
        "aggregates": aggregate(runs, config) if status == "completed" else None,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": config.device,
            "device_name": (
                torch.cuda.get_device_name(torch.device(config.device))
                if torch.cuda.is_available() and config.device.startswith("cuda")
                else "cpu"
            ),
        },
        "method_boundaries": {
            "operator_specific_internal_gauges": True,
            "factor_only_optimizer": True,
            "rank_changes": False,
            "token_space_descrambler": False,
            "online_construction_cost_counted": False,
            "realized_sparse_kernel_timing": False,
            "cross_checkpoint_shared_gauge_evidence": False,
        },
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=GaugeChannelizationConfig.checkpoint)
    parser.add_argument("--token-cache", default=GaugeChannelizationConfig.token_cache)
    parser.add_argument("--parent-aggregate", default=GaugeChannelizationConfig.parent_aggregate)
    parser.add_argument("--preregistration", default=GaugeChannelizationConfig.preregistration)
    parser.add_argument("--seeds", default="101,211,307,401,503")
    parser.add_argument("--lengths", default="64,128,256")
    parser.add_argument("--layers", default=",".join(str(item) for item in range(8)))
    parser.add_argument("--heads", default="0,3,7")
    parser.add_argument("--optimizer-updates", type=int, default=96)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_h2_gauge_channelization/20260830_registered"
        ),
    )
    parser.add_argument("--shakedown", action="store_true")
    args = parser.parse_args(argv)
    config = GaugeChannelizationConfig(
        checkpoint=args.checkpoint,
        token_cache=args.token_cache,
        parent_aggregate=args.parent_aggregate,
        preregistration=args.preregistration,
        evaluation_seeds=_ints(args.seeds),
        lengths=_ints(args.lengths),
        primary_length=max(_ints(args.lengths)),
        layers=_ints(args.layers),
        heads=_ints(args.heads),
        optimizer_updates=args.optimizer_updates,
        device=args.device,
        shakedown=args.shakedown,
    )
    campaign = run_campaign(config, args.output)
    print(
        json.dumps(
            {
                "status": campaign["status"],
                "summary": campaign["summary"],
                "aggregates": campaign["aggregates"],
            },
            indent=2,
        )
    )
    return 0 if campaign["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
