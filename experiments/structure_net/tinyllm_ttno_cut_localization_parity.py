#!/usr/bin/env python3
"""Execute preregistered A5 TTNO cut-localization and parity controls."""

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
        balanced_tree_nodes,
        causal_attention,
        extract_selected_qk,
        numerical_rank_from_singular_values,
        paired_operator_tensor,
        paired_tensor_to_operator,
    )
    from experiments.structure_net import tinyllm_dynamic_ttno_rank as a4
except ModuleNotFoundError:  # Direct execution sets the script directory on sys.path.
    from tinyllm_dynamic_ttno_rank import (  # type: ignore[no-redef]
        _load_model,
        _sha256,
        _validation_segment,
        balanced_tree_nodes,
        causal_attention,
        extract_selected_qk,
        numerical_rank_from_singular_values,
        paired_operator_tensor,
        paired_tensor_to_operator,
    )
    import tinyllm_dynamic_ttno_rank as a4  # type: ignore[no-redef]


SCHEMA_VERSION = "nal.tinyllm-ttno-cut-localization-parity.v1"
HYPOTHESIS_ID = "tinyllm-ttno-cut-localization-parity-v1"
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
    "9d059361d84d07fa27da35a0833253a04bc3757534158b3d99b0ae2380fc6d24"
)


@dataclass(frozen=True)
class CutLocalizationConfig:
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
    layers: tuple[int, ...] = tuple(range(8))
    heads: tuple[int, ...] = (0, 3, 7)
    epsilons: tuple[float, ...] = (1e-2, 1e-3)
    primary_epsilon: float = 1e-2
    parity_artifact_threshold: float = 0.75
    topology_reduction_threshold: float = 0.25
    topology_sensitive_fraction: float = 0.80
    intrinsic_artifact_ceiling: float = 0.25
    intrinsic_topology_fraction_ceiling: float = 0.20
    device: str = "cuda:0"
    shakedown: bool = False

    def __post_init__(self) -> None:
        if not self.evaluation_seeds:
            raise ValueError("at least one evaluation seed is required")
        if not self.layers or not self.heads:
            raise ValueError("layers and heads must be non-empty")
        if self.primary_epsilon not in self.epsilons:
            raise ValueError("primary epsilon must be one of epsilons")
        if self.validation_tokens <= 256:
            raise ValueError("validation suffix must exceed 256 tokens")


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
    config: CutLocalizationConfig,
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


def gray_code_order(length: int) -> np.ndarray:
    """Binary-reflected Gray-code token order."""

    if length < 2 or length & (length - 1):
        raise ValueError("length must be a power of two")
    indices = np.arange(length, dtype=np.int64)
    return indices ^ (indices >> 1)


def zero_pad_operator(operator: np.ndarray) -> np.ndarray:
    """Embed a square operator in the upper-left block of a doubled zero matrix."""

    value = np.asarray(operator, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("operator must be square")
    output = np.zeros((2 * len(value), 2 * len(value)), dtype=np.float64)
    output[: len(value), : len(value)] = value
    return output


def duplicate_operator(operator: np.ndarray) -> np.ndarray:
    """Return direct sum I2 kron operator."""

    value = np.asarray(operator, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("operator must be square")
    return np.kron(np.eye(2, dtype=np.float64), value)


def mode_order(mode_count: int, topology: str) -> tuple[int, ...]:
    """Return the frozen paired-bit mode ordering for a topology arm."""

    natural = tuple(range(mode_count))
    if topology == "msb_balanced":
        return natural
    if topology == "lsb_balanced":
        return natural[::-1]
    if topology == "odd_even_modes":
        return natural[1::2] + natural[0::2]
    raise ValueError(f"unknown topology {topology!r}")


def ordered_qttno_rank_profile(
    operator: np.ndarray,
    epsilons: Sequence[float],
    topology: str,
    computation_device: str = "cpu",
) -> dict[str, Any]:
    """A4 QTTNO profile after one preregistered paired-mode permutation."""

    tensor = paired_operator_tensor(operator)
    order = mode_order(tensor.ndim, topology)
    ordered = tensor.transpose(order)
    edges: list[dict[str, Any]] = []
    for node in balanced_tree_nodes(tuple(range(ordered.ndim))):
        complement = tuple(index for index in range(ordered.ndim) if index not in node)
        rows = 4 ** len(node)
        matrix = ordered.transpose(node + complement).reshape(rows, -1)
        if torch.device(computation_device).type == "cuda":
            torch_matrix = torch.as_tensor(
                np.ascontiguousarray(matrix),
                dtype=torch.float64,
                device=computation_device,
            )
            gram = (
                torch_matrix @ torch_matrix.T
                if matrix.shape[0] <= matrix.shape[1]
                else torch_matrix.T @ torch_matrix
            )
            eigenvalues = torch.linalg.eigvalsh(gram)
            singular_values = (
                torch.sqrt(torch.clamp(eigenvalues, min=0.0))
                .flip(0)
                .cpu()
                .numpy()
            )
        else:
            singular_values = a4._singular_values(matrix)
        ranks = {
            f"{epsilon:.12g}": numerical_rank_from_singular_values(
                singular_values, epsilon
            )
            for epsilon in epsilons
        }
        edges.append(
            {
                "tree_positions": list(node),
                "modes": [order[position] for position in node],
                "shape": list(matrix.shape),
                "ranks": ranks,
            }
        )
    maxima = {
        f"{epsilon:.12g}": max(
            edge["ranks"][f"{epsilon:.12g}"] for edge in edges
        )
        for epsilon in epsilons
    }
    return {
        "topology": topology,
        "mode_order": list(order),
        "max_ranks": maxima,
        "edges": edges,
    }


def _a4_compatible_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Drop A5 topology annotations to reproduce the exact A4 record shape."""

    return {
        "max_ranks": dict(profile["max_ranks"]),
        "edges": [
            {
                "modes": edge["modes"],
                "shape": edge["shape"],
                "ranks": edge["ranks"],
            }
            for edge in profile["edges"]
        ],
    }


def _critical_edge(profile: Mapping[str, Any], epsilon_key: str) -> dict[str, Any]:
    edge = max(
        profile["edges"],
        key=lambda item: (int(item["ranks"][epsilon_key]), -len(item["modes"])),
    )
    return {
        "modes": edge["modes"],
        "shape": edge["shape"],
        "rank": int(edge["ranks"][epsilon_key]),
    }


def _new_root_split_rank(profile: Mapping[str, Any], epsilon_key: str) -> int:
    mode_count = len(profile["mode_order"])
    if mode_count % 2:
        raise ValueError("new root split is defined for an even mode count")
    half = mode_count // 2
    values = [
        int(edge["ranks"][epsilon_key])
        for edge in profile["edges"]
        if len(edge["tree_positions"]) == half
    ]
    if not values:
        raise RuntimeError("root-child edges are missing")
    return max(values)


def _roundtrip_exact(operator: np.ndarray) -> bool:
    return bool(
        np.array_equal(
            paired_tensor_to_operator(paired_operator_tensor(operator)),
            np.asarray(operator, dtype=np.float64),
        )
    )


def _parent_cells(parent_run: Mapping[str, Any]) -> dict[tuple[int, int], Any]:
    return {
        (int(cell["layer"]), int(cell["head"])): cell
        for cell in parent_run["natural_cells"]
    }


def analyze_cell(
    query: np.ndarray,
    key: np.ndarray,
    parent_cell: Mapping[str, Any],
    config: CutLocalizationConfig,
) -> dict[str, Any]:
    """Analyze all frozen A5 arms for one layer/head/prefix cell."""

    attention128 = causal_attention(query[:128], key[:128])
    attention256 = causal_attention(query[:256], key[:256])
    zero = zero_pad_operator(attention128)
    duplicate = duplicate_operator(attention128)
    gray = gray_code_order(256)
    gray_attention = attention256[np.ix_(gray, gray)]

    arms = {
        "natural_128_msb": ordered_qttno_rank_profile(
            attention128, config.epsilons, "msb_balanced", config.device
        ),
        "natural_256_msb": ordered_qttno_rank_profile(
            attention256, config.epsilons, "msb_balanced", config.device
        ),
        "natural_256_lsb": ordered_qttno_rank_profile(
            attention256, config.epsilons, "lsb_balanced", config.device
        ),
        "natural_256_odd_even": ordered_qttno_rank_profile(
            attention256, config.epsilons, "odd_even_modes", config.device
        ),
        "natural_256_gray": ordered_qttno_rank_profile(
            gray_attention, config.epsilons, "msb_balanced", config.device
        ),
        "zero_pad_128_to_256": ordered_qttno_rank_profile(
            zero, config.epsilons, "msb_balanced", config.device
        ),
        "duplicate_128_to_256": ordered_qttno_rank_profile(
            duplicate, config.epsilons, "msb_balanced", config.device
        ),
    }

    parent128 = parent_cell["lengths"]["128"]["orders"]["chronological"]["qttno"]
    parent256 = parent_cell["lengths"]["256"]["orders"]["chronological"]["qttno"]
    replay128 = _a4_compatible_profile(arms["natural_128_msb"])
    replay256 = _a4_compatible_profile(arms["natural_256_msb"])
    parent_reproduced = replay128 == parent128 and replay256 == parent256

    roundtrips = {
        "natural_128": _roundtrip_exact(attention128),
        "natural_256": _roundtrip_exact(attention256),
        "gray_token_order": _roundtrip_exact(gray_attention),
        "zero_pad_128_to_256": _roundtrip_exact(zero),
        "duplicate_128_to_256": _roundtrip_exact(duplicate),
    }
    epsilon_key = f"{config.primary_epsilon:.12g}"
    rank = {
        name: int(profile["max_ranks"][epsilon_key])
        for name, profile in arms.items()
    }
    r128 = rank["natural_128_msb"]
    r256 = rank["natural_256_msb"]
    rzero = rank["zero_pad_128_to_256"]
    rduplicate = rank["duplicate_128_to_256"]
    alternatives = {
        "lsb_balanced": rank["natural_256_lsb"],
        "odd_even_modes": rank["natural_256_odd_even"],
        "gray_token_order": rank["natural_256_gray"],
    }
    winning_topology = min(alternatives, key=lambda item: (alternatives[item], item))
    ralt = alternatives[winning_topology]
    cliff = max(r256 - r128, 1)
    zero_fraction = float(np.clip((rzero - r128) / cliff, 0.0, 1.0))
    duplicate_fraction = float(
        np.clip((rduplicate - r128) / cliff, 0.0, 1.0)
    )
    topology_reduction = float((r256 - ralt) / r256) if r256 else 0.0
    root_rank = _new_root_split_rank(arms["natural_256_msb"], epsilon_key)
    root_fraction = float(np.clip((root_rank - r128) / cliff, 0.0, 1.0))

    return {
        "validity": {
            "parent_ranks_reproduced": parent_reproduced,
            "tensor_roundtrips": roundtrips,
            "pass": parent_reproduced and all(roundtrips.values()),
        },
        "primary": {
            "r128": r128,
            "r256": r256,
            "rzero": rzero,
            "rduplicate": rduplicate,
            "ralt": ralt,
            "winning_alternative_topology": winning_topology,
            "cliff": cliff,
            "zero_artifact_fraction": zero_fraction,
            "duplicate_artifact_fraction": duplicate_fraction,
            "maximum_artifact_fraction": max(zero_fraction, duplicate_fraction),
            "topology_reduction": topology_reduction,
            "new_root_split_rank": root_rank,
            "new_root_cliff_fraction": root_fraction,
            "critical_edge_128": _critical_edge(
                arms["natural_128_msb"], epsilon_key
            ),
            "critical_edge_256": _critical_edge(
                arms["natural_256_msb"], epsilon_key
            ),
        },
        "arms": arms,
    }


def analyze_seed(
    config: CutLocalizationConfig,
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
    for layer in config.layers:
        for head in config.heads:
            cell = analyze_cell(
                *selected[(layer, head)], parents[(layer, head)], config
            )
            cells.append({"layer": layer, "head": head, **cell})

    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "parent_hypothesis_id": PARENT_HYPOTHESIS_ID,
        "experiment_id": f"tinyllm-ttno-cut-localization-parity-seed{seed}",
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
        },
        "cells": cells,
        "analysis_seconds": time.perf_counter() - started,
    }
    _write_json(output / "runs" / f"seed_{seed}" / "result.json", result)
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def aggregate(
    runs: Sequence[Mapping[str, Any]], config: CutLocalizationConfig
) -> dict[str, Any]:
    cells = [cell for run in runs for cell in run["cells"]]
    valid = [bool(cell["validity"]["pass"]) for cell in cells]
    artifact = [float(cell["primary"]["maximum_artifact_fraction"]) for cell in cells]
    topology = [float(cell["primary"]["topology_reduction"]) for cell in cells]
    root = [float(cell["primary"]["new_root_cliff_fraction"]) for cell in cells]
    parity_gate = float(np.median(artifact)) >= config.parity_artifact_threshold
    topology_fraction = float(
        np.mean(np.asarray(topology) >= config.topology_reduction_threshold)
    )
    topology_gate = topology_fraction >= config.topology_sensitive_fraction
    intrinsic_gate = (
        float(np.median(artifact)) < config.intrinsic_artifact_ceiling
        and topology_fraction < config.intrinsic_topology_fraction_ceiling
    )
    validity_gate = bool(valid) and all(valid)
    if not validity_gate:
        classification = "invalid_parent_or_tensor_contract"
    elif parity_gate:
        classification = "new_cut_artifact_dominant"
    elif topology_gate:
        classification = "bit_topology_sensitive"
    elif intrinsic_gate:
        classification = "intrinsic_operator_rank_growth"
    else:
        classification = "mixed_cut_and_operator_effect"

    rank_keys = (
        "r128",
        "r256",
        "rzero",
        "rduplicate",
        "ralt",
        "cliff",
        "new_root_split_rank",
    )
    rank_summary = {
        key: {
            "median": float(np.median([cell["primary"][key] for cell in cells])),
            "p90": _percentile([cell["primary"][key] for cell in cells], 90),
            "range": [
                int(min(cell["primary"][key] for cell in cells)),
                int(max(cell["primary"][key] for cell in cells)),
            ],
        }
        for key in rank_keys
    }
    winner_counts = {
        name: sum(
            cell["primary"]["winning_alternative_topology"] == name
            for cell in cells
        )
        for name in ("lsb_balanced", "odd_even_modes", "gray_token_order")
    }
    return {
        "classification": classification,
        "validity": {
            "passing_cells": sum(valid),
            "required_cells": len(cells),
            "pass": validity_gate,
        },
        "parity_artifact": {
            "median_maximum_artifact_fraction": float(np.median(artifact)),
            "p90_maximum_artifact_fraction": _percentile(artifact, 90),
            "threshold": config.parity_artifact_threshold,
            "pass": parity_gate,
        },
        "topology_sensitivity": {
            "cell_fraction_at_least_25_percent_reduction": topology_fraction,
            "required_fraction": config.topology_sensitive_fraction,
            "median_reduction": float(np.median(topology)),
            "p90_reduction": _percentile(topology, 90),
            "winning_topology_counts": winner_counts,
            "pass": topology_gate,
        },
        "intrinsic_persistence": {
            "artifact_median_ceiling": config.intrinsic_artifact_ceiling,
            "topology_fraction_ceiling": config.intrinsic_topology_fraction_ceiling,
            "pass": intrinsic_gate,
        },
        "new_root_split": {
            "median_cliff_fraction": float(np.median(root)),
            "p90_cliff_fraction": _percentile(root, 90),
        },
        "rank_summary": rank_summary,
        "counts": {
            "evaluation_seeds": len(runs),
            "cells": len(cells),
            "profiles": len(cells) * 7,
        },
    }


def _load_parent(config: CutLocalizationConfig) -> tuple[dict[str, Any], dict[int, Any]]:
    path = Path(config.parent_aggregate)
    if _sha256(path) != PARENT_AGGREGATE_SHA256:
        raise RuntimeError("A4 aggregate hash differs from the preregistration")
    campaign = json.loads(path.read_text(encoding="utf-8"))
    if campaign.get("hypothesis_id") != PARENT_HYPOTHESIS_ID:
        raise RuntimeError("unexpected A4 hypothesis ID")
    if campaign.get("implementation_sha256") != PARENT_IMPLEMENTATION_SHA256:
        raise RuntimeError("unexpected A4 implementation hash")
    runs = {}
    for item in campaign["results"]:
        run = json.loads(Path(item["result"]).read_text(encoding="utf-8"))
        runs[int(run["seed"])] = run
    return campaign, runs


def run_campaign(config: CutLocalizationConfig, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(config.checkpoint)
    token_cache = Path(config.token_cache)
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint hash differs from the preregistration")
    if _sha256(token_cache) != TOKEN_STREAM_SHA256:
        raise RuntimeError("token stream hash differs from the preregistration")
    _, parent_runs = _load_parent(config)
    if any(seed not in parent_runs for seed in config.evaluation_seeds):
        raise RuntimeError("a requested A5 seed is absent from A4")

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
            "A5 localizes the A4 rank cliff and does not construct a TTNO or H2 operator.",
            "Alternative mode orders are fixed topology controls, not a tree search.",
            "Zero padding and duplication are representation controls, not natural attention distributions.",
            "The frozen checkpoint cannot supply a natural 512-token extension.",
        ],
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CutLocalizationConfig.checkpoint)
    parser.add_argument("--token-cache", default=CutLocalizationConfig.token_cache)
    parser.add_argument(
        "--parent-aggregate", default=CutLocalizationConfig.parent_aggregate
    )
    parser.add_argument("--evaluation-seeds", default="101,211,307,401,503")
    parser.add_argument("--layers", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--heads", default="0,3,7")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_ttno_cut_localization_parity/"
            "20260830_registered"
        ),
    )
    args = parser.parse_args(argv)
    seeds = _ints(args.evaluation_seeds)
    layers = _ints(args.layers)
    heads = _ints(args.heads)
    if args.shakedown:
        seeds = seeds[:1]
        layers = layers[:1]
        heads = heads[:1]
    config = CutLocalizationConfig(
        checkpoint=args.checkpoint,
        token_cache=args.token_cache,
        parent_aggregate=args.parent_aggregate,
        evaluation_seeds=seeds,
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
