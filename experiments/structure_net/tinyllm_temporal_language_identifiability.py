#!/usr/bin/env python3
"""L1 identifiability campaign for the temporal-phase language task.

Fine-tunes TinyLLM d8 systems (BabyLM-pretrained and scratch initializations)
on calibrated, uncalibrated, and UTC-oracle variants of the temporal report
task, then gates identifiability per the locked preregistration:

  G1 oracle competence, G2 uncalibrated control at chance, G3 calibration use.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import experiments.structure_net.tinyllm_temporal_language_task as temporal
from structure_net.components.models import TinyLLMModel
from structure_net.components.models.tinyllm_model import TinyLLMConfig


SCHEMA_VERSION = "nal.tinyllm-temporal-language-identifiability.v1"
HYPOTHESIS_ID = "tinyllm-temporal-language-identifiability-v1"
EVIDENCE_ROLE = "preregistered_language_identifiability_training_campaign"
TOKENIZER_SHA256 = (
    "ffb45dbe848de6ab2bdfc40c55e577a429e45791d6047c1fd0401b2b3311e0cf"
)
PRETRAIN_CHECKPOINT = (
    "data/experiments/tinyllm_babylm_pretrain/20260812_d8_seed7/"
    "checkpoint_step12000.pt"
)
PRETRAIN_CHECKPOINT_SHA256 = (
    "5a7491b4231c30feaaabda7babf0a831dd4d72fbb4e7f3e757114cadf098df09"
)
INITIALIZATIONS = ("babylm_pretrained", "scratch")
MODES = ("calibrated", "uncalibrated", "utc_oracle")
PRIMARY_SEEDS = (7, 17, 29, 41, 53)
EVAL_REGIMES = ("interpolation", "composition", "extrapolation")
TRAIN_SEED_BASE = 100_000
EVAL_SEEDS = {"interpolation": 200_001, "composition": 200_002, "extrapolation": 200_003}
GATES = {
    "g1_oracle_interpolation": 0.375,
    "g1_oracle_composition": 0.25,
    "g2_uncalibrated_ceiling": 0.125,
    "g3_calibrated_floor": 0.30,
    "g3_paired_margin": 0.15,
    "required_seed_passes": 4,
}


@dataclass(frozen=True)
class TemporalIdentifiabilityConfig:
    train_fibers: int = 4_096
    evaluation_samples: int = 1_024
    training_steps: int = 600
    batch_fibers: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    block_size: int = 256
    preset: str = "d8"
    seeds: Tuple[int, ...] = PRIMARY_SEEDS
    device: str = "cuda:1"
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if self.evaluation_samples % 2 or self.train_fibers < 1:
            raise ValueError("evaluation samples must pair fibers")
        if not self.allow_underpowered:
            if self.seeds != PRIMARY_SEEDS:
                raise ValueError("the primary fine-tune seeds are fixed")
            if (
                self.train_fibers != 4_096
                or self.evaluation_samples != 1_024
                or self.training_steps != 600
                or self.batch_fibers != 32
                or self.learning_rate != 3e-4
            ):
                raise ValueError("the primary training protocol is fixed")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _implementation_digest() -> str:
    material = {
        "runner": _sha256(Path(__file__)),
        "task": _sha256(Path(temporal.__file__)),
        "model": _sha256(Path(inspect.getfile(TinyLLMModel))),
        "schema": SCHEMA_VERSION,
        "gates": GATES,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _answer_logits(
    model: TinyLLMModel, input_ids: torch.Tensor, answer_ids: torch.Tensor
) -> torch.Tensor:
    logits, _ = model(input_ids, return_full_logits=True)
    return logits[:, -1, :].index_select(-1, answer_ids)


def _evaluate(
    model: TinyLLMModel,
    dataset: temporal.TemporalLanguageDataset,
    answer_ids: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    cross_entropy = 0.0
    total = len(dataset.input_ids)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            inputs = dataset.input_ids[start:stop].to(device)
            targets = dataset.target_posteriors[start:stop].to(device)
            logits = _answer_logits(model, inputs, answer_ids)
            log_probabilities = torch.log_softmax(logits, dim=-1)
            cross_entropy += float(-(targets * log_probabilities).sum(-1).sum())
            predicted = logits.argmax(-1)
            correct += int(
                (predicted == dataset.target_bins[start:stop].to(device)).sum()
            )
    model.train()
    return {
        "exact_bin_accuracy": correct / total,
        "mean_target_cross_entropy": cross_entropy / total,
    }


def _fingerprint(
    config: TemporalIdentifiabilityConfig,
    implementation: str,
    initialization: str,
    mode: str,
    seed: int,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "initialization": initialization,
        "mode": mode,
        "seed": seed,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "pretrain_checkpoint_sha256": PRETRAIN_CHECKPOINT_SHA256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def train_cell(
    config: TemporalIdentifiabilityConfig,
    tokenizer,
    task: temporal.TemporalLanguageTaskConfig,
    initialization: str,
    mode: str,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model_config = TinyLLMConfig.from_preset(
        config.preset,
        block_size=config.block_size,
        vocab_size=task.vocab_size,
        initialization_seed=seed,
    )
    model = TinyLLMModel(
        model_config, name=f"TemporalL1_{initialization}_{mode}_{seed}"
    )
    if initialization == "babylm_pretrained":
        checkpoint_path = Path(PRETRAIN_CHECKPOINT)
        if _sha256(checkpoint_path) != PRETRAIN_CHECKPOINT_SHA256:
            raise ValueError("pretraining checkpoint digest mismatch")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state"])
    model.to(device)

    train_dataset = temporal.generate_paired_temporal_dataset(
        task,
        tokenizer,
        sample_count=2 * config.train_fibers,
        seed=TRAIN_SEED_BASE + seed,
        regime="train",
        mode=mode,
    )
    evaluations = {
        regime: temporal.generate_paired_temporal_dataset(
            task,
            tokenizer,
            sample_count=config.evaluation_samples,
            seed=EVAL_SEEDS[regime],
            regime=regime,
            mode=mode,
        )
        for regime in EVAL_REGIMES
    }
    dataset_digests = {
        "train": temporal.dataset_digest(train_dataset),
        **{
            regime: temporal.dataset_digest(dataset)
            for regime, dataset in evaluations.items()
        },
    }

    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator().manual_seed(seed + 6_013)
    history: List[Dict[str, float]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, config.training_steps + 1):
        fibers = torch.randint(
            0, config.train_fibers, (config.batch_fibers,), generator=generator
        )
        indices = torch.stack((2 * fibers, 2 * fibers + 1), dim=1).reshape(-1)
        inputs = train_dataset.input_ids[indices].to(device)
        targets = train_dataset.target_posteriors[indices].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _answer_logits(model, inputs, answer_ids)
        loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == config.training_steps:
            history.append({"step": step, "loss": float(loss)})
    metrics = {
        regime: _evaluate(model, dataset, answer_ids, device)
        for regime, dataset in evaluations.items()
    }
    checkpoint_path = output_dir / "model.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "model_config": asdict(model_config)},
        checkpoint_path,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "metrics": metrics,
        "history": history,
        "dataset_digests": dataset_digests,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "training_seconds": time.perf_counter() - started,
        "finite_contract": all(
            math.isfinite(value)
            for regime in metrics.values()
            for value in regime.values()
        ),
    }


def _passes(values: List[bool], config: TemporalIdentifiabilityConfig) -> bool:
    required = (
        len(config.seeds)
        if config.allow_underpowered
        else GATES["required_seed_passes"]
    )
    return sum(values) >= required


def classify_campaign(gate_results: Mapping[str, Mapping[str, bool]]) -> str:
    if not all(
        gate_results["g2_uncalibrated"][init] for init in INITIALIZATIONS
    ):
        return "identifiability_control_leak"
    g1 = {init: gate_results["g1_oracle"][init] for init in INITIALIZATIONS}
    g3 = {init: gate_results["g3_calibrated"][init] for init in INITIALIZATIONS}
    if g1["babylm_pretrained"] and g3["babylm_pretrained"]:
        return "identifiable_and_learnable"
    if not any(g1.values()):
        return "oracle_task_too_hard"
    if not any(g3.values()):
        return "in_context_calibration_unused"
    if g1["scratch"] and g3["scratch"]:
        return "pretraining_interference"
    return "in_context_calibration_unused"


def run_campaign(
    config: TemporalIdentifiabilityConfig, output: Path
) -> Dict[str, Any]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    implementation = _implementation_digest()
    task = temporal.TemporalLanguageTaskConfig()
    if _sha256(Path(task.tokenizer_path)) != TOKENIZER_SHA256:
        raise ValueError("tokenizer digest does not match the preregistration")
    tokenizer = temporal.load_tokenizer(task)

    results: List[Dict[str, Any]] = []
    for initialization in INITIALIZATIONS:
        for mode in MODES:
            for seed in config.seeds:
                cell_dir = output / "runs" / initialization / mode / f"seed_{seed}"
                result_path = cell_dir / "result.json"
                fingerprint = _fingerprint(
                    config, implementation, initialization, mode, seed
                )
                if result_path.is_file():
                    existing = json.loads(result_path.read_text(encoding="utf-8"))
                    if existing.get("scientific_fingerprint") == fingerprint:
                        results.append(existing)
                        print(f"reuse {initialization}/{mode} seed {seed}", flush=True)
                        continue
                    raise ValueError(f"incompatible completed cell {result_path}")
                cell = train_cell(
                    config, tokenizer, task, initialization, mode, seed, device, cell_dir
                )
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "hypothesis_id": HYPOTHESIS_ID,
                    "experiment_id": (
                        f"temporal-l1-{initialization}-{mode}-seed{seed}"
                    ),
                    "status": "completed",
                    "evidence_role": (
                        "systems_lifecycle_only_not_quality_evidence"
                        if config.allow_underpowered
                        else EVIDENCE_ROLE
                    ),
                    "completed_at": _utc_now(),
                    "initialization": initialization,
                    "mode": mode,
                    "seed": seed,
                    "configuration": asdict(config),
                    "scientific_fingerprint": fingerprint,
                    "implementation_sha256": implementation,
                    "provenance": {
                        "tokenizer_sha256": TOKENIZER_SHA256,
                        "pretrain_checkpoint": PRETRAIN_CHECKPOINT,
                        "pretrain_checkpoint_sha256": PRETRAIN_CHECKPOINT_SHA256,
                    },
                    **cell,
                }
                _write_json(result_path, record)
                results.append(record)
                accuracy = record["metrics"]["interpolation"]["exact_bin_accuracy"]
                print(
                    f"completed {initialization}/{mode} seed {seed}: "
                    f"interpolation accuracy {accuracy:.4f}",
                    flush=True,
                )

    indexed = {
        (r["initialization"], r["mode"], int(r["seed"])): r for r in results
    }

    def accuracy(init: str, mode: str, seed: int, regime: str) -> float:
        return indexed[(init, mode, seed)]["metrics"][regime]["exact_bin_accuracy"]

    gate_results = {"g1_oracle": {}, "g2_uncalibrated": {}, "g3_calibrated": {}}
    seed_tables: Dict[str, Any] = {}
    for init in INITIALIZATIONS:
        g1 = [
            accuracy(init, "utc_oracle", seed, "interpolation")
            >= GATES["g1_oracle_interpolation"]
            and accuracy(init, "utc_oracle", seed, "composition")
            >= GATES["g1_oracle_composition"]
            for seed in config.seeds
        ]
        g2 = [
            accuracy(init, "uncalibrated", seed, "interpolation")
            <= GATES["g2_uncalibrated_ceiling"]
            for seed in config.seeds
        ]
        margins = [
            accuracy(init, "calibrated", seed, "interpolation")
            - accuracy(init, "uncalibrated", seed, "interpolation")
            for seed in config.seeds
        ]
        g3 = [
            accuracy(init, "calibrated", seed, "interpolation")
            >= GATES["g3_calibrated_floor"]
            and margin >= GATES["g3_paired_margin"]
            for seed, margin in zip(config.seeds, margins)
        ]
        gate_results["g1_oracle"][init] = _passes(g1, config)
        gate_results["g2_uncalibrated"][init] = _passes(g2, config)
        gate_results["g3_calibrated"][init] = _passes(g3, config)
        seed_tables[init] = {
            "g1_seed_passes": g1,
            "g2_seed_passes": g2,
            "g3_seed_passes": g3,
            "calibrated_minus_uncalibrated": margins,
        }

    digest_parity = all(
        indexed[("babylm_pretrained", mode, seed)]["dataset_digests"]
        == indexed[("scratch", mode, seed)]["dataset_digests"]
        for mode in MODES
        for seed in config.seeds
    )
    finite = all(record["finite_contract"] for record in results)
    valid = bool(digest_parity and finite)
    classification = (
        "systems_lifecycle_only_not_quality_evidence"
        if config.allow_underpowered
        else ("invalid" if not valid else classify_campaign(gate_results))
    )
    mean_table = {
        init: {
            mode: {
                regime: float(
                    np.mean(
                        [accuracy(init, mode, seed, regime) for seed in config.seeds]
                    )
                )
                for regime in EVAL_REGIMES
            }
            for mode in MODES
        }
        for init in INITIALIZATIONS
    }
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "evidence_role": (
            "systems_lifecycle_only_not_quality_evidence"
            if config.allow_underpowered
            else EVIDENCE_ROLE
        ),
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "implementation_sha256": implementation,
        "gates": GATES,
        "provenance": {
            "tokenizer_sha256": TOKENIZER_SHA256,
            "pretrain_checkpoint": PRETRAIN_CHECKPOINT,
            "pretrain_checkpoint_sha256": PRETRAIN_CHECKPOINT_SHA256,
        },
        "aggregates": {
            "valid": valid,
            "dataset_digest_parity_contract": digest_parity,
            "finite_numerical_contract": finite,
            "classification": classification,
            "gate_results": gate_results,
            "seed_tables": seed_tables,
            "mean_exact_bin_accuracy": mean_table,
        },
        "summary": {
            "requested": len(INITIALIZATIONS) * len(MODES) * len(config.seeds),
            "completed": len(results),
            "failed": 0,
            "trained_models": len(
                [r for r in results if "reuse" not in r.get("status", "")]
            ),
        },
        "results": [
            {
                "experiment_id": record["experiment_id"],
                "initialization": record["initialization"],
                "mode": record["mode"],
                "seed": record["seed"],
                "scientific_fingerprint": record["scientific_fingerprint"],
                "metrics": record["metrics"],
                "path": str(
                    output
                    / "runs"
                    / record["initialization"]
                    / record["mode"]
                    / f"seed_{record['seed']}"
                    / "result.json"
                ),
            }
            for record in results
        ],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "analysis_seconds": time.perf_counter() - started,
        "method_boundaries": [
            "Templated language with a closed generative grammar; not open text.",
            "One pretraining seed initializes every babylm_pretrained cell.",
            "Gates are preregistered floors, not comparisons against circle-task numbers.",
            "Extrapolation accuracies are descriptive at L1; no gate is set on them.",
        ],
    }
    _write_json(output / "campaign_results.json", campaign)
    return campaign


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_temporal_language_identifiability/"
            "20260812_l1_preregistered"
        ),
    )
    args = parser.parse_args()
    if args.shakedown:
        config = TemporalIdentifiabilityConfig(
            train_fibers=256,
            evaluation_samples=256,
            training_steps=60,
            seeds=(7,),
            device=args.device,
            allow_underpowered=True,
        )
    else:
        config = TemporalIdentifiabilityConfig(device=args.device)
    campaign = run_campaign(config, args.output)
    print(json.dumps(campaign["aggregates"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
