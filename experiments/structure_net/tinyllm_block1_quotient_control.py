#!/usr/bin/env python3
"""Causally control TinyLLM's block-1 horizontal/vertical quotient geometry.

The preregistered d8/N3 comparison retains the five ordinary checkpoints from
the nuisance-support campaign and trains five matched models with two losses at
the post-MLP residual of block 1: cosine reconstruction and a cosine-conditioned
branch adversary connected through gradient reversal.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net.tinyllm_conditional_branch_depth_scan import (
    extract_depth_representations,
)
from experiments.structure_net.tinyllm_nuisance_support_scaling import (
    NuisanceScalingConfig,
    analyze_cell,
    extract_block1_representations,
    generate_paired_dataset,
    serialize_nuisance_inputs,
    train_cell,
)
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    _state_digest,
    _tinyllm_config,
)
from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-block1-quotient-control.v1"
HYPOTHESIS_ID = "tinyllm-block1-horizontal-vertical-control-v1"
SOURCE_SCHEMA_VERSION = "nal.tinyllm-nuisance-support-scaling.v1"
CONDITIONS = ("ordinary_n3", "block1_quotient_controlled_n3")
CUTS = ("query", "post_attention", "post_mlp", "full")
PRIMARY_CUTS = ("post_mlp", "full")
REGIMES = ("in_distribution", "composition", "extrapolation")
PRIMARY_REGIMES = ("composition", "extrapolation")


@dataclass(frozen=True)
class Block1QuotientControlConfig:
    preset: str = "d8"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    training_steps: int = 600
    train_samples: int = 4_096
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    base_loss_weight: float = 0.2
    invariance_loss_weight: float = 0.2
    adversary_width: int = 128
    local_cosines: tuple[float, ...] = (-0.7, -0.35, 0.0, 0.35, 0.7)
    local_delta: float = 0.05
    local_replicates: int = 16
    local_whitening_floor: float = 1e-3
    analysis_seed: int = 83
    probe_train_samples: int = 2_048
    probe_validation_samples: int = 512
    probe_test_samples: int = 1_024
    activation_batch_size: int = 256
    probe_batch_size: int = 256
    probe_steps: int = 240
    probe_width: int = 128
    validation_interval: int = 20
    early_stopping_patience: int = 5
    fiber_cosines: tuple[float, ...] = (-0.8, -0.4, 0.0, 0.4, 0.8)
    fiber_replicates: int = 12
    baseline_campaign: str = (
        "data/experiments/tinyllm_nuisance_support_scaling/20260806_d8_full"
    )
    device_ids: tuple[int, ...] = (0,)
    gpu_slots_per_device: int = 0
    gpu_memory_per_experiment_gb: Optional[float] = 2.75
    max_gpu_slots_per_device: int = 2
    max_parallel_experiments: int = 2
    max_retries: int = 1
    resume: bool = True
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("preregistered campaign requires five seeds")
        if self.preset != "d8" and not self.allow_underpowered:
            raise ValueError("preregistered campaign fixes the d8 preset")
        if self.training_steps < 1 or self.train_samples < 4:
            raise ValueError("training steps and sample count must be positive")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("batch size must be positive and even")
        if self.base_loss_weight < 0.0 or self.invariance_loss_weight < 0.0:
            raise ValueError("intervention weights cannot be negative")
        if (
            not self.allow_underpowered
            and (self.base_loss_weight == 0.0 or self.invariance_loss_weight == 0.0)
        ):
            raise ValueError("confirmatory intervention weights must be positive")
        if self.adversary_width < 2:
            raise ValueError("adversary width must be at least two")
        if self.local_replicates < 2 or not 0.0 < self.local_delta < 0.2:
            raise ValueError("invalid local derivative design")
        if any(abs(value) + self.local_delta >= 1.0 for value in self.local_cosines):
            raise ValueError("local cosine stencil must remain inside (-1, 1)")


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor]:
        return (-gradient,)


def gradient_reverse(value: torch.Tensor) -> torch.Tensor:
    """Identity in the forward pass and sign reversal in the backward pass."""
    return _GradientReversal.apply(value)


class ConditionalBranchAdversary(nn.Module):
    def __init__(self, residual_width: int, hidden_width: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(residual_width + 1, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )

    def forward(self, residual: torch.Tensor, cosine: torch.Tensor) -> torch.Tensor:
        conditioned = torch.cat((residual, cosine[:, None]), dim=-1)
        return self.network(conditioned).squeeze(-1)


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


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _scientific_fingerprint(parameters: Mapping[str, Any]) -> str:
    serialized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _implementation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _experiment_fingerprint(experiment: Experiment) -> str:
    """Fingerprint only scientific inputs, stable across process serialization."""
    config = _config_from_mapping(experiment.parameters["configuration"])
    task = CircleTaskConfig(**experiment.parameters["task_config"])
    return _scientific_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "configuration": asdict(config),
            "task_config": asdict(task),
            "condition": str(experiment.parameters["condition"]),
            "seed": int(experiment.seed),
            "implementation_sha256": str(
                experiment.parameters["implementation_sha256"]
            ),
        }
    )


def _analysis_config(config: Block1QuotientControlConfig) -> NuisanceScalingConfig:
    return NuisanceScalingConfig(
        preset=config.preset,
        seeds=config.seeds,
        supports=("N3",),
        training_arms=("standard_final",),
        training_steps=config.training_steps,
        train_samples=config.train_samples,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip=config.gradient_clip,
        probe_train_samples=config.probe_train_samples,
        probe_validation_samples=config.probe_validation_samples,
        probe_test_samples=config.probe_test_samples,
        activation_batch_size=config.activation_batch_size,
        probe_batch_size=config.probe_batch_size,
        probe_steps=config.probe_steps,
        probe_width=config.probe_width,
        validation_interval=config.validation_interval,
        early_stopping_patience=config.early_stopping_patience,
        fiber_cosines=config.fiber_cosines,
        fiber_replicates=config.fiber_replicates,
        analysis_seed=config.analysis_seed,
        device_ids=config.device_ids,
        allow_underpowered=config.allow_underpowered,
    )


def _forward_from_inputs(
    model: TinyLLMModel, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(inputs.shape[1], device=inputs.device)
    value = model.transformer["wte"](inputs) + model.transformer["wpe"](positions)
    block1 = model.transformer["h"][0](value)
    value = block1
    for block in model.transformer["h"][1:]:
        value = block(value)
    return block1, value


def _task_logits(
    model: TinyLLMModel, residual: torch.Tensor, answer_ids: torch.Tensor
) -> torch.Tensor:
    logits = model.lm_head(model.transformer["ln_f"](residual))[:, -1, :]
    return logits.index_select(-1, answer_ids)


def _initialize_heads(
    residual_width: int, adversary_width: int, seed: int, device: torch.device
) -> tuple[nn.Linear, ConditionalBranchAdversary]:
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed + 91_009)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 91_009)
        base_head = nn.Linear(residual_width, 1).to(device)
        adversary = ConditionalBranchAdversary(
            residual_width, adversary_width
        ).to(device)
    return base_head, adversary


def _protocol_material(
    task: CircleTaskConfig,
    config: Block1QuotientControlConfig,
    seed: int,
) -> tuple[Any, torch.Tensor, str, str]:
    paired = generate_paired_dataset(
        task,
        sample_count=config.train_samples,
        seed=seed + 1_001,
        support="N3",
        shuffle=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 6_013)
    pair_batches = torch.randint(
        0,
        config.train_samples // 2,
        (config.training_steps, config.batch_size // 2),
        generator=generator,
    )
    data_hash = _tensor_digest(
        paired.circle.input_ids,
        paired.circle.target_posteriors,
        paired.fiber.cosine,
        paired.fiber.branch,
    )
    batch_hash = _tensor_digest(pair_batches)
    return paired, pair_batches, data_hash, batch_hash


def train_controlled(
    task: CircleTaskConfig,
    config: Block1QuotientControlConfig,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[TinyLLMModel, Dict[str, Any]]:
    """Train one intervention cell with task-matched data and minibatches."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyLLMModel(
        _tinyllm_config(config.preset, task, seed),
        name=f"Block1QuotientControl_N3_{seed}",
    ).to(device)
    initial_state = _state_digest(model)
    base_head, adversary = _initialize_heads(
        model.config.n_embd, config.adversary_width, seed, device
    )
    model_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    auxiliary_optimizer = torch.optim.AdamW(
        [*base_head.parameters(), *adversary.parameters()],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    paired, pair_batches, data_hash, batch_hash = _protocol_material(
        task, config, seed
    )
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    history: List[Dict[str, float]] = []
    started = time.perf_counter()
    model.train()
    base_head.train()
    adversary.train()
    for step in range(1, config.training_steps + 1):
        pairs = pair_batches[step - 1]
        indices = torch.stack((2 * pairs, 2 * pairs + 1), dim=1).reshape(-1)
        inputs = paired.circle.input_ids[indices].to(device, non_blocking=True)
        targets = paired.circle.target_posteriors[indices].to(
            device, non_blocking=True
        )
        cosine = paired.fiber.cosine[indices].to(device, non_blocking=True)
        branch = paired.fiber.branch[indices].to(device, non_blocking=True)
        model_optimizer.zero_grad(set_to_none=True)
        auxiliary_optimizer.zero_grad(set_to_none=True)
        block1, final = _forward_from_inputs(model, inputs)
        block1_query = block1[:, -1, :]
        task_logits = _task_logits(model, final, answer_ids)
        task_loss = -(
            targets * torch.log_softmax(task_logits, dim=-1)
        ).sum(-1).mean()
        normalized_block1 = F.layer_norm(
            block1_query, (block1_query.shape[-1],)
        )
        base_prediction = base_head(normalized_block1).squeeze(-1)
        base_loss = F.mse_loss(base_prediction, cosine)
        adversary_logits = adversary(
            gradient_reverse(normalized_block1), cosine
        )
        adversary_loss = F.binary_cross_entropy_with_logits(
            adversary_logits, branch
        )
        loss = (
            task_loss
            + config.base_loss_weight * base_loss
            + config.invariance_loss_weight * adversary_loss
        )
        loss.backward()
        model_gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        auxiliary_gradient_norm = torch.nn.utils.clip_grad_norm_(
            [*base_head.parameters(), *adversary.parameters()],
            config.gradient_clip,
        )
        model_optimizer.step()
        auxiliary_optimizer.step()
        if step == 1 or step % 25 == 0 or step == config.training_steps:
            with torch.no_grad():
                adversary_accuracy = (
                    (adversary_logits >= 0.0) == (branch >= 0.5)
                ).float().mean()
            history.append(
                {
                    "step": float(step),
                    "total_loss": float(loss.detach()),
                    "task_loss": float(task_loss.detach()),
                    "base_mse": float(base_loss.detach()),
                    "adversary_cross_entropy": float(adversary_loss.detach()),
                    "online_adversary_accuracy": float(adversary_accuracy),
                    "model_gradient_norm": float(model_gradient_norm),
                    "auxiliary_gradient_norm": float(auxiliary_gradient_norm),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    final_state = _state_digest(model)
    checkpoint = output_dir / "model.pt"
    model.to("cpu").save_checkpoint(
        checkpoint,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "condition": "block1_quotient_controlled_n3",
            "seed": seed,
            "base_loss_weight": config.base_loss_weight,
            "invariance_loss_weight": config.invariance_loss_weight,
            "training_data_sha256": data_hash,
            "minibatch_schedule_sha256": batch_hash,
        },
    )
    heads_checkpoint = output_dir / "temporary_heads.pt"
    torch.save(
        {
            "base_head": base_head.to("cpu").state_dict(),
            "conditional_branch_adversary": adversary.to("cpu").state_dict(),
        },
        heads_checkpoint,
    )
    model.to(device)
    return model, {
        "training_seconds": training_seconds,
        "training_history": history,
        "initial_state_sha256": initial_state,
        "final_state_sha256": final_state,
        "training_data_sha256": data_hash,
        "minibatch_schedule_sha256": batch_hash,
        "checkpoint": str(checkpoint),
        "temporary_heads_checkpoint": str(heads_checkpoint),
        "loss_sign_convention": (
            "L_task + lambda_base*MSE + lambda_inv*BCE(GRL(r1), cosine, branch); "
            "GRL negates only the transformer-side BCE gradient"
        ),
    }


def _baseline_paths(
    config: Block1QuotientControlConfig, seed: int
) -> tuple[Path, Path]:
    root = Path(config.baseline_campaign) / "runs" / "N3" / "standard_final"
    cell = root / f"seed_{seed}"
    return cell / "model.pt", cell / "result.json"


def verify_baseline(
    task: CircleTaskConfig,
    config: Block1QuotientControlConfig,
    seed: int,
) -> Dict[str, Any]:
    checkpoint, detail_path = _baseline_paths(config, seed)
    if not checkpoint.is_file() or not detail_path.is_file():
        raise FileNotFoundError(f"missing retained N3 baseline for seed {seed}")
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    if (
        detail.get("schema_version") != SOURCE_SCHEMA_VERSION
        or detail.get("status") != "completed"
        or detail.get("support") != "N3"
        or detail.get("training_arm") != "standard_final"
        or int(detail.get("seed", -1)) != seed
    ):
        raise ValueError(f"incompatible retained baseline: {detail_path}")
    campaign_path = Path(config.baseline_campaign) / "campaign_results.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    source = campaign["configuration"]
    expected = {
        "preset": config.preset,
        "training_steps": config.training_steps,
        "train_samples": config.train_samples,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "gradient_clip": config.gradient_clip,
    }
    mismatches = {
        key: (source.get(key), value)
        for key, value in expected.items()
        if source.get(key) != value
    }
    if campaign.get("task_config") != asdict(task) or mismatches:
        raise ValueError(f"baseline protocol mismatch: {mismatches}")
    _, _, data_hash, batch_hash = _protocol_material(task, config, seed)
    return {
        "source_result": str(detail_path),
        "checkpoint": str(checkpoint),
        "final_state_sha256": detail["training"]["final_state_sha256"],
        "training_depth_schedule_sha256": detail["training"][
            "training_depth_schedule_sha256"
        ],
        "training_data_sha256": data_hash,
        "minibatch_schedule_sha256": batch_hash,
        "source_task_metrics_on_training_support": detail["training"][
            "task_metrics_on_training_support"
        ],
        "source_analysis": detail["analysis"],
    }


def _phase_for_cosine(value: float, branch: int) -> float:
    positive = math.acos(value)
    return positive if branch == 0 else 2.0 * math.pi - positive


def _local_inputs(
    task: CircleTaskConfig,
    config: Block1QuotientControlConfig,
    regime: str,
    seed: int,
) -> tuple[torch.Tensor, Dict[str, slice], np.ndarray, Dict[str, float]]:
    support_regime = "interpolation" if regime == "in_distribution" else regime
    pieces: List[torch.Tensor] = []
    slices: Dict[str, slice] = {}
    cosines: List[float] = []
    offset = 0
    for cosine_index, cosine in enumerate(config.local_cosines):
        count = config.local_replicates
        for branch in (0, 1):
            directions_rng = np.random.default_rng(
                seed + cosine_index * 1_009 + branch * 101
            )
            directions = directions_rng.choice(np.array([-1.0, 1.0]), count)
            for stencil, value in (
                ("plus", cosine + config.local_delta),
                ("minus", cosine - config.local_delta),
            ):
                future = np.full(count, _phase_for_cosine(value, branch))
                current = (future - directions * task.future_delta) % (2.0 * math.pi)
                inputs = serialize_nuisance_inputs(
                    task,
                    current,
                    directions,
                    support="N3",
                    regime=support_regime,
                    seed=seed + 30_001 + cosine_index * 1_009 + branch * 101,
                )
                key = f"u{cosine_index}_b{branch}_{stencil}"
                slices[key] = slice(offset, offset + count)
                offset += count
                pieces.append(inputs)
                cosines.extend([cosine] * count)
        branch_rng = np.random.default_rng(
            seed + 70_001 + cosine_index * 2_003
        )
        directions = branch_rng.choice(np.array([-1.0, 1.0]), count)
        branch_nuisance_seed = seed + 90_001 + cosine_index * 2_003
        for branch in (0, 1):
            future = np.full(count, _phase_for_cosine(cosine, branch))
            current = (future - directions * task.future_delta) % (2.0 * math.pi)
            inputs = serialize_nuisance_inputs(
                task,
                current,
                directions,
                support="N3",
                regime=support_regime,
                seed=branch_nuisance_seed,
            )
            key = f"u{cosine_index}_branch{branch}"
            slices[key] = slice(offset, offset + count)
            offset += count
            pieces.append(inputs)
            cosines.extend([cosine] * count)
    combined = torch.cat(pieces)
    base_hamming = []
    branch_hamming = []
    for index, _ in enumerate(config.local_cosines):
        for branch in (0, 1):
            plus = combined[slices[f"u{index}_b{branch}_plus"]]
            minus = combined[slices[f"u{index}_b{branch}_minus"]]
            base_hamming.append(float((plus != minus).float().mean()))
        positive = combined[slices[f"u{index}_branch0"]]
        negative = combined[slices[f"u{index}_branch1"]]
        branch_hamming.append(float((positive != negative).float().mean()))
    return combined, slices, np.asarray(cosines), {
        "base_stencil_token_hamming_fraction": float(np.mean(base_hamming)),
        "matched_branch_token_hamming_fraction": float(np.mean(branch_hamming)),
        "branch_chord_uses_identical_nuisance_and_direction": True,
    }


def _local_ratio_for_cut(
    features: np.ndarray,
    slices: Mapping[str, slice],
    config: Block1QuotientControlConfig,
) -> Dict[str, Any]:
    nuisance_residuals = []
    for value_slice in slices.values():
        values = features[value_slice]
        nuisance_residuals.append(values - values.mean(axis=0, keepdims=True))
    residuals = np.concatenate(nuisance_residuals, axis=0)
    scale = np.sqrt(np.mean(np.square(residuals), axis=0))
    positive = scale[scale > 0.0]
    reference = float(np.median(positive)) if len(positive) else 1.0
    scale = np.maximum(scale, config.local_whitening_floor * reference)
    records = []
    for index, cosine in enumerate(config.local_cosines):
        base_norms = []
        for branch in (0, 1):
            plus = features[slices[f"u{index}_b{branch}_plus"]]
            minus = features[slices[f"u{index}_b{branch}_minus"]]
            derivative = (plus - minus) / (2.0 * config.local_delta)
            base_norms.extend(np.linalg.norm(derivative / scale, axis=1))
        positive_branch = features[slices[f"u{index}_branch0"]]
        negative_branch = features[slices[f"u{index}_branch1"]]
        branch_difference = positive_branch - negative_branch
        branch_norms = np.linalg.norm(branch_difference / scale, axis=1)
        base_norm = float(np.median(base_norms))
        branch_norm = float(np.median(branch_norms))
        records.append(
            {
                "cosine": float(cosine),
                "base_sensitivity_norm": base_norm,
                "branch_difference_norm": branch_norm,
                "q_local": base_norm / (branch_norm + 1e-8),
            }
        )
    return {
        "whitening": "diagonal nuisance-residual whitening",
        "delta": config.local_delta,
        "replicates": config.local_replicates,
        "by_cosine": records,
        "median_base_sensitivity_norm": float(
            np.median([record["base_sensitivity_norm"] for record in records])
        ),
        "median_branch_difference_norm": float(
            np.median([record["branch_difference_norm"] for record in records])
        ),
        "median_q_local": float(
            np.median([record["q_local"] for record in records])
        ),
    }


def local_derivative_diagnostic(
    model: TinyLLMModel,
    task: CircleTaskConfig,
    config: Block1QuotientControlConfig,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    for regime_index, regime in enumerate(REGIMES):
        inputs, slices, _, input_diagnostic = _local_inputs(
            task,
            config,
            regime,
            config.analysis_seed + seed * 101 + regime_index * 10_007,
        )
        block1 = extract_block1_representations(
            model,
            inputs,
            device,
            batch_size=config.activation_batch_size,
        )
        full = extract_depth_representations(
            model,
            inputs,
            (float(model.config.n_layer),),
            device,
            batch_size=config.activation_batch_size,
        )[float(model.config.n_layer)]
        features = {**block1, "full": full}
        records[regime] = {
            cut: _local_ratio_for_cut(features[cut], slices, config) for cut in CUTS
        }
        records[regime]["input_diagnostic"] = input_diagnostic
    return records


@torch.no_grad()
def shifted_task_metrics(
    model: TinyLLMModel,
    task: CircleTaskConfig,
    config: Block1QuotientControlConfig,
    device: torch.device,
) -> Dict[str, Any]:
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    metrics = {}
    for index, regime in enumerate(REGIMES):
        generator_regime = "interpolation" if regime == "in_distribution" else regime
        dataset = generate_paired_dataset(
            task,
            sample_count=config.probe_test_samples,
            seed=config.analysis_seed + 50_003 + index * 1_009,
            support="N3",
            regime=generator_regime,
        ).circle
        probabilities: Dict[str, List[torch.Tensor]] = {
            "post_mlp": [],
            "full": [],
        }
        model.eval()
        for start in range(0, len(dataset.input_ids), config.activation_batch_size):
            inputs = dataset.input_ids[
                start : start + config.activation_batch_size
            ].to(device)
            block1, full = _forward_from_inputs(model, inputs)
            for cut, residual in (("post_mlp", block1), ("full", full)):
                probabilities[cut].append(
                    torch.softmax(_task_logits(model, residual, answer_ids), dim=-1).cpu()
                )

        targets = dataset.target_posteriors
        target_bins = targets.argmax(1)
        by_cut: Dict[str, Dict[str, float]] = {}
        for cut, pieces in probabilities.items():
            predicted = torch.cat(pieces)
            predicted_bins = predicted.argmax(1)
            bin_delta = (predicted_bins - target_bins).abs()
            circular_bin_delta = torch.minimum(
                bin_delta, targets.shape[1] - bin_delta
            )
            kl = (
                targets
                * (
                    targets.clamp_min(1e-12).log()
                    - predicted.clamp_min(1e-12).log()
                )
            ).sum(1)
            by_cut[cut] = {
                "mean_kl": float(kl.mean()),
                "exact_bin_accuracy": float(
                    (predicted_bins == target_bins).float().mean()
                ),
                "mean_circular_error_radians": float(
                    circular_bin_delta.float().mean()
                    * (2.0 * math.pi / targets.shape[1])
                ),
                "mean_target_cross_entropy": float(
                    -(targets * predicted.clamp_min(1e-12).log()).sum(1).mean()
                ),
            }
        metrics[regime] = by_cut
    return metrics


def _condition_directory(root: Path, condition: str, seed: int) -> Path:
    return root / "runs" / condition / f"seed_{seed}"


def _config_from_mapping(values: Mapping[str, Any]) -> Block1QuotientControlConfig:
    converted = dict(values)
    for field in ("seeds", "local_cosines", "fiber_cosines", "device_ids"):
        converted[field] = tuple(converted[field])
    return Block1QuotientControlConfig(**converted)


def quotient_control_worker(
    experiment: Experiment, device_id: int
) -> ExperimentResult:
    config = _config_from_mapping(experiment.parameters["configuration"])
    task = CircleTaskConfig(**experiment.parameters["task_config"])
    condition = str(experiment.parameters["condition"])
    seed = int(experiment.seed)
    expected_implementation = str(experiment.parameters["implementation_sha256"])
    actual_implementation = _implementation_digest()
    if actual_implementation != expected_implementation:
        raise RuntimeError(
            "worker implementation changed after campaign construction: "
            f"expected {expected_implementation}, got {actual_implementation}"
        )
    output_dir = _condition_directory(
        Path(experiment.parameters["output_dir"]), condition, seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if device_id < 0 else f"cuda:{device_id}")
    if device.type == "cuda":
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    baseline: Optional[Dict[str, Any]] = None
    try:
        baseline = verify_baseline(task, config, seed)
    except (FileNotFoundError, ValueError):
        if not config.allow_underpowered:
            raise
    if condition == "ordinary_n3":
        if baseline is None:
            model, training = train_cell(
                task,
                _analysis_config(config),
                "N3",
                "standard_final",
                seed,
                device,
                output_dir,
            )
            training["source"] = "trained_underpowered_shakedown_baseline"
            analysis = analyze_cell(
                model,
                task,
                _analysis_config(config),
                "N3",
                "standard_final",
                seed,
                device,
            )
        else:
            model = TinyLLMModel.from_checkpoint(
                baseline["checkpoint"], map_location=device
            )
            actual_digest = _state_digest(model)
            if actual_digest != baseline["final_state_sha256"]:
                raise ValueError(
                    f"retained baseline checkpoint digest mismatch for seed {seed}"
                )
            training = {
                key: value
                for key, value in baseline.items()
                if key != "source_analysis"
            }
            training["source"] = "retained_exact_matched_baseline"
            analysis = baseline["source_analysis"]
    else:
        model, training = train_controlled(
            task, config, seed, device, output_dir
        )
        analysis = analyze_cell(
            model,
            task,
            _analysis_config(config),
            "N3",
            "block1_quotient_controlled",
            seed,
            device,
        )
    task_metrics = shifted_task_metrics(model, task, config, device)
    local = local_derivative_diagnostic(model, task, config, seed, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    detail = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": experiment.id,
        "status": "completed",
        "completed_at": _utc_now(),
        "condition": condition,
        "seed": seed,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "training": training,
        "analysis": analysis,
        "task_metrics": task_metrics,
        "local_derivative": local,
        "peak_cuda_allocated_gb": peak_allocated,
        "peak_cuda_reserved_gb": peak_reserved,
        "wall_seconds": elapsed,
        "scientific_fingerprint": _experiment_fingerprint(experiment),
        "implementation_sha256": actual_implementation,
    }
    detail_path = output_dir / "result.json"
    _write_json(detail_path, detail)
    composition = analysis["cuts"]["post_mlp"]["probe"]["evaluations"][
        "composition"
    ]
    extrapolation = analysis["cuts"]["post_mlp"]["probe"]["evaluations"][
        "extrapolation"
    ]
    metrics = {
        "post_mlp_composition_cosine": float(composition["cosine_pearson"]),
        "post_mlp_composition_branch": float(composition["balanced_accuracy"]),
        "post_mlp_extrapolation_cosine": float(extrapolation["cosine_pearson"]),
        "post_mlp_extrapolation_branch": float(extrapolation["balanced_accuracy"]),
        "peak_cuda_allocated_gb": peak_allocated,
    }
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=HYPOTHESIS_ID,
        metrics=metrics,
        primary_metric=min(
            metrics["post_mlp_composition_cosine"],
            metrics["post_mlp_extrapolation_cosine"],
        ),
        model_architecture=[
            task.vocab_size,
            *([model.config.n_embd] * model.config.n_layer),
            task.phase_bins,
        ],
        model_parameters=detail["model_parameters"],
        training_time=elapsed,
        training_history=training.get("training_history", []),
        model_checkpoint=training["checkpoint"],
        observations=[f"detail={detail_path}", f"condition={condition}"],
    )


def _probe_metrics(detail: Mapping[str, Any], cut: str, regime: str) -> Mapping[str, Any]:
    source_regime = "interpolation" if regime == "in_distribution" else regime
    return detail["analysis"]["cuts"][cut]["probe"]["evaluations"][source_regime]


def endpoint_pass(metrics: Mapping[str, Any]) -> bool:
    return bool(
        metrics["cosine_pearson"] >= 0.90
        and metrics["balanced_accuracy"] <= 0.55
        and metrics["conditional_log_loss_gain_over_cosine_only"] <= 0.02
    )


def aggregate_details(
    details: Sequence[Mapping[str, Any]],
    config: Block1QuotientControlConfig,
) -> Dict[str, Any]:
    cells: Dict[str, Any] = {}
    by_key = {(item["condition"], int(item["seed"])): item for item in details}
    for condition in CONDITIONS:
        selected = [item for item in details if item["condition"] == condition]
        cuts: Dict[str, Any] = {}
        for cut in CUTS:
            cuts[cut] = {}
            for regime in REGIMES:
                values = [_probe_metrics(item, cut, regime) for item in selected]
                passes = [endpoint_pass(value) for value in values]
                cuts[cut][regime] = {
                    "seed_count": len(values),
                    "cosine_pearson_mean": float(
                        np.mean([value["cosine_pearson"] for value in values])
                    ),
                    "branch_accuracy_mean": float(
                        np.mean([value["balanced_accuracy"] for value in values])
                    ),
                    "conditional_log_loss_gain_mean": float(
                        np.mean(
                            [
                                value[
                                    "conditional_log_loss_gain_over_cosine_only"
                                ]
                                for value in values
                            ]
                        )
                    ),
                    "pass_count": int(sum(passes)),
                    "pass_by_seed": {
                        str(item["seed"]): passed
                        for item, passed in zip(selected, passes)
                    },
                }
        cells[condition] = {"cuts": cuts}

    task_comparisons: Dict[str, Any] = {}
    common_seeds = sorted(
        set(seed for condition, seed in by_key if condition == CONDITIONS[0])
        & set(seed for condition, seed in by_key if condition == CONDITIONS[1])
    )
    for regime in REGIMES:
        rows = []
        for seed in common_seeds:
            baseline = by_key[(CONDITIONS[0], seed)]["task_metrics"][regime][
                "full"
            ]["exact_bin_accuracy"]
            controlled = by_key[(CONDITIONS[1], seed)]["task_metrics"][regime][
                "full"
            ]["exact_bin_accuracy"]
            rows.append(
                {
                    "seed": seed,
                    "baseline_accuracy": baseline,
                    "controlled_accuracy": controlled,
                    "controlled_minus_baseline": controlled - baseline,
                    "within_three_points": controlled >= baseline - 0.03,
                }
            )
        task_comparisons[regime] = {
            "by_seed": rows,
            "mean_controlled_minus_baseline": float(
                np.mean([row["controlled_minus_baseline"] for row in rows])
            ),
            "pass_count": int(sum(row["within_three_points"] for row in rows)),
        }

    controlled = cells[CONDITIONS[1]]["cuts"]
    required = 4 if len(config.seeds) >= 5 else len(config.seeds)
    joint_by_seed: Dict[str, Any] = {}
    for seed in common_seeds:
        representation_cells = {
            f"{cut}:{regime}": bool(
                controlled[cut][regime]["pass_by_seed"][str(seed)]
            )
            for cut in PRIMARY_CUTS
            for regime in PRIMARY_REGIMES
        }
        task_cells = {
            regime: bool(
                next(
                    row["within_three_points"]
                    for row in task_comparisons[regime]["by_seed"]
                    if row["seed"] == seed
                )
            )
            for regime in REGIMES
        }
        joint_by_seed[str(seed)] = {
            "representation_cells": representation_cells,
            "representation_pass": all(representation_cells.values()),
            "task_cells": task_cells,
            "task_pass": all(task_cells.values()),
            "joint_pass": all(representation_cells.values())
            and all(task_cells.values()),
        }
    representation_pass_count = sum(
        value["representation_pass"] for value in joint_by_seed.values()
    )
    task_pass_count = sum(value["task_pass"] for value in joint_by_seed.values())
    joint_pass_count = sum(value["joint_pass"] for value in joint_by_seed.values())
    representation_success = representation_pass_count >= required
    task_success = task_pass_count >= required
    joint_success = joint_pass_count >= required
    composition_success = sum(
        all(
            controlled[cut]["composition"]["pass_by_seed"][str(seed)]
            for cut in PRIMARY_CUTS
        )
        for seed in common_seeds
    ) >= required
    extrapolation_success = sum(
        all(
            controlled[cut]["extrapolation"]["pass_by_seed"][str(seed)]
            for cut in PRIMARY_CUTS
        )
        for seed in common_seeds
    ) >= required
    if joint_success:
        conclusion = "stable_internal_quotient_created"
    elif composition_success and not extrapolation_success:
        conclusion = "support_relative_quotient_next_equivariant_encoder"
    else:
        post_mlp = controlled["post_mlp"]
        base_preserved = all(
            post_mlp[regime]["cosine_pearson_mean"] >= 0.9
            for regime in PRIMARY_REGIMES
        )
        branch_erased = all(
            post_mlp[regime]["branch_accuracy_mean"] <= 0.55
            and post_mlp[regime]["conditional_log_loss_gain_mean"] <= 0.02
            for regime in PRIMARY_REGIMES
        )
        if branch_erased and not base_preserved:
            conclusion = "compression_not_quotient"
        elif base_preserved and not branch_erased:
            conclusion = "base_preserved_branch_remains"
        else:
            conclusion = "no_joint_improvement_test_equivariant_front_end"
    local_summary = {
        condition: {
            cut: {
                regime: float(
                    np.mean(
                        [
                            item["local_derivative"][regime][cut]["median_q_local"]
                            for item in details
                            if item["condition"] == condition
                        ]
                    )
                )
                for regime in REGIMES
            }
            for cut in CUTS
        }
        for condition in CONDITIONS
    }
    return {
        "cells": cells,
        "task_performance": task_comparisons,
        "local_q_mean": local_summary,
        "preregistered_gate": {
            "representation_success": representation_success,
            "task_success": task_success,
            "overall_success": joint_success,
            "joint_pass_by_seed": joint_by_seed,
            "representation_pass_count": representation_pass_count,
            "task_pass_count": task_pass_count,
            "joint_pass_count": joint_pass_count,
            "required_seed_passes": required,
            "cosine_threshold": 0.90,
            "branch_accuracy_ceiling": 0.55,
            "conditional_log_loss_gain_ceiling": 0.02,
            "task_drop_ceiling": 0.03,
        },
        "conclusion": conclusion,
    }


def _experiments(
    config: Block1QuotientControlConfig,
    task: CircleTaskConfig,
    output_dir: Path,
) -> List[Experiment]:
    common = {
        "configuration": asdict(config),
        "task_config": asdict(task),
        "output_dir": str(output_dir),
        "worker_schema_version": SCHEMA_VERSION,
        "implementation_sha256": _implementation_digest(),
    }
    return [
        Experiment(
            id=f"tinyllm-block1-quotient-{condition}-seed{seed}",
            hypothesis_id=HYPOTHESIS_ID,
            name=f"TinyLLM block-1 quotient control {condition} seed {seed}",
            parameters={**common, "condition": condition, "seed": seed},
            seed=seed,
        )
        for condition in CONDITIONS
        for seed in config.seeds
    ]


def _existing_detail(experiment: Experiment, output_dir: Path) -> Optional[Dict[str, Any]]:
    condition = str(experiment.parameters["condition"])
    path = _condition_directory(output_dir, condition, int(experiment.seed)) / "result.json"
    if not path.is_file():
        return None
    try:
        detail = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if (
        detail.get("schema_version") != SCHEMA_VERSION
        or detail.get("status") != "completed"
        or detail.get("condition") != condition
        or int(detail.get("seed", -1)) != experiment.seed
        or detail.get("scientific_fingerprint")
        != _experiment_fingerprint(experiment)
    ):
        return None
    checkpoint = Path(detail.get("training", {}).get("checkpoint", ""))
    if not checkpoint.is_file():
        return None
    return detail


async def run_campaign(
    config: Block1QuotientControlConfig,
    task: CircleTaskConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = _experiments(config, task, output_dir)
    existing = {
        experiment.id: detail
        for experiment in experiments
        if (detail := _existing_detail(experiment, output_dir)) is not None
    }
    pending = [experiment for experiment in experiments if experiment.id not in existing]
    lab = LabConfig(
        project_name="tinyllm_block1_quotient_control",
        results_dir=str(output_dir),
        device_ids=list(config.device_ids),
        max_parallel_experiments=config.max_parallel_experiments,
        gpu_slots_per_device=config.gpu_slots_per_device,
        gpu_memory_per_experiment_gb=config.gpu_memory_per_experiment_gb,
        max_gpu_slots_per_device=config.max_gpu_slots_per_device,
        max_experiment_retries=config.max_retries,
        resume_completed_experiments=config.resume,
        auto_balance=False,
        enable_wandb=False,
        verbose=True,
    )
    runner = AsyncExperimentRunner(lab, quotient_control_worker)
    results = await runner.run_experiments(pending) if pending else []
    successful_ids = {result.experiment_id for result in results if result.error is None}
    details = list(existing.values())
    for experiment in pending:
        if experiment.id not in successful_ids:
            continue
        path = _condition_directory(
            output_dir,
            str(experiment.parameters["condition"]),
            int(experiment.seed),
        ) / "result.json"
        details.append(json.loads(path.read_text(encoding="utf-8")))
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed" if len(details) == len(experiments) else "partial",
        "completed_at": _utc_now(),
        "configuration": asdict(config),
        "task_config": asdict(task),
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "implementation_sha256": _implementation_digest(),
        "baseline_reuse": (
            "The five ordinary N3 checkpoints are retained exact protocol matches; "
            "no baseline was selected or altered after intervention outcomes."
        ),
        "summary": {
            "requested": len(experiments),
            "reused": len(existing),
            "scheduled": len(pending),
            "completed": len(details),
            "failed": len(experiments) - len(details),
        },
        "scheduler": {
            "backend": "nal_local_spawn",
            "logical_device_slots": list(runner.slot_plan.slots),
            "slots_by_device": {
                str(key): value for key, value in runner.slot_plan.slots_by_device.items()
            },
            "calibration": runner.slot_plan.calibration,
        },
        "aggregates": (
            aggregate_details(details, config)
            if len(details) == len(experiments)
            else {}
        ),
        "results": [
            {
                "experiment_id": result.experiment_id,
                "status": result.status.value,
                "metrics": result.metrics,
                "error": result.error,
                "model_checkpoint": result.model_checkpoint,
            }
            for result in results
        ],
        "method_boundaries": [
            "The nonlinear probes measure tested conditional decodability, not certified mutual information.",
            "The local diagnostic uses diagonal whitening of within-group nuisance residuals.",
            "The baseline weights are retained from the exact matched precursor campaign.",
            "The temporary training heads are stored but excluded from task inference.",
        ],
    }
    _write_json(output_dir / "campaign_results.json", bundle)
    return bundle


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_devices(value: str) -> tuple[int, ...]:
    if value.strip().lower() == "cpu":
        return (-1,)
    if value.strip().lower() == "auto":
        return tuple(range(torch.cuda.device_count())) if torch.cuda.is_available() else (-1,)
    return _comma_ints(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--train-samples", type=int, default=4_096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-weight", type=float, default=0.2)
    parser.add_argument("--invariance-weight", type=float, default=0.2)
    parser.add_argument("--probe-steps", type=int, default=240)
    parser.add_argument("--probe-train-samples", type=int, default=2_048)
    parser.add_argument("--probe-validation-samples", type=int, default=512)
    parser.add_argument("--probe-test-samples", type=int, default=1_024)
    parser.add_argument("--local-replicates", type=int, default=16)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--slots-per-gpu", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--baseline-campaign",
        default=(
            "data/experiments/tinyllm_nuisance_support_scaling/20260806_d8_full"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_block1_quotient_control/"
            "20260806_d8_preregistered"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    preset = "d8"
    if args.shakedown:
        preset = "tiny"
        args.seeds = "7"
        args.steps = 2
        args.train_samples = 32
        args.batch_size = 8
        args.probe_steps = 20
        args.probe_train_samples = 64
        args.probe_validation_samples = 32
        args.probe_test_samples = 32
        args.local_replicates = 2
        args.allow_underpowered = True
    config = Block1QuotientControlConfig(
        preset=preset,
        seeds=_comma_ints(args.seeds),
        training_steps=args.steps,
        train_samples=args.train_samples,
        batch_size=args.batch_size,
        base_loss_weight=args.base_weight,
        invariance_loss_weight=args.invariance_weight,
        probe_steps=args.probe_steps,
        probe_train_samples=args.probe_train_samples,
        probe_validation_samples=args.probe_validation_samples,
        probe_test_samples=args.probe_test_samples,
        local_replicates=args.local_replicates,
        baseline_campaign=args.baseline_campaign,
        device_ids=_parse_devices(args.gpus),
        gpu_slots_per_device=args.slots_per_gpu,
        max_parallel_experiments=args.max_parallel,
        max_retries=args.retries,
        resume=args.resume,
        allow_underpowered=args.allow_underpowered,
    )
    task = CircleTaskConfig(train_samples=config.train_samples)
    bundle = asyncio.run(run_campaign(config, task, args.output))
    print(
        json.dumps(
            {
                "summary": bundle["summary"],
                "conclusion": bundle.get("aggregates", {}).get("conclusion"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(args.output / "campaign_results.json")
    return 0 if bundle["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
