#!/usr/bin/env python3
"""Run the preregistered TinyLLM invariant-front-end causal comparison.

The three arms are the retained raw N3 model, a fixed observation-only
canonicalizer, and a learned sensor encoder whose vector output is SO(2)
equivariant by construction.  Frozen probes measure base retention and
conditional branch leakage at four declared representation cuts.
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
    _add_log_loss_gains,
    fit_conditional_probe,
    fit_cosine_only_null,
)
from experiments.structure_net.tinyllm_nuisance_support_scaling import (
    NuisanceScalingConfig,
    PairedCircleDataset,
    generate_paired_dataset,
)
from experiments.structure_net.tinyllm_predictive_circle import (
    CircleTaskConfig,
    _state_digest,
    _tinyllm_config,
)
from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-invariant-frontend-causal.v1"
HYPOTHESIS_ID = "tinyllm-invariant-frontend-stable-cosine-quotient-v1"
SOURCE_SCHEMA_VERSION = "nal.tinyllm-nuisance-support-scaling.v1"
CONDITIONS = ("raw_n3", "analytic_invariant", "learned_equivariant")
CUTS = ("frontend", "post_attention", "post_mlp", "full")
REGIMES = ("in_distribution", "composition", "extrapolation")
PRIMARY_REGIMES = ("composition", "extrapolation")


@dataclass(frozen=True)
class InvariantFrontendConfig:
    preset: str = "d8"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    training_steps: int = 600
    train_samples: int = 4_096
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    vector_channels: int = 16
    speed_grid_size: int = 91
    speed_grid_min: float = 0.10
    speed_grid_max: float = 0.60
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
    baseline_campaign: str = (
        "data/experiments/tinyllm_nuisance_support_scaling/20260806_d8_full"
    )
    device_ids: tuple[int, ...] = (0,)
    gpu_slots_per_device: int = 0
    gpu_memory_per_experiment_gb: Optional[float] = 3.0
    max_gpu_slots_per_device: int = 2
    max_parallel_experiments: int = 2
    max_retries: int = 1
    resume: bool = True
    allow_underpowered: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and distinct")
        if len(self.seeds) < 5 and not self.allow_underpowered:
            raise ValueError("the preregistered campaign requires five seeds")
        if self.preset != "d8" and not self.allow_underpowered:
            raise ValueError("the preregistered campaign fixes the d8 preset")
        if self.training_steps < 1 or self.train_samples < 4:
            raise ValueError("training steps and samples must be positive")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("batch size must be positive and even")
        if self.vector_channels < 2:
            raise ValueError("vector_channels must be at least two")
        if self.speed_grid_size < 3 or self.speed_grid_min >= self.speed_grid_max:
            raise ValueError("invalid analytic speed grid")


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


def _module_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _implementation_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def decode_sensor_tokens(input_ids: torch.Tensor, task: CircleTaskConfig) -> torch.Tensor:
    """Decode the serialized quantization bins to their observed bin centers."""
    tokens = input_ids[:, 1:-1].reshape(-1, task.sensor_steps, 3)
    offsets = 100 + task.value_bins * torch.arange(3, device=tokens.device)
    quantized = tokens - offsets[None, None, :]
    if bool(((quantized < 0) | (quantized >= task.value_bins)).any()):
        raise ValueError("input contains tokens outside the declared sensor bins")
    unit = (quantized.to(torch.float32) + 0.5) / float(task.value_bins)
    return unit * (2.0 * task.quantization_limit) - task.quantization_limit


class AnalyticInvariantCanonicalizer:
    """Observation-only sinusoid fit after removing the known nuisance subgroup."""

    def __init__(self, task: CircleTaskConfig, config: InvariantFrontendConfig):
        self.task = task
        self.speeds = np.linspace(
            config.speed_grid_min,
            config.speed_grid_max,
            config.speed_grid_size,
            dtype=np.float64,
        )
        history = np.arange(task.sensor_steps, dtype=np.float64) - (task.sensor_steps - 1)
        self.designs = []
        self.pseudoinverses = []
        for speed in self.speeds:
            design = np.column_stack(
                (np.ones_like(history), history, np.cos(speed * history), np.sin(speed * history))
            )
            self.designs.append(design)
            self.pseudoinverses.append(np.linalg.pinv(design))
        self.designs = np.asarray(self.designs)
        self.pseudoinverses = np.asarray(self.pseudoinverses)

    def __call__(self, sensor: torch.Tensor) -> torch.Tensor:
        planar = sensor[..., :2].detach().cpu().double().numpy()
        coefficients = np.einsum("gkt,ntc->ngkc", self.pseudoinverses, planar)
        fitted = np.einsum("gtk,ngkc->ngtc", self.designs, coefficients)
        errors = np.square(fitted - planar[:, None]).mean(axis=(2, 3))
        chosen = errors.argmin(axis=1)
        coeff = coefficients[np.arange(len(planar)), chosen]
        cx_cos, cy_cos = coeff[:, 2, 0], coeff[:, 2, 1]
        cx_sin, cy_sin = coeff[:, 3, 0], coeff[:, 3, 1]
        plus_error = np.square(cx_sin + cy_cos) + np.square(cy_sin - cx_cos)
        minus_error = np.square(cx_sin - cy_cos) + np.square(cy_sin + cx_cos)
        positive = plus_error <= minus_error
        cos_endpoint = np.where(
            positive, 0.5 * (cx_cos + cy_sin), 0.5 * (cx_cos - cy_sin)
        )
        sin_endpoint = np.where(
            positive, 0.5 * (cy_cos - cx_sin), 0.5 * (cy_cos + cx_sin)
        )
        norm = np.sqrt(np.square(cos_endpoint) + np.square(sin_endpoint)).clip(1e-8)
        cos_endpoint /= norm
        sin_endpoint /= norm
        delta = np.full(len(planar), self.task.future_delta, dtype=np.float64)
        direction = np.where(positive, 1.0, -1.0)
        future_cosine = cos_endpoint * np.cos(delta) - direction * sin_endpoint * np.sin(delta)
        return torch.from_numpy(future_cosine.astype(np.float32)).to(sensor.device)[:, None]


class LearnedEquivariantEncoder(nn.Module):
    """SO(2)-equivariant planar encoder with structural trend/scale removal."""

    def __init__(self, sensor_steps: int, vector_channels: int = 16):
        super().__init__()
        history = torch.arange(sensor_steps, dtype=torch.float32) - (sensor_steps - 1)
        trend = torch.stack((torch.ones_like(history), history), dim=1)
        projector = torch.eye(sensor_steps) - trend @ torch.linalg.pinv(trend)
        self.register_buffer("trend_projector", projector)
        self.temporal_weights = nn.Parameter(torch.empty(vector_channels, sensor_steps))
        invariant_width = vector_channels * vector_channels
        hidden = max(32, 2 * vector_channels)
        self.scalar_mixer = nn.Sequential(
            nn.Linear(invariant_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, vector_channels),
        )
        nn.init.normal_(self.temporal_weights, std=1.0 / math.sqrt(sensor_steps))

    def vector(self, sensor: torch.Tensor) -> torch.Tensor:
        planar = sensor[..., :2]
        centered = torch.einsum("ts,bsc->btc", self.trend_projector, planar)
        scale = centered.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-6)
        normalized = centered / scale
        channels = torch.einsum("kt,btc->bkc", self.temporal_weights, normalized)
        gram = torch.einsum("bkc,blc->bkl", channels, channels)
        weights = torch.softmax(self.scalar_mixer(gram.flatten(1)), dim=-1)
        vector = torch.einsum("bk,bkc->bc", weights, channels)
        return vector / vector.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, sensor: torch.Tensor) -> torch.Tensor:
        return self.vector(sensor)[:, :1]


class StructuredTinyLLM(nn.Module):
    """TinyLLM receiving one scalar sensor feature between BOS and query."""

    def __init__(
        self,
        model: TinyLLMModel,
        condition: str,
        task: CircleTaskConfig,
        config: InvariantFrontendConfig,
    ):
        super().__init__()
        if condition not in {"analytic_invariant", "learned_equivariant"}:
            raise ValueError(f"unsupported structured condition {condition}")
        self.model = model
        self.condition = condition
        self.scalar_embedding = nn.Linear(1, model.config.n_embd)
        self.encoder: Optional[LearnedEquivariantEncoder]
        self.encoder = (
            LearnedEquivariantEncoder(task.sensor_steps, config.vector_channels)
            if condition == "learned_equivariant"
            else None
        )

    def feature(
        self, sensor: torch.Tensor, analytic_feature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.encoder is not None:
            return self.encoder(sensor)
        if analytic_feature is None:
            raise ValueError("analytic_feature is required for the fixed canonicalizer")
        return analytic_feature

    def forward_cuts(
        self,
        input_ids: torch.Tensor,
        sensor: torch.Tensor,
        analytic_feature: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feature = self.feature(sensor, analytic_feature)
        positions = torch.arange(3, device=input_ids.device)
        bos = self.model.transformer["wte"](input_ids[:, :1])
        query = self.model.transformer["wte"](input_ids[:, -1:])
        value = torch.cat((bos, self.scalar_embedding(feature)[:, None, :], query), dim=1)
        value = value + self.model.transformer["wpe"](positions)
        block = self.model.transformer["h"][0]
        attended = value + block.attn(block.ln_1(value))
        post_mlp = attended + block.mlp(block.ln_2(attended))
        final = post_mlp
        for later in self.model.transformer["h"][1:]:
            final = later(final)
        return {
            "frontend": feature,
            "post_attention": attended[:, -1, :],
            "post_mlp": post_mlp[:, -1, :],
            "full": final[:, -1, :],
        }


def _raw_cuts(
    model: TinyLLMModel, input_ids: torch.Tensor, sensor: torch.Tensor
) -> Dict[str, torch.Tensor]:
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    value = model.transformer["wte"](input_ids) + model.transformer["wpe"](positions)
    block = model.transformer["h"][0]
    attended = value + block.attn(block.ln_1(value))
    post_mlp = attended + block.mlp(block.ln_2(attended))
    final = post_mlp
    for later in model.transformer["h"][1:]:
        final = later(final)
    return {
        "frontend": sensor.flatten(1),
        "post_attention": attended[:, -1, :],
        "post_mlp": post_mlp[:, -1, :],
        "full": final[:, -1, :],
    }


def _task_logits(model: TinyLLMModel, residual: torch.Tensor, answer_ids: torch.Tensor) -> torch.Tensor:
    logits = model.lm_head(model.transformer["ln_f"](residual))
    return logits.index_select(-1, answer_ids)


def _protocol_material(
    task: CircleTaskConfig, config: InvariantFrontendConfig, seed: int
) -> tuple[PairedCircleDataset, torch.Tensor, str, str]:
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
    return paired, pair_batches, data_hash, _tensor_digest(pair_batches)


def _condition_directory(root: Path, condition: str, seed: int) -> Path:
    return root / "runs" / condition / f"seed_{seed}"


def _baseline_paths(config: InvariantFrontendConfig, seed: int) -> tuple[Path, Path]:
    cell = (
        Path(config.baseline_campaign)
        / "runs"
        / "N3"
        / "standard_final"
        / f"seed_{seed}"
    )
    return cell / "model.pt", cell / "result.json"


def verify_baseline(
    task: CircleTaskConfig, config: InvariantFrontendConfig, seed: int
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
    campaign = json.loads(
        (Path(config.baseline_campaign) / "campaign_results.json").read_text(encoding="utf-8")
    )
    expected = {
        "preset": config.preset,
        "training_steps": config.training_steps,
        "train_samples": config.train_samples,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "gradient_clip": config.gradient_clip,
    }
    mismatch = {key: (campaign["configuration"].get(key), value) for key, value in expected.items() if campaign["configuration"].get(key) != value}
    if (campaign.get("task_config") != asdict(task) or mismatch) and not config.allow_underpowered:
        raise ValueError(f"baseline protocol mismatch: {mismatch}")
    paired, batches, data_hash, batch_hash = _protocol_material(task, config, seed)
    del paired, batches
    return {
        "checkpoint": str(checkpoint),
        "source_result": str(detail_path),
        "final_state_sha256": detail["training"]["final_state_sha256"],
        "training_data_sha256": data_hash,
        "minibatch_schedule_sha256": batch_hash,
    }


def train_structured(
    task: CircleTaskConfig,
    config: InvariantFrontendConfig,
    condition: str,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[StructuredTinyLLM, Dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TinyLLMModel(
        _tinyllm_config(config.preset, task, seed),
        name=f"InvariantFrontend_{condition}_{seed}",
    )
    structured = StructuredTinyLLM(model, condition, task, config).to(device)
    initial_model = _state_digest(model)
    initial_frontend = _module_digest(structured)
    paired, pair_batches, data_hash, batch_hash = _protocol_material(task, config, seed)
    sensor = decode_sensor_tokens(paired.circle.input_ids, task)
    canonicalizer = AnalyticInvariantCanonicalizer(task, config)
    analytic = canonicalizer(sensor) if condition == "analytic_invariant" else None
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    optimizer = torch.optim.AdamW(
        structured.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: List[Dict[str, float]] = []
    started = time.perf_counter()
    structured.train()
    for step in range(1, config.training_steps + 1):
        pairs = pair_batches[step - 1]
        indices = torch.stack((2 * pairs, 2 * pairs + 1), dim=1).reshape(-1)
        inputs = paired.circle.input_ids[indices].to(device, non_blocking=True)
        observed = sensor[indices].to(device, non_blocking=True)
        fixed = analytic[indices].to(device, non_blocking=True) if analytic is not None else None
        targets = paired.circle.target_posteriors[indices].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        cuts = structured.forward_cuts(inputs, observed, fixed)
        logits = _task_logits(model, cuts["full"], answer_ids)
        loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            structured.parameters(), config.gradient_clip
        )
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == config.training_steps:
            history.append(
                {
                    "step": float(step),
                    "task_loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    final_model = _state_digest(model)
    final_frontend = _module_digest(structured)
    checkpoint = output_dir / "model.pt"
    model.to("cpu").save_checkpoint(
        checkpoint,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "condition": condition,
            "seed": seed,
            "training_data_sha256": data_hash,
            "minibatch_schedule_sha256": batch_hash,
        },
    )
    frontend_checkpoint = output_dir / "frontend.pt"
    torch.save(
        {
            "condition": condition,
            "scalar_embedding": structured.scalar_embedding.to("cpu").state_dict(),
            "encoder": structured.encoder.to("cpu").state_dict() if structured.encoder else None,
        },
        frontend_checkpoint,
    )
    structured.to(device)
    return structured, {
        "source": "new_matched_task_only_training",
        "training_seconds": training_seconds,
        "training_history": history,
        "initial_model_state_sha256": initial_model,
        "initial_structured_state_sha256": initial_frontend,
        "final_model_state_sha256": final_model,
        "final_structured_state_sha256": final_frontend,
        "training_data_sha256": data_hash,
        "minibatch_schedule_sha256": batch_hash,
        "checkpoint": str(checkpoint),
        "frontend_checkpoint": str(frontend_checkpoint),
        "objective": "ordinary cosine-interval task cross-entropy only",
    }


@torch.no_grad()
def extract_features(
    condition: str,
    model: TinyLLMModel,
    structured: Optional[StructuredTinyLLM],
    canonicalizer: AnalyticInvariantCanonicalizer,
    dataset: PairedCircleDataset,
    task: CircleTaskConfig,
    config: InvariantFrontendConfig,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    if structured is not None:
        structured.eval()
    sensor = decode_sensor_tokens(dataset.circle.input_ids, task)
    analytic = canonicalizer(sensor) if condition == "analytic_invariant" else None
    captured: Dict[str, List[torch.Tensor]] = {cut: [] for cut in CUTS}
    for start in range(0, len(sensor), config.activation_batch_size):
        stop = start + config.activation_batch_size
        inputs = dataset.circle.input_ids[start:stop].to(device)
        observed = sensor[start:stop].to(device)
        if structured is None:
            cuts = _raw_cuts(model, inputs, observed)
        else:
            fixed = analytic[start:stop].to(device) if analytic is not None else None
            cuts = structured.forward_cuts(inputs, observed, fixed)
        for cut, value in cuts.items():
            captured[cut].append(value.detach().cpu())
    return {cut: torch.cat(parts).float().numpy() for cut, parts in captured.items()}


def _probe_config(config: InvariantFrontendConfig) -> NuisanceScalingConfig:
    return NuisanceScalingConfig(
        preset=config.preset,
        seeds=config.seeds,
        supports=("N3",),
        training_arms=("standard_final",),
        training_steps=config.training_steps,
        train_samples=config.train_samples,
        batch_size=config.batch_size,
        probe_train_samples=config.probe_train_samples,
        probe_validation_samples=config.probe_validation_samples,
        probe_test_samples=config.probe_test_samples,
        activation_batch_size=config.activation_batch_size,
        probe_batch_size=config.probe_batch_size,
        probe_steps=config.probe_steps,
        probe_width=config.probe_width,
        validation_interval=config.validation_interval,
        early_stopping_patience=config.early_stopping_patience,
        allow_underpowered=config.allow_underpowered,
    )


def analyze(
    condition: str,
    model: TinyLLMModel,
    structured: Optional[StructuredTinyLLM],
    task: CircleTaskConfig,
    config: InvariantFrontendConfig,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    canonicalizer = AnalyticInvariantCanonicalizer(task, config)
    datasets = {
        "train": generate_paired_dataset(task, sample_count=config.probe_train_samples, seed=config.analysis_seed + 101, support="N3"),
        "validation": generate_paired_dataset(task, sample_count=config.probe_validation_samples, seed=config.analysis_seed + 211, support="N3"),
        "in_distribution": generate_paired_dataset(task, sample_count=config.probe_test_samples, seed=config.analysis_seed + 307, support="N3"),
        "composition": generate_paired_dataset(task, sample_count=config.probe_test_samples, seed=config.analysis_seed + 1_316, support="N3", regime="composition"),
        "extrapolation": generate_paired_dataset(task, sample_count=config.probe_test_samples, seed=config.analysis_seed + 2_325, support="N3", regime="extrapolation"),
    }
    features = {
        name: extract_features(
            condition, model, structured, canonicalizer, dataset, task, config, device
        )
        for name, dataset in datasets.items()
    }
    probe_config = _probe_config(config)
    evaluation_fibers = {name: datasets[name].fiber for name in REGIMES}
    null = fit_cosine_only_null(
        datasets["train"].fiber,
        datasets["validation"].fiber,
        evaluation_fibers,
        probe_config,
        device,
        kind="nonlinear",
        seed=config.analysis_seed + seed * 101,
    )
    cuts: Dict[str, Any] = {}
    for cut_index, cut in enumerate(CUTS):
        probe = fit_conditional_probe(
            features["train"][cut],
            features["validation"][cut],
            {name: features[name][cut] for name in REGIMES},
            datasets["train"].fiber,
            datasets["validation"].fiber,
            evaluation_fibers,
            probe_config,
            device,
            kind="nonlinear",
            seed=config.analysis_seed + seed * 10_007 + cut_index * 1_009,
        )
        _add_log_loss_gains(probe, null)
        cuts[cut] = {"probe": probe}
    task_metrics = evaluate_task_metrics(
        condition, model, structured, canonicalizer, datasets, task, config, device
    )
    return {"cuts": cuts, "task_metrics": task_metrics}


@torch.no_grad()
def evaluate_task_metrics(
    condition: str,
    model: TinyLLMModel,
    structured: Optional[StructuredTinyLLM],
    canonicalizer: AnalyticInvariantCanonicalizer,
    datasets: Mapping[str, PairedCircleDataset],
    task: CircleTaskConfig,
    config: InvariantFrontendConfig,
    device: torch.device,
) -> Dict[str, Any]:
    answer_ids = torch.tensor(task.answer_token_ids, device=device)
    result: Dict[str, Any] = {}
    for regime in REGIMES:
        dataset = datasets[regime]
        sensor = decode_sensor_tokens(dataset.circle.input_ids, task)
        analytic = canonicalizer(sensor) if condition == "analytic_invariant" else None
        probabilities = []
        for start in range(0, len(sensor), config.activation_batch_size):
            stop = start + config.activation_batch_size
            inputs = dataset.circle.input_ids[start:stop].to(device)
            observed = sensor[start:stop].to(device)
            if structured is None:
                residual = _raw_cuts(model, inputs, observed)["full"]
            else:
                fixed = analytic[start:stop].to(device) if analytic is not None else None
                residual = structured.forward_cuts(inputs, observed, fixed)["full"]
            probabilities.append(torch.softmax(_task_logits(model, residual, answer_ids), -1).cpu())
        predicted = torch.cat(probabilities)
        target = dataset.circle.target_posteriors
        predicted_bins = predicted.argmax(1)
        target_bins = target.argmax(1)
        delta = (predicted_bins - target_bins).abs()
        circular = torch.minimum(delta, target.shape[1] - delta)
        result[regime] = {
            "exact_bin_accuracy": float((predicted_bins == target_bins).float().mean()),
            "mean_circular_error_radians": float(circular.float().mean() * (2.0 * math.pi / target.shape[1])),
            "mean_target_cross_entropy": float(-(target * predicted.clamp_min(1e-12).log()).sum(1).mean()),
        }
    return result


def _config_from_mapping(values: Mapping[str, Any]) -> InvariantFrontendConfig:
    converted = dict(values)
    for field in ("seeds", "device_ids"):
        converted[field] = tuple(converted[field])
    return InvariantFrontendConfig(**converted)


def _scientific_fingerprint(experiment: Experiment) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "configuration": experiment.parameters["configuration"],
        "task_config": experiment.parameters["task_config"],
        "condition": experiment.parameters["condition"],
        "seed": experiment.seed,
        "implementation_sha256": experiment.parameters["implementation_sha256"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def invariant_frontend_worker(experiment: Experiment, device_id: int) -> ExperimentResult:
    config = _config_from_mapping(experiment.parameters["configuration"])
    task = CircleTaskConfig(**experiment.parameters["task_config"])
    condition = str(experiment.parameters["condition"])
    seed = int(experiment.seed)
    if _implementation_digest() != experiment.parameters["implementation_sha256"]:
        raise RuntimeError("implementation changed after campaign construction")
    output_dir = _condition_directory(Path(experiment.parameters["output_dir"]), condition, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if device_id < 0 else f"cuda:{device_id}")
    if device.type == "cuda":
        torch.cuda.set_device(device_id)
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    structured: Optional[StructuredTinyLLM] = None
    if condition == "raw_n3":
        baseline = verify_baseline(task, config, seed)
        model = TinyLLMModel.from_checkpoint(baseline["checkpoint"], map_location=device)
        if _state_digest(model) != baseline["final_state_sha256"]:
            raise ValueError(f"retained baseline checkpoint digest mismatch for seed {seed}")
        training = {**baseline, "source": "retained_exact_matched_baseline"}
    else:
        structured, training = train_structured(
            task, config, condition, seed, device, output_dir
        )
        model = structured.model
    analysis = analyze(condition, model, structured, task, config, seed, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    detail = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "experiment_id": experiment.id,
        "status": "completed",
        "completed_at": _utc_now(),
        "condition": condition,
        "seed": seed,
        "device": str(device),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training": training,
        "analysis": analysis,
        "peak_cuda_allocated_gb": peak_allocated,
        "wall_seconds": elapsed,
        "scientific_fingerprint": _scientific_fingerprint(experiment),
        "implementation_sha256": _implementation_digest(),
    }
    detail_path = output_dir / "result.json"
    _write_json(detail_path, detail)
    composition = analysis["cuts"]["full"]["probe"]["evaluations"]["composition"]
    extrapolation = analysis["cuts"]["full"]["probe"]["evaluations"]["extrapolation"]
    metrics = {
        "full_composition_cosine": float(composition["cosine_pearson"]),
        "full_composition_branch": float(composition["balanced_accuracy"]),
        "full_extrapolation_cosine": float(extrapolation["cosine_pearson"]),
        "full_extrapolation_branch": float(extrapolation["balanced_accuracy"]),
        "peak_cuda_allocated_gb": peak_allocated,
    }
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=HYPOTHESIS_ID,
        metrics=metrics,
        primary_metric=min(metrics["full_composition_cosine"], metrics["full_extrapolation_cosine"]),
        model_architecture=[task.vocab_size, *([model.config.n_embd] * model.config.n_layer), task.phase_bins],
        model_parameters=detail["model_parameters"],
        training_time=elapsed,
        training_history=training.get("training_history", []),
        model_checkpoint=training["checkpoint"],
        observations=[f"detail={detail_path}", f"condition={condition}"],
    )


def endpoint_pass(metrics: Mapping[str, Any]) -> bool:
    return bool(metrics["cosine_pearson"] >= 0.90 and metrics["balanced_accuracy"] <= 0.55)


def aggregate_details(
    details: Sequence[Mapping[str, Any]], config: InvariantFrontendConfig
) -> Dict[str, Any]:
    required = 4 if len(config.seeds) >= 5 else len(config.seeds)
    arms: Dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = sorted(
            (item for item in details if item["condition"] == condition), key=lambda item: item["seed"]
        )
        cut_summary: Dict[str, Any] = {}
        for cut in CUTS:
            cut_summary[cut] = {}
            for regime in REGIMES:
                values = [item["analysis"]["cuts"][cut]["probe"]["evaluations"][regime] for item in selected]
                passes = [endpoint_pass(value) for value in values]
                cut_summary[cut][regime] = {
                    "cosine_pearson_mean": float(np.mean([value["cosine_pearson"] for value in values])),
                    "branch_balanced_accuracy_mean": float(np.mean([value["balanced_accuracy"] for value in values])),
                    "conditional_log_loss_gain_mean": float(np.mean([value["conditional_log_loss_gain_over_cosine_only"] for value in values])),
                    "pass_count": int(sum(passes)),
                    "pass_by_seed": {str(item["seed"]): passed for item, passed in zip(selected, passes)},
                }
        joint_by_seed = {}
        for item in selected:
            seed = str(item["seed"])
            cells = {
                f"{cut}:{regime}": cut_summary[cut][regime]["pass_by_seed"][seed]
                for cut in CUTS
                for regime in PRIMARY_REGIMES
            }
            joint_by_seed[seed] = {"cells": cells, "joint_pass": all(cells.values())}
        joint_count = sum(value["joint_pass"] for value in joint_by_seed.values())
        arms[condition] = {
            "cuts": cut_summary,
            "joint_pass_by_seed": joint_by_seed,
            "joint_pass_count": joint_count,
            "success": joint_count >= required,
        }
    analytic_success = arms["analytic_invariant"]["success"]
    learned_success = arms["learned_equivariant"]["success"]
    if analytic_success and learned_success:
        conclusion = "architectural_symmetry_stabilizes_quotient"
    elif analytic_success:
        conclusion = "analytic_attainable_learned_family_or_optimization_insufficient"
    else:
        structured = ("analytic_invariant", "learned_equivariant")
        branch_erased = all(
            arms[arm]["cuts"][cut][regime]["branch_balanced_accuracy_mean"] <= 0.55
            for arm in structured for cut in CUTS for regime in PRIMARY_REGIMES
        )
        base_preserved = any(
            arms[arm]["cuts"][cut][regime]["cosine_pearson_mean"] >= 0.90
            for arm in structured for cut in CUTS for regime in PRIMARY_REGIMES
        )
        conclusion = (
            "current_absolute_cosine_quotient_not_identifiable"
            if branch_erased and not base_preserved
            else "architectural_frontends_do_not_meet_joint_endpoint"
        )
    return {
        "arms": arms,
        "preregistered_gate": {
            "required_seed_passes": required,
            "cosine_threshold": 0.90,
            "branch_accuracy_ceiling": 0.55,
            "cuts": list(CUTS),
            "regimes": list(PRIMARY_REGIMES),
        },
        "conclusion": conclusion,
    }


def _experiments(
    config: InvariantFrontendConfig, task: CircleTaskConfig, output_dir: Path
) -> List[Experiment]:
    common = {
        "configuration": asdict(config),
        "task_config": asdict(task),
        "output_dir": str(output_dir),
        "implementation_sha256": _implementation_digest(),
    }
    return [
        Experiment(
            id=f"tinyllm-invariant-frontend-{condition}-seed{seed}",
            hypothesis_id=HYPOTHESIS_ID,
            name=f"TinyLLM invariant front-end {condition} seed {seed}",
            parameters={**common, "condition": condition, "seed": seed},
            seed=seed,
        )
        for condition in CONDITIONS
        for seed in config.seeds
    ]


def _existing_detail(experiment: Experiment, output_dir: Path) -> Optional[Dict[str, Any]]:
    path = _condition_directory(output_dir, str(experiment.parameters["condition"]), int(experiment.seed)) / "result.json"
    if not path.is_file():
        return None
    try:
        detail = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if (
        detail.get("schema_version") != SCHEMA_VERSION
        or detail.get("status") != "completed"
        or detail.get("scientific_fingerprint") != _scientific_fingerprint(experiment)
    ):
        return None
    checkpoint = Path(detail.get("training", {}).get("checkpoint", ""))
    return detail if checkpoint.is_file() else None


async def run_campaign(
    config: InvariantFrontendConfig, task: CircleTaskConfig, output_dir: Path
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
        project_name="tinyllm_invariant_frontend_causal",
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
    runner = AsyncExperimentRunner(lab, invariant_frontend_worker)
    results = await runner.run_experiments(pending) if pending else []
    successful = {result.experiment_id for result in results if result.error is None}
    details = list(existing.values())
    for experiment in pending:
        if experiment.id in successful:
            path = _condition_directory(output_dir, str(experiment.parameters["condition"]), int(experiment.seed)) / "result.json"
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
            "slots_by_device": {str(key): value for key, value in runner.slot_plan.slots_by_device.items()},
            "calibration": runner.slot_plan.calibration,
        },
        "aggregates": aggregate_details(details, config) if len(details) == len(experiments) else {},
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
            "The current N3 observation has an exact orientation/harmonic-phase gauge ambiguity, so absolute cosine is not exactly identifiable.",
            "The structured vector relation is equivariant to planar orientation and invariant to positive scale, constant/affine drift, and the discarded harmonic channel.",
            "The nonlinear branch probes measure tested conditional decodability, not certified mutual information.",
            "The analytic canonicalizer is observation-only and is a positive control, not a learned result.",
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
    parser.add_argument("--probe-steps", type=int, default=240)
    parser.add_argument("--probe-train-samples", type=int, default=2_048)
    parser.add_argument("--probe-validation-samples", type=int, default=512)
    parser.add_argument("--probe-test-samples", type=int, default=1_024)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--slots-per-gpu", type=int, default=0)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-underpowered", action="store_true")
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument("--baseline-campaign", default="data/experiments/tinyllm_nuisance_support_scaling/20260806_d8_full")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/tinyllm_invariant_frontend_causal/20260806_d8_preregistered"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    preset = "d8"
    if args.shakedown:
        args.seeds = "7"
        args.steps = 2
        args.train_samples = 32
        args.batch_size = 8
        args.probe_steps = 20
        args.probe_train_samples = 64
        args.probe_validation_samples = 32
        args.probe_test_samples = 32
        args.allow_underpowered = True
    config = InvariantFrontendConfig(
        preset=preset,
        seeds=_comma_ints(args.seeds),
        training_steps=args.steps,
        train_samples=args.train_samples,
        batch_size=args.batch_size,
        probe_steps=args.probe_steps,
        probe_train_samples=args.probe_train_samples,
        probe_validation_samples=args.probe_validation_samples,
        probe_test_samples=args.probe_test_samples,
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
    print(json.dumps({"summary": bundle["summary"], "conclusion": bundle.get("aggregates", {}).get("conclusion")}, indent=2, sort_keys=True))
    print(args.output / "campaign_results.json")
    return 0 if bundle["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
