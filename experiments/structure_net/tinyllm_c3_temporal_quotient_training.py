#!/usr/bin/env python3
"""Stage and train an exact observable-C3 temporal TinyLLM experiment.

The public CLI currently exposes systems lifecycle modes only. Primary training
remains deliberately unavailable until a dated numeric preregistration exists.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as preflight
import experiments.structure_net.tinyllm_joint_physical_scalar_interface as joint
from structure_net.components.models import TinyLLMConfig, TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-c3-temporal-quotient-stage0.v1"
HYPOTHESIS_ID = "tinyllm-c3-temporal-quotient-training-v1"
EVIDENCE_ROLE = "systems_lifecycle_only_not_scientific_evidence"
ARMS = ("raw", "analytic", "learned_c3")
STRUCTURED_ARMS = ("analytic", "learned_c3")
REGIMES = ("composition", "extrapolation")
STAGE0_REGISTRATION_PATH = Path(
    "docs/07 - Status Reports/"
    "2026-08-11_tinyllm-c3-temporal-quotient-stage0-registration.md"
)
STAGE0_REGISTRATION_SHA256 = (
    "c50c7fd3c437f3e4e14ddae672d5a5b251ea0b9589c73d717867fd824987e892"
)
PREFLIGHT_IMPLEMENTATION_SHA256 = (
    "812ac0a574bab812ff9fbbea7a0997c71e8ed0c50f98b270b8344234956ec118"
)
TRAINING_RANGES = preflight.REGIME_RANGES["composition"]
BASELINE_CALIBRATION = {"amplitude": 1.2, "offset": 0.0, "drift": 0.0}


@dataclass(frozen=True)
class C3TaskConfig:
    time_steps: int = preflight.TIME_STEPS
    channels: int = preflight.CHANNELS
    phase_bins: int = 16
    vocab_size: int = 50_257
    answer_token_start: int = 32_000
    bos_token_id: int = 0
    query_token_id: int = 1

    def __post_init__(self) -> None:
        if self.time_steps != preflight.TIME_STEPS or self.channels != 3:
            raise ValueError("the registered temporal shape changed")
        if self.phase_bins != 16:
            raise ValueError("the registered answer resolution changed")
        if self.answer_token_start + self.phase_bins > self.vocab_size:
            raise ValueError("answer tokens exceed the vocabulary")

    @property
    def sequence_length(self) -> int:
        return self.time_steps + 2

    @property
    def answer_token_ids(self) -> tuple[int, ...]:
        return tuple(
            range(self.answer_token_start, self.answer_token_start + self.phase_bins)
        )


@dataclass(frozen=True)
class LifecycleConfig:
    preset: str
    steps: int
    split_step: int
    train_samples: int
    batch_size: int
    evaluation_samples: int
    seed: int = 7
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    learned_feature_width: int = 16
    learned_character_channels: int = 8

    def __post_init__(self) -> None:
        if self.preset not in ("tiny", "d6"):
            raise ValueError("Stage 0 permits only tiny CPU or d6 CUDA")
        if not 0 < self.split_step < self.steps:
            raise ValueError("resume split must be inside the lifecycle")
        if self.train_samples < 4 or self.train_samples % 2:
            raise ValueError("training samples must be positive and paired")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("batch size must contain complete pairs")
        if self.evaluation_samples < 2:
            raise ValueError("evaluation sample count is too small")


@dataclass(frozen=True)
class C3TrainingDataset:
    canonical_tokens: torch.Tensor
    tokens: torch.Tensor
    calibration: torch.Tensor
    target: torch.Tensor
    target_posteriors: torch.Tensor
    target_bins: torch.Tensor
    phase: torch.Tensor
    speed: torch.Tensor
    deck: torch.Tensor
    pair_id: torch.Tensor
    saturation_count: int

    def __len__(self) -> int:
        return int(self.tokens.shape[0])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _digest_tensors(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in values:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _update_object_digest(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        digest.update(b"tensor")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: str(item)):
            _update_object_digest(digest, str(key))
            _update_object_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"sequence")
        for item in value:
            _update_object_digest(digest, item)
    elif value is None or isinstance(value, (bool, int, float, str)):
        digest.update(json.dumps(value, sort_keys=True).encode("utf-8"))
    else:
        raise TypeError(f"unsupported digest value {type(value).__name__}")


def _optimizer_digest(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    _update_object_digest(digest, optimizer.state_dict())
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "stage0_registration": STAGE0_REGISTRATION_PATH,
        "generator_preflight": Path(preflight.__file__),
        "interval_likelihood": Path(joint.__file__),
        "tinyllm_model": Path(inspect.getfile(TinyLLMModel)),
    }
    values = {name: _sha256(path) for name, path in paths.items()}
    if values["stage0_registration"] != STAGE0_REGISTRATION_SHA256:
        raise RuntimeError("C3 Stage-0 registration changed")
    if values["generator_preflight"] != PREFLIGHT_IMPLEMENTATION_SHA256:
        raise RuntimeError("pinned C3 generator preflight changed")
    return values


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _uniform(
    count: int, bounds: tuple[float, float], generator: torch.Generator
) -> torch.Tensor:
    return torch.empty(count, dtype=torch.float64).uniform_(
        *bounds, generator=generator
    )


def _targets(target: torch.Tensor, phase_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    posterior = joint.interval_posterior_unclipped(target, phase_bins).float()
    return posterior, posterior.argmax(dim=-1)


def generate_training_dataset(
    task: C3TaskConfig, *, sample_count: int, seed: int
) -> C3TrainingDataset:
    """Generate paired sheets with one calibration nuisance active per latent."""
    if sample_count % 2:
        raise ValueError("training examples must form two-sheet pairs")
    latent_count = sample_count // 2
    generator = torch.Generator(device="cpu").manual_seed(seed)
    phase = 2.0 * math.pi * torch.rand(
        latent_count, dtype=torch.float64, generator=generator
    )
    speed_magnitude = _uniform(latent_count, TRAINING_RANGES["speed"], generator)
    speed_sign = (
        2.0
        * torch.randint(0, 2, (latent_count,), generator=generator).double()
        - 1.0
    )
    speed = speed_magnitude * speed_sign
    nuisance_family = torch.randint(0, 4, (latent_count,), generator=generator)
    amplitude_draw = _uniform(
        latent_count, TRAINING_RANGES["amplitude"], generator
    )
    offset_draw = _uniform(latent_count, TRAINING_RANGES["offset"], generator)
    drift_draw = _uniform(latent_count, TRAINING_RANGES["drift"], generator)
    amplitude = torch.full(
        (latent_count,), BASELINE_CALIBRATION["amplitude"], dtype=torch.float64
    )
    offset = torch.full(
        (latent_count,), BASELINE_CALIBRATION["offset"], dtype=torch.float64
    )
    drift = torch.full(
        (latent_count,), BASELINE_CALIBRATION["drift"], dtype=torch.float64
    )
    amplitude = torch.where(nuisance_family == 1, amplitude_draw, amplitude)
    offset = torch.where(nuisance_family == 2, offset_draw, offset)
    drift = torch.where(nuisance_family == 3, drift_draw, drift)
    continuous = preflight._continuous_observation(
        phase, speed, amplitude, offset, drift
    )
    saturation_count = int(
        (continuous.abs() >= preflight.QUANTIZATION_LIMIT).sum()
    )
    canonical = preflight.quantize(continuous)
    first_deck = torch.randint(0, 3, (latent_count,), generator=generator)
    separation = 1 + torch.randint(0, 2, (latent_count,), generator=generator)
    deck = torch.stack((first_deck, (first_deck + separation) % 3), dim=1)
    observed = torch.stack(
        [
            preflight.apply_deck_action(canonical[index], int(element))
            for index, pair in enumerate(deck)
            for element in pair
        ]
    )
    repeated_canonical = canonical.repeat_interleave(2, dim=0)
    calibration = torch.stack((amplitude, offset, drift), dim=-1).repeat_interleave(
        2, dim=0
    )
    target = torch.cos(3.0 * (phase + task.time_steps * speed)).repeat_interleave(2)
    posterior, bins = _targets(target, task.phase_bins)
    return C3TrainingDataset(
        canonical_tokens=repeated_canonical,
        tokens=observed,
        calibration=calibration,
        target=target,
        target_posteriors=posterior,
        target_bins=bins,
        phase=phase.repeat_interleave(2),
        speed=speed.repeat_interleave(2),
        deck=deck.reshape(-1),
        pair_id=torch.arange(latent_count).repeat_interleave(2),
        saturation_count=saturation_count,
    )


def generate_evaluation_dataset(
    task: C3TaskConfig, regime: str, sample_count: int
) -> C3TrainingDataset:
    source = preflight.generate_dataset(regime)
    selection = slice(0, sample_count)
    target = source.target[selection]
    posterior, bins = _targets(target, task.phase_bins)
    return C3TrainingDataset(
        canonical_tokens=source.base_tokens[selection],
        tokens=source.tokens[selection],
        calibration=source.calibration[selection],
        target=target,
        target_posteriors=posterior,
        target_bins=bins,
        phase=source.phase[selection],
        speed=source.speed[selection],
        deck=source.deck[selection],
        pair_id=torch.arange(sample_count),
        saturation_count=source.saturation_count,
    )


def protocol_material(
    task: C3TaskConfig, config: LifecycleConfig
) -> tuple[C3TrainingDataset, torch.Tensor, str, str]:
    dataset = generate_training_dataset(
        task, sample_count=config.train_samples, seed=config.seed + 1_001
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 6_013)
    pair_batches = torch.randint(
        0,
        config.train_samples // 2,
        (config.steps, config.batch_size // 2),
        generator=generator,
    )
    data_hash = _digest_tensors(
        dataset.canonical_tokens,
        dataset.tokens,
        dataset.calibration,
        dataset.target,
        dataset.target_posteriors,
        dataset.phase,
        dataset.speed,
        dataset.deck,
        dataset.pair_id,
    )
    return dataset, pair_batches, data_hash, _digest_tensors(pair_batches)


def corrected_channels(
    tokens: torch.Tensor, calibration: torch.Tensor
) -> torch.Tensor:
    observed = preflight.decode(tokens).float()
    packet = calibration.float()
    time = torch.arange(
        preflight.TIME_STEPS, dtype=observed.dtype, device=observed.device
    )[None, :, None]
    return (
        observed
        - packet[:, None, 1:2]
        - packet[:, None, 2:3] * time
    ) / packet[:, None, 0:1]


class LearnedC3InvariantEncoder(nn.Module):
    """Exact invariant obtained by cubing a learned first-character carrier."""

    def __init__(self, hidden: int = 16, character_channels: int = 8):
        super().__init__()
        self.shared_map = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, character_channels),
        )
        self.mixer_real = nn.Parameter(torch.empty(character_channels))
        self.mixer_imag = nn.Parameter(torch.empty(character_channels))
        nn.init.normal_(self.mixer_real, std=1.0 / math.sqrt(character_channels))
        nn.init.normal_(self.mixer_imag, std=1.0 / math.sqrt(character_channels))
        angles = 2.0 * math.pi * torch.arange(3, dtype=torch.float32) / 3.0
        self.register_buffer("character_real", torch.cos(-angles))
        self.register_buffer("character_imag", torch.sin(-angles))

    def forward(self, corrected: torch.Tensor) -> torch.Tensor:
        features = self.shared_map(corrected.unsqueeze(-1))
        real = torch.einsum("btck,c->btk", features, self.character_real)
        imaginary = torch.einsum("btck,c->btk", features, self.character_imag)
        mixed_real = torch.einsum("btk,k->bt", real, self.mixer_real) - torch.einsum(
            "btk,k->bt", imaginary, self.mixer_imag
        )
        mixed_imaginary = torch.einsum(
            "btk,k->bt", real, self.mixer_imag
        ) + torch.einsum("btk,k->bt", imaginary, self.mixer_real)
        magnitude = torch.sqrt(mixed_real.square() + mixed_imaginary.square())
        denominator = magnitude.clamp_min(1e-6)
        x = mixed_real / denominator
        y = mixed_imaginary / denominator
        return torch.stack(
            (x.pow(3) - 3.0 * x * y.square(), 3.0 * x.square() * y - y.pow(3)),
            dim=-1,
        )


class C3TemporalTinyLLM(nn.Module):
    def __init__(
        self,
        model: TinyLLMModel,
        arm: str,
        *,
        learned_feature_width: int = 16,
        learned_character_channels: int = 8,
    ):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown C3 arm {arm!r}")
        self.model = model
        self.arm = arm
        feature_width = 3 if arm == "raw" else 2
        # Construct the shared structured injection before the learned encoder
        # so analytic and learned arms consume the same initialization stream.
        self.sequence_embedding = nn.Linear(feature_width, model.config.n_embd)
        self.encoder: LearnedC3InvariantEncoder | None = None
        if arm == "learned_c3":
            self.encoder = LearnedC3InvariantEncoder(
                learned_feature_width, learned_character_channels
            )

    def feature(
        self, tokens: torch.Tensor, calibration: torch.Tensor
    ) -> torch.Tensor:
        corrected = corrected_channels(tokens, calibration)
        if self.arm == "raw":
            return corrected
        if self.arm == "analytic":
            carrier, _ = preflight.analytic_carrier(
                tokens.detach().cpu(), calibration.detach().cpu()
            )
            return torch.stack((carrier.real, carrier.imag), dim=-1).float().to(
                tokens.device
            )
        if self.encoder is None:
            raise AssertionError("learned C3 encoder missing")
        return self.encoder(corrected)

    def residual_cuts(
        self, tokens: torch.Tensor, calibration: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        feature = self.feature(tokens, calibration)
        batch = tokens.shape[0]
        bos_ids = torch.full(
            (batch, 1), 0, dtype=torch.long, device=tokens.device
        )
        query_ids = torch.full(
            (batch, 1), 1, dtype=torch.long, device=tokens.device
        )
        bos = self.model.transformer["wte"](bos_ids)
        sequence = self.sequence_embedding(feature)
        query = self.model.transformer["wte"](query_ids)
        value = torch.cat((bos, sequence, query), dim=1)
        positions = torch.arange(value.shape[1], device=value.device)
        frontend = value + self.model.transformer["wpe"](positions)
        block = self.model.transformer["h"][0]
        attended = frontend + block.attn(block.ln_1(frontend))
        post_mlp = attended + block.mlp(block.ln_2(attended))
        final = post_mlp
        for later in self.model.transformer["h"][1:]:
            final = later(final)
        return {
            "feature": feature,
            "frontend": frontend,
            "post_attention": attended,
            "post_mlp": post_mlp,
            "full": final,
        }

    def task_logits(
        self,
        tokens: torch.Tensor,
        calibration: torch.Tensor,
        answer_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.residual_cuts(tokens, calibration)["full"][:, -1, :]
        logits = self.model.lm_head(self.model.transformer["ln_f"](residual))
        return logits.index_select(-1, answer_ids)


def _model_config(preset: str, task: C3TaskConfig, seed: int) -> TinyLLMConfig:
    if preset == "tiny":
        return TinyLLMConfig(
            block_size=task.sequence_length,
            vocab_size=task.vocab_size,
            n_layer=2,
            n_head=2,
            n_embd=32,
            initialization_seed=seed,
        )
    return TinyLLMConfig.from_preset(
        preset,
        block_size=task.sequence_length,
        vocab_size=task.vocab_size,
        initialization_seed=seed,
    )


def build_system(
    task: C3TaskConfig,
    config: LifecycleConfig,
    arm: str,
    device: torch.device,
) -> C3TemporalTinyLLM:
    torch.manual_seed(config.seed + 91_001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + 91_001)
    model = TinyLLMModel(
        _model_config(config.preset, task, config.seed),
        name=f"C3Temporal_{config.preset}_{arm}_{config.seed}",
    )
    return C3TemporalTinyLLM(
        model,
        arm,
        learned_feature_width=config.learned_feature_width,
        learned_character_channels=config.learned_character_channels,
    ).to(device)


def parameter_counts(system: C3TemporalTinyLLM) -> dict[str, int]:
    model_count = sum(value.numel() for value in system.model.parameters())
    injection_count = sum(
        value.numel() for value in system.sequence_embedding.parameters()
    )
    encoder_count = (
        sum(value.numel() for value in system.encoder.parameters())
        if system.encoder is not None
        else 0
    )
    total = sum(value.numel() for value in system.parameters())
    if model_count + injection_count + encoder_count != total:
        raise RuntimeError("parameter accounting does not close")
    return {
        "tinyllm": model_count,
        "sequence_injection": injection_count,
        "learned_encoder": encoder_count,
        "total": total,
    }


def _feature_contract(
    system: C3TemporalTinyLLM,
    dataset: C3TrainingDataset,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    system.eval()
    with torch.no_grad():
        tokens = dataset.tokens[:32].to(next(system.parameters()).device)
        calibration = dataset.calibration[:32].to(tokens.device)
        baseline = system.feature(tokens, calibration)
        cells = []
        for element in range(3):
            transformed = system.feature(
                preflight.apply_deck_action(tokens, element), calibration
            )
            cells.append(
                {
                    "element": element,
                    "maximum_absolute_change": float(
                        (transformed - baseline).abs().max()
                    ),
                }
            )
    maximum_nonidentity = max(
        item["maximum_absolute_change"] for item in cells if item["element"]
    )
    passed = (
        maximum_nonidentity > 1e-3
        if system.arm == "raw"
        else maximum_nonidentity <= tolerance
    )
    return {
        "arm": system.arm,
        "cells": cells,
        "maximum_nonidentity_change": maximum_nonidentity,
        "tolerance": tolerance,
        "expected": "sheet_visible" if system.arm == "raw" else "invariant",
        "pass": passed,
    }


def learned_registered_state_contract(
    task: C3TaskConfig,
    config: LifecycleConfig,
    dataset: C3TrainingDataset,
    device: torch.device,
) -> dict[str, Any]:
    system = build_system(task, config, "learned_c3", device)
    initial = _feature_contract(system, dataset)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 92_003)
    with torch.no_grad():
        if system.encoder is None:
            raise AssertionError("learned encoder missing")
        for parameter in system.encoder.parameters():
            noise = torch.randn(
                parameter.shape, dtype=parameter.dtype, generator=generator
            ).to(parameter.device)
            parameter.add_(0.05 * noise)
    perturbed = _feature_contract(system, dataset)
    del system
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "initial": initial,
        "deterministic_perturbed": perturbed,
        "pass": initial["pass"] and perturbed["pass"],
    }


def validate_training_protocol(dataset: C3TrainingDataset) -> dict[str, Any]:
    pair_target_error = float(
        (dataset.target[0::2] - dataset.target[1::2]).abs().max()
    )
    pair_calibration_error = float(
        (dataset.calibration[0::2] - dataset.calibration[1::2]).abs().max()
    )
    distinct_decks = bool(torch.all(dataset.deck[0::2] != dataset.deck[1::2]))
    canonical_action_errors = 0
    for index in range(len(dataset)):
        expected = preflight.apply_deck_action(
            dataset.canonical_tokens[index], int(dataset.deck[index])
        )
        canonical_action_errors += int((expected != dataset.tokens[index]).sum())
    latent_count = len(dataset) // 2
    derangement = torch.roll(torch.arange(latent_count), shifts=latent_count // 2)
    deranged_target = dataset.target[2 * derangement]
    source_target = dataset.target[0::2]
    exact_target_fixed_points = int((deranged_target == source_target).sum())
    mean_target_change = float((deranged_target - source_target).abs().mean())
    pass_value = (
        dataset.saturation_count == 0
        and pair_target_error == 0.0
        and pair_calibration_error == 0.0
        and distinct_decks
        and canonical_action_errors == 0
        and exact_target_fixed_points == 0
        and mean_target_change >= 0.25
    )
    return {
        "quantizer_saturation_count": dataset.saturation_count,
        "maximum_pair_target_error": pair_target_error,
        "maximum_pair_calibration_error": pair_calibration_error,
        "all_pairs_use_distinct_decks": distinct_decks,
        "canonical_action_token_error_count": canonical_action_errors,
        "target_derangement_exact_fixed_points": exact_target_fixed_points,
        "target_derangement_mean_absolute_change": mean_target_change,
        "pass": pass_value,
    }


def _optimizer(
    system: C3TemporalTinyLLM, config: LifecycleConfig
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        system.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _training_indices(pair_batch: torch.Tensor) -> torch.Tensor:
    return torch.stack((2 * pair_batch, 2 * pair_batch + 1), dim=-1).reshape(-1)


def train_steps(
    system: C3TemporalTinyLLM,
    optimizer: torch.optim.AdamW,
    dataset: C3TrainingDataset,
    pair_batches: torch.Tensor,
    task: C3TaskConfig,
    config: LifecycleConfig,
    device: torch.device,
    start_step: int,
    stop_step: int,
) -> list[dict[str, float]]:
    answer = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    history = []
    system.train()
    for step in range(start_step, stop_step):
        indices = _training_indices(pair_batches[step])
        optimizer.zero_grad(set_to_none=True)
        logits = system.task_logits(
            dataset.tokens[indices].to(device),
            dataset.calibration[indices].to(device),
            answer,
        )
        targets = dataset.target_posteriors[indices].to(device)
        loss = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            system.parameters(), config.gradient_clip
        )
        optimizer.step()
        history.append(
            {
                "step": float(step + 1),
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
            }
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return history


@torch.no_grad()
def evaluate(
    system: C3TemporalTinyLLM,
    dataset: C3TrainingDataset,
    task: C3TaskConfig,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[dict[str, Any], torch.Tensor]:
    system.eval()
    answer = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    probabilities = []
    for start in range(0, len(dataset), batch_size):
        stop = start + batch_size
        logits = system.task_logits(
            dataset.tokens[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
            answer,
        )
        probabilities.append(torch.softmax(logits, dim=-1).cpu())
    posterior = torch.cat(probabilities)
    centers = torch.linspace(-1.0, 1.0, task.phase_bins)
    predicted = posterior @ centers
    target = dataset.target.float()
    centered_prediction = predicted - predicted.mean()
    centered_target = target - target.mean()
    correlation = float(
        (centered_prediction * centered_target).sum()
        / (
            torch.linalg.vector_norm(centered_prediction)
            * torch.linalg.vector_norm(centered_target)
        ).clamp_min(1e-12)
    )
    slope = float(
        (centered_prediction * centered_target).sum()
        / centered_target.square().sum().clamp_min(1e-12)
    )
    predicted_bins = posterior.argmax(dim=-1)
    predicted_angle = torch.acos(predicted.clamp(-1.0, 1.0))
    target_angle = torch.acos(target.clamp(-1.0, 1.0))
    metrics = {
        "exact_bin_accuracy": float(
            (predicted_bins == dataset.target_bins).float().mean()
        ),
        "target_cross_entropy": float(
            -(
                dataset.target_posteriors
                * posterior.clamp_min(1e-12).log()
            ).sum(-1).mean()
        ),
        "posterior_mean_correlation": correlation,
        "posterior_mean_rmse": float(
            torch.sqrt((predicted - target).square().mean())
        ),
        "mean_triple_angle_error_radians": float(
            (predicted_angle - target_angle).abs().mean()
        ),
        "predicted_bin_coverage": int(torch.unique(predicted_bins).numel()),
        "calibration_slope": slope,
    }
    return metrics, posterior


def save_training_checkpoint(
    path: Path,
    system: C3TemporalTinyLLM,
    optimizer: torch.optim.AdamW,
    *,
    arm: str,
    config: LifecycleConfig,
    step: int,
    data_hash: str,
    batch_hash: str,
    sources: Mapping[str, str],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_role": EVIDENCE_ROLE,
        "arm": arm,
        "configuration": asdict(config),
        "step": step,
        "data_sha256": data_hash,
        "minibatch_sha256": batch_hash,
        "source_hashes": dict(sources),
        "system_state": system.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
    torch.save(payload, path)
    return {"path": str(path), "sha256": _sha256(path)}


def load_training_checkpoint(
    path: Path,
    system: C3TemporalTinyLLM,
    optimizer: torch.optim.AdamW,
    *,
    arm: str,
    config: LifecycleConfig,
    data_hash: str,
    batch_hash: str,
    sources: Mapping[str, str],
    device: torch.device,
) -> int:
    payload = torch.load(path, map_location=device, weights_only=True)
    expected = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_role": EVIDENCE_ROLE,
        "arm": arm,
        "configuration": asdict(config),
        "data_sha256": data_hash,
        "minibatch_sha256": batch_hash,
        "source_hashes": dict(sources),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"checkpoint provenance mismatch for {key}")
    system.load_state_dict(payload["system_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["cpu_rng_state"].cpu())
    if device.type == "cuda" and payload["cuda_rng_state"]:
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in payload["cuda_rng_state"]]
        )
    return int(payload["step"])


def exact_resume_lifecycle(
    task: C3TaskConfig,
    config: LifecycleConfig,
    arm: str,
    device: torch.device,
    root: Path,
    sources: Mapping[str, str],
) -> dict[str, Any]:
    dataset, pair_batches, data_hash, batch_hash = protocol_material(task, config)
    evaluations = {
        regime: generate_evaluation_dataset(task, regime, config.evaluation_samples)
        for regime in REGIMES
    }

    continuous = build_system(task, config, arm, device)
    initial_model = _state_digest(continuous.model)
    initial_injection = _state_digest(continuous.sequence_embedding)
    initial_system = _state_digest(continuous)
    counts = parameter_counts(continuous)
    feature_before = _feature_contract(continuous, evaluations["composition"])
    continuous_optimizer = _optimizer(continuous, config)
    continuous_history = train_steps(
        continuous,
        continuous_optimizer,
        dataset,
        pair_batches,
        task,
        config,
        device,
        0,
        config.steps,
    )
    continuous_state = _state_digest(continuous)
    continuous_optimizer_state = _optimizer_digest(continuous_optimizer)
    continuous_metrics: dict[str, Any] = {}
    continuous_posteriors = {}
    for regime, evaluation in evaluations.items():
        metrics, posterior = evaluate(continuous, evaluation, task, device)
        continuous_metrics[regime] = metrics
        continuous_posteriors[regime] = posterior
    del continuous_optimizer, continuous
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    split = build_system(task, config, arm, device)
    split_optimizer = _optimizer(split, config)
    split_history = train_steps(
        split,
        split_optimizer,
        dataset,
        pair_batches,
        task,
        config,
        device,
        0,
        config.split_step,
    )
    checkpoint_path = root / arm / "resume.pt"
    artifact = save_training_checkpoint(
        checkpoint_path,
        split,
        split_optimizer,
        arm=arm,
        config=config,
        step=config.split_step,
        data_hash=data_hash,
        batch_hash=batch_hash,
        sources=sources,
    )
    del split_optimizer, split
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    resumed = build_system(task, config, arm, device)
    resumed_optimizer = _optimizer(resumed, config)
    loaded_step = load_training_checkpoint(
        checkpoint_path,
        resumed,
        resumed_optimizer,
        arm=arm,
        config=config,
        data_hash=data_hash,
        batch_hash=batch_hash,
        sources=sources,
        device=device,
    )
    resumed_history = train_steps(
        resumed,
        resumed_optimizer,
        dataset,
        pair_batches,
        task,
        config,
        device,
        loaded_step,
        config.steps,
    )
    resumed_state = _state_digest(resumed)
    resumed_optimizer_state = _optimizer_digest(resumed_optimizer)
    posterior_errors = {}
    resumed_metrics = {}
    for regime, evaluation in evaluations.items():
        metrics, posterior = evaluate(resumed, evaluation, task, device)
        resumed_metrics[regime] = metrics
        posterior_errors[regime] = float(
            (posterior - continuous_posteriors[regime]).abs().max()
        )
    feature_after = _feature_contract(resumed, evaluations["composition"])
    history = split_history + resumed_history
    finite_history = all(
        math.isfinite(item["loss"]) and math.isfinite(item["gradient_norm"])
        for item in history
    )
    exact = {
        "system_state": resumed_state == continuous_state,
        "optimizer_state": resumed_optimizer_state == continuous_optimizer_state,
        "loss_history": history == continuous_history,
        "evaluation_posteriors": all(value == 0.0 for value in posterior_errors.values()),
    }
    passed = (
        all(exact.values())
        and finite_history
        and continuous_state != initial_system
        and feature_before["pass"]
        and feature_after["pass"]
    )
    result = {
        "arm": arm,
        "configuration": asdict(config),
        "data_sha256": data_hash,
        "minibatch_sha256": batch_hash,
        "initial_tinyllm_sha256": initial_model,
        "initial_injection_sha256": initial_injection,
        "initial_system_sha256": initial_system,
        "final_system_sha256": resumed_state,
        "final_optimizer_sha256": resumed_optimizer_state,
        "parameter_counts": counts,
        "history": history,
        "metrics": resumed_metrics,
        "continuous_metrics": continuous_metrics,
        "feature_contract_before": feature_before,
        "feature_contract_after": feature_after,
        "resume_checkpoint": artifact,
        "loaded_step": loaded_step,
        "maximum_posterior_resume_error": posterior_errors,
        "exact_resume": exact,
        "finite_history": finite_history,
        "nonzero_parameter_change": continuous_state != initial_system,
        "pass": passed,
    }
    del resumed_optimizer, resumed
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _lifecycle_task(mode: str) -> tuple[C3TaskConfig, LifecycleConfig]:
    if mode == "cpu":
        return (
            C3TaskConfig(vocab_size=256, answer_token_start=32),
            LifecycleConfig(
                preset="tiny",
                steps=2,
                split_step=1,
                train_samples=64,
                batch_size=8,
                evaluation_samples=96,
            ),
        )
    return (
        C3TaskConfig(),
        LifecycleConfig(
            preset="d6",
            steps=64,
            split_step=32,
            train_samples=512,
            batch_size=64,
            evaluation_samples=512,
        ),
    )


def _arm_initialization_contract(
    records: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    model_hashes = {arm: record["initial_tinyllm_sha256"] for arm, record in records.items()}
    injection_hashes = {
        arm: records[arm]["initial_injection_sha256"] for arm in STRUCTURED_ARMS
        if arm in records
    }
    model_match = len(set(model_hashes.values())) == 1
    structured_injection_match = len(injection_hashes) < 2 or len(set(injection_hashes.values())) == 1
    return {
        "tinyllm_by_arm": model_hashes,
        "structured_injection_by_arm": injection_hashes,
        "tinyllm_match": model_match,
        "structured_injection_match": structured_injection_match,
        "pass": model_match and structured_injection_match,
    }


def run_cpu_stage0(output: Path) -> dict[str, Any]:
    sources = _source_hashes()
    preflight_record = preflight.build_preflight()
    task, config = _lifecycle_task("cpu")
    dataset, _, data_hash, batch_hash = protocol_material(task, config)
    protocol = validate_training_protocol(dataset)
    registered_learned = learned_registered_state_contract(
        task, config, dataset, torch.device("cpu")
    )
    cells = {
        arm: exact_resume_lifecycle(
            task, config, arm, torch.device("cpu"), output.parent / "stage0_artifacts_cpu", sources
        )
        for arm in ARMS
    }
    initialization = _arm_initialization_contract(cells)
    matched_hashes = (
        {record["data_sha256"] for record in cells.values()} == {data_hash}
        and {record["minibatch_sha256"] for record in cells.values()} == {batch_hash}
    )
    gates = {
        "pinned_preflight": preflight_record["classification"]
        == "c3_temporal_quotient_preflight_passed",
        "training_protocol": protocol["pass"],
        "learned_two_state_invariance": registered_learned["pass"],
        "matched_data_and_minibatches": matched_hashes,
        "matched_initialization": initialization["pass"],
        "all_cpu_lifecycles": all(record["pass"] for record in cells.values()),
        "no_primary_cells": True,
    }
    passed = all(gates.values())
    record = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_role": EVIDENCE_ROLE,
        "status": "completed" if passed else "invalid",
        "completed_at": _utc_now(),
        "classification": (
            "c3_stage0_cpu_passed_pending_cuda"
            if passed
            else "c3_temporal_quotient_stage0_invalid"
        ),
        "mode": "cpu",
        "source_hashes": sources,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "task": asdict(task),
        "configuration": asdict(config),
        "preflight_classification": preflight_record["classification"],
        "training_protocol": protocol,
        "learned_registered_state_contract": registered_learned,
        "initialization_contract": initialization,
        "cells": cells,
        "gates": gates,
        "optimizer_steps_executed": len(ARMS) * config.steps * 2,
        "primary_optimizer_steps_executed": 0,
        "method_boundaries": [
            "All outcomes are systems lifecycle checks, not model-quality evidence.",
            "The tiny CPU preset validates code paths but is not a primary architecture.",
            "CUDA analytic validation and a numeric primary preregistration remain required.",
        ],
    }
    if not _finite(record):
        raise RuntimeError("non-finite CPU Stage-0 record")
    _write_json(output, record)
    return record


def run_cuda_stage0(output: Path, cpu_result: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Stage 0 requested but CUDA is unavailable")
    sources = _source_hashes()
    cpu = json.loads(cpu_result.read_text(encoding="utf-8"))
    if (
        cpu.get("classification") != "c3_stage0_cpu_passed_pending_cuda"
        or cpu.get("source_hashes") != sources
    ):
        raise RuntimeError("matching passing CPU Stage-0 artifact is required")
    task, config = _lifecycle_task("cuda")
    device = torch.device("cuda:0")
    cell = exact_resume_lifecycle(
        task,
        config,
        "analytic",
        device,
        output.parent / "stage0_artifacts_cuda",
        sources,
    )
    gates = {
        "cpu_stage0": True,
        "analytic_cuda_exact_resume": cell["pass"],
        "analytic_carrier_before": cell["feature_contract_before"]["pass"],
        "analytic_carrier_after": cell["feature_contract_after"]["pass"],
        "no_raw_or_learned_d6_cells": True,
        "no_primary_cells": True,
    }
    passed = all(gates.values())
    record = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "evidence_role": EVIDENCE_ROLE,
        "status": "completed" if passed else "invalid",
        "completed_at": _utc_now(),
        "classification": (
            "c3_temporal_quotient_stage0_passed"
            if passed
            else "c3_temporal_quotient_stage0_invalid"
        ),
        "mode": "cuda",
        "source_hashes": sources,
        "cpu_stage0": {
            "path": str(cpu_result),
            "sha256": _sha256(cpu_result),
            "classification": cpu["classification"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
        },
        "task": asdict(task),
        "configuration": asdict(config),
        "analytic_cuda_cell": cell,
        "gates": gates,
        "optimizer_steps_executed": config.steps * 2,
        "primary_optimizer_steps_executed": 0,
        "method_boundaries": [
            "This d6 analytic run is an underpowered systems shakedown.",
            "Its metrics may anchor prospective thresholds but are not scientific evidence.",
            "No primary raw, analytic, or learned seed has run.",
        ],
    }
    if not _finite(record):
        raise RuntimeError("non-finite CUDA Stage-0 record")
    _write_json(output, record)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/tinyllm-c3-temporal-quotient-stage0.json"),
    )
    parser.add_argument(
        "--cpu-result",
        type=Path,
        default=Path("/tmp/tinyllm-c3-temporal-quotient-stage0-cpu.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.use_deterministic_algorithms(True)
    if args.stage0 == "cpu":
        record = run_cpu_stage0(args.output)
    else:
        record = run_cuda_stage0(args.output, args.cpu_result)
    print(
        json.dumps(
            {
                "status": record["status"],
                "classification": record["classification"],
                "evidence_role": record["evidence_role"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if record["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
