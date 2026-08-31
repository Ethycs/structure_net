#!/usr/bin/env python3
"""Exercise CALM-style conversion through the Structure Net component lifecycle.

The run trains a tiny source TinyLLM, copies its backbone into a continuous
autoregressive model, executes both CALM training stages, generates one chunk,
and verifies exact checkpoint restoration.  It is systems evidence only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Optional

import torch

import structure_net.components.models.calm_tinyllm_model as calm_model_module
import structure_net.components.trainers.continuous_autoregressive_trainer as calm_trainer_module
import structure_net.components.models.tinyllm_model as tinyllm_model_module
from structure_net.components.models import (
    CALMTinyLLMConfig,
    CALMTinyLLMModel,
    TinyLLMConfig,
    TinyLLMModel,
)
from structure_net.components.trainers import (
    CausalLanguageModelTrainer,
    ContinuousAutoregressiveTrainer,
)
from structure_net.core import EvolutionContext


SCHEMA_VERSION = "nal.tinyllm-calm-framework-shakedown.v1"
HYPOTHESIS_ID = "tinyllm-calm-framework-lifecycle-v1"
CLAIM_SCOPE = "systems_lifecycle_only_not_quality_evidence"


@dataclass(frozen=True)
class ShakedownConfig:
    seed: int = 7
    device: str = "cpu"
    vocab_size: int = 32
    token_sequence_length: int = 12
    batch_size: int = 16
    patch_size: int = 2
    backbone_layers: int = 2
    backbone_heads: int = 2
    backbone_width: int = 32
    latent_size: int = 8
    autoencoder_hidden_size: int = 32
    autoencoder_intermediate_size: int = 64
    autoencoder_dropout: float = 0.1
    generator_layers: int = 1
    noise_size: int = 8
    energy_samples: int = 4
    target_samples: int = 8
    source_training_steps: int = 4
    autoencoder_training_steps: int = 32
    energy_training_steps: int = 16
    source_learning_rate: float = 0.01
    autoencoder_learning_rate: float = 0.01
    energy_learning_rate: float = 0.003

    def __post_init__(self) -> None:
        if self.token_sequence_length % self.patch_size:
            raise ValueError("token_sequence_length must be divisible by patch_size")
        if self.token_sequence_length < 2 * self.patch_size:
            raise ValueError("the energy stage needs at least two complete chunks")
        if self.backbone_width % self.backbone_heads:
            raise ValueError("backbone_width must be divisible by backbone_heads")
        if self.autoencoder_hidden_size != self.backbone_width:
            raise ValueError("the exercise requires matching embedding widths for warm start")
        if min(
            self.source_training_steps,
            self.autoencoder_training_steps,
            self.energy_training_steps,
        ) <= 0:
            raise ValueError("all lifecycle stages require at least one optimizer step")
        if torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("device must be CPU or CUDA")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _scientific_fingerprint(config: ShakedownConfig) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "configuration": asdict(config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


MappingLike = Dict[str, Any]


def _batches(config: ShakedownConfig) -> List[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 101)
    count = max(
        config.source_training_steps,
        config.autoencoder_training_steps,
        config.energy_training_steps,
    )
    positions = torch.arange(config.token_sequence_length)
    batches = []
    for _ in range(count):
        start = torch.randint(
            1,
            config.vocab_size,
            (config.batch_size, 1),
            generator=generator,
        )
        stride = torch.randint(1, 4, (config.batch_size, 1), generator=generator)
        batches.append((start + stride * positions) % config.vocab_size)
    return batches


def _build_configs(config: ShakedownConfig) -> tuple[TinyLLMConfig, CALMTinyLLMConfig]:
    backbone = TinyLLMConfig(
        block_size=config.token_sequence_length,
        vocab_size=config.vocab_size,
        n_layer=config.backbone_layers,
        n_head=config.backbone_heads,
        n_embd=config.backbone_width,
        initialization_seed=config.seed,
    )
    calm = CALMTinyLLMConfig(
        backbone=backbone,
        patch_size=config.patch_size,
        latent_size=config.latent_size,
        autoencoder_hidden_size=config.autoencoder_hidden_size,
        autoencoder_intermediate_size=config.autoencoder_intermediate_size,
        autoencoder_dropout=config.autoencoder_dropout,
        generator_layers=config.generator_layers,
        noise_size=config.noise_size,
        energy_samples=config.energy_samples,
        target_samples=config.target_samples,
        initialization_seed=config.seed + 1,
    )
    return backbone, calm


@torch.no_grad()
def _autoencoder_metrics(model: CALMTinyLLMModel, batch: torch.Tensor) -> Dict[str, float]:
    model.autoencoder.eval()
    output = model.autoencoder(batch)
    predictions = output.logits.argmax(dim=-1).reshape_as(batch)
    chunks = batch.reshape(batch.shape[0], -1, model.config.patch_size)
    predicted_chunks = predictions.reshape_as(chunks)
    return {
        "loss": float(output.loss.detach().cpu()),
        "reconstruction_loss": float(output.reconstruction_loss.detach().cpu()),
        "kl_loss": float(output.kl_loss.detach().cpu()),
        "token_accuracy": float((predictions == batch).float().mean().cpu()),
        "exact_chunk_accuracy": float(
            (predicted_chunks == chunks).all(dim=-1).float().mean().cpu()
        ),
    }


def _fixed_noises(
    model: CALMTinyLLMModel, batch: torch.Tensor, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chunk_predictions = batch.shape[1] // model.config.patch_size - 1
    prediction_noise = torch.rand(
        (
            model.config.energy_samples,
            batch.shape[0],
            chunk_predictions,
            model.config.noise_size,
        ),
        generator=generator,
    ) - 0.5
    target_noise = torch.randn(
        (
            model.config.target_samples,
            batch.shape[0],
            chunk_predictions,
            model.config.latent_size,
        ),
        generator=generator,
    )
    return prediction_noise.to(batch.device), target_noise.to(batch.device)


@torch.no_grad()
def _energy_metrics(
    model: CALMTinyLLMModel,
    batch: torch.Tensor,
    prediction_noise: torch.Tensor,
    target_noise: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    output = model(batch, noise=prediction_noise, target_noise=target_noise)
    return {
        "loss": float(output.loss.detach().cpu()),
        "prediction_mean_norm": float(
            output.latent_predictions.mean(dim=0).norm(dim=-1).mean().cpu()
        ),
        "target_mean_norm": float(output.target_mean.norm(dim=-1).mean().cpu()),
    }


def _environment(device: torch.device) -> Dict[str, Any]:
    value: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        value.update(
            {
                "cuda_runtime": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return value


def run_shakedown(config: ShakedownConfig, output_dir: Path) -> Dict[str, Any]:
    device = torch.device(config.device)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable to PyTorch")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("requested CUDA ordinal is unavailable")

    run_dir = output_dir / "runs" / "framework_lifecycle" / f"seed_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    previous_threads = torch.get_num_threads()
    previous_rng = torch.random.get_rng_state()
    previous_cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    started = time.perf_counter()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
            torch.cuda.reset_peak_memory_stats(device)

        backbone_config, calm_config = _build_configs(config)
        batches = _batches(config)

        source = TinyLLMModel(backbone_config, name="CALMConversionSource").to(device)
        source_trainer = CausalLanguageModelTrainer(
            optimizer_kwargs={"lr": config.source_learning_rate, "weight_decay": 0.0}
        )
        source_history = [
            source_trainer.train_step(
                source,
                batches[index],
                EvolutionContext(device=str(device)),
            )
            for index in range(config.source_training_steps)
        ]
        source.to("cpu")
        source_state = {
            name: tensor.detach().clone() for name, tensor in source.state_dict().items()
        }
        model = CALMTinyLLMModel.from_tinyllm(source, calm_config).to(device)
        copied_exactly = all(
            torch.equal(value, model.backbone.state_dict()[name].detach().cpu())
            for name, value in source_state.items()
        )

        evaluation_batch = batches[0].to(device)
        ae_before = _autoencoder_metrics(model, evaluation_batch)
        ae_trainer = ContinuousAutoregressiveTrainer(
            phase="autoencoder",
            optimizer_kwargs={
                "lr": config.autoencoder_learning_rate,
                "weight_decay": 0.0,
            },
        )
        ae_history = [
            ae_trainer.train_step(
                model,
                batches[index].to(device),
                EvolutionContext(device=str(device)),
            )
            for index in range(config.autoencoder_training_steps)
        ]
        ae_after = _autoencoder_metrics(model, evaluation_batch)

        energy_trainer = ContinuousAutoregressiveTrainer(
            phase="energy",
            optimizer_kwargs={"lr": config.energy_learning_rate, "weight_decay": 0.0},
        )
        energy_trainer.attach(model)
        autoencoder_digest_before_energy = _state_digest(model.autoencoder)
        fixed_prediction_noise, fixed_target_noise = _fixed_noises(
            model, evaluation_batch, config.seed + 303
        )
        energy_before = _energy_metrics(
            model,
            evaluation_batch,
            fixed_prediction_noise,
            fixed_target_noise,
        )
        energy_history = [
            energy_trainer.train_step(
                model,
                batches[index].to(device),
                EvolutionContext(device=str(device)),
            )
            for index in range(config.energy_training_steps)
        ]
        energy_after = _energy_metrics(
            model,
            evaluation_batch,
            fixed_prediction_noise,
            fixed_target_noise,
        )
        autoencoder_digest_after_energy = _state_digest(model.autoencoder)

        generation_noise = torch.rand(
            (1, evaluation_batch.shape[0], calm_config.noise_size),
            generator=torch.Generator(device="cpu").manual_seed(config.seed + 404),
        ).to(device) - 0.5
        generated_chunk = model.generate_next_chunk(
            evaluation_batch[:, : 2 * config.patch_size], noise=generation_noise
        )

        peak_memory = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        )
        model.to("cpu")
        evaluation_batch_cpu = evaluation_batch.cpu()
        fixed_prediction_noise_cpu = fixed_prediction_noise.cpu()
        fixed_target_noise_cpu = fixed_target_noise.cpu()
        expected = model(
            evaluation_batch_cpu,
            noise=fixed_prediction_noise_cpu,
            target_noise=fixed_target_noise_cpu,
        )
        model_digest = _state_digest(model)
        checkpoint = run_dir / "model.pt"
        model.save_checkpoint(
            checkpoint,
            metadata={
                "schema_version": SCHEMA_VERSION,
                "hypothesis_id": HYPOTHESIS_ID,
                "claim_scope": CLAIM_SCOPE,
            },
        )
        restored = CALMTinyLLMModel.from_checkpoint(checkpoint).eval()
        restored_output = restored(
            evaluation_batch_cpu,
            noise=fixed_prediction_noise_cpu,
            target_noise=fixed_target_noise_cpu,
        )
        checkpoint_loss_delta = float(
            (expected.loss - restored_output.loss).abs().detach().cpu()
        )
        checkpoint_prediction_delta = float(
            (
                expected.latent_predictions - restored_output.latent_predictions
            ).abs().max().detach().cpu()
        )
        restored_digest = _state_digest(restored)

        source_files = {
            "experiment": Path(__file__).resolve(),
            "calm_model": Path(calm_model_module.__file__).resolve(),
            "calm_trainer": Path(calm_trainer_module.__file__).resolve(),
            "tinyllm_model": Path(tinyllm_model_module.__file__).resolve(),
        }
        implementation = {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in source_files.items()
        }
        lifecycle_gates = {
            "source_backbone_tensors_copied_exactly": copied_exactly,
            "autoencoder_backward_finite_nonzero": all(
                item["grad_norm"] > 0.0 for item in ae_history
            ),
            "energy_backward_finite_nonzero": all(
                item["grad_norm"] > 0.0 for item in energy_history
            ),
            "autoencoder_frozen_during_energy": (
                autoencoder_digest_before_energy == autoencoder_digest_after_energy
            ),
            "generated_exactly_one_in_vocabulary_chunk": (
                tuple(generated_chunk.shape)
                == (config.batch_size, config.patch_size)
                and int(generated_chunk.min()) >= 0
                and int(generated_chunk.max()) < config.vocab_size
            ),
            "checkpoint_state_digest_exact": model_digest == restored_digest,
            "checkpoint_fixed_noise_output_exact": (
                checkpoint_loss_delta == 0.0 and checkpoint_prediction_delta == 0.0
            ),
        }
        lifecycle_pass = all(lifecycle_gates.values())
        elapsed = time.perf_counter() - started
        result = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "experiment_id": f"tinyllm-calm-framework-lifecycle-seed{config.seed}-{device.type}",
            "status": "completed" if lifecycle_pass else "failed_gate",
            "condition": "framework_lifecycle",
            "seed": config.seed,
            "claim_scope": CLAIM_SCOPE,
            "scientific_fingerprint": _scientific_fingerprint(config),
            "configuration": asdict(config),
            "model_configuration": calm_config.to_dict(),
            "environment": _environment(device),
            "implementation": implementation,
            "training": {
                "source_history": source_history,
                "autoencoder_history": ae_history,
                "energy_history": energy_history,
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_bytes": peak_memory,
            },
            "analysis": {
                "autoencoder_before": ae_before,
                "autoencoder_after": ae_after,
                "energy_before": energy_before,
                "energy_after": energy_after,
                "generated_chunk_shape": list(generated_chunk.shape),
                "generated_chunk_min": int(generated_chunk.min()),
                "generated_chunk_max": int(generated_chunk.max()),
            },
            "lifecycle_gates": lifecycle_gates,
            "lifecycle_pass": lifecycle_pass,
            "model_architecture": model.get_architecture_summary(),
            "artifacts": {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256_file(checkpoint),
                "model_state_sha256": model_digest,
                "autoencoder_state_sha256": autoencoder_digest_after_energy,
                "checkpoint_loss_max_abs_delta": checkpoint_loss_delta,
                "checkpoint_prediction_max_abs_delta": checkpoint_prediction_delta,
            },
            "method_boundaries": [
                "systems shakedown; not language-model quality evidence",
                "GPT-2 adaptation; not an exact Llama-backbone reproduction",
                "synthetic arithmetic tokens; no BrierLM or perplexity claim",
                "base-temperature chunk sampling only",
                "copied tensors do not imply preservation of the source function",
                "checkpoint excludes optimizer, dataloader, scheduler, and RNG state",
            ],
        }
        _atomic_json(run_dir / "result.json", result)
        campaign = {
            "schema_version": SCHEMA_VERSION,
            "hypothesis_id": HYPOTHESIS_ID,
            "status": result["status"],
            "claim_scope": CLAIM_SCOPE,
            "configuration": asdict(config),
            "environment": result["environment"],
            "summary": {
                "requested": 1,
                "completed": int(lifecycle_pass),
                "failed": int(not lifecycle_pass),
                "reused": 0,
            },
            "aggregates": {
                "lifecycle_pass_count": int(lifecycle_pass),
                "lifecycle_gates": lifecycle_gates,
                "conclusion": (
                    "framework_lifecycle_validated"
                    if lifecycle_pass
                    else "framework_lifecycle_gate_failed"
                ),
            },
            "results": [
                {
                    "experiment_id": result["experiment_id"],
                    "status": result["status"],
                    "result": str(run_dir / "result.json"),
                    "checkpoint": str(checkpoint),
                }
            ],
            "method_boundaries": result["method_boundaries"],
        }
        _atomic_json(output_dir / "campaign_results.json", campaign)
        return campaign
    finally:
        if previous_cuda_rng is not None:
            torch.cuda.set_rng_state_all(previous_cuda_rng)
        torch.random.set_rng_state(previous_rng)
        torch.set_num_threads(previous_threads)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--source-steps", type=int, default=4)
    parser.add_argument("--autoencoder-steps", type=int, default=32)
    parser.add_argument("--energy-steps", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_calm_framework_shakedown/20260811_cpu_seed7"
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    campaign = run_shakedown(
        ShakedownConfig(
            seed=args.seed,
            device=args.device,
            source_training_steps=args.source_steps,
            autoencoder_training_steps=args.autoencoder_steps,
            energy_training_steps=args.energy_steps,
        ),
        args.output,
    )
    print(json.dumps(campaign["aggregates"], indent=2, sort_keys=True))
    print(args.output / "campaign_results.json")
    return 0 if campaign["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
