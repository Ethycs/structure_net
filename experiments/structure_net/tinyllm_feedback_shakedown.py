#!/usr/bin/env python3
"""Run matched TinyLLM baseline, recompute, and random-feedback controls.

This deterministic CPU systems shakedown validates construction, training,
component-driven growth, and checkpoint restoration on an arithmetic token
task. It is not evidence of language-model or sensor-task quality.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, List, Optional

import torch

from structure_net.components.evolvers import FeedbackGrowthEvolver
from structure_net.components.models import TinyLLMConfig, TinyLLMModel
from structure_net.components.strategies import RandomFeedbackGrowthStrategy
from structure_net.components.trainers import CausalLanguageModelTrainer
from structure_net.core import AnalysisReport, EvolutionContext


SCHEMA_VERSION = "nal.tinyllm-feedback-shakedown.v1"


@dataclass(frozen=True)
class ShakedownConfig:
    preset: str = "tiny"
    vocab_size: int = 64
    sequence_length: int = 12
    batch_size: int = 8
    training_steps: int = 8
    learning_rate: float = 0.01
    seed: int = 7
    device: str = "cpu"
    tiny_layers: int = 3
    tiny_heads: int = 2
    tiny_width: int = 32
    feedback_connections: int = 2
    feedback_neurons: int = 4
    feedback_connection_density: float = 0.25
    feedback_gate_init: float = 1e-3

    def __post_init__(self) -> None:
        if self.sequence_length < 2 or self.batch_size <= 0 or self.training_steps < 0:
            raise ValueError("sequence_length, batch_size, and training_steps are invalid")
        if self.feedback_connections <= 0 or self.feedback_neurons <= 0:
            raise ValueError("feedback connection count and width must be positive")
        if not 0.0 < self.feedback_connection_density <= 1.0:
            raise ValueError("feedback_connection_density must be in (0, 1]")
        if torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("The shakedown runner supports CPU or CUDA devices")


def _model_config(config: ShakedownConfig) -> TinyLLMConfig:
    if config.preset == "tiny":
        return TinyLLMConfig(
            block_size=config.sequence_length,
            vocab_size=config.vocab_size,
            n_layer=config.tiny_layers,
            n_head=config.tiny_heads,
            n_embd=config.tiny_width,
            initialization_seed=config.seed,
        )
    return TinyLLMConfig.from_preset(config.preset, initialization_seed=config.seed)


def _batches(config: ShakedownConfig, vocab_size: int) -> List[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    positions = torch.arange(config.sequence_length + 1)
    return [
        (
            torch.randint(
                0,
                vocab_size,
                (config.batch_size, 1),
                generator=generator,
            )
            + positions
        )
        % vocab_size
        for _ in range(max(1, config.training_steps))
    ]


def _state_digest(model: TinyLLMModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _evaluate(
    model: TinyLLMModel,
    batch: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    tokens = batch.to(device)
    _, loss = model(
        tokens[:, :-1],
        tokens[:, 1:],
        return_full_logits=True,
    )
    assert loss is not None
    return float(loss.cpu())


def _configure_arm(
    arm: str,
    model: TinyLLMModel,
    trainer: CausalLanguageModelTrainer,
    config: ShakedownConfig,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if arm == "baseline":
        return None, None
    if arm == "recompute_control":
        model.refinement_steps = 1
        return None, None
    if arm != "random_feedback":
        raise ValueError(f"Unknown shakedown arm: {arm}")

    strategy = RandomFeedbackGrowthStrategy(
        count=config.feedback_connections,
        num_neurons=config.feedback_neurons,
        connection_density=config.feedback_connection_density,
        gate_init=config.feedback_gate_init,
        refinement_steps=1,
        seed=config.seed,
    )
    plan = strategy.propose_plan(
        AnalysisReport(), EvolutionContext(seed=config.seed, device=config.device)
    )
    growth = FeedbackGrowthEvolver().apply_plan(
        plan,
        model,
        trainer,
        trainer.optimizer,
    )
    return dict(plan), growth


def _run_arm(
    arm: str,
    model_config: TinyLLMConfig,
    initial_state: Dict[str, torch.Tensor],
    batches: Iterable[torch.Tensor],
    config: ShakedownConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    model = TinyLLMModel(model_config, name=f"TinyLLMShakedown_{arm}")
    model.load_state_dict(initial_state)
    device = torch.device(config.device)
    model.to(device)
    batch_list = list(batches)
    pre_intervention_loss = _evaluate(model, batch_list[0], device)

    trainer = CausalLanguageModelTrainer(
        optimizer_kwargs={"lr": config.learning_rate, "weight_decay": 0.0}
    )
    trainer.attach(model)
    growth_plan, growth_result = _configure_arm(arm, model, trainer, config)
    initial_loss = _evaluate(model, batch_list[0], device)
    context = EvolutionContext(
        device=str(device), refinement_steps=model.refinement_steps
    )
    history: List[Dict[str, float]] = [
        trainer.train_step(model, batch, context)
        for batch in batch_list[: config.training_steps]
    ]
    final_loss = _evaluate(model, batch_list[0], device)
    architecture = model.get_architecture_summary()
    active_connections = sum(
        item["active_connections"] for item in architecture["feedback_connections"]
    )
    token_count = sum(item["tokens"] for item in history)

    checkpoint = output_dir / f"{arm}.pt"
    model.to("cpu").save_checkpoint(
        checkpoint,
        metadata={"experiment_family": SCHEMA_VERSION, "arm": arm},
    )
    restored = TinyLLMModel.from_checkpoint(checkpoint).eval()
    probe = batch_list[0][:2, :-1]
    with torch.no_grad():
        expected_logits, _ = model(probe, return_full_logits=True)
        restored_logits, _ = restored(probe, return_full_logits=True)
    checkpoint_delta = float((expected_logits - restored_logits).abs().max())
    topology_equal = (
        restored.get_feedback_topology() == model.get_feedback_topology()
    )
    if checkpoint_delta != 0.0 or not topology_equal:
        raise RuntimeError(f"{arm} checkpoint round trip was not exact")

    return {
        "experiment_id": f"tinyllm-feedback-{arm}-seed{config.seed}",
        "hypothesis_id": "tinyllm-delayed-feedback-lifecycle-v1",
        "status": "completed",
        "arm": arm,
        "growth_plan": growth_plan,
        "growth_result": growth_result,
        "metrics": {
            "pre_intervention_loss": pre_intervention_loss,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_change": final_loss - initial_loss,
            "mean_training_loss": (
                statistics.fmean(item["loss"] for item in history)
                if history
                else initial_loss
            ),
            "tokens": token_count,
            "total_parameters": architecture["total_parameters"],
            "feedback_parameters": architecture["feedback_parameters"],
            "active_feedback_connections": active_connections,
            "refinement_steps": model.refinement_steps,
        },
        "training_history": history,
        "model_architecture": architecture,
        "model_state_sha256": _state_digest(model),
        "model_checkpoint": checkpoint.name,
        "checkpoint_round_trip": {
            "max_abs_logit_delta": checkpoint_delta,
            "topology_equal": topology_equal,
        },
    }


def run_shakedown(config: ShakedownConfig, output_dir: Path) -> Dict[str, Any]:
    """Run all matched arms and persist a deterministic result bundle."""
    device = torch.device(config.device)
    if device.type == "cuda":
        # Required by deterministic CUDA matrix multiplication on CUDA >= 10.2.
        # The canonical CLI sets this before CUDA creates a cuBLAS handle.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {config.device!r} was requested, but PyTorch cannot "
                "access a CUDA driver in this process"
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_rng_state = torch.random.get_rng_state()
    previous_cuda_rng_states = (
        torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    )
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        model_config = _model_config(config)
        template = TinyLLMModel(model_config)
        initial_state = {
            key: value.detach().clone() for key, value in template.state_dict().items()
        }
        batches = _batches(config, model_config.vocab_size)
        results = [
            _run_arm(arm, model_config, initial_state, batches, config, output_dir)
            for arm in ("baseline", "recompute_control", "random_feedback")
        ]
        by_arm = {result["arm"]: result for result in results}
        baseline = by_arm["baseline"]
        recompute = by_arm["recompute_control"]
        feedback = by_arm["random_feedback"]
        pre_intervention_losses = {
            result["metrics"]["pre_intervention_loss"] for result in results
        }
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "experiment_family": "tinyllm_feedback_shakedown",
            "claim_scope": "systems_lifecycle_only_not_quality_evidence",
            "configuration": asdict(config),
            "model_config": asdict(model_config),
            "results": results,
            "comparisons": {
                "all_arms_share_pre_intervention_loss": len(pre_intervention_losses) == 1,
                "recompute_matches_baseline_initial": (
                    recompute["metrics"]["initial_loss"]
                    == baseline["metrics"]["initial_loss"]
                ),
                "recompute_matches_baseline_final": (
                    recompute["metrics"]["final_loss"]
                    == baseline["metrics"]["final_loss"]
                ),
                "feedback_final_loss_delta": (
                    feedback["metrics"]["final_loss"]
                    - baseline["metrics"]["final_loss"]
                ),
                "all_checkpoint_round_trips_exact": all(
                    result["checkpoint_round_trip"]["max_abs_logit_delta"] == 0.0
                    and result["checkpoint_round_trip"]["topology_equal"]
                    for result in results
                ),
            },
            "unsupported_boundaries": [
                "feedback graphs cannot export as GPT2LMHeadModel or GGUF",
                "feedback inference does not support a standard GPT-2 KV cache",
                "model checkpoints do not include optimizer, scheduler, dataloader, or RNG state",
                "masked feedback weights use dense storage and kernels",
                "this arithmetic-token shakedown is not task-quality evidence",
            ],
        }
        if not bundle["comparisons"]["all_arms_share_pre_intervention_loss"]:
            raise RuntimeError("Shakedown arms did not share the same initial model")
        if not bundle["comparisons"]["all_checkpoint_round_trips_exact"]:
            raise RuntimeError("A shakedown checkpoint failed exact restoration")
        result_path = output_dir / "results.json"
        result_path.write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return bundle
    finally:
        if previous_cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(previous_cuda_rng_states)
        torch.random.set_rng_state(previous_rng_state)
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.set_num_threads(previous_threads)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        default="tiny",
        choices=["tiny", "d6", "d8", "d10", "d11", "d12"],
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/tinyllm_feedback_shakedown/seed_7"),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = run_shakedown(
        ShakedownConfig(
            preset=args.preset,
            training_steps=args.steps,
            seed=args.seed,
            device=args.device,
        ),
        args.output,
    )
    print(json.dumps(bundle["comparisons"], indent=2, sort_keys=True))
    print(args.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
