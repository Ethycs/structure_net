#!/usr/bin/env python3
"""Pretrain a TinyLLM on the BabyLM strict-small corpus.

Causal language modeling over the cleaned 10M-word BabyLM text, tokenized
with the task BPE tokenizer (specials pinned, answer ids reserved above the
text vocabulary). The resulting checkpoint initializes the temporal-phase
language-task fine-tuning arms.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from experiments.structure_net.tinyllm_temporal_language_task import (
    BOS_ID,
    TemporalLanguageTaskConfig,
    load_tokenizer,
)
from structure_net.components.models import TinyLLMModel
from structure_net.components.models.tinyllm_model import TinyLLMConfig


@dataclass(frozen=True)
class BabylmPretrainConfig:
    corpus_dir: str = "data/corpora/babylm_10M"
    tokenizer_path: str = "data/corpora/babylm_10M_bpe16k.tokenizer.json"
    preset: str = "d8"
    vocab_size: int = 50_257
    block_size: int = 256
    batch_size: int = 24
    training_steps: int = 12_000
    warmup_steps: int = 200
    learning_rate: float = 3e-4
    final_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 7
    validation_tokens: int = 262_144
    validation_batches: int = 16
    log_every: int = 100
    validate_every: int = 500
    checkpoint_every: int = 4_000
    device: str = "cuda:1"
    use_amp: bool = True

    def __post_init__(self) -> None:
        if self.training_steps < 1 or self.batch_size < 1 or self.block_size < 8:
            raise ValueError("training_steps, batch_size, block_size must be positive")
        if self.warmup_steps >= self.training_steps:
            raise ValueError("warmup must be shorter than training")
        if self.validation_tokens < self.block_size * self.validation_batches:
            raise ValueError("validation split is too small")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_token_cache(config: BabylmPretrainConfig, cache_path: Path) -> Path:
    """Encode the corpus once into a flat token array (documents BOS-joined)."""
    corpus_files = sorted(Path(config.corpus_dir).glob("*.txt"))
    if not corpus_files:
        raise FileNotFoundError(f"no corpus files under {config.corpus_dir}")
    manifest = {path.name: _sha256_file(path) for path in corpus_files}
    tokenizer_sha = _sha256_file(Path(config.tokenizer_path))
    meta_path = cache_path.with_suffix(".meta.json")
    if cache_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("manifest") == manifest
            and meta.get("tokenizer_sha256") == tokenizer_sha
        ):
            return cache_path
    task = TemporalLanguageTaskConfig(tokenizer_path=config.tokenizer_path)
    tokenizer = load_tokenizer(task)
    stream: List[np.ndarray] = []
    for path in corpus_files:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for start in range(0, len(lines), 2_000):
            batch = tokenizer.encode_batch(lines[start : start + 2_000])
            for encoding in batch:
                stream.append(np.array([BOS_ID] + encoding.ids, dtype=np.int32))
    tokens = np.concatenate(stream)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, tokens)
    meta_path.write_text(
        json.dumps(
            {
                "manifest": manifest,
                "tokenizer_sha256": tokenizer_sha,
                "token_count": int(len(tokens)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return cache_path


def _cosine_lr(step: int, config: BabylmPretrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(
        1, config.training_steps - config.warmup_steps
    )
    return config.final_learning_rate + 0.5 * (
        config.learning_rate - config.final_learning_rate
    ) * (1.0 + math.cos(math.pi * progress))


def _batch(
    tokens: np.ndarray,
    generator: np.random.Generator,
    config: BabylmPretrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = generator.integers(0, len(tokens) - config.block_size - 1, config.batch_size)
    rows = np.stack([tokens[s : s + config.block_size + 1] for s in starts])
    block = torch.from_numpy(rows.astype(np.int64)).to(device, non_blocking=True)
    return block[:, :-1], block[:, 1:]


def run_pretraining(config: BabylmPretrainConfig, output: Path) -> Dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)

    cache_path = prepare_token_cache(
        config, Path(config.corpus_dir).parent / "babylm_10M_bpe16k.tokens.npy"
    )
    tokens = np.load(cache_path)
    validation = tokens[-config.validation_tokens :]
    training = tokens[: -config.validation_tokens]
    print(
        f"tokens: train={len(training):,} val={len(validation):,}", flush=True
    )

    model_config = TinyLLMConfig.from_preset(
        config.preset,
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        initialization_seed=config.seed,
    )
    model = TinyLLMModel(model_config, name=f"BabylmPretrain_{config.preset}_{config.seed}")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")
    generator = np.random.default_rng(config.seed)
    validation_generator = np.random.default_rng((config.seed, 424_243))
    validation_batches = [
        _batch(validation, validation_generator, config, torch.device("cpu"))
        for _ in range(config.validation_batches)
    ]

    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for inputs, labels in validation_batches:
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.autocast(
                    device_type=device.type, enabled=scaler.is_enabled()
                ):
                    logits, _ = model(inputs, return_full_logits=True)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                    )
                losses.append(float(loss))
        model.train()
        return float(np.mean(losses))

    history: List[Dict[str, float]] = []
    started = time.perf_counter()
    model.train()
    for step in range(1, config.training_steps + 1):
        lr = _cosine_lr(step - 1, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, labels = _batch(training, generator, config, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits, _ = model(inputs, return_full_logits=True)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % config.log_every == 0:
            record = {"step": step, "loss": float(loss), "lr": lr}
            if step == 1 or step % config.validate_every == 0:
                record["validation_loss"] = evaluate()
            history.append(record)
            print(json.dumps(record), flush=True)
        if step % config.checkpoint_every == 0 or step == config.training_steps:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": asdict(model_config),
                    "pretrain_config": asdict(config),
                    "step": step,
                },
                output / f"checkpoint_step{step}.pt",
            )

    final_validation = evaluate()
    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "configuration": asdict(config),
        "token_cache_sha256": _sha256_file(cache_path),
        "tokenizer_sha256": _sha256_file(Path(config.tokenizer_path)),
        "final_validation_loss": final_validation,
        "final_validation_perplexity": math.exp(final_validation),
        "history": history,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "training_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "final_checkpoint": str(output / f"checkpoint_step{config.training_steps}.pt"),
    }
    (output / "pretrain_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"final_validation_loss": final_validation}), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--shakedown", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/tinyllm_babylm_pretrain/20260812_d8_seed7"),
    )
    args = parser.parse_args()
    steps = 30 if args.shakedown else args.steps
    config = BabylmPretrainConfig(
        seed=args.seed,
        training_steps=steps,
        warmup_steps=min(200, max(1, steps // 10)),
        validate_every=min(500, steps),
        checkpoint_every=min(4_000, steps),
        device=args.device,
    )
    run_pretraining(config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
