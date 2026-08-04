#!/usr/bin/env python3
"""Re-evaluate archived MNIST checkpoints on the canonical full test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.torch_version import TorchVersion
from torch.utils.data import DataLoader

from structure_net.data_factory.datasets import load_dataset


DEFAULT_CHECKPOINT_DIR = Path(
    "archive/old_research/foundational_results/test_simple_models"
)


def build_dense_mlp(architecture: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_size, output_size) in enumerate(
        zip(architecture, architecture[1:])
    ):
        layers.append(nn.Linear(input_size, output_size))
        if index < len(architecture) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def evaluate_checkpoint(path: Path, loader: DataLoader) -> dict:
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    architecture = [int(size) for size in checkpoint["architecture"]]
    model = build_dense_mlp(architecture)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    correct = 0
    examples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.view(inputs.size(0), -1))
            correct += (logits.argmax(dim=1) == targets).sum().item()
            examples += targets.numel()
    state = checkpoint["model_state_dict"]
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "architecture": architecture,
        "seed": checkpoint.get("seed"),
        "stored_epoch": checkpoint.get("epoch"),
        "stored_accuracy": float(checkpoint.get("accuracy", 0.0)),
        "canonical_full_test_accuracy": correct / examples,
        "correct": correct,
        "test_examples": examples,
        "stored_minus_canonical": float(checkpoint.get("accuracy", 0.0))
        - correct / examples,
        "has_sparse_masks": any(key.endswith("mask") for key in state),
        "state_dict_keys": list(state),
        "status": "audited",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = args.output or Path(
        f"data/experiments/mnist_checkpoint_audit/{timestamp}/results.json"
    )

    test_dataset = load_dataset("mnist", train=False, download=False)
    loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    checkpoints = [
        evaluate_checkpoint(path, loader)
        for path in sorted(args.checkpoint_dir.glob("*.pt"))
    ]
    result = {
        "schema_version": "nal.checkpoint-audit.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_contract": {
            "dataset": "mnist",
            "split": "canonical full test set",
            "examples": len(test_dataset),
            "normalization": {"mean": [0.1307], "stddev": [0.3081]},
            "no_training": True,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "checkpoints": checkpoints,
        "summary": {
            "checkpoint_count": len(checkpoints),
            "all_dense": all(not item["has_sparse_masks"] for item in checkpoints),
            "stored_accuracy_reproduced": all(
                abs(item["stored_minus_canonical"]) < 1e-12 for item in checkpoints
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)
    for checkpoint in checkpoints:
        print(
            f"{Path(checkpoint['path']).name}: "
            f"stored={checkpoint['stored_accuracy']:.2%}, "
            f"canonical={checkpoint['canonical_full_test_accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()
