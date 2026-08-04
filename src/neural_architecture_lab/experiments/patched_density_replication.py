"""Paired replication of the archived MNIST patched-density finding.

The historical experiment intended to train a 2%-connected scaffold, detect
high-activation hidden units, and attach small dense paths from those units to
the classifier output.  This module makes that intended mechanism executable
and compares it with an identically initialized continuation-only control.
"""

from __future__ import annotations

import copy
import json
import math
import platform
import random
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from structure_net.components.layers import StandardSparseLayer
from structure_net.data_factory.datasets import load_dataset


@dataclass(frozen=True)
class ReplicationConfig:
    """Fixed protocol for one or more paired dataset/seed replications."""

    datasets: Tuple[str, ...] = ("mnist", "fashion_mnist")
    seeds: Tuple[int, ...] = (4,)
    architecture: Tuple[int, int, int] = (784, 128, 10)
    connection_density: float = 0.02
    scaffold_epochs: int = 20
    continuation_epochs: int = 30
    batch_size: int = 64
    eval_batch_size: int = 1000
    scaffold_learning_rate: float = 1e-3
    continuation_learning_rate: float = 1e-4
    patch_learning_rate: float = 5e-4
    patch_width: int = 10
    patch_scale: float = 0.1
    max_patches: int = 5
    extrema_stddevs: float = 2.0
    probe_split: str = "test"
    probe_batches: int = 1
    subset_fraction: Optional[float] = None
    num_workers: int = 0
    torch_threads: Optional[int] = None
    device: str = "auto"
    cache_dataset_in_memory: bool = True

    def __post_init__(self) -> None:
        if len(self.architecture) != 3:
            raise ValueError("patched-density replication requires [input, hidden, output]")
        if not 0 < self.connection_density <= 1:
            raise ValueError("connection_density must be in (0, 1]")
        if self.probe_split not in {"train", "test"}:
            raise ValueError("probe_split must be 'train' or 'test'")
        if self.probe_batches < 1:
            raise ValueError("probe_batches must be positive")


class PatchedDensityClassifier(nn.Module):
    """Canonical sparse scaffold with extrema-selected dense output patches."""

    def __init__(
        self,
        architecture: Sequence[int],
        connection_density: float,
        patch_width: int = 10,
        patch_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if len(architecture) != 3:
            raise ValueError("architecture must contain input, hidden, and output sizes")

        input_size, hidden_size, output_size = (int(size) for size in architecture)
        self.architecture = (input_size, hidden_size, output_size)
        self.connection_density = float(connection_density)
        self.patch_width = int(patch_width)
        self.patch_scale = float(patch_scale)
        self.hidden_layer = StandardSparseLayer(
            input_size, hidden_size, self.connection_density
        )
        self.output_layer = StandardSparseLayer(
            hidden_size, output_size, self.connection_density
        )
        self.patches = nn.ModuleDict()
        self.patch_sources: Dict[str, int] = {}

    def hidden_activations(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened = inputs.view(inputs.size(0), -1)
        return torch.relu(self.hidden_layer(flattened))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden_activations(inputs)
        logits = self.output_layer(hidden)
        for name, patch in self.patches.items():
            source = self.patch_sources[name]
            logits = logits + self.patch_scale * patch(hidden[:, source : source + 1])
        return logits

    def add_high_extrema_patches(self, neuron_indices: Iterable[int]) -> int:
        """Attach one dense scalar-to-logit path per selected hidden neuron."""
        output_size = self.architecture[-1]
        for neuron_index in neuron_indices:
            neuron_index = int(neuron_index)
            if not 0 <= neuron_index < self.architecture[1]:
                raise ValueError(f"hidden neuron index out of range: {neuron_index}")
            name = f"high_{neuron_index}"
            if name in self.patches:
                continue
            self.patches[name] = nn.Sequential(
                nn.Linear(1, self.patch_width),
                nn.ReLU(),
                nn.Linear(self.patch_width, output_size),
            )
            self.patch_sources[name] = neuron_index
        return len(self.patches)

    def parameter_counts(self) -> Dict[str, int]:
        scaffold_active_weights = int(
            self.hidden_layer.mask.sum().item() + self.output_layer.mask.sum().item()
        )
        scaffold_biases = int(
            self.hidden_layer.linear.bias.numel() + self.output_layer.linear.bias.numel()
        )
        patch_parameters = sum(parameter.numel() for parameter in self.patches.parameters())
        stored_parameters = sum(parameter.numel() for parameter in self.parameters())
        dense_scaffold_parameters = (
            self.architecture[0] * self.architecture[1]
            + self.architecture[1]
            + self.architecture[1] * self.architecture[2]
            + self.architecture[2]
        )
        return {
            "active_scaffold_weights": scaffold_active_weights,
            "scaffold_biases": scaffold_biases,
            "patch_parameters": patch_parameters,
            "effective_parameters": scaffold_active_weights
            + scaffold_biases
            + patch_parameters,
            "stored_trainable_parameters": stored_parameters,
            "dense_scaffold_parameters": dense_scaffold_parameters,
        }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


def _materialize_dataset(dataset: Dataset) -> TensorDataset:
    """Evaluate deterministic transforms once instead of once per training epoch."""
    loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=0)
    inputs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for batch_inputs, batch_targets in loader:
        inputs.append(batch_inputs)
        targets.append(batch_targets)
    return TensorDataset(torch.cat(inputs), torch.cat(targets))


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_examples = targets.size(0)
        total_loss += loss.item() * batch_examples
        total_examples += batch_examples
    return total_loss / max(total_examples, 1)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            batch_examples = targets.size(0)
            total_loss += criterion(logits, targets).item() * batch_examples
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total_examples += batch_examples
    return total_loss / max(total_examples, 1), correct / max(total_examples, 1)


def _detect_high_extrema(
    model: PatchedDensityClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    stddevs: float,
    max_patches: int,
    probe_batches: int,
) -> Dict[str, Any]:
    model.eval()
    activation_sum = torch.zeros(model.architecture[1], device=device)
    examples = 0
    with torch.no_grad():
        for batch_index, (inputs, _) in enumerate(loader):
            if batch_index >= probe_batches:
                break
            activations = model.hidden_activations(inputs.to(device))
            activation_sum += activations.sum(dim=0)
            examples += activations.size(0)

    means = activation_sum / max(examples, 1)
    threshold = means.mean() + stddevs * means.std(unbiased=False)
    candidates = torch.where(means > threshold)[0]
    ranked = sorted(
        (int(index) for index in candidates.tolist()),
        key=lambda index: float(means[index]),
        reverse=True,
    )
    selected = ranked[:max_patches]
    return {
        "probe_examples": examples,
        "mean_activation": float(means.mean()),
        "activation_stddev": float(means.std(unbiased=False)),
        "high_threshold": float(threshold),
        "candidate_indices": ranked,
        "selected_indices": selected,
        "selected_mean_activations": [float(means[index]) for index in selected],
    }


def _train_arm(
    *,
    name: str,
    model: PatchedDensityClassifier,
    train_dataset: Dataset,
    test_loader: DataLoader,
    epochs: int,
    batch_size: int,
    shuffle_seed: int,
    num_workers: int,
    device: torch.device,
    parameter_groups: List[Dict[str, Any]],
    verbose: bool,
) -> Tuple[List[Dict[str, float]], float]:
    loader = _loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=shuffle_seed,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.Adam(parameter_groups)
    criterion = nn.CrossEntropyLoss()
    history: List[Dict[str, float]] = []
    best_accuracy = 0.0
    for epoch in range(epochs):
        train_loss = _train_epoch(model, loader, optimizer, criterion, device)
        test_loss, test_accuracy = _evaluate(model, test_loader, criterion, device)
        best_accuracy = max(best_accuracy, test_accuracy)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
        if verbose:
            print(
                f"  {name:9s} epoch {epoch + 1:02d}/{epochs}: "
                f"loss={train_loss:.4f}, test={test_accuracy:.2%}",
                flush=True,
            )
    return history, best_accuracy


def _run_pair(
    dataset_name: str,
    seed: int,
    config: ReplicationConfig,
    device: torch.device,
    *,
    verbose: bool,
) -> Dict[str, Any]:
    _seed_everything(seed)
    started = time.perf_counter()
    train_dataset = load_dataset(
        dataset_name,
        train=True,
        download=True,
        subset_fraction=config.subset_fraction,
        seed=seed,
    )
    test_dataset = load_dataset(
        dataset_name,
        train=False,
        download=True,
        subset_fraction=config.subset_fraction,
        seed=seed,
    )
    if config.cache_dataset_in_memory:
        train_dataset = _materialize_dataset(train_dataset)
        test_dataset = _materialize_dataset(test_dataset)
    test_loader = _loader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        seed=seed + 1,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = PatchedDensityClassifier(
        config.architecture,
        config.connection_density,
        config.patch_width,
        config.patch_scale,
    ).to(device)
    scaffold_history, scaffold_best = _train_arm(
        name="scaffold",
        model=model,
        train_dataset=train_dataset,
        test_loader=test_loader,
        epochs=config.scaffold_epochs,
        batch_size=config.batch_size,
        shuffle_seed=seed + 1000,
        num_workers=config.num_workers,
        device=device,
        parameter_groups=[
            {"params": model.parameters(), "lr": config.scaffold_learning_rate}
        ],
        verbose=verbose,
    )
    criterion = nn.CrossEntropyLoss()
    scaffold_loss, scaffold_accuracy = _evaluate(model, test_loader, criterion, device)

    probe_dataset = test_dataset if config.probe_split == "test" else train_dataset
    probe_loader = _loader(
        probe_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        seed=seed + 2,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    extrema = _detect_high_extrema(
        model,
        probe_loader,
        device,
        stddevs=config.extrema_stddevs,
        max_patches=config.max_patches,
        probe_batches=config.probe_batches,
    )

    control = copy.deepcopy(model)
    patched = copy.deepcopy(model)
    _seed_everything(seed + 3000)
    patched.add_high_extrema_patches(extrema["selected_indices"])
    patched.to(device)

    control_history, control_best = _train_arm(
        name="control",
        model=control,
        train_dataset=train_dataset,
        test_loader=test_loader,
        epochs=config.continuation_epochs,
        batch_size=config.batch_size,
        shuffle_seed=seed + 2000,
        num_workers=config.num_workers,
        device=device,
        parameter_groups=[
            {"params": control.parameters(), "lr": config.continuation_learning_rate}
        ],
        verbose=verbose,
    )
    patched_history, patched_best = _train_arm(
        name="patched",
        model=patched,
        train_dataset=train_dataset,
        test_loader=test_loader,
        epochs=config.continuation_epochs,
        batch_size=config.batch_size,
        shuffle_seed=seed + 2000,
        num_workers=config.num_workers,
        device=device,
        parameter_groups=[
            {
                "params": [
                    *patched.hidden_layer.parameters(),
                    *patched.output_layer.parameters(),
                ],
                "lr": config.continuation_learning_rate,
            },
            {"params": patched.patches.parameters(), "lr": config.patch_learning_rate},
        ],
        verbose=verbose,
    )
    control_loss, control_accuracy = _evaluate(control, test_loader, criterion, device)
    patched_loss, patched_accuracy = _evaluate(patched, test_loader, criterion, device)
    counts = patched.parameter_counts()

    return {
        "experiment_id": f"patched-density-{dataset_name}-seed-{seed}",
        "hypothesis_id": "archived-patched-density-high-accuracy",
        "dataset": dataset_name,
        "seed": seed,
        "status": "completed",
        "metrics": {
            "scaffold_accuracy": scaffold_accuracy,
            "scaffold_best_accuracy": scaffold_best,
            "control_accuracy": control_accuracy,
            "control_best_accuracy": control_best,
            "patched_accuracy": patched_accuracy,
            "patched_best_accuracy": patched_best,
            "patched_minus_scaffold": patched_accuracy - scaffold_accuracy,
            "patched_minus_control": patched_accuracy - control_accuracy,
            "scaffold_loss": scaffold_loss,
            "control_loss": control_loss,
            "patched_loss": patched_loss,
            "patch_count": len(patched.patches),
            **counts,
            "effective_parameter_fraction_of_dense": counts["effective_parameters"]
            / counts["dense_scaffold_parameters"],
        },
        "extrema": extrema,
        "training_history": {
            "scaffold": scaffold_history,
            "control": control_history,
            "patched": patched_history,
        },
        "model_architecture": list(config.architecture),
        "training_time_seconds": time.perf_counter() - started,
        "observations": [
            "Control and patched arms start from the same trained scaffold.",
            "Continuation arms receive the same deterministic minibatch order.",
        ],
        "anomalies": [],
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def _summarize(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_dataset: Dict[str, Dict[str, Any]] = {}
    for dataset_name in sorted({run["dataset"] for run in runs}):
        selected = [run for run in runs if run["dataset"] == dataset_name]
        patched = [run["metrics"]["patched_accuracy"] for run in selected]
        controls = [run["metrics"]["control_accuracy"] for run in selected]
        paired_deltas = [run["metrics"]["patched_minus_control"] for run in selected]
        by_dataset[dataset_name] = {
            "runs": len(selected),
            "mean_scaffold_accuracy": _mean(
                run["metrics"]["scaffold_accuracy"] for run in selected
            ),
            "mean_control_accuracy": _mean(controls),
            "mean_patched_accuracy": _mean(patched),
            "mean_patched_minus_control": _mean(paired_deltas),
            "patched_accuracy_stddev": statistics.pstdev(patched)
            if len(patched) > 1
            else 0.0,
            "replicated_95_percent_accuracy": _mean(patched) >= 0.95,
            "patches_outperform_control": _mean(paired_deltas) > 0,
        }
    return {"by_dataset": by_dataset}


def run_replication(
    config: ReplicationConfig,
    *,
    output_dir: Path,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Path]:
    """Run all configured pairs and persist one self-describing JSON artifact."""
    if config.torch_threads is not None:
        torch.set_num_threads(config.torch_threads)
    device = _resolve_device(config.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    runs: List[Dict[str, Any]] = []
    for dataset_name in config.datasets:
        for seed in config.seeds:
            if verbose:
                print(f"\n[{dataset_name} seed={seed}] device={device}", flush=True)
            runs.append(_run_pair(dataset_name, seed, config, device, verbose=verbose))

    result = {
        "schema_version": "nal.paired-replication.v1",
        "experiment": {
            "name": "Patched-density MNIST replication and Fashion-MNIST transfer",
            "question": (
                "Does extrema-guided dense patching reproduce the archived high "
                "MNIST accuracy, and does the paired gain transfer to Fashion-MNIST?"
            ),
            "historical_claim": {
                "mnist_best_accuracy": 0.9767,
                "status": "unverified archival transcript",
                "source": "archive/old_research/notes/Breakthroughs.md",
            },
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
        "protocol": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "torch_threads": torch.get_num_threads(),
        },
        "runs": runs,
        "summary": _summarize(runs),
        "caveats": [
            "The archived success code cannot reproduce its own transcript because its patch output width does not match the logits.",
            "This protocol repairs that routing according to the stated design intent.",
            "Masked dense tensors reduce effective connections but not stored parameter memory or dense-kernel compute.",
            "The archival probe used held-out test inputs; probe_split records whether that behavior is retained.",
        ],
    }
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result, output_path
