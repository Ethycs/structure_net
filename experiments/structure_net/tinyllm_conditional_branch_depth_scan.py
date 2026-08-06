#!/usr/bin/env python3
"""Test whether TinyLLM forms a cosine quotient inside its residual stream.

The source models are frozen.  At real-valued depths, linear and nonlinear
held-out probes predict the phase branch from exact cosine-matched pairs.  A
cosine-only conditional null, paired-label shuffle controls, shifted nuisances,
posterior topology, finite-fiber component proxies, and attention/MLP cuts
separate residual-stream quotient formation from collapse in the shared head.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import pearsonr
import torch
from torch import nn
import torch.nn.functional as F

try:
    from experiments.structure_net.tinyllm_depth_graded_quotient import (
        TRAINING_ARMS,
        DepthGradedCampaignConfig,
        run_arm,
        soft_depth_outputs,
    )
    from experiments.structure_net.tinyllm_internal_quotient_probe import (
        FiberDataset,
        _inputs_for_family,
        generate_matched_fiber_dataset,
    )
    from experiments.structure_net.tinyllm_predictive_circle import (
        CircleTaskConfig,
        _persistence_record,
        _resolve_device,
    )
    from experiments.structure_net.tinyllm_task_geometry_atlas import (
        extract_sublayer_representations,
        stage_labels,
    )
except ModuleNotFoundError:  # Direct ``python path/to/script.py`` execution.
    from tinyllm_depth_graded_quotient import (  # type: ignore[no-redef]
        TRAINING_ARMS,
        DepthGradedCampaignConfig,
        run_arm,
        soft_depth_outputs,
    )
    from tinyllm_internal_quotient_probe import (  # type: ignore[no-redef]
        FiberDataset,
        _inputs_for_family,
        generate_matched_fiber_dataset,
    )
    from tinyllm_predictive_circle import (  # type: ignore[no-redef]
        CircleTaskConfig,
        _persistence_record,
        _resolve_device,
    )
    from tinyllm_task_geometry_atlas import (  # type: ignore[no-redef]
        extract_sublayer_representations,
        stage_labels,
    )
from structure_net.components.analyzers import (
    fisher_rao_distance_matrix,
    representation_distance_matrix,
)
from structure_net.components.models import TinyLLMModel


SCHEMA_VERSION = "nal.tinyllm-conditional-branch-depth-scan.v1"
HYPOTHESIS_ID = "tinyllm-joint-depth-internal-cosine-quotient-v1"
QUOTIENTS = ("phase_circle", "cosine_interval")
BASE_DEPTHS = (
    0.0,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    8.0,
)
STANDARD_FINE_DEPTHS = (1.70, 1.75, 1.80, 1.825, 1.85, 1.875, 1.90, 1.95)


@dataclass(frozen=True)
class BranchDepthCampaignConfig:
    preset: str = "d8"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    training_arms: tuple[str, ...] = TRAINING_ARMS
    depths: tuple[float, ...] = BASE_DEPTHS
    standard_fine_depths: tuple[float, ...] = STANDARD_FINE_DEPTHS
    training_steps: int = 600
    train_samples: int = 2_048
    validation_samples: int = 512
    test_samples: int = 1_024
    activation_batch_size: int = 256
    probe_batch_size: int = 256
    probe_steps: int = 240
    probe_width: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_interval: int = 20
    early_stopping_patience: int = 5
    posterior_phase_points: int = 64
    fiber_cosines: tuple[float, ...] = (-0.8, -0.4, 0.0, 0.4, 0.8)
    fiber_replicates: int = 12
    include_phase_seed7_controls: bool = True
    device: str = "cuda:auto"
    seed: int = 71

    def __post_init__(self) -> None:
        if len(set(self.seeds)) < 5:
            raise ValueError("at least five distinct cosine seeds are required")
        if 7 not in self.seeds:
            raise ValueError("the retained seed 7 must be included")
        if set(self.training_arms).difference(TRAINING_ARMS):
            raise ValueError(f"training arms must be drawn from {TRAINING_ARMS}")
        if set(self.training_arms) != set(TRAINING_ARMS):
            raise ValueError("all three depth-training arms are required")
        if any(value < 4 or value % 2 for value in (
            self.train_samples,
            self.validation_samples,
            self.test_samples,
        )):
            raise ValueError("matched-fiber sample counts must be even and at least four")
        if self.probe_steps < 1 or self.probe_width < 2:
            raise ValueError("probe steps and width must be positive")
        if self.fiber_replicates < 2:
            raise ValueError("fiber_replicates must be at least two")
        if any(not -1.0 < value < 1.0 for value in self.fiber_cosines):
            raise ValueError("fiber cosine values must be interior to (-1, 1)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def depths_for_arm(
    arm: str, campaign: BranchDepthCampaignConfig, layer_count: int
) -> tuple[float, ...]:
    values = list(campaign.depths)
    if arm == "standard_final":
        values.extend(campaign.standard_fine_depths)
    return tuple(sorted({float(value) for value in values if value <= layer_count}))


def _checkpoint_name(arm: str, quotient: str, seed: int) -> str:
    return f"{arm}_{quotient}_seed{seed}.pt"


def ensure_checkpoints(
    task: CircleTaskConfig,
    campaign: BranchDepthCampaignConfig,
    device: torch.device,
    source_checkpoint_dir: Path,
    output_dir: Path,
) -> tuple[List[Dict[str, Any]], List[Path]]:
    """Reuse seed 7 and train the missing cosine checkpoints reproducibly."""
    records: List[Dict[str, Any]] = []
    checkpoints: List[Path] = []
    generated_dir = output_dir / "checkpoints"
    generated_dir.mkdir(parents=True, exist_ok=True)
    for seed in campaign.seeds:
        for arm in campaign.training_arms:
            name = _checkpoint_name(arm, "cosine_interval", seed)
            source = source_checkpoint_dir / name if seed == 7 else generated_dir / name
            if source.exists():
                records.append(
                    {
                        "training_arm": arm,
                        "quotient": "cosine_interval",
                        "seed": seed,
                        "checkpoint": str(source),
                        "source": "retained_seed7" if seed == 7 else "resumed_generated",
                    }
                )
                checkpoints.append(source)
                continue
            training_campaign = DepthGradedCampaignConfig(
                preset=campaign.preset,
                seed=seed,
                quotients=("cosine_interval",),
                training_arms=(arm,),
                training_steps=campaign.training_steps,
                checkpoints=(0, campaign.training_steps),
                phase_points=32,
                max_phase_points=64,
                max_row_phase_points=64,
                depth_step=1.0,
                device=str(device),
                save_checkpoints=True,
            )
            training = run_arm(
                arm=arm,
                quotient="cosine_interval",
                task=task,
                campaign=training_campaign,
                device=device,
                output_dir=output_dir,
                expected_standard_hash="",
            )
            if not source.exists():
                raise RuntimeError(f"training did not create {source}")
            records.append(
                {
                    "training_arm": arm,
                    "quotient": "cosine_interval",
                    "seed": seed,
                    "checkpoint": str(source),
                    "source": "trained_for_campaign",
                    "training_seconds": training["training_seconds"],
                    "training_depth_schedule_sha256": training[
                        "training_depth_schedule_sha256"
                    ],
                    "final_state_sha256": training["final_state_sha256"],
                    "final_shallow_depth_accuracy": training[
                        "final_shallow_depth_accuracy"
                    ],
                    "final_full_depth_accuracy": training[
                        "final_full_depth_accuracy"
                    ],
                }
            )
            checkpoints.append(source)
    if campaign.include_phase_seed7_controls:
        for arm in campaign.training_arms:
            source = source_checkpoint_dir / _checkpoint_name(
                arm, "phase_circle", 7
            )
            if not source.exists():
                raise FileNotFoundError(source)
            records.append(
                {
                    "training_arm": arm,
                    "quotient": "phase_circle",
                    "seed": 7,
                    "checkpoint": str(source),
                    "source": "retained_seed7_control",
                }
            )
            checkpoints.append(source)
    return records, checkpoints


def _matched_datasets(
    task: CircleTaskConfig, campaign: BranchDepthCampaignConfig
) -> Dict[str, FiberDataset]:
    return {
        "train": generate_matched_fiber_dataset(
            task,
            sample_count=campaign.train_samples,
            seed=campaign.seed + 101,
            family="training_support",
        ),
        "validation": generate_matched_fiber_dataset(
            task,
            sample_count=campaign.validation_samples,
            seed=campaign.seed + 211,
            family="training_support",
        ),
        "training_family_fit": generate_matched_fiber_dataset(
            task,
            sample_count=campaign.test_samples,
            seed=campaign.seed + 307,
            family="training_support",
        ),
        "overlapping_held_out": generate_matched_fiber_dataset(
            task,
            sample_count=campaign.test_samples,
            seed=campaign.seed + 401,
            family="training_support",
        ),
        "disjoint_nuisance": generate_matched_fiber_dataset(
            task,
            sample_count=campaign.test_samples,
            seed=campaign.seed + 503,
            family="disjoint_third_harmonic_drift",
        ),
    }


def generate_fiber_component_dataset(
    task: CircleTaskConfig,
    *,
    cosine_values: Sequence[float],
    replicates: int,
    family: str,
    seed: int,
) -> FiberDataset:
    """Sample nuisance clouds on both branches of several exact fibers."""
    generator = np.random.default_rng(seed)
    future: List[float] = []
    branch: List[float] = []
    fiber_id: List[int] = []
    cosine: List[float] = []
    for index, value in enumerate(cosine_values):
        positive = math.acos(float(value))
        negative = (2.0 * math.pi - positive) % (2.0 * math.pi)
        for label, phase in enumerate((positive, negative)):
            future.extend([phase] * replicates)
            branch.extend([float(label)] * replicates)
            fiber_id.extend([index] * replicates)
            cosine.extend([float(value)] * replicates)
    future_array = np.asarray(future)
    directions = generator.choice(np.array([-1.0, 1.0]), len(future_array))
    current = (future_array - directions * task.future_delta) % (2.0 * math.pi)
    inputs = _inputs_for_family(
        task, current, directions, family=family, seed=seed + 10_003
    )
    permutation = generator.permutation(len(future_array))
    return FiberDataset(
        input_ids=inputs[permutation],
        cosine=torch.tensor(np.asarray(cosine)[permutation], dtype=torch.float32),
        branch=torch.tensor(np.asarray(branch)[permutation], dtype=torch.float32),
        phase=torch.tensor(future_array[permutation], dtype=torch.float32),
        fiber_id=torch.tensor(np.asarray(fiber_id)[permutation], dtype=torch.int64),
    )


@torch.no_grad()
def extract_depth_representations(
    model: TinyLLMModel,
    inputs: torch.Tensor,
    depths: Sequence[float],
    device: torch.device,
    *,
    batch_size: int,
) -> Dict[float, np.ndarray]:
    """Capture query residuals, sharing exact block prefixes across depths."""
    model.eval()
    ordered = tuple(sorted({float(value) for value in depths}))
    captured: Dict[float, List[torch.Tensor]] = {depth: [] for depth in ordered}
    by_prefix: Dict[int, List[tuple[float, float]]] = {}
    for depth in ordered:
        prefix = min(int(math.floor(depth)), model.config.n_layer)
        by_prefix.setdefault(prefix, []).append((depth, depth - prefix))
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(device)
        positions = torch.arange(batch.shape[1], device=device, dtype=torch.long)
        value = model.transformer["wte"](batch) + model.transformer["wpe"](positions)
        for prefix in range(model.config.n_layer + 1):
            for depth, fraction in by_prefix.get(prefix, []):
                residual = value
                if prefix < model.config.n_layer and fraction > 0.0:
                    residual = model.transformer["h"][prefix].forward_gated(
                        value, fraction
                    )
                captured[depth].append(residual[:, -1, :].detach().cpu())
            if prefix < model.config.n_layer:
                value = model.transformer["h"][prefix](value)
    return {
        depth: torch.cat(chunks).float().numpy()
        for depth, chunks in captured.items()
    }


class _SmallProbe(nn.Module):
    def __init__(self, inputs: int, width: int, outputs: int, kind: str):
        super().__init__()
        if kind == "linear":
            self.network = nn.Linear(inputs, outputs)
        elif kind == "nonlinear":
            self.network = nn.Sequential(
                nn.Linear(inputs, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Linear(width, outputs),
            )
        else:
            raise ValueError(f"unknown probe kind: {kind}")

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class _ConditionalSuite(nn.Module):
    def __init__(self, features: int, width: int, kind: str):
        super().__init__()
        self.branch = _SmallProbe(features + 1, width, 1, kind)
        self.cosine = _SmallProbe(features, width, 1, kind)


def _paired_random_labels(dataset: FiberDataset, seed: int) -> np.ndarray:
    """Randomize each pair while preserving one label of each class per cosine."""
    generator = np.random.default_rng(seed)
    result = np.empty(len(dataset.branch), dtype=np.float32)
    fibers = dataset.fiber_id.numpy()
    for fiber in np.unique(fibers):
        indices = np.flatnonzero(fibers == fiber)
        if len(indices) != 2:
            raise ValueError("shuffle controls require exactly two samples per fiber")
        result[indices] = generator.permutation(np.array([0.0, 1.0]))
    return result


def _classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor
) -> Dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predicted = probabilities >= 0.5
    truth = labels.bool()
    positive = truth
    negative = ~truth
    sensitivity = float((predicted[positive] == truth[positive]).float().mean())
    specificity = float((predicted[negative] == truth[negative]).float().mean())
    return {
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "accuracy": float((predicted == truth).float().mean()),
        "log_loss": float(F.binary_cross_entropy_with_logits(logits, labels)),
        "mean_probability": float(probabilities.mean()),
    }


def fit_conditional_probe(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    evaluation_features: Mapping[str, np.ndarray],
    train_dataset: FiberDataset,
    validation_dataset: FiberDataset,
    evaluation_datasets: Mapping[str, FiberDataset],
    campaign: BranchDepthCampaignConfig,
    device: torch.device,
    *,
    kind: str,
    seed: int,
    shuffled_labels: bool = False,
) -> Dict[str, Any]:
    """Fit a frozen-representation branch/cosine suite with early stopping."""
    torch.manual_seed(seed)
    mean = train_features.mean(axis=0, dtype=np.float64).astype(np.float32)
    deviation = train_features.std(axis=0, dtype=np.float64).astype(np.float32)
    deviation = np.maximum(deviation, 1e-5)

    def normalize(values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((values - mean) / deviation).float().to(device)

    train_x = normalize(train_features)
    validation_x = normalize(validation_features)
    evaluation_x = {key: normalize(value) for key, value in evaluation_features.items()}
    train_cosine = train_dataset.cosine.to(device)
    validation_cosine = validation_dataset.cosine.to(device)
    evaluation_cosine = {
        key: dataset.cosine.to(device) for key, dataset in evaluation_datasets.items()
    }
    if shuffled_labels:
        train_labels = torch.from_numpy(
            _paired_random_labels(train_dataset, seed + 101)
        ).to(device)
        validation_labels = torch.from_numpy(
            _paired_random_labels(validation_dataset, seed + 211)
        ).to(device)
        evaluation_labels = {
            key: torch.from_numpy(
                _paired_random_labels(dataset, seed + 307 + 101 * index)
            ).to(device)
            for index, (key, dataset) in enumerate(evaluation_datasets.items())
        }
    else:
        train_labels = train_dataset.branch.to(device)
        validation_labels = validation_dataset.branch.to(device)
        evaluation_labels = {
            key: dataset.branch.to(device) for key, dataset in evaluation_datasets.items()
        }
    width = min(campaign.probe_width, max(16, train_features.shape[1]))
    suite = _ConditionalSuite(train_features.shape[1], width, kind).to(device)
    optimizer = torch.optim.AdamW(
        suite.parameters(),
        lr=campaign.learning_rate,
        weight_decay=campaign.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 401)

    def loss(
        values: torch.Tensor, cosine: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        branch_input = torch.cat((values, cosine[:, None]), dim=1)
        branch_logits = suite.branch(branch_input).squeeze(1)
        predicted_cosine = suite.cosine(values).squeeze(1)
        return F.binary_cross_entropy_with_logits(
            branch_logits, labels
        ) + F.mse_loss(predicted_cosine, cosine)

    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_step = 0
    stale = 0
    steps_run = 0
    for step in range(1, campaign.probe_steps + 1):
        indices = torch.randint(
            0,
            len(train_x),
            (campaign.probe_batch_size,),
            generator=generator,
        ).to(device)
        suite.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = loss(
            train_x[indices], train_cosine[indices], train_labels[indices]
        )
        train_loss.backward()
        optimizer.step()
        steps_run = step
        if step % campaign.validation_interval == 0 or step == campaign.probe_steps:
            suite.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss(validation_x, validation_cosine, validation_labels)
                )
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in suite.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= campaign.early_stopping_patience:
                    break
    if best_state is None:
        raise RuntimeError("probe training did not produce a validation checkpoint")
    suite.load_state_dict(best_state)
    suite.eval()
    evaluations: Dict[str, Any] = {}
    with torch.no_grad():
        for key, values in evaluation_x.items():
            branch_input = torch.cat(
                (values, evaluation_cosine[key][:, None]), dim=1
            )
            branch_logits = suite.branch(branch_input).squeeze(1)
            predicted_cosine = suite.cosine(values).squeeze(1)
            cosine_np = predicted_cosine.cpu().numpy()
            target_np = evaluation_cosine[key].cpu().numpy()
            correlation = 0.0
            if np.ptp(cosine_np) > 1e-6 and np.ptp(target_np) > 1e-6:
                correlation = float(pearsonr(cosine_np, target_np).statistic)
                if not np.isfinite(correlation):
                    correlation = 0.0
            evaluations[key] = {
                **_classification_metrics(branch_logits, evaluation_labels[key]),
                "cosine_pearson": correlation,
                "cosine_rmse": float(
                    torch.sqrt(F.mse_loss(predicted_cosine, evaluation_cosine[key]))
                ),
            }
    return {
        "probe_kind": kind,
        "shuffled_labels": shuffled_labels,
        "best_step": best_step,
        "steps_run": steps_run,
        "best_validation_loss": best_loss,
        "evaluations": evaluations,
    }


def fit_cosine_only_null(
    train_dataset: FiberDataset,
    validation_dataset: FiberDataset,
    evaluation_datasets: Mapping[str, FiberDataset],
    campaign: BranchDepthCampaignConfig,
    device: torch.device,
    *,
    kind: str,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Cross-validated conditional null p(branch | cosine)."""
    torch.manual_seed(seed)
    train_x = train_dataset.cosine[:, None].to(device)
    validation_x = validation_dataset.cosine[:, None].to(device)
    train_y = train_dataset.branch.to(device)
    validation_y = validation_dataset.branch.to(device)
    width = min(campaign.probe_width, 32)
    model = _SmallProbe(1, width, 1, kind).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=campaign.learning_rate, weight_decay=campaign.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    best_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stale = 0
    for step in range(1, campaign.probe_steps + 1):
        indices = torch.randint(
            0,
            len(train_x),
            (campaign.probe_batch_size,),
            generator=generator,
        ).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x[indices]).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, train_y[indices])
        loss.backward()
        optimizer.step()
        if step % campaign.validation_interval == 0 or step == campaign.probe_steps:
            model.eval()
            with torch.no_grad():
                value = float(
                    F.binary_cross_entropy_with_logits(
                        model(validation_x).squeeze(1), validation_y
                    )
                )
            if value < best_loss - 1e-5:
                best_loss = value
                best_state = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= campaign.early_stopping_patience:
                    break
    if best_state is None:
        raise RuntimeError("conditional null did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    result: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for key, dataset in evaluation_datasets.items():
            logits = model(dataset.cosine[:, None].to(device)).squeeze(1)
            result[key] = _classification_metrics(logits, dataset.branch.to(device))
    return result


def fiber_component_record(
    features: np.ndarray, dataset: FiberDataset
) -> Dict[str, Any]:
    """Finite-sample one-vs-two component proxy inside exact cosine fibers.

    Each branch nuisance cloud is first connected by its Euclidean MST.  If the
    nearest cross-branch edge appears only at a larger scale than both within-
    branch MST bottlenecks, the sampled fiber has two components at that scale.
    """
    fibers = dataset.fiber_id.numpy()
    branches = dataset.branch.numpy().astype(int)
    records = []
    for fiber in np.unique(fibers):
        indices = np.flatnonzero(fibers == fiber)
        values = features[indices]
        labels = branches[indices]
        distances = representation_distance_matrix(values, metric="euclidean")
        within_bottlenecks = []
        for label in (0, 1):
            local = distances[np.ix_(labels == label, labels == label)]
            tree = minimum_spanning_tree(local).toarray()
            edges = tree[tree > 0.0]
            within_bottlenecks.append(float(edges.max()) if len(edges) else 0.0)
        within_scale = max(within_bottlenecks)
        cross = distances[np.ix_(labels == 0, labels == 1)]
        cross_scale = float(cross.min())
        ratio = cross_scale / max(within_scale, 1e-12)
        silhouettes = []
        for local_index, label in enumerate(labels):
            same = labels == label
            same[local_index] = False
            other = labels != label
            a = float(distances[local_index, same].mean())
            b = float(distances[local_index, other].mean())
            silhouettes.append((b - a) / max(a, b, 1e-12))
        records.append(
            {
                "fiber_id": int(fiber),
                "cosine": float(dataset.cosine[indices[0]]),
                "within_branch_mst_bottleneck": within_scale,
                "nearest_cross_branch_distance": cross_scale,
                "merger_scale_ratio": ratio,
                "branch_silhouette": float(np.mean(silhouettes)),
                "component_count_proxy": 2 if ratio > 1.0 else 1,
            }
        )
    counts = [record["component_count_proxy"] for record in records]
    ratios = [record["merger_scale_ratio"] for record in records]
    return {
        "fibers": records,
        "median_merger_scale_ratio": float(np.median(ratios)),
        "mean_branch_silhouette": float(
            np.mean([record["branch_silhouette"] for record in records])
        ),
        "one_component_fraction": float(np.mean(np.asarray(counts) == 1)),
        "majority_component_count_proxy": 1
        if sum(value == 1 for value in counts) >= math.ceil(len(counts) / 2)
        else 2,
        "scope": "finite-sample MST scale proxy; not a certified Reeb graph or cosheaf",
    }


def _decoder_cosine_record(
    posteriors: np.ndarray,
    phases: np.ndarray,
    task: CircleTaskConfig,
    quotient: str,
) -> Dict[str, float]:
    if quotient == "cosine_interval":
        centers = np.linspace(-1.0, 1.0, task.phase_bins)
    elif quotient == "phase_circle":
        centers = np.cos(2.0 * math.pi * np.arange(task.phase_bins) / task.phase_bins)
    else:
        raise ValueError(f"unknown quotient: {quotient}")
    predicted = posteriors @ centers
    target = np.cos(phases)
    correlation = 0.0
    if np.ptp(predicted) > 1e-8:
        correlation = float(pearsonr(predicted, target).statistic)
        if not np.isfinite(correlation):
            correlation = 0.0
    return {
        "pearson_correlation": correlation,
        "root_mean_squared_error": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "predicted_range": float(np.ptp(predicted)),
    }


def decoder_depth_record(
    model: TinyLLMModel,
    task: CircleTaskConfig,
    depth: float,
    quotient: str,
    campaign: BranchDepthCampaignConfig,
    device: torch.device,
) -> Dict[str, Any]:
    phases = np.linspace(
        0.0, 2.0 * math.pi, campaign.posterior_phase_points, endpoint=False
    )
    posteriors, _ = soft_depth_outputs(model, task, phases, depth, device)
    persistence = _persistence_record(fisher_rao_distance_matrix(posteriors))
    return {
        "cosine_map": _decoder_cosine_record(posteriors, phases, task, quotient),
        "posterior_normalized_h1_lifetime": persistence["summary"][
            "normalized_h1_lifetime"
        ],
    }


def _add_log_loss_gains(
    probe: Dict[str, Any], null: Mapping[str, Mapping[str, float]]
) -> None:
    for family, metrics in probe["evaluations"].items():
        metrics["conditional_log_loss_gain_over_cosine_only"] = (
            float(null[family]["log_loss"]) - float(metrics["log_loss"])
        )


def _residual_quotient_cell(cell: Mapping[str, Any]) -> bool:
    nonlinear = cell["probes"]["nonlinear"]["evaluations"]
    fibers = cell["fiber_components"]
    return (
        nonlinear["overlapping_held_out"]["cosine_pearson"] >= 0.9
        and nonlinear["overlapping_held_out"]["balanced_accuracy"] <= 0.55
        and nonlinear["overlapping_held_out"][
            "conditional_log_loss_gain_over_cosine_only"
        ]
        <= 0.02
        and fibers["overlapping_held_out"]["majority_component_count_proxy"] == 1
    )


def _decoder_supported_quotient_cell(cell: Mapping[str, Any]) -> bool:
    return (
        _residual_quotient_cell(cell)
        and cell["decoder"]["cosine_map"]["pearson_correlation"] >= 0.9
    )


def _internal_quotient_cell(cell: Mapping[str, Any]) -> bool:
    nonlinear = cell["probes"]["nonlinear"]["evaluations"]
    fibers = cell["fiber_components"]
    return (
        _decoder_supported_quotient_cell(cell)
        and nonlinear["disjoint_nuisance"]["cosine_pearson"] >= 0.9
        and nonlinear["disjoint_nuisance"]["balanced_accuracy"] <= 0.55
        and fibers["disjoint_nuisance"]["majority_component_count_proxy"] == 1
    )


def analyze_checkpoint(
    checkpoint: Path,
    task: CircleTaskConfig,
    campaign: BranchDepthCampaignConfig,
    device: torch.device,
    datasets: Mapping[str, FiberDataset],
    fiber_datasets: Mapping[str, FiberDataset],
) -> Dict[str, Any]:
    started = time.perf_counter()
    model = TinyLLMModel.from_checkpoint(checkpoint, map_location="cpu").to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    arm = str(metadata["training_arm"])
    quotient = str(metadata["quotient"])
    seed = int(metadata["seed"])
    depths = depths_for_arm(arm, campaign, model.config.n_layer)
    all_datasets = {**datasets, **{f"fiber_{key}": value for key, value in fiber_datasets.items()}}
    representations = {
        key: extract_depth_representations(
            model,
            dataset.input_ids,
            depths,
            device,
            batch_size=campaign.activation_batch_size,
        )
        for key, dataset in all_datasets.items()
    }
    evaluation_names = (
        "training_family_fit",
        "overlapping_held_out",
        "disjoint_nuisance",
    )
    evaluation_datasets = {key: datasets[key] for key in evaluation_names}
    conditional_nulls = {
        kind: fit_cosine_only_null(
            datasets["train"],
            datasets["validation"],
            evaluation_datasets,
            campaign,
            device,
            kind=kind,
            seed=campaign.seed + seed * 101 + index * 1_009,
        )
        for index, kind in enumerate(("linear", "nonlinear"))
    }
    cells = []
    for depth_index, depth in enumerate(depths):
        probes: Dict[str, Any] = {}
        controls: Dict[str, Any] = {}
        for kind_index, kind in enumerate(("linear", "nonlinear")):
            probe_seed = (
                campaign.seed
                + seed * 10_007
                + depth_index * 1_009
                + kind_index * 101
            )
            evaluation_features = {
                key: representations[key][depth] for key in evaluation_names
            }
            probe = fit_conditional_probe(
                representations["train"][depth],
                representations["validation"][depth],
                evaluation_features,
                datasets["train"],
                datasets["validation"],
                evaluation_datasets,
                campaign,
                device,
                kind=kind,
                seed=probe_seed,
            )
            _add_log_loss_gains(probe, conditional_nulls[kind])
            control = fit_conditional_probe(
                representations["train"][depth],
                representations["validation"][depth],
                evaluation_features,
                datasets["train"],
                datasets["validation"],
                evaluation_datasets,
                campaign,
                device,
                kind=kind,
                seed=probe_seed + 50_021,
                shuffled_labels=True,
            )
            probes[kind] = probe
            controls[kind] = control
        cell: Dict[str, Any] = {
            "depth": depth,
            "probes": probes,
            "shuffled_label_controls": controls,
            "decoder": decoder_depth_record(
                model, task, depth, quotient, campaign, device
            ),
            "fiber_components": {
                family: fiber_component_record(
                    representations[f"fiber_{family}"][depth], dataset
                )
                for family, dataset in fiber_datasets.items()
            },
        }
        is_cosine = quotient == "cosine_interval"
        cell["residual_quotient_criterion_id"] = (
            is_cosine and _residual_quotient_cell(cell)
        )
        cell["decoder_supported_quotient_criterion_id"] = (
            is_cosine and _decoder_supported_quotient_cell(cell)
        )
        cell["nuisance_robust_quotient_criterion"] = (
            is_cosine and _internal_quotient_cell(cell)
        )
        cell["internal_quotient_criterion"] = cell[
            "nuisance_robust_quotient_criterion"
        ]
        cells.append(cell)
        primary = probes["nonlinear"]["evaluations"]["overlapping_held_out"]
        shifted = probes["nonlinear"]["evaluations"]["disjoint_nuisance"]
        print(
            checkpoint.stem,
            f"depth={depth:g}",
            f"branch={primary['balanced_accuracy']:.4f}",
            f"shifted={shifted['balanced_accuracy']:.4f}",
            f"cos={primary['cosine_pearson']:.4f}",
            flush=True,
        )
    residual_front = next(
        (cell["depth"] for cell in cells if cell["residual_quotient_criterion_id"]),
        None,
    )
    decoder_front = next(
        (
            cell["depth"]
            for cell in cells
            if cell["decoder_supported_quotient_criterion_id"]
        ),
        None,
    )
    robust_front = next(
        (
            cell["depth"]
            for cell in cells
            if cell["nuisance_robust_quotient_criterion"]
        ),
        None,
    )
    sublayer = analyze_transition_sublayers(
        model,
        task,
        campaign,
        device,
        datasets,
        fiber_datasets,
        conditional_nulls,
        residual_front,
        seed,
    )
    result = {
        "experiment_id": f"tinyllm-conditional-branch-{campaign.preset}-{arm}-{quotient}-seed{seed}",
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "checkpoint": str(checkpoint),
        "training_arm": arm,
        "quotient": quotient,
        "seed": seed,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "conditional_nulls": conditional_nulls,
        "depth_cells": cells,
        "residual_quotient_front_id": residual_front,
        "decoder_supported_quotient_front_id": decoder_front,
        "nuisance_robust_quotient_front": robust_front,
        "internal_quotient_front": robust_front,
        "transition_sublayer_scan": sublayer,
        "analysis_seconds": time.perf_counter() - started,
    }
    del model, representations
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def analyze_transition_sublayers(
    model: TinyLLMModel,
    task: CircleTaskConfig,
    campaign: BranchDepthCampaignConfig,
    device: torch.device,
    datasets: Mapping[str, FiberDataset],
    fiber_datasets: Mapping[str, FiberDataset],
    conditional_nulls: Mapping[str, Mapping[str, Mapping[str, float]]],
    quotient_front: Optional[float],
    seed: int,
) -> Dict[str, Any]:
    """Bracket the detected transition by pre/attention/MLP residual cuts."""
    target_block = model.config.n_layer if quotient_front is None else max(
        1, min(model.config.n_layer, int(math.ceil(quotient_front)))
    )
    labels = stage_labels(model.config.n_layer)
    indices = (2 * (target_block - 1), 2 * target_block - 1, 2 * target_block)
    all_datasets = {**datasets, **{f"fiber_{key}": value for key, value in fiber_datasets.items()}}
    activations = {
        key: extract_sublayer_representations(
            model,
            dataset.input_ids,
            device,
            batch_size=campaign.activation_batch_size,
        )
        for key, dataset in all_datasets.items()
    }
    evaluation_names = (
        "training_family_fit",
        "overlapping_held_out",
        "disjoint_nuisance",
    )
    evaluation_datasets = {key: datasets[key] for key in evaluation_names}
    stages = []
    for offset, index in enumerate(indices):
        probes = {}
        for kind_offset, kind in enumerate(("linear", "nonlinear")):
            probe = fit_conditional_probe(
                activations["train"][index],
                activations["validation"][index],
                {key: activations[key][index] for key in evaluation_names},
                datasets["train"],
                datasets["validation"],
                evaluation_datasets,
                campaign,
                device,
                kind=kind,
                seed=campaign.seed + seed * 20_011 + offset * 1_009 + kind_offset * 101,
            )
            _add_log_loss_gains(probe, conditional_nulls[kind])
            probes[kind] = probe
        stages.append(
            {
                "stage": labels[index],
                "stage_index": index,
                "probes": probes,
                "fiber_components": {
                    family: fiber_component_record(
                        activations[f"fiber_{family}"][index], dataset
                    )
                    for family, dataset in fiber_datasets.items()
                },
            }
        )
    return {
        "target_block": target_block,
        "selection_basis": "block containing first internal quotient cell; final block if none",
        "stages": stages,
        "gate_note": (
            "The fractional-depth gate scales attention and MLP together. These full "
            "sublayer cuts bracket mechanism but do not decompose a fractional gate exactly."
        ),
    }


def aggregate_results(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    cosine_runs = [run for run in runs if run["quotient"] == "cosine_interval"]
    phase_runs = [run for run in runs if run["quotient"] == "phase_circle"]
    arms: Dict[str, Any] = {}
    for arm in TRAINING_ARMS:
        cells = [run for run in cosine_runs if run["training_arm"] == arm]
        robust_fronts = [
            float(run["internal_quotient_front"])
            for run in cells
            if run["internal_quotient_front"] is not None
        ]
        residual_fronts = [
            float(run["residual_quotient_front_id"])
            for run in cells
            if run.get("residual_quotient_front_id") is not None
        ]
        decoder_fronts = [
            float(run["decoder_supported_quotient_front_id"])
            for run in cells
            if run.get("decoder_supported_quotient_front_id") is not None
        ]
        full = [run["depth_cells"][-1] for run in cells]
        arms[arm] = {
            "seed_count": len(cells),
            "nuisance_robust_fronts": {
                str(run["seed"]): run["internal_quotient_front"] for run in cells
            },
            "residual_quotient_fronts_id": {
                str(run["seed"]): run.get("residual_quotient_front_id")
                for run in cells
            },
            "decoder_supported_quotient_fronts_id": {
                str(run["seed"]): run.get("decoder_supported_quotient_front_id")
                for run in cells
            },
            "nuisance_robust_front_found_fraction": len(robust_fronts) / len(cells)
            if cells
            else 0.0,
            "residual_front_found_fraction_id": len(residual_fronts) / len(cells)
            if cells
            else 0.0,
            "median_residual_quotient_front_id": float(np.median(residual_fronts))
            if residual_fronts
            else None,
            "median_decoder_supported_quotient_front_id": float(
                np.median(decoder_fronts)
            )
            if decoder_fronts
            else None,
            "median_nuisance_robust_quotient_front": float(
                np.median(robust_fronts)
            )
            if robust_fronts
            else None,
            "full_depth_mean_branch_accuracy": float(
                np.mean(
                    [
                        cell["probes"]["nonlinear"]["evaluations"][
                            "overlapping_held_out"
                        ]["balanced_accuracy"]
                        for cell in full
                    ]
                )
            )
            if full
            else None,
            "full_depth_mean_shifted_branch_accuracy": float(
                np.mean(
                    [
                        cell["probes"]["nonlinear"]["evaluations"][
                            "disjoint_nuisance"
                        ]["balanced_accuracy"]
                        for cell in full
                    ]
                )
            )
            if full
            else None,
        }
    phase_controls = {}
    for run in phase_runs:
        full = run["depth_cells"][-1]
        phase_controls[run["training_arm"]] = {
            "full_depth_branch_accuracy": full["probes"]["nonlinear"][
                "evaluations"
            ]["overlapping_held_out"]["balanced_accuracy"],
            "full_depth_shifted_branch_accuracy": full["probes"]["nonlinear"][
                "evaluations"
            ]["disjoint_nuisance"]["balanced_accuracy"],
            "full_depth_cosine_pearson": full["probes"]["nonlinear"][
                "evaluations"
            ]["overlapping_held_out"]["cosine_pearson"],
        }
    standard = arms["standard_final"]["median_nuisance_robust_quotient_front"]
    discrete = arms["discrete_multi_exit"][
        "median_nuisance_robust_quotient_front"
    ]
    continuous = arms["continuous_gate"][
        "median_nuisance_robust_quotient_front"
    ]
    criteria = {
        "five_seeds_per_cosine_arm": all(
            arms[arm]["seed_count"] >= 5 for arm in TRAINING_ARMS
        ),
        "joint_arms_form_quotient_by_first_tenth_in_at_least_four_seeds": all(
            sum(
                value is not None and float(value) <= 0.1
                for value in arms[arm]["nuisance_robust_fronts"].values()
            )
            >= 4
            for arm in ("discrete_multi_exit", "continuous_gate")
        ),
        "ordinary_front_later_than_joint_medians": (
            standard is not None
            and discrete is not None
            and continuous is not None
            and standard > max(discrete, continuous)
        ),
        "cosine_full_depth_branch_at_chance_in_all_arms": all(
            arms[arm]["full_depth_mean_branch_accuracy"] is not None
            and arms[arm]["full_depth_mean_branch_accuracy"] <= 0.55
            for arm in TRAINING_ARMS
        ),
        "phase_seed7_controls_retain_branch": bool(phase_controls)
        and all(
            value["full_depth_branch_accuracy"] >= 0.8
            for value in phase_controls.values()
        ),
    }
    return {
        "cosine_arms": arms,
        "phase_seed7_controls": phase_controls,
        "descriptive_id_comparisons": {
            "all_cosine_runs_have_residual_quotient_front": all(
                arms[arm]["residual_front_found_fraction_id"] == 1.0
                for arm in TRAINING_ARMS
            ),
            "joint_median_residual_front_earlier_than_ordinary": (
                arms["standard_final"]["median_residual_quotient_front_id"]
                is not None
                and all(
                    arms[arm]["median_residual_quotient_front_id"]
                    < arms["standard_final"]["median_residual_quotient_front_id"]
                    for arm in ("discrete_multi_exit", "continuous_gate")
                )
            ),
        },
        "pre_registered_criteria": criteria,
        "criteria_passed": all(criteria.values()),
    }


def _refresh_run_derived_fields(run: Dict[str, Any]) -> None:
    """Backfill the three nested quotient criteria without refitting probes."""
    is_cosine = run["quotient"] == "cosine_interval"
    for cell in run["depth_cells"]:
        cell["residual_quotient_criterion_id"] = (
            is_cosine and _residual_quotient_cell(cell)
        )
        cell["decoder_supported_quotient_criterion_id"] = (
            is_cosine and _decoder_supported_quotient_cell(cell)
        )
        cell["nuisance_robust_quotient_criterion"] = (
            is_cosine and _internal_quotient_cell(cell)
        )
        cell["internal_quotient_criterion"] = cell[
            "nuisance_robust_quotient_criterion"
        ]

    def first(field: str) -> Optional[float]:
        return next(
            (float(cell["depth"]) for cell in run["depth_cells"] if cell[field]),
            None,
        )

    run["residual_quotient_front_id"] = first(
        "residual_quotient_criterion_id"
    )
    run["decoder_supported_quotient_front_id"] = first(
        "decoder_supported_quotient_criterion_id"
    )
    run["nuisance_robust_quotient_front"] = first(
        "nuisance_robust_quotient_criterion"
    )
    run["internal_quotient_front"] = run["nuisance_robust_quotient_front"]


def refresh_existing_campaign(
    results_path: Path,
    *,
    requested_device: str,
    recompute_sublayers: bool = True,
) -> Dict[str, Any]:
    """Refresh derived criteria and localize cosine sublayers at the ID front."""
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    bundle = json.loads(results_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported results schema in {results_path}")
    task = CircleTaskConfig(**bundle["task_config"])
    campaign_values = dict(bundle["campaign_config"])
    for field in (
        "seeds",
        "training_arms",
        "depths",
        "standard_fine_depths",
        "fiber_cosines",
    ):
        campaign_values[field] = tuple(campaign_values[field])
    campaign_values["device"] = requested_device
    campaign = BranchDepthCampaignConfig(**campaign_values)
    for run in bundle["runs"]:
        _refresh_run_derived_fields(run)

    partial = results_path.with_name("results.partial.json")
    if recompute_sublayers:
        device = _resolve_device(requested_device)
        datasets = _matched_datasets(task, campaign)
        fiber_datasets = {
            "overlapping_held_out": generate_fiber_component_dataset(
                task,
                cosine_values=campaign.fiber_cosines,
                replicates=campaign.fiber_replicates,
                family="training_support",
                seed=campaign.seed + 60_013,
            ),
            "disjoint_nuisance": generate_fiber_component_dataset(
                task,
                cosine_values=campaign.fiber_cosines,
                replicates=campaign.fiber_replicates,
                family="disjoint_third_harmonic_drift",
                seed=campaign.seed + 70_019,
            ),
        }
        for run in bundle["runs"]:
            if run["quotient"] != "cosine_interval":
                continue
            front = run["residual_quotient_front_id"]
            expected_block = max(1, int(math.ceil(front))) if front is not None else 1
            if run["transition_sublayer_scan"]["target_block"] == expected_block:
                continue
            model = TinyLLMModel.from_checkpoint(
                Path(run["checkpoint"]), map_location="cpu"
            ).to(device)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            run["transition_sublayer_scan"] = analyze_transition_sublayers(
                model,
                task,
                campaign,
                device,
                datasets,
                fiber_datasets,
                run["conditional_nulls"],
                front,
                int(run["seed"]),
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            bundle["derived_fields_refreshed_at"] = _utc_now()
            _write_json(partial, bundle)
            print(
                run["experiment_id"],
                f"sublayer_block={expected_block}",
                flush=True,
            )
    bundle["aggregates"] = aggregate_results(bundle["runs"])
    criteria = bundle["aggregates"]["pre_registered_criteria"]
    descriptive = bundle["aggregates"]["descriptive_id_comparisons"]
    bundle["claim_status"] = {
        "confirmed": bool(bundle["aggregates"]["criteria_passed"]),
        "internal_quotient_evidence": bool(
            criteria["cosine_full_depth_branch_at_chance_in_all_arms"]
        ),
        "in_distribution_residual_quotient_evidence": bool(
            descriptive["all_cosine_runs_have_residual_quotient_front"]
        ),
        "early_joint_depth_quotient_evidence": bool(
            criteria[
                "joint_arms_form_quotient_by_first_tenth_in_at_least_four_seeds"
            ]
        ),
        "joint_depth_moves_id_residual_front_earlier": bool(
            descriptive["joint_median_residual_front_earlier_than_ordinary"]
        ),
        "interpretation": (
            "All preregistered multi-seed residual-quotient criteria passed."
            if bundle["aggregates"]["criteria_passed"]
            else "Internal quotient evidence was observed in distribution, but at "
            "least one preregistered nuisance-robust criterion failed."
        ),
    }
    bundle["derived_fields_refreshed_at"] = _utc_now()
    _write_json(results_path, bundle)
    _write_json(partial, bundle)
    return bundle


def run_campaign(
    task: CircleTaskConfig,
    campaign: BranchDepthCampaignConfig,
    source_checkpoint_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(campaign.device)
    torch.use_deterministic_algorithms(True)
    partial = output_dir / "results.partial.json"
    fresh_bundle: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "training_checkpoints",
        "started_at": _utc_now(),
        "task_config": asdict(task),
        "campaign_config": asdict(campaign),
        "source_checkpoint_dir": str(source_checkpoint_dir),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
        },
        "pre_registered_prediction": (
            "Joint-depth cosine supervision moves the depth where conditional phase-"
            "branch information becomes unrecoverable into the first tenth of block 1, "
            "while residual and decoder cosine maps remain strong under held-out and "
            "disjoint nuisances. Ordinary final-depth training forms the quotient later."
        ),
        "protocol": {
            "source_models_frozen_during_probes": True,
            "conditional_target": "branch = sign(sin(phase)) given exact matched cosine",
            "primary_statistic": "balanced held-out branch accuracy; chance 0.5",
            "conditional_log_loss_comparison": "L(branch|cosine) - L(branch|residual,cosine)",
            "probe_families": ["linear", "two-hidden-layer nonlinear"],
            "shuffle_control": "one randomized label of each class within every exact cosine pair",
            "nuisance_evaluations": [
                "training-family held-out sample",
                "overlapping held-out sample",
                "disjoint third-harmonic drift family",
            ],
            "fiber_component_method": "within-branch MST bottleneck versus nearest cross-branch edge",
        },
        "method_boundaries": [
            "probe log-loss gain is a variational surrogate, not a calibrated mutual-information estimate",
            "the finite-fiber MST statistic is not a certified Reeb graph or cosheaf",
            "posterior H1 remains decoder-conditioned",
            "the fractional residual gate is a numerical depth family, not a neural-ODE flow",
            "only the cosine arms have five seeds; phase arms are retained seed-7 controls",
        ],
        "checkpoint_training": [],
        "runs": [],
    }
    if partial.exists():
        resumed = json.loads(partial.read_text(encoding="utf-8"))
        if (
            resumed.get("schema_version") != SCHEMA_VERSION
            or resumed.get("campaign_config") != fresh_bundle["campaign_config"]
            or resumed.get("task_config") != fresh_bundle["task_config"]
        ):
            raise ValueError(
                f"existing partial campaign is incompatible with this invocation: {partial}"
            )
        bundle = resumed
        bundle["status"] = "resuming"
    else:
        bundle = fresh_bundle
    _write_json(partial, bundle)
    training_records, checkpoints = ensure_checkpoints(
        task, campaign, device, source_checkpoint_dir, output_dir
    )
    bundle["checkpoint_training"] = training_records
    bundle["status"] = "probing"
    _write_json(partial, bundle)
    datasets = _matched_datasets(task, campaign)
    fiber_datasets = {
        "overlapping_held_out": generate_fiber_component_dataset(
            task,
            cosine_values=campaign.fiber_cosines,
            replicates=campaign.fiber_replicates,
            family="training_support",
            seed=campaign.seed + 60_013,
        ),
        "disjoint_nuisance": generate_fiber_component_dataset(
            task,
            cosine_values=campaign.fiber_cosines,
            replicates=campaign.fiber_replicates,
            family="disjoint_third_harmonic_drift",
            seed=campaign.seed + 70_019,
        ),
    }
    completed = {run["checkpoint"] for run in bundle["runs"]}
    for checkpoint in checkpoints:
        if str(checkpoint) in completed:
            continue
        run = analyze_checkpoint(
            checkpoint, task, campaign, device, datasets, fiber_datasets
        )
        bundle["runs"].append(run)
        _write_json(partial, bundle)
    bundle["aggregates"] = aggregate_results(bundle["runs"])
    bundle["status"] = "completed"
    bundle["completed_at"] = _utc_now()
    bundle["claim_status"] = {
        "confirmed": bool(bundle["aggregates"]["criteria_passed"]),
        "internal_quotient_evidence": bool(
            bundle["aggregates"]["pre_registered_criteria"][
                "cosine_full_depth_branch_at_chance_in_all_arms"
            ]
        ),
        "early_joint_depth_quotient_evidence": bool(
            bundle["aggregates"]["pre_registered_criteria"][
                "joint_arms_form_quotient_by_first_tenth_in_at_least_four_seeds"
            ]
        ),
        "interpretation": (
            "All preregistered multi-seed residual-quotient criteria passed."
            if bundle["aggregates"]["criteria_passed"]
            else "At least one preregistered residual-quotient criterion failed."
        ),
    }
    _write_json(output_dir / "results.json", bundle)
    _write_json(partial, bundle)
    return bundle


def _comma_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint-dir",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_depth_graded_quotient/"
            "20260805_d8_seed7/checkpoints"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/tinyllm_conditional_branch_depth_scan/"
            "20260805_d8_five_seed"
        ),
    )
    parser.add_argument("--seeds", default="7,17,29,41,53")
    parser.add_argument("--device", default="cuda:auto")
    parser.add_argument("--training-steps", type=int, default=600)
    parser.add_argument("--probe-steps", type=int, default=240)
    parser.add_argument("--train-samples", type=int, default=2_048)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--test-samples", type=int, default=1_024)
    parser.add_argument("--no-phase-controls", action="store_true")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="refresh derived fronts and sublayer localization in an existing result",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.refresh_existing:
        bundle = refresh_existing_campaign(
            args.output / "results.json", requested_device=args.device
        )
        print(json.dumps(bundle["claim_status"], indent=2, sort_keys=True))
        print(args.output / "results.json")
        return 0
    campaign = BranchDepthCampaignConfig(
        seeds=_comma_ints(args.seeds),
        training_steps=args.training_steps,
        probe_steps=args.probe_steps,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        include_phase_seed7_controls=not args.no_phase_controls,
        device=args.device,
    )
    bundle = run_campaign(
        CircleTaskConfig(), campaign, args.source_checkpoint_dir, args.output
    )
    print(json.dumps(bundle["claim_status"], indent=2, sort_keys=True))
    print(args.output / "results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
