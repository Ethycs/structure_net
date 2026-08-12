"""Analysis primitives for the preregistered observable-C3 TinyLLM campaign."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import torch
from torch import nn

import experiments.structure_net.tinyllm_c3_temporal_quotient_preflight as preflight
import experiments.structure_net.tinyllm_c3_temporal_quotient_training as stage0


REPRESENTATION_CUTS = ("frontend", "post_attention", "post_mlp", "full")
CAUSAL_CUTS = REPRESENTATION_CUTS
REGIMES = ("composition", "extrapolation")
PROBE_SEEDS = {
    "train": 231_003,
    "validation": 231_021,
    "composition": 331_003,
    "extrapolation": 331_021,
}


@dataclass(frozen=True)
class ProbeConfig:
    train_latents: int = 2_048
    validation_latents: int = 512
    test_latents: int = 1_024
    steps: int = 240
    width: int = 128
    batch_size: int = 256
    validation_interval: int = 20
    patience: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    extraction_batch_size: int = 256

    def __post_init__(self) -> None:
        if min(
            self.train_latents,
            self.validation_latents,
            self.test_latents,
            self.steps,
            self.width,
            self.batch_size,
        ) < 1:
            raise ValueError("probe sizes must be positive")
        if self.validation_interval < 1 or self.patience < 1:
            raise ValueError("probe validation controls must be positive")


def _uniform(
    count: int, bounds: tuple[float, float], generator: torch.Generator
) -> torch.Tensor:
    return torch.empty(count, dtype=torch.float64).uniform_(
        *bounds, generator=generator
    )


def generate_shift_dataset(
    task: stage0.C3TaskConfig,
    *,
    regime: str,
    latent_count: int,
    seed: int,
    all_sheets: bool,
) -> stage0.C3TrainingDataset:
    """Generate a fresh declared shift with either one or all exact deck sheets."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    if latent_count < 1:
        raise ValueError("latent count must be positive")
    ranges = preflight.REGIME_RANGES[regime]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    phase = 2.0 * math.pi * torch.rand(
        latent_count, dtype=torch.float64, generator=generator
    )
    speed_magnitude = _uniform(latent_count, ranges["speed"], generator)
    speed_sign = (
        2.0
        * torch.randint(0, 2, (latent_count,), generator=generator).double()
        - 1.0
    )
    speed = speed_magnitude * speed_sign
    amplitude = _uniform(latent_count, ranges["amplitude"], generator)
    offset = _uniform(latent_count, ranges["offset"], generator)
    drift = _uniform(latent_count, ranges["drift"], generator)
    continuous = preflight._continuous_observation(
        phase, speed, amplitude, offset, drift
    )
    saturation_count = int(
        (continuous.abs() >= preflight.QUANTIZATION_LIMIT).sum()
    )
    canonical = preflight.quantize(continuous)
    calibration = torch.stack((amplitude, offset, drift), dim=-1)
    target = torch.cos(3.0 * (phase + task.time_steps * speed))
    if all_sheets:
        repeats = 3
        tokens = torch.stack(
            [
                preflight.apply_deck_action(row, element)
                for row in canonical
                for element in range(3)
            ]
        )
        deck = torch.arange(3).repeat(latent_count)
    else:
        repeats = 1
        deck = torch.randint(0, 3, (latent_count,), generator=generator)
        tokens = torch.stack(
            [
                preflight.apply_deck_action(row, int(element))
                for row, element in zip(canonical, deck)
            ]
        )
    repeated_target = target.repeat_interleave(repeats)
    posterior, bins = stage0._targets(repeated_target, task.phase_bins)
    return stage0.C3TrainingDataset(
        canonical_tokens=canonical.repeat_interleave(repeats, dim=0),
        tokens=tokens,
        calibration=calibration.repeat_interleave(repeats, dim=0),
        target=repeated_target,
        target_posteriors=posterior,
        target_bins=bins,
        phase=phase.repeat_interleave(repeats),
        speed=speed.repeat_interleave(repeats),
        deck=deck,
        pair_id=torch.arange(latent_count).repeat_interleave(repeats),
        saturation_count=saturation_count,
    )


def latent_fingerprint(dataset: stage0.C3TrainingDataset) -> str:
    first = torch.zeros(len(dataset), dtype=torch.bool)
    first[0] = True
    first[1:] = dataset.pair_id[1:] != dataset.pair_id[:-1]
    return stage0._digest_tensors(
        dataset.phase[first],
        dataset.speed[first],
        dataset.calibration[first],
        dataset.target[first],
    )


def dataset_hash(dataset: stage0.C3TrainingDataset) -> str:
    return stage0._digest_tensors(
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


def build_probe_datasets(
    task: stage0.C3TaskConfig, config: ProbeConfig
) -> dict[str, stage0.C3TrainingDataset]:
    specs = {
        "train": ("composition", config.train_latents),
        "validation": ("composition", config.validation_latents),
        "composition": ("composition", config.test_latents),
        "extrapolation": ("extrapolation", config.test_latents),
    }
    return {
        name: generate_shift_dataset(
            task,
            regime=regime,
            latent_count=count,
            seed=PROBE_SEEDS[name],
            all_sheets=True,
        )
        for name, (regime, count) in specs.items()
    }


def build_task_datasets(
    task: stage0.C3TaskConfig, test_latents: int
) -> dict[str, stage0.C3TrainingDataset]:
    return {
        regime: generate_shift_dataset(
            task,
            regime=regime,
            latent_count=test_latents,
            seed=PROBE_SEEDS[regime],
            all_sheets=False,
        )
        for regime in REGIMES
    }


def validate_split_contract(
    probe_datasets: Mapping[str, stage0.C3TrainingDataset],
    task_datasets: Mapping[str, stage0.C3TrainingDataset],
) -> dict[str, Any]:
    probe_fingerprints = {
        name: latent_fingerprint(dataset)
        for name, dataset in probe_datasets.items()
    }
    task_fingerprints = {
        name: latent_fingerprint(dataset)
        for name, dataset in task_datasets.items()
    }
    # The one-sheet task set and all-sheet probe set deliberately share final
    # latent histories. Fit splits remain distinct from both final families.
    fit = {probe_fingerprints["train"], probe_fingerprints["validation"]}
    final = {
        probe_fingerprints["composition"],
        probe_fingerprints["extrapolation"],
    }
    final_match = all(
        probe_fingerprints[regime] == task_fingerprints[regime]
        for regime in REGIMES
    )
    saturation = {
        name: dataset.saturation_count
        for name, dataset in {**probe_datasets, **task_datasets}.items()
    }
    all_sheet_contract = all(
        torch.equal(
            dataset.deck.reshape(-1, 3),
            torch.arange(3).expand(len(dataset) // 3, 3),
        )
        for dataset in probe_datasets.values()
    )
    passed = (
        len(fit) == 2
        and len(final) == 2
        and fit.isdisjoint(final)
        and final_match
        and all(value == 0 for value in saturation.values())
        and all_sheet_contract
    )
    return {
        "probe_latent_fingerprints": probe_fingerprints,
        "task_latent_fingerprints": task_fingerprints,
        "fit_final_disjoint": fit.isdisjoint(final),
        "one_sheet_all_sheet_final_match": final_match,
        "saturation_counts": saturation,
        "all_sheet_deck_order_exact": all_sheet_contract,
        "pass": passed,
    }


def target_derangement(dataset: stage0.C3TrainingDataset) -> torch.Tensor:
    if len(dataset) % 3:
        raise ValueError("target derangement requires all three sheets")
    bins = dataset.target_bins[0::3]
    count = len(bins)
    order = torch.argsort(bins, stable=True)
    shifted = torch.roll(order, shifts=count // 2)
    permutation = torch.empty_like(order)
    permutation[order] = shifted
    if bool(torch.any(bins[permutation] == bins)):
        raise RuntimeError("registered target-bin derangement has a fixed bin")
    return permutation


@torch.no_grad()
def extract_representations(
    system: stage0.C3TemporalTinyLLM,
    dataset: stage0.C3TrainingDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    system.eval()
    collected = {cut: [] for cut in REPRESENTATION_CUTS}
    for start in range(0, len(dataset), batch_size):
        stop = start + batch_size
        cuts = system.residual_cuts(
            dataset.tokens[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
        )
        collected["frontend"].append(cuts["feature"].flatten(1).cpu())
        for cut in ("post_attention", "post_mlp", "full"):
            collected[cut].append(cuts[cut][:, -1, :].cpu())
    return {cut: torch.cat(values).float() for cut, values in collected.items()}


def analytic_condition(dataset: stage0.C3TrainingDataset) -> torch.Tensor:
    carrier, _ = preflight.analytic_carrier(
        dataset.tokens, dataset.calibration
    )
    history = torch.stack((carrier.real, carrier.imag), dim=-1).flatten(1).float()
    return torch.cat((history, dataset.target.float()[:, None]), dim=-1)


def _standardize(
    train: torch.Tensor, values: Mapping[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    mean = train.double().mean(0).float()
    scale = train.double().std(0, unbiased=False).float().clamp_min(1e-5)
    return (
        (train - mean) / scale,
        {name: (value - mean) / scale for name, value in values.items()},
        {"mean": mean, "scale": scale},
    )


def _balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predicted = logits.argmax(-1)
    recalls = []
    for label in range(3):
        selected = labels == label
        recalls.append(float((predicted[selected] == label).float().mean()))
    return sum(recalls) / 3.0


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.double().reshape(-1)
    y = right.double().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    return float(
        (x * y).sum()
        / (
            torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
        ).clamp_min(1e-12)
    )


def _fit_probe(
    train_x: torch.Tensor,
    validation_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_y: torch.Tensor,
    *,
    output_width: int,
    classification: bool,
    config: ProbeConfig,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], config.width),
        nn.GELU(),
        nn.Linear(config.width, output_width),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 17_003)
    batches = torch.randint(
        0,
        len(train_x),
        (config.steps, min(config.batch_size, len(train_x))),
        generator=generator,
    )
    validation_x_device = validation_x.to(device)
    validation_y_device = validation_y.to(device)
    best = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    ran = 0
    for step in range(1, config.steps + 1):
        index = batches[step - 1]
        x = train_x[index].to(device)
        y = train_y[index].to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = (
            nn.functional.cross_entropy(prediction, y.long())
            if classification
            else nn.functional.mse_loss(prediction[:, 0], y.float())
        )
        loss.backward()
        optimizer.step()
        ran = step
        if step % config.validation_interval == 0 or step == config.steps:
            model.eval()
            with torch.no_grad():
                prediction = model(validation_x_device)
                value = float(
                    nn.functional.cross_entropy(
                        prediction, validation_y_device.long()
                    )
                    if classification
                    else nn.functional.mse_loss(
                        prediction[:, 0], validation_y_device.float()
                    )
                )
            model.train()
            if value < best - 1e-6:
                best = value
                stale = 0
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            else:
                stale += 1
                if stale >= config.patience:
                    break
    if best_state is None:
        raise RuntimeError("probe never produced a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "validation_loss": best,
        "steps_run": ran,
        "parameter_count": sum(value.numel() for value in model.parameters()),
    }


@torch.no_grad()
def _semantic_evaluations(
    model: nn.Module,
    features: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    result = {}
    for regime in REGIMES:
        prediction = model(features[regime].to(device))[:, 0].cpu()
        target = targets[regime].float()
        result[regime] = {
            "target_correlation": _correlation(prediction, target),
            "target_rmse": float(
                torch.sqrt((prediction.double() - target.double()).square().mean())
            ),
        }
    return result


@torch.no_grad()
def _deck_evaluations(
    model: nn.Module,
    features: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    result = {}
    for regime in REGIMES:
        logits = model(features[regime].to(device)).cpu()
        target = labels[regime].long()
        result[regime] = {
            "balanced_accuracy": _balanced_accuracy(logits, target),
            "cross_entropy": float(nn.functional.cross_entropy(logits, target)),
        }
    return result


def representation_analysis(
    system: stage0.C3TemporalTinyLLM,
    datasets: Mapping[str, stage0.C3TrainingDataset],
    config: ProbeConfig,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    features = {
        name: extract_representations(
            system, dataset, device, config.extraction_batch_size
        )
        for name, dataset in datasets.items()
    }
    conditions = {name: analytic_condition(dataset) for name, dataset in datasets.items()}
    labels = {name: dataset.deck.long() for name, dataset in datasets.items()}
    targets = {name: dataset.target.float() for name, dataset in datasets.items()}
    condition_train, condition_values, condition_scaling = _standardize(
        conditions["train"],
        {name: value for name, value in conditions.items() if name != "train"},
    )
    null_model, null_fit = _fit_probe(
        condition_train,
        condition_values["validation"],
        labels["train"],
        labels["validation"],
        output_width=3,
        classification=True,
        config=config,
        seed=seed + 71_003,
        device=device,
    )
    null_evaluations = _deck_evaluations(
        null_model,
        {regime: condition_values[regime] for regime in REGIMES},
        labels,
        device,
    )
    cuts: dict[str, Any] = {}
    for cut_index, cut in enumerate(REPRESENTATION_CUTS):
        semantic_train, semantic_values, semantic_scaling = _standardize(
            features["train"][cut],
            {
                name: values[cut]
                for name, values in features.items()
                if name != "train"
            },
        )
        semantic_model, semantic_fit = _fit_probe(
            semantic_train,
            semantic_values["validation"],
            targets["train"],
            targets["validation"],
            output_width=1,
            classification=False,
            config=config,
            seed=seed + cut_index * 1_009 + 81_007,
            device=device,
        )
        semantic_evaluations = _semantic_evaluations(
            semantic_model,
            {regime: semantic_values[regime] for regime in REGIMES},
            targets,
            device,
        )
        joint_train = torch.cat((semantic_train, condition_train), dim=-1)
        joint_values = {
            name: torch.cat((semantic_values[name], condition_values[name]), dim=-1)
            for name in ("validation", *REGIMES)
        }
        deck_model, deck_fit = _fit_probe(
            joint_train,
            joint_values["validation"],
            labels["train"],
            labels["validation"],
            output_width=3,
            classification=True,
            config=config,
            seed=seed + cut_index * 1_009 + 91_009,
            device=device,
        )
        deck_evaluations = _deck_evaluations(
            deck_model,
            {regime: joint_values[regime] for regime in REGIMES},
            labels,
            device,
        )
        for regime in REGIMES:
            deck_evaluations[regime]["condition_only_cross_entropy"] = (
                null_evaluations[regime]["cross_entropy"]
            )
            deck_evaluations[regime]["conditional_log_loss_gain"] = (
                null_evaluations[regime]["cross_entropy"]
                - deck_evaluations[regime]["cross_entropy"]
            )
        cuts[cut] = {
            "semantic": {
                "fit": semantic_fit,
                "evaluations": semantic_evaluations,
                "input_width": int(semantic_train.shape[1]),
                "scaling_sha256": stage0._digest_tensors(
                    semantic_scaling["mean"], semantic_scaling["scale"]
                ),
            },
            "conditional_deck": {
                "fit": deck_fit,
                "evaluations": deck_evaluations,
                "input_width": int(joint_train.shape[1]),
            },
        }
    return {
        "cuts": cuts,
        "condition_only_null": {
            "fit": null_fit,
            "evaluations": null_evaluations,
            "scaling_sha256": stage0._digest_tensors(
                condition_scaling["mean"], condition_scaling["scale"]
            ),
        },
    }


def task_metrics_from_logits(
    logits: torch.Tensor,
    target_posteriors: torch.Tensor,
    target_bins: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    posterior = torch.softmax(logits.float(), dim=-1).cpu()
    centers = torch.linspace(-1.0, 1.0, posterior.shape[1])
    predicted = posterior @ centers
    predicted_bins = posterior.argmax(-1)
    predicted_angle = torch.acos(predicted.clamp(-1.0, 1.0))
    target_angle = torch.acos(target.float().clamp(-1.0, 1.0))
    return {
        "exact_bin_accuracy": float(
            (predicted_bins == target_bins.cpu()).float().mean()
        ),
        "target_cross_entropy": float(
            -(
                target_posteriors.cpu()
                * posterior.clamp_min(1e-12).log()
            ).sum(-1).mean()
        ),
        "posterior_mean_correlation": _correlation(predicted, target.cpu()),
        "posterior_mean_rmse": float(
            torch.sqrt((predicted.double() - target.cpu().double()).square().mean())
        ),
        "mean_triple_angle_error_radians": float(
            (predicted_angle - target_angle.cpu()).abs().mean()
        ),
        "predicted_bin_coverage": int(torch.unique(predicted_bins).numel()),
    }


def _task_logits_from_state(
    system: stage0.C3TemporalTinyLLM,
    state: torch.Tensor,
    answer_ids: torch.Tensor,
) -> torch.Tensor:
    query = state[:, -1, :]
    logits = system.model.lm_head(system.model.transformer["ln_f"](query))
    return logits.index_select(-1, answer_ids)


def continue_from_cut(
    system: stage0.C3TemporalTinyLLM,
    state: torch.Tensor,
    cut: str,
) -> torch.Tensor:
    if cut == "frontend":
        value = state
        for block in system.model.transformer["h"]:
            value = block(value)
        return value
    block0 = system.model.transformer["h"][0]
    if cut == "post_attention":
        value = state + block0.mlp(block0.ln_2(state))
        for block in system.model.transformer["h"][1:]:
            value = block(value)
        return value
    if cut == "post_mlp":
        value = state
        for block in system.model.transformer["h"][1:]:
            value = block(value)
        return value
    if cut == "full":
        return state
    raise ValueError(f"unknown causal cut {cut!r}")


def _preservation(
    natural: Mapping[str, Any], patched: Mapping[str, Any]
) -> dict[str, Any]:
    accuracy_loss = float(natural["exact_bin_accuracy"] - patched["exact_bin_accuracy"])
    cross_entropy_increase = float(
        patched["target_cross_entropy"] - natural["target_cross_entropy"]
    )
    angle_increase = float(
        patched["mean_triple_angle_error_radians"]
        - natural["mean_triple_angle_error_radians"]
    )
    passed = (
        accuracy_loss <= 0.03
        and cross_entropy_increase <= 0.10
        and angle_increase <= math.pi / 16.0
    )
    return {
        "accuracy_loss": accuracy_loss,
        "cross_entropy_increase": cross_entropy_increase,
        "triple_angle_error_increase": angle_increase,
        "pass": passed,
    }


@torch.no_grad()
def causal_orbit_analysis(
    system: stage0.C3TemporalTinyLLM,
    dataset: stage0.C3TrainingDataset,
    task: stage0.C3TaskConfig,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if len(dataset) % 3:
        raise ValueError("causal analysis requires complete C3 orbits")
    system.eval()
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    collected = {cut: [] for cut in CAUSAL_CUTS}
    natural_logits = []
    # ``batch_size`` is a latent-orbit count. Preserve the same flattened
    # three-sheet batch shape here and in identity continuation so the replay
    # gate measures graph identity rather than CUDA kernel batch-shape drift.
    sheet_batch_size = 3 * batch_size
    for start in range(0, len(dataset), sheet_batch_size):
        stop = start + sheet_batch_size
        cuts = system.residual_cuts(
            dataset.tokens[start:stop].to(device),
            dataset.calibration[start:stop].to(device),
        )
        for cut in CAUSAL_CUTS:
            collected[cut].append(cuts[cut].cpu())
        natural_logits.append(
            _task_logits_from_state(system, cuts["full"], answer_ids).cpu()
        )
    states = {cut: torch.cat(values) for cut, values in collected.items()}
    natural_logits_tensor = torch.cat(natural_logits)
    latent_count = len(dataset) // 3
    latent_target = dataset.target[0::3]
    latent_posterior = dataset.target_posteriors[0::3]
    latent_bins = dataset.target_bins[0::3]
    natural = task_metrics_from_logits(
        natural_logits_tensor,
        dataset.target_posteriors,
        dataset.target_bins,
        dataset.target,
    )
    permutation = target_derangement(dataset)
    cells: dict[str, Any] = {}
    for cut in CAUSAL_CUTS:
        state = states[cut].reshape(latent_count, 3, *states[cut].shape[1:])
        barycenter = state.mean(1)
        action_error = float((state - state[:, :1]).abs().max())
        replay_logits = []
        barycenter_logits = []
        deranged_logits = []
        for start in range(0, latent_count, batch_size):
            stop = start + batch_size
            replay_state = state[start:stop].flatten(0, 1).to(device)
            replay_final = continue_from_cut(system, replay_state, cut)
            replay_logits.append(
                _task_logits_from_state(system, replay_final, answer_ids).cpu()
            )
            patched = barycenter[start:stop].to(device)
            patched_final = continue_from_cut(system, patched, cut)
            barycenter_logits.append(
                _task_logits_from_state(system, patched_final, answer_ids).cpu()
            )
            deranged = barycenter[permutation[start:stop]].to(device)
            deranged_final = continue_from_cut(system, deranged, cut)
            deranged_logits.append(
                _task_logits_from_state(system, deranged_final, answer_ids).cpu()
            )
        replay = torch.cat(replay_logits)
        barycenter_output = torch.cat(barycenter_logits)
        deranged_output = torch.cat(deranged_logits)
        replay_error = float((replay - natural_logits_tensor).abs().max())
        patched_metrics = task_metrics_from_logits(
            barycenter_output,
            latent_posterior,
            latent_bins,
            latent_target,
        )
        deranged_metrics = task_metrics_from_logits(
            deranged_output,
            latent_posterior,
            latent_bins,
            latent_target,
        )
        cells[cut] = {
            "maximum_orbit_state_error": action_error,
            "maximum_identity_replay_logit_error": replay_error,
            "orbit_barycenter_metrics": patched_metrics,
            "orbit_barycenter_preservation": _preservation(natural, patched_metrics),
            "target_derangement_metrics": deranged_metrics,
            "target_derangement_preservation": _preservation(natural, deranged_metrics),
        }
    return {
        "natural_metrics": natural,
        "target_derangement_sha256": stage0._digest_tensors(permutation),
        "cuts": cells,
    }


def _continue_after_sublayer(
    system: stage0.C3TemporalTinyLLM,
    state: torch.Tensor,
    block_index: int,
    sublayer: str,
) -> torch.Tensor:
    value = state
    block = system.model.transformer["h"][block_index]
    if sublayer == "attention":
        value = value + block.mlp(block.ln_2(value))
    for later in system.model.transformer["h"][block_index + 1 :]:
        value = later(value)
    return value


def _sublayer(
    block: nn.Module, state: torch.Tensor, kind: str
) -> torch.Tensor:
    if kind == "attention":
        return state + block.attn(block.ln_1(state))
    if kind == "mlp":
        return state + block.mlp(block.ln_2(state))
    raise ValueError(kind)


@torch.no_grad()
def raw_reynolds_analysis(
    system: stage0.C3TemporalTinyLLM,
    dataset: stage0.C3TrainingDataset,
    task: stage0.C3TaskConfig,
    *,
    device: torch.device,
    latent_limit: int = 256,
) -> dict[str, Any]:
    """Decompose exact raw Reynolds defects and symmetry-allowed low orders."""
    if system.arm != "raw":
        return {"status": "not_applicable_structured_arm"}
    latent_count = min(latent_limit, len(dataset) // 3)
    selected = slice(0, 3 * latent_count)
    tokens = dataset.tokens[selected].to(device)
    calibration = dataset.calibration[selected].to(device)
    cuts = system.residual_cuts(tokens, calibration)
    state = cuts["frontend"].reshape(latent_count, 3, *cuts["frontend"].shape[1:])
    answer_ids = torch.tensor(task.answer_token_ids, dtype=torch.long, device=device)
    natural_logits = _task_logits_from_state(
        system, cuts["full"], answer_ids
    ).cpu()
    natural = task_metrics_from_logits(
        natural_logits,
        dataset.target_posteriors[selected],
        dataset.target_bins[selected],
        dataset.target[selected],
    )
    latent_target = dataset.target[selected][0::3]
    latent_posterior = dataset.target_posteriors[selected][0::3]
    latent_bins = dataset.target_bins[selected][0::3]
    records = []
    first_synthesis: str | None = None
    omega_angles = -2.0 * math.pi * torch.arange(3, device=device) / 3.0
    char_real = torch.cos(omega_angles)
    char_imag = torch.sin(omega_angles)
    for block_index, block in enumerate(system.model.transformer["h"]):
        for kind in ("attention", "mlp"):
            barycenter = state.mean(1)
            delta = state - barycenter[:, None]
            sheet_next = _sublayer(
                block, state.flatten(0, 1), kind
            ).reshape_as(state)
            actual_next = sheet_next.mean(1)
            propagated = _sublayer(block, barycenter, kind)
            defect = actual_next - propagated
            symmetric_terms = []
            for sheet in range(3):
                plus = _sublayer(block, barycenter + delta[:, sheet], kind)
                minus = _sublayer(block, barycenter - delta[:, sheet], kind)
                symmetric_terms.append(0.5 * (plus + minus - 2.0 * propagated))
            quadratic = torch.stack(symmetric_terms).mean(0)
            higher = defect - quadratic
            propagated_final = _continue_after_sublayer(
                system, propagated, block_index, kind
            )
            actual_final = _continue_after_sublayer(
                system, actual_next, block_index, kind
            )
            propagated_metrics = task_metrics_from_logits(
                _task_logits_from_state(system, propagated_final, answer_ids).cpu(),
                latent_posterior,
                latent_bins,
                latent_target,
            )
            actual_metrics = task_metrics_from_logits(
                _task_logits_from_state(system, actual_final, answer_ids).cpu(),
                latent_posterior,
                latent_bins,
                latent_target,
            )
            propagated_pass = _preservation(natural, propagated_metrics)["pass"]
            actual_pass = _preservation(natural, actual_metrics)["pass"]
            name = f"block{block_index}_{kind}"
            if first_synthesis is None and not propagated_pass and actual_pass:
                first_synthesis = name
            c1_real = torch.einsum("bjtd,j->btd", delta, char_real) / 3.0
            c1_imag = torch.einsum("bjtd,j->btd", delta, char_imag) / 3.0
            defect_norm = torch.linalg.vector_norm(defect.double()).clamp_min(1e-12)
            records.append(
                {
                    "sublayer": name,
                    "propagated_barycenter_pass": propagated_pass,
                    "actual_next_barycenter_pass": actual_pass,
                    "regime": (
                        "cover_required"
                        if not propagated_pass and not actual_pass
                        else "synthesis"
                        if not propagated_pass and actual_pass
                        else "closed"
                        if propagated_pass and actual_pass
                        else "corruption"
                    ),
                    "exact_defect_relative_norm": float(
                        defect_norm
                        / torch.linalg.vector_norm(actual_next.double()).clamp_min(1e-12)
                    ),
                    "quadratic_residual_fraction": float(
                        torch.linalg.vector_norm((defect - quadratic).double())
                        / defect_norm
                    ),
                    "higher_order_fraction": float(
                        torch.linalg.vector_norm(higher.double()) / defect_norm
                    ),
                    "first_character_energy": float(
                        c1_real.double().square().mean()
                        + c1_imag.double().square().mean()
                    ),
                    "propagated_metrics": propagated_metrics,
                    "actual_next_metrics": actual_metrics,
                }
            )
            state = sheet_next
    return {
        "status": "completed",
        "latent_histories": latent_count,
        "first_synthesis_sublayer": first_synthesis,
        "sublayers": records,
        "method_boundary": (
            "Symmetric quadratic terms are local finite-difference approximations; "
            "the exact Reynolds defect and frozen continuation define synthesis."
        ),
    }
