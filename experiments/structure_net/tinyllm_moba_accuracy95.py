#!/usr/bin/env python3
"""Task-supervised MoBA adaptation toward 95% held-out BLiMP accuracy."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.structure_net import tinyllm_moba_distillation as base

SCHEMA_VERSION = "nal.tinyllm-moba-accuracy95.v1"
HYPOTHESIS_ID = "tinyllm-moba-accuracy95-v1"
PROTOCOL = Path("docs/07 - Status Reports/2026-09-09_tinyllm-moba-accuracy95-optimization.md")
PARENT = Path("data/experiments/tinyllm_moba_distillation/20260909_registered")
PARENT_SHA = "abcb16c37bc37ffc62afe8f1060075944f86225ae9af38fc82acd8c47f86f6d6"
BASE_SHA = "3c863504bc4898863ca2a8e8ff1b1e16230a59dd3061f495c19a53366c91f792"


@dataclass(frozen=True)
class AccuracyConfig:
    device: str = "cuda:0"
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    sequence_length: int = 256
    block_size: int = 32
    target_k: int = 2
    pair_batch_size: int = 8
    check_every: int = 100
    validation_windows: int = 16
    final_windows: int = 32
    validation_tokens: int = 262144
    margin: float = 1.0
    good_lm_weight: float = 0.1
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    validation_accuracy_min: float = 0.98
    final_accuracy_min: float = 0.95
    nll_ratio_max: float = 1.05
    logit_temperature: float = 1.0
    route_temperature: float = 1.0
    shakedown: bool = False

    def __post_init__(self):
        if self.sequence_length != 256 or self.block_size != 32 or self.target_k != 2:
            raise ValueError("architecture and sparsity budget are fixed")
        if min(self.pair_batch_size, self.check_every, self.validation_windows, self.final_windows) < 1:
            raise ValueError("counts must be positive")
        if not 0 < self.final_accuracy_min <= self.validation_accuracy_min <= 1:
            raise ValueError("invalid accuracy thresholds")
        if not self.shakedown and not self.device.startswith("cuda"):
            raise ValueError("optimization requires CUDA")


@dataclass(frozen=True)
class Candidate:
    name: str
    learning_rate: float
    replay_lm_weight: float
    replay_kd_weight: float
    max_steps: int


CANDIDATES = (
    Candidate("pair_replay", 2e-5, .25, .5, 1600),
    Candidate("pair_replay_slow", 1e-5, .25, .5, 3200),
    Candidate("pair_replay_strong", 2e-5, .5, 1.0, 1600),
)


def split_pairs(records):
    """Deterministic ID splits with complete-sentence leakage exclusion."""
    spans = {"historical": (0, 32), "train": (32, 800), "validation": (800, 900), "test": (900, 1000)}
    parts = {name: [r for r in records if lo <= int(r["pairID"]) < hi] for name, (lo, hi) in spans.items()}
    def sentences(rows):
        return {r[k].strip().casefold() for r in rows for k in ("sentence_good", "sentence_bad")}
    protected = sentences(parts["historical"] + parts["test"])
    parts["validation"] = [r for r in parts["validation"] if not sentences([r]) & protected]
    protected |= sentences(parts["validation"])
    parts["train"] = [r for r in parts["train"] if not sentences([r]) & protected]
    if sentences(parts["train"]) & protected:
        raise RuntimeError("training/evaluation sentence overlap")
    if sentences(parts["historical"]) & sentences(parts["test"]):
        raise RuntimeError("historical and fresh tests overlap")
    return parts


def load_pairs(config):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(base.TOKENIZER)
    if tokenizer.token_to_id("<bos>") != 1:
        raise RuntimeError("BOS mismatch")
    records = []
    for suite, expected in base.BLIMP_HASHES.items():
        path = base.BLIMP_ROOT / (suite + ".jsonl")
        if base.sha256(path) != expected:
            raise RuntimeError("BLiMP source hash mismatch")
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    parts = split_pairs(records)
    for rows in parts.values():
        for row in rows:
            row["good"] = [1] + tokenizer.encode(row["sentence_good"]).ids
            row["bad"] = [1] + tokenizer.encode(row["sentence_bad"]).ids
            if max(len(row["good"]), len(row["bad"])) - 1 > config.sequence_length:
                raise RuntimeError("sentence exceeds native length; no truncation")
    return parts


def pair_batch(rows, device):
    sequences = [r[k] for r in rows for k in ("good", "bad")]
    length = max(len(s) - 1 for s in sequences)
    ids = torch.zeros(len(sequences), length, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    for index, seq in enumerate(sequences):
        ids[index, :len(seq) - 1] = torch.tensor(seq[:-1], device=device)
        labels[index, :len(seq) - 1] = torch.tensor(seq[1:], device=device)
    return ids, labels


def pair_objective(logits, labels, config):
    nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="none").reshape_as(labels)
    scores = -nll.sum(-1)
    differences = scores[0::2] - scores[1::2]
    pair_loss = F.softplus(config.margin - differences).mean()
    good_lm = nll[0::2].sum() / (labels[0::2] != -100).sum()
    return pair_loss + config.good_lm_weight * good_lm, scores, {"pair_loss": pair_loss.detach(), "good_lm": good_lm.detach()}


@torch.no_grad()
def evaluate_pairs(model, rows, config):
    model.eval()
    records = []
    for start in range(0, len(rows), config.pair_batch_size):
        selected = rows[start:start + config.pair_batch_size]
        ids, labels = pair_batch(selected, config.device)
        logits, _ = base.forward(model, ids, config, config.target_k, capture=False)
        _, scores, _ = pair_objective(logits, labels, config)
        values = scores.cpu().tolist()
        for index, row in enumerate(selected):
            good, bad = values[2 * index:2 * index + 2]
            score = float(good > bad) if good != bad else .5
            records.append({"suite": row["UID"], "pair_id": row["pairID"], "good_logp": good, "bad_logp": bad, "accuracy": score})
    return {"accuracy": sum(r["accuracy"] for r in records) / len(records), "count": len(records),
            "per_suite": {suite: float(np.mean([r["accuracy"] for r in records if r["suite"] == suite]))
                          for suite in sorted({r["suite"] for r in records})}, "records": records}


def make_schedules(token_count, pair_count, seed, config):
    max_steps = 4 if config.shakedown else max(c.max_steps for c in CANDIDATES)
    split = token_count - config.validation_tokens
    width = config.sequence_length + 1
    half = config.validation_tokens // 2
    pair_indices = np.random.default_rng((seed, 95117)).integers(0, pair_count, (max_steps, config.pair_batch_size))
    replay = np.random.default_rng((seed, 95217)).integers(0, split - width + 1, max_steps)
    val_bins = np.arange(half // width) * width + split
    validation = np.random.default_rng((seed, 95317)).choice(val_bins, config.validation_windows, replace=False)
    exposed = set()
    for parent_seed in (7, 17, 29, 41, 53):
        with np.load(PARENT / "schedules" / f"seed_{parent_seed}.npz") as old:
            exposed.update(int(v) for v in old["test"])
    test_bins = np.array([v for v in np.arange(half // width) * width + split + half if int(v) not in exposed])
    test = np.random.default_rng((seed, 95417)).choice(test_bins, config.final_windows, replace=False)
    return {"pairs": pair_indices, "replay": replay, "validation": validation, "test": test}


@torch.no_grad()
def text_nll(model, tokens, starts, config, *, full=False):
    total = 0.0
    for start in starts:
        x, y = base.batch(tokens, [start], config)
        logits, _ = base.forward(model, x, config, 8 if full else config.target_k, capture=False)
        total += float(F.cross_entropy(logits.flatten(0, 1), y.flatten()))
    return total / len(starts)


def validation_gate(accuracy, nll, teacher_nll, config):
    return accuracy >= config.validation_accuracy_min and nll <= config.nll_ratio_max * teacher_nll


def final_gate(historical_accuracy, fresh_accuracy, nll, teacher_nll, config):
    accuracy = historical_accuracy >= config.final_accuracy_min and fresh_accuracy >= config.final_accuracy_min
    quality = nll <= config.nll_ratio_max * teacher_nll
    return {"accuracy": accuracy, "nll": quality, "joint": accuracy and quality}


def load_model(path, device, teacher=False):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = base.TinyLLMModel(base.TinyLLMConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval().requires_grad_(not teacher)


def parent_checkpoint(seed):
    path = PARENT / "runs" / "KD" / f"seed_{seed}" / "result.json"
    parent = json.loads((PARENT / "campaign_results.json").read_text())
    expected = next(r["sha256"] for r in parent["results"] if r["path"] == str(path))
    if base.sha256(path) != expected:
        raise RuntimeError("parent per-run evidence hash mismatch")
    result = json.loads(path.read_text())
    artifact = result["artifacts"]["checkpoint"]
    if base.sha256(artifact["path"]) != artifact["sha256"]:
        raise RuntimeError("parent KD checkpoint hash mismatch")
    return artifact


def manifest(root, config, mode, pairs, selection=None):
    if base.sha256(PARENT / "campaign_results.json") != PARENT_SHA or base.sha256(base.__file__) != BASE_SHA:
        raise RuntimeError("frozen parent evidence or helper changed")
    paths = sorted(Path("src/structure_net").rglob("*.py")) + [Path(__file__), Path(base.__file__), PROTOCOL]
    sources = {str(p.resolve().relative_to(Path.cwd())): base.sha256(p) for p in paths}
    inputs = {base.CHECKPOINT: base.sha256(base.CHECKPOINT), base.TOKEN_CACHE: base.sha256(base.TOKEN_CACHE),
              base.TOKENIZER: base.sha256(base.TOKENIZER),
              **{str(base.BLIMP_ROOT / (s + ".jsonl")): h for s, h in base.BLIMP_HASHES.items()}}
    if inputs[base.CHECKPOINT] != base.CHECKPOINT_SHA256 or inputs[base.TOKEN_CACHE] != base.TOKEN_SHA256:
        raise RuntimeError("teacher or token hash mismatch")
    identities = {name: [f"{r['UID']}/{r['pairID']}" for r in rows] for name, rows in pairs.items()}
    value = {"schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
             "configuration": asdict(config), "candidates": [asdict(c) for c in CANDIDATES], "mode": mode,
             "sources": sources, "inputs": inputs, "pairs": identities, "selection": selection,
             "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                             "device": config.device, "device_name": torch.cuda.get_device_name() if config.device.startswith("cuda") else "CPU",
                             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda": torch.version.cuda}}
    value["fingerprint"] = base.digest(value)
    existing = root / "producer.json"
    if existing.exists():
        if json.loads(existing.read_text())["fingerprint"] != value["fingerprint"]:
            raise RuntimeError("producer changed; preserve previous root")
    else:
        for path in sources:
            target = root / "producer_source" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        base.write_json(existing, {**value, "created_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv})
    return value


def save_checkpoint(path, model, fingerprint, step):
    base.atomic_save(path, {"model_config": asdict(model.config), "model_state": model.state_dict(),
                            "scientific_fingerprint": fingerprint, "step": step, "moba_top_k": 2, "moba_block_size": 32})
    return {"path": str(path), "sha256": base.sha256(path), "step": step}


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_candidate(seed, candidate, config, producer, root, teacher, tokens, parts, schedule, *, fixed_steps=None):
    directory = root / "runs" / candidate.name / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    initial = parent_checkpoint(7 if config.shakedown else seed)
    schedule_path = root / "schedules" / f"seed_{seed}.npz"
    fingerprint = base.digest({"producer": producer["fingerprint"], "candidate": asdict(candidate), "seed": seed,
                               "schedule": base.sha256(schedule_path), "initial_checkpoint": initial, "fixed_steps": fixed_steps})
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result["scientific_fingerprint"] != fingerprint or result["status"] != "completed":
            raise RuntimeError("completed candidate fingerprint/status mismatch")
        for item in result["history"]:
            if base.sha256(item["checkpoint"]["path"]) != item["checkpoint"]["sha256"]:
                raise RuntimeError("saved checkpoint changed")
        return result
    if any(directory.iterdir()):
        raise RuntimeError("incomplete candidate: preserve its evidence and use a new root")
    start = time.monotonic()
    torch.manual_seed(seed)
    model = load_model(initial["path"], config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=candidate.learning_rate, betas=(.9, .95),
                                 eps=1e-8, weight_decay=config.weight_decay)
    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    validation_rows = parts["validation"][:8] if config.shakedown else parts["validation"]
    teacher_nll = text_nll(teacher, tokens, schedule["validation"], config, full=True)
    history = []
    def validate(step, checkpoint):
        accuracy = evaluate_pairs(model, validation_rows, config)
        nll = text_nll(model, tokens, schedule["validation"], config)
        record = {"step": step, "validation": accuracy, "nll": nll, "teacher_nll": teacher_nll,
                  "nll_ratio": nll / teacher_nll,
                  "pass": validation_gate(accuracy["accuracy"], nll, teacher_nll, config), "checkpoint": checkpoint}
        history.append(record)
        base.write_json(directory / f"validation_step{step}.json", record)
        print(json.dumps({"candidate": candidate.name, "seed": seed, "step": step,
                          "validation_accuracy": accuracy["accuracy"], "nll_ratio": nll / teacher_nll,
                          "pass": record["pass"]}), flush=True)
    validate(0, {**initial, "step": 0})
    end = fixed_steps or candidate.max_steps
    selected = False
    stage_losses = {"pair_loss": 0.0, "good_lm": 0.0, "replay_lm": 0.0, "replay_kd": 0.0}
    try:
        for step in range(1, end + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            rows = [parts["train"][int(index)] for index in schedule["pairs"][step - 1]]
            ids, labels = pair_batch(rows, config.device)
            logits, _ = base.forward(model, ids, config, config.target_k, capture=False)
            pair_loss, scores, losses = pair_objective(logits, labels, config)
            pair_loss.backward()
            for name, value in losses.items():
                stage_losses[name] += float(value)
            del logits, scores, pair_loss, losses
            x, y = base.batch(tokens, [schedule["replay"][step - 1]], config)
            with torch.no_grad():
                tl, _ = base.forward(teacher, x, config, 8, capture=False)
            sl, _ = base.forward(model, x, config, config.target_k, capture=False)
            lm = F.cross_entropy(sl.flatten(0, 1), y.flatten())
            kd = base.logits_kl(sl, tl)
            replay_loss = candidate.replay_lm_weight * lm + candidate.replay_kd_weight * kd
            replay_loss.backward()
            stage_losses["replay_lm"] += float(lm.detach())
            stage_losses["replay_kd"] += float(kd.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            del sl, tl, replay_loss, lm, kd
            if step % config.check_every == 0 or step == end:
                checkpoint = save_checkpoint(directory / f"checkpoint_step{step}.pt", model, fingerprint, step)
                validate(step, checkpoint)
                if not fixed_steps and not config.shakedown and len(history) >= 3 and all(r["pass"] for r in history[-2:]):
                    selected = True
                    break
        # Verify the final persisted model has tensor-exact reload behavior.
        checkpoint = history[-1]["checkpoint"]
        state = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
        if any(not torch.equal(v.detach().cpu(), state["model_state"][k]) for k, v in model.state_dict().items()):
            raise RuntimeError("checkpoint reload mismatch")
        result = {"schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
                  "status": "completed", "candidate": asdict(candidate), "seed": seed,
                  "scientific_fingerprint": fingerprint, "configuration": asdict(config),
                  "profile": "systems_lifecycle_only_not_quality_evidence" if config.shakedown else "EXPLORATORY_TASK_SUPERVISED",
                  "selected": selected, "training_steps": step, "checkpoint": checkpoint,
                  "initial_checkpoint": initial, "history": history,
                  "training_pair_presentations": step * config.pair_batch_size,
                  "replay_target_tokens": step * config.sequence_length,
                  "mean_losses": {k: v / step for k, v in stage_losses.items()}, "checkpoint_reload_exact": True,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if config.device.startswith("cuda") else None,
                  "elapsed_seconds": time.monotonic() - start, "errors": []}
        base.write_json(result_path, result)
        return result
    except Exception as exc:
        base.write_json(directory / f"failure_{time.time_ns()}.json", {"status": "failed", "error": repr(exc), "scientific_fingerprint": fingerprint})
        raise
    finally:
        del model, optimizer
        if config.device.startswith("cuda"):
            torch.cuda.empty_cache()


def run(config, root, mode, selection_path=None):
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".producer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if config.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required; no fallback")
            torch.cuda.set_device(torch.device(config.device))
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        parts = load_pairs(config)
        selection = None
        if selection_path:
            selection = json.loads(selection_path.read_text())
            if selection["classification"] != "candidate_selected" or selection["profile"] != "EXPLORATORY_TASK_SUPERVISED":
                raise RuntimeError("a non-systems candidate selection is required")
        producer = manifest(root, config, mode, parts, selection)
        if (root / "campaign_results.json").exists():
            raise RuntimeError("completed aggregate is immutable; inspect it instead of rerunning")
        tokens = np.load(base.TOKEN_CACHE, mmap_mode="r")
        teacher = load_model(base.CHECKPOINT, config.device, teacher=True)
        parent_teacher_hash = state_hash(teacher)
        (root / "schedules").mkdir(exist_ok=True)
        schedules = {}
        for seed in config.seeds:
            schedules[seed] = make_schedules(len(tokens), len(parts["train"]), seed, config)
            path = root / "schedules" / f"seed_{seed}.npz"
            if path.exists():
                with np.load(path) as stored:
                    if any(not np.array_equal(v, stored[k]) for k, v in schedules[seed].items()):
                        raise RuntimeError("schedule changed")
            else:
                np.savez(path, **schedules[seed])
        results = []
        if mode == "search":
            seed = config.seeds[0]
            for candidate in CANDIDATES:
                if config.shakedown:
                    candidate = replace(candidate, max_steps=4)
                result = train_candidate(seed, candidate, config, producer, root, teacher, tokens, parts, schedules[seed])
                results.append(result)
                if result["selected"] or config.shakedown:
                    break
            chosen = next((r for r in results if r["selected"]), None)
            aggregate = {"classification": "systems_lifecycle_only_not_quality_evidence" if config.shakedown else ("candidate_selected" if chosen else "search_target_not_reached"),
                         "selection": {"candidate": chosen["candidate"], "training_steps": chosen["training_steps"], "checkpoint": chosen["checkpoint"]} if chosen else None,
                         "test_evaluated": False, "attempted_candidates": len(results),
                         "candidate_results": [{"candidate": r["candidate"], "training_steps": r["training_steps"], "last_validation_accuracy": r["history"][-1]["validation"]["accuracy"],
                                                "last_nll_ratio": r["history"][-1]["nll_ratio"], "selected": r["selected"], "checkpoint": r["checkpoint"]} for r in results]}
        else:
            if selection is None:
                raise RuntimeError("replication needs a frozen selection")
            chosen = selection["selection"]
            candidate = Candidate(**chosen["candidate"])
            if candidate not in CANDIDATES or not 0 < chosen["training_steps"] <= candidate.max_steps:
                raise RuntimeError("selected configuration outside the declared search")
            if base.sha256(chosen["checkpoint"]["path"]) != chosen["checkpoint"]["sha256"]:
                raise RuntimeError("selected checkpoint changed")
            for seed in config.seeds:
                if seed == 7:
                    result = {"checkpoint": chosen["checkpoint"], "scientific_fingerprint": base.digest({
                        "producer": producer["fingerprint"], "reused_selection": chosen, "seed": seed}),
                        "reused_selected_checkpoint": True, "training_steps": chosen["training_steps"]}
                    base.write_json(root / "runs" / candidate.name / f"seed_{seed}" / "reuse.json", result)
                else:
                    result = train_candidate(seed, candidate, config, producer, root, teacher, tokens, parts, schedules[seed], fixed_steps=chosen["training_steps"])
                model = load_model(result["checkpoint"]["path"], config.device)
                historical = evaluate_pairs(model, parts["historical"], config)
                fresh = evaluate_pairs(model, parts["test"], config)
                nll = text_nll(model, tokens, schedules[seed]["test"], config)
                teacher_nll = text_nll(teacher, tokens, schedules[seed]["test"], config, full=True)
                final = {"seed": seed, "historical": historical, "fresh": fresh, "nll": nll, "teacher_nll": teacher_nll,
                         "nll_ratio": nll / teacher_nll, "gate": final_gate(historical["accuracy"], fresh["accuracy"], nll, teacher_nll, config),
                         "checkpoint": result["checkpoint"], "scientific_fingerprint": result["scientific_fingerprint"]}
                base.write_json(root / "runs" / candidate.name / f"seed_{seed}" / "final_evaluation.json", final)
                results.append(final)
                del model
                torch.cuda.empty_cache()
                print(json.dumps({"seed": seed, "historical_accuracy": historical["accuracy"], "fresh_accuracy": fresh["accuracy"], "nll_ratio": nll / teacher_nll, "gate": final["gate"]}), flush=True)
            aggregate = {"classification": "accuracy95_achieved" if all(r["gate"]["joint"] for r in results) else "accuracy95_not_confirmed",
                         "selection": chosen, "test_evaluated": True, "per_seed": results,
                         "summary": {"requested": len(config.seeds), "completed": len(results), "failed": 0,
                                     "accuracy_passes": sum(r["gate"]["accuracy"] for r in results), "joint_passes": sum(r["gate"]["joint"] for r in results)}}
        after_hash = state_hash(teacher)
        if after_hash != parent_teacher_hash:
            raise RuntimeError("teacher mutated")
        aggregate.update(schema_version=SCHEMA_VERSION, hypothesis_id=HYPOTHESIS_ID, status="completed",
                         profile="SHAKEDOWN" if config.shakedown else "EXPLORATORY_TASK_SUPERVISED",
                         configuration=asdict(config), producer_fingerprint=producer["fingerprint"],
                         completed_at=datetime.now(timezone.utc).isoformat(), teacher_frozen=True)
        base.write_json(root / "campaign_results.json", aggregate)
        print(json.dumps({"classification": aggregate["classification"], "output": str(root)}), flush=True)
        return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("search", "replicate"), default="search")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--shakedown", action="store_true")
    args = parser.parse_args(argv)
    config = AccuracyConfig(device=args.device, **({"shakedown": True, "seeds": (997,), "check_every": 2, "validation_windows": 1, "final_windows": 1} if args.shakedown else {}))
    run(config, args.output, args.mode, args.selection)


if __name__ == "__main__":
    main()
