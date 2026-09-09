#!/usr/bin/env python3
"""Preregistered, checkpoint-paired MoBA distillation; dense-mask reference."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from structure_net.components.models.tinyllm_model import TinyLLMConfig, TinyLLMModel

SCHEMA_VERSION = "nal.tinyllm-moba-distillation.v1"
HYPOTHESIS_ID = "tinyllm-moba-distillation-v1"
PREREGISTRATION = "docs/07 - Status Reports/2026-09-09_tinyllm-moba-distillation-preregistration.md"
CHECKPOINT = "data/experiments/tinyllm_babylm_pretrain/20260812_d8_seed7/checkpoint_step12000.pt"
CHECKPOINT_SHA256 = "5a7491b4231c30feaaabda7babf0a831dd4d72fbb4e7f3e757114cadf098df09"
TOKEN_CACHE = "data/corpora/babylm_10M_bpe16k.tokens.npy"
TOKEN_SHA256 = "f339655453b970ae6cc1cbdc7c78f8a5234c42437bc7bba8fb31a1dee5c9d765"
TOKENIZER = "data/corpora/babylm_10M_bpe16k.tokenizer.json"
BLIMP_ROOT = Path("data/corpora/blimp_moba_20260909")
BLIMP_HASHES = {
    "regular_plural_subject_verb_agreement_1": "1a18d94062c8e792c0a200a3b00dff0e051a92bb084c05e9b5e5c73bee32a620",
    "regular_plural_subject_verb_agreement_2": "e5b4668811e63b00ac4c9569cb9553281a39c21df43256905ad78e59ae24942c",
    "anaphor_number_agreement": "05c16b8bfe4feb5b3552dc2c53d1742ce0c91d9a1c896684335bdddd4373cc9b",
    "determiner_noun_agreement_1": "f48065fd760d0fd1a895012860d38bdf3c2ac115cd221e521684cc430c0f1855",
}
ARMS = ("F", "M0", "LM", "KD", "route_mass", "route_contribution",
        "combined_mass", "combined_contribution")


@dataclass(frozen=True)
class StudyConfig:
    seeds: tuple[int, ...] = (7, 17, 29, 41, 53)
    sequence_length: int = 256
    block_size: int = 32
    steps: int = 128
    train_batch_size: int = 1
    monitor_windows: int = 4
    test_windows: int = 16
    validation_tokens: int = 262144
    learning_rate: float = 3e-5
    betas: tuple[float, float] = (0.9, 0.95)
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    route_temperature: float = 1.0
    logit_temperature: float = 1.0
    lm_weight: float = 1.0
    logits_weight: float = 1.0
    route_weight: float = 1.0
    hidden_weight: float = 0.1
    attention_weight: float = 0.1
    target_k: int = 2
    evaluation_ks: tuple[int, ...] = (1, 2, 4, 8)
    stage_ks: tuple[int, ...] = (8, 4, 2, 2)
    parity_tolerance: float = 1e-4
    material_kl: float = 0.001
    kl_ratio_max: float = 0.8
    recall_gain_min: float = 0.05
    nll_ratio_max: float = 1.05
    required_seeds: int = 4
    blimp_pairs_per_suite: int = 32
    device: str = "cuda:0"
    shakedown: bool = False
    synthetic: bool = False

    def __post_init__(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if self.steps < 4 or self.steps % 4:
            raise ValueError("four equal positive training stages required")
        if self.block_size <= 0 or self.sequence_length <= 0 or self.sequence_length % self.block_size:
            raise ValueError("sequence must divide into blocks")
        if min(self.train_batch_size, self.monitor_windows, self.test_windows) < 1:
            raise ValueError("batch and evaluation counts must be positive")
        if self.target_k not in self.evaluation_ks or self.target_k < 2:
            raise ValueError("target k must include local and at least one routed block")
        if self.validation_tokens // 2 < (self.sequence_length + 1) * max(self.test_windows, self.monitor_windows):
            raise ValueError("held-out partitions too short")
        if any(not math.isfinite(v) or v <= 0 for v in (
            self.learning_rate, self.route_temperature, self.logit_temperature,
            self.gradient_clip, self.adam_epsilon, self.parity_tolerance,
        )):
            raise ValueError("invalid numerical configuration")
        if not self.shakedown:
            locked = {
                "seeds": (7, 17, 29, 41, 53), "sequence_length": 256,
                "block_size": 32, "steps": 128, "train_batch_size": 1,
                "monitor_windows": 4, "test_windows": 16, "validation_tokens": 262144,
                "learning_rate": 3e-5, "betas": (0.9, 0.95), "adam_epsilon": 1e-8,
                "weight_decay": 0.01, "gradient_clip": 1.0, "route_temperature": 1.0,
                "logit_temperature": 1.0, "lm_weight": 1.0, "logits_weight": 1.0,
                "route_weight": 1.0, "hidden_weight": 0.1, "attention_weight": 0.1,
                "target_k": 2, "evaluation_ks": (1, 2, 4, 8), "stage_ks": (8, 4, 2, 2),
                "parity_tolerance": 1e-4, "material_kl": 0.001,
                "kl_ratio_max": 0.8, "recall_gain_min": 0.05, "nll_ratio_max": 1.05,
                "required_seeds": 4, "blimp_pairs_per_suite": 32, "synthetic": False,
            }
            if any(getattr(self, k) != v for k, v in locked.items()):
                raise ValueError("primary configuration is preregistered and frozen")
            if not self.device.startswith("cuda"):
                raise ValueError("primary campaign requires CUDA")
        if self.synthetic and not self.shakedown:
            raise ValueError("synthetic data is systems-only")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text)
    temp.replace(path)


def atomic_save(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def past_blocks(length, block_size, device):
    query_block = torch.arange(length, device=device) // block_size
    return torch.arange(math.ceil(length / block_size), device=device)[None, :] < query_block[:, None]


def routing_scores(q, k, block_size):
    means = torch.stack([part.mean(-2) for part in k.split(block_size, dim=-2)], dim=-2)
    return q @ means.transpose(-1, -2)


def select_blocks(scores, block_size, top_k):
    """Exactly top-k including local; no future/local key means influence choice."""
    length, count = scores.shape[-2:]
    eligible = past_blocks(length, block_size, scores.device)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    n_remote = min(max(top_k - 1, 0), count)
    if n_remote:
        order = scores.masked_fill(~eligible, -torch.inf).argsort(dim=-1, descending=True, stable=True)
        mask.scatter_(-1, order[..., :n_remote], True)
        mask &= eligible
    local = torch.arange(length, device=scores.device) // block_size
    mask.scatter_(-1, local.view(1, 1, length, 1).expand(*scores.shape[:-1], 1), True)
    return mask


def attention(q, k, v, block_size, top_k):
    scores = routing_scores(q, k, block_size)
    selected = select_blocks(scores, block_size, top_k)
    length = q.shape[-2]
    causal = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
    allowed = selected.repeat_interleave(block_size, -1)[..., :length] & causal
    probabilities = ((q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])).masked_fill(~allowed, -torch.inf).softmax(-1)
    return probabilities @ v, probabilities, scores, selected, allowed


def teacher_targets(probabilities, v, projection, block_size):
    """Per-head projected contribution norm via W_O^T W_O, without bias."""
    length = v.shape[-2]
    blocks = math.ceil(length / block_size)
    padded_p = F.pad(probabilities, (0, blocks * block_size - length))
    mass = padded_p.reshape(*probabilities.shape[:-1], blocks, block_size).sum(-1)
    c = torch.stack([probabilities[..., start:start + block_size] @ v[..., start:start + block_size, :]
                     for start in range(0, length, block_size)], dim=-2)
    heads, dim = v.shape[1], v.shape[-1]
    w = projection.reshape(projection.shape[0], heads, dim).permute(1, 0, 2)
    gram = w.transpose(-1, -2) @ w
    contribution = torch.einsum("bhtnd,hde,bhtne->bhtn", c, gram, c).clamp_min(0).sqrt()
    return mass, contribution


def normalize_target(values, eligible, fallback=None):
    values = values * eligible
    total = values.sum(-1, keepdim=True)
    normalized = values / total.clamp_min(1e-30)
    if fallback is not None:
        normalized = torch.where(total > 1e-30, normalized, normalize_target(fallback, eligible))
    return normalized


def route_kl(scores, target, block_size, temperature=1.0):
    eligible = past_blocks(scores.shape[-2], block_size, scores.device)
    rows = eligible.any(-1)
    if not rows.any():
        return scores.sum() * 0
    logits = (scores[..., rows, :] / temperature).masked_fill(~eligible[rows], -torch.inf)
    logp = logits.log_softmax(-1).masked_fill(~eligible[rows], 0)
    p = target[..., rows, :].detach()
    return (p * (p.clamp_min(1e-30).log() - logp)).sum(-1).mean()


def forward(model, ids, config, top_k, *, capture=True, targets=False):
    """Baseline TinyLLM graph with attention replaced, preserving all weights."""
    if ids.shape[1] > model.config.block_size or model.refinement_steps or len(model.feedback_connections):
        raise ValueError("requires native-length baseline TinyLLM")
    length = ids.shape[1]
    x = model.transformer["wte"](ids) + model.transformer["wpe"](torch.arange(length, device=ids.device))
    traces = []
    for block in model.transformer["h"]:
        a = block.attn
        batch, _, channels = x.shape
        qkv = a.c_attn(block.ln_1(x)).split(channels, -1)
        q, k, v = [t.reshape(batch, length, a.n_head, channels // a.n_head).transpose(1, 2) for t in qkv]
        attended, prob, scores, selected, allowed = attention(q, k, v, config.block_size, top_k)
        out = a.c_proj(attended.transpose(1, 2).reshape(batch, length, channels))
        x = x + out
        x = x + block.mlp(block.ln_2(x))
        if capture:
            trace = {"scores": scores, "selected": selected, "attention": out, "hidden": x,
                     "edge_fraction": allowed.float().sum() / (batch * a.n_head * length * (length + 1) / 2)}
            if targets:
                mass, contribution = teacher_targets(prob, v, a.c_proj.weight, config.block_size)
                eligible = past_blocks(length, config.block_size, ids.device)
                trace.update(mass_raw=mass, mass=normalize_target(mass, eligible),
                             contribution=normalize_target(contribution, eligible, mass))
            traces.append(trace)
    return model.lm_head(model.transformer["ln_f"](x)), traces


def logits_kl(student, teacher, temperature=1.0):
    logp = (student / temperature).log_softmax(-1)
    logq = (teacher.detach() / temperature).log_softmax(-1)
    return (logq.exp() * (logq - logp)).sum(-1).mean() * temperature ** 2


def loss_terms(student_logits, teacher_logits, student, teacher, labels, config):
    zero = student_logits.sum() * 0
    terms = {"lm": F.cross_entropy(student_logits.flatten(0, 1), labels.flatten()),
             "logits": logits_kl(student_logits, teacher_logits, config.logit_temperature),
             "hidden": zero, "attention": zero, "route_mass": zero, "route_contribution": zero}
    for s, t in zip(student, teacher):
        terms["hidden"] = terms["hidden"] + F.mse_loss(s["hidden"], t["hidden"].detach()) / len(student)
        terms["attention"] = terms["attention"] + F.mse_loss(s["attention"], t["attention"].detach()) / len(student)
        for name in ("mass", "contribution"):
            terms["route_" + name] = terms["route_" + name] + route_kl(s["scores"], t[name], config.block_size, config.route_temperature) / len(student)
    return terms


def objective(terms, arm, stage, config):
    value = config.lm_weight * terms["lm"]
    if stage == 3 or arm == "LM":
        return value
    if arm == "KD" or arm.startswith("combined"):
        value = value + config.logits_weight * terms["logits"]
    if arm.startswith(("route_", "combined_")):
        target = arm.split("_", 1)[1]
        value = value + config.route_weight * terms["route_" + target]
    if arm.startswith("combined"):
        value = value + config.hidden_weight * terms["hidden"]
        if stage == 1:
            value = value + config.attention_weight * terms["attention"]
    return value


def configure_stage(model, stage, config):
    for name, p in model.named_parameters():
        p.requires_grad_(stage >= 2 or (".attn." in name and (stage == 1 or ".c_attn." in name)))
        p.grad = None
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=config.learning_rate, betas=config.betas, eps=config.adam_epsilon,
                            weight_decay=0 if stage == 0 else config.weight_decay)


def mask_value_gradients(model):
    for block in model.transformer["h"]:
        a = block.attn
        for p in (a.c_attn.weight, a.c_attn.bias):
            if p.grad is not None:
                p.grad[2 * a.n_embd:] = 0


def make_schedules(token_count, seed, config):
    length = config.sequence_length + 1
    split = token_count - config.validation_tokens
    if split < length:
        raise ValueError("training prefix too short")
    rng = np.random.default_rng((seed, 1103))
    train = rng.integers(0, split - length + 1, (config.steps, config.train_batch_size))
    half = config.validation_tokens // 2
    def windows(start, count, salt):
        bins = np.arange(half // length) * length + start
        return np.random.default_rng((seed, salt)).choice(bins, count, replace=False)
    return {"train": train, "monitor": windows(split, config.monitor_windows, 2203),
            "test": windows(split + half, config.test_windows, 3301)}


def batch(tokens, starts, config):
    rows = np.stack([tokens[int(start):int(start) + config.sequence_length + 1] for start in np.atleast_1d(starts)])
    rows = torch.tensor(rows, dtype=torch.long, device=config.device)
    return rows[:, :-1], rows[:, 1:]


def recall(scores, target, block_size, top_k):
    if top_k <= 1:
        return None
    eligible = past_blocks(scores.shape[-2], block_size, scores.device)
    selected = select_blocks(scores, block_size, top_k) & eligible
    oracle = select_blocks(target, block_size, top_k) & eligible
    rows = eligible.any(-1)
    denominator = eligible.sum(-1).clamp_max(top_k - 1)[rows]
    return ((selected & oracle).sum(-1)[..., rows] / denominator).float().mean()


@torch.no_grad()
def evaluate(model, teacher, tokens, starts, config, top_k):
    model.eval()
    total = {}
    for start in starts:
        x, y = batch(tokens, [start], config)
        tl, tt = forward(teacher, x, config, math.ceil(x.shape[1] / config.block_size), targets=True)
        sl, st = forward(model, x, config, top_k)
        terms = loss_terms(sl, tl, st, tt, y, config)
        metrics = {"nll": terms["lm"], "logits_kl": terms["logits"],
                   "hidden_mse": terms["hidden"], "attention_mse": terms["attention"],
                   "route_mass_kl": terms["route_mass"], "route_contribution_kl": terms["route_contribution"],
                   "next_token_accuracy": (sl.argmax(-1) == y).float().mean(),
                   "edge_fraction": torch.stack([s["edge_fraction"] for s in st]).mean(),
                   "teacher_mass_captured": torch.stack([(s["selected"] * t["mass_raw"]).sum(-1).mean() for s, t in zip(st, tt)]).mean()}
        for target in ("mass", "contribution"):
            values = [recall(s["scores"], t[target], config.block_size, top_k) for s, t in zip(st, tt)]
            if values[0] is not None:
                metrics[target + "_recall"] = torch.stack(values).mean()
        for name, value in metrics.items():
            total[name] = total.get(name, 0.0) + float(value) / len(starts)
    total["perplexity"] = math.exp(total["nll"])
    total["tokens"] = len(starts) * config.sequence_length
    if top_k <= 1:
        total.update(mass_recall=None, contribution_recall=None,
                     recall_status="undefined_no_routed_blocks_at_k1")
    digest(total)  # Reject all nonfinite measurements before persisting.
    return total


def load_blimp(config):
    if config.synthetic:
        return []
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER)
    if tokenizer.token_to_id("<bos>") != 1:
        raise ValueError("corpus BOS identity mismatch")
    pairs = []
    for suite in BLIMP_HASHES:
        records = [json.loads(s) for s in (BLIMP_ROOT / (suite + ".jsonl")).read_text().splitlines()]
        for record in records[:config.blimp_pairs_per_suite]:
            good, bad = [[1] + tokenizer.encode(record[key]).ids for key in ("sentence_good", "sentence_bad")]
            if max(len(good), len(bad)) - 1 > config.sequence_length:
                raise ValueError("BLiMP sentence exceeds native length; truncation forbidden")
            pairs.append({"suite": suite, "id": record["pairID"], "good": good, "bad": bad})
    return pairs


@torch.no_grad()
def evaluate_blimp(model, pairs, config, top_k):
    if not pairs:
        return {"status": "not_applicable_synthetic_shakedown"}
    records, suites = [], {}
    for pair in pairs:
        scores = []
        for key in ("good", "bad"):
            ids = torch.tensor([pair[key]], device=config.device)
            logits, _ = forward(model, ids[:, :-1], config, top_k, capture=False)
            scores.append(-float(F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten(), reduction="sum")))
        accuracy = float(scores[0] > scores[1]) if scores[0] != scores[1] else 0.5
        suites.setdefault(pair["suite"], []).append(accuracy)
        records.append({"suite": pair["suite"], "id": pair["id"], "good_logp": scores[0],
                        "bad_logp": scores[1], "accuracy": accuracy,
                        "sparsity_active": max(len(pair["good"]), len(pair["bad"])) - 1 > top_k * config.block_size})
    return {"status": "completed", "accuracy": sum(r["accuracy"] for r in records) / len(records),
            "per_suite": {k: sum(v) / len(v) for k, v in suites.items()}, "pair_count": len(records),
            "sparsity_active_fraction": sum(r["sparsity_active"] for r in records) / len(records), "records": records}


def seed_gate(full, zero, lm, combined, config):
    gates = {"kl": combined["logits_kl"] <= config.kl_ratio_max * lm["logits_kl"],
             "recall": combined["mass_recall"] >= zero["mass_recall"] + config.recall_gain_min,
             "nll": combined["nll"] <= config.nll_ratio_max * full["nll"]}
    return {**gates, "joint": all(gates.values())}


def classify(gates, config):
    if config.shakedown:
        return "systems_lifecycle_only_not_quality_evidence"
    if len(gates) != len(config.seeds):
        return "incomplete"
    if sum(g["joint"] for g in gates) >= config.required_seeds:
        return "routing_distillation_supported"
    if sum(g["kl"] for g in gates) >= config.required_seeds:
        return "behavioral_gain_without_full_joint_gate"
    return "distillation_not_confirmed"


def producer_manifest(config, root):
    paths = sorted(Path("src/structure_net").rglob("*.py")) + [Path(__file__).resolve(), Path(PREREGISTRATION)]
    source = {str(p.resolve().relative_to(Path.cwd())): sha256(p) for p in paths}
    data = {} if config.synthetic else {CHECKPOINT: sha256(CHECKPOINT), TOKEN_CACHE: sha256(TOKEN_CACHE),
                                      TOKENIZER: sha256(TOKENIZER),
                                      **{str(BLIMP_ROOT / (s + ".jsonl")): sha256(BLIMP_ROOT / (s + ".jsonl")) for s in BLIMP_HASHES}}
    if not config.synthetic:
        expected = {CHECKPOINT: CHECKPOINT_SHA256, TOKEN_CACHE: TOKEN_SHA256,
                    **{str(BLIMP_ROOT / (s + ".jsonl")): h for s, h in BLIMP_HASHES.items()}}
        if any(data[p] != h for p, h in expected.items()):
            raise RuntimeError("input scientific hash mismatch")
    environment = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                   "cuda": torch.version.cuda, "device": config.device,
                   "device_name": torch.cuda.get_device_name() if config.device.startswith("cuda") else "CPU",
                   "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "tf32": False,
                   "deterministic_algorithms": True, "cpu_threads": torch.get_num_threads()}
    identity = {"source": source, "data": data, "configuration": asdict(config), "environment": environment}
    fingerprint = digest(identity)
    path = root / "producer.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["fingerprint"] != fingerprint:
            raise RuntimeError("producer changed; use a new dated root")
        for p, h in source.items():
            if sha256(root / "producer_source" / p) != h:
                raise RuntimeError("preserved source hash mismatch")
        return existing
    for p in source:
        target = root / "producer_source" / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    result = {**identity, "fingerprint": fingerprint, "created_at": datetime.now(timezone.utc).isoformat(),
              "command": sys.argv, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
              "scheduler": {"workers": 1, "slots_per_gpu": 1, "isolated_timing": False}}
    write_json(path, result)
    return result


def reuse_result(path, fingerprint):
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result.get("status") != "completed" or result.get("scientific_fingerprint") != fingerprint or result.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("refuse result reuse: status/schema/fingerprint mismatch")
    for artifact in result["artifacts"].values():
        if not Path(artifact["path"]).exists() or sha256(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("refuse result reuse: artifact hash mismatch")
    return result


def run_arm(arm, seed, teacher, tokens, schedules, config, root, producer, pairs, baseline=None):
    directory = root / "runs" / arm / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    schedule_path = root / "schedules" / f"seed_{seed}.npz"
    fingerprint = digest({"producer": producer["fingerprint"], "arm": arm, "seed": seed,
                          "schedule": sha256(schedule_path), "schema": SCHEMA_VERSION, "hypothesis": HYPOTHESIS_ID})
    result_path = directory / "result.json"
    reused = reuse_result(result_path, fingerprint)
    if reused is not None:
        return reused, True
    if any(directory.iterdir()):
        raise RuntimeError(f"incomplete evidence at {directory}; preserve it and use a new root")
    start = time.monotonic()
    torch.manual_seed(seed)
    model = copy.deepcopy(teacher).requires_grad_(False)
    if config.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    history = []
    losses = []
    if arm not in ("F", "M0"):
        history.append({"step": 0, "metrics": baseline["monitor"]})
        for stage in range(4):
            optimizer = configure_stage(model, stage, config)
            stage_loss = {}
            for step in range(stage * config.steps // 4, (stage + 1) * config.steps // 4):
                x, y = batch(tokens, schedules["train"][step], config)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                sl, st = forward(model, x, config, config.stage_ks[stage])
                # Teacher-free final stage and LM control perform no teacher forward.
                if stage == 3 or arm == "LM":
                    terms = {"lm": F.cross_entropy(sl.flatten(0, 1), y.flatten())}
                else:
                    with torch.no_grad():
                        tl, tt = forward(teacher, x, config, config.sequence_length // config.block_size, targets=True)
                    terms = loss_terms(sl, tl, st, tt, y, config)
                value = objective(terms, arm, stage, config)
                if not torch.isfinite(value):
                    raise RuntimeError("nonfinite training objective")
                value.backward()
                if stage == 0:
                    mask_value_gradients(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                optimizer.step()
                for key, v in {**terms, "objective": value, "gradient_norm": norm}.items():
                    stage_loss[key] = stage_loss.get(key, 0.0) + float(v.detach()) / (config.steps // 4)
                del sl, st, terms, value
            # Release graph/logit references before checkpoint evaluation.
            if stage != 3 and arm != "LM":
                del tl, tt
            del optimizer
            step = (stage + 1) * config.steps // 4
            metrics = evaluate(model, teacher, tokens, schedules["monitor"], config, config.target_k)
            history.append({"step": step, "metrics": metrics})
            losses.append({"stage": stage + 1, "top_k": config.stage_ks[stage], "losses": stage_loss})
            print(json.dumps({"arm": arm, "seed": seed, "step": step,
                              "monitor_nll": metrics["nll"], "monitor_kl": metrics["logits_kl"]}), flush=True)
    ks = (config.sequence_length // config.block_size,) if arm == "F" else config.evaluation_ks
    curve = {str(k): evaluate(model, teacher, tokens, schedules["test"], config, k) for k in ks}
    primary = curve[str(ks[0] if arm == "F" else config.target_k)]
    monitor = evaluate(model, teacher, tokens, schedules["monitor"], config, ks[0] if arm == "F" else config.target_k) if arm in ("F", "M0") else history[-1]["metrics"]
    blimp = evaluate_blimp(model, pairs, config, ks[0] if arm == "F" else config.target_k)
    difference = sum(float((p.detach() - q.detach()).square().sum()) for p, q in zip(model.parameters(), teacher.parameters()))
    reference = sum(float(p.detach().square().sum()) for p in teacher.parameters())
    checkpoint = directory / "model.pt"
    atomic_save(checkpoint, {"model_config": asdict(model.config), "model_state": model.state_dict(),
                             "moba": {"block_size": config.block_size, "top_k": ks[0] if arm == "F" else config.target_k},
                             "scientific_fingerprint": fingerprint})
    # Reload tensor state and check both weights and reference outputs exactly.
    reloaded = torch.load(checkpoint, map_location=config.device, weights_only=True)
    if any(not torch.equal(value, reloaded["model_state"][key]) for key, value in model.state_dict().items()):
        raise RuntimeError("checkpoint reload tensor mismatch")
    x, _ = batch(tokens, [schedules["test"][0]], config)
    with torch.no_grad():
        before, _ = forward(model, x, config, config.target_k, capture=False)
        model.load_state_dict(reloaded["model_state"], strict=True)
        after, _ = forward(model, x, config, config.target_k, capture=False)
        reload_error = float((before - after).abs().max())
    if reload_error != 0:
        raise RuntimeError("checkpoint reload output mismatch")
    del reloaded, before, after
    result = {"schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
              "experiment_id": f"{HYPOTHESIS_ID}/{arm}/seed_{seed}", "arm": arm, "seed": seed,
              "status": "completed", "scientific_fingerprint": fingerprint,
              "configuration": asdict(config), "producer_fingerprint": producer["fingerprint"],
              "profile": "systems_lifecycle_only_not_quality_evidence" if config.shakedown else "PREREGISTERED",
              "metrics": primary, "primary_metric": primary["logits_kl"], "curve": curve,
              "monitor": monitor, "history": history, "stage_losses": losses, "blimp": blimp,
              "training_updates": 0 if arm in ("F", "M0") else config.steps,
              "training_tokens": 0 if arm in ("F", "M0") else config.steps * config.train_batch_size * config.sequence_length,
              "teacher_free_final_updates": 0 if arm in ("F", "M0") else config.steps // 4,
              "relative_parameter_deformation": math.sqrt(difference / reference),
              "checkpoint_reload_max_error": reload_error,
              "model_parameters": sum(p.numel() for p in model.parameters()),
              "elapsed_seconds": time.monotonic() - start,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if config.device.startswith("cuda") else None,
              "artifacts": {"checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
                            "schedule": {"path": str(schedule_path), "sha256": sha256(schedule_path)}},
              "errors": []}
    write_json(result_path, result)
    del model
    if config.device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(json.dumps({"completed": arm, "seed": seed, "seconds": result["elapsed_seconds"],
                      "peak_MB": (result["peak_cuda_allocated_bytes"] or 0) / 1e6}), flush=True)
    return result, False


def run_campaign(config, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # An exclusive process lock prevents duplicate live producers, including resume.
    import fcntl
    with (root / ".producer.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_campaign(config, root)


def _run_campaign(config, root):
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if config.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(config.device))
    producer = producer_manifest(config, root)
    if (root / "campaign_results.json").exists():
        aggregate = json.loads((root / "campaign_results.json").read_text())
        if aggregate.get("status") != "completed" or aggregate["producer"]["fingerprint"] != producer["fingerprint"]:
            raise RuntimeError("existing aggregate is not reusable")
        for artifact in aggregate["results"]:
            if sha256(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError("completed per-run result changed")
            stored = json.loads(Path(artifact["path"]).read_text())
            reuse_result(Path(artifact["path"]), stored["scientific_fingerprint"])
        print(json.dumps({"status": "completed", "reused_campaign": True, "classification": aggregate["classification"]}), flush=True)
        return aggregate
    if config.synthetic:
        model_config = TinyLLMConfig(block_size=config.sequence_length, vocab_size=64, n_layer=2, n_head=2, n_embd=16)
        teacher = TinyLLMModel(model_config).to(config.device).eval().requires_grad_(False)
        tokens = np.random.default_rng(901).integers(0, 64, 8192, dtype=np.int32)
    else:
        payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model_config = TinyLLMConfig(**payload["model_config"])
        if (model_config.n_layer, model_config.n_head, model_config.n_embd, model_config.block_size, model_config.vocab_size) != (8, 8, 512, 256, 50257):
            raise RuntimeError("teacher architecture mismatch")
        teacher = TinyLLMModel(model_config)
        teacher.load_state_dict(payload["model_state"], strict=True)
        teacher = teacher.to(config.device).eval().requires_grad_(False)
        del payload
        tokens = np.load(TOKEN_CACHE, mmap_mode="r")
    teacher_state_hash = digest({k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k, v in teacher.state_dict().items()})
    pairs = load_blimp(config)
    schedules = {}
    (root / "schedules").mkdir(exist_ok=True)
    parity = []
    for seed in config.seeds:
        schedules[seed] = make_schedules(len(tokens), seed, config)
        path = root / "schedules" / f"seed_{seed}.npz"
        if path.exists():
            with np.load(path) as previous:
                if any(not np.array_equal(v, previous[k]) for k, v in schedules[seed].items()):
                    raise RuntimeError("schedule reuse mismatch")
        else:
            np.savez(path, **schedules[seed])
        x, _ = batch(tokens, [schedules[seed]["test"][0]], config)
        with torch.no_grad():
            expected, _ = teacher(x, return_full_logits=True)
            actual, _ = forward(teacher, x, config, config.sequence_length // config.block_size, capture=False)
            err = float((actual - expected).abs().max())
        parity.append({"seed": seed, "max_absolute_logit_error": err,
                       "pass": math.isfinite(err) and err <= config.parity_tolerance})
    write_json(root / "validity.json", {"parity": parity, "teacher_state_sha256": teacher_state_hash})
    results, reused_count = [], 0
    def execute(arm, seed, baseline=None):
        nonlocal reused_count
        result, reused = run_arm(arm, seed, teacher, tokens, schedules[seed], config, root, producer, pairs, baseline)
        results.append(result)
        reused_count += int(reused)
        return result
    classification = "invalid_full_budget_conversion"
    gates = []
    try:
        if all(p["pass"] for p in parity):
            zero = {}
            for seed in config.seeds:
                execute("F", seed)
                zero[seed] = execute("M0", seed)
            material = sum(r["metrics"]["logits_kl"] >= config.material_kl for r in zero.values())
            classification = "no_material_conversion_gap"
            if config.shakedown or material >= config.required_seeds:
                for seed in config.seeds:
                    for arm in ARMS[2:]:
                        execute(arm, seed, zero[seed])
                    by_arm = {r["arm"]: r for r in results if r["seed"] == seed}
                    gate = seed_gate(*(by_arm[a]["metrics"] for a in ("F", "M0", "LM", "combined_mass")), config)
                    gates.append({"seed": seed, **gate})
                classification = classify(gates, config)
    except Exception as exc:
        write_json(root / f"failure_{time.time_ns()}.json", {"status": "failed", "error": repr(exc),
                   "completed_results": len(results), "producer_fingerprint": producer["fingerprint"]})
        raise
    final_teacher_hash = digest({k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k, v in teacher.state_dict().items()})
    if final_teacher_hash != teacher_state_hash:
        raise RuntimeError("frozen teacher mutated")
    arm_means = {}
    for arm in ARMS:
        selected = [r for r in results if r["arm"] == arm]
        if selected:
            arm_means[arm] = {key: float(np.mean([r["metrics"][key] for r in selected])) for key in selected[0]["metrics"] if isinstance(selected[0]["metrics"][key], (float, int))}
            blimp_values = [r["blimp"].get("accuracy") for r in selected]
            if blimp_values[0] is not None:
                arm_means[arm]["blimp_accuracy"] = float(np.mean(blimp_values))
    aggregate = {"schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
                 "status": "completed", "classification": classification,
                 "confirmed": classification == "routing_distillation_supported" and not config.shakedown,
                 "profile": "SHAKEDOWN" if config.shakedown else "PREREGISTERED",
                 "configuration": asdict(config), "producer": producer,
                 "completed_at": datetime.now(timezone.utc).isoformat(), "validity": parity,
                 "teacher_unchanged": True, "per_seed_gates": gates, "arm_means": arm_means,
                 "summary": {"requested": len(config.seeds) * len(ARMS), "completed": len(results),
                             "failed": 0, "retried": 0, "excluded": 0, "reused": reused_count,
                             "unexecuted_by_gate": len(config.seeds) * len(ARMS) - len(results),
                             "joint_passes": sum(g["joint"] for g in gates)},
                 "results": [{"experiment_id": r["experiment_id"],
                              "path": str(root / "runs" / r["arm"] / f"seed_{r['seed']}" / "result.json"),
                              "sha256": sha256(root / "runs" / r["arm"] / f"seed_{r['seed']}" / "result.json")} for r in results],
                 "limitations": ["One pretrained checkpoint; seeds are adaptation schedules.",
                                 "Dense-mask reference: no sparse-kernel acceleration claim.",
                                 "Projected contribution norm is not an ablation-based causal effect.",
                                 "Native context 256; no long-context extrapolation.",
                                 "One fixed token budget cannot establish sample-complexity scaling.",
                                 "BLiMP is a small fixed subset; text-level training overlap unaudited.",
                                 "Parent validation suffix reused, with disjoint adaptation/monitor/test tokens."]}
    write_json(root / "campaign_results.json", aggregate)
    print(json.dumps({"classification": classification, "summary": aggregate["summary"]}), flush=True)
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shakedown", action="store_true", help="Four updates, one seed; systems-only")
    parser.add_argument("--synthetic", action="store_true", help="Tiny random model/data, requires --shakedown")
    args = parser.parse_args(argv)
    if args.shakedown:
        config = StudyConfig(device=args.device, shakedown=True, synthetic=args.synthetic,
                             seeds=(997,), steps=4, monitor_windows=1, test_windows=1,
                             blimp_pairs_per_suite=1,
                             **({"sequence_length": 32, "block_size": 4, "validation_tokens": 2048} if args.synthetic else {}))
    elif args.synthetic:
        parser.error("--synthetic requires --shakedown")
    else:
        config = StudyConfig(device=args.device)
    run_campaign(config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
