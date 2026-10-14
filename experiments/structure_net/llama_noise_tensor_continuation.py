"""Paired noisy tensor continuation and block-coordinate global Llama recovery."""
import argparse
import copy
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.structure_net.llama_neighbourhood_tensorization import (
    digest, factor_state, generate, install_factors, load_example, load_teacher, run_blocks)
from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH, sha256, write_json

HYPOTHESIS_ID = "llama32-3b-noise-tensor-continuation-v1"
SCHEMA_VERSION = "nal.llama-noise-tensor-continuation.v1"
BASE = Path("data/experiments/llama_neighbourhood_tensorization")
FIXTURES = BASE / "20261013_fixtures_v1.json"
CACHE = BASE / "20261013_cache_v1/cache.json"
PARENT = BASE / "20261013_training_v4/training.json"
PLAN = Path("docs/07 - Status Reports/2026-10-14_llama32-noise-tensor-continuation-plan.md")
RESEARCH = Path("docs/08 - Analysis/2026-10-14_embedded-neighbourhood-distillation-research.md")


@dataclass(frozen=True)
class Config:
    rank: int = 1024
    seed: int = 41014
    local_steps: int = 256
    local_lr: float = 0.00002
    interval: int = 64
    weight_noise: float = 0.01
    input_noise: float = 0.05
    global_steps: int = 32
    global_lr: float = 0.00001
    energy_floor: float = 1e-8
    scaler_initial: float = 16.
    clip_norm: float = 1.
    response_weight: float = 0.5
    kl_weight: float = 0.5
    ce_weight: float = 0.5

    def __post_init__(self):
        if self.rank != 1024 or min(self.local_steps, self.global_steps, self.interval) < 1:
            raise ValueError("fixed rank and positive update counts required")
        for name in ("local_lr", "global_lr", "energy_floor", "scaler_initial", "clip_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError("positive finite optimizer settings required")
        for name in ("weight_noise", "input_noise"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("nonnegative finite noise required")
        if self.response_weight != .5 or self.kl_weight != .5 or self.ce_weight != .5:
            raise ValueError("equal fixed objective weights required")


def gaussian_like(x, scale, generator):
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("invalid noise")
    return torch.randn(x.shape, dtype=torch.float32, device=x.device, generator=generator) * scale


@torch.no_grad()
def perturb_factors(parameters, amount, generator):
    records = []
    for p in parameters:
        rms = p.float().square().mean().sqrt()
        delta = gaussian_like(p, amount, generator) * rms
        p.add_(delta.to(p.dtype))
        records.append({"factor_rms": float(rms), "noise_rms": float(delta.square().mean().sqrt())})
    return records


def noisy_pair(x, cached_y, teacher_fn, amount, generator):
    """Never train changed inputs against stale original-input targets."""
    if amount == 0:
        return x, cached_y.detach()
    with torch.no_grad():
        channel_rms = x.float().square().mean(dim=(0, 1), keepdim=True).sqrt()
        altered = (x.float() + gaussian_like(x, amount, generator) * channel_rms).to(x.dtype)
        target = teacher_fn(altered).detach()
    return altered, target


def balanced_hidden_loss(prediction, target, prompt_length, floor=1e-8):
    if prediction.shape != target.shape or not 1 <= prompt_length < prediction.shape[1]:
        raise ValueError("paired full sequence and nonempty response required")
    target = target.detach().float()
    relative = (prediction.float() - target).square().mean(-1) / target.square().mean(-1).clamp_min(floor)
    # Hidden at last prompt token predicts first answer; exclude final EOS input.
    return .5 * relative.mean() + .5 * relative[:, prompt_length - 1:-1].mean()


def output_objective(student_logits, teacher_logits, labels):
    if student_logits.shape != teacher_logits.shape or student_logits.shape[:-1] != labels.shape:
        raise ValueError("aligned teacher, student and reference tokens required")
    student = F.log_softmax(student_logits.float(), dim=-1)
    teacher = F.log_softmax(teacher_logits.detach().float(), dim=-1)
    kl = F.kl_div(student, teacher, reduction="none", log_target=True).sum(-1).mean()
    ce = F.cross_entropy(student_logits.float().reshape(-1, student_logits.shape[-1]), labels.reshape(-1))
    return .5 * (kl + ce), kl, ce


def schedule(count, steps, seed):
    rng = random.Random(seed)
    order = list(range(count))
    result = []
    for step in range(steps):
        if step % count == 0:
            rng.shuffle(order)
        result.append(order[step % count])
    return result


def require_cuda():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("one visible CUDA device required; no CPU fallback")


def seal(output, stage, arm, paths=()):
    if output.exists():
        raise ValueError("new producing root required")
    output.mkdir(parents=True)
    paths = [Path(__file__), PLAN, RESEARCH, FIXTURES, CACHE, PARENT,
             Path("experiments/structure_net/llama_neighbourhood_tensorization.py"),
             Path("experiments/structure_net/prepare_llama_cotenancy.py"),
             Path("data/models/Llama-3.2-3B-cotenancy/manifest.json")] + list(paths)
    hashes = {str(p): sha256(p) for p in paths}
    snapshots = output / "producer_source"
    snapshots.mkdir()
    for i, p in enumerate(paths):
        shutil.copyfile(p, snapshots / f"{i:03d}_{p.name}")
    import transformers
    record = {"schema": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
              "stage": stage, "arm": arm, "config": asdict(Config()),
              "source_hashes": hashes, "command": sys.argv, "python": sys.version,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "transformers": transformers.__version__,
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    record["fingerprint"] = digest(record)
    write_json(output / "producer.json", record)
    return record


def prepare(args):
    seal(args.output, "original_teacher_response_targets", "shared")
    fixtures = {e["id"]: e for e in json.loads(FIXTURES.read_text())["examples"] if e["split"] != "heldout"}
    parent = json.loads(PARENT.read_text())
    cache = json.loads(CACHE.read_text())
    if cache["status"] != "completed" or parent["status"] != "completed":
        raise ValueError("completed parent and teacher cache required")
    for group in parent["groups"]:
        if sha256(group["selected_checkpoint"]) != group["selected_sha256"]:
            raise ValueError("changed parent factors")
    dataset = []
    for entry in cache["examples"]:
        e = fixtures[entry["id"]]
        for f in entry["files"]:
            if sha256(f["path"]) != f["sha256"]:
                raise ValueError("changed original teacher cache")
        target = torch.load(entry["files"][-1]["path"], map_location="cpu", weights_only=True)["target"]
        start = len(e["prompt_ids"]) - 1
        response = target[:, start:-1].clone().contiguous()
        labels = torch.tensor([e["training_ids"][start+1:]], dtype=torch.long)
        if response.shape[1] != labels.shape[1]:
            raise ValueError("unaligned teacher response cache")
        dataset.append({"id": e["id"], "split": e["split"], "prompt_length": len(e["prompt_ids"]),
                        "ids": torch.tensor([e["training_ids"]]), "labels": labels, "teacher_response": response})
    path = args.output / "responses.pt"
    torch.save(dataset, path)
    write_json(args.output / "prepared.json", {"status": "completed", "dataset": str(path),
               "dataset_sha256": sha256(path), "train": 64, "development": 8, "heldout_included": False,
               "cache_sha256": sha256(CACHE), "parent_sha256": sha256(PARENT), "fixtures_sha256": sha256(FIXTURES),
               "boundary_files_verified": sum(len(e["files"]) for e in cache["examples"])})


def validated_data(prepared):
    record = json.loads((prepared / "prepared.json").read_text())
    if record["status"] != "completed" or record["heldout_included"] or sha256(record["dataset"]) != record["dataset_sha256"]:
        raise ValueError("invalid prepared teacher responses")
    for name, path in (("cache", CACHE), ("parent", PARENT), ("fixtures", FIXTURES)):
        if sha256(path) != record[name + "_sha256"]:
            raise ValueError("changed continuation inputs")
    return torch.load(record["dataset"], map_location="cpu", weights_only=True)


@torch.no_grad()
def local_dev(blocks, rotary, development, group, fixture_map, config):
    values = []
    for e in development:
        x, y = load_example(e, group)
        values.append(float(balanced_hidden_loss(run_blocks(blocks, x, rotary), y,
                                                len(fixture_map[e["id"]]["prompt_ids"]), config.energy_floor)))
    return sum(values) / len(values)


def gradient_step(loss, optimizer, scaler, parameters, frozen):
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if not all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in parameters):
        raise FloatingPointError("missing, zero or nonfinite active factor gradients")
    if any(p.grad is not None for p in frozen):
        raise RuntimeError("frozen weights received gradients")
    norm = float(nn.utils.clip_grad_norm_(parameters, Config().clip_norm))
    scaler.step(optimizer)
    scaler.update()
    return norm


def local(args):
    require_cuda()
    validated_data(args.prepared)
    config = Config()
    stage = "systems_lifecycle_only_not_quality_evidence" if args.pilot else "local_continuation"
    producer = seal(args.output, stage, args.arm, [args.prepared / "prepared.json"])
    cache = json.loads(CACHE.read_text())
    parent = json.loads(PARENT.read_text())
    fixtures = {e["id"]: e for e in json.loads(FIXTURES.read_text())["examples"]}
    train = [e for e in cache["examples"] if e["split"] == "train"]
    dev = [e for e in cache["examples"] if e["split"] == "development"]
    if args.pilot:
        dev = [max(dev, key=lambda e: e["tokens"])]
    model = load_teacher("cpu")
    rotary = model.model.rotary_emb.to("cuda")
    records = []
    for group in (range(1) if args.pilot else range(14)):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        teacher = nn.ModuleList([copy.deepcopy(b).cuda() for b in model.model.layers[2*group:2*group+2]])
        teacher.eval().requires_grad_(False)
        blocks = copy.deepcopy(teacher)
        state = torch.load(parent["groups"][group]["selected_checkpoint"], map_location="cpu", weights_only=True)
        install_factors(blocks, state, "cuda", trainable=True)
        params = [p for p in blocks.parameters() if p.requires_grad]
        versions = [(p, p._version) for p in blocks.parameters() if not p.requires_grad]
        weight_rng = torch.Generator(device="cuda").manual_seed(config.seed + 100000 + group)
        input_rng = torch.Generator(device="cuda").manual_seed(config.seed + 200000 + group)
        noise = perturb_factors(params, config.weight_noise if args.arm == "noise" else 0., weight_rng)
        initial = factor_state(blocks)
        initial_path = args.output / f"group_{group:02d}_initial.pt"
        torch.save(initial, initial_path)
        evaluation = copy.deepcopy(blocks).half().requires_grad_(False)
        initial_error = local_dev(evaluation, rotary, dev, group, fixtures, config)
        del evaluation
        best, best_step, best_state = initial_error, 0, initial
        history = [{"step": 0, "development_balanced_loss": initial_error}]
        opt = torch.optim.AdamW(params, lr=config.local_lr, weight_decay=0., foreach=False)
        scaler = torch.amp.GradScaler("cuda", init_scale=config.scaler_initial)
        steps = 4 if args.pilot else config.local_steps
        order = schedule(len(train), steps, config.seed + 70000 + group)
        for step in range(1, steps+1):
            e = max(train, key=lambda e: e["tokens"]) if args.pilot else train[order[step-1]]
            x, y = load_example(e, group)
            noisy = args.arm == "noise" and step % 2 == 0
            x, y = noisy_pair(x, y, lambda value: run_blocks(teacher, value, rotary),
                              config.input_noise if noisy else 0., input_rng)
            opt.zero_grad(set_to_none=True)
            prediction = run_blocks(blocks, x, rotary)
            loss = balanced_hidden_loss(prediction, y, len(fixtures[e["id"]]["prompt_ids"]), config.energy_floor)
            norm = gradient_step(loss, opt, scaler, params, list(teacher.parameters()) + [p for p, _ in versions])
            row = {"step": step, "example": e["id"], "training_balanced_loss": float(loss.detach()),
                   "gradient_norm": norm, "input_noise_applied": noisy, "teacher_target_recomputed": noisy}
            del prediction, loss, x, y
            if step % config.interval == 0 or step == steps:
                current = factor_state(blocks)
                evaluation = copy.deepcopy(blocks).half().requires_grad_(False)
                err = local_dev(evaluation, rotary, dev, group, fixtures, config)
                del evaluation
                row["development_balanced_loss"] = err
                if err < best:
                    best, best_step, best_state = err, step, current
                print(json.dumps({"arm": args.arm, "group": group, **row}), flush=True)
            history.append(row)
        path = args.output / f"group_{group:02d}_selected.pt"
        torch.save(best_state, path)
        unchanged = all(p._version == version for p, version in versions)
        if not unchanged:
            raise RuntimeError("frozen block weights changed")
        x, _ = load_example(dev[0], group)
        install_factors(blocks, best_state, "cuda")
        with torch.no_grad():
            expected = run_blocks(blocks, x, rotary)
        install_factors(blocks, torch.load(path, map_location="cpu", weights_only=True), "cuda")
        with torch.no_grad():
            actual = run_blocks(blocks, x, rotary)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        record = {"group": group, "status": "completed", "history": history, "updates": steps,
                  "initial_development_balanced_loss": initial_error, "selected_development_balanced_loss": best,
                  "selected_step": best_step, "selected_checkpoint": str(path), "selected_sha256": sha256(path),
                  "initial_checkpoint": str(initial_path), "initial_sha256": sha256(initial_path),
                  "parent_checkpoint": parent["groups"][group]["selected_checkpoint"],
                  "parent_sha256": parent["groups"][group]["selected_sha256"],
                  "weight_noise": noise, "optimizer_state_resumed": False,
                  "all_factors_have_gradients": True, "teacher_frozen": True, "frozen_weights_unchanged": unchanged,
                  "checkpoint_reload_exact": True, "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                  "seconds": time.perf_counter()-started}
        records.append(record)
        write_json(args.output / f"group_{group:02d}.json", record)
        print(json.dumps({k: v for k, v in record.items() if k not in ("history", "weight_noise")}), flush=True)
        del blocks, teacher, opt, scaler, params, versions, initial, best_state, state, x, actual, expected
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / "training.json", {"status": "completed", "hypothesis_id": HYPOTHESIS_ID,
               "stage": stage, "arm": args.arm, "config": asdict(config), "groups": records,
               "producer_fingerprint": producer["fingerprint"]})


def assembled(root):
    training = json.loads((root / "training.json").read_text())
    if training["status"] != "completed" or len(training["groups"]) != 14:
        raise ValueError("complete local/global factors required")
    model = load_teacher("cpu")
    for g in training["groups"]:
        if sha256(g["selected_checkpoint"]) != g["selected_sha256"]:
            raise ValueError("factor checkpoint changed")
        state = torch.load(g["selected_checkpoint"], map_location="cpu", weights_only=True)
        install_factors(model.model.layers[2*g["group"]:2*g["group"]+2], state, "cpu")
    return model.cuda().eval().requires_grad_(False)


def global_loss(model, example):
    ids, labels = example["ids"].cuda(), example["labels"].cuda()
    start = example["prompt_length"] - 1
    with torch.no_grad():
        teacher_logits = model.lm_head(model.model.norm(example["teacher_response"].cuda())).float()
    hidden = model.model(ids, use_cache=False).last_hidden_state[:, start:-1]
    logits = model.lm_head(hidden).float()
    return output_objective(logits, teacher_logits, labels)


@torch.no_grad()
def global_dev(model, development):
    rows = []
    for e in development:
        loss, kl, ce = global_loss(model, e)
        rows.append({"id": e["id"], "loss": float(loss), "kl": float(kl), "ce": float(ce)})
    return {"rows": rows, "mean_kl": sum(r["kl"] for r in rows)/len(rows),
            "mean_ce": sum(r["ce"] for r in rows)/len(rows)}


def global_train(args):
    require_cuda()
    dataset = validated_data(args.prepared)
    config = Config()
    stage = "systems_lifecycle_only_not_quality_evidence" if args.pilot else "global_recovery"
    producer = seal(args.output, stage, args.arm, [args.prepared / "prepared.json", args.local / "training.json",
                                               args.local / "producer.json"])
    model = assembled(args.local)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    train = [e for e in dataset if e["split"] == "train"]
    dev = [e for e in dataset if e["split"] == "development"]
    if args.pilot:
        dev = [max(dev, key=lambda e: e["ids"].shape[1])]
    base_versions = [(p, p._version) for name, p in model.named_parameters() if not name.endswith((".left", ".right"))]
    records = []
    initial_dev = global_dev(model, dev)
    for group in (range(1) if args.pilot else range(14)):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        blocks = model.model.layers[2*group:2*group+2]
        state = factor_state(blocks)
        install_factors(blocks, state, "cuda", trainable=True)
        params = [p for p in model.parameters() if p.requires_grad]
        expected = {id(p) for b in blocks for p in b.parameters() if p.requires_grad}
        if {id(p) for p in params} != expected or not params:
            raise RuntimeError("gradient routing active group mismatch")
        frozen = [p for p in model.parameters() if not p.requires_grad]
        opt = torch.optim.AdamW(params, lr=config.global_lr, weight_decay=0., foreach=False)
        scaler = torch.amp.GradScaler("cuda", init_scale=config.scaler_initial)
        steps = 2 if args.pilot else config.global_steps
        order = schedule(len(train), steps, config.seed + 300000 + group)
        history = []
        for step in range(1, steps+1):
            e = max(train, key=lambda e: e["ids"].shape[1]) if args.pilot else train[order[step-1]]
            opt.zero_grad(set_to_none=True)
            loss, kl, ce = global_loss(model, e)
            norm = gradient_step(loss, opt, scaler, params, frozen)
            row = {"step": step, "example": e["id"], "loss": float(loss.detach()), "kl": float(kl.detach()),
                   "ce": float(ce.detach()), "gradient_norm": norm}
            history.append(row)
            del loss, kl, ce
        state = factor_state(blocks)
        path = args.output / f"group_{group:02d}_selected.pt"
        torch.save(state, path)
        del opt, scaler, params, frozen
        install_factors(blocks, state, "cuda")
        probe = dev[0]
        with torch.no_grad():
            expected_logits = model.model(probe["ids"].cuda(), use_cache=False).last_hidden_state[:, -2:-1].clone()
        install_factors(blocks, torch.load(path, map_location="cpu", weights_only=True), "cuda")
        with torch.no_grad():
            actual_logits = model.model(probe["ids"].cuda(), use_cache=False).last_hidden_state[:, -2:-1]
        torch.testing.assert_close(expected_logits, actual_logits, rtol=0, atol=0)
        if any(p._version != v for p, v in base_versions):
            raise RuntimeError("frozen original weights changed during global recovery")
        evaluation = global_dev(model, dev)
        record = {"group": group, "status": "completed", "history": history, "updates": steps,
                  "selected_checkpoint": str(path), "selected_sha256": sha256(path), "selection": "final_fixed_step",
                  "development": evaluation, "all_factors_have_gradients": True,
                  "frozen_weights_unchanged": True, "checkpoint_reload_exact": True,
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "seconds": time.perf_counter()-started}
        records.append(record)
        write_json(args.output / f"group_{group:02d}.json", record)
        print(json.dumps({"arm": args.arm, "group": group, "updates": steps, "development_kl": evaluation["mean_kl"],
                          "development_ce": evaluation["mean_ce"], "peak_cuda_bytes": record["peak_cuda_bytes"],
                          "seconds": record["seconds"]}), flush=True)
        del state, blocks, expected_logits, actual_logits
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / "training.json", {"status": "completed", "hypothesis_id": HYPOTHESIS_ID,
               "stage": stage, "arm": args.arm, "config": asdict(config), "groups": records,
               "initial_development": initial_dev, "final_development": records[-1]["development"],
               "producer_fingerprint": producer["fingerprint"], "local_root": str(args.local),
               "local_training_sha256": sha256(args.local / "training.json")})


def evaluate(args):
    require_cuda()
    from transformers import AutoTokenizer
    seal(args.output, args.split + "_quality_and_runtime", args.arm, [args.global_root / "training.json"])
    model = assembled(args.global_root)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    examples = [e for e in json.loads(FIXTURES.read_text())["examples"] if e["split"] == args.split]
    generate(model, examples[0]["prompt_ids"][:64], 8, False)
    rows = []
    for e in examples:
        row = generate(model, e["prompt_ids"], e["cap"])
        row.update(id=e["id"], arm=args.arm, text=tokenizer.decode(row["output_ids"], skip_special_tokens=True),
                   prompt_tokens=e["prompt_tokens"])
        rows.append(row)
        write_json(args.output / "outputs.json", {"schema": SCHEMA_VERSION, "arm": args.arm, "split": args.split,
                   "rows": rows, "parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()),
                   "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "lookup": False})
        print(json.dumps(row), flush=True)
    timing = []
    for repeat in range(3):
        for e in examples[:4]:
            row = generate(model, e["prompt_ids"], 32, False)
            row.pop("output_ids")
            timing.append({"id": e["id"], "repeat": repeat, **row})
    write_json(args.output / "runtime.json", {"arm": args.arm, "rows": timing, "quality_evidence": False,
               "aggregate_tokens_per_second": 32*len(timing)/sum(r["wall_seconds"] for r in timing)})
    write_json(args.output / "result.json", {"status": "completed", "outputs_sha256": sha256(args.output / "outputs.json"),
               "runtime_sha256": sha256(args.output / "runtime.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "local", "global", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--local", type=Path)
    parser.add_argument("--global-root", type=Path)
    parser.add_argument("--arm", choices=("clean", "noise"), default="clean")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--split", choices=("development", "heldout"), default="development")
    args = parser.parse_args()
    torch.set_num_threads(2)
    existed = args.output.exists()
    try:
        {"prepare": prepare, "local": local, "global": global_train, "evaluate": evaluate}[args.command](args)
    except Exception as exc:
        if not existed and args.output.is_dir():
            write_json(args.output / "failure.json", {"status": "failed", "type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
