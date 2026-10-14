"""Independent two-layer Llama distillation from factorized MLP initialization."""
import argparse
import copy
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH, sha256, write_json
from experiments.structure_net.llama_summary_throughput import make_fixture

SCHEMA_VERSION = "nal.llama-neighbourhood-tensorization.v1"
HYPOTHESIS_ID = "llama32-3b-neighbourhood-tensorization-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-13_llama32-neighbourhood-tensorization-plan.md")
AMENDMENT = Path("docs/07 - Status Reports/2026-10-13_llama32-neighbourhood-tensorization-stability-amendment.md")
MANIFEST = Path("data/models/Llama-3.2-3B-cotenancy/manifest.json")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    rank: int = 1024
    steps: int = 96
    seed: int = 31013
    lr: float = 0.00002
    interval: int = 32
    neighbourhood_size: int = 2

    def __post_init__(self):
        if not 0 < self.rank <= 3072 or self.steps < 1 or self.interval < 1:
            raise ValueError("positive feasible rank, steps and interval required")
        if self.neighbourhood_size != 2 or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("two-layer neighbourhood and positive finite LR required")


class FactorLinear(nn.Module):
    """Execute factors directly; no reconstructed weight or dense correction."""
    def __init__(self, left, right):
        super().__init__()
        self.left = nn.Parameter(left.detach().clone())
        self.right = nn.Parameter(right.detach().clone())
        self.in_features, self.out_features = right.shape[1], left.shape[0]

    def forward(self, x):
        return F.linear(F.linear(x, self.right.to(x.dtype)), self.left.to(x.dtype))

    @classmethod
    @torch.no_grad()
    def from_weight(cls, weight, scale, rank, seed):
        out_size, in_size = weight.shape
        if rank * (out_size + in_size) >= weight.numel():
            raise ValueError("factorization must reduce parameter count")
        scale = scale.float().clamp_min(1e-4)
        if scale.shape != (in_size,) or not torch.isfinite(scale).all():
            raise ValueError("invalid calibration scale")
        devices = [weight.device.index] if weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            matrix = weight.float() * scale[None, :]
            if min(matrix.shape) <= 128:
                u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
            else:
                u, s, v = torch.svd_lowrank(matrix, q=min(rank + 16, min(matrix.shape)), niter=2)
                vh = v.T
            root = s[:rank].sqrt()
            left = u[:, :rank] * root
            right = (root[:, None] * vh[:rank]) / scale[None, :]
        return cls(left, right)


def normalized_error(predicted, target):
    return (predicted.float() - target.detach().float()).square().mean() / target.detach().float().square().mean().clamp_min(1e-12)


def run_blocks(blocks, x, rotary):
    positions = torch.arange(x.shape[1], device=x.device)[None, :]
    embeddings = rotary(x, positions)
    for block in blocks:
        x = block(x, position_ids=positions, position_embeddings=embeddings, use_cache=False)[0]
    return x


def factor_state(blocks):
    return {f"{i}.{name}.{part}": getattr(getattr(block.mlp, name), part).detach().half().cpu()
            for i, block in enumerate(blocks) for name in PROJECTIONS for part in ("left", "right")}


def install_factors(blocks, state, device, trainable=False):
    dtype = torch.float32 if trainable else torch.float16
    for i, block in enumerate(blocks):
        for name in PROJECTIONS:
            factors = [state[f"{i}.{name}.{part}"].to(device=device, dtype=dtype) for part in ("left", "right")]
            setattr(block.mlp, name, FactorLinear(*factors).requires_grad_(trainable))


def prepare_fixtures(output):
    from transformers import AutoTokenizer
    if output.exists():
        raise ValueError("preserve fixtures")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    all_examples = []
    specs = ((1, 500, "raw"), (1, 1980, "raw"), (8, 6220, "raw"), (1, 1000, "hierarchy"))
    for split, start, count in (("train", 21000, 64), ("development", 22000, 8), ("heldout", 23000, 20)):
        for i in range(count):
            example = make_fixture(tokenizer, start + i, *specs[i % 4], split)
            labels = ", ".join(f["id"] for f in example["fragments"])
            body = "\n\n".join(f"[{f['id']}]\n{f['source']}" for f in example["fragments"])
            instruction = (f"\n\nSummarize these {len(example['fragments'])} fragments in exactly {len(example['fragments'])} lines labeled {labels}, then stop. "
                           "Include each project's final owner, approved budget and deadline. Ignore superseded proposals and background. "
                           "Maximum 128 tokens per line. Output only summaries.")
            ids = tokenizer.apply_chat_template([
                {"role": "system", "content": "Summarize the supplied evidence accurately and concisely."},
                {"role": "user", "content": body + instruction}], add_generation_prompt=True)
            reference = "\n".join(f"{f['id']}: {f['reference_fact']}" for f in example["fragments"])
            example.update(prompt_ids=ids, prompt_tokens=len(ids), reference=reference,
                           training_ids=ids + tokenizer.encode(reference, add_special_tokens=False) + [128009])
            if len(ids) > 6300 or any(f["source_tokens"] > 2048 for f in example["fragments"]):
                raise ValueError("workload limits exceeded")
            all_examples.append(example)
    hashes = [digest(e["prompt_ids"]) for e in all_examples]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicated requests across splits")
    write_json(output, {"schema": SCHEMA_VERSION, "examples": all_examples, "synthetic_proxy": True})
    print(json.dumps({"fixtures": str(output), "sha256": sha256(output), "examples": len(all_examples)}), flush=True)


def seal(output, config, paths, stage):
    if output.exists():
        raise ValueError("new artifact root required")
    output.mkdir(parents=True)
    paths = [Path(__file__), PLAN, AMENDMENT, MANIFEST,
             Path("experiments/structure_net/prepare_llama_cotenancy.py"),
             Path("experiments/structure_net/llama_summary_throughput.py")] + list(paths)
    hashes = {str(p): sha256(p) for p in paths}
    snapshot = output / "producer_source"
    snapshot.mkdir()
    for i, path in enumerate(paths):
        shutil.copyfile(path, snapshot / f"{i:03d}_{path.name}")
    record = {"schema": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID, "stage": stage,
              "config": asdict(config), "source_hashes": hashes, "command": sys.argv,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "python": sys.version, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}
    import transformers
    record["transformers"] = transformers.__version__
    record["fingerprint"] = digest(record)
    write_json(output / "producer.json", record)
    return record


def load_teacher(device="cuda"):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, torch_dtype=torch.float16,
        attn_implementation="flash_attention_2", low_cpu_mem_usage=True).to(device).eval().requires_grad_(False)


def cache_teacher(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    manifest = json.loads(MANIFEST.read_text())
    for name, item in manifest["files"].items():
        if sha256(MODEL_PATH / name) != item["sha256"]:
            raise ValueError("teacher identity mismatch: " + name)
    seal(args.output, Config(), [args.fixtures], "original_teacher_boundary_cache")
    model = load_teacher()
    scales, counts, states = {}, {}, {}
    hooks = []
    collecting_scales = [False]

    def boundary(index):
        def hook(module, inputs, result):
            states[index] = result[0].detach().cpu()
        return hook

    hooks.append(model.model.layers[0].register_forward_pre_hook(lambda module, inputs: states.update({0: inputs[0].detach().cpu()})))
    for i, block in enumerate(model.model.layers):
        if i % 2:
            hooks.append(block.register_forward_hook(boundary((i + 1) // 2)))
        for name in PROJECTIONS:
            key = f"{i}.{name}"
            def hook(module, inputs, key=key):
                if collecting_scales[0]:
                    x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])[::8].float()
                    value = x.square().sum(0)
                    scales[key] = scales.get(key, 0) + value
                    counts[key] = counts.get(key, 0) + x.shape[0]
            hooks.append(getattr(block.mlp, name).register_forward_pre_hook(hook))
    examples = [e for e in json.loads(args.fixtures.read_text())["examples"] if e["split"] != "heldout"]
    inventory = []
    for n, e in enumerate(examples):
        collecting_scales[0] = e["split"] == "train"
        with torch.no_grad():
            model.model(torch.tensor([e["training_ids"]], device="cuda"), use_cache=False)
        files = []
        for group in range(14):
            path = args.output / f"group_{group:02d}" / f"{e['id']}.pt"
            path.parent.mkdir(exist_ok=True)
            torch.save({"input": states[group], "target": states[group + 1]}, path)
            files.append({"path": str(path), "sha256": sha256(path)})
        inventory.append({"id": e["id"], "split": e["split"], "tokens": len(e["training_ids"]), "files": files})
        states.clear()
        print(json.dumps({"cache": n + 1, "total": len(examples), "tokens": len(e["training_ids"])}), flush=True)
    for h in hooks:
        h.remove()
    scale_path = args.output / "scales.pt"
    torch.save({k: (v / counts[k]).sqrt().cpu() for k, v in scales.items()}, scale_path)
    write_json(args.output / "cache.json", {"schema": SCHEMA_VERSION, "status": "completed", "examples": inventory,
               "scales": str(scale_path), "scales_sha256": sha256(scale_path), "fixtures_sha256": sha256(args.fixtures),
               "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "teacher_gradients": any(p.grad is not None for p in model.parameters())})


def load_example(cache, group, device="cuda"):
    item = cache["files"][group]
    value = torch.load(item["path"], map_location="cpu", weights_only=True)
    return value["input"].to(device), value["target"].to(device)


@torch.no_grad()
def dev_error(blocks, rotary, examples, group):
    numerator, denominator = 0., 0.
    for e in examples:
        x, y = load_example(e, group)
        result = run_blocks(blocks, x, rotary)
        numerator += float((result.float() - y.float()).square().sum())
        denominator += float(y.float().square().sum())
    return numerator / max(denominator, 1e-12)


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    config = Config(steps=8 if args.pilot else 96)
    cache_path = args.cache / "cache.json"
    cache = json.loads(cache_path.read_text())
    if cache["status"] != "completed" or sha256(cache["scales"]) != cache["scales_sha256"]:
        raise ValueError("cache identity mismatch")
    for e in cache["examples"]:
        for f in e["files"]:
            if sha256(f["path"]) != f["sha256"]:
                raise ValueError("boundary cache changed")
    producer = seal(args.output, config, [cache_path, args.cache / "producer.json"],
                    "systems_lifecycle_only_not_quality_evidence" if args.pilot else "training")
    model = load_teacher("cpu")
    rotary = model.model.rotary_emb.to("cuda")
    scales = torch.load(cache["scales"], map_location="cpu", weights_only=True)
    training = [e for e in cache["examples"] if e["split"] == "train"]
    development = [e for e in cache["examples"] if e["split"] == "development"]
    records = []
    for group in (range(1) if args.pilot else range(14)):
        started = time.perf_counter()
        blocks = nn.ModuleList([copy.deepcopy(model.model.layers[2 * group + i]).to("cuda") for i in range(2)])
        blocks.eval().requires_grad_(False)
        before_parameters = sum(p.numel() for p in blocks.parameters())
        control_x, control_y = load_example(development[0], group)
        with torch.no_grad():
            control = run_blocks(blocks, control_x, rotary)
            dense_control_nmse = float(normalized_error(control, control_y))
        if dense_control_nmse > 1e-6:
            raise RuntimeError("uncorrected block extraction does not match cached teacher")
        del control_x, control_y, control
        for i, block in enumerate(blocks):
            for j, name in enumerate(PROJECTIONS):
                layer = 2 * group + i
                original = getattr(block.mlp, name)
                factor = FactorLinear.from_weight(original.weight, scales[f"{layer}.{name}"].cuda(), config.rank,
                                                  config.seed + 1000 * layer + j)
                setattr(block.mlp, name, factor)
                del original
        initial = factor_state(blocks)
        torch.save(initial, args.output / f"group_{group:02d}_initial.pt")
        params = [p for p in blocks.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=0., foreach=False)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.)
        if args.pilot:
            development = [max(development, key=lambda e: e["tokens"])]
        init_error = dev_error(blocks, rotary, development, group)
        best, best_step = init_error, 0
        best_state = initial
        history = [{"step": 0, "development_nmse": init_error}]
        schedule = random.Random(config.seed + 70000 + group)
        order = list(range(len(training)))
        for step in range(1, config.steps + 1):
            if (step - 1) % len(order) == 0:
                schedule.shuffle(order)
            e = max(training, key=lambda e: e["tokens"]) if args.pilot else training[order[(step - 1) % len(order)]]
            x, y = load_example(e, group)
            optimizer.zero_grad(set_to_none=True)
            prediction = run_blocks(blocks, x, rotary)
            loss = normalized_error(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            all_gradients = all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in params)
            if not all_gradients:
                raise FloatingPointError("missing, zero, or nonfinite factor gradient")
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("teacher received gradients")
            norm = float(torch.nn.utils.clip_grad_norm_(params, 1.))
            scaler.step(optimizer)
            scaler.update()
            row = {"step": step, "training_nmse": float(loss.detach()), "gradient_norm": norm, "example": e["id"]}
            del prediction, loss, x, y
            if step % config.interval == 0 or step == config.steps:
                # Select in the actual FP16 deployment representation.
                current = factor_state(blocks)
                evaluation = copy.deepcopy(blocks).half().requires_grad_(False)
                err = dev_error(evaluation, rotary, development, group)
                del evaluation
                row["development_nmse"] = err
                if err < best:
                    best, best_step, best_state = err, step, current
                print(json.dumps({"group": group, **row}), flush=True)
            history.append(row)
        path = args.output / f"group_{group:02d}_selected.pt"
        torch.save(best_state, path)
        # Independent deserialization and operational output check.
        x, y = load_example(development[0], group)
        install_factors(blocks, best_state, "cuda")
        with torch.no_grad():
            expected = run_blocks(blocks, x, rotary)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        install_factors(blocks, loaded, "cuda")
        with torch.no_grad():
            actual = run_blocks(blocks, x, rotary)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        after_parameters = sum(p.numel() for p in blocks.parameters())
        record = {"group": group, "status": "completed", "initial_development_nmse": init_error,
                  "selected_development_nmse": best, "selected_step": best_step,
                  "parameters_before": before_parameters, "parameters_after": after_parameters,
                  "trained_parameters": sum(p.numel() for p in params), "history": history,
                  "selected_checkpoint": str(path), "selected_sha256": sha256(path),
                  "initial_checkpoint": str(args.output / f"group_{group:02d}_initial.pt"),
                  "initial_sha256": sha256(args.output / f"group_{group:02d}_initial.pt"),
                  "checkpoint_reload_exact": True, "teacher_frozen": True, "all_factors_have_gradients": bool(all_gradients),
                  "dense_control_nmse": dense_control_nmse,
                  "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "seconds": time.perf_counter() - started}
        records.append(record)
        write_json(args.output / f"group_{group:02d}.json", record)
        print(json.dumps({k: v for k, v in record.items() if k != "history"}), flush=True)
        del blocks, optimizer, scaler, params, best_state, initial, actual, expected, x, y
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output / "training.json", {"schema": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID,
               "status": "completed", "stage": producer["stage"], "config": asdict(config),
               "producer_fingerprint": producer["fingerprint"], "groups": records})


@torch.inference_mode()
def generate(model, ids, cap, stop_eos=True):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    prompt = torch.tensor([ids], device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    logits = None
    # Matched bounded prefill chunks; avoid allocating logits for the full prompt.
    for offset in range(0, len(ids), 512):
        hidden = model.model(prompt[:, offset:offset+512], past_key_values=cache, use_cache=True).last_hidden_state
    logits = model.lm_head(hidden[:, -1:]).float()
    torch.cuda.synchronize()
    ttft = time.perf_counter() - started
    generated = []
    eos = model.generation_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    for step in range(cap):
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite generation logits")
        token = int(logits[0, -1].argmax())
        generated.append(token)
        if stop_eos and token in eos:
            break
        if step + 1 < cap:
            hidden = model.model(torch.tensor([[token]], device="cuda"), past_key_values=cache, use_cache=True).last_hidden_state
            logits = model.lm_head(hidden[:, -1:]).float()
    torch.cuda.synchronize()
    return {"output_ids": generated, "wall_seconds": time.perf_counter() - started, "ttft_seconds": ttft,
            "eos": generated[-1] in eos, "cap": cap}


def evaluate(args):
    from transformers import AutoTokenizer
    paths = [args.fixtures]
    training = None
    if args.arm != "dense":
        paths += [args.training / "training.json", args.training / "producer.json"]
        training = json.loads((args.training / "training.json").read_text())
        if training["status"] != "completed" or len(training["groups"]) != 14:
            raise ValueError("complete full-model training required")
    seal(args.output, Config(), paths, "development_quality_and_runtime" if args.split == "development" else "heldout")
    model = load_teacher("cpu")
    if training:
        kind = "initial" if args.arm == "initial" else "selected"
        for r in training["groups"]:
            path = Path(r[f"{kind}_checkpoint"])
            if sha256(path) != r[f"{kind}_sha256"]:
                raise ValueError("checkpoint changed")
            g = r["group"]
            install_factors(model.model.layers[2*g:2*g+2], torch.load(path, map_location="cpu", weights_only=True), "cpu")
    model.to("cuda").eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    examples = [e for e in json.loads(args.fixtures.read_text())["examples"] if e["split"] == args.split]
    warm = generate(model, examples[0]["prompt_ids"][:64], 8, False)
    rows = []
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    for e in examples:
        result = generate(model, e["prompt_ids"], e["cap"])
        result.update(id=e["id"], arm=args.arm, text=tokenizer.decode(result["output_ids"], skip_special_tokens=True),
                      prompt_tokens=len(e["prompt_ids"]))
        rows.append(result)
        write_json(args.output / "outputs.json", {"schema": SCHEMA_VERSION, "arm": args.arm,
                   "split": args.split, "rows": rows, "parameter_bytes": parameter_bytes,
                   "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "lookup": False,
                   "measurement_boundary": "warm tokenized fresh request, chunked prefill and greedy decoding; excludes model load and detokenization"})
        print(json.dumps(result), flush=True)
    # Fixed 32 new-token diagnostic avoids rewarding verbosity or early EOS.
    timing = []
    for repeat in range(3):
        for e in examples[:4]:
            result = generate(model, e["prompt_ids"], 32, False)
            result.pop("output_ids")
            timing.append({"id": e["id"], "repeat": repeat, **result})
    write_json(args.output / "runtime.json", {"arm": args.arm, "rows": timing, "quality_evidence": False,
               "aggregate_tokens_per_second": 32 * len(timing) / sum(r["wall_seconds"] for r in timing)})
    write_json(args.output / "result.json", {"status": "completed", "arm": args.arm,
               "outputs_sha256": sha256(args.output / "outputs.json"), "runtime_sha256": sha256(args.output / "runtime.json")})


def tiny_pilot(output):
    from transformers import LlamaConfig, LlamaForCausalLM
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    seal(output, Config(rank=8, steps=1), [], "systems_lifecycle_only_not_quality_evidence")
    torch.manual_seed(91)
    c = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    c._attn_implementation = "flash_attention_2"
    teacher = LlamaForCausalLM(c).half().cuda().eval().requires_grad_(False)
    blocks = copy.deepcopy(teacher.model.layers)
    for i, b in enumerate(blocks):
        for name in PROJECTIONS:
            w = getattr(b.mlp, name).weight
            setattr(b.mlp, name, FactorLinear.from_weight(w, torch.ones(w.shape[1], device="cuda"), 8, 20+i))
    x = torch.randn(1, 31, 64, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        target = run_blocks(teacher.model.layers, x, teacher.model.rotary_emb)
    opt = torch.optim.AdamW([p for p in blocks.parameters() if p.requires_grad], lr=.001)
    loss = normalized_error(run_blocks(blocks, x, teacher.model.rotary_emb), target)
    loss.backward()
    gradients = all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in blocks.parameters() if p.requires_grad)
    assert gradients and not any(p.grad is not None for p in teacher.parameters())
    opt.step()
    path = output / "factors.pt"
    torch.save(factor_state(blocks), path)
    install_factors(blocks, torch.load(path, weights_only=True), "cuda")
    with torch.no_grad():
        actual = run_blocks(blocks, x, teacher.model.rotary_emb)
    assert torch.isfinite(actual).all()
    write_json(output / "result.json", {"status": "completed", "stage": "systems_lifecycle_only_not_quality_evidence",
               "loss": float(loss.detach()), "gradient_routing_pass": bool(gradients), "checkpoint_sha256": sha256(path),
               "peak_cuda_bytes": torch.cuda.max_memory_allocated()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fixtures", "tiny-pilot", "cache", "train", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--arm", choices=("dense", "initial", "distilled"), default="dense")
    parser.add_argument("--split", choices=("development", "heldout"), default="development")
    args = parser.parse_args()
    torch.set_num_threads(2)
    output_existed = args.output.exists()
    try:
        if args.command == "fixtures":
            prepare_fixtures(args.output)
        elif args.command == "tiny-pilot":
            tiny_pilot(args.output)
        elif args.command == "cache":
            cache_teacher(args)
        elif args.command == "train":
            train(args)
        else:
            evaluate(args)
    except Exception as exc:
        if not output_existed and args.output.is_dir() and not (args.output / "failure.json").exists():
            write_json(args.output / "failure.json", {"status": "failed", "type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
