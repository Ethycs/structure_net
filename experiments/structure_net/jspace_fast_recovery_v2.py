"""Version 2: inference-safe deployment/reset lifecycle; v1 artifacts stay immutable."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from . import jspace_compact_performance as perf
from . import jspace_qwen_affine as base
from .jspace_bench_data import sha, write
from .jspace_bench_tensor import accuracy, comparison, model_bytes, run_cases
from .jspace_full_capture import read_bits
from .jspace_full_lease import wait_ready
from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_affine_fast import FastAffineQwenBlock

ROOT = base.ROOT
DEFAULT_ROOT = ROOT / "data/experiments/jspace_fast_recovery/20261113_v2"
LAYERKD = ROOT / "data/experiments/jspace_qwen_layerkd/20261113_v1"
ARMS = ("control", "fast")
CONFIG = {"seed": 861, "epochs": 4, "learning_rate": 3e-5, "accumulation": 4,
          "weight_decay": 0., "gradient_clip": 1., "temperature": 2., "kl_weight": .7,
          "ce_weight": .3, "early_stopping": False, "maximum_relative_dev_loss_increase": .001,
          "selection": "minimum packed development endpoint KL+CE, including epoch zero",
          "screen": "direct validation only when all dev choices agree and endpoint loss increases by at most 0.1%",
          "scope": "single-final-block faster-structure validation; conditional matched four-epoch recovery"}


def recovery_required(result):
    reference, changed = result["baseline_development"]["loss"], result["development"]["loss"]
    if not all(math.isfinite(x) for x in (reference, changed)) or reference <= 0:
        raise ValueError("Invalid development screen")
    return not (result["development_comparison"]["agreement"] == 1. and
                changed <= reference * (1 + CONFIG["maximum_relative_dev_loss_increase"]))


def validate_selection(selection, source_arm, candidate, benchmark_path, recover):
    if (selection["source_arm"] != source_arm or selection["candidate"] != candidate or
            selection["validation_mode"] != ("recovery" if recover else "direct") or
            selection["bench_result_sha256"] != sha(benchmark_path) or selection["heldout_inspected"]):
        raise ValueError("Frozen pre-test selection does not match this validation")


def merged_exclusions(parent, layerkd):
    splits = {k: list(v) for k, v in parent["splits"].items()}
    splits["layerkd_evaluation"] = list(layerkd["splits"]["evaluation"])
    return {"splits": splits}


def validate_fresh(fresh, excluded):
    identities = [c["identity"] for c in fresh]
    seen = {c["identity"] for rows in excluded["splits"].values() for c in rows}
    if len(identities) != 512 or len(set(identities)) != 512 or seen.intersection(identities):
        raise ValueError("Fresh evaluation must contain 512 unique unseen items")
    if any(sum(c["family"] == f for c in fresh) != 128 for f in base.DATASETS):
        raise ValueError("Fresh evaluation must balance the four task families")


def sources():
    return sorted(set(perf.sources() + [Path(__file__), Path(__file__).with_name("jspace_fast_recovery.py"),
        ROOT / "tests/structure_net/test_jspace_fast_recovery.py",
        ROOT / "tests/structure_net/test_jspace_fast_recovery_v2.py"]))


def prepare(root, performance=perf.DEFAULT_ROOT, source_arm="r1536_h64_gb", candidate="combined"):
    base.cpu_only()
    if root.exists():
        raise FileExistsError(root)
    if source_arm not in perf.ARMS or candidate not in perf.CANDIDATES or not perf.CANDIDATES[candidate]["enabled"]:
        raise ValueError("Select a benchmarked fused structure before fresh evaluation")
    previous = perf.verify(performance, inputs=True); perf.checked(performance, "bench")
    selection = json.loads((performance / "selection.json").read_text())
    bench = json.loads((performance / "bench/result.json").read_text())[source_arm + "/" + candidate]
    if not perf.runtime_candidate_valid(bench):
        raise ValueError("Invalid numerical or kernel benchmark")
    validate_selection(selection, source_arm, candidate, performance / "bench/result.json", recovery_required(bench))
    parent = Path(previous["parent_root"])
    training = json.loads((parent / ("training_" + source_arm + ".json")).read_text())
    initializer = parent / ("train_" + source_arm) / f"epoch_{training['selected_epoch']:02d}.pt"
    manifest = base.checked(parent, "train_" + source_arm)
    if sha(initializer) != manifest["artifact_sha256"][str(initializer.relative_to(parent))]:
        raise ValueError("Selected unpacked initializer changed")
    old = json.loads((parent / "tasks.json").read_text())
    layerkd = json.loads((LAYERKD / "tasks.json").read_text())
    excluded = merged_exclusions(old, layerkd)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(previous["model"]["path"], local_files_only=True, trust_remote_code=False)
    fresh, skipped = base.fresh_cases(tokenizer, excluded); validate_fresh(fresh, excluded)
    tasks = {"splits": {"train": old["splits"]["train"], "development": old["splits"]["development"], "evaluation": fresh},
             "fresh_exclusions": skipped, "excluded_identity_count": len({c["identity"] for rows in excluded["splits"].values() for c in rows})}
    if len(tasks["splits"]["train"]) != 512 or len(tasks["splits"]["development"]) != 64:
        raise ValueError("Frozen recovery budget requires 512 train and 64 development cases")
    inputs = dict(previous["input_sha256"])
    extra = [performance / "plan.json", performance / "selection.json", performance / "bench_manifest.json", performance / "bench/result.json",
             performance / "bench" / (source_arm + "_baseline_development.json"),
             performance / "bench" / (source_arm + "_" + candidate + "_development.json"),
             initializer, LAYERKD / "tasks.json"]
    inputs.update({str(p): sha(p) for p in extra})
    plan = {"study_id": "JSPACE-FAST-RECOVERY-01", "config": CONFIG, "parent_root": str(parent),
            "performance_root": str(performance), "performance_fingerprint": previous["fingerprint"],
            "source_arm": source_arm, "candidate": candidate, "fast_options": perf.CANDIDATES[candidate],
            "recover": recovery_required(bench), "screen_values": {"agreement": bench["development_comparison"]["agreement"],
                "reference_loss": bench["baseline_development"]["loss"], "fast_loss": bench["development"]["loss"]},
            "initializer": str(initializer), "initializer_sha256": sha(initializer), "selected_parent_epoch": training["selected_epoch"],
            "model": previous["model"], "cache_root": previous["cache_root"], "label_ids": previous["label_ids"],
            "task_digest": base.digest(tasks), "input_sha256": inputs,
            "created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": {str(p): sha(p) for p in sources()}}
    plan["fingerprint"] = base.digest(plan); root.mkdir(parents=True)
    for p in sources():
        target = root / "producer" / p.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
    write(root / "tasks.json", tasks); write(root / "plan.json", plan)
    return {"root": str(root), "fingerprint": plan["fingerprint"], "recover": plan["recover"], "fresh_count": len(fresh)}


def verify(root, inputs=False, model=False):
    plan = json.loads((root / "plan.json").read_text())
    if plan["config"] != CONFIG or plan["fingerprint"] != base.digest({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed fast-recovery protocol")
    for name, expected in plan["source_sha256"].items():
        if sha(name) != expected or sha(root / "producer" / Path(name).relative_to(ROOT)) != expected:
            raise ValueError("Changed producer: " + name)
    if base.digest(json.loads((root / "tasks.json").read_text())) != plan["task_digest"]:
        raise ValueError("Changed fresh task split")
    if inputs:
        for name, expected in plan["input_sha256"].items():
            if sha(name) != expected:
                raise ValueError("Changed frozen input: " + name)
    if model:
        for name, expected in plan["model"]["files"].items():
            if sha(Path(plan["model"]["path"]) / name) != expected:
                raise ValueError("Changed native weights")
    return plan


def checked(root, name):
    plan = verify(root); value = json.loads((root / (name + "_manifest.json")).read_text())
    if value["status"] != "completed" or value["fingerprint"] != plan["fingerprint"]:
        raise ValueError("Incomplete fast-recovery stage")
    for name, expected in value["artifact_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed stage artifact: " + name)
    return value


def selected_checkpoint(root, arm, recover):
    report = (json.loads((root / ("training_" + arm + ".json")).read_text()) if recover else
              json.loads((root / "deployment.json").read_text())["checkpoints"][arm])
    path = root / (arm + ".pt")
    if sha(path) != report["checkpoint_sha256"]:
        raise ValueError("Changed selected checkpoint")
    return path


class Stage(base.Stage):
    def __init__(self, root, name):
        self.plan = verify(root, inputs=True, model=True); self.lease = wait_ready([0]); self.guard = wait_ready
        self.root, self.name, self.path = root, name, root / name
        self.path.mkdir(exist_ok=False); self.started = time.perf_counter(); self.counts = Counter()
        torch.set_num_threads(4); torch.manual_seed(CONFIG["seed"]); torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(); self.parent = Path(self.plan["parent_root"])
        self.model, self.tokenizer = base.load_small_qwen(self.plan["model"]["path"], device="cuda:0", dtype=torch.bfloat16)
        self.native = copy.deepcopy(self.model.model.layers[-1]).cpu()
        self.native_norm = {k: v.detach().cpu().clone() for k, v in self.model.model.norm.state_dict().items()}
        self.tasks = json.loads((root / "tasks.json").read_text())["splits"]
        self.label_ids = torch.tensor(self.plan["label_ids"], device="cuda:0")
        cache = Path(self.plan["cache_root"]); self.items = json.loads((cache / "items.json").read_text())["items"]
        self.input = np.load(cache / "boundary_35.npy", mmap_mode="r"); self.targets = np.load(cache / "teacher_logits.npy", mmap_mode="r")
        self.indices = {c["identity"]: i for i, c in enumerate(self.items)}
        self.training = [i for i, c in enumerate(self.items) if c["split"] == "train"]
        self.fit = torch.load(self.parent / "fit.pt", map_location="cpu", weights_only=True)
        self.relation = torch.load(self.parent / "relation.pt", map_location="cpu", weights_only=True)
        write(self.path / "lease.json", {k: self.lease[k] for k in ("id", "holder", "allocation", "state", "ready", "end")})

    def install(self, arm, trainable=False, packed=False, selected=False):
        selected_path = selected_checkpoint(self.root, arm, self.plan["recover"]) if selected else None
        self.reset()
        if arm == "native":
            return self.model.model.layers[-1]
        if arm not in ARMS:
            raise ValueError("Unknown recovery arm")
        spec = base.ARMS[self.plan["source_arm"]]
        block = AffineQwenBlock(self.model.model.layers[-1], self.fit["basis"][:, :spec["rank"]].cuda(), self.fit["means"].cuda(),
            trainable=trainable, parameter_dtype=torch.float32,
            head_projection={"rope_pairs": self.fit["rope_pairs"].cuda(), "value_basis": self.fit["value_basis"].cuda()})
        if spec.get("groebner"):
            block.compile_polynomial(self.relation, self.relation["removed"])
        state = torch.load(self.plan["initializer"], map_location="cpu", weights_only=True)
        block.load_state_dict(state["block"], strict=True); self.model.model.norm.float().load_state_dict(state["norm"])
        if arm == "fast":
            block = FastAffineQwenBlock(block, **self.plan["fast_options"])
        self.model.model.layers[-1] = block; self.model.model.norm.requires_grad_(trainable); self.model.eval()
        if packed:
            block.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
            self.model.model.norm.to(dtype=torch.bfloat16).requires_grad_(False)
        if selected:
            state = torch.load(selected_path, map_location="cpu", weights_only=True)
            block.load_state_dict(state["block"], strict=True); self.model.model.norm.load_state_dict(state["norm"])
        return block

    def finish(self, extra=()):
        self.guard([0]); verify(self.root); peak = torch.cuda.max_memory_reserved()
        if peak > sum(self.lease["allocation"].values()) * 1024**2:
            raise RuntimeError("Reservation exceeded")
        files = [p for p in self.path.rglob("*") if p.is_file()] + [self.root / name for name in extra]
        result = {"status": "completed", "stage": self.name, "fingerprint": self.plan["fingerprint"],
            "elapsed_seconds": time.perf_counter() - self.started, "peak_reserved_bytes": peak,
            "lease_id": self.lease["id"], "gpu": torch.cuda.get_device_name(0),
            "artifact_sha256": {str(p.relative_to(self.root)): sha(p) for p in files}}
        write(self.root / (self.name + "_manifest.json"), result)
        return {k: v for k, v in result.items() if k != "artifact_sha256"}


def training_order(indices, epoch):
    result = list(indices); random.Random(CONFIG["seed"] + epoch).shuffle(result); return result


def train(root, arm):
    plan = verify(root)
    if not plan["recover"] or arm not in ARMS:
        raise ValueError("Training requires a frozen recovery decision and known arm")
    stage = Stage(root, "train_" + arm); block = stage.install(arm, trainable=True)
    spec = base.ARMS[plan["source_arm"]]
    parameters = [p for p in stage.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=CONFIG["learning_rate"], weight_decay=0.)
    initial, _ = base.packed_development(stage, "epoch_00"); best, best_epoch, history = initial["loss"], 0, []
    base.save(stage.path / "epoch_00.pt", base.checkpoint(stage, spec["rank"], spec["head_dim"]))
    for epoch in range(1, CONFIG["epochs"] + 1):
        order = training_order(stage.training, epoch); optimizer.zero_grad(set_to_none=True); losses = []
        for step, index in enumerate(order):
            fraction = ((epoch - 1) * len(order) + step) / (CONFIG["epochs"] * len(order))
            for group in optimizer.param_groups:
                group["lr"] = CONFIG["learning_rate"] * .5 * (1 + math.cos(math.pi * fraction))
            case = stage.items[index]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = base.cached_logits(stage, case)
            loss, kd, ce = base.objective(logits, read_bits(stage.targets, index), case)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite recovery loss")
            (loss / CONFIG["accumulation"]).backward(); losses.append(float(loss.detach()))
            if (step + 1) % CONFIG["accumulation"] == 0:
                torch.nn.utils.clip_grad_norm_(parameters, CONFIG["gradient_clip"], error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        dev, _ = base.packed_development(stage, f"epoch_{epoch:02d}")
        history.append({"epoch": epoch, "loss": sum(losses) / len(losses), "development": dev}); stage.event("epoch", arm=arm, **history[-1])
        if dev["loss"] < best:
            best, best_epoch = dev["loss"], epoch
            base.save(stage.path / f"epoch_{epoch:02d}.pt", base.checkpoint(stage, spec["rank"], spec["head_dim"]))
    del optimizer, parameters
    state = torch.load(stage.path / f"epoch_{best_epoch:02d}.pt", map_location="cpu", weights_only=True)
    block.load_state_dict(state["block"], strict=True); stage.model.model.norm.load_state_dict(state["norm"])
    block.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16); stage.model.model.norm.to(dtype=torch.bfloat16).requires_grad_(False)
    selected, rows = base.development(stage, "selected_packed")
    if rows != json.loads((stage.path / f"epoch_{best_epoch:02d}_development.json").read_text()):
        raise ValueError("Selected packed checkpoint changed")
    target = root / (arm + ".pt"); base.save(target, base.checkpoint(stage, spec["rank"], spec["head_dim"]))
    write(root / ("training_" + arm + ".json"), {"arm": arm, "initializer_sha256": plan["initializer_sha256"],
        "completed_epochs": len(history), "updates": len(history) * len(stage.training) // CONFIG["accumulation"],
        "selected_epoch": best_epoch, "initial": initial, "selected_packed": selected, "history": history,
        "checkpoint_sha256": sha(target), "tensor_bytes": model_bytes(stage.model)})
    return stage.finish((target.name, "training_" + arm + ".json"))


def require_ready(root):
    plan = verify(root)
    if plan["recover"]:
        for arm in ARMS:
            checked(root, "train_" + arm)
    return plan


def equal_state(left, right):
    return left.keys() == right.keys() and all(a.dtype == right[k].dtype and torch.equal(a.detach().cpu(), right[k].detach().cpu())
                                               for k, a in left.items())


@torch.inference_mode()
def validate_direct_tensors(stage, arm):
    """Prove the exported structure is the same one timed from the parent checkpoint."""
    live, norm = stage.model.model.layers[-1], copy.deepcopy(stage.model.model.norm)
    try:
        expected = stage.install("control", packed=True)
        state = torch.load(stage.parent / (stage.plan["source_arm"] + ".pt"), map_location="cpu", weights_only=True)
        expected.load_state_dict(state["block"], strict=True); stage.model.model.norm.load_state_dict(state["norm"])
        if arm == "fast":
            expected = perf.candidate_factory(expected, stage.plan["candidate"])
        if not equal_state(live.state_dict(), expected.state_dict()) or not equal_state(norm.state_dict(), stage.model.model.norm.state_dict()):
            raise ValueError("Direct export tensors differ from the timed parent-derived structure")
    finally:
        stage.model.model.layers[-1], stage.model.model.norm = live, norm
    return True


@torch.inference_mode()
def validate_fast(stage):
    fast = stage.model.model.layers[-1]
    reference = FastAffineQwenBlock.from_packed(copy.deepcopy(fast.original), **{**stage.plan["fast_options"], "enabled": False})
    errors = []
    for case in perf.selected_cases(stage.items):
        x = stage.x(case); perf.ensure_fusion(fast, stage.plan["candidate"], x)
        for mode in ("prefill", "decode"):
            error = perf.numerical_error(perf.block_call(stage, reference, x, mode)(), perf.block_call(stage, fast, x, mode)())
            if error["relative_rmse"] > perf.CONFIG["maximum_kernel_relative_rmse"]:
                raise ValueError("Fused recovery kernel diverged from the same-geometry Torch reference")
            errors.append({"identity": case["identity"], "mode": mode, **error})
    write(stage.path / "fast_kernel_validation.json", errors)


@torch.inference_mode()
def evaluate(root):
    plan = require_ready(root); stage = Stage(root, "evaluation"); results, records = {}, {}
    # Validate every deployment on development before exposing any fresh results.
    development, deployments = {}, {}
    for arm in ("native", *ARMS):
        stage.install(arm, packed=arm != "native", selected=plan["recover"] and arm != "native")
        dev, rows = base.development(stage, arm + "_preflight"); development[arm] = dev
        if arm == "fast":
            validate_fast(stage)
        if arm != "native":
            if plan["recover"]:
                expected = root / ("train_" + arm) / "selected_packed_development.json"
            else:
                tag = plan["source_arm"] + ("_baseline" if arm == "control" else "_" + plan["candidate"])
                expected = Path(plan["performance_root"]) / "bench" / (tag + "_development.json")
            if rows != json.loads(expected.read_text()):
                raise ValueError("Packed initialization/reload differs from frozen development records")
            direct_exact = validate_direct_tensors(stage, arm) if not plan["recover"] else None
            if not plan["recover"]:
                spec = base.ARMS[plan["source_arm"]]
                base.save(root / (arm + ".pt"), base.checkpoint(stage, spec["rank"], spec["head_dim"]))
            deployments[arm] = {"checkpoint_sha256": sha(root / (arm + ".pt")), "trained": plan["recover"],
                "timing_tensors_match_parent_candidate": direct_exact, "tensor_bytes": model_bytes(stage.model),
                "parameter_count": sum(p.numel() for p in stage.model.parameters())}
    write(root / "deployment.json", {"checkpoints": deployments, "source_arm": plan["source_arm"], "candidate": plan["candidate"],
                                     "initializer_sha256": plan["initializer_sha256"], "fast_options": plan["fast_options"]})
    for arm in ("native", *ARMS):
        stage.install(arm, packed=arm != "native", selected=arm != "native")
        rows = run_cases(stage, stage.tasks["evaluation"], arm + "_fresh"); records[arm] = rows
        value = {"accuracy": accuracy(rows), "development": development[arm], "tensor_bytes": model_bytes(stage.model),
                 "parameter_count": sum(p.numel() for p in stage.model.parameters()), "trained_in_this_study": plan["recover"] and arm != "native"}
        if arm != "native":
            value["comparison_to_native"] = comparison(records["native"], rows)
        if arm == "fast":
            value["comparison_to_control"] = comparison(records["control"], rows)
        results[arm] = value; stage.event("complete", arm=arm, correct=value["accuracy"]["correct"])
    write(root / "result.json", {"study_id": "JSPACE-FAST-RECOVERY-01", "status": "completed", "confirmed": False, "fingerprint": plan["fingerprint"],
        "recover": plan["recover"], "performance_root": plan["performance_root"], "parent_root": plan["parent_root"],
        "source_arm": plan["source_arm"], "candidate": plan["candidate"], "fresh_evaluation_count": 512, "results": results,
        "timing_provenance": {"root": str(root) if plan["recover"] else plan["performance_root"],
            "stage": "runtime" if plan["recover"] else "full_" + plan["source_arm"] + "_" + plan["candidate"],
            "tensor_identity_verified": not plan["recover"]}})
    return stage.finish(("result.json", "deployment.json", "control.pt", "fast.pt"))


@torch.inference_mode()
def reload(root):
    checked(root, "evaluation"); stage = Stage(root, "reload"); result = {}
    for arm in ARMS:
        stage.install(arm, packed=True, selected=True)
        rows = run_cases(stage, stage.tasks["evaluation"], arm + "_fresh")
        if rows != json.loads((root / "evaluation" / (arm + "_fresh.json")).read_text()):
            raise ValueError("Fresh-process reload changed held-out records")
        result[arm] = {"records_exact": True, "accuracy": accuracy(rows)}
    write(stage.path / "verification.json", result); return stage.finish()


@torch.inference_mode()
def runtime(root):
    checked(root, "reload"); stage = Stage(root, "runtime"); result = {}
    for arm in ("native", *ARMS, "native_repeat"):
        stage.install("native" if arm == "native_repeat" else arm, packed=arm in ARMS, selected=arm in ARMS)
        result[arm] = base.runtime(stage, arm)
    write(stage.path / "result.json", result); return stage.finish()


def campaign(root):
    plan = verify(root); wait_ready([0]); schedule = [("train", arm) for arm in ARMS] if plan["recover"] else []
    schedule += [("evaluate", None), ("reload", None)] + ([("runtime", None)] if plan["recover"] else [])
    for command, arm in schedule:
        name = "train_" + arm if arm else "evaluation" if command == "evaluate" else command
        if (root / (name + "_manifest.json")).exists():
            checked(root, name); continue
        args = [sys.executable, "-m", "experiments.structure_net.jspace_fast_recovery_v2", command, "--root", str(root)]
        subprocess.run(args + (["--arm", arm] if arm else []), cwd=ROOT, check=True)
    return {"status": "completed", "recover": plan["recover"], "root": str(root)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "reload", "runtime", "campaign"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT); parser.add_argument("--performance", type=Path, default=perf.DEFAULT_ROOT)
    parser.add_argument("--source-arm", choices=perf.ARMS, default="r1536_h64_gb"); parser.add_argument("--candidate", choices=perf.CANDIDATES, default="combined")
    parser.add_argument("--arm", choices=ARMS); args = parser.parse_args()
    value = prepare(args.root, args.performance, args.source_arm, args.candidate) if args.command == "prepare" else train(args.root, args.arm) if args.command == "train" else {"evaluate": evaluate, "reload": reload, "runtime": runtime, "campaign": campaign}[args.command](args.root)
    print(json.dumps(value, allow_nan=False), flush=True)
