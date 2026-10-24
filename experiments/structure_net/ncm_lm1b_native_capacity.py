"""Fixed16 TRAIN learnability control for the fully trainable native NCM head."""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_lm1b_native_finetuning_pilot_v2 as lifecycle
from experiments.structure_net import ncm_lm1b_grounded_development as inference
from experiments.structure_net import ncm_additional_training_sources as source_data

native, training, q, ROOT = lifecycle.native, lifecycle.training, lifecycle.q, lifecycle.ROOT
HYPOTHESIS = "ncm-lm1b-native16-TRAIN-capacity-v1"
MODULE = "experiments.structure_net.ncm_lm1b_native_capacity"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-native-capacity-plan.md")
TEST = Path("tests/structure_net/test_ncm_lm1b_native_capacity.py")
AUDITOR = Path("experiments/neural_architecture_lab/audit_ncm_lm1b_native_capacity.py")
WORKFLOW = ROOT / "20261024_lm1b_native16_capacity_workflow_v1"
DATA = ROOT / "20261024_lm1b_native16_capacity_data_v1"
PILOT = ROOT / "20261024_lm1b_native16_capacity_pilot_v1"
TRAIN = ROOT / "20261024_lm1b_native16_capacity_train_v1"
SYSTEMS = ROOT / "20261024_lm1b_native_finetuning_cuda_pilot_results_v2"
SYSTEMS_SHA = "33b7a2aca4b0e4b733c2ae6ac75203d189f7f49525d7449316b75b39c590e020"
INITIAL = "b4b901f589a6fa8b6da89a8f92b7e52f2757a0afff481c898022ed5b448236a5"
BANDS = ((192, 511), (512, 767), (768, 1023), (1024, 1535), (1536, 2048))
QUOTAS = {0: 8, 1: 2, 2: 2, 3: 2, 4: 2}


@dataclass(frozen=True)
class Config:
    updates: int = 256
    microbatch: int = 4
    accumulation: int = 4
    warmup: int = 16
    max_lr: float = 5e-5
    min_lr: float = 5e-6
    sampler_seed: int = 115142
    neighbor_seed: int = 116142
    dropout_seed: int = 117142


def rate(step):
    c = Config()
    if not 1 <= step <= c.updates:
        raise ValueError("step outside fixed256 TRAIN-control budget")
    if step <= c.warmup:
        return c.max_lr * step / c.warmup
    ratio = (step - c.warmup) / (c.updates - c.warmup)
    return c.min_lr + .5 * (c.max_lr - c.min_lr) * (1 + math.cos(math.pi * ratio))


def require_systems():
    path = SYSTEMS / "campaign_results.json"
    receipt = q.read(SYSTEMS / (lifecycle.HYPOTHESIS + "_registry_readback.json"))
    if (lifecycle.file_hash(path) != SYSTEMS_SHA or receipt["record"]["campaign_sha256"] != SYSTEMS_SHA
            or receipt["exact"]["campaign_sha256"] != SYSTEMS_SHA
            or q.read(SYSTEMS / "backend_exact_readback.json")["status"] != "independent_backend_exact_readback_passed"):
        raise ValueError("actual audited registered native CUDA lifecycle required")


def choose(rows, encoding):
    candidates = []
    for row in rows:
        if row["target_origin"] != "original_approved_target_unchanged":
            continue
        count = len(encoding.encode(row["source"], disallowed_special=()))
        band = next((i for i, (low, high) in enumerate(BANDS) if low <= count <= high), None)
        if band is not None:
            candidates.append({**row, "capacity_control_bin": band, "capacity_source_tokens": count})
    selected = []
    for band, count in QUOTAS.items():
        options = sorted((r for r in candidates if r["capacity_control_bin"] == band),
            key=lambda r: (q.digest([HYPOTHESIS, "case", r["id"]]), r["id"]))
        if len(options) < count:
            raise ValueError("all16 predeclared TRAIN-source quotas required")
        selected.extend(options[:count])
    for key in ("id", "source_id", "group_id", "source_sha256"):
        if len({r[key] for r in selected}) != 16:
            raise ValueError("distinct TRAIN control sources required")
    return selected


def producing(root, stage, inputs=(), extra=None):
    require_systems()
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(lifecycle.__file__), lifecycle.PLAN,
        Path(lifecycle.mechanism.__file__), Path(lifecycle.objective.__file__), Path(lifecycle.objective.original.__file__),
        Path(lifecycle.head.__file__), Path(lifecycle.head.concepts.__file__), Path(training.__file__), Path(inference.__file__), Path(source_data.__file__)]
    common = [SYSTEMS / "campaign_results.json", SYSTEMS / (lifecycle.HYPOTHESIS + "_registry_readback.json"),
        SYSTEMS / "backend_exact_readback.json", lifecycle.WARM / "model.pt", lifecycle.WARM / "result.json",
        lifecycle.WARM_WORKFLOW / "tool_terminal_receipt.json", training.CACHE / "outputs.json"]
    common += [native.RELEASE / "files" / name for name in native.FILES]
    common += [native.RELEASE / "files/lm1b/small/config.yaml", *sorted(p for p in (native.INVENTORY / "bert-tokenizer").iterdir() if p.is_file())]
    p = native.vendor.freeze(root, stage, [*inputs, *common, *files], {**asdict(Config()), **(extra or {}),
        "hypothesis_id": HYPOTHESIS, "TRAIN_learnability_only": True, "source_disjoint_task_accuracy": None,
        "no_DEV_or_final_model_selection": True, "all_final_sources_closed": True, "no_semantic_judges_in_this_workflow": True,
        "warm_checkpoint_sha256": lifecycle.WARM_SHA, "pilot_weights_reused": False, "no_new_random_weights": True,
        "native_trainable_parameters": 127732736, "native_encoder_and_VQ_frozen": True, "Llama_body_loaded": False,
        "precision": "FP32weights/BF16training/FP32hardinference", "TF32": False, "peak_limit_bytes": 8_000_000_000,
        "loss": "clean.5C+.7W,noisy.5C+.3W dividedaccum;firstmicroeachupdate self.25C+1W unscaled viahard-forward wordgradient",
        "noise": "max.25 percaseuniform/rootprotected,absolute5120+step", "checkpoint_selection": "fixed final256 only"})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = lifecycle.file_hash(path)
        (root / "producer_source" / f"native_capacity_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    lifecycle.write_json(root / "producer.json", p)
    return p


def prepare():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU source selection only")
    native.vendor.terminal(lifecycle.WARM)
    if lifecycle.file_hash(lifecycle.WARM / "model.pt") != lifecycle.WARM_SHA:
        raise ValueError("fixed original native-training initializer required")
    rows = choose(q.read(training.DATA / "cases.json"), source_data.storage_encoding())
    manifest = q.read(training.DATA / "manifest.json")
    pilot = [0, min(range(16), key=lambda i: (-manifest[rows[i]["id"]]["source_tokens"], rows[i]["id"]))]
    if len(set(pilot)) != 2:
        raise ValueError("distinct first/longest control sources required")
    p = producing(DATA, "fixed16_original_TRAIN_control_sources", [training.DATA / "cases.json", training.DATA / "manifest.json"])
    lifecycle.write_json(DATA / "cases.json", rows)
    lifecycle.write_json(DATA / "manifest.json", {r["id"]: manifest[r["id"]] for r in rows})
    lifecycle.write_json(DATA / "selection.json", {"pilot_indices": pilot})
    native.vendor.verify(p)
    lifecycle.write_json(DATA / "result.json", {"status": "fixed16_TRAIN_sources_prepared", "cases": 16,
        "counts_by_bin": dict(Counter(r["capacity_control_bin"] for r in rows)), "source_disjoint_task_accuracy": None})


def fresh():
    model = lifecycle.head.from_domain(device="cuda")
    saved = torch.load(lifecycle.WARM / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    model.load_attention_state(saved["attention"])
    if native.vendor.module_hash(model) != lifecycle.WARM_MODEL:
        raise ValueError("fresh control must reset original warm weights, never reuse a pilot")
    lifecycle.mechanism.enable(model)
    if lifecycle.trainable_hash(model) != INITIAL or sum(v.numel() for v in model.parameters() if v.requires_grad) != 127732736:
        raise ValueError("complete initial native trainable state differs")
    return model


@torch.no_grad()
def outputs(model, rows, manifest):
    model.eval()
    tokenizer = native.tokenizer()
    values = []
    for row in rows:
        source = {k: v for k, v in row.items() if k not in inference.FORBIDDEN}
        values.append(inference.predict(model, source, manifest, tokenizer))
    return values


def save(path, model, opt, sampler, neighbor, step, producer):
    state = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "optimizer": opt.state_dict(),
        "sampler_rng": sampler.get_state(), "neighbor_rng": neighbor.get_state(), "CPU_rng": torch.get_rng_state(),
        "CUDA_rng": torch.cuda.get_rng_state_all(), "step": step, "config": asdict(Config()), "producer_fingerprint": producer["fingerprint"]}
    torch.save(state, path)


def train(pilot=False):
    training.require_gpu()
    native.vendor.corpus.quality.dependencies()
    native.vendor.terminal(DATA)
    if not pilot:
        native.vendor.terminal(PILOT)
        if not q.read(PILOT / "result.json")["strict_reload_passed"]:
            raise ValueError("actual data-path native training pilot required")
    rows, manifest = q.read(DATA / "cases.json"), q.read(DATA / "manifest.json")
    codes = {r["id"]: r["codes"] for r in q.read(training.CACHE / "outputs.json")}
    root = PILOT if pilot else TRAIN
    p = producing(root, "native16_data_path_pilot" if pilot else "native16_fixed256_TRAIN_training",
        [DATA / n for n in ("cases.json", "manifest.json", "selection.json", "worker_terminal_receipt.json")]
        + [Path(manifest[r["id"]]["path"]) for r in rows], {"pilot": pilot, "actual_updates": 2 if pilot else 256,
            "actual_accumulation": 1 if pilot else 4, "pilot_lr": Config().max_lr if pilot else None})
    features = training.load_features(rows, manifest)
    model = fresh()
    initial_outputs = outputs(model, rows, manifest)
    if not pilot and initial_outputs != q.read(PILOT / "initial_outputs.json"):
        raise ValueError("full control did not reset to the identical warm model")
    lifecycle.write_json(root / "initial_outputs.json", initial_outputs)
    sampler = torch.Generator().manual_seed(Config().sampler_seed)
    neighbor = torch.Generator().manual_seed(Config().neighbor_seed)
    torch.manual_seed(Config().dropout_seed)
    torch.cuda.manual_seed_all(Config().dropout_seed)
    opt = lifecycle.optimizer(model)
    updates, accumulation = (2, 1) if pilot else (256, 4)
    torch.cuda.reset_peak_memory_stats()
    with (root / "trace.jsonl").open("x") as stream:
        for step in range(1, updates + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            learning_rate = Config().max_lr if pilot else rate(step)
            for group in opt.param_groups:
                group["lr"] = learning_rate
            permutation = torch.randperm(16, generator=sampler).tolist()
            micro = []
            for index in range(accumulation):
                indices = (q.read(DATA / "selection.json")["pilot_indices"] * 2) if pilot else permutation[4 * index:4 * index + 4]
                f, m, c, w = training.batch(rows, indices, features, codes)
                if index == 0:
                    loss = lifecycle.objective.backward_microbatch(model, f, m, c, w, neighbor, 5120 + step, accumulation=accumulation, temperature=1.)
                else:
                    loss = lifecycle.objective.original.backward_microbatch(model, f, m, c, w, neighbor, 5120 + step, accumulation=accumulation, self_history=False)
                micro.append({"case_ids": [rows[i]["id"] for i in indices], "source_positions": f.shape[1], "losses": loss})
            if any(v.grad is not None for module in (model.codec.encoder, model.codec.vq) for v in module.parameters()):
                raise ValueError("protected target-code geometry received gradients")
            norm = torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad], 1.)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite native control gradient")
            opt.step()
            stream.write(json.dumps({"step": step, "lr": learning_rate, "permutation": permutation,
                "gradient_norm": float(norm), "microbatches": micro}, allow_nan=False) + "\n")
            stream.flush()
            if pilot or step % 16 == 0:
                print(json.dumps({"step": step, "updates": updates, "self_word_loss": micro[0]["losses"]["self_word_loss"]}), flush=True)
    del f, m, c, w, norm, group
    if lifecycle.protected(model) != {"encoder": lifecycle.ENCODER, "VQ": lifecycle.VQ}:
        raise ValueError("protected encoder/VQ changed")
    final_hash = lifecycle.trainable_hash(model)
    native_hashes = {"generator": native.vendor.module_hash(model.concepts.generator), "word_decoder": native.vendor.module_hash(model.codec.decoder)}
    if (final_hash == INITIAL or native_hashes["generator"] == "60316ea14c56e6276be581112a2a1d81437e3ad8bb4ea3ddf8a98b946805e26a"
            or native_hashes["word_decoder"] == "74d6bb3fc88e2314722c3fcbb9f8e2cb62803bfc076c1f7806d33db0a6988fca"):
        raise ValueError("native generator and word decoder must actually train")
    before_reload = outputs(model, rows, manifest)
    save(root / "model.pt", model, opt, sampler, neighbor, updates, p)
    model.close()
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    saved = torch.load(root / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    model = lifecycle.mechanism.enable(lifecycle.head.from_domain(device="cuda"))
    model.load_state_dict(saved["model"], strict=True)
    opt = lifecycle.optimizer(model)
    opt.load_state_dict(saved["optimizer"])
    sampler.set_state(saved["sampler_rng"])
    neighbor.set_state(saved["neighbor_rng"])
    torch.set_rng_state(saved["CPU_rng"])
    torch.cuda.set_rng_state_all(saved["CUDA_rng"])
    if (outputs(model, rows, manifest) != before_reload or not lifecycle.same(model.state_dict(), saved["model"])
            or not lifecycle.same(opt.state_dict(), saved["optimizer"]) or lifecycle.trainable_hash(model) != final_hash
            or not torch.equal(sampler.get_state(), saved["sampler_rng"]) or not torch.equal(neighbor.get_state(), saved["neighbor_rng"])
            or not torch.equal(torch.get_rng_state(), saved["CPU_rng"]) or not lifecycle.same(torch.cuda.get_rng_state_all(), saved["CUDA_rng"])
            or lifecycle.protected(model) != {"encoder": lifecycle.ENCODER, "VQ": lifecycle.VQ}):
        raise ValueError("strict full-native control optimizer/allRNG/all16source-only reload failed")
    peak = torch.cuda.max_memory_allocated()
    if peak > 8_000_000_000:
        raise ValueError("native TRAIN control exceeds8GBplan")
    lifecycle.write_json(root / "outputs.json", before_reload)
    correct, total = 0, 0
    for row, value in zip(rows, before_reload):
        correct += sum(a == b for a, b in zip(row["native_word_ids"], value["raw_word_ids"]) if a != 0)
        total += sum(v != 0 for v in row["native_word_ids"])
    native.vendor.verify(p)
    result = {"status": "native16_training_completed_semantic_TRAIN_diagnostic_pending", "updates": updates, "cases": 16,
        "trainable_parameters": 127732736, "initial_trainable_sha256": INITIAL, "final_trainable_sha256": final_hash,
        "warm_reset_exact": True, "pilot_weights_reused": False, "strict_reload_passed": True,
        "protected_codec_hashes": lifecycle.protected(model), "peak_allocated_bytes": peak,
        "final_native_hashes": native_hashes,
        "checkpoint_sha256": lifecycle.file_hash(root / "model.pt"), "word_ID_fit_diagnostic_only": correct / total,
        "source_disjoint_task_accuracy": None, "semantic_TRAIN_accuracy": None, "all_final_sources_closed": True,
        "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    lifecycle.write_json(root / "result.json", result)
    model.close()
    print(json.dumps(result), flush=True)


def workflow():
    if any(p.exists() for p in (WORKFLOW, DATA, PILOT, TRAIN)):
        raise ValueError("fresh native TRAIN-control roots required")
    p = producing(WORKFLOW, "supervise_native16_TRAIN_capacity_control")
    receipts = []
    for stage, root, gpu in (("prepare", DATA, ""), ("pilot", PILOT, "0"), ("train", TRAIN, "0")):
        native.vendor.verify(p)
        command = [sys.executable, "-u", "-m", MODULE, stage]
        child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
        lifecycle.write_json(WORKFLOW / "progress.json", {"active_stage": stage, "child_pid": child.pid, "stages": receipts})
        code = child.wait()
        receipt = {"stage": stage, "exit_code": code, "command": command, "child_pid": child.pid, "observer_pid": os.getpid(), "observer": "subprocess.wait"}
        lifecycle.write_json(WORKFLOW / (stage + "_receipt.json"), receipt)
        if root.exists():
            lifecycle.write_json(root / "worker_terminal_receipt.json", receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError("native TRAIN-control stage failed:" + stage)
    native.vendor.verify(p)
    lifecycle.write_json(WORKFLOW / "result.json", {"status": "native16_TRAIN_control_workflow_completed_semantic_review_pending", "stages": receipts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "train", "workflow"))
    args = parser.parse_args()
    {"prepare": prepare, "pilot": lambda: train(True), "train": train, "workflow": workflow}[args.stage]()
