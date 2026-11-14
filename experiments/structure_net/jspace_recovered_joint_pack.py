"""Separately verify FP32 packing of a completed joint-recovery selection.

This is deployment arithmetic parity on the original64 development prompts,
not confirmation, generation qualification, or a throughput measurement.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_recovered_joint as recovered

assembly, recipe, joint = recovered.assembly, recovered.recipe, recovered.joint
SCHEMA = "jspace.recovered-joint-pack.v1"
SCOPE = "FP32 packing parity on original development cases only; no speed or confirmation claim"


def authenticated(root):
    root = Path(root).resolve(); plan = recovered.verify(root)
    result = recovered.completed_training(root, plan)
    fresh = recipe.read(root/"verification/result.json"); receipt = recipe.read(root/"verification/manifest.json")
    if (receipt.get("status") != "completed" or receipt.get("passed") is not True
            or receipt.get("fingerprint") != plan["fingerprint"]
            or receipt.get("result_sha256") != recipe.sha(root/"verification/result.json")
            or fresh.get("status") != "completed" or fresh.get("passed") is not True
            or fresh.get("fingerprint") != plan["fingerprint"]
            or fresh.get("selected_manifest_sha256") != result["selected_manifest_sha256"]
            or fresh.get("selected_epoch") != result["selected_epoch"]
            or not all(fresh.get(k) is True for k in ("strict_state_load", "fixed_state_unchanged", "predictions_exact", "logits_pass"))):
        raise ValueError("Completed matching fresh strict verification required before packing")
    selected = recipe.read(root/"train/selected/manifest.json")
    expected = {f"block_{i:02d}.pt" for i in range(36)} | {"norm_gain.pt"}
    if (selected.get("schema") != joint.SCHEMA or selected.get("blocks") != 36
            or set(selected["artifacts"]) != expected or selected["metadata"].get("epoch") != result["selected_epoch"]
            or selected["metadata"].get("development") != result["selected"]):
        raise ValueError("Selected checkpoint architecture or metadata changed")
    return plan, result, selected


def compare(records, selected_reference, fresh_reference, cases):
    count = sum(r["prediction"] == r["answer"] for r in selected_reference["records"])
    checks = {name:assembly.compare_control(records, reference, cases, count)
        for name, reference in (("selected_epoch", selected_reference), ("fresh_verification", fresh_reference))}
    return dict(passed=all(v["passed"] for v in checks.values()), correct=count, references=checks)


def prepare(root, training_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("Hide CUDA for CPU preparation")
    root, training_root = Path(root).resolve(), Path(training_root).resolve()
    if root.exists(): raise FileExistsError(root)
    parent, result, selected = authenticated(training_root)
    cases = [c for c in recipe.read(training_root/"items.json")["items"] if c["split"] == "development"]
    if len(cases) != 64 or len({c["identity"] for c in cases}) != 64: raise ValueError("Original64 development cases required")
    selected_report = training_root/"train"/f"epoch_{result['selected_epoch']:02d}_development.json"
    fresh_report = training_root/"verification/result.json"
    references = [recipe.read(p) for p in (selected_report, fresh_report)]
    if not compare(references[0]["records"], *references, cases)["passed"]:
        raise ValueError("Selected and fresh references disagree")
    inputs = dict(parent["input_sha256"])
    for name in ("plan.json", "items.json", "train/result.json", "train/manifest.json", "train/selected/manifest.json",
                 "verification/result.json", "verification/manifest.json"):
        assembly._pin(inputs, training_root/name)
    assembly._pin(inputs, selected_report)
    for name, digest in selected["artifacts"].items(): assembly._pin(inputs, training_root/"train/selected"/name, digest)
    sources = dict(parent["source_sha256"])
    for path in (Path(__file__).resolve(), recipe.ROOT/"tests/structure_net/test_jspace_recovered_joint_pack.py"):
        assembly._merge_hashes(sources, {str(path):recipe.sha(path)})
    plan = dict(schema=SCHEMA, training_root=str(training_root), training_fingerprint=parent["fingerprint"],
        model=parent["model"], blocks=parent["blocks"], selected_epoch=result["selected_epoch"],
        selected_directory=str(training_root/"train/selected"), checkpoint_artifacts=selected["artifacts"],
        selected_manifest_sha256=result["selected_manifest_sha256"], selected_reference=str(selected_report),
        fresh_reference=str(fresh_report), cases=cases, label_ids=parent["label_ids"],
        selected_correct=result["selected"]["correct"], training_quality_pass=result["quality_pass"],
        input_sha256=inputs, input_identity={p:recipe.file_identity(p) for p in inputs}, source_sha256=sources,
        protocol=str(recovered.PROTOCOL), precision="All selected tensors and geometry packedFP32; original tiedE plus trained RMS gain",
        gpu_prefix="GPU-0370fdb0", allocation_mib=7168, microbatch=4, confirmed=False, scope=SCOPE)
    plan["fingerprint"] = recipe.digest(plan); root.mkdir(parents=True)
    for path, digest in sources.items():
        if recipe.sha(path) != digest: raise ValueError("Changed packing producer")
        destination = root/"producer"/Path(path).relative_to(recipe.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
    recipe.write(root/"plan.json", plan)
    return dict(status="prepared", fingerprint=plan["fingerprint"])


def verify(root):
    root = Path(root).resolve(); plan = recipe.read(root/"plan.json")
    if plan.get("schema") != SCHEMA or plan["fingerprint"] != recipe.digest({k:v for k,v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed packing plan")
    parent, result, _ = authenticated(plan["training_root"])
    if parent["fingerprint"] != plan["training_fingerprint"] or result["selected_manifest_sha256"] != plan["selected_manifest_sha256"]:
        raise ValueError("Changed selected training source")
    for path, digest in plan["source_sha256"].items():
        if recipe.sha(path) != digest or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT)) != digest:
            raise ValueError("Changed packing source")
    for path, identity in plan["input_identity"].items():
        if recipe.file_identity(path) != identity: raise ValueError("Changed packing input")
    return plan


def load_packed_block(plan, layer, *, device="cpu", loader=None):
    from .jspace_qwen_parallel_model import load_native_block
    if type(layer) is not int or not 0 <= layer < 36: raise ValueError("Original absolute layer0..35 required")
    name = f"block_{layer:02d}.pt"; path = Path(plan["selected_directory"])/name
    if recipe.sha(path) != plan["checkpoint_artifacts"][name]: raise ValueError("Changed selected block checkpoint")
    loader = load_native_block if loader is None else loader
    native = loader(plan["model"]["path"], layer, device=device, dtype=torch.bfloat16)
    block = recovered.semantic.build_enriched_block(native, recipe.load(plan["blocks"][str(layer)]["fit_path"]),
        parameter_dtype=torch.float32)
    del native
    state = recipe.load(path); original = plan["blocks"][str(layer)]
    if recipe.sha(original["checkpoint"]) != original["checkpoint_sha256"]: raise ValueError("Changed recovered initializer")
    initial_state = recipe.load(original["checkpoint"])["block"]
    # Compare immutable buffers against their exact authenticated initializer,
    # avoiding device-dependent recomputation rounding in projection shifts.
    # Numerator offsets are trainable parameters, not geometric buffers.
    for key, value in block.named_buffers():
        if (key not in state or key not in initial_state or value.dtype != state[key].dtype
                or not torch.equal(initial_state[key], state[key])):
            raise ValueError("Selected checkpoint changed fixed geometry: "+key)
        if key.endswith(".basis") and not torch.equal(value.cpu(), state[key]): raise ValueError("Selected basis differs from fit")
    if any(key not in state or state[key].dtype != parameter.dtype for key, parameter in block.named_parameters()):
        raise ValueError("Selected parameter precision changed")
    block.load_state_dict(state, strict=True)
    basis = block.basis.detach().clone()
    block.pack(dtype=torch.float32, buffer_dtype=torch.float32)
    if not torch.equal(block.basis, basis): raise ValueError("Packing changed fixed basis")
    return block.eval().requires_grad_(False)


def load_norm(plan, norm):
    path = Path(plan["selected_directory"])/"norm_gain.pt"
    if recipe.sha(path) != plan["checkpoint_artifacts"][path.name]: raise ValueError("Changed selected final norm")
    gain = recipe.load(path)
    if gain.dtype != torch.float32 or gain.shape != norm.weight.shape or not torch.isfinite(gain).all():
        raise ValueError("Finite native-width FP32 trained norm required")
    with torch.no_grad(): norm.weight.copy_(gain)
    return norm.eval().requires_grad_(False)


def cap_allocator(lease, *, cuda=None, visible=None):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "") if visible is None else visible
    allocation = lease["allocation"]
    if len(allocation) != 1 or visible not in allocation or not visible.startswith("GPU-0370fdb0") or allocation[visible] != 7168:
        raise ValueError("Packing requires exactly the leased3060 at7168MiB")
    return assembly.sequential.cap_allocator(lease, cuda=cuda, visible=visible)


def run(root):
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_readout_norm
    root = Path(root).resolve(); plan = verify(root); directory = root/"verification"
    directory.mkdir(exist_ok=False); lease = allocator = None; started = time.monotonic()
    telemetry = dict(calls=0, wall_seconds=0.); cuda_started = False
    try:
        lease = wait_ready([0]); allocator = cap_allocator(lease)
        torch.cuda.init(); cuda_started = True; torch.cuda.reset_peak_memory_stats()
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        def guard():
            began = time.monotonic()
            try:
                current = wait_ready([0])
                if current["id"] != lease["id"] or current["allocation"] != lease["allocation"]: raise RuntimeError("Packing lease changed")
                if torch.cuda.max_memory_reserved() > allocator["allocator_limit_bytes"]: raise MemoryError("Packing allocator cap exceeded")
            finally:
                telemetry["calls"] += 1; telemetry["wall_seconds"] += time.monotonic()-began
        embedding, norm, rotary = load_readout_norm(plan["model"]["path"], device="cpu", dtype=torch.bfloat16)
        embedding, norm, rotary = embedding.float(), load_norm(plan, norm.float()), rotary.float().to("cuda:0")
        endpoints = assembly.tensor_storage(embedding, norm, rotary); inventory = []
        def loader(layer):
            guard(); block = load_packed_block(plan, layer, device="cuda:0")
            inventory.append(dict(layer=layer, fixed_geometry_verified=True, **assembly.tensor_storage(block)))
            return block
        value = assembly.single.stream_score(plan["cases"], input_embedding=embedding, block_loader=loader,
            final_readout=(norm, embedding), label_ids=plan["label_ids"], scratch_dir=directory/"scratch",
            rotary=rotary, layer_count=36, microbatch=plan["microbatch"], device="cuda:0", guard=guard,
            progress=lambda **kw:recipe.event(root, "packing_progress", detail=kw))
        parity = compare(value["records"], recipe.read(plan["selected_reference"]), recipe.read(plan["fresh_reference"]), plan["cases"])
        guard()
        value.update(status="completed", passed=parity["passed"], fingerprint=plan["fingerprint"], parity=parity,
            selected_epoch=plan["selected_epoch"], selected_manifest_sha256=plan["selected_manifest_sha256"],
            accuracy=assembly.single.ranking_summary(value["records"]), training_quality_pass=plan["training_quality_pass"],
            fixed_geometry_verified=True, strict_state_load=True, precision="float32", confirmed=False, scope=SCOPE,
            tensor_state=dict(storage_bytes=endpoints["storage_bytes"]+sum(b["storage_bytes"] for b in inventory),
                endpoints=endpoints, blocks=inventory, accounting="TiedE once, trained norm/rotary once, packed blocks including fixed buffers; no activations or KV cache"),
            throughput_measured=False, confirmation_used=False)
        recipe.write(directory/"result.json", value)
        if not parity["passed"]: raise RuntimeError("Packed model differs from the selected or fresh-verified logits")
        recipe.write(directory/"manifest.json", dict(status="completed", passed=True, fingerprint=plan["fingerprint"],
            result_sha256=recipe.sha(directory/"result.json"), lease_id=lease["id"], allocation=lease["allocation"],
            allocator=allocator, peak_reserved_bytes=torch.cuda.max_memory_reserved(), reservation_guard=telemetry,
            elapsed_seconds=time.monotonic()-started))
        return dict(status="completed", passed=True, accuracy=value["accuracy"], fingerprint=plan["fingerprint"])
    except BaseException as error:
        recipe.write(directory/"failure.json", dict(status="failed", fingerprint=plan["fingerprint"], error=repr(error),
            lease_id=None if lease is None else lease["id"], allocation=None if lease is None else lease["allocation"], allocator=allocator,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None, reservation_guard=telemetry))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run")); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path); args = parser.parse_args()
    if args.stage == "prepare" and args.training_root is None: parser.error("prepare requires --training-root")
    print(json.dumps(prepare(args.root, args.training_root) if args.stage == "prepare" else run(args.root), allow_nan=False), flush=True)
