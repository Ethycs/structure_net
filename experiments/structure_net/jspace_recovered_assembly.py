"""Actual end-to-end assembly of independently recovered Qwen blocks.

All arms start from original token embeddings and execute every decoder block.
One block is loaded at a time; cached student or native internal boundaries are
never used to shortcut assembly. Only the original64 development tasks are read.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_independent_layer_distill as independent
from . import jspace_next_layer_distill as sequential
from .jspace_single_block_verify import validate_records

single, recipe, joint = sequential.single, sequential.recipe, sequential.joint
SCHEMA = "jspace.recovered-assembly.v1"
ARMS = ("native", "accepted_two", "all_compact", "hybrid")
PROTOCOL = recipe.ROOT/"docs/07 - Status Reports/jspace/2026-11-14_qwen-recovered-assembly.md"
SCOPE = "Development-only full36-block assembly; no confirmation, speed, or deployment-readiness claim"


def compact_layers(arm):
    if arm == "native": return []
    if arm == "accepted_two": return [0, 1]
    if arm == "all_compact": return list(range(36))
    if arm == "hybrid": return [i for i in range(36) if i not in (12, 15)]
    raise ValueError("Unknown assembly arm")


def _merge_hashes(destination, incoming):
    for path, expected in incoming.items():
        if path in destination and destination[path] != expected:
            raise ValueError("Conflicting source fingerprints for "+path)
        destination[path] = expected


def _pin(inputs, path, expected=None):
    path = Path(path).resolve(); observed = recipe.sha(path)
    if expected is not None and observed != expected:
        raise ValueError("Changed consumed assembly input: "+str(path))
    _merge_hashes(inputs, {str(path):observed})
    return observed


def verified_independent(root):
    """A scientific isolated miss is allowed; a reload/parity failure is not."""
    root = Path(root).resolve(); plan = independent.verify(root)
    independent.verify_pilot(root, plan)
    result = sequential.completed_training(root, plan)
    manifest = recipe.read(root/"verification/manifest.json")
    verification = recipe.read(root/"verification/result.json")
    selected_sha = result["selected_checkpoint_sha256"]
    if (manifest.get("status") != "completed" or manifest.get("passed") is not True
            or manifest.get("fingerprint") != plan["fingerprint"]
            or manifest.get("result_sha256") != recipe.sha(root/"verification/result.json")
            or manifest.get("checkpoint_sha256") != selected_sha
            or verification.get("status") != "completed" or verification.get("passed") is not True
            or verification.get("strict_state_load") is not True
            or verification.get("fingerprint") != plan["fingerprint"]
            or verification.get("campaign_fingerprint") != plan["fingerprint"]
            or verification.get("checkpoint_sha256") != selected_sha
            or verification.get("selected_epoch") != result["selected_epoch"]
            or verification.get("training_quality_pass") != result["quality_pass"]):
        raise ValueError("Matching completed fresh verification required for every independent block")
    cases = [c for c in recipe.read(root/"items.json")["items"] if c["split"] == "development"]
    validate_records(verification["records"], cases)
    selected = recipe.read(root/"train"/f"epoch_{result['selected_epoch']:02d}_development.json")
    if any(a["prediction"] != b["prediction"] for a,b in zip(selected["records"], verification["records"])):
        raise ValueError("Fresh verification changed selected predictions")
    return plan, result


def bind_selection(root, plan, result, verification_root, inputs, sources):
    root, verification_root = Path(root).resolve(), Path(verification_root).resolve()
    epoch = result["selected_epoch"]
    history = next(r for r in result["history"] if r["epoch"] == epoch)
    checkpoint = Path(result["selected_checkpoint"]).resolve()
    checkpoint_sha = result.get("selected_checkpoint_sha256", history["checkpoint_sha256"])
    _pin(inputs, checkpoint, checkpoint_sha)
    _pin(inputs, plan["fit_path"], plan["input_sha256"][plan["fit_path"]])
    for name in ("plan.json", "items.json", "train/result.json", "train/manifest.json"):
        _pin(inputs, root/name)
    for row in result["history"]:
        _pin(inputs, root/"train"/f"epoch_{row['epoch']:02d}_development.json", row["development_sha256"])
    for name in ("plan.json", "verification/result.json", "verification/manifest.json"):
        _pin(inputs, verification_root/name)
    if (root/"runtime_policy.json").exists():
        for name in ("runtime_policy.json", "runtime_train.json"):
            _pin(inputs, root/name)
    _merge_hashes(sources, plan["source_sha256"])
    _merge_hashes(sources, recipe.read(verification_root/"plan.json")["source_sha256"])
    return dict(root=str(root), fingerprint=plan["fingerprint"], fit_path=plan["fit_path"],
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha, selected_epoch=epoch,
        selected_correct=result["selected"]["correct"], isolated_quality_pass=result["quality_pass"],
        verification_root=str(verification_root), rank=plan["rank"], head_dim=plan["head_dim"],
        intermediate_size=plan["intermediate_size"])


def prepare(root, source, block0_root, block0_verification_root, block1_root, parallel_root, native_cache_root):
    from . import jspace_native_dev_cache as cache
    from . import jspace_parallel_single_layer as parallel
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA for CPU assembly preparation")
    root, source, block0_root, block0_verification_root, block1_root, parallel_root, native_cache_root = (
        Path(p).resolve() for p in (root, source, block0_root, block0_verification_root, block1_root, parallel_root, native_cache_root))
    if root.exists(): raise FileExistsError(root)
    original = recipe.verify(source); items = recipe.read(source/"items.json")
    cases = [c for c in items["items"] if c["split"] == "development"]
    if len(cases) != 64 or len({c["identity"] for c in cases}) != 64:
        raise ValueError("Original64 development cases required")
    inputs, sources, blocks = {}, dict(original["source_sha256"]), {}
    for layer, campaign, verifier in ((0, block0_root, block0_verification_root), (1, block1_root, block1_root)):
        previous = sequential.accepted_predecessor(campaign, verifier)
        plan, result = previous["plan"], recipe.read(campaign/"train/result.json")
        if (previous["layer"] != layer or plan["source_fingerprint"] != original["fingerprint"]
                or plan["items_digest"] != recipe.digest(items) or result["selected"]["correct"] != (48 if layer == 0 else 49)):
            raise ValueError("Accepted block0/block1 sources or qualification differ")
        blocks[str(layer)] = bind_selection(campaign, plan, result, verifier, inputs, sources)
    queue_plan = parallel.verify_state(parallel_root); queue_result = recipe.read(parallel_root/"result.json")
    if (queue_result.get("status") != "completed" or queue_result.get("fingerprint") != queue_plan["fingerprint"]
            or queue_result.get("verified_layers") != list(range(2, 36))
            or queue_result.get("failed_layers") != [] or queue_result.get("unattempted_layers") != []
            or queue_result.get("isolated_quality_misses") != [12, 15]
            or Path(queue_plan["source"]).resolve() != source
            or Path(queue_plan["quality_prefix_root"]).resolve() != native_cache_root):
        raise ValueError("Complete34-block independent queue with isolated misses12/15 required")
    _merge_hashes(sources, queue_plan["source_sha256"])
    for name in ("plan.json", "result.json", "pilots_gate.json"):
        _pin(inputs, parallel_root/name)
    for layer in range(2, 36):
        campaign = parallel_root/"blocks"/f"block_{layer:02d}"
        plan, result = verified_independent(campaign)
        if (plan["layer"] != layer or plan["source_fingerprint"] != original["fingerprint"]
                or plan["items_digest"] != recipe.digest(items)
                or Path(plan["quality_prefix_root"]).resolve() != native_cache_root
                or result["quality_pass"] != (layer not in (12, 15))):
            raise ValueError("Independent selection identity or qualification changed")
        blocks[str(layer)] = bind_selection(campaign, plan, result, campaign, inputs, sources)
        for name in ("pilot/result.json", "pilot/manifest.json"):
            _pin(inputs, campaign/name)
    native_plan = cache.verify(native_cache_root, completed=False)
    native_manifest = recipe.read(native_cache_root/"cache/manifest.json")
    native_reference = recipe.read(native_cache_root/"cache/result.json")
    if (native_plan["source_fingerprint"] != original["fingerprint"]
            or native_manifest.get("status") != "completed" or native_manifest.get("fingerprint") != native_plan["fingerprint"]
            or native_reference.get("status") != "completed" or native_reference.get("fingerprint") != native_plan["fingerprint"]
            or native_reference.get("correct") != 50 or native_reference.get("count") != 64
            or native_reference.get("native_baseline_predictions_exact") is not True):
        raise ValueError("Matching pure-native50/64 reference required")
    validate_records(native_reference["records"], cases)
    _pin(inputs, native_cache_root/"cache/result.json", native_manifest["artifact_sha256"]["cache/result.json"])
    for path in (native_cache_root/"plan.json", native_cache_root/"cache/manifest.json", source/"plan.json", source/"items.json"):
        _pin(inputs, path)
    _merge_hashes(sources, native_plan["source_sha256"])
    for path in (Path(__file__).resolve(), recipe.ROOT/"tests/structure_net/test_jspace_recovered_assembly.py", PROTOCOL):
        _merge_hashes(sources, {str(path):recipe.sha(path)})
    plan = dict(schema=SCHEMA, source=str(source), source_fingerprint=original["fingerprint"], model=original["model"],
        blocks=blocks, arms={arm:compact_layers(arm) for arm in ARMS}, primary_arm="hybrid",
        native_reference=str(native_cache_root/"cache/result.json"),
        accepted_two_reference=str(block1_root/"verification/result.json"), expected_native_correct=50,
        expected_accepted_two_correct=49, minimum_correct=47, count=64, maximum_accuracy_drop=.05,
        cases=cases, protocol=str(PROTOCOL), scope=SCOPE, confirmed=False,
        precision="Native BF16-rounded weights promoted toFP32; recovered parameters/geometry packedFP32; full native-width residual",
        input_sha256=inputs, input_identity={p:recipe.file_identity(p) for p in inputs}, source_sha256=sources)
    plan["fingerprint"] = recipe.digest(plan); root.mkdir(parents=True)
    for path, expected in sources.items():
        if recipe.sha(path) != expected: raise ValueError("Changed assembly producer")
        destination = root/"producer"/Path(path).relative_to(recipe.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
    recipe.write(root/"plan.json", plan)
    return dict(status="prepared", fingerprint=plan["fingerprint"], arms=list(ARMS), primary="hybrid")


def verify(root):
    root = Path(root); plan = recipe.read(root/"plan.json")
    if plan.get("schema") != SCHEMA or plan["fingerprint"] != recipe.digest({k:v for k,v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed assembly plan")
    if recipe.verify(plan["source"])["fingerprint"] != plan["source_fingerprint"]:
        raise ValueError("Changed original model source")
    for path, expected in plan["source_sha256"].items():
        if recipe.sha(path) != expected or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT)) != expected:
            raise ValueError("Changed assembly producer")
    for path, identity in plan["input_identity"].items():
        if recipe.file_identity(path) != identity: raise ValueError("Changed frozen assembly input")
    return plan


def tensor_storage(*values):
    """Count actual parameter/buffer storage, including fixed/nonpersistent geometry."""
    tensors = []
    for value in values:
        tensors.extend([value] if isinstance(value, torch.Tensor) else [*value.parameters(), *value.buffers()])
    storages, objects = {}, set()
    logical = 0
    for tensor in tensors:
        if tensor.device.type == "meta": raise ValueError("Cannot count unmaterialized model storage")
        if id(tensor) not in objects:
            objects.add(id(tensor)); logical += tensor.numel()*tensor.element_size()
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        storages[key] = storage.nbytes()
    return dict(storage_bytes=sum(storages.values()), logical_tensor_bytes=logical, unique_storages=len(storages),
        unique_tensor_objects=len(objects))


def load_block(plan, arm, layer, *, device="cuda:0", guard=lambda:None, loader=None):
    from .jspace_qwen_parallel_model import load_native_block
    if type(layer) is not int or not 0 <= layer < 36: raise ValueError("Absolute decoder index0..35 required")
    guard(); loader = load_native_block if loader is None else loader
    native = loader(plan["model"]["path"], layer, device=device, dtype=torch.bfloat16)
    if layer not in plan["arms"][arm]: return native.float().eval().requires_grad_(False)
    selected = plan["blocks"][str(layer)]
    if recipe.sha(selected["checkpoint"]) != selected["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed immediately before assembly load")
    block = single.build_enriched_block(native, recipe.load(selected["fit_path"]), parameter_dtype=torch.float32)
    del native
    checkpoint = recipe.load(selected["checkpoint"])
    if checkpoint["epoch"] != selected["selected_epoch"] or (layer > 0 and checkpoint.get("layer") != layer):
        raise ValueError("Wrong selected epoch/layer in assembly checkpoint")
    block.load_state_dict(checkpoint["block"], strict=True)
    del checkpoint
    if (block.rank, block.head_dim, block.intermediate_size) != (selected["rank"], selected["head_dim"], selected["intermediate_size"]):
        raise ValueError("Selected block geometry changed")
    return block.pack(dtype=torch.float32, buffer_dtype=torch.float32)


def completed_arm(root, plan, arm):
    path = Path(root)/arm
    manifest, result = recipe.read(path/"manifest.json"), recipe.read(path/"result.json")
    if (manifest.get("status") != "completed" or manifest.get("fingerprint") != plan["fingerprint"]
            or manifest.get("arm") != arm or manifest.get("result_sha256") != recipe.sha(path/"result.json")
            or result.get("status") != "completed" or result.get("fingerprint") != plan["fingerprint"]
            or result.get("arm") != arm):
        raise ValueError("Matching completed assembly arm required")
    validate_records(result["records"], plan["cases"])
    return result


def compare_control(records, reference, cases, expected_correct):
    from .jspace_single_block_verify import Tolerances
    validate_records(records, cases); validate_records(reference["records"], cases)
    changed = [a["identity"] for a,b in zip(records, reference["records"]) if a["prediction"] != b["prediction"]]
    correct = sum(r["prediction"] == r["answer"] for r in records)
    tolerance = Tolerances()
    logits_pass = all(abs(observed-expected) <= tolerance.logit_atol+tolerance.logit_rtol*abs(expected)
        for row, ref in zip(records, reference["records"]) for observed, expected in zip(row["logits"], ref["logits"]))
    return dict(passed=not changed and correct == expected_correct and logits_pass, predictions_exact=not changed,
        logits_pass=logits_pass, logit_atol=tolerance.logit_atol, logit_rtol=tolerance.logit_rtol,
        correct=correct, expected_correct=expected_correct, changed_prediction_identities=changed,
        logit_max_abs_delta=max(abs(x-y) for a,b in zip(records, reference["records"]) for x,y in zip(a["logits"], b["logits"])))


def run(root, arm):
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_readout_norm
    root = Path(root).resolve(); plan = verify(root)
    if arm not in plan["arms"]: raise ValueError("Unknown frozen assembly arm")
    directory = root/arm
    if directory.exists(): raise FileExistsError(directory)
    if arm in ("all_compact", "hybrid"):
        for control in ("native", "accepted_two"):
            if completed_arm(root, plan, control)["control"]["passed"] is not True:
                raise ValueError("Native and accepted-two controls must pass before candidate assembly")
    directory.mkdir(exist_ok=False); lease = None; cuda_started = False; started = time.monotonic()
    guard_telemetry = dict(calls=0, wall_seconds=0.)
    try:
        lease = wait_ready([0]); allocation = dict(lease["allocation"])
        allocator = sequential.cap_allocator(lease)
        torch.cuda.init(); cuda_started = True; torch.cuda.reset_peak_memory_stats()
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        def guard():
            began = time.monotonic()
            try:
                current = wait_ready([0])
                if current["id"] != lease["id"] or current["allocation"] != allocation:
                    raise RuntimeError("Assembly reservation changed")
                joint.check_reserved_peak(allocation, torch.cuda.max_memory_reserved())
            finally:
                guard_telemetry["calls"] += 1; guard_telemetry["wall_seconds"] += time.monotonic()-began
        embedding, norm, rotary = load_readout_norm(plan["model"]["path"], device="cpu", dtype=torch.bfloat16)
        embedding, norm, rotary = embedding.float(), norm.float(), rotary.float().to("cuda:0")
        endpoints = tensor_storage(embedding, norm, rotary); block_bytes = []
        def loader(layer):
            block = load_block(plan, arm, layer, device="cuda:0", guard=guard)
            block_bytes.append(dict(layer=layer, compact=layer in plan["arms"][arm], **tensor_storage(block)))
            return block
        value = single.stream_score(plan["cases"], input_embedding=embedding, block_loader=loader,
            final_readout=(norm, embedding), label_ids=recipe.read(Path(plan["source"])/"plan.json")["label_ids"],
            scratch_dir=directory/"scratch", rotary=rotary, layer_count=36, microbatch=4, device="cuda:0", guard=guard,
            progress=lambda **kw:recipe.event(root, "progress", arm=arm, detail=kw))
        control = None
        if arm in ("native", "accepted_two"):
            key = "native" if arm == "native" else "accepted_two"
            control = compare_control(value["records"], recipe.read(plan[key+"_reference"]), plan["cases"], plan["expected_"+key+"_correct"])
        guard()
        value.update(status="completed", arm=arm, fingerprint=plan["fingerprint"], compact_layers=plan["arms"][arm],
            accuracy=single.ranking_summary(value["records"]), control=control, confirmed=False, scope=SCOPE,
            resident_tensor_state=dict(storage_bytes=endpoints["storage_bytes"]+sum(b["storage_bytes"] for b in block_bytes),
                logical_tensor_bytes=endpoints["logical_tensor_bytes"]+sum(b["logical_tensor_bytes"] for b in block_bytes),
                endpoints=endpoints, blocks=block_bytes, dtype="float32",
                accounting="Tied embedding/readout once, final norm/rotary once, each packed block's parameters AND fixed buffers; storage aliases deduplicated within each owner. Excludes temporary activations, KV cache, scorer label-row copies and allocator reserve."),
            lease_id=lease["id"], allocation=allocation, allocator=allocator,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), reservation_guard=guard_telemetry,
            elapsed_seconds=time.monotonic()-started)
        recipe.write(directory/"result.json", value)
        if control is not None and not control["passed"]:
            raise RuntimeError("Assembly control predictions did not reproduce accepted reference")
        recipe.write(directory/"manifest.json", dict(status="completed", arm=arm, fingerprint=plan["fingerprint"],
            result_sha256=recipe.sha(directory/"result.json"), lease_id=lease["id"], allocation=allocation,
            peak_reserved_bytes=value["peak_reserved_bytes"]))
        return dict(status="completed", arm=arm, accuracy=value["accuracy"], control=control)
    except BaseException as error:
        recipe.write(directory/"failure.json", dict(status="failed", fingerprint=plan["fingerprint"], error=repr(error),
            lease_id=None if lease is None else lease["id"], allocation=None if lease is None else lease["allocation"],
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None, reservation_guard=guard_telemetry))
        raise


def summarize(root):
    from .jspace_qwen_parallel_evaluate import paired_quality
    root = Path(root); plan = verify(root)
    values = {arm:completed_arm(root, plan, arm) for arm in ARMS}
    if not all(values[arm]["control"]["passed"] for arm in ("native", "accepted_two")):
        raise ValueError("Both assembly controls must pass")
    native = values["native"]; size = native["resident_tensor_state"]["storage_bytes"]
    comparison = {arm:dict(paired=paired_quality(native["records"], values[arm]["records"]),
        tensor_storage_bytes=values[arm]["resident_tensor_state"]["storage_bytes"],
        tensor_storage_reduction=1-values[arm]["resident_tensor_state"]["storage_bytes"]/size)
        for arm in ARMS if arm != "native"}
    hybrid = comparison["hybrid"]["paired"]["overall"]
    result = dict(status="completed", fingerprint=plan["fingerprint"], confirmed=False, scope=SCOPE,
        primary_arm="hybrid", minimum_correct=47, maximum_accuracy_drop=.05,
        primary_quality_pass=hybrid["compact_correct"] >= 47 and hybrid["accuracy_drop"] <= .05+1e-12,
        accuracy={arm:values[arm]["accuracy"] for arm in ARMS}, comparisons=comparison,
        native_tensor_storage_bytes=size, storage_comparison="Matched FP32 native versus packedFP32 assembled tensors; not a reduction versus the originalBF16 stored model",
        original_native_bf16_storage_measured=False, original_native_bf16_tensor_bytes=None,
        speed_measured=False, confirmation_used=False,
        source_result_sha256={arm:recipe.sha(root/arm/"result.json") for arm in ARMS})
    recipe.write(root/"paired_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize")); parser.add_argument("--root", type=Path, required=True)
    for name in ("source", "block0-root", "block0-verification-root", "block1-root", "parallel-root", "native-cache-root"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.stage == "prepare":
        values = [args.source, args.block0_root, args.block0_verification_root, args.block1_root, args.parallel_root, args.native_cache_root]
        if any(v is None for v in values): parser.error("prepare requires all source, recovered study and native cache roots")
        value = prepare(args.root, *values)
    elif args.stage == "run":
        if args.arm is None: parser.error("run requires --arm")
        value = run(args.root, args.arm)
    else: value = summarize(args.root)
    print(json.dumps(value, allow_nan=False), flush=True)


if __name__ == "__main__": main()
