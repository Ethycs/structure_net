"""Fresh-process verification of a completed single-block recovery checkpoint.

This consumes only the original development population. A verification pass
does not itself establish held-out quality or authorize another compression.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_single_block_distill as single

recipe = single.recipe
SCHEMA = "jspace.single-block-independent-verification.v1"


@dataclass(frozen=True)
class Tolerances:
    feature_atol: float = 1e-5
    feature_rtol: float = 2e-4
    logit_atol: float = 5e-4
    logit_rtol: float = 1e-5

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Finite nonnegative numerical tolerances required")


def completed_training(campaign):
    """Authenticate metadata and selected bytes before any checkpoint load."""
    campaign = Path(campaign).resolve()
    runtime_inputs = {}; runtime_fingerprint = None
    if (campaign / "runtime_policy.json").exists():
        from . import jspace_single_block_reserved as reserved
        plan, policy = reserved.verify(campaign)
        reserved.verify_runtime_receipt(campaign, "train", plan, policy)
        runtime_fingerprint = policy["fingerprint"]
        runtime_inputs = {str(path): recipe.sha(path) for path in
            (campaign / "runtime_policy.json", campaign / "runtime_train.json")}
    else:
        if any(campaign.glob("runtime_*.json")):
            raise ValueError("Runtime receipts require their frozen allocator policy")
        plan = single.verify(campaign)
    directory = campaign / "train"
    manifest = recipe.read(directory / "manifest.json")
    result_path = directory / "result.json"
    if (manifest.get("status") != "completed" or manifest.get("systems_only") is not False
            or manifest.get("fingerprint") != plan["fingerprint"]
            or recipe.sha(result_path) != manifest.get("result_sha256")):
        raise ValueError("Completed matching scientific training manifest required")
    result = recipe.read(result_path)
    if (result.get("status") != "completed" or result.get("systems_only") is not False
            or result.get("strict_reload_exact") is not True):
        raise ValueError("Completed training and its strict reload are required")
    history = result["history"]
    if [row["epoch"] for row in history] != list(range(plan["config"]["epochs"] + 1)):
        raise ValueError("Complete ordered epoch history required")
    selected_epoch = result["selected_epoch"]
    expected = max(history, key=lambda row: (row["metrics"]["correct"], -row["metrics"]["relative_mse"]))
    if expected["epoch"] != selected_epoch or expected["metrics"] != result["selected"]:
        raise ValueError("Selected checkpoint differs from the frozen selection rule")
    checkpoint = (directory / f"epoch_{selected_epoch:02d}.pt").resolve()
    reference_path = directory / f"epoch_{selected_epoch:02d}_development.json"
    if (Path(result["selected_checkpoint"]).resolve() != checkpoint
            or expected["checkpoint"] != checkpoint.name
            or recipe.sha(checkpoint) != expected["checkpoint_sha256"]
            or recipe.sha(reference_path) != expected.get("development_sha256")):
        raise ValueError("Selected checkpoint or authenticated development reference changed")
    reference = recipe.read(reference_path)
    items = recipe.read(campaign / "items.json")["items"]
    cases = [case for case in items if case["split"] == "development"]
    if reference["metrics"] != result["selected"] or len(cases) != 64:
        raise ValueError("The original 64 development cases and selected metrics are required")
    validate_records(reference["records"], cases)
    if sum(r["prediction"] == r["answer"] for r in reference["records"]) != result["selected"]["correct"]:
        raise ValueError("Reference predictions disagree with selected accuracy")
    inputs = {str(p): recipe.sha(p) for p in (campaign / "plan.json", campaign / "items.json",
        directory / "manifest.json", result_path, checkpoint, reference_path)}
    inputs.update(runtime_inputs)
    # The fit and native feature banks are consumed again by this process.
    for key in ("fit_path", "input_path", "target_path"):
        path = plan[key]
        expected_hash = plan["input_sha256"][path]
        if recipe.sha(path) != expected_hash:
            raise ValueError("Changed native pair or enriched geometry: " + path)
        inputs[path] = expected_hash
    return dict(plan=plan, result=result, reference=reference, checkpoint=checkpoint,
                items=items, input_sha256=inputs, runtime_policy_fingerprint=runtime_fingerprint)


def validate_records(records, cases):
    if len(records) != len(cases):
        raise ValueError("Development record count changed")
    for record, case in zip(records, cases):
        if any(record[key] != case[key] for key in ("identity", "family", "answer")):
            raise ValueError("Development identities, order or labels changed")
        logits = record["logits"]
        if (len(logits) != len(case["choices"]) or not all(math.isfinite(v) for v in logits)
                or record["prediction"] != max(range(len(logits)), key=logits.__getitem__)):
            raise ValueError("Finite choice logits and their argmax are required")


def compare(reference, features, records, cases, tolerances=Tolerances()):
    validate_records(reference["records"], cases)
    validate_records(records, cases)
    feature_checks = {}
    for key, value in features.items():
        expected = reference["metrics"][key]
        passed = math.isfinite(value) and math.isclose(value, expected,
            abs_tol=tolerances.feature_atol, rel_tol=tolerances.feature_rtol)
        feature_checks[key] = dict(reference=expected, observed=value,
                                  absolute_delta=abs(value - expected), passed=passed)
    maximum = 0.; logits_pass = True; changed = []
    for before, after in zip(reference["records"], records):
        if before["prediction"] != after["prediction"]:
            changed.append(before["identity"])
        for expected, value in zip(before["logits"], after["logits"]):
            maximum = max(maximum, abs(value - expected))
            logits_pass &= abs(value - expected) <= tolerances.logit_atol + tolerances.logit_rtol * abs(expected)
    feature_pass = bool(feature_checks) and all(v["passed"] for v in feature_checks.values())
    return dict(passed=feature_pass and logits_pass and not changed, feature_pass=feature_pass,
        logits_pass=bool(logits_pass), predictions_exact=not changed,
        changed_prediction_identities=changed, logit_max_abs_delta=maximum,
        feature_checks=feature_checks, tolerances=asdict(tolerances), count=len(cases),
        correct=sum(r["prediction"] == r["answer"] for r in records))


def prepare(campaign, root, *, tolerances=Tolerances()):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA for CPU-only preparation")
    campaign, root = Path(campaign).resolve(), Path(root).resolve()
    if root.exists():
        raise FileExistsError(root)
    checked = completed_training(campaign)
    producers = [Path(__file__), recipe.ROOT / "tests/structure_net/test_jspace_single_block_verify.py"]
    plan = dict(schema=SCHEMA, campaign=str(campaign), campaign_fingerprint=checked["plan"]["fingerprint"],
        runtime_policy_fingerprint=checked["runtime_policy_fingerprint"],
        selected_epoch=checked["result"]["selected_epoch"], selected_checkpoint=str(checked["checkpoint"]),
        checkpoint_sha256=checked["input_sha256"][str(checked["checkpoint"])],
        tolerances=asdict(tolerances), input_sha256=checked["input_sha256"],
        source_sha256={str(p): recipe.sha(p) for p in producers},
        scope="Fresh-process development-only reload and packed execution parity; no quality threshold or confirmation claim")
    plan["fingerprint"] = recipe.digest(plan)
    root.mkdir(parents=True)
    for path in producers:
        target = root / "producer" / path.relative_to(recipe.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    recipe.write(root / "plan.json", plan)
    return dict(status="prepared", fingerprint=plan["fingerprint"])


def verify(root):
    root = Path(root).resolve(); plan = recipe.read(root / "plan.json")
    if plan["schema"] != SCHEMA or plan["fingerprint"] != recipe.digest({k:v for k,v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed verification plan")
    for path, digest in plan["source_sha256"].items():
        if recipe.sha(path) != digest or recipe.sha(root / "producer" / Path(path).relative_to(recipe.ROOT)) != digest:
            raise ValueError("Changed verifier source")
    for path, digest in plan["input_sha256"].items():
        if recipe.sha(path) != digest:
            raise ValueError("Changed verification input: " + path)
    checked = completed_training(plan["campaign"])
    if (checked["plan"]["fingerprint"] != plan["campaign_fingerprint"]
            or str(checked["checkpoint"]) != plan["selected_checkpoint"]
            or checked["runtime_policy_fingerprint"] != plan["runtime_policy_fingerprint"]):
        raise ValueError("Verification campaign or selection changed")
    return plan, checked


def load_selected(block, checked):
    """Call only after completion/hash gates; recheck the exact bytes at load."""
    path = checked["checkpoint"]
    if recipe.sha(path) != checked["input_sha256"][str(path)]:
        raise ValueError("Selected checkpoint changed before loading")
    state = recipe.load(path)
    if state["epoch"] != checked["result"]["selected_epoch"] or state["normalization"] != checked["result"]["normalization"]:
        raise ValueError("Checkpoint epoch or training-only normalization changed")
    block.load_state_dict(state["block"], strict=True)
    return block.eval().requires_grad_(False)


@torch.no_grad()
def feature_metrics(block, rotary, data, *, floor, microbatch, device, guard):
    totals = {}
    for start in range(0, len(data.development), microbatch):
        guard(); indices = data.development[start:start + microbatch]
        batch = data.batch(indices, device)
        output = block(batch["hidden"], attention_mask=batch["mask"], position_ids=batch["positions"],
                       position_embeddings=rotary(batch["hidden"], batch["positions"]))
        values = single.feature_losses(output, batch["hidden"], batch["target"], batch["valid"],
                                       floor=floor, basis=block.basis)
        for key, value in values.items():
            totals[key] = totals.get(key, 0.) + float(value) * len(indices)
    return {key: value / len(data.development) for key, value in totals.items()}


@torch.no_grad()
def run(root, *, device="cuda:0"):
    from .jspace_qwen_parallel_model import load_native_block, load_readout_norm
    from .jspace_full_lease import wait_ready
    root = Path(root).resolve(); plan, checked = verify(root)
    directory = root / "verification"
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(); started = time.monotonic(); lease = None; data = None
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("CPU or reserved CUDA execution required")
    try:
        if target.type == "cuda":
            lease = wait_ready([target.index or 0]); torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(target)
        def guard():
            if lease is not None:
                current = wait_ready([target.index or 0])
                if current["id"] != lease["id"] or current["allocation"] != lease["allocation"]:
                    raise RuntimeError("Verification reservation changed")
                single.joint.check_reserved_peak(lease["allocation"], torch.cuda.max_memory_reserved(target))
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        parent = recipe.read(Path(checked["plan"]["source"]) / "plan.json")
        guard(); native = load_native_block(parent["model"]["path"], 0, device=device, dtype=torch.bfloat16)
        block = single.build_enriched_block(native, recipe.load(checked["plan"]["fit_path"]), parameter_dtype=torch.float32)
        del native
        block = load_selected(block, checked)
        rotary = recipe.rotary_module(parent, device)
        data = single.NativePairs(checked["items"], checked["plan"]["input_path"], checked["plan"]["target_path"])
        kwargs = dict(floor=checked["result"]["normalization"]["floor"],
            microbatch=checked["plan"]["config"]["microbatch"], device=device, guard=guard)
        unpacked_features = feature_metrics(block, rotary, data, **kwargs)
        packed = deepcopy(block).pack(dtype=torch.float32, buffer_dtype=torch.float32)
        del block
        packed_features = feature_metrics(packed, rotary, data, **kwargs)
        data.close(); data = None
        cases = [case for case in checked["items"] if case["split"] == "development"]
        embedding, norm, _ = load_readout_norm(parent["model"]["path"], device="cpu", dtype=torch.bfloat16)
        embedding, norm = embedding.float(), norm.float()
        snapshots = [packed]; del packed
        def loader(layer):
            guard()
            return snapshots.pop() if layer == 0 else load_native_block(parent["model"]["path"], layer,
                device=device, dtype=torch.bfloat16).float()
        scored = single.stream_score(cases, input_embedding=embedding, block_loader=loader,
            final_readout=(norm, embedding), label_ids=parent["label_ids"],
            scratch_dir=directory / "scratch", rotary=rotary, layer_count=36, device=device,
            guard=guard, microbatch=4)
        tolerance = Tolerances(**plan["tolerances"])
        unpacked = compare(checked["reference"], unpacked_features, scored["records"], cases, tolerance)
        packed = compare(checked["reference"], packed_features, scored["records"], cases, tolerance)
        guard()
        result = dict(status="completed", fingerprint=plan["fingerprint"], campaign_fingerprint=plan["campaign_fingerprint"],
            runtime_policy_fingerprint=plan["runtime_policy_fingerprint"],
            checkpoint_sha256=plan["checkpoint_sha256"], selected_epoch=plan["selected_epoch"],
            passed=unpacked["passed"] and packed["passed"], strict_state_load=True,
            unpacked=unpacked, packed=packed, records=scored["records"],
            training_quality_pass=checked["result"]["quality_pass"], confirmed=False,
            elapsed_seconds=time.monotonic()-started, process_id=os.getpid(),
            lease_id=None if lease is None else lease["id"], allocation=None if lease is None else lease["allocation"],
            peak_reserved_bytes=None if lease is None else torch.cuda.max_memory_reserved(target), scope=plan["scope"])
        recipe.write(directory / "result.json", result)
        recipe.write(directory / "manifest.json", dict(status="completed" if result["passed"] else "failed",
            fingerprint=plan["fingerprint"], result_sha256=recipe.sha(directory / "result.json"),
            checkpoint_sha256=plan["checkpoint_sha256"], passed=result["passed"]))
        if not result["passed"]:
            raise RuntimeError("Independent selected checkpoint parity failed; inspect verification/result.json")
        return result
    except BaseException as error:
        recipe.write(directory / "failure.json", dict(status="failed", fingerprint=plan["fingerprint"], error=repr(error)))
        raise
    finally:
        if data is not None:
            data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run")); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path); parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "prepare" and args.campaign is None:
        parser.error("prepare requires --campaign")
    result = prepare(args.campaign, args.root) if args.stage == "prepare" else run(args.root, device=args.device)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
