"""Teacher-assisted projection after an accepted student prefix (development only).

The surrogate x + U U^T(t-x) is Euclidean-optimal among updates in U. It uses
the original teacher's post-block states at all valid tokens and is unattainable at inference.
It is NOT an upper bound on task accuracy or evidence that training can attain
the same features. This first campaign is explicitly limited to block1.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import jspace_next_layer_distill as stage
from .jspace_prefix_localize import validate_development_rows
from .jspace_update_projection_oracle import project_update

recipe = stage.recipe
SCHEMA = "jspace.next-layer-projection.v1"
ARMS = ("native_boundary", "projected_correction")
SCOPE = ("Development-only, unattainable teacher-assisted Euclidean projection surrogate after the accepted "
         "student prefix; NOT a task-accuracy upper bound; no recovery, confirmation, or speed claim")


@torch.inference_mode()
def reconstruct(cases, items, prefix_path, native_input_path, target_path, basis, *, arm,
                normalization_floor, device="cpu", guard=None, chunk_tokens=256):
    """Read development slices only; preserve FP32 student-prefix precision."""
    if arm not in ARMS or type(chunk_tokens) is not int or chunk_tokens < 1:
        raise ValueError("Known arm and positive integer chunk size required")
    if (type(normalization_floor) not in (int, float) or not math.isfinite(normalization_floor)
            or normalization_floor <= 0):
        raise ValueError("A positive finite frozen training normalization floor is required")
    dev = validate_development_rows(cases, items)
    if torch.device(device).type == "cuda" and guard is None:
        raise ValueError("GPU projection requires a reservation guard")
    guard = guard or (lambda: None)
    guard()
    basis = basis.detach().to(device=device, dtype=torch.float32)
    if basis.ndim != 2 or not 0 < basis.shape[1] <= basis.shape[0] or not torch.isfinite(basis).all():
        raise ValueError("Finite nonempty native-width basis required")
    orthogonality = float((basis.T@basis-torch.eye(basis.shape[1], device=device)).abs().max())
    if orthogonality > 5e-5:
        raise ValueError("Orthogonal projection requires an orthonormal basis")
    banks = []
    try:
        for path in (prefix_path, native_input_path, target_path):
            banks.append(np.load(path, mmap_mode="r", allow_pickle=False))
        if (banks[0].dtype != np.float32 or any(b.dtype != np.int16 for b in banks[1:])
                or any(b.ndim != 2 or b.shape != banks[0].shape for b in banks)
                or banks[0].shape[1] != basis.shape[0]):
            raise ValueError("Aligned FP32 prefix and original BF16-bit teacher banks required")
        tokens = sum(len(c["input_ids"]) for c in cases)
        table = torch.empty(tokens, basis.shape[0], dtype=torch.float32)
        virtual, records, cursor = [], [], 0
        for case in cases:
            original = dev[case["identity"]]
            start, stop, length = original["offset"], original["stop"], len(case["input_ids"])
            if not 0 <= start < stop <= len(banks[0]) or stop-start != length:
                raise ValueError("Development offsets differ from aligned boundary banks")
            errors, update_energies, correction_energies = [], [], []
            for local in range(0, length, chunk_tokens):
                guard(); end = min(length, local+chunk_tokens)
                rows = [torch.from_numpy(np.array(b[start+local:start+end], copy=True)) for b in banks]
                x = rows[0].to(device=device, dtype=torch.float32)
                native, target = [r.view(torch.bfloat16).to(device=device, dtype=torch.float32) for r in rows[1:]]
                if not all(torch.isfinite(v).all() for v in (x, native, target)):
                    raise FloatingPointError("Nonfinite development boundary states")
                value = target if arm == "native_boundary" else project_update(x, target, basis)
                if not torch.isfinite(value).all():
                    raise FloatingPointError("Nonfinite teacher-assisted projection")
                table[cursor+local:cursor+end] = value.cpu()
                errors.append((value-target).square().mean(-1).cpu())
                update_energies.append((target-native).square().mean(-1).cpu())
                correction_energies.append((target-x).square().mean(-1).cpu())
            error, energy, correction = map(torch.cat, (errors, update_energies, correction_energies))
            relative = error/energy.clamp_min(normalization_floor)
            copied = deepcopy(case)
            copied["input_ids"] = list(range(cursor, cursor+length)); virtual.append(copied)
            records.append(dict(identity=case["identity"], family=case["family"], original_length=length,
                original_input_ids_digest=recipe.digest(case["input_ids"]), native_offset=start, native_stop=stop,
                virtual_offset=cursor, virtual_stop=cursor+length, mse=float(error.mean()),
                relative_mse=float(relative.mean()), first_relative_mse=float(relative[0]),
                answer_relative_mse=float(relative[-1]),
                content_relative_mse=None if length <= 2 else float(relative[1:-1].mean()),
                original_teacher_update_energy=float(energy.mean()), student_to_target_energy=float(correction.mean()),
                outside_correction_energy_fraction=float(error.sum()/correction.sum().clamp_min(1e-12)),
                zero_native_update_tokens=int((energy == 0).sum())))
            cursor += length
        return table, virtual, dict(records=records, total_tokens=tokens, rank=basis.shape[1],
            native_width=basis.shape[0], basis_orthogonality_max=orthogonality,
            mean_case_mse=sum(r["mse"] for r in records)/len(records),
            mean_case_relative_mse=sum(r["relative_mse"] for r in records)/len(records),
            loss=sum(r["relative_mse"] for r in records)/len(records),
            formula="native_h[l+1]" if arm == "native_boundary" else "x_student + ((native_h[l+1]-x_student) @ U) @ U.T",
            normalization="Equal-case/token MSE divided by ORIGINAL teacher update mean((h[l+1]-h[l])^2), clamped at the frozen pilot's training-derived floor",
            training_derived_normalization_floor=normalization_floor, prefix_dtype="float32 without BF16 round-trip",
            virtual_id_scope="Hidden-table row addresses; original token lengths and RoPE positions retained")
    finally:
        for bank in banks:
            bank._mmap.close()


def pilot_normalization(campaign, parent):
    """Authenticate the existing TRAIN-only normalizer; never fit on these cases."""
    stage.joint.verify_pilot(campaign, parent["fingerprint"])
    value = recipe.read(Path(campaign)/"pilot/result.json")["normalization"]
    median, floor, fraction = (value[k] for k in ("median_teacher_update_energy", "floor", "floor_fraction"))
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (median, floor, fraction))
            or value.get("train_case_count") != 512 or type(value.get("train_token_count")) is not int
            or value["train_token_count"] <= 0 or fraction != parent["config"]["floor_fraction"]
            or not math.isclose(floor, median*fraction, rel_tol=1e-12, abs_tol=0)):
        raise ValueError("Completed pilot must contain the matching TRAIN-only teacher-update normalization")
    return value


def prepare(root, campaign):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA for CPU preparation")
    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    if root.exists():
        raise FileExistsError(root)
    parent = stage.verify(campaign, prefix=True)
    if (parent["layer"] != 1 or parent["trained_layers"] != [1] or parent["rank"] != 1024
            or parent["native_suffix"] != list(range(2, 36))):
        raise ValueError("This diagnostic is bounded to the prepared block1 campaign")
    normalization = pilot_normalization(campaign, parent)
    items = recipe.read(campaign/"items.json")["items"]
    cases = [c for c in items if c["split"] == "development"]
    if len(cases) != 64 or Counter(c["family"] for c in cases) != {f:16 for f in ("arc", "boolq", "hellaswag", "winogrande")}:
        raise ValueError("Original balanced64 development cases required")
    validate_development_rows(cases, items)
    prefix = campaign/"prefix/input.npy"
    prefix_manifest = recipe.read(campaign/"prefix/manifest.json")
    inputs = {str(prefix):prefix_manifest["artifact_sha256"]}
    for key in ("fit_path", "native_input_path", "target_path"):
        path = parent[key]
        inputs[path] = parent["input_sha256"][path]
    for path in (campaign/"plan.json", campaign/"items.json", campaign/"prefix/manifest.json",
                 campaign/"pilot/result.json", campaign/"pilot/manifest.json"):
        inputs[str(path)] = recipe.sha(path)
    for path, expected in inputs.items():
        if recipe.sha(path) != expected:
            raise ValueError("Changed prefix, teacher boundary, or fitted basis")
    sources = dict(parent["source_sha256"])
    for path in (Path(__file__).resolve(), recipe.ROOT/"tests/structure_net/test_jspace_next_projection.py",
                 Path(__file__).with_name("jspace_update_projection_oracle.py"),
                 Path(__file__).with_name("jspace_prefix_localize.py")):
        sources[str(path)] = recipe.sha(path)
    plan = dict(schema=SCHEMA, campaign=str(campaign), campaign_fingerprint=parent["fingerprint"],
        source=parent["source"], source_fingerprint=parent["source_fingerprint"], layer=1,
        prefix_path=str(prefix), fit_path=parent["fit_path"], native_input_path=parent["native_input_path"],
        target_path=parent["target_path"], input_sha256=inputs, source_sha256=sources,
        arms=list(ARMS), case_identities=[c["identity"] for c in cases], original_suffix=list(range(2, 36)),
        basis_fingerprint=parent["basis_fingerprint"], current_learned_checkpoint_used=False,
        accepted_prefix_checkpoint_used=True, split="development", confirmation_used=False,
        normalization=normalization, normalization_source="Authenticated completed pilot/result.json; fitted on original512 training cases only",
        dtype="FP32 accepted prefix/projection/suffix; original BF16 teacher cache promoted toFP32", scope=SCOPE)
    plan["fingerprint"] = recipe.digest(plan); root.mkdir(parents=True)
    for path, expected in sources.items():
        if recipe.sha(path) != expected:
            raise ValueError("Changed diagnostic producer")
        destination = root/"producer"/Path(path).relative_to(recipe.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
    recipe.write(root/"plan.json", plan)
    return dict(status="prepared", fingerprint=plan["fingerprint"], count=64, arms=list(ARMS))


def verify(root):
    root = Path(root); plan = recipe.read(root/"plan.json")
    if plan.get("schema") != SCHEMA or plan["fingerprint"] != recipe.digest({k:v for k,v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed projection diagnostic plan")
    parent = stage.verify(plan["campaign"], prefix=True)
    if parent["fingerprint"] != plan["campaign_fingerprint"]:
        raise ValueError("Changed parent campaign")
    for path, expected in plan["input_sha256"].items():
        if recipe.sha(path) != expected:
            raise ValueError("Changed diagnostic input: "+path)
    for path, expected in plan["source_sha256"].items():
        if recipe.sha(path) != expected or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT)) != expected:
            raise ValueError("Changed diagnostic producer")
    items = recipe.read(Path(plan["campaign"])/"items.json")["items"]
    cases = [c for c in items if c["split"] == "development"]
    if [c["identity"] for c in cases] != plan["case_identities"]:
        raise ValueError("Changed development identities")
    return plan, items, cases


def run(root, arm):
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_native_block, load_readout_norm
    root = Path(root).resolve(); plan, items, cases = verify(root)
    if arm not in plan["arms"]:
        raise ValueError("Arm was not frozen in the diagnostic plan")
    directory = root/arm; directory.mkdir(exist_ok=False)
    lease = None; cuda_started = False; started = time.monotonic()
    try:
        lease = wait_ready([0]); allocation = dict(lease["allocation"])
        allocator = stage.cap_allocator(lease)
        torch.cuda.init(); cuda_started = True; torch.cuda.reset_peak_memory_stats()
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        def guard():
            current = wait_ready([0])
            if current["id"] != lease["id"] or current["allocation"] != allocation:
                raise RuntimeError("Projection diagnostic reservation changed")
            stage.joint.check_reserved_peak(allocation, torch.cuda.max_memory_reserved())
        original = recipe.read(Path(plan["source"])/"plan.json")
        table, virtual, geometry = reconstruct(cases, items, plan["prefix_path"], plan["native_input_path"],
            plan["target_path"], recipe.load(plan["fit_path"])["basis"], arm=arm,
            normalization_floor=plan["normalization"]["floor"], device="cuda:0", guard=guard)
        embedding, norm, _ = load_readout_norm(original["model"]["path"], device="cpu", dtype=torch.bfloat16)
        def loader(local):
            guard()
            return load_native_block(original["model"]["path"], plan["original_suffix"][local],
                device="cuda:0", dtype=torch.bfloat16).float()
        value = stage.single.stream_score(virtual, input_embedding=table, block_loader=loader,
            final_readout=(norm.float(), embedding.float()), label_ids=original["label_ids"],
            scratch_dir=directory/"scratch", rotary=recipe.rotary_module(original, "cuda:0"),
            layer_count=len(plan["original_suffix"]), device="cuda:0", guard=guard, microbatch=4,
            progress=lambda **kw:recipe.event(root, "progress", arm=arm, detail=kw))
        guard()
        value.update(status="completed", fingerprint=plan["fingerprint"], arm=arm, reconstruction=geometry,
            accuracy=stage.single.ranking_summary(value["records"]), scope=SCOPE, confirmed=False,
            elapsed_seconds=time.monotonic()-started, lease_id=lease["id"], allocation=allocation,
            allocator=allocator, peak_reserved_bytes=torch.cuda.max_memory_reserved())
        recipe.write(directory/"result.json", value)
        recipe.write(directory/"manifest.json", dict(status="completed", fingerprint=plan["fingerprint"], arm=arm,
            result_sha256=recipe.sha(directory/"result.json"), lease_id=lease["id"], allocation=allocation,
            peak_reserved_bytes=value["peak_reserved_bytes"]))
        return dict(status="completed", arm=arm, accuracy=value["accuracy"])
    except BaseException as error:
        recipe.write(directory/"failure.json", dict(status="failed", fingerprint=plan["fingerprint"],
            error=repr(error), lease_id=None if lease is None else lease["id"],
            allocation=None if lease is None else lease["allocation"],
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None))
        raise


def summarize(root):
    from .jspace_single_block_verify import validate_records
    root = Path(root); plan, _, cases = verify(root); values = {}
    for arm in ARMS:
        path = root/arm/"result.json"; manifest = recipe.read(root/arm/"manifest.json")
        if (manifest.get("status") != "completed" or manifest.get("fingerprint") != plan["fingerprint"]
                or manifest.get("arm") != arm or manifest.get("result_sha256") != recipe.sha(path)):
            raise ValueError("Both authenticated completed diagnostic arms required")
        values[arm] = recipe.read(path)
        validate_records(values[arm]["records"], cases)
    before, after = (values[arm]["records"] for arm in ARMS)
    result = dict(status="completed", fingerprint=plan["fingerprint"], count=len(cases),
        gained=sum(a["prediction"] != a["answer"] and b["prediction"] == b["answer"] for a,b in zip(before, after)),
        lost=sum(a["prediction"] == a["answer"] and b["prediction"] != b["answer"] for a,b in zip(before, after)),
        changed_predictions=sum(a["prediction"] != b["prediction"] for a,b in zip(before, after)),
        accuracy={arm:values[arm]["accuracy"] for arm in ARMS}, scope=SCOPE, confirmed=False,
        source_result_sha256={arm:recipe.sha(root/arm/"result.json") for arm in ARMS})
    recipe.write(root/"paired_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.stage == "prepare" and args.campaign is None:
        parser.error("prepare requires --campaign")
    if args.stage == "run" and args.arm is None:
        parser.error("run requires --arm")
    result = prepare(args.root, args.campaign) if args.stage == "prepare" else (
        run(args.root, args.arm) if args.stage == "run" else summarize(args.root))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
