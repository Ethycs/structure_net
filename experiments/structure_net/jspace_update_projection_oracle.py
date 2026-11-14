"""Native block0 update projection, followed by the original frozen suffix.

The projected update is the closest teacher update in the fixed basis under
Euclidean feature error. It is a surrogate, NOT an upper bound on task accuracy:
a trained block may produce different features that give better answers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import jspace_single_block_distill as single
from .jspace_prefix_localize import validate_development_rows

recipe = single.recipe
ARMS = ("native_boundary", "projected_update")
SCHEMA = "jspace.native-update-projection-oracle.v1"
SCOPE = ("Development-only teacher-update Euclidean projection surrogate, not a task-accuracy upper bound; "
         "original layers1..35 and readout; no learned checkpoint, confirmation, or speed claim")


def project_update(hidden, target, basis):
    if (hidden.shape != target.shape or hidden.ndim != 2 or basis.ndim != 2
            or hidden.shape[1] != basis.shape[0] or not 0 < basis.shape[1] <= basis.shape[0]):
        raise ValueError("Aligned native token rows and a nonempty basis required")
    return hidden + ((target - hidden) @ basis) @ basis.T


@torch.inference_mode()
def reconstruct(cases, items, input_path, target_path, basis, *, arm, device="cpu", guard=None, chunk_tokens=256):
    """Read only development slices; virtual token IDs address hidden rows."""
    if arm not in ARMS or chunk_tokens < 1:
        raise ValueError("Known oracle arm and positive chunk size required")
    dev = validate_development_rows(cases, items)
    target_device = torch.device(device)
    if target_device.type == "cuda" and guard is None:
        raise ValueError("GPU projection requires a reservation guard")
    guard = guard or (lambda: None); guard()
    basis = basis.detach().to(device=device, dtype=torch.float32)
    if basis.ndim != 2 or not 0 < basis.shape[1] <= basis.shape[0] or not torch.isfinite(basis).all():
        raise ValueError("Finite native-width basis required")
    orthogonality = float((basis.T @ basis - torch.eye(basis.shape[1], device=device)).abs().max())
    if orthogonality > 5e-5:
        raise ValueError("Orthogonal projection requires an orthonormal basis")
    banks = []
    try:
        for path in (input_path, target_path):
            banks.append(np.load(path, mmap_mode="r", allow_pickle=False))
        if (banks[0].dtype != np.int16 or banks[1].dtype != np.int16
                or banks[0].ndim != 2 or banks[0].shape != banks[1].shape or banks[0].shape[1] != basis.shape[0]):
            raise ValueError("Aligned original BF16-bit native boundary banks required")
        tokens = sum(len(case["input_ids"]) for case in cases)
        hidden_table = torch.empty(tokens, basis.shape[0], dtype=torch.float32, device="cpu")
        virtual, records, cursor = [], [], 0
        for case in cases:
            source = dev[case["identity"]]; start, stop = source["offset"], source["stop"]
            length = len(case["input_ids"])
            if not 0 <= start < stop <= len(banks[0]) or stop-start != length:
                raise ValueError("Development offsets differ from original native banks")
            errors, energies = [], []
            for local in range(0, length, chunk_tokens):
                guard(); end = min(length, local+chunk_tokens)
                h0, h1 = [torch.from_numpy(np.array(bank[start+local:start+end], copy=True)).view(torch.bfloat16)
                          .to(device=device, dtype=torch.float32) for bank in banks]
                if not torch.isfinite(h0).all() or not torch.isfinite(h1).all():
                    raise FloatingPointError("Nonfinite cached development states")
                values = h1 if arm == "native_boundary" else project_update(h0, h1, basis)
                hidden_table[cursor+local:cursor+end] = values.cpu()
                errors.append((values-h1).square().mean(-1).cpu())
                energies.append((h1-h0).square().mean(-1).cpu())
            error, energy = torch.cat(errors), torch.cat(energies)
            relative = error / energy.clamp_min(torch.finfo(torch.float32).tiny)
            copied = deepcopy(case); copied["input_ids"] = list(range(cursor, cursor+length)); virtual.append(copied)
            records.append(dict(identity=case["identity"], family=case["family"], original_length=length,
                original_input_ids_digest=recipe.digest(case["input_ids"]), native_offset=start, native_stop=stop,
                virtual_offset=cursor, virtual_stop=cursor+length, mse=float(error.mean()), relative_mse=float(relative.mean()),
                first_relative_mse=float(relative[0]), answer_relative_mse=float(relative[-1]),
                content_relative_mse=None if length <= 2 else float(relative[1:-1].mean()),
                teacher_update_energy_min=float(energy.min()), zero_update_tokens=int((energy == 0).sum())))
            cursor += length
        return hidden_table, virtual, dict(records=records, total_tokens=tokens, rank=basis.shape[1],
            native_width=basis.shape[0], basis_orthogonality_max=orthogonality,
            mean_case_relative_mse=sum(r["relative_mse"] for r in records)/len(records),
            mean_case_mse=sum(r["mse"] for r in records)/len(records),
            formula="cached h1" if arm == "native_boundary" else "cached h0 + ((cached h1 - cached h0) @ U) @ U.T",
            normalization="Equal-case mean of equal-token MSE / native update energy; zero-energy updates have zero error",
            logical_native_read_bytes=tokens*basis.shape[0]*2*2,
            virtual_id_scope="Internal hidden-table row addresses, never vocabulary IDs or retokenized prompts")
    finally:
        for bank in banks:
            bank._mmap.close()


def prepare(root, campaign):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA for CPU preparation")
    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    if root.exists():
        raise FileExistsError(root)
    parent = single.verify(campaign)
    if parent["trained_layers"] != [0] or parent["native_suffix"] != list(range(1, 36)) or parent["rank"] != 1024:
        raise ValueError("The frozen single block0, rank1024 intervention is required")
    items = recipe.read(campaign / "items.json")["items"]
    cases = [case for case in items if case["split"] == "development"]
    if len(cases) != 64 or Counter(c["family"] for c in cases) != {f:16 for f in ("arc", "boolq", "hellaswag", "winogrande")}:
        raise ValueError("Original balanced 64 development cases required")
    validate_development_rows(cases, items)
    basis_plan = recipe.read(Path(parent["basis_root"]) / "plan.json")
    training = {case["identity"] for case in items if case["split"] == "train"}
    fitting, checks = set(basis_plan["fit_training_identities"]), set(basis_plan["internal_training_check_identities"])
    if not fitting or not checks or fitting & checks or not (fitting | checks) <= training:
        raise ValueError("Fixed enriched basis must use training identities only")
    inputs = {str(path): recipe.sha(path) for path in (campaign / "plan.json", campaign / "items.json",
        Path(parent["source"]) / "plan.json", Path(parent["basis_root"]) / "plan.json")}
    for key in ("fit_path", "input_path", "target_path"):
        path = parent[key]; expected = parent["input_sha256"][path]
        if recipe.sha(path) != expected:
            raise ValueError("Changed enriched basis or native boundary bank")
        inputs[path] = expected
    paths = [Path(__file__), Path(__file__).with_name("jspace_prefix_localize.py"),
             recipe.ROOT / "tests/structure_net/test_jspace_update_projection_oracle.py"]
    plan = dict(schema=SCHEMA, campaign=str(campaign), campaign_fingerprint=parent["fingerprint"],
        source=parent["source"], source_fingerprint=parent["source_fingerprint"], basis_fingerprint=parent["basis_fingerprint"],
        fit_path=parent["fit_path"], input_path=parent["input_path"], target_path=parent["target_path"],
        input_sha256=inputs, source_sha256={str(path):recipe.sha(path) for path in paths},
        arms=list(ARMS), case_identities=[case["identity"] for case in cases], original_suffix=list(range(1,36)),
        basis_fitted_on_training_only=True, selected_checkpoint_used=False, confirmation_used=False,
        dtype="Native BF16 cached boundary bits decoded losslessly into FP32; FP32 projection and native suffix",
        scope=SCOPE)
    plan["fingerprint"] = recipe.digest(plan); root.mkdir(parents=True)
    for path in paths:
        target = root / "producer" / path.relative_to(recipe.ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    recipe.write(root / "plan.json", plan)
    return dict(status="prepared", fingerprint=plan["fingerprint"], arms=list(ARMS), count=64)


def verify(root):
    root = Path(root); plan = recipe.read(root / "plan.json")
    if plan["schema"] != SCHEMA or plan["fingerprint"] != recipe.digest({k:v for k,v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed projection-oracle plan")
    parent = single.verify(plan["campaign"])
    if parent["fingerprint"] != plan["campaign_fingerprint"]:
        raise ValueError("Changed frozen single-block campaign")
    for path, digest in plan["input_sha256"].items():
        if recipe.sha(path) != digest:
            raise ValueError("Changed oracle input: " + path)
    for path, digest in plan["source_sha256"].items():
        if recipe.sha(path) != digest or recipe.sha(root / "producer" / Path(path).relative_to(recipe.ROOT)) != digest:
            raise ValueError("Changed oracle producer")
    items = recipe.read(Path(plan["campaign"]) / "items.json")["items"]
    cases = [case for case in items if case["split"] == "development"]
    if [c["identity"] for c in cases] != plan["case_identities"]:
        raise ValueError("Development population changed")
    return plan, items, cases


def run(root, arm):
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_native_block, load_readout_norm
    root = Path(root).resolve(); plan, items, cases = verify(root)
    if arm not in plan["arms"]:
        raise ValueError("Arm was not frozen in the oracle plan")
    directory = root / arm; directory.mkdir(exist_ok=False)
    started = time.monotonic(); lease = None
    try:
        lease = wait_ready([0]); torch.cuda.init(); torch.cuda.reset_peak_memory_stats()
        def guard():
            current = wait_ready([0])
            if current["id"] != lease["id"] or current["allocation"] != lease["allocation"]:
                raise RuntimeError("Projection-oracle reservation changed")
            single.joint.check_reserved_peak(lease["allocation"], torch.cuda.max_memory_reserved())
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        original = recipe.read(Path(plan["source"]) / "plan.json")
        basis = recipe.load(plan["fit_path"])["basis"]
        hidden, virtual, reconstruction = reconstruct(cases, items, plan["input_path"], plan["target_path"],
            basis, arm=arm, device="cuda:0", guard=guard)
        del basis
        embedding, norm, _ = load_readout_norm(original["model"]["path"], device="cpu", dtype=torch.bfloat16)
        def loader(layer):
            guard()
            return load_native_block(original["model"]["path"], layer+1, device="cuda:0", dtype=torch.bfloat16).float()
        value = single.stream_score(virtual, input_embedding=hidden, block_loader=loader,
            final_readout=(norm.float(), embedding.float()), label_ids=original["label_ids"],
            scratch_dir=directory / "scratch", rotary=recipe.rotary_module(original, "cuda:0"),
            layer_count=35, device="cuda:0", guard=guard, microbatch=4,
            progress=lambda **kw: print(json.dumps(dict(arm=arm, **kw)), flush=True))
        guard(); records = value["records"]
        value.update(status="completed", fingerprint=plan["fingerprint"], arm=arm, reconstruction=reconstruction,
            accuracy=single.ranking_summary(records), scope=SCOPE, confirmed=False,
            elapsed_seconds=time.monotonic()-started, lease_id=lease["id"], allocation=lease["allocation"],
            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        recipe.write(directory / "result.json", value)
        recipe.write(directory / "manifest.json", dict(status="completed", fingerprint=plan["fingerprint"],
            result_sha256=recipe.sha(directory / "result.json"), arm=arm, lease_id=lease["id"]))
        return dict(status="completed", arm=arm, accuracy=value["accuracy"], result_sha256=recipe.sha(directory / "result.json"))
    except BaseException as error:
        recipe.write(directory / "failure.json", dict(status="failed", fingerprint=plan["fingerprint"],
            error=repr(error), lease_id=None if lease is None else lease["id"]))
        raise


def paired_summary(root):
    root = Path(root); plan, _, cases = verify(root); values = {}
    for arm in ARMS:
        directory = root / arm; manifest = recipe.read(directory / "manifest.json")
        path = directory / "result.json"
        if (manifest.get("status") != "completed" or manifest.get("fingerprint") != plan["fingerprint"]
                or manifest.get("arm") != arm or manifest.get("result_sha256") != recipe.sha(path)):
            raise ValueError("Both completed and authenticated oracle arms required")
        values[arm] = recipe.read(path)
    before, after = (values[arm]["records"] for arm in ARMS)
    for rows in (before, after):
        if len(rows) != len(cases) or any(any(row[k] != case[k] for k in ("identity", "family", "answer")) for row,case in zip(rows,cases)):
            raise ValueError("Oracle arms must preserve exactly matched development cases")
    gained = sum(a["prediction"] != a["answer"] and b["prediction"] == b["answer"] for a,b in zip(before,after))
    lost = sum(a["prediction"] == a["answer"] and b["prediction"] != b["answer"] for a,b in zip(before,after))
    result = dict(status="completed", fingerprint=plan["fingerprint"], count=len(cases), gained=gained, lost=lost,
        changed_predictions=sum(a["prediction"] != b["prediction"] for a,b in zip(before,after)),
        accuracy={arm:values[arm]["accuracy"] for arm in ARMS}, scope=SCOPE, confirmed=False,
        source_result_sha256={arm:recipe.sha(root / arm / "result.json") for arm in ARMS})
    recipe.write(root / "paired_result.json", result); return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--campaign", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.stage == "prepare" and args.campaign is None: parser.error("prepare requires --campaign")
    if args.stage == "run" and args.arm is None: parser.error("run requires --arm")
    result = (prepare(args.root, args.campaign) if args.stage == "prepare" else
              run(args.root, args.arm) if args.stage == "run" else paired_summary(args.root))
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__": main()
