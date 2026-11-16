"""Paired, singleton BF16 sanity check of the accepted fourteen-block suffix.

The native phase computes raw-token answers and stores its unchanged prefix.
The compact phase consumes those lossless states. Neither phase trains, selects
weights, reads final confirmation questions, or measures inference throughput.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import jspace_3b_feature_resident_campaign as source
from . import jspace_3b_sanity_data as data_source
from . import jspace_3b_final_evaluate as execution
from .jspace_full_lease import wait_ready
from .jspace_qwen_parallel_evaluate import paired_quality

base, old, read, builder = source.base, source.old, source.read, source.model
SCHEMA = "jspace.3b-fourteen-sanity.v1"
MODULE = "experiments.structure_net.jspace_3b_sanity_evaluate"
PROTOCOL = base.ROOT/"docs/07 - Status Reports/jspace/2026-11-15_qwen3b-fourteen-block-benchmark-sanity.md"
CUT = 22
CHECKPOINT_SHA = "814563a4de04d91c17bb02d670ed2d5a7609d3bb3460ed92073d721deac7e401"
POLICY = dict(dtype="bfloat16", attention="eager", batch_size=1, padding=False,
    tf32=False, readout="identical_choice_row_linear", boundary="pre_native_block22_lossless_bfloat16_bits",
    count=128, per_family=32, maximum_net_lost_correct=6, confirmed=False)


def _record(case, hidden, E, label_ids):
    count = len(case["choices"])
    if not 0 < count <= len(label_ids):
        raise ValueError("Invalid choice count")
    logits = F.linear(hidden, E[label_ids[:count]])[0].float()
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite choice logits")
    return dict(identity=case["identity"], family=case["family"], answer=case["answer"],
        prediction=int(logits.argmax()), logits=logits.cpu().tolist())


@torch.inference_mode()
def native_case(model, case, label_ids, cut, guard=lambda: None):
    """Return choice record and exact pre-cut residual [1,L,d] on CPU."""
    guard()
    if type(cut) is not int or not 0 <= cut < len(model.model.layers):
        raise ValueError("Invalid native cut")
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError("Expected original tied embedding/readout")
    captured = []
    def capture(_module, args):
        captured.append(args[0].detach().cpu().clone())
    handle = model.model.layers[cut].register_forward_pre_hook(capture)
    try:
        ids = torch.tensor([case["input_ids"]], device=model.lm_head.weight.device, dtype=torch.long)
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
        result = _record(case, hidden, model.lm_head.weight, label_ids)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape[1] != len(case["input_ids"]):
        raise ValueError("Native boundary hook did not capture exactly one complete prompt")
    if not torch.isfinite(captured[0]).all():
        raise ValueError("Nonfinite native boundary")
    return result, captured[0]


@torch.inference_mode()
def compact_case(stack, norm, E, rotary, boundary, case, label_ids, guard=lambda: None):
    """Execute only the selected suffix, retaining original absolute positions."""
    guard()
    length = len(case["input_ids"])
    if boundary.ndim != 3 or boundary.shape[:2] != (1, length) or boundary.shape[-1] != E.shape[1]:
        raise ValueError("Boundary shape does not match the original unpadded prompt")
    if boundary.dtype != E.dtype or not torch.isfinite(boundary).all():
        raise ValueError("Boundary must preserve the original finite execution dtype")
    hidden = boundary.to(device=E.device)
    positions = torch.arange(length, device=E.device).unsqueeze(0)
    mask = torch.full((length, length), torch.finfo(hidden.dtype).min, device=E.device,
                      dtype=hidden.dtype).triu(1)[None, None]
    output = stack(hidden, attention_mask=mask, position_ids=positions,
        cache_position=positions[0], position_embeddings=rotary(hidden, positions), use_cache=False)
    output = output[0] if isinstance(output, tuple) else output
    return _record(case, norm(output[:, -1]), E, label_ids)


def boundary_control(actual, expected):
    if (actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16 or actual.shape != expected.shape
            or not torch.isfinite(actual).all() or not torch.isfinite(expected).all()):
        raise ValueError("Boundary control requires aligned finite BF16 residuals")
    maximum = float((actual.float()-expected.float()).abs().max())
    exact = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    if not exact:
        raise ValueError(f"Native singleton boundary differs from pinned capture: max_abs={maximum}")
    return dict(passed=True, bit_exact=True, max_abs=maximum, shape=list(actual.shape),
                scope="One existing development prompt; pre-layer residual, not final task accuracy")


def validate_records(records, cases):
    if len(records) != len(cases) or not cases or len({c["identity"] for c in cases}) != len(cases):
        raise ValueError("Complete distinct paired cases required")
    for row, case in zip(records, cases):
        logits = row.get("logits", [])
        if (any(row.get(k) != case[k] for k in ("identity", "family", "answer"))
                or len(logits) != len(case["choices"]) or not all(math.isfinite(x) for x in logits)
                or type(row.get("prediction")) is not int or
                row["prediction"] != max(range(len(logits)), key=logits.__getitem__)):
            raise ValueError("Unpaired or invalid choice records")


def paired_summary(native, compact, cases):
    validate_records(native, cases); validate_records(compact, cases)
    quality = paired_quality(native, compact)
    return dict(quality=quality, descriptive_screen_passed=quality["overall"]["accuracy_drop"] <= .05,
        maximum_accuracy_drop=.05, maximum_net_lost_correct=math.floor(.05*len(cases)),
        confirmed=False, full36=False, speed_measured=False,
        scope="Exploratory paired multiple-choice sanity only; not a population 5pp bound, final confirmation, or generation-quality test")


def _source_files():
    return source.inherited.local_import_closure([Path(__file__),
        base.ROOT/"tests/structure_net/test_jspace_3b_sanity_evaluate.py", PROTOCOL])


def prepare(root, source_root, data_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Preparation requires CUDA hidden")
    root, source_root, data_root = map(lambda x: Path(x).resolve(), (root, source_root, data_root))
    if root.exists(): raise FileExistsError(root)
    parent, stage = source.verify_stage(source_root, CUT)
    directory = source.stage_dir(source_root, CUT)
    verified = source.checked(directory, "verify", stage)
    accepted = verified["acceptance"]
    checkpoint = directory/"verify/accepted.pt"
    if (not accepted.get("accepted") or not accepted.get("strict_state_load") or
            not accepted.get("fixed_state_unchanged") or not accepted.get("exact_packed_replay") or
            accepted["candidate"]["checkpoint_sha256"] != CHECKPOINT_SHA or
            accepted["metrics"]["correct"] != 50 or base.sha(checkpoint) != CHECKPOINT_SHA):
        raise ValueError("Expected the freshly verified, exact accepted fourteen-block checkpoint")
    data_source.verify(data_root)
    dp = read(data_root/"plan.json")
    if dp["model"] != parent["model"] or dp["label_ids"] != parent["label_ids"]:
        raise ValueError("Sanity tokenizer/model/answer-token identity differs from candidate")
    cases = read(data_root/"tasks.json")["splits"]["sanity"]
    if len(cases) != 128 or Counter(c["family"] for c in cases) != dict.fromkeys(("arc","boolq","hellaswag","winogrande"),32):
        raise ValueError("Exactly 128 balanced sanity cases required")
    items = read(parent["items_path"])["items"]
    control = next(c for c in items if c["split"] == "development")
    boundary = Path(parent["native_cache"])/"boundary_22.npy"
    inputs = {}
    for path in (source_root/"plan.json", directory/"plan.json", parent["items_path"],
                 directory/"verify/manifest.json", directory/"verify/result.json",
                 directory/"verify/accepted_development.json", data_root/"plan.json", data_root/"manifest.json",
                 data_root/"tasks.json", data_root/"exclusions.json"):
        old.pin(inputs, path)
    old.pin(inputs, checkpoint, CHECKPOINT_SHA)
    old.pin(inputs, boundary, parent["input_sha256"][str(boundary)])
    for name, sha in parent["model"]["files"].items(): old.pin(inputs, Path(parent["model"]["path"])/name, sha)
    for fit in stage["fits"].values():
        for name in ("fit.pt", "relation.pt"): old.pin(inputs, Path(fit)/name)
    offset = 0; packed_items = []
    for case in cases:
        packed_items.append(dict(case, offset=offset, stop=offset+len(case["input_ids"])))
        offset += len(case["input_ids"])
    plan = dict(schema=SCHEMA, policy=POLICY, source_root=str(source_root), source_fingerprint=parent["fingerprint"],
        source_stage_fingerprint=stage["fingerprint"], data_root=str(data_root), data_fingerprint=dp["fingerprint"],
        model=parent["model"], fits=stage["fits"], cut=CUT, checkpoint=str(checkpoint), checkpoint_sha256=CHECKPOINT_SHA,
        label_ids=dp["label_ids"], native_boundary=str(boundary), development_control=control,
        cases_digest=base.digest(packed_items), total_tokens=offset, hidden_size=2048,
        allocation_mib=11264, allocator_mib=11008, confirmed=False,
        input_sha256=inputs, input_identity={p:old.file_identity(p) for p in inputs},
        source_sha256={str(p):base.sha(p) for p in sorted(_source_files())})
    plan["fingerprint"] = base.digest(plan)
    root.mkdir(parents=True)
    for name, digest in plan["source_sha256"].items():
        target = root/"producer"/Path(name).relative_to(base.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(name, target)
        if base.sha(target) != digest: raise ValueError("Producer archive changed during preparation")
    base.write(root/"items.json", dict(items=packed_items))
    base.write(root/"plan.json", plan)
    return verify(root)


def verify(root):
    root = Path(root); plan = read(root/"plan.json")
    if (plan.get("schema") != SCHEMA or plan.get("policy") != POLICY or plan.get("cut") != CUT or
            plan.get("checkpoint_sha256") != CHECKPOINT_SHA or plan.get("confirmed") is not False or
            plan.get("allocation_mib") != 11264 or plan.get("allocator_mib") != 11008 or
            plan["fingerprint"] != base.digest({k:v for k,v in plan.items() if k != "fingerprint"})):
        raise ValueError("Changed sanity plan/policy")
    for path, expected in plan["input_identity"].items():
        if old.file_identity(path) != expected: raise ValueError("Pinned sanity input changed: "+path)
    for path, expected in plan["source_sha256"].items():
        if base.sha(path) != expected or base.sha(root/"producer"/Path(path).relative_to(base.ROOT)) != expected:
            raise ValueError("Changed live or archived sanity producer")
    if base.sha(plan["checkpoint"]) != CHECKPOINT_SHA:
        raise ValueError("Accepted checkpoint bytes changed")
    cases = read(root/"items.json")["items"]
    if base.digest(cases) != plan["cases_digest"]: raise ValueError("Sanity case order/tokens changed")
    return plan


def checked(root, phase, plan=None):
    root = Path(root); plan = verify(root) if plan is None else plan
    if phase not in ("native", "compact"): raise ValueError("Unknown sanity phase")
    output = root/phase; manifest = read(output/"manifest.json")
    if (manifest.get("status") != "completed" or manifest.get("phase") != phase or
            manifest.get("fingerprint") != plan["fingerprint"]):
        raise ValueError("Incomplete or mismatched sanity phase")
    allocation, allocator = manifest.get("allocation", {}), manifest.get("allocator", {})
    uuid = allocator.get("physical_uuid", "")
    if (not uuid.startswith("GPU-0370fdb0") or allocation != {uuid:11264} or
            allocator.get("allocator_limit_bytes") != 11008*1024**2 or
            allocator.get("lease_bytes") != 11264*1024**2 or
            not 0 < manifest.get("peak_reserved_bytes", 0) <= allocator["allocator_limit_bytes"]):
        raise ValueError("Sanity phase lacks the matching bounded RTX3060 reservation")
    for name, expected in manifest["artifacts"].items():
        path = output/name
        if not path.resolve().is_relative_to(output.resolve()) or base.sha(path) != expected:
            raise ValueError("Sanity phase artifact changed")
    result = read(output/"result.json")
    validate_records(result["records"], read(root/"items.json")["items"])
    if (manifest["artifacts"].get("result.json") != base.sha(output/"result.json") or
            result.get("status") != "completed" or result.get("phase") != phase or
            result.get("fingerprint") != plan["fingerprint"] or result.get("count") != len(result["records"]) or
            result.get("correct") != sum(r["prediction"]==r["answer"] for r in result["records"])):
        raise ValueError("Sanity result identity changed")
    if phase == "native" and (not result["boundary_control"].get("bit_exact") or
            not result["boundary_control"].get("passed") or result["boundary_control"].get("max_abs") != 0 or
            "boundary_22.npy" not in manifest["artifacts"]):
        raise ValueError("Native boundary control or bank absent")
    if phase == "compact":
        checked(root, "native", plan)
        inherited = result.get("inheritance", {})
        if (not result.get("strict_state_load") or result.get("checkpoint_sha256") != CHECKPOINT_SHA or
                result.get("boundary_sha256") != base.sha(root/"native/boundary_22.npy") or
                not inherited.get("passed") or not inherited.get("norm_exact") or
                inherited.get("unchanged_layers") != list(range(CUT,36))):
            raise ValueError("Compact phase checkpoint, norm, or native prefix provenance changed")
    return result


def run(root, phase):
    root = Path(root).resolve(); plan = verify(root)
    if phase not in ("native", "compact"): raise ValueError("Unknown GPU phase")
    if phase == "compact": checked(root, "native", plan)
    output = root/phase
    if output.exists(): raise FileExistsError(output)
    lease = settings = None; started = time.monotonic(); cuda_started = False
    try:
        lease = wait_ready([0])
        settings = source.inherited.cap_allocator(lease, plan)
        torch.cuda.init(); cuda_started = True
        torch.cuda.reset_peak_memory_stats(); torch.set_num_threads(4)
        def guard():
            current = wait_ready([0])
            if current["id"] != lease["id"] or current["allocation"] != lease["allocation"]:
                raise RuntimeError("Reservation changed")
            if torch.cuda.max_memory_reserved() > settings["allocator_limit_bytes"]:
                raise MemoryError("Reservation allocator limit exceeded")
        output.mkdir()
        cases = read(root/"items.json")["items"]; records = []
        with execution.arithmetic_policy():
            if phase == "native":
                model = execution.load_native_model(plan["model"]["path"], device="cuda:0", attention="eager", guard=guard)
                _, actual = native_case(model, plan["development_control"], plan["label_ids"], CUT, guard)
                c = plan["development_control"]
                original = np.load(plan["native_boundary"], mmap_mode="r")
                if original.dtype != np.int16: raise ValueError("Original native cache must store lossless BF16 bits")
                expected = torch.from_numpy(np.array(original[c["offset"]:c["stop"]], copy=True)).view(torch.bfloat16)[None]
                control = dict(boundary_control(actual, expected), identity=c["identity"])
                del original, actual, expected
                bank = np.lib.format.open_memmap(output/"boundary_22.npy", mode="w+", dtype=np.int16,
                    shape=(plan["total_tokens"], plan["hidden_size"]))
                for index, case in enumerate(cases):
                    record, value = native_case(model, case, plan["label_ids"], CUT, guard)
                    bank[case["offset"]:case["stop"]] = value[0].contiguous().view(torch.int16).numpy()
                    records.append(record)
                    base.event(root/"progress.jsonl", "score", phase=phase, completed=index+1, total=len(cases))
                bank.flush(); del bank, model
                detail = dict(boundary_control=control, prefix="Raw tokens, native layers0..21; no padding, singleton eager")
            else:
                stack, norm, E, rotary, inheritance = builder.build_stack(plan["model"]["path"],
                    {int(k):v for k,v in plan["fits"].items()}, CUT, plan["checkpoint"], device="cpu", trainable=False,
                    checkpoint_blocks=False)
                packed = builder.packed_clone(stack, device="cpu"); del stack
                packed, norm, E, rotary = packed.to("cuda:0"), norm.to("cuda:0"), E.to("cuda:0"), rotary.to("cuda:0")
                bank = np.load(root/"native/boundary_22.npy", mmap_mode="r")
                if bank.dtype != np.int16 or list(bank.shape) != [plan["total_tokens"], plan["hidden_size"]]:
                    raise ValueError("Changed native boundary shape or encoding")
                for index, case in enumerate(cases):
                    value = torch.from_numpy(np.array(bank[case["offset"]:case["stop"]], copy=True)).view(torch.bfloat16)[None]
                    records.append(compact_case(packed, norm, E, rotary, value, case, plan["label_ids"], guard))
                    base.event(root/"progress.jsonl", "score", phase=phase, completed=index+1, total=len(cases))
                del bank
                detail = dict(strict_state_load=True, inheritance=inheritance, checkpoint_sha256=plan["checkpoint_sha256"],
                    boundary_sha256=base.sha(root/"native/boundary_22.npy"),
                    prefix="Exact cached native prefix computed from the same raw prompts in native phase")
        validate_records(records, cases); guard()
        result = dict(status="completed", phase=phase, fingerprint=plan["fingerprint"], records=records,
            count=len(records), correct=sum(r["prediction"]==r["answer"] for r in records), **detail)
        base.write(output/"result.json", result)
        artifacts = {p.name:base.sha(p) for p in output.iterdir() if p.is_file()}
        base.write(output/"manifest.json", dict(status="completed", phase=phase, fingerprint=plan["fingerprint"],
            artifacts=artifacts, lease_id=lease["id"], allocation=lease["allocation"], allocator=settings,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), elapsed_seconds=time.monotonic()-started))
        return result
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        base.write(output/"failure.json", dict(status="failed", phase=phase, fingerprint=plan["fingerprint"],
            error=f"{type(error).__name__}: {error}", lease=lease, allocator=settings,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None))
        raise
    finally:
        gc.collect()


def finalize(root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("Finalization requires CUDA hidden")
    root = Path(root); plan = verify(root)
    native, compact = (checked(root, phase, plan) for phase in ("native", "compact"))
    result = dict(status="completed", fingerprint=plan["fingerprint"],
        checkpoint_sha256=plan["checkpoint_sha256"], compressed_layers=14,
        **paired_summary(native["records"], compact["records"], read(root/"items.json")["items"]),
        phase_manifest_sha256={phase:base.sha(root/phase/"manifest.json") for phase in ("native","compact")})
    base.write(root/"paired_result.json", result)
    base.write(root/"manifest.json", dict(status="completed", fingerprint=plan["fingerprint"],
        artifacts={"paired_result.json":base.sha(root/"paired_result.json")}, confirmed=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare","native","compact","finalize"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--source"); parser.add_argument("--data-root")
    args = parser.parse_args()
    if args.phase == "prepare":
        if not args.source or not args.data_root: parser.error("prepare requires --source and --data-root")
        result = prepare(args.root, args.source, args.data_root)
    else: result = finalize(args.root) if args.phase == "finalize" else run(args.root, args.phase)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
