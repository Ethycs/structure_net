"""Append-only actual-model full-vocabulary response-path parity probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch
from transformers import AutoTokenizer

from . import jspace_3b_response_model as construction
from .jspace_3b_response_train import ResponseModel
from .jspace_full_lease import wait_ready

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "jspace.complete-response-systems.v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(path):
    p = Path(path).resolve(); s = p.stat()
    return dict(path=str(p), size=s.st_size, inode=s.st_ino, mtime_ns=s.st_mtime_ns,
                ctime_ns=s.st_ctime_ns, device=s.st_dev)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")


def prepare(root, arm, source_plan, checkpoint, *, allocation_mib, limit_mib,
            expected_checkpoint_sha256=None):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Preparation must run with CUDA hidden")
    root, source_plan, checkpoint = map(lambda x: Path(x).resolve(), (root, source_plan, checkpoint))
    if root.exists(): raise FileExistsError(root)
    source = json.loads(source_plan.read_text())
    construction.progressive.source.check_plan(source)
    inputs = {str(source_plan): sha(source_plan), str(checkpoint): sha(checkpoint)}
    if expected_checkpoint_sha256 and inputs[str(checkpoint)] != expected_checkpoint_sha256:
        raise ValueError("Scientific initializer differs from the requested checkpoint")
    model_path = Path(source["model"]["path"])
    for name, digest in source["model"]["files"].items():
        path = model_path/name
        actual = sha(path)
        if actual != digest: raise ValueError("Pinned native model changed: "+str(path))
        inputs[str(path.resolve())] = actual
    fits = dict(source["fit_map"] if arm == "all36" else source["parent_fits"])
    if arm == "progressive": fits[str(source["cut"])] = str(Path(source["fit_root"])/f'block_{source["cut"]:02d}')
    for directory in fits.values():
        for name in ("fit.pt", "relation.pt"):
            path = Path(directory)/name
            inputs[str(path.resolve())] = sha(path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    messages = [dict(role="user", content="Summarize this in one sentence, preserving the numbers: "
        "The trial enrolled 120 adults. Thirty withdrew before week 8, leaving 90 participants. "
        "The final review is scheduled for 12 December.")]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    files = construction.progressive.source.inherited.local_import_closure([
        Path(__file__), ROOT/"src/neural_architecture_lab/distillation_training.py",
        ROOT/"src/neural_architecture_lab/distillation_data.py",
        ROOT/"docs/05 - Standards/NAL-STD-DISTILLATION-v1.md"])
    p = dict(schema=SCHEMA, arm=arm, source_plan=str(source_plan), checkpoint=str(checkpoint),
        checkpoint_sha256=inputs[str(checkpoint)], model_path=str(model_path), messages=messages,
        prompt_ids=ids, steps=3, allocation_mib=allocation_mib, limit_mib=limit_mib,
        input_sha256=inputs, input_identity={path:identity(path) for path in inputs},
        source_sha256={str(f):sha(f) for f in sorted(files)}, atol=.03125, rtol=1/128,
        scope="Systems parity only; no optimizer, training, task accuracy or recovery claim")
    p["fingerprint"] = construction.progressive.base.digest(p)
    root.mkdir(parents=True)
    write(root/"plan.json", p)
    for name, digest in p["source_sha256"].items():
        out = root/"producer"/Path(name).relative_to(ROOT)
        out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(name, out)
        if sha(out) != digest: raise ValueError("Probe producer archive mismatch")
    return dict(status="prepared", fingerprint=p["fingerprint"], checkpoint_sha256=p["checkpoint_sha256"])


def prepare_unpacked_from_prior(root, prior, *, allocation_mib, limit_mib):
    """Reuse authenticated unchanged large inputs; archive a new execution control."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Preparation must run with CUDA hidden")
    root, prior = Path(root).resolve(), Path(prior).resolve()
    if root.exists(): raise FileExistsError(root)
    p = json.loads((prior/"plan.json").read_text())
    construction.progressive.source.check_plan(p)
    old_result = json.loads((prior/"result.json").read_text())
    if old_result.get("status") != "completed" or old_result.get("fingerprint") != p["fingerprint"]:
        raise ValueError("Completed actual prior probe required")
    for name, old in p["input_identity"].items():
        if identity(name) != old: raise ValueError("Previously hashed input changed: "+name)
    for name, digest in p["source_sha256"].items():
        if sha(prior/"producer"/Path(name).relative_to(ROOT)) != digest:
            raise ValueError("Prior archived producer changed")
    old_fingerprint = p.pop("fingerprint")
    p.update(reference_mode="unpacked", prior_probe=str(prior), prior_fingerprint=old_fingerprint,
             allocation_mib=allocation_mib, limit_mib=limit_mib)
    for path in (prior/"plan.json", prior/"result.json"):
        p["input_sha256"][str(path)] = sha(path)
        p["input_identity"][str(path)] = identity(path)
    files = construction.progressive.source.inherited.local_import_closure([
        Path(__file__), ROOT/"src/neural_architecture_lab/distillation_training.py",
        ROOT/"src/neural_architecture_lab/distillation_data.py",
        ROOT/"docs/05 - Standards/NAL-STD-DISTILLATION-v1.md"])
    p["source_sha256"] = {str(f):sha(f) for f in sorted(files)}
    p["fingerprint"] = construction.progressive.base.digest(p)
    root.mkdir(parents=True)
    write(root/"plan.json", p)
    for name, digest in p["source_sha256"].items():
        out = root/"producer"/Path(name).relative_to(ROOT)
        out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(name, out)
        if sha(out) != digest: raise ValueError("Probe producer archive mismatch")
    return dict(status="prepared", fingerprint=p["fingerprint"], checkpoint_sha256=p["checkpoint_sha256"],
                reference_mode="unpacked", large_inputs_reused_with_unchanged_file_identity=True)


def run(root):
    root = Path(root).resolve(); p = json.loads((root/"plan.json").read_text())
    construction.progressive.source.check_plan(p)
    for name, old in p["input_identity"].items():
        if identity(name) != old: raise ValueError("Frozen probe input changed: "+name)
    for name, digest in p["source_sha256"].items():
        if sha(name) != digest or sha(root/"producer"/Path(name).relative_to(ROOT)) != digest:
            raise ValueError("Frozen probe source changed: "+name)
    if (root/"result.json").exists() or (root/"failure.json").exists():
        raise FileExistsError("Probe attempt already has a terminal receipt")
    started = time.monotonic(); lease = None
    try:
        lease = wait_ready([0]); torch.cuda.set_device(0); torch.set_num_threads(2)
        if len(lease["allocation"]) != 1 or next(iter(lease["allocation"].values())) != p["allocation_mib"]:
            raise ValueError("Probe requires its declared single-card reservation")
        props = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(p["limit_mib"]*1024**2/props.total_memory, 0)
        torch.cuda.reset_peak_memory_stats()
        def guard():
            active = wait_ready([0])
            if active["id"] != lease["id"] or active["allocation"] != lease["allocation"]:
                raise RuntimeError("Probe reservation changed")
        print(json.dumps(dict(phase="building", arm=p["arm"], lease_id=lease["id"], allocation=lease["allocation"])), flush=True)
        stack, norm, embedding, rotary, proof = construction.build_student(
            p["arm"], p["source_plan"], p["checkpoint"], torch.device("cuda:0"), guard)
        if proof["checkpoint_sha256"] != p["checkpoint_sha256"]:
            raise ValueError("Loaded initializer differs from frozen probe")
        model = ResponseModel(stack, norm, embedding, rotary, max_length=416, expected_layers=36, guard=guard)
        print(json.dumps(dict(phase="parity", arm=p["arm"], cut=model.cut, lease_id=lease["id"])), flush=True)
        parity = construction.check_full_vocabulary_parity(model, p["model_path"], p["prompt_ids"],
            guard=guard, steps=p["steps"], atol=p["atol"], rtol=p["rtol"],
            reference_mode=p.get("reference_mode", "packed"))
        result = dict(status="completed", passed=parity["passed"], fingerprint=p["fingerprint"],
            construction=proof, parity=parity, lease_id=lease["id"], allocation=lease["allocation"],
            gpu=props.name, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            seconds=time.monotonic()-started, scope=p["scope"], confirmed=False)
        write(root/"result.json", result)
        return result
    except BaseException as error:
        write(root/"failure.json", dict(status="execution_failure", error=repr(error), lease=lease,
                                        seconds=time.monotonic()-started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run")); parser.add_argument("--root", required=True)
    parser.add_argument("--arm", choices=("progressive", "all36")); parser.add_argument("--source-plan")
    parser.add_argument("--checkpoint"); parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--prior", help="Prior completed packed probe to qualify the separate unpacked path")
    parser.add_argument("--allocation-mib", type=int); parser.add_argument("--limit-mib", type=int)
    args = parser.parse_args()
    if args.phase == "run": result = run(args.root)
    elif args.prior:
        result = prepare_unpacked_from_prior(args.root, args.prior,
            allocation_mib=args.allocation_mib, limit_mib=args.limit_mib)
    else:
        result = prepare(args.root, args.arm, args.source_plan,
            args.checkpoint, allocation_mib=args.allocation_mib, limit_mib=args.limit_mib,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__": main()
