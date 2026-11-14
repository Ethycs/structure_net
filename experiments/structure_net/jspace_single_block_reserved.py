"""Reservation-bounded allocator policy around the unchanged single-block run.

Preparation delegates the frozen scientific plan, then records an independent
runtime policy. Each fresh process obtains its lease before querying CUDA and
caps the native PyTorch caching allocator at that GPU's lease minus256MiB.
The cap can prevent cache growth beyond the lease; it cannot make an oversized
live working set fit. Training, initialization and checkpoint selection remain
the original12-epoch experiment, with a separate discarded systems pilot.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_single_block_distill as single

SCHEMA="jspace.single-block-reserved-runtime.v1"
HEADROOM_MIB=256
MIB=1024**2


def prepare(root,source,basis_root,native_baseline):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Hide CUDA for CPU preparation")
    root=Path(root).resolve()
    result=single.prepare(root,source,basis_root,native_baseline)
    files=[Path(__file__),single.recipe.ROOT/"tests/structure_net/test_jspace_single_block_reserved.py"]
    policy=dict(schema=SCHEMA,base_schema=single.SCHEMA,base_plan_fingerprint=result["fingerprint"],
        base_plan_sha256=single.recipe.sha(root/"plan.json"),headroom_mib=HEADROOM_MIB,
        allocator_backend="native",device=0,requires_fresh_cuda_process=True,
        budget_source="MiB allocation for the first and only CUDA_VISIBLE_DEVICES GPU UUID",
        formula="fraction=(lease_bytes-256*1024**2)/device_total_memory",
        scientific_config="Delegated unchanged single-block12-epoch initialization, objective and selection; clean restart",
        source_sha256={str(p):single.recipe.sha(p) for p in files})
    policy["fingerprint"]=single.recipe.digest(policy)
    for path in files:
        target=root/"runtime_producer"/path.relative_to(single.recipe.ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    single.recipe.write(root/"runtime_policy.json",policy)
    return dict(**result,runtime_policy_fingerprint=policy["fingerprint"])


def verify(root):
    root=Path(root).resolve();plan=single.verify(root)
    policy=single.recipe.read(root/"runtime_policy.json")
    if (policy.get("schema")!=SCHEMA or policy.get("fingerprint")!=single.recipe.digest({k:v for k,v in policy.items() if k!="fingerprint"})
            or policy.get("base_schema")!=single.SCHEMA or policy.get("base_plan_fingerprint")!=plan["fingerprint"]
            or policy.get("base_plan_sha256")!=single.recipe.sha(root/"plan.json")
            or policy.get("headroom_mib")!=HEADROOM_MIB or policy.get("allocator_backend")!="native"
            or policy.get("device")!=0 or policy.get("requires_fresh_cuda_process") is not True):
        raise ValueError("Changed or mismatched frozen allocator policy")
    expected={str(Path(__file__)),str(single.recipe.ROOT/"tests/structure_net/test_jspace_single_block_reserved.py")}
    if set(policy["source_sha256"])!=expected:raise ValueError("Changed runtime producer closure")
    for path,digest in policy["source_sha256"].items():
        snapshot=root/"runtime_producer"/Path(path).relative_to(single.recipe.ROOT)
        if single.recipe.sha(path)!=digest or single.recipe.sha(snapshot)!=digest:
            raise ValueError("Changed frozen runtime producer")
    return plan,policy


def allocator_settings(lease,policy,*,cuda=None,visible=None):
    """Configure an already-authorized lease; injectable CUDA enables CPU tests."""
    cuda=torch.cuda if cuda is None else cuda
    visible=os.environ.get("CUDA_VISIBLE_DEVICES","") if visible is None else visible
    ids=visible.split(",")
    allocation=lease["allocation"]
    if len(ids)!=1 or not ids[0] or ids[0] not in allocation:
        raise ValueError("Single visible leased GPU UUID required")
    mib=allocation[ids[0]]
    if type(mib) is not int or mib<=policy["headroom_mib"]:
        raise ValueError("GPU allocation must exceed reserved allocator headroom")
    if cuda.is_initialized():raise RuntimeError("Allocator-capped stage requires a fresh CUDA process")
    # The caller has acquired the reservation BEFORE this initializes CUDA.
    total=int(cuda.get_device_properties(0).total_memory)
    allowed=mib*MIB-policy["headroom_mib"]*MIB
    if not 0<allowed<=total:raise ValueError("Allocator budget exceeds the physical device")
    backend=cuda.memory.get_allocator_backend()
    if backend!=policy["allocator_backend"]:
        raise RuntimeError("Frozen allocator policy requires the native PyTorch backend")
    fraction=float(allowed/total)
    cuda.set_per_process_memory_fraction(fraction,device=0)
    return dict(lease_id=lease["id"],allocation=dict(allocation),visible_gpu=ids[0],device=0,
        lease_bytes=mib*MIB,headroom_bytes=policy["headroom_mib"]*MIB,allocator_limit_bytes=allowed,
        device_total_memory=total,memory_fraction=fraction,allocator_backend=backend,
        torch_version=str(torch.__version__),allocator_environment=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))


def reservation():
    from .jspace_full_lease import wait_ready
    return wait_ready([0])


def verify_runtime_receipt(root,stage,plan,policy):
    root=Path(root);receipt=single.recipe.read(root/f"runtime_{stage}.json")
    if (receipt.get("status")!="completed" or receipt.get("stage")!=stage
            or receipt.get("base_plan_fingerprint")!=plan["fingerprint"]
            or receipt.get("runtime_policy_fingerprint")!=policy["fingerprint"]
            or receipt.get("base_stage_manifest_sha256")!=single.recipe.sha(root/stage/"manifest.json")):
        raise ValueError("A matching completed allocator-capped stage receipt is required")
    return receipt


def run(root,stage):
    if stage not in ("pilot","train"):raise ValueError("Only pilot/train execute GPU work")
    root=Path(root).resolve();plan,policy=verify(root)
    if (root/stage).exists() or (root/f"runtime_{stage}.json").exists():raise FileExistsError("Stage already attempted")
    if stage=="train":
        single.joint.verify_pilot(root,plan["fingerprint"])
        verify_runtime_receipt(root,"pilot",plan,policy)
    started=time.monotonic();lease=None;settings=None
    try:
        lease=reservation()  # No CUDA query precedes authorization.
        settings=allocator_settings(lease,policy)
        single.recipe.write(root/f"runtime_{stage}_started.json",dict(stage=stage,
            base_plan_fingerprint=plan["fingerprint"],runtime_policy_fingerprint=policy["fingerprint"],settings=settings))
        result=single.run(root,stage)
        manifest=single.recipe.read(root/stage/"manifest.json")
        if manifest.get("status")!="completed" or manifest.get("fingerprint")!=plan["fingerprint"]:
            raise RuntimeError("Unchanged trainer did not complete the matching stage")
        if manifest["peak_reserved_bytes"]>settings["allocator_limit_bytes"]:
            raise MemoryError("Observed allocator peak exceeded the configured runtime cap")
        receipt=dict(status="completed",stage=stage,base_plan_fingerprint=plan["fingerprint"],
            runtime_policy_fingerprint=policy["fingerprint"],settings=settings,
            base_stage_manifest_sha256=single.recipe.sha(root/stage/"manifest.json"),
            peak_reserved_bytes=manifest["peak_reserved_bytes"],elapsed_seconds=time.monotonic()-started)
        single.recipe.write(root/f"runtime_{stage}.json",receipt)
        return dict(**result,runtime=receipt)
    except Exception as error:
        single.recipe.write(root/f"runtime_{stage}.json",dict(status="failed",stage=stage,
            base_plan_fingerprint=plan["fingerprint"],runtime_policy_fingerprint=policy["fingerprint"],
            lease_id=None if lease is None else lease["id"],settings=settings,
            error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","pilot","train"));parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--source",type=Path);parser.add_argument("--basis-root",type=Path);parser.add_argument("--native-baseline",type=Path)
    args=parser.parse_args()
    if args.stage=="prepare":
        if any(v is None for v in (args.source,args.basis_root,args.native_baseline)):parser.error("prepare requires source, basis-root, native-baseline")
        value=prepare(args.root,args.source,args.basis_root,args.native_baseline)
    else:value=run(args.root,args.stage)
    print(json.dumps(value,allow_nan=False),flush=True)


if __name__=="__main__":main()
