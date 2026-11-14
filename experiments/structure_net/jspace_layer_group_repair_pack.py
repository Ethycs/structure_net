"""Verify packing of the full compact assembly plus its selected three-block delta.

Original native embedding/norm/readout remain fixed. This checks arithmetic and
actual tensor storage on64 development cases, not generation speed or generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_layer_group_repair as repair
from . import jspace_recovered_joint_pack as previous

assembly,recipe=repair.base.assembly,repair.recipe
SCHEMA="jspace.layer-group-repair-pack.v1"
SCOPE="Full compact assembly packing parity on original64 development cases; no speed or confirmation claim"
compare,cap_allocator=previous.compare,previous.cap_allocator


def authenticated(root):
    root=Path(root).resolve();plan=repair.verify(root)
    repair.checked(root,"pilot",plan);result=repair.checked(root,"train",plan)
    fresh=repair.checked(root,"verification",plan);receipt=recipe.read(root/"verification/manifest.json")
    if (receipt.get("passed") is not True or fresh.get("passed") is not True
            or fresh.get("fingerprint")!=plan["fingerprint"] or fresh.get("selected_layers")!=plan["selected_layers"]
            or fresh.get("selected_manifest_sha256")!=result["selected_manifest_sha256"]
            or fresh.get("selected_epoch")!=result["selected_epoch"]
            or fresh.get("training_quality_pass")!=result["quality_pass"]
            or not all(fresh.get(k) is True for k in ("strict_state_load","frozen_state_unchanged","predictions_exact","logits_pass","metrics_pass"))):
        raise ValueError("Completed matching fresh subgroup verification required before packing")
    selected=recipe.read(root/"train/selected/manifest.json")
    if (selected.get("schema")!=repair.SCHEMA or selected.get("selected_layers")!=plan["selected_layers"]
            or selected.get("parent_assembly_fingerprint")!=plan["assembly_fingerprint"]
            or selected.get("search_fingerprint")!=plan["selection"]["search_fingerprint"]
            or set(selected["artifacts"])!={f"block_{i:02d}.pt" for i in plan["selected_layers"]}
            or selected["metadata"].get("epoch")!=result["selected_epoch"]
            or selected["metadata"].get("development")!=result["selected"]):
        raise ValueError("Delta3 architecture, fixed parent or selected metadata changed")
    parent=assembly.verify(plan["assembly_root"])
    if parent["fingerprint"]!=plan["assembly_fingerprint"] or parent["arms"]["all_compact"]!=list(range(36)):
        raise ValueError("Authenticated full36 compact parent required; no native-restoration mask")
    return plan,result,selected,parent


def prepare(root,training_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Hide CUDA for CPU preparation")
    root,training_root=Path(root).resolve(),Path(training_root).resolve()
    if root.exists():raise FileExistsError(root)
    parent,result,selected,assembled=authenticated(training_root)
    cases=[c for c in recipe.read(training_root/"items.json")["items"] if c["split"]=="development"]
    if len(cases)!=64 or len({c["identity"] for c in cases})!=64:raise ValueError("Original64 development prompts required")
    selected_report=training_root/"train"/f"epoch_{result['selected_epoch']:02d}_development.json"
    fresh_report=training_root/"verification/result.json"
    references=[recipe.read(p) for p in (selected_report,fresh_report)]
    if not compare(references[0]["records"],*references,cases)["passed"]:raise ValueError("Selected and fresh references disagree")
    inputs=dict(parent["input_sha256"])
    for name in ("plan.json","items.json","pilot/result.json","pilot/manifest.json","train/result.json","train/manifest.json",
            "train/frozen_state.json","train/selected/manifest.json","verification/result.json","verification/manifest.json"):
        assembly._pin(inputs,training_root/name)
    for row in result["history"]:
        assembly._pin(inputs,training_root/"train"/f"epoch_{row['epoch']:02d}_development.json",row["development_sha256"])
    for name,digest in selected["artifacts"].items():assembly._pin(inputs,training_root/"train/selected"/name,digest)
    sources=dict(parent["source_sha256"])
    for path in (Path(__file__).resolve(),Path(previous.__file__).resolve(),
            recipe.ROOT/"tests/structure_net/test_jspace_layer_group_repair_pack.py"):
        assembly._merge_hashes(sources,{str(path):recipe.sha(path)})
    plan=dict(schema=SCHEMA,training_root=str(training_root),training_fingerprint=parent["fingerprint"],
        assembly_root=parent["assembly_root"],assembly_fingerprint=parent["assembly_fingerprint"],selection=parent["selection"],
        model=parent["model"],blocks=assembled["blocks"],selected_layers=parent["selected_layers"],
        compact_layers=list(range(36)),native_restored_layers=[],selected_epoch=result["selected_epoch"],
        selected_directory=str(training_root/"train/selected"),checkpoint_artifacts=selected["artifacts"],
        selected_manifest_sha256=result["selected_manifest_sha256"],selected_reference=str(selected_report),
        fresh_reference=str(fresh_report),cases=cases,label_ids=parent["label_ids"],selected_correct=result["selected"]["correct"],
        training_quality_pass=result["quality_pass"],input_sha256=inputs,input_identity={p:recipe.file_identity(p) for p in inputs},
        source_sha256=sources,protocol=str(repair.PROTOCOL),
        precision="All36 compact blocks and geometry packedFP32; original native norm and tiedE unchanged",
        gpu_prefix="GPU-0370fdb0",allocation_mib=7168,microbatch=4,confirmed=False,scope=SCOPE)
    plan["fingerprint"]=recipe.digest(plan);root.mkdir(parents=True)
    for path,digest in sources.items():
        if recipe.sha(path)!=digest:raise ValueError("Changed packing producer")
        destination=root/"producer"/Path(path).relative_to(recipe.ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
    recipe.write(root/"plan.json",plan);return dict(status="prepared",fingerprint=plan["fingerprint"])


def verify(root):
    root=Path(root).resolve();plan=recipe.read(root/"plan.json")
    if (plan.get("schema")!=SCHEMA or plan["fingerprint"]!=recipe.digest({k:v for k,v in plan.items() if k!="fingerprint"})
            or plan.get("compact_layers")!=list(range(36)) or plan.get("native_restored_layers")!=[]):
        raise ValueError("Changed full-compact packing plan")
    parent,result,_,_=authenticated(plan["training_root"])
    if (parent["fingerprint"]!=plan["training_fingerprint"] or result["selected_manifest_sha256"]!=plan["selected_manifest_sha256"]
            or parent["selected_layers"]!=plan["selected_layers"]):raise ValueError("Changed selected training source")
    for path,digest in plan["source_sha256"].items():
        if recipe.sha(path)!=digest or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT))!=digest:
            raise ValueError("Changed packing producer")
    for path,identity in plan["input_identity"].items():
        if recipe.file_identity(path)!=identity:raise ValueError("Changed consumed packing input")
    return plan


def load_packed_block(plan,layer,*,device="cpu",loader=None):
    """Only the selected triple comes from delta; every other block stays compact."""
    if type(layer) is not int or not 0<=layer<36:raise ValueError("Original absolute layer0..35 required")
    if layer in plan["selected_layers"]:
        block=previous.load_packed_block(plan,layer,device=device,loader=loader)
    else:
        view=dict(model=plan["model"],blocks=plan["blocks"],arms={"all_compact":list(range(36))})
        block=assembly.load_block(view,"all_compact",layer,device=device,loader=loader)
    from .jspace_residual_carry import ResidualCarryBlock
    if not isinstance(block,ResidualCarryBlock):raise ValueError("Every deployment layer must be a compact residual-carry block")
    if any(p.requires_grad for p in block.parameters()):raise ValueError("Packed deployment weights must be frozen")
    if any(t.is_floating_point() and t.dtype!=torch.float32 for t in [*block.parameters(),*block.buffers()]):
        raise ValueError("FP32 packed weights and fixed geometry required")
    return block


def norm_fingerprint(norm):
    value=norm.weight.detach().cpu().contiguous()
    return dict(sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest(),dtype=str(value.dtype),
        shape=list(value.shape),epsilon=float(norm.variance_epsilon))


def run(root):
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_readout_norm
    root=Path(root).resolve();plan=verify(root);directory=root/"verification"
    directory.mkdir(exist_ok=False);lease=allocator=None;started=time.monotonic();cuda_started=False
    telemetry=dict(calls=0,wall_seconds=0.)
    try:
        lease=wait_ready([0]);allocator=cap_allocator(lease)
        torch.cuda.init();cuda_started=True;torch.cuda.reset_peak_memory_stats()
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        def guard():
            began=time.monotonic()
            try:
                current=wait_ready([0])
                if current["id"]!=lease["id"] or current["allocation"]!=lease["allocation"]:raise RuntimeError("Packing reservation changed")
                if torch.cuda.max_memory_reserved()>allocator["allocator_limit_bytes"]:raise MemoryError("Packing allocator cap exceeded")
            finally:
                telemetry["calls"]+=1;telemetry["wall_seconds"]+=time.monotonic()-began
        embedding,norm,rotary=load_readout_norm(plan["model"]["path"],device="cpu",dtype=torch.bfloat16)
        embedding=embedding.float();norm=norm.float().eval().requires_grad_(False);rotary=rotary.float().to("cuda:0")
        original_norm=norm_fingerprint(norm);embedding_version=embedding._version
        endpoints=assembly.tensor_storage(embedding,norm,rotary);inventory=[]
        def loader(layer):
            guard();block=load_packed_block(plan,layer,device="cuda:0")
            inventory.append(dict(layer=layer,compact=True,selected_delta=layer in plan["selected_layers"],
                fixed_geometry_verified=True,**assembly.tensor_storage(block)))
            return block
        value=assembly.single.stream_score(plan["cases"],input_embedding=embedding,block_loader=loader,
            final_readout=(norm,embedding),label_ids=plan["label_ids"],scratch_dir=directory/"scratch",
            rotary=rotary,layer_count=36,microbatch=plan["microbatch"],device="cuda:0",guard=guard,
            progress=lambda **kw:recipe.event(root,"packing_progress",detail=kw))
        parity=compare(value["records"],recipe.read(plan["selected_reference"]),recipe.read(plan["fresh_reference"]),plan["cases"])
        fixed_endpoints=norm_fingerprint(norm)==original_norm and embedding._version==embedding_version
        if [row["layer"] for row in inventory]!=list(range(36)):raise RuntimeError("Packing did not execute the full compact assembly")
        guard();value.update(status="completed",passed=parity["passed"] and fixed_endpoints,fingerprint=plan["fingerprint"],parity=parity,
            selected_epoch=plan["selected_epoch"],selected_manifest_sha256=plan["selected_manifest_sha256"],selected_layers=plan["selected_layers"],
            compact_layers=list(range(36)),native_restored_layers=[],strict_state_load=True,fixed_geometry_verified=True,
            native_norm_fingerprint=original_norm,native_norm_and_tied_embedding_unchanged=fixed_endpoints,
            accuracy=assembly.single.ranking_summary(value["records"]),training_quality_pass=plan["training_quality_pass"],
            precision="float32",confirmed=False,scope=SCOPE,throughput_measured=False,confirmation_used=False,
            tensor_state=dict(storage_bytes=endpoints["storage_bytes"]+sum(row["storage_bytes"] for row in inventory),
                endpoints=endpoints,blocks=inventory,
                accounting="TiedE once, original native norm/rotary once, all36 packed compact blocks including fixed geometry; no activations/KV cache"))
        recipe.write(directory/"result.json",value)
        if not value["passed"]:raise RuntimeError("Packed full assembly differs from selected/fresh logits or fixed endpoints")
        recipe.write(directory/"manifest.json",dict(status="completed",passed=True,fingerprint=plan["fingerprint"],
            result_sha256=recipe.sha(directory/"result.json"),lease_id=lease["id"],allocation=lease["allocation"],allocator=allocator,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),reservation_guard=telemetry,elapsed_seconds=time.monotonic()-started))
        return dict(status="completed",passed=True,accuracy=value["accuracy"],fingerprint=plan["fingerprint"])
    except BaseException as error:
        recipe.write(directory/"failure.json",dict(status="failed",fingerprint=plan["fingerprint"],error=repr(error),
            lease_id=None if lease is None else lease["id"],allocation=None if lease is None else lease["allocation"],allocator=allocator,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None,reservation_guard=telemetry));raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("prepare","run"))
    parser.add_argument("--root",required=True,type=Path);parser.add_argument("--training-root",type=Path);args=parser.parse_args()
    if args.stage=="prepare" and args.training_root is None:parser.error("prepare requires --training-root")
    print(json.dumps(prepare(args.root,args.training_root) if args.stage=="prepare" else run(args.root),allow_nan=False),flush=True)
