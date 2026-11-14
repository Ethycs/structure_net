"""Independent, matched-FP32 scoring of completed joint residual recovery.

Only a completed, development-selected checkpoint may unlock confirmation.
Each block is reconstructed and strictly reloaded independently. The scorer
stages choice-label rows, but the model retains its original full-vocabulary
tied E. Streaming wall time is an evaluation diagnostic, never a speed claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_joint_recovery as joint
from . import jspace_quadratic_recipe as recipe
from .jspace_quadratic_evaluate import paired_summary, stream_score

SCHEMA="jspace.joint-evaluation.v1"


def tensor_storage_bytes(*objects):
    """Count physical tensor storage once, including shared/tied views."""
    storages={}
    for value in objects:
        tensors=[value] if isinstance(value,torch.Tensor) else [*value.parameters(),*value.buffers()]
        for tensor in tensors:
            storage=tensor.untyped_storage()
            key=(str(tensor.device),storage.data_ptr(),storage.nbytes())
            storages[key]=storage.nbytes()
    return sum(storages.values())


def validate_confirmation(cases,excluded,*,expected_count=512):
    identities=[c["identity"] for c in cases]
    if len(cases)!=expected_count or len(set(identities))!=len(cases):
        raise ValueError("Confirmation population/identity uniqueness mismatch")
    if set(identities)&set(excluded):
        raise ValueError("Confirmation overlaps training, development or previously exposed tasks")
    return cases


def completed_training(root):
    root=Path(root).resolve();plan=joint.verify(root)
    joint.verify_pilot(root,plan["fingerprint"])
    manifest=recipe.read(root/"train/manifest.json")
    result=recipe.read(root/"train/result.json")
    if (manifest.get("status")!="completed" or manifest.get("systems_only") is not False
            or manifest.get("fingerprint")!=plan["fingerprint"]
            or manifest.get("result_sha256")!=recipe.sha(root/"train/result.json")
            or result.get("status")!="completed" or result.get("systems_only") is not False
            or result.get("strict_reload_exact") is not True):
        raise ValueError("Completed joint training with exact selected reload is required before evaluation")
    selected=Path(result["selected_directory"]).resolve()
    if selected.parent!=root/"train" or selected.name!=f"epoch_{result['selected_epoch']:02d}_selected":
        raise ValueError("Selected checkpoint must be the completed run's chosen epoch")
    checkpoint=recipe.read(selected/"manifest.json")
    if (checkpoint.get("schema")!=joint.SCHEMA or checkpoint.get("blocks")!=plan["source_config"]["layers"]
            or checkpoint["metadata"]["epoch"]!=result["selected_epoch"]
            or checkpoint["metadata"]["development"]!=result["selected"]):
        raise ValueError("Selected checkpoint metadata differs from development selection")
    expected={f"block_{i:02d}.pt" for i in range(checkpoint["blocks"])}|{"norm_gain.pt"}
    if set(checkpoint["artifacts"])!=expected:raise ValueError("Incomplete selected checkpoint")
    return plan,result,selected,checkpoint


def load_candidate_block(training_plan,selected,layer,device,*,block_factory=None):
    """Reload one BF16-weight/FP32-geometry block, then promote for scoring."""
    from .jspace_residual_carry import ResidualCarryBlock
    selected=Path(selected)
    manifest=recipe.read(selected/"manifest.json")
    path=selected/f"block_{layer:02d}.pt"
    if recipe.sha(path)!=manifest["artifacts"][path.name]:raise ValueError("Changed selected block shard")
    if block_factory is None:
        wrapped=recipe.build_block(training_plan["source"],training_plan["arm"],layer,device=device,selected=False)
        block=ResidualCarryBlock(wrapped)
    else:block=block_factory(layer,device)
    for parameter in block.parameters():parameter.data=parameter.data.to(torch.bfloat16)
    block.load_state_dict(recipe.load(path),strict=True)
    block.eval().requires_grad_(False)
    checkpoint_bytes=tensor_storage_bytes(block)
    block.float()
    return block,dict(checkpoint_tensor_bytes=checkpoint_bytes,fp32_evaluation_tensor_bytes=tensor_storage_bytes(block))


def prepare(root,training_root,*,split="development",confirmation_root=None):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Hide CUDA for CPU evaluation preparation")
    root,training_root=Path(root).resolve(),Path(training_root).resolve()
    if root.exists():raise FileExistsError(root)
    if split not in ("development","confirmation"):raise ValueError("Only development or confirmation evaluation")
    # This gate precedes ANY confirmation data read.
    training,result,selected,checkpoint=completed_training(training_root)
    source=Path(training["source"])
    inputs=[training_root/"plan.json",training_root/"pilot/manifest.json",training_root/"pilot/result.json",
        training_root/"train/manifest.json",training_root/"train/result.json",selected/"manifest.json",source/"plan.json"]
    verified_hashes={}
    for name,expected in checkpoint["artifacts"].items():
        actual=recipe.sha(selected/name)
        if actual!=expected:raise ValueError("Changed selected checkpoint")
        verified_hashes[str(selected/name)]=actual
        inputs.append(selected/name)
    items=recipe.read(training_root/"items.json")["items"]
    confirmation_fingerprint=None
    if split=="development":
        cases=[c for c in items if c["split"]=="development"]
        if not cases or len({c["identity"] for c in cases})!=len(cases):raise ValueError("Invalid development identities")
        inputs.append(training_root/"items.json")
    else:
        if confirmation_root is None:raise ValueError("A frozen confirmation root is required")
        from . import jspace_recovery_data as fresh
        confirmation_root=Path(confirmation_root).resolve()
        confirmation_fingerprint=fresh.verify(confirmation_root)["fingerprint"]
        tasks=recipe.read(confirmation_root/"tasks.json")
        exclusions=recipe.read(confirmation_root/"exclusions.json")
        old_tasks=recipe.read(source/"tasks.json")
        forbidden=set(exclusions["identities"])|fresh.all_identities(old_tasks)|{c["identity"] for c in items}
        cases=validate_confirmation(tasks["splits"]["confirmation"],forbidden)
        inputs += [source/"tasks.json",training_root/"items.json"]
        inputs += [confirmation_root/name for name in ("plan.json","tasks.json","exclusions.json","manifest.json")]
    cases_payload=dict(split=split,cases=cases)
    files=[Path(__file__),recipe.ROOT/"tests/structure_net/test_jspace_joint_evaluate.py",
        Path(__file__).with_name("jspace_quadratic_evaluate.py")]
    plan=dict(schema=SCHEMA,training_root=str(training_root),training_fingerprint=training["fingerprint"],
        selected_directory=str(selected),selected_epoch=result["selected_epoch"],split=split,
        confirmation_root=None if confirmation_root is None else str(confirmation_root),
        confirmation_fingerprint=confirmation_fingerprint,cases_digest=recipe.digest(cases_payload),
        identities=[c["identity"] for c in cases],cases=len(cases),maximum_accuracy_drop=.05,
        precision="Original BF16-rounded values promoted to FP32; candidate recovered BF16 parameters promoted to FP32; FP32 geometry",
        selection="No evaluator checkpoint selection; completed training choice fixed before opening confirmation",
        input_sha256={str(p):verified_hashes.get(str(p)) or recipe.sha(p) for p in inputs},
        input_identity={str(p):recipe.file_identity(p) for p in inputs},
        source_sha256={str(p):recipe.sha(p) for p in files},speed_claim=False,
        byte_scope="One original tied E + final RMS gain + rotary + every native/compact block; scorer label-row temporaries excluded")
    plan["fingerprint"]=recipe.digest(plan)
    root.mkdir(parents=True)
    for path in files:
        target=root/"producer"/path.relative_to(recipe.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    recipe.write(root/"plan.json",plan);recipe.write(root/"cases.json",cases_payload)
    return dict(status="prepared",fingerprint=plan["fingerprint"],split=split,cases=len(cases))


def verify(root):
    root=Path(root);plan=recipe.read(root/"plan.json")
    if plan["schema"]!=SCHEMA or plan["fingerprint"]!=recipe.digest({k:v for k,v in plan.items() if k!="fingerprint"}):
        raise ValueError("Changed evaluation plan")
    training,result,selected,checkpoint=completed_training(plan["training_root"])
    if training["fingerprint"]!=plan["training_fingerprint"] or str(selected)!=plan["selected_directory"]:
        raise ValueError("Changed selected joint run")
    if recipe.digest(recipe.read(root/"cases.json"))!=plan["cases_digest"]:raise ValueError("Changed evaluation cases")
    for path,expected in plan["source_sha256"].items():
        if recipe.sha(path)!=expected or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT))!=expected:
            raise ValueError("Changed evaluator producer")
    for path,identity in plan["input_identity"].items():
        if recipe.file_identity(path)!=identity:raise ValueError("Changed frozen evaluation input metadata")
    return plan,training,checkpoint


def aggregate(root):
    root=Path(root);plan,_,_=verify(root)
    values={}
    for arm in ("native","candidate"):
        directory=root/arm
        if not (directory/"manifest.json").exists():return None
        manifest=recipe.read(directory/"manifest.json")
        if (manifest["status"]!="completed" or manifest["fingerprint"]!=plan["fingerprint"]
                or manifest["result_sha256"]!=recipe.sha(directory/"result.json")):
            raise ValueError("Mismatched evaluation arm")
        values[arm]=recipe.read(directory/"result.json")
    if (root/"result.json").exists():return recipe.read(root/"result.json")
    comparison=paired_summary(values["native"]["records"],values["candidate"]["records"],maximum_accuracy_drop=.05)
    sizes={arm:values[arm]["state_bytes"] for arm in values}
    for name in ("checkpoint_tensor_bytes","fp32_evaluation_tensor_bytes"):
        comparison[name+"_reduction"]=1-sizes["candidate"][name]/sizes["native"][name]
    result=dict(status="completed",fingerprint=plan["fingerprint"],split=plan["split"],
        comparison=comparison,state_bytes=sizes,speed_claim=False,
        scope="Full-vocabulary models; label scores only. Mixed-precision checkpoint storage and matched-FP32 scoring storage reported separately.")
    recipe.write(root/"result.json",result)
    return result


def run(root,arm):
    if arm not in ("native","candidate"):raise ValueError("Expected native or candidate")
    root=Path(root).resolve();plan,training,checkpoint=verify(root);directory=root/arm
    if directory.exists():raise FileExistsError(directory)
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_native_block,load_readout_norm
    lease=wait_ready([0]);allocation=dict(lease["allocation"])
    torch.set_num_threads(4);torch.cuda.init();torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    def guard():
        current=wait_ready([0])
        if current["id"]!=lease["id"] or current["allocation"]!=allocation:raise RuntimeError("Evaluation reservation changed")
        joint.check_reserved_peak(allocation,torch.cuda.max_memory_reserved())
    directory.mkdir(exist_ok=False);started=time.monotonic()
    try:
        parent=recipe.read(Path(training["source"])/"plan.json")
        embedding,norm,rotary=load_readout_norm(parent["model"]["path"],device="cpu",dtype=torch.bfloat16)
        if arm=="candidate":
            norm_path=Path(plan["selected_directory"])/"norm_gain.pt"
            if recipe.sha(norm_path)!=checkpoint["artifacts"][norm_path.name]:raise ValueError("Changed selected RMS gain")
            gain=recipe.load(norm_path)
            if gain.shape!=norm.weight.shape or gain.dtype!=torch.float32:raise ValueError("Invalid recovered native RMS gain")
            norm.weight=torch.nn.Parameter(gain.clone(),requires_grad=False)
        checkpoint_shared=tensor_storage_bytes(embedding,norm,rotary)
        embedding=embedding.float();norm.float();rotary.float()
        fp32_shared=tensor_storage_bytes(embedding,norm,rotary)
        rotary.to("cuda:0")
        rows=[]
        def loader(layer):
            guard()
            if arm=="candidate":
                block,sizes=load_candidate_block(training,plan["selected_directory"],layer,"cuda:0")
            else:
                block=load_native_block(parent["model"]["path"],layer,device="cuda:0",dtype=torch.bfloat16)
                before=tensor_storage_bytes(block);block.float()
                sizes=dict(checkpoint_tensor_bytes=before,fp32_evaluation_tensor_bytes=tensor_storage_bytes(block))
            rows.append(dict(layer=layer,**sizes))
            return block
        cases=recipe.read(root/"cases.json")["cases"]
        result=stream_score(cases,input_embedding=embedding,block_loader=loader,final_readout=(norm,embedding),
            label_ids=parent["label_ids"],scratch_dir=directory/"scratch",rotary=rotary,
            layer_count=parent["config"]["layers"],device="cuda:0",guard=guard,microbatch=4,
            progress=lambda **values:recipe.event(root,arm=arm,**values))
        guard()
        result["state_bytes"]=dict(checkpoint_tensor_bytes=checkpoint_shared+sum(r["checkpoint_tensor_bytes"] for r in rows),
            fp32_evaluation_tensor_bytes=fp32_shared+sum(r["fp32_evaluation_tensor_bytes"] for r in rows),
            checkpoint_shared_bytes=checkpoint_shared,fp32_shared_bytes=fp32_shared,blocks=rows,
            tied_embedding_copies=1,full_vocabulary=embedding.shape[0],label_row_temporaries_counted=False)
        result.update(status="completed",arm=arm,fingerprint=plan["fingerprint"],speed_claim=False)
        recipe.write(directory/"result.json",result)
        recipe.write(directory/"manifest.json",dict(status="completed",fingerprint=plan["fingerprint"],
            result_sha256=recipe.sha(directory/"result.json"),lease_id=lease["id"],allocation=allocation,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            elapsed_seconds=time.monotonic()-started))
        aggregate(root)
        return dict(status="completed",arm=arm,cases=len(cases),state_bytes=result["state_bytes"])
    except Exception as error:
        recipe.write(directory/"failure.json",dict(status="failed",fingerprint=plan["fingerprint"],arm=arm,
            error_type=type(error).__name__,error=str(error),lease_id=lease["id"],allocation=allocation,
            peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","run","aggregate"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--training-root",type=Path)
    parser.add_argument("--split",choices=("development","confirmation"),default="development")
    parser.add_argument("--confirmation-root",type=Path)
    parser.add_argument("--arm",choices=("native","candidate"))
    args=parser.parse_args()
    if args.stage=="prepare":
        if args.training_root is None:parser.error("prepare requires --training-root")
        value=prepare(args.root,args.training_root,split=args.split,confirmation_root=args.confirmation_root)
    elif args.stage=="run":
        if args.arm is None:parser.error("run requires --arm")
        value=run(args.root,args.arm)
    else:value=aggregate(args.root)
    print(json.dumps(value,allow_nan=False),flush=True)


if __name__=="__main__":main()
