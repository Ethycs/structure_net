"""Stream the fixed locked-J/input-update enrichment through all36 native blocks."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import gc
import shutil

import numpy as np
import torch

from . import jspace_quadratic_recipe as recipe
from .jspace_first_block_enrich import enrich_basis, guard_device, projection_metrics

FAMILIES = ("arc","boolq","hellaswag","winogrande")


def training_selection(items, *, start=0, per_family=32, positions=16):
    cases=[case for family in FAMILIES for case in
        [case for case in items if case["split"]=="train" and case["family"]==family][start:start+per_family]]
    if len(cases)!=per_family*len(FAMILIES) or len({c["identity"] for c in cases})!=len(cases):
        raise ValueError("Distinct balanced training-only population required")
    indices=[];zero=[]
    for case in cases:
        length=len(case["input_ids"])
        if case["stop"]-case["offset"]!=length or length<positions:
            raise ValueError("Cached offsets or sampling length changed")
        selected=torch.linspace(0,length-1,positions).long().tolist()
        indices.extend(case["offset"]+position for position in selected)
        zero.extend(position==0 for position in selected)
    return dict(identities=[c["identity"] for c in cases],indices=indices,position0=zero)


def sample_boundary(path, selection):
    bank=np.load(path,mmap_mode="r")
    try:
        if bank.dtype!=np.int16 or bank.ndim!=2 or max(selection["indices"])>=len(bank):
            raise ValueError("Expected original BF16-bit boundary bank")
        return torch.from_numpy(np.array(bank[selection["indices"]],copy=True)).view(torch.bfloat16).float()
    finally:bank._mmap.close()


def diagnostic_groups(basis,mean,h0,h1,position0,g0,g1=None):
    return {name:projection_metrics(basis,mean,h0[mask],h1[mask],g0,g1) for name,mask in
        (("all_positions",torch.ones_like(position0)),("position0",position0),("non_position0",~position0))}


@torch.no_grad()
def fit_layer(original,h0,h1,position0,*,guard=None):
    """Preserve calibration head data; only the channel basis changes."""
    basis,metadata=enrich_basis(original["basis"],original["means"],h0,h1,position0,
        functional_columns=original["metadata"]["retained_functional_directions"],guard=guard)
    result={**original,"basis":basis,
        "metadata":{**original["metadata"],"enrichment":metadata}}
    return result,metadata


def producers():
    return [Path(__file__),Path(__file__).with_name("jspace_first_block_enrich.py"),
        Path(__file__).with_name("jspace_residual_carry.py"),Path(__file__).with_name("jspace_prefix_localize.py"),
        recipe.ROOT/"tests/structure_net/test_jspace_all_layer_enrich.py"]


def fit(root,source,device="cpu"):
    from .jspace_full_lease import wait_ready
    root,source=Path(root).resolve(),Path(source).resolve()
    if root.exists():raise FileExistsError(root)
    lease=wait_ready([0]) if torch.device(device).type=="cuda" else None
    guard=(lambda:wait_ready([0])) if lease else None
    device=guard_device(device,guard);torch.set_num_threads(2)
    if device.type=="cuda":torch.cuda.init();torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    parent=recipe.verify(source)
    if parent["config"]["layers"]!=36:raise ValueError("Original36-block source required")
    cache=Path(parent["teacher_root"])/"capture"
    inputs={str(path):recipe.sha(path) for path in (source/"plan.json",source/"items.json",source/"basis/manifest.json")}
    original_manifest=recipe.checked(source,source/"basis",hashes=False)
    for path in [cache/"statistics.pt",*[cache/f"boundary_{layer:02d}.npy" for layer in range(37)]]:
        expected=parent["input_sha256"][str(path)]
        if recipe.sha(path)!=expected:raise ValueError("Changed original training bank: "+str(path))
        inputs[str(path)]=expected
    for layer in range(36):
        path=recipe.fit_path(source,layer);expected=original_manifest["artifacts"][str(path.relative_to(source))]
        if recipe.sha(path)!=expected:raise ValueError("Changed original geometry")
        inputs[str(path)]=expected
    items=recipe.read(source/"items.json")["items"]
    selected,check=training_selection(items),training_selection(items,start=32)
    if set(selected["identities"])&set(check["identities"]):raise ValueError("Fitting/check training cases overlap")
    plan=dict(schema="jspace.all-layer-enriched-basis.v1",source_root=str(source),source_fingerprint=parent["fingerprint"],
        layers=list(range(36)),rank=1024,stored_head_projection_dim=64,
        head_scope="Copied original64-dimensional calibration projection tensors; executed head width is frozen separately in the development-probe plan",
        fit_training_identities=selected["identities"],internal_training_check_identities=check["identities"],
        positions_per_case=16,
        objective="Identical frozen first-block enrichment: locked resolved input-J columns and global/position0 means, then equal trace-normalized input/update moments at nonzero positions",
        input_sha256=inputs,source_sha256={str(path):recipe.sha(path) for path in producers()},
        scope="Training-only basis extension; no native weights or decoder training; internal training checks are not held-out evidence",
        memory_scope="Only one layer's sampled input/output states, Gramians and eigensystem on device; native banks are mapped and selected rows copied")
    plan["fingerprint"]=recipe.digest(plan);root.mkdir(parents=True)
    for path in producers():
        target=root/"producer"/path.relative_to(recipe.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    recipe.write(root/"plan.json",plan)
    statistics=recipe.load(cache/"statistics.pt")
    zero=torch.tensor(selected["position0"],device=device,dtype=torch.bool)
    check_zero=torch.tensor(check["position0"],device=device,dtype=torch.bool)
    artifacts={};summaries={}
    try:
        for layer in range(36):
            guard_device(device,guard)
            original=recipe.load(recipe.fit_path(source,layer))
            current={**original,"basis":original["basis"].to(device),"means":original["means"].to(device)}
            h0,h1,ch0,ch1=[sample_boundary(cache/f"boundary_{index:02d}.npy",selection).to(device)
                for selection,index in ((selected,layer),(selected,layer+1),(check,layer),(check,layer+1))]
            g0=statistics["grams"][layer].to(device);g0=g0/g0.trace()
            g1=None
            if layer+1<len(statistics["grams"]):
                g1=statistics["grams"][layer+1].to(device);g1=g1/g1.trace()
            candidate,metadata=fit_layer(current,h0,h1,zero,guard=guard)
            metadata.update(layer=layer,output_jacobian_available=g1 is not None,
                original_ritz_metadata_scope="Inherited parent Ritz fields describe the original basis only; use enrichment.diagnostics for this candidate",
                stored_head_projection_dim=64,diagnostics={})
            for name,basis in (("original",current["basis"]),("enriched",candidate["basis"])):
                guard_device(device,guard)
                metadata["diagnostics"][name]=dict(input_gram_trace_retained=float((basis*(g0@basis)).double().sum()),
                    output_gram_trace_retained=None if g1 is None else float((basis*(g1@basis)).double().sum()),
                    fitting_train=diagnostic_groups(basis,current["means"],h0,h1,zero,g0,g1),
                    separate_training_check=diagnostic_groups(basis,current["means"],ch0,ch1,check_zero,g0,g1))
            directory=root/"basis"/f"block_{layer:02d}";directory.mkdir(parents=True)
            saved={**original,"basis":candidate["basis"].cpu(),
                "metadata":{**original["metadata"],"enrichment":metadata}}
            recipe.save(directory/"fit.pt",saved);recipe.write(directory/"fit.json",metadata)
            for path in (directory/"fit.pt",directory/"fit.json"):artifacts[str(path.relative_to(root))]=recipe.sha(path)
            summary=dict(layer=layer,functional_columns_locked=metadata["functional_columns_locked"],
                input_G_energy=metadata["diagnostics"]["enriched"]["input_gram_trace_retained"],
                training_check=metadata["diagnostics"]["enriched"]["separate_training_check"])
            summaries[str(layer)]=summary;recipe.event(root,"enriched_layer",**summary)
            del original,current,h0,h1,ch0,ch1,g0,g1,candidate,metadata,saved,basis
            gc.collect()
            if device.type=="cuda":torch.cuda.empty_cache()
        guard_device(device,guard)
        peak=torch.cuda.max_memory_reserved() if device.type=="cuda" else None
        if lease and peak>sum(lease["allocation"].values())*1024**2:raise RuntimeError("All-layer basis fitting exceeded reservation")
        recipe.write(root/"basis/manifest.json",dict(status="completed",fingerprint=plan["fingerprint"],
            source_fingerprint=parent["fingerprint"],artifacts=artifacts,layers=summaries,
            lease_id=None if lease is None else lease["id"],peak_reserved_bytes=peak))
        return dict(status="completed",root=str(root),fingerprint=plan["fingerprint"],layers=36)
    except BaseException as error:
        recipe.write(root/"failed.json",dict(error=repr(error),fingerprint=plan["fingerprint"]));raise


def probe(root,source,fit_root,head_dim=128):
    from .jspace_full_lease import wait_ready
    from .jspace_prefix_localize import development_cases
    from .jspace_qwen_parallel_model import load_native_block,load_readout_norm
    from .jspace_qwen_affine_block import AffineQwenBlock
    from .jspace_quadratic_qwen import QuadraticQwenBlock
    from .jspace_residual_carry import ResidualCarryBlock
    from .jspace_quadratic_evaluate import stream_score
    root,source,fit_root=map(lambda p:Path(p).resolve(),(root,source,fit_root))
    if root.exists():raise FileExistsError(root)
    if head_dim not in (64,128):raise ValueError("Explicit head width64 or128 required")
    parent=recipe.verify(source);fitted=recipe.read(fit_root/"plan.json");manifest=recipe.read(fit_root/"basis/manifest.json")
    if (fitted["fingerprint"]!=recipe.digest({k:v for k,v in fitted.items() if k!="fingerprint"}) or
            fitted["source_fingerprint"]!=parent["fingerprint"] or fitted["layers"]!=list(range(36)) or
            manifest["status"]!="completed" or manifest["fingerprint"]!=fitted["fingerprint"]):
        raise ValueError("Incomplete or mismatched all-layer fit")
    for path,expected in fitted["source_sha256"].items():
        if recipe.sha(path)!=expected:raise ValueError("Enrichment source changed after fitting")
    cases,_=development_cases(source)
    plan=dict(schema="jspace.all-layer-enrichment-probe.v1",source_root=str(source),source_fingerprint=parent["fingerprint"],
        fit_root=str(fit_root),fit_fingerprint=fitted["fingerprint"],fit_manifest_sha256=recipe.sha(fit_root/"basis/manifest.json"),
        layers=list(range(36)),rank=1024,executed_head_dim=head_dim,stored_calibration_head_projection_dim=64,
        attention="softmax; no learned metric",initializer="original native BF16-rounded weights promoted toFP32",
        native_embedding_and_readout=True,residual="full native carry, dynamic RMS complement energy",
        split="development",identities=[case["identity"] for case in cases],
        source_sha256={str(path):recipe.sha(path) for path in producers()},
        scope="Full36 development-only geometry probe before decoder training; no fresh confirmation or throughput claim")
    plan["fingerprint"]=recipe.digest(plan)
    lease=wait_ready([0]);guard=lambda:wait_ready([0])
    torch.cuda.init();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    root.mkdir(parents=True);recipe.write(root/"plan.json",plan)
    for path in producers():
        target=root/"producer"/path.relative_to(recipe.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    try:
        def loader(layer):
            guard();path=fit_root/"basis"/f"block_{layer:02d}"/"fit.pt"
            if recipe.sha(path)!=manifest["artifacts"][str(path.relative_to(fit_root))]:raise ValueError("Enriched basis changed")
            fit=recipe.load(path);native=load_native_block(parent["model"]["path"],layer,device="cuda:0",dtype=torch.bfloat16).float()
            head={k:fit[k].to("cuda:0") for k in ("rope_pairs","value_basis")} if head_dim==64 else None
            affine=AffineQwenBlock(native,fit["basis"].to("cuda:0"),fit["means"].to("cuda:0"),
                parameter_dtype=torch.float32,trainable=False,head_projection=head)
            return ResidualCarryBlock(QuadraticQwenBlock(affine,mode="softmax",learn_metric=False)).pack(dtype=torch.float32,buffer_dtype=torch.float32)
        embedding,norm,_=load_readout_norm(parent["model"]["path"],device="cpu",dtype=torch.bfloat16)
        embedding,norm=embedding.float(),norm.float()
        result=stream_score(cases,input_embedding=embedding,block_loader=loader,final_readout=(norm,embedding),
            label_ids=parent["label_ids"],scratch_dir=root/"scratch",rotary=recipe.rotary_module(parent,"cuda:0"),
            device="cuda:0",guard=guard,layer_count=36,microbatch=4,
            progress=lambda **kw:recipe.event(root,"progress",detail=kw))
        correct=sum(row["prediction"]==row["answer"] for row in result["records"]);peak=torch.cuda.max_memory_reserved()
        if peak>sum(lease["allocation"].values())*1024**2:raise RuntimeError("Enriched all-layer probe exceeded reservation")
        guard();result.update(status="completed",fingerprint=plan["fingerprint"],correct=correct,count=len(cases),
            accuracy=correct/len(cases),executed_head_dim=head_dim,scope=plan["scope"],
            prediction_counts=dict(Counter(row["prediction"] for row in result["records"])),
            by_family={family:dict(correct=sum(row["prediction"]==row["answer"] for row in result["records"] if row["family"]==family),
                count=sum(row["family"]==family for row in result["records"])) for family in FAMILIES},
            peak_reserved_bytes=peak,lease_id=lease["id"])
        recipe.write(root/"result.json",result)
        recipe.write(root/"manifest.json",dict(status="completed",fingerprint=plan["fingerprint"],
            artifacts={"plan.json":recipe.sha(root/"plan.json"),"result.json":recipe.sha(root/"result.json")}))
        return {k:result[k] for k in ("status","correct","count","accuracy","executed_head_dim")}
    except BaseException as error:
        recipe.write(root/"failed.json",dict(error=repr(error),fingerprint=plan["fingerprint"]));raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=("fit","probe"))
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--source",type=Path,default=recipe.DEFAULT_ROOT)
    parser.add_argument("--device",default="cpu");parser.add_argument("--fit-root",type=Path)
    parser.add_argument("--head-dim",type=int,choices=(64,128),default=128)
    args=parser.parse_args()
    print(fit(args.root,args.source,args.device) if args.stage=="fit" else probe(args.root,args.source,args.fit_root,args.head_dim))
