"""All-layer coordinate folding, progressive adaptation and matched timing."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import jspace_full_capture as base
from . import jspace_full_fit as fitting
from .jspace_bench_data import DATASETS,sha,write
from .jspace_bench_tensor import accuracy,comparison,model_bytes,run_cases
from .jspace_compact_model import RotatedBlock,RotatedEmbedding,RotatedFinalNorm
from .jspace_full_lease import StableStage
from .jspace_full_fit_v2 import DEFAULT_ROOT as FIT_ROOT

DEFAULT_ROOT=base.ROOT/"data/experiments/jspace_full_compact/20261113_v1"
SHAKEDOWN_ROOT=base.ROOT/"data/experiments/jspace_full_compact/20261113_training_shakedown_v1"


def prepare(root,parent,systems_only=False):
    base.checked(parent,"fit")
    names=["jspace_compact_model.py","jspace_full_lease.py","jspace_stacked_graph_decode.py","jspace_stacked_graph_decode_v2.py",
           "jspace_stacked_graph_decode_v3.py","jspace_bench_stacked.py","jspace_bench_stacked_timing.py","jspace_packed_core.py"]
    paths=[Path(__file__),base.ROOT/"tests/structure_net/test_jspace_full_compact.py"]
    paths += [Path(__file__).parent/n for n in names]
    return fitting.derive_root(root,parent,paths,[parent/"fit_manifest.json",parent/"fit.pt",parent/"fit.json"],
                               fit_root=str(parent.resolve()),systems_only=systems_only)


def start(root,name):
    plan=base.verify(root);base.checked(Path(plan["fit_root"]),"fit")
    stage=StableStage(root,name)
    fit=torch.load(Path(plan["fit_root"])/"fit.pt",map_location="cpu",weights_only=True)
    return stage,fit


def rotate(stage,fit):
    q=fit["basis"].cuda()
    for i in range(36):
        original=stage.model.model.layers[i]
        stage.model.model.layers[i]=RotatedBlock(original,q,stage.model.config)
        del original
        stage.event("rotate_layer",layer=i)
    embedding=RotatedEmbedding(stage.model.model.embed_tokens,q)
    stage.model.model.norm=RotatedFinalNorm(stage.model.model.norm,embedding)
    stage.model.model.embed_tokens=embedding
    stage.model.requires_grad_(False)
    del q;gc.collect();torch.cuda.empty_cache()


def compile_layer(stage,fit,i):
    block=stage.model.model.layers[i];entry=fit["layers"][str(i)]
    block.self_attn.compile(base.CONFIG["functional_rank"],entry["mean_coordinates"],entry["attention_group"],
                           entry["rope_pairs"],entry["value_basis"])
    block.mlp.compile(entry,entry["removed"])
    return block


def full_mask(x):
    length=x.shape[1]
    return torch.full((length,length),torch.finfo(x.dtype).min,device=x.device,dtype=x.dtype).triu(1)[None,None]


def position_embeddings(model,x):
    return model.model.rotary_emb(x,torch.arange(x.shape[1],device=x.device)[None])


def block_forward(block,x,pos,mask):
    return block(x,attention_mask=mask,position_embeddings=pos,use_cache=False)[0]


def tail_logits(stage,local,layer,pos,mask,positions):
    h=local
    for block in stage.model.model.layers[layer+1:]:h=block_forward(block,h,pos,mask)
    return F.linear(stage.model.model.norm(h[:,positions]),stage.model.lm_head.weight)[0].float()


def losses(stage,x,target,teacher_input,teacher_logits,case,layer):
    pos=position_embeddings(stage.model,x);mask=full_mask(x)
    local=block_forward(stage.model.model.layers[layer],x,pos,mask)
    logits=tail_logits(stage,local,layer,pos,mask,case["loss_positions"])
    temperature=base.CONFIG["temperature"]
    log_target=(teacher_logits.float()/temperature).log_softmax(-1)
    kd=(log_target.exp()*(log_target-(logits/temperature).log_softmax(-1))).sum(-1).mean()
    ce=F.cross_entropy(logits,torch.tensor(case["targets"],device=x.device))
    local_mse=(local.float()-target.float()).square().mean()/(target.float()-teacher_input.float()).square().mean().clamp_min(1e-6)
    total=base.CONFIG["kd_weight"]*temperature**2*kd+base.CONFIG["ce_weight"]*ce+base.CONFIG["local_weight"]*local_mse
    return total,{"loss":float(total.detach()),"kl":float(kd.detach()),"ce":float(ce.detach()),"local_relative_mse":float(local_mse.detach())}


def bits(root,name):return np.load(root/name,mmap_mode="r")


def cache_for(stage):
    root=Path(stage.plan["capture_root"])/"capture"
    meta=json.loads((root/"items.json").read_text())
    return root,meta,bits(root,"teacher_logits.npy")


def rotate_state(stage,array,case):
    native=base.read_bits(array,slice(case["offset"],case["stop"]))[None]
    return F.linear(native,stage.model.model.embed_tokens.basis.T)


def compact_state(block):
    return {name:p.detach().cpu().clone() for name,p in block.named_parameters() if p.requires_grad}


def restore_compact(block,state):
    parameters=dict(block.named_parameters())
    with torch.no_grad():
        for name,value in state.items():parameters[name].copy_(value.to(parameters[name]))


def timing_cases(stage):
    # Fixed pilot prompts, independent of the expanded adaptation split.
    old=json.loads((base.PARENT/"tasks.json").read_text())["splits"]["development"]
    return [c for family in DATASETS for c in [v for v in old if v["family"]==family][:2]]


def timed_generation(stage,tag):
    from .jspace_stacked_graph_decode_v3 import DynamicPrefillGraphDecoder
    cases=timing_cases(stage);decoder=DynamicPrefillGraphDecoder(stage.model)
    setup=decoder.capture();warm=torch.tensor([cases[0]["input_ids"]],device="cuda:0")
    with torch.inference_mode():
        a=decoder.generate(warm,False);b=decoder.generate(warm,True)
        if not torch.equal(a,b):raise ValueError("Graph/uncaptured mismatch")
    results={};outputs={}
    for arm in ("hf","cuda_graph"):
        rows=[];outputs[arm]=[]
        def generate(ids):
            if arm=="cuda_graph":return decoder.generate(ids,True)
            return stage.model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,use_cache=True,
                max_new_tokens=32,min_new_tokens=32,pad_token_id=stage.tokenizer.eos_token_id)[0,-32:]
        with torch.inference_mode():
            generate(warm);torch.cuda.synchronize()
            for case in cases:
                stage.guard([0]);ids=torch.tensor([case["input_ids"]],device="cuda:0")
                torch.cuda.synchronize();t=time.perf_counter();tokens=generate(ids);torch.cuda.synchronize()
                rows.append({"identity":case["identity"],"prompt_tokens":ids.shape[1],"new_tokens":len(tokens),"seconds":time.perf_counter()-t})
                outputs[arm].append(tokens.cpu().tolist())
        results[arm]={"tokens_per_second":sum(r["new_tokens"] for r in rows)/sum(r["seconds"] for r in rows),"samples":rows}
        stage.event("timing",model=tag,arm=arm,tokens_per_second=results[arm]["tokens_per_second"])
    graph_rows=[]
    with torch.inference_mode():
        for case in stage.tasks["evaluation"]:
            h=decoder.prefill(torch.tensor([case["input_ids"]],device="cuda:0"))
            logits=F.linear(h,stage.model.lm_head.weight[stage.label_ids[:len(case["choices"])]]).float()[0]
            if not torch.isfinite(logits).all():raise FloatingPointError("Nonfinite graph prefill")
            graph_rows.append({"identity":case["identity"],"family":case["family"],"answer":case["answer"],
                               "prediction":int(logits.argmax()),"logits":logits.cpu().tolist()})
    value={"timings":results,"graph_capture_seconds":setup,"generated_tokens":outputs,"graph_static_warm_equal":True,
           "resident_tensor_bytes":model_bytes(stage.model),"graph_prefill_accuracy":accuracy(graph_rows),
           "scope":"batch one; same eight pilot prompts; 32 greedy tokens; prefill included, capture amortized"}
    write(stage.path/(tag+"_timing.json"),value);write(stage.path/(tag+"_graph_evaluation.json"),graph_rows)
    del decoder;gc.collect();torch.cuda.empty_cache()
    return value


def controls(root):
    stage,fit=start(root,"controls")
    value={}
    for tag in ("original","rotated","initialized_compact"):
        if tag=="rotated":rotate(stage,fit)
        elif tag=="initialized_compact":
            for i in range(36):compile_layer(stage,fit,i).pack()
        evaluation=run_cases(stage,stage.tasks["evaluation"],tag+"_evaluation")
        development=run_cases(stage,stage.tasks["development"],tag+"_development")
        value[tag]={"evaluation":accuracy(evaluation),"development":accuracy(development),
                    "parameters":sum(p.numel() for p in stage.model.parameters()),"tensor_bytes":model_bytes(stage.model)}
        if tag=="original":reference=evaluation;value[tag]["runtime"]=timed_generation(stage,tag)
        else:value[tag]["comparison_to_original"]=comparison(reference,evaluation)
        stage.event("control_complete",arm=tag,correct=value[tag]["evaluation"]["correct"])
    write(root/"controls.json",value)
    return stage.finish(["controls.json"])


def shakedown(root):
    stage,fit=start(root,"shakedown");rotate(stage,fit)
    source,meta,teacher_logits=cache_for(stage)
    index=max((i for i,c in enumerate(meta["items"]) if c["split"]=="train"),key=lambda i:len(meta["items"][i]["input_ids"]))
    case=meta["items"][index];report=[]
    for layer in (0,35):
        block=compile_layer(stage,fit,layer)
        x=rotate_state(stage,bits(source,f"boundary_{layer:02d}.npy"),case)
        target=rotate_state(stage,bits(source,f"boundary_{layer+1:02d}.npy"),case)
        teachers=base.read_bits(teacher_logits,index)
        parameters=block.compact_parameters();optimizer=torch.optim.AdamW(parameters,lr=base.CONFIG["learning_rate"])
        before=compact_state(block)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss,values=losses(stage,x,target,x,teachers,case,layer)
            if not torch.isfinite(loss):raise FloatingPointError("Invalid shakedown loss")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,base.CONFIG["gradient_clip"],error_if_nonfinite=True)
            if not all(p.grad is not None for p in parameters):raise ValueError("Missing compact gradient")
            if any(p.grad is not None for p in block.parameters() if not p.requires_grad):raise ValueError("Native weight gradient")
            optimizer.step();stage.event("shakedown_backward",layer=layer,step=step,gradient_norm=float(norm),**values)
        after=compact_state(block)
        if not any(not torch.equal(v,after[k]) for k,v in before.items()):raise ValueError("No coefficient update")
        base.save(stage.path/f"layer_{layer:02d}.pt",after)
        with torch.no_grad():
            pos=position_embeddings(stage.model,x);mask=full_mask(x);expected=block_forward(block,x,pos,mask)
            restore_compact(block,before);restore_compact(block,torch.load(stage.path/f"layer_{layer:02d}.pt",weights_only=True))
            actual=block_forward(block,x,pos,mask)
            if not torch.equal(expected,actual):raise ValueError("Checkpoint reload mismatch")
        report.append({"layer":layer,"updated":True,"reload_exact":True,"loss":values})
        optimizer.zero_grad(set_to_none=True);del optimizer,parameters,loss,x,target,teachers
        block.pack();gc.collect();torch.cuda.empty_cache()
    for layer in range(1,35):compile_layer(stage,fit,layer).pack()
    from .jspace_stacked_graph_decode_v3 import DynamicPrefillGraphDecoder
    decoder=DynamicPrefillGraphDecoder(stage.model);seconds=decoder.capture()
    ids=torch.tensor([timing_cases(stage)[0]["input_ids"]],device="cuda:0")
    if not torch.equal(decoder.generate(ids,False),decoder.generate(ids,True)):raise ValueError("Compact graph mismatch")
    write(stage.path/"checks.json",{"backward":report,"graph_capture_seconds":seconds,"graph_tokens_exact":True})
    return stage.finish(systems_only=True)


def train(root):
    base.checked(root,"controls")
    stage,fit=start(root,"train");rotate(stage,fit)
    source,meta,teacher_logits=cache_for(stage)
    train_indices=[i for i,c in enumerate(meta["items"]) if c["split"]=="train"]
    references=json.loads((root/"controls/original_evaluation.json").read_text())
    previous=None;trajectory={};layer_summaries=[]
    for layer in range(36):
        block=compile_layer(stage,fit,layer)
        parameters=block.compact_parameters()
        if {id(p) for p in stage.model.parameters() if p.requires_grad}!={id(p) for p in parameters}:
            raise ValueError("Optimizer scope extends beyond current compact layer")
        optimizer=torch.optim.AdamW(parameters,lr=base.CONFIG["learning_rate"])
        native_input=bits(source,f"boundary_{layer:02d}.npy");native_target=bits(source,f"boundary_{layer+1:02d}.npy")
        order=train_indices.copy();random.Random(base.CONFIG["seed"]+layer).shuffle(order)
        history=[];started=time.perf_counter()
        for step in range(base.CONFIG["adapt_steps"]):
            index=order[step%len(order)];case=meta["items"][index]
            teacher_input=rotate_state(stage,native_input,case)
            x=teacher_input if previous is None else base.read_bits(previous,slice(case["offset"],case["stop"]))[None]
            target=rotate_state(stage,native_target,case);teachers=base.read_bits(teacher_logits,index)
            optimizer.zero_grad(set_to_none=True)
            loss,values=losses(stage,x,target,teacher_input,teachers,case,layer)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite adaptation loss")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,base.CONFIG["gradient_clip"],error_if_nonfinite=True)
            optimizer.step()
            row={"layer":layer,"step":step,"identity":case["identity"],"gradient_norm":float(norm),**values};history.append(row)
            stage.event("adapt",**row)
            del x,target,teachers,teacher_input,loss
        state=compact_state(block)
        base.save(stage.path/f"compact_{layer:02d}.pt",state)
        optimizer.zero_grad(set_to_none=True);del optimizer,parameters,state
        write(stage.path/f"losses_{layer:02d}.json",history)
        block.pack()
        next_path=stage.path/f"student_boundary_{layer+1:02d}.npy"
        upcoming=np.lib.format.open_memmap(next_path,mode="w+",dtype=np.int16,shape=(meta["tokens"],2048))
        with torch.inference_mode():
            for index,case in enumerate(meta["items"]):
                x=rotate_state(stage,native_input,case) if previous is None else base.read_bits(previous,slice(case["offset"],case["stop"]))[None]
                output=block_forward(block,x,position_embeddings(stage.model,x),full_mask(x))
                if not torch.isfinite(output).all():raise FloatingPointError("Nonfinite propagated prefix")
                base.write_bits(upcoming,slice(case["offset"],case["stop"]),output[0])
        upcoming.flush();previous=bits(stage.path,next_path.name)
        values={"layer":layer,"steps":len(history),"seconds":time.perf_counter()-started,
                "first_16_mean_loss":sum(v["loss"] for v in history[:16])/16,"last_16_mean_loss":sum(v["loss"] for v in history[-16:])/16}
        layer_summaries.append(values);stage.event("layer_complete",**values)
        del native_input,native_target,upcoming,x,output;gc.collect();torch.cuda.empty_cache()
        if layer+1 in (12,24,36):
            dev=run_cases(stage,stage.tasks["development"],f"adapted_{layer+1:02d}_development")
            evaluation=run_cases(stage,stage.tasks["evaluation"],f"adapted_{layer+1:02d}_evaluation")
            trajectory[str(layer+1)]={"development":accuracy(dev),"evaluation":accuracy(evaluation),
                                       "comparison_to_original":comparison(references,evaluation)}
            stage.event("trajectory",layers=layer+1,correct=trajectory[str(layer+1)]["evaluation"]["correct"])
    runtime=timed_generation(stage,"adapted")
    native=json.loads((root/"controls.json").read_text())
    final_bytes=model_bytes(stage.model)
    checkpoint={k:v.detach().cpu() for k,v in stage.model.state_dict().items()}
    base.save(root/"compact_model.pt",checkpoint);del checkpoint
    result={"study_id":base.STUDY_ID,"status":"completed","confirmed":False,"scope":base.CONFIG["scope"],
        "fingerprint":stage.plan["scientific_fingerprint"],"controls":native,"trajectory":trajectory,"layers":layer_summaries,
        "adapted_runtime":runtime,"final_parameters":sum(p.numel() for p in stage.model.parameters()),"final_tensor_bytes":final_bytes,
        "tensor_byte_reduction_fraction":1-final_bytes/native["original"]["tensor_bytes"],
        "matched_graph_speedup":runtime["timings"]["cuda_graph"]["tokens_per_second"]/native["original"]["runtime"]["timings"]["cuda_graph"]["tokens_per_second"],
        "matched_hf_speedup":runtime["timings"]["hf"]["tokens_per_second"]/native["original"]["runtime"]["timings"]["hf"]["tokens_per_second"],
        "checkpoint_bytes":(root/"compact_model.pt").stat().st_size,"checkpoint_sha256":sha(root/"compact_model.pt")}
    write(root/"result.json",result)
    return stage.finish(["result.json","compact_model.pt"])


def reload(root):
    base.checked(root,"train")
    stage,fit=start(root,"reload");rotate(stage,fit)
    for i in range(36):compile_layer(stage,fit,i).pack()
    checkpoint=torch.load(root/"compact_model.pt",map_location="cpu",weights_only=True)
    stage.model.load_state_dict(checkpoint,strict=True);del checkpoint
    records=run_cases(stage,stage.tasks["evaluation"],"reloaded_evaluation")
    expected=json.loads((root/"train/adapted_36_evaluation.json").read_text())
    if records!=expected:raise ValueError("Materialized checkpoint changed evaluation logits")
    write(stage.path/"verification.json",{"strict_state_load":True,"all_256_records_exact":True,
        "accuracy":accuracy(records),"model_tensor_bytes":model_bytes(stage.model),"checkpoint_sha256":sha(root/"compact_model.pt")})
    return stage.finish()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("prepare","shakedown","controls","train","reload"))
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT);p.add_argument("--fit-root",type=Path,default=FIT_ROOT)
    p.add_argument("--systems-only",action="store_true");a=p.parse_args()
    result=prepare(a.root,a.fit_root,a.systems_only) if a.stage=="prepare" else {"shakedown":shakedown,"controls":controls,"train":train,"reload":reload}[a.stage](a.root)
    print(json.dumps(result,allow_nan=False),flush=True)
