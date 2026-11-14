"""Disk-streamed teacher boundaries and all-layer functional calibration."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from .jspace_bench_data import DATASETS,fetch,normalize,sha,write
from .jspace_lens import load_small_qwen
from .jspace_bench_tensor import accuracy,comparison,model_bytes,timing


ROOT=Path(__file__).resolve().parents[2]
STUDY_ID="JSPACE-FULL-COMPACT-01"
DEFAULT_ROOT=ROOT/"data/experiments/jspace_full_compact/20261113_capture_v1"
SHAKEDOWN_ROOT=ROOT/"data/experiments/jspace_full_compact/20261113_capture_shakedown_v1"
PARENT=ROOT/"data/experiments/jspace_bench_tensor/20261113_graph_v3"
DATA_ROOT=ROOT/"data/benchmarks/jspace_four_tasks_20261113"
PROTOCOL=ROOT/"docs/07 - Status Reports/jspace/2026-11-13_jspace-full-compact-plan.md"
RECIPE=Path("/home/rabbit/.codex/attachments/3a9e1366-cc2e-4086-92d0-193f8890637b/Pasted text.txt")
CONFIG={"seed":861,"train_per_task":128,"dev_per_task":16,"eval_per_task":64,"max_tokens":384,
        "calibration_per_task":32,"sample_tokens":16,"functional_rank":1024,"anchors":16,
        "removed_per_layer":1024,"rope_pairs":48,"ridge":.01,"adapt_steps":96,"learning_rate":1e-4,
        "temperature":2.,"kd_weight":.7,"ce_weight":.3,"local_weight":.1,"gradient_clip":1.,
        "generation_tokens":32,"shared_relation_nrmse":.5,"layers":list(range(36)),"dtype":"bfloat16",
        "scope":"exploratory_all_layer_compact_component_plus_native_residual"}


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise RuntimeError("CPU stage requires CUDA_VISIBLE_DEVICES=''")


def save(path,value):
    if path.exists():raise FileExistsError(path)
    torch.save(value,path)


def expanded_tasks(tokenizer):
    import pyarrow.parquet as pq
    previous=json.loads((PARENT/"tasks.json").read_text())
    evaluation=previous["splits"]["evaluation"]
    seen={c["identity"] for c in evaluation}
    result={"train":[],"development":[],"evaluation":evaluation}
    manifest=fetch(DATA_ROOT);excluded={}
    for fi,(family,entry) in enumerate(manifest.items()):
        rows=pq.read_table(DATA_ROOT/entry["files"]["train"]["local"]).to_pylist()
        indices=list(range(len(rows)));random.Random(CONFIG["seed"]+100*fi).shuffle(indices)
        accepted=[];skipped=Counter()
        for index in indices:
            question,choices,answer=normalize(family,rows[index])
            identity=hashlib.sha256(json.dumps([question,choices]).encode()).hexdigest()
            if identity in seen:skipped["duplicate_or_fixed_evaluation"]+=1;continue
            user=question+"\n\n"+"\n".join(f"{chr(65+i)}. {choice}" for i,choice in enumerate(choices))
            messages=[{"role":"system","content":"Answer the multiple-choice question. Reply with only the letter of the correct option."},
                      {"role":"user","content":user}]
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
            if len(ids)>CONFIG["max_tokens"]:skipped["long"]+=1;continue
            seen.add(identity)
            accepted.append({"family":family,"source_split":"train","source_row":index,"identity":identity,
                             "prompt":user,"input_ids":ids,"choices":choices,"answer":answer})
            if len(accepted)==CONFIG["train_per_task"]+CONFIG["dev_per_task"]:break
        if len(accepted)!=144:raise ValueError("Insufficient training pool")
        result["train"]+=accepted[:128];result["development"]+=accepted[128:]
        excluded[family]=dict(skipped)
    identities=[c["identity"] for cases in result.values() for c in cases]
    if len(identities)!=len(set(identities)):raise ValueError("Split leakage")
    return {"splits":result,"datasets":manifest,"exclusions":excluded,"scope":CONFIG["scope"],
            "evaluation_reused_from":str(PARENT/"tasks.json")}


def prepare(root,systems_only=False):
    cpu_only()
    from transformers import AutoTokenizer
    if root.exists():raise FileExistsError(root)
    parent=json.loads((PARENT/"plan.json").read_text())
    model=parent["model"]
    tokenizer=AutoTokenizer.from_pretrained(model["path"],local_files_only=True,trust_remote_code=False)
    tasks=expanded_tasks(tokenizer)
    paths=[Path(__file__),PROTOCOL,ROOT/"experiments/structure_net/jspace_bench_data.py",
           ROOT/"experiments/structure_net/jspace_bench_tensor.py",ROOT/"experiments/structure_net/jspace_groebner_tensor.py",
           ROOT/"experiments/structure_net/jspace_lens.py",ROOT/"pixi.lock",ROOT/"pyproject.toml"]
    paths+=sorted((ROOT/"src/structure_net_resources").glob("*.py"))
    value={"study_id":STUDY_ID,"config":CONFIG,"model":model,"label_ids":parent["label_ids"],
           "systems_only":systems_only,"task_digest":digest(tasks),"recipe_sha256":sha(RECIPE),
           "source_sha256":{str(p):sha(p) for p in paths},"created_at":datetime.now(timezone.utc).isoformat(),
           "input_sha256":{str(PARENT/"result.json"):sha(PARENT/"result.json"),str(PARENT/"tasks.json"):sha(PARENT/"tasks.json"),
                            str(DATA_ROOT/"manifest.json"):sha(DATA_ROOT/"manifest.json")}}
    value["scientific_fingerprint"]=digest(value)
    root.mkdir(parents=True)
    for path in paths:
        target=root/"producer"/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    shutil.copy2(RECIPE,root/"user_recipe.txt")
    write(root/"tasks.json",tasks);write(root/"plan.json",value)
    return {"fingerprint":value["scientific_fingerprint"],"systems_only":systems_only,
            "questions":{k:len(v) for k,v in tasks["splits"].items()}}


def verify(root,checkpoint=False):
    plan=json.loads((root/"plan.json").read_text())
    if (plan["config"]!=CONFIG or plan["study_id"]!=STUDY_ID or
            plan["scientific_fingerprint"]!=digest({k:v for k,v in plan.items() if k!="scientific_fingerprint"})):
        raise ValueError("Scientific fingerprint mismatch")
    for name,expected in plan["source_sha256"].items():
        if sha(name)!=expected or sha(root/"producer"/Path(name).relative_to(ROOT))!=expected:raise ValueError("Frozen source changed: "+name)
    for name,expected in plan["input_sha256"].items():
        if sha(name)!=expected:raise ValueError("Frozen input changed: "+name)
    if sha(root/"user_recipe.txt")!=plan["recipe_sha256"]:raise ValueError("Recipe changed")
    if digest(json.loads((root/"tasks.json").read_text()))!=plan["task_digest"]:raise ValueError("Tasks changed")
    if checkpoint:
        for name,expected in plan["model"]["files"].items():
            if sha(Path(plan["model"]["path"])/name)!=expected:raise ValueError("Checkpoint changed")
    return plan


def checked(root,stage):
    plan=verify(root);value=json.loads((root/f"{stage}_manifest.json").read_text())
    if value["status"]!="completed" or value["fingerprint"]!=plan["scientific_fingerprint"]:raise ValueError("Incomplete stage")
    for path,expected in value["artifact_sha256"].items():
        if sha(root/path)!=expected:raise ValueError("Stage artifact changed: "+path)
    return value


class Stage:
    def __init__(self,root,name):
        from structure_net_resources.client import require_gpu_lease
        self.lease=require_gpu_lease([0]);self.guard=require_gpu_lease
        self.plan=verify(root,checkpoint=True);self.root=root;self.name=name;self.path=root/name
        self.path.mkdir(exist_ok=False);self.started=time.perf_counter();self.counts=Counter()
        torch.set_num_threads(4);torch.manual_seed(CONFIG["seed"])
        torch.backends.cuda.matmul.allow_tf32=False
        torch.cuda.reset_peak_memory_stats()
        self.model,self.tokenizer=load_small_qwen(self.plan["model"]["path"],device="cuda:0",dtype=torch.bfloat16)
        self.tasks=json.loads((root/"tasks.json").read_text())["splits"]
        self.label_ids=torch.tensor(self.plan["label_ids"],device="cuda:0")
        self.original_parameters=sum(p.numel() for p in self.model.parameters())
        write(self.path/"lease.json",{k:self.lease[k] for k in ("id","holder","state","ready","allocation","end")})

    def event(self,kind,**values):
        self.counts[kind]+=1
        row={"kind":kind,"elapsed_seconds":time.perf_counter()-self.started,**values}
        with (self.path/"events.jsonl").open("a") as stream:stream.write(json.dumps(row,allow_nan=False)+"\n")
        if self.counts[kind]==1 or self.counts[kind]%8==0 or kind=="layer_complete":
            self.guard([0]);print(json.dumps(row,allow_nan=False),flush=True)

    def finish(self,names=(),**extras):
        self.guard([0]);verify(self.root)
        peak=torch.cuda.max_memory_reserved()
        if peak>sum(self.lease["allocation"].values())*1024**2:raise RuntimeError("Exceeded reservation")
        files=[p for p in self.path.rglob("*") if p.is_file()]+[self.root/n for n in names]
        value={"study_id":STUDY_ID,"status":"completed","stage":self.name,"fingerprint":self.plan["scientific_fingerprint"],
               "elapsed_seconds":time.perf_counter()-self.started,"peak_reserved_bytes":peak,"lease_id":self.lease["id"],
               "original_parameters":self.original_parameters,"gpu":torch.cuda.get_device_name(0),"torch":str(torch.__version__),
               "counts":dict(self.counts),"artifact_sha256":{str(p.relative_to(self.root)):sha(p) for p in files},**extras}
        write(self.root/f"{self.name}_manifest.json",value)
        return {k:v for k,v in value.items() if k!="artifact_sha256"}


def write_bits(array,index,tensor):
    array[index]=tensor.detach().contiguous().view(torch.int16).cpu().numpy()


def read_bits(array,index,device="cuda:0"):
    return torch.from_numpy(np.array(array[index],copy=True)).view(torch.bfloat16).to(device)


def capture(root):
    stage=Stage(root,"capture")
    items=[{**case,"split":split} for split in ("train","development") for case in stage.tasks[split]]
    if stage.plan["systems_only"]:
        items=[max([c for c in items if c["family"]==f and c["split"]=="train"],key=lambda c:len(c["input_ids"])) for f in DATASETS]+sorted(
            [c for c in items if c["split"]=="development"],key=lambda c:len(c["input_ids"]),reverse=True)[:2]
    calibration=[]
    for f in DATASETS:calibration += [i for i,c in enumerate(items) if c["family"]==f and c["split"]=="train"][:CONFIG["calibration_per_task"]]
    cal_map={i:j for j,i in enumerate(calibration)}
    dev=[i for i,c in enumerate(items) if c["split"]=="development"];dev_map={i:j for j,i in enumerate(dev)}
    offsets=[0]
    for c in items:offsets.append(offsets[-1]+len(c["input_ids"]))
    shape=(offsets[-1],2048)
    boundary=[np.lib.format.open_memmap(stage.path/f"boundary_{i:02d}.npy",mode="w+",dtype=np.int16,shape=shape) for i in range(37)]
    acts={split:[np.lib.format.open_memmap(stage.path/f"mlp_{split}_{i:02d}.npy",mode="w+",dtype=np.int16,
        shape=(count*16,11008)) for i in range(36)] for split,count in (("calibration",len(calibration)),("development",len(dev)))}
    jac=[np.lib.format.open_memmap(stage.path/f"mlp_jacobian_{i:02d}.npy",mode="w+",dtype=np.float32,
         shape=(len(calibration)*2,11008)) for i in range(36)]
    logits=np.lib.format.open_memmap(stage.path/"teacher_logits.npy",mode="w+",dtype=np.int16,shape=(len(items),4,151936))
    grams=torch.zeros(36,2048,2048,device="cuda:0")
    means=torch.zeros(36,2048,device="cuda:0")
    pair_energy=torch.zeros(36,2,64,device="cuda:0");group_energy=torch.zeros(36,2,device="cuda:0")
    value_sum=torch.zeros(36,2,128,device="cuda:0");value_cov=torch.zeros(36,2,128,128,device="cuda:0")
    metadata=[];records=[]
    for index,case in enumerate(items):
        states={};hidden_mlp={};projections={};handles=[];gradient=index in cal_map
        for layer,block in enumerate(stage.model.model.layers):
            def boundary_hook(module,args,layer=layer):
                if gradient and layer==0:args[0].requires_grad_(True)
                states[layer]=args[0]
            handles.append(block.register_forward_pre_hook(boundary_hook))
            handles.append(block.mlp.down_proj.register_forward_pre_hook(lambda m,a,layer=layer:hidden_mlp.__setitem__(layer,a[0])))
            if gradient:
                for kind in ("q","k","v"):
                    handles.append(getattr(block.self_attn,kind+"_proj").register_forward_hook(
                        lambda m,a,o,layer=layer,kind=kind:projections.__setitem__((layer,kind),o)))
        handles.append(stage.model.model.norm.register_forward_pre_hook(lambda m,a:states.__setitem__(36,a[0])))
        ids=torch.tensor([case["input_ids"]],device="cuda:0")
        positions=torch.linspace(0,len(case["input_ids"])-1,16,device="cuda:0").long()
        loss_positions=torch.cat((torch.linspace(0,len(case["input_ids"])-2,3,device="cuda:0").long(),
                                   torch.tensor([len(case["input_ids"])-1],device="cuda:0")))
        try:
            with torch.set_grad_enabled(gradient):
                output=stage.model.model(input_ids=ids,use_cache=False).last_hidden_state
                choice=F.linear(output[:,-1],stage.model.lm_head.weight[stage.label_ids[:len(case["choices"])]]).float()[0]
                if not torch.isfinite(choice).all():raise FloatingPointError("Nonfinite teacher")
                if gradient:
                    others=[i for i in range(len(choice)) if i!=case["answer"]]
                    target=choice[case["answer"]]-choice[others].mean()
                    tensors=[states[i] for i in range(36)]+[hidden_mlp[i] for i in range(36)]+[
                        projections[i,k] for i in range(36) for k in ("q","k","v")]
                    derivatives=torch.autograd.grad(target,tensors)
            with torch.no_grad():
                teacher=F.linear(output[0,loss_positions].detach(),stage.model.lm_head.weight)
                write_bits(logits,index,teacher)
                for layer in range(37):write_bits(boundary[layer],slice(offsets[index],offsets[index+1]),states[layer][0])
                if gradient:
                    row=cal_map[index]
                    for layer in range(36):
                        g=derivatives[layer][0,positions].float()
                        if not torch.isfinite(g).all():raise FloatingPointError("Nonfinite task Jacobian")
                        grams[layer].add_(g.T@g);means[layer].add_(states[layer][0,positions].float().sum(0))
                        gm=derivatives[36+layer][0].float()
                        jac[layer][2*row:2*row+2]=torch.stack((gm[-1],gm.mean(0))).cpu().numpy()
                        write_bits(acts["calibration"][layer],slice(row*16,(row+1)*16),hidden_mlp[layer][0,positions])
                        q,k,v=[projections[layer,key][0,positions].float() for key in ("q","k","v")]
                        gq,gk,gv=[derivatives[72+3*layer+j][0,positions].float() for j in range(3)]
                        importance=(q*gq).reshape(16,2,8,128).square().sum((0,2))+(k*gk).reshape(16,2,128).square().sum(0)
                        pair_energy[layer].add_(importance[:,:64]+importance[:,64:])
                        group_energy[layer].add_(importance.sum(1)+(v*gv).reshape(16,2,128).square().sum((0,2)))
                        vv=v.reshape(16,2,128).transpose(0,1)
                        value_sum[layer].add_(vv.sum(1));value_cov[layer].add_(vv.transpose(1,2)@vv)
                if index in dev_map:
                    row=dev_map[index]
                    for layer in range(36):write_bits(acts["development"][layer],slice(row*16,(row+1)*16),hidden_mlp[layer][0,positions])
                targets=[case["input_ids"][int(p)+1] for p in loss_positions[:-1]]+[int(stage.label_ids[case["answer"]])]
                metadata.append({**case,"offset":offsets[index],"stop":offsets[index+1],"loss_positions":loss_positions.cpu().tolist(),"targets":targets})
                records.append({"identity":case["identity"],"family":case["family"],"answer":case["answer"],
                                "prediction":int(choice.argmax()),"logits":choice.cpu().tolist()})
        finally:
            for handle in handles:handle.remove()
        stage.event("teacher_capture",completed=index+1,total=len(items),gradient=gradient)
        del states,hidden_mlp,projections,output,teacher,choice
        if gradient:del derivatives,tensors,target
        if (index+1)%32==0:
            for array in boundary:array.flush()
            logits.flush();gc.collect()
    for arrays in (boundary,jac,*acts.values()):
        for array in arrays:array.flush()
    logits.flush()
    count=len(calibration)*16
    save(stage.path/"statistics.pt",{"grams":grams.cpu(),"mean":(means/count).cpu(),"pair_energy":pair_energy.cpu(),
        "group_energy":group_energy.cpu(),"value_sum":value_sum.cpu(),"value_cov":value_cov.cpu(),"count":count})
    write(stage.path/"items.json",{"items":metadata,"calibration_indices":calibration,"development_indices":dev,
        "calibration_families":[items[i]["family"] for i in calibration for _ in range(16)],
        "development_families":[items[i]["family"] for i in dev for _ in range(16)],"tokens":offsets[-1],
        "storage":"memory-mapped signed int16 arrays preserving BF16 bits"})
    write(stage.path/"teacher_records.json",records)
    # Check the on-disk boundary/logit representation can be read without a dtype conversion.
    reloaded=read_bits(np.load(stage.path/"teacher_logits.npy",mmap_mode="r"),0,device="cpu")
    if not torch.isfinite(reloaded).all():raise FloatingPointError("Teacher cache reload failed")
    return stage.finish(systems_only=stage.plan["systems_only"],scope="systems_only" if stage.plan["systems_only"] else CONFIG["scope"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","capture"));parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--systems-only",action="store_true")
    args=parser.parse_args()
    value=prepare(args.root,args.systems_only) if args.stage=="prepare" else capture(args.root)
    print(json.dumps(value,allow_nan=False),flush=True)
