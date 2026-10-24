"""Full690 exploratory frozen-body NCM training with audited, possibly noisy labels."""
import argparse
from dataclasses import asdict,dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_lm1b_grounded_head as head
from experiments.structure_net import ncm_lm1b_neighborhood_loss as objective
from experiments.structure_net import ncm_lm1b_capacity_labels as labels
from experiments.structure_net.ncm_lm1b_decoder_repair import same
from experiments.structure_net.ncm_memory_data import file_hash,write_json

native,q,ROOT=head.concepts.release,head.concepts.release.q,head.ROOT
HYPOTHESIS="ncm-lm1b-full690-source-grounded-neighborhood-v1"
MODULE="experiments.structure_net.ncm_lm1b_grounded_training"
PLAN=Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-grounded-training-plan.md")
WORKFLOW=ROOT/"20261024_lm1b_grounded_training_workflow_v1"
DATA=ROOT/"20261024_lm1b_grounded_training_data_v1"
CACHE_PILOT=ROOT/"20261024_lm1b_grounded_training_cache_pilot_v1"
CACHE=ROOT/"20261024_lm1b_grounded_training_cache_v1"
PILOT=ROOT/"20261024_lm1b_grounded_training_pilot_v1"
TRAIN=ROOT/"20261024_lm1b_grounded_training_train_v1"
LABEL_DATA=ROOT/"20261024_lm1b_capacity_revision_data_v1"
LABEL_DRAFT=ROOT/"20261024_lm1b_capacity_revision_draft_v1"
MANIFEST=ROOT/"20261023_reviewed_draft_data_v1/manifest.json"
DEV=ROOT/"20261024_pinned_date_development_sources_v1/cases.json"
SYSTEMS=ROOT/"20261024_lm1b_neighborhood_cuda_pilot_v1"


@dataclass(frozen=True)
class Config:
    updates:int=4096
    microbatch:int=4
    accumulation:int=4
    warmup:int=128
    max_lr:float=1e-3
    min_lr:float=1e-4
    sampler_seed:int=95142
    neighbor_seed:int=96142
    dropout_seed:int=94142
    self_history_start:int=256
    self_history_interval:int=4


def rate(step):
    c=Config()
    if not 1<=step<=c.updates:raise ValueError("step outside fixed budget")
    if step<=c.warmup:return c.max_lr*step/c.warmup
    fraction=(step-c.warmup)/(c.updates-c.warmup)
    return c.min_lr+(c.max_lr-c.min_lr)*.5*(1+math.cos(math.pi*fraction))


def disjoint(train,dev):
    parts=[train,*[dev[k] for k in ("short_development","long_diagnostic")]]
    if [len(v) for v in parts]!=[690,64,64]:raise ValueError("full690/64/64 populations required")
    fields=("id","source_id","group_id","source_sha256")
    proof={}
    for key in fields:
        sets=[{r[key] for r in part} for part in parts]
        if any(len(s)!=len(p) for s,p in zip(sets,parts)) or any(sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2))):
            raise ValueError("TRAIN and permanent task DEV identity/session/group/source overlap")
        proof[key]={"train_unique":len(sets[0]),"short_unique":len(sets[1]),"long_unique":len(sets[2]),"overlap":0}
    return proof


def producing(root,stage,inputs=(),config=None):
    files=[Path(__file__),PLAN,Path(head.__file__),Path(objective.__file__),
        Path("tests/structure_net/test_ncm_lm1b_grounded_training.py"),
        Path("experiments/structure_net/ncm_lm1b_source_attention.py"),Path(labels.__file__)]
    common=[native.CASES,MANIFEST,DEV,head.DOMAIN_CHECKPOINT,SYSTEMS/"producer.json",SYSTEMS/"result.json",
        SYSTEMS/"worker_terminal_receipt.json",LABEL_DRAFT/"outputs.json",LABEL_DRAFT/"result.json",LABEL_DRAFT/"worker_terminal_receipt.json",*files]
    cfg={**asdict(Config()),**(config or {}),"hypothesis_id":HYPOTHESIS,"schema":"nal.ncm-lm1b-full690-grounded-neighborhood.v1",
        "conformance":"EXPLORATORY non-rescuing end-to-end training; prior codec and label gates failed",
        "potentially_noisy_TRAIN_labels_allowed":True,"strict_label_qualification_claim":False,
        "label_choice":"latest genuine capacity35 candidates, all retained regardless semantic verdict;655 original texts unchanged",
        "all_TRAIN_cases":690,"all_task_DEV_cases":128,"all_final_sources_closed":True,
        "old_body_and_MLPs_loaded":False,"old_body_and_MLPs_updated":False,"source_features":"full cached FP32 frozen-body features",
        "native_generator_and_domain_codec_frozen":True,"new_trainable_parameters":11796480,
        "model_training_executed":stage in ("full_objective_training_pilot","fixed4096_full690_training"),
        "precision":"FP32weights/BF16training;FP32source-only self-history and evaluation", "TF32":False,
        "task_accuracy":None,"source_only_accuracy_measured":False,"speed_tested":False,"final_test_opened":False,
        "checkpoint_selection":"fixed final4096 only","all_cases_equal_loss_weight":True,
        "loss":"equal12levelCE;clean.5concept+.7word,noisy.5concept+.3word;self.25*(concept+word) firstmicrobatch every4after256",
        "corruption":"root protected;per-case uniform rate0to.25 ramp over1024updates;uniform8192replacement"}
    p=native.producing(root,stage,[*inputs,*common],cfg);p["hypothesis_id"]=HYPOTHESIS;p["config"].update(cfg)
    for i,f in enumerate(files):
        p["source_hashes"][str(f)]=file_hash(f);(root/"producer_source"/f"grounded_training_{i}_{f.name}").write_bytes(f.read_bytes())
    p["fingerprint"]=q.digest({k:v for k,v in p.items() if k!="fingerprint"});write_json(root/"producer.json",p);return p


def prepare():
    labels.configure()
    import tiktoken
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or torch.cuda.is_initialized():raise ValueError("CPU data preparation only")
    native.vendor.terminal(LABEL_DRAFT)
    if file_hash(native.CASES)!=native.CASES_SHA:raise ValueError("original690 source dataset changed")
    rows=q.read(native.CASES)["train"];proposals=q.read(LABEL_DRAFT/"outputs.json")
    selected={r["id"] for r in q.read(LABEL_DATA/"cases.json")}
    if (len(proposals)!=35 or {r["id"] for r in proposals}!=selected
            or any(r["status"]!="valid" or not r["capacity"]["fits"] for r in proposals)):
        raise ValueError("all35 schema-valid capacity-compatible latest candidates required")
    merged=labels.merge(rows,{r["id"]:r["text"] for r in proposals})
    manifest=q.read(MANIFEST);proof=disjoint(merged,q.read(DEV));t=native.tokenizer();storage=tiktoken.get_encoding("cl100k_base")
    for original,row in zip(rows,merged):
        if any(row[k]!=v for k,v in original.items() if k not in ("word_ids","concept_ids","target_text")):
            raise ValueError("source/context/alignment changed with label revision")
        cap=labels.capacity(row["target_text"],t,storage)
        if not cap["fits"]:raise ValueError("complete native target cannot be clipped")
        ids=t.encode(row["target_text"],add_special_tokens=False)
        row["native_word_ids"]=[101,*ids,101]+[0]*(126-len(ids))
        row["target_origin"]="latest_capacity35_candidate_potentially_noisy" if row["id"] in selected else "original_approved_target_unchanged"
        if row["id"] not in manifest:raise ValueError("missing full-source feature")
    if len(merged)!=690 or sum(a["target_text"]!=b["target_text"] for a,b in zip(rows,merged))!=35:
        raise ValueError("all690 cases,exactly35 new texts and655 unchanged required")
    labels.reference.verify_public(merged)
    pilot=[merged[0]["id"],min(merged,key=lambda r:(-manifest[r["id"]]["source_tokens"],r["id"]))["id"]]
    p=producing(DATA,"complete690_potentially_noisy_native_targets",[LABEL_DATA/"cases.json"],{"CPU_only":True})
    write_json(DATA/"cases.json",merged);write_json(DATA/"manifest.json",manifest)
    write_json(DATA/"split_proof.json",proof);write_json(DATA/"selection.json",{"pilot_ids":pilot})
    native.vendor.verify(p)
    write_json(DATA/"result.json",{"status":"all690_capacity_fit_source_disjoint_targets_prepared","cases":690,
        "strict_semantic_label_qualification_claim":False,"label_noise_acknowledged":True,"source_disjoint":True,
        "hashes":{n:file_hash(DATA/n) for n in ("cases.json","manifest.json","split_proof.json","selection.json")},"producer_fingerprint":p["fingerprint"]})


def cache(pilot=False):
    native.require_gpu();native.vendor.terminal(DATA)
    rows=q.read(DATA/"cases.json");ids=q.read(DATA/"selection.json")["pilot_ids"]
    chosen=[next(r for r in rows if r["id"]==i) for i in ids] if pilot else rows
    root=CACHE_PILOT if pilot else CACHE;extra=[]
    if not pilot:
        native.vendor.terminal(CACHE_PILOT);extra=[CACHE_PILOT/n for n in ("outputs.json","result.json","worker_terminal_receipt.json")]
    p=producing(root,"native_target_cache_pilot" if pilot else "all690_fresh_native_code_hierarchies",
        [DATA/n for n in ("cases.json","result.json","worker_terminal_receipt.json")]+extra)
    codec=native.load_codec();before=native.vendor.module_hash(codec);values=[]
    originals={r["id"]:r for r in q.read(ROOT/"20261024_lm1b_framed_codec_reconstruction_v1/outputs.json")}
    prior={r["id"]:r for r in q.read(CACHE_PILOT/"outputs.json")} if not pilot else {}
    with torch.no_grad():
        for row in chosen:
            codes=codec.encode(torch.tensor([row["native_word_ids"]],device="cuda",dtype=torch.long)).cpu().tolist()[0]
            if len(codes)!=sum(native.LEVELS):raise ValueError("complete native641 target hierarchy required")
            value={"id":row["id"],"codes":codes}
            if row["target_origin"]=="original_approved_target_unchanged" and [codes]!=originals[row["id"]]["codes"]:
                raise ValueError("unchanged655 native target hierarchy changed")
            if row["id"] in prior and value!=prior[row["id"]]:raise ValueError("cache pilot fresh-load parity differs")
            values.append(value)
            if pilot or len(values)%32==0 or len(values)==len(chosen):
                write_json(root/"outputs.json",values);print(json.dumps({"cached":len(values),"total":len(chosen)}),flush=True)
    if native.vendor.module_hash(codec)!=before:raise ValueError("frozen native target encoder/books changed")
    native.vendor.verify(p)
    write_json(root/"result.json",{"status":"native_target_cache_measured","cases":len(values),"codec_hash":before,
        "outputs_sha256":file_hash(root/"outputs.json"),"task_accuracy":None,"producer_fingerprint":p["fingerprint"]})


def require_gpu():
    if (os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or os.environ.get("CUDA_DEVICE_ORDER")!="PCI_BUS_ID"
            or not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name()
            or not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info()[0]<8_000_000_000):
        raise ValueError("physicalGPU0 RTX3060 nativeBF16 with8GBfree required")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def load_features(rows,manifest):
    available=int(next(v.split()[1] for v in Path("/proc/meminfo").read_text().splitlines() if v.startswith("MemAvailable:")))*1024
    size=sum(manifest[r["id"]]["source_tokens"] for r in rows)*3072*4
    if size>8_000_000_000 or available<12_000_000_000:raise ValueError("bounded8GBfeature cache and12GBavailable host memory required")
    features={}
    for row in rows:
        e=manifest[row["id"]]
        if file_hash(e["path"])!=e["sha256"]:raise ValueError("frozen full-source feature bytes changed")
        value=torch.load(e["path"],map_location="cpu",weights_only=True)
        if (value["id"]!=row["id"] or value["source_sha256"]!=row["source_sha256"]
                or value["native_prompt_sha256"]!=row["native_prompt_sha256"]
                or value["features"].shape!=(e["source_tokens"],3072) or value["features"].dtype!=torch.float32
                or not torch.isfinite(value["features"]).all()):raise ValueError("source-only feature identity/shape/precision differs")
        features[row["id"]]=value["features"].detach().clone()
    return features


def batch(rows,indices,features,codes):
    chosen=[rows[i] for i in indices];length=max(len(features[r["id"]]) for r in chosen)
    source=torch.stack([F.pad(features[r["id"]],(0,0,0,length-len(features[r["id"]]))) for r in chosen]).cuda()
    mask=torch.stack([torch.arange(length)<len(features[r["id"]]) for r in chosen]).cuda()
    target_codes=torch.tensor([codes[r["id"]] for r in chosen],device="cuda",dtype=torch.long)
    words=torch.tensor([r["native_word_ids"] for r in chosen],device="cuda",dtype=torch.long)
    return source,mask,target_codes,words


@torch.no_grad()
def probes(model,rows,ids,features,codes):
    mode=model.training;model.eval();result={}
    for identity in ids:
        i=next(i for i,r in enumerate(rows) if r["id"]==identity)
        f,m,_,_=batch(rows,[i],features,codes)
        result[identity]={k:v.cpu().tolist() for k,v in model.generate(f,m).items()}
    model.train(mode);return result


def optimizer(model):
    return torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=Config().max_lr,betas=(.9,.95),eps=1e-8,weight_decay=.01)


def attention_hash(model):
    digest=hashlib.sha256()
    for name,value in sorted((n,v) for n,v in model.named_parameters() if v.requires_grad):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save(path,model,opt,sampler,neighbor,step,p):
    state={"attention":model.attention_state(),"optimizer":opt.state_dict(),"sampler_rng":sampler.get_state(),
        "neighbor_rng":neighbor.get_state(),"CPU_rng":torch.get_rng_state(),"CUDA_rng":torch.cuda.get_rng_state_all(),
        "step":step,"config":asdict(Config()),"producer_fingerprint":p["fingerprint"]}
    tmp=path.with_suffix(".tmp");torch.save(state,tmp);tmp.replace(path)


def train(pilot=False):
    require_gpu();native.vendor.terminal(DATA);native.vendor.terminal(CACHE);native.vendor.terminal(SYSTEMS)
    if not q.read(SYSTEMS/"result.json")["clean_noisy_and_label_free_selfhistory_backward_executed"]:
        raise ValueError("actual full-objective representative CUDA lifecycle required")
    if not pilot:
        native.vendor.terminal(PILOT)
        if q.read(PILOT/"result.json")["status"]!="systems_full690_training_reload_passed":raise ValueError("actual data-path training pilot required")
    rows,manifest=q.read(DATA/"cases.json"),q.read(DATA/"manifest.json")
    codes={r["id"]:r["codes"] for r in q.read(CACHE/"outputs.json")}
    if len(rows)!=690 or set(codes)!={r["id"] for r in rows}:raise ValueError("complete690 code/data population required")
    root=PILOT if pilot else TRAIN
    inputs=[DATA/n for n in ("producer.json","result.json","cases.json","manifest.json","worker_terminal_receipt.json")]
    inputs+=[CACHE/n for n in ("producer.json","result.json","outputs.json","worker_terminal_receipt.json")]
    inputs+=[Path(manifest[r["id"]]["path"]) for r in rows]
    if not pilot:inputs+=[PILOT/n for n in ("result.json","initial_probes.json","worker_terminal_receipt.json")]
    p=producing(root,"full_objective_training_pilot" if pilot else "fixed4096_full690_training",inputs,
        {"pilot":pilot,"actual_updates":2 if pilot else Config().updates,"actual_accumulation":1 if pilot else Config().accumulation,
        "peak_limit_bytes":7_500_000_000,"systems_pilot_forces_max_noise_and_selfhistory":pilot})
    features=load_features(rows,manifest);model=head.from_domain(device="cuda").train()
    frozen={"generator":native.vendor.module_hash(model.concepts.generator),"codec":native.vendor.module_hash(model.codec)}
    if frozen!={"generator":"60316ea14c56e6276be581112a2a1d81437e3ad8bb4ea3ddf8a98b946805e26a","codec":head.DOMAIN_CODEC_HASH}:
        raise ValueError("exact measured frozen native states required")
    initial_attention=attention_hash(model)
    if sum(v.numel() for v in model.parameters() if v.requires_grad)!=11796480:raise ValueError("only new11.8Mattention parameters may train")
    ids=q.read(DATA/"selection.json")["pilot_ids"];initial=probes(model,rows,ids,features,codes)
    if not pilot and initial!=q.read(PILOT/"initial_probes.json"):raise ValueError("full training must reset to fresh zero-output initialization")
    write_json(root/"initial_probes.json",initial)
    sampler=torch.Generator().manual_seed(Config().sampler_seed);neighbor=torch.Generator().manual_seed(Config().neighbor_seed)
    torch.manual_seed(Config().dropout_seed);torch.cuda.manual_seed_all(Config().dropout_seed)
    opt=optimizer(model);updates=2 if pilot else Config().updates;accumulation=1 if pilot else Config().accumulation
    torch.cuda.reset_peak_memory_stats()
    with (root/"trace.jsonl").open("x") as trace:
        for step in range(1,updates+1):
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group["lr"]=rate(step)
            micro=[]
            for index in range(accumulation):
                indices=torch.randint(len(rows),(Config().microbatch,),generator=sampler).tolist()
                if pilot:indices=[next(i for i,r in enumerate(rows) if r["id"]==ids[j%2]) for j in range(4)]
                f,m,c,w=batch(rows,indices,features,codes)
                use_history=pilot or (step>Config().self_history_start and step%Config().self_history_interval==0 and index==0)
                losses=objective.backward_microbatch(model,f,m,c,w,neighbor,1024 if pilot else step,
                    accumulation=accumulation,self_history=use_history)
                micro.append({"case_ids":[rows[i]["id"] for i in indices],"source_positions":f.shape[1],"losses":losses})
            if any(v.grad is not None for module in (model.codec,model.concepts.generator) for v in module.parameters()):
                raise ValueError("frozen pretrained weights received gradients")
            if pilot and step==2:
                active=[n for n,v in model.named_parameters() if v.requires_grad and v.grad is not None and v.grad.abs().sum()>0]
                if any(not any(n.startswith(prefix) for n in active) for prefix in
                        ("concepts.source_memory.","concepts.adapters.","word_memory.","word_adapters.")):
                    raise ValueError("new data-path source attention gradients inactive")
            norm=torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.)
            if not torch.isfinite(norm):raise ValueError("nonfinite full690 training gradients")
            opt.step();trace.write(json.dumps({"step":step,"lr":rate(step),"gradient_norm":float(norm),"microbatches":micro},allow_nan=False)+"\n");trace.flush()
            if pilot or step%32==0:print(json.dumps({"step":step,"updates":updates,"clean_word_loss":sum(v["losses"]["clean_word_loss"] for v in micro)/len(micro)}),flush=True)
            if not pilot and step%1024==0 and step<updates:save(root/f"recovery_step{step}.pt",model,opt,sampler,neighbor,step,p)
    if any(native.vendor.module_hash(module)!=frozen[k] for k,module in (("generator",model.concepts.generator),("codec",model.codec))):
        raise ValueError("frozen native generator/domain codec state drift")
    final_attention=attention_hash(model)
    if initial_attention==final_attention:raise ValueError("new attention did not update")
    before=probes(model,rows,ids,features,codes);save(root/"model.pt",model,opt,sampler,neighbor,updates,p)
    model.close();del model,opt;gc.collect();torch.cuda.empty_cache()
    saved=torch.load(root/"model.pt",map_location="cpu",weights_only=True,mmap=True)
    model=head.from_domain(device="cuda");model.load_attention_state(saved["attention"]);opt=optimizer(model);opt.load_state_dict(saved["optimizer"])
    sampler.set_state(saved["sampler_rng"]);neighbor.set_state(saved["neighbor_rng"])
    torch.set_rng_state(saved["CPU_rng"]);torch.cuda.set_rng_state_all(saved["CUDA_rng"])
    if (probes(model,rows,ids,features,codes)!=before or not same(model.attention_state(),saved["attention"])
            or not same(opt.state_dict(),saved["optimizer"]) or not torch.equal(sampler.get_state(),saved["sampler_rng"])
            or not torch.equal(neighbor.get_state(),saved["neighbor_rng"]) or not torch.equal(torch.get_rng_state(),saved["CPU_rng"])
            or not same(torch.cuda.get_rng_state_all(),saved["CUDA_rng"])):raise ValueError("full data-path model/optimizer/allRNG/source-only reload failed")
    peak=torch.cuda.max_memory_allocated()
    if peak>7_500_000_000:raise ValueError("full690 source-attention training exceeds7.5GBplan")
    write_json(root/"reload_probes.json",before);native.vendor.verify(p)
    result={"status":"systems_full690_training_reload_passed" if pilot else "source_grounded_NCM_trained_source_only_task_accuracy_pending",
        "updates":updates,"training_cases":690,"new_trainable_parameters":11796480,"frozen_native_states_unchanged":frozen,
        "initial_attention_sha256":initial_attention,"final_attention_sha256":final_attention,"new_attention_updated":True,
        "exact_attention_optimizer_sampler_neighbor_CPU_CUDA_reload":True,"source_only_reload_exact":True,
        "peak_allocated_bytes":peak,"checkpoint_sha256":file_hash(root/"model.pt"),"label_noise_acknowledged":True,
        "task_accuracy":None,"final_test_opened":False,"speed_tested":False,"producer_fingerprint":p["fingerprint"]}
    write_json(root/"result.json",result);print(json.dumps(result),flush=True)


def workflow():
    if any(p.exists() for p in (WORKFLOW,DATA,CACHE_PILOT,CACHE,PILOT,TRAIN)):raise ValueError("fresh full690 NCM workflow required")
    p=producing(WORKFLOW,"full690_exploratory_source_grounded_training")
    (WORKFLOW/"stage_receipts").mkdir();journal={"status":"running","observer_pid":os.getpid(),"stages":[],"producer_fingerprint":p["fingerprint"]}
    try:
        for stage,root,gpu in (("prepare",DATA,""),("cache-pilot",CACHE_PILOT,"2"),("cache",CACHE,"2"),("pilot",PILOT,"0"),("train",TRAIN,"0")):
            native.vendor.verify(p);command=[sys.executable,"-u","-m",MODULE,stage]
            child=subprocess.Popen(command,env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu});journal.update(active_stage=stage,child_pid=child.pid);write_json(WORKFLOW/"progress.json",journal)
            code=child.wait();receipt={"stage":stage,"exit_code":code,"child_pid":child.pid,"observer_pid":os.getpid(),"observer":"subprocess.wait","command":command}
            write_json(WORKFLOW/"stage_receipts"/(stage+".json"),receipt)
            if root.exists():write_json(root/"worker_terminal_receipt.json",receipt)
            journal["stages"].append(receipt)
            if code:raise RuntimeError("full690 source-grounded stage failed:"+stage)
        native.vendor.verify(p);journal.update(status="full690_source_grounded_training_completed_accuracy_pending",active_stage=None,child_pid=None);write_json(WORKFLOW/"result.json",journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved",error_type=type(error).__name__,error=str(error));write_json(WORKFLOW/"failure.json",journal);raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("stage",choices=("prepare","cache-pilot","cache","pilot","train","workflow"));args=parser.parse_args()
    {"prepare":prepare,"cache-pilot":lambda:cache(True),"cache":cache,"pilot":lambda:train(True),"train":train,"workflow":workflow}[args.stage]()
