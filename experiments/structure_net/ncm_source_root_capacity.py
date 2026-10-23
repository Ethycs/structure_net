"""Direct TRAIN-only source-to-root capacity experiment after archived quality failure."""
import argparse
from dataclasses import dataclass,asdict
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_word_final_generation_gap as gap
from experiments.structure_net.ncm_memory_data import file_hash,write_json

study=gap.study
PLAN=Path("docs/07 - Status Reports/2026-10-23_ncm-source-root-capacity-plan.md")
HYPOTHESIS="ncm-word-final-source-root-capacity-v1"
PILOT=study.data.ROOT/"20261023_word_final_root_capacity_pilot_v1"
TRAIN=study.data.ROOT/"20261023_word_final_root_capacity_train_v1"
COMPLETED=study.data.ROOT/"20261023_word_final_results_v1"


@dataclass(frozen=True)
class Config:
    updates:int=256
    warmup:int=16
    max_lr:float=3e-4
    min_lr:float=3e-5
    model_seed:int=13143
    sampler_seed:int=14143
    pad_weight:float=.25


def rate(step,config=Config()):
    if not 1<=step<=config.updates:raise ValueError("step outside fixed capacity budget")
    if step<=config.warmup:return config.max_lr*step/config.warmup
    return config.min_lr+(config.max_lr-config.min_lr)*.5*(1+math.cos(math.pi*(step-config.warmup)/(config.updates-config.warmup)))


def scope(net):
    net.requires_grad_(False)
    net.source_head.requires_grad_(True)
    for head in net.generator.lm_heads[1:]:head.requires_grad_(False)
    return [p for p in net.parameters() if p.requires_grad]


def root_logits(net,features,source_mask,source_ids):
    source=net.source_head.source_latents(features,source_mask,source_ids)
    prefix=net.generator.transformer.wte(source)
    query=net.generator.build_l0_query_input(len(source),dtype=source.dtype)
    hidden=net._level_hidden(prefix,source_mask,query,0,0)
    return net.generator.lm_heads[0](hidden)


def active_mask(words,root_width):
    if words.ndim!=2 or words.shape[1]!=192 or root_width!=16:
        raise ValueError("complete four-block words and16root positions required")
    return words.reshape(len(words),4,48).ne(50257).any(-1).repeat_interleave(4,dim=1)


def loss(logits,labels,active,config=Config()):
    if logits.shape[:2]!=labels.shape or labels.shape!=active.shape or not active.any():
        raise ValueError("complete root labels and active blocks required")
    values=F.cross_entropy(logits.float().flatten(0,1),labels.flatten(),reduction="none").reshape_as(labels)
    primary=values[active].mean()
    padded=values[~active].mean() if (~active).any() else primary*0
    return primary+config.pad_weight*padded,{"active_CE":float(primary.detach()),"padded_CE":float(padded.detach())}


def frozen_hash(net):
    return study.recovery.base.tensor_digest({k:v for k,v in net.named_parameters() if not v.requires_grad})


@torch.no_grad()
def evaluate(net,rows,manifest):
    previous=net.training;net.eval();outputs=[]
    for row in rows:
        batch=study.recovery.base.batch_for(row,manifest)
        study.set_source_frame(net.generator,row["source_frame"],maximum=8192)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits=root_logits(net,batch["features"],batch["mask"],batch["source_ids"])
        if not torch.isfinite(logits).all():raise ValueError("nonfinite root inference")
        outputs.append({"id":row["id"],"codes":logits.argmax(-1)[0].cpu().tolist()})
    net.train(previous)
    counts={"correct":0,"total":0,"active_correct":0,"active_total":0,"shuffled_active_correct":0}
    for i,(row,prediction) in enumerate(zip(rows,outputs)):
        labels=torch.tensor(row["concept_ids"][:16]);actual=torch.tensor(prediction["codes"])
        mask=active_mask(torch.tensor([row["word_ids"]]),16)[0]
        shuffled=torch.tensor(outputs[(i+1)%len(outputs)]["codes"])
        counts["correct"]+=int((actual==labels).sum());counts["total"]+=16
        counts["active_correct"]+=int(((actual==labels)&mask).sum());counts["active_total"]+=int(mask.sum())
        counts["shuffled_active_correct"]+=int(((shuffled==labels)&mask).sum())
    counts.update(all_root_recovery=counts["correct"]/counts["total"],
        active_root_recovery=counts["active_correct"]/counts["active_total"],
        shuffled_active_recovery=counts["shuffled_active_correct"]/counts["active_total"])
    return {"counts":counts,"outputs":outputs,"task_accuracy":None}


def producing(root,pilot,rows,manifest):
    inputs=[COMPLETED/n for n in ("campaign_results.json","ncm-word-final-frozen-parent-v1_registry_readback.json")]
    inputs += [gap.DEST/n for n in ("producer.json","result.json","selection.json","case_measurements.json","worker_terminal_receipt.json")]
    inputs += [study.data.DATA/"cases.json",study.data.FEATURES/"manifest.json"]
    inputs += [study.recovery.TRAIN/n for n in ("producer.json","result.json","model.pt","worker_terminal_receipt.json")]
    if not pilot:inputs += [PILOT/n for n in ("producer.json","result.json","worker_terminal_receipt.json")]
    for row in rows:
        entry=manifest[row["id"]]
        if file_hash(entry["path"])!=entry["sha256"]:raise ValueError("frozen TRAINfeature changed")
        inputs.append(Path(entry["path"]))
    files=[Path(__file__),PLAN,Path(gap.__file__),gap.PLAN]
    p=study.producing(root,"source_root_gradient_reload_pilot" if pilot else "source_root_fixed_capacity_training",
        inputs+files,{**asdict(Config()),"pilot":pilot,"actual_updates":2 if pilot else 256,"cases":len(rows)})
    p["hypothesis_id"]=HYPOTHESIS
    p["config"].update(hypothesis_id=HYPOTHESIS,coarse_levels=1,word_levels=0,
        source_prediction=True,oracle_target_codes=False,TRAIN_only=True,task_accuracy=None,
        original_parent_loaded=False,word_modules_frozen=True,nonroot_heads_frozen=True)
    for i,path in enumerate(files):
        p["source_hashes"][str(path)]=file_hash(path)
        (root/"producer_source"/f"source_root_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=study.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(root/"producer.json",p)
    return p


def save(path,net,optimizer,sampler,noise,step,p):
    if path.exists():raise ValueError("checkpoint overwrite forbidden")
    torch.save({"model":net.state_dict(),"optimizer":optimizer.state_dict(),"step":step,
        "source_frame":net.generator.prefix_len,"sampler_rng":sampler.get_state(),"noise_rng":noise.get_state(),
        "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
        "config":asdict(Config()),"producer_fingerprint":p["fingerprint"]},path)


def train(pilot=False):
    study.data.require_gpu();study.data.terminal(gap.DEST)
    receipt=study.q.read(COMPLETED/"ncm-word-final-frozen-parent-v1_registry_readback.json")
    if (receipt["status"]!="registered_and_read_back"
            or receipt["record"]["campaign_sha256"]!=file_hash(COMPLETED/"campaign_results.json")
            or study.q.read(COMPLETED/"campaign_results.json")["classification"]!="word_final_NCM_development_quality_failed"):
        raise ValueError("complete archived preceding task failure required before new capacity training")
    rows=gap.selection(study.q.read(study.data.DATA/"cases.json")["train"])
    manifest=study.q.read(study.data.FEATURES/"manifest.json")
    first=rows[0];longest=sorted(rows,key=lambda r:(-len(r["source_ids"]),r["id"]))[0]
    probes=[first,longest]
    if not pilot:
        study.data.terminal(PILOT)
        if study.q.read(PILOT/"result.json")["status"]!="systems_source_root_pilot_passed":
            raise ValueError("actual two-update capacity gradient/reload pilot required")
    root=PILOT if pilot else TRAIN;p=producing(root,pilot,rows,manifest)
    config=Config();torch.manual_seed(config.model_seed);torch.cuda.manual_seed_all(config.model_seed)
    net,codec,_=study.checkpoint();parameters=scope(net)
    initial=study.recovery.base.tensor_digest(dict(net.named_parameters()));frozen=frozen_hash(net)
    codec_hash=study.core.released.module_hash(codec)
    baseline=evaluate(net,rows,manifest)
    if file_hash(gap.DEST/"case_measurements.json")!=study.q.read(gap.DEST/"result.json")["case_measurements_sha256"]:
        raise ValueError("preceding component measurements changed")
    expected=[{"id":r["id"],"codes":r["source_only_codes"][0]} for r in study.q.read(gap.DEST/"case_measurements.json")]
    if baseline["outputs"]!=expected:raise ValueError("initial source-only roots must reproduce preceding final512 exactly")
    if not pilot and initial!=study.q.read(PILOT/"result.json")["initial_parameter_sha256"]:
        raise ValueError("full capacity run must start fresh from the same final512 as pilot")
    write_json(root/"initial_root_measurements.json",baseline)
    optimizer=torch.optim.AdamW(parameters,lr=config.max_lr,betas=(.9,.95),weight_decay=.01)
    sampler=torch.Generator().manual_seed(config.sampler_seed);noise=torch.Generator().manual_seed(15143)
    torch.manual_seed(config.model_seed);torch.cuda.manual_seed_all(config.model_seed)
    net.train();torch.cuda.reset_peak_memory_stats();(root/"checkpoints").mkdir()
    updates=2 if pilot else config.updates
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group["lr"]=rate(step+1)
        row=probes[step] if pilot else rows[int(torch.randint(len(rows),(1,),generator=sampler))]
        batch=study.recovery.base.batch_for(row,manifest,frame=2816 if pilot and step==1 else None)
        study.set_source_frame(net.generator,batch["source_ids"].shape[1],maximum=8192)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits=root_logits(net,batch["features"],batch["mask"],batch["source_ids"])
        value,terms=loss(logits,batch["codes"][:,:16],active_mask(batch["words"],16),config)
        if not torch.isfinite(value):raise ValueError("nonfinite root loss")
        value.backward()
        if (any(v.grad is not None for v in net.parameters() if not v.requires_grad)
                or any(v.grad is not None for v in codec.parameters())
                or net.source_head.source_projection.weight.grad is None
                or not torch.count_nonzero(net.source_head.source_projection.weight.grad)
                or net.generator.lm_heads[0].proj.weight.grad is None
                or not torch.count_nonzero(net.generator.lm_heads[0].proj.weight.grad)):
            raise ValueError("root/source versus frozen gradient routing failed")
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
        if torch.cuda.max_memory_allocated()>8_000_000_000:raise ValueError("capacity memory budget exceeded")
        trace={"step":step+1,"id":row["id"],"source_frame":net.generator.prefix_len,
            "lr":rate(step+1),"loss":float(value.detach()),"gradient_norm":float(norm),**terms}
        with (root/"trace.jsonl").open("a") as handle:handle.write(json.dumps(trace,allow_nan=False)+"\n")
        if pilot or (step+1)%16==0:print(json.dumps(trace),flush=True)
        del batch,logits,value
        if not pilot and step+1==128:save(root/"checkpoints"/"step_000128.pt",net,optimizer,sampler,noise,step+1,p)
    save(root/"model.pt",net,optimizer,sampler,noise,updates,p)
    study.recovery.configure()
    reloaded=study.recovery.base.reload_check(root/"model.pt",net,optimizer,sampler,noise,probes,manifest)
    final=evaluate(net,rows,manifest);write_json(root/"final_root_measurements.json",final)
    if (frozen_hash(net)!=frozen or study.core.released.module_hash(codec)!=codec_hash
            or study.recovery.base.tensor_digest(dict(net.named_parameters()))==initial):
        raise ValueError("trainable roots must change while frozen word/head/codec parameters remain exact")
    study.data.verify(p);write_json(root/"source_only_reload_probe.json",reloaded)
    c=final["counts"];passed=(c["active_root_recovery"]>=.95 and c["all_root_recovery"]>=.95
        and c["active_root_recovery"]-c["shuffled_active_recovery"]>=.5)
    result={"status":"systems_source_root_pilot_passed" if pilot else "TRAIN_only_source_root_capacity_completed",
        "updates":updates,"initial_root_counts":baseline["counts"],"final_root_counts":c,
        "capacity_gate":passed if not pilot else None,"task_accuracy":None,"TRAIN_only":True,
        "frozen_words_other_heads_and_codec_unchanged":True,"reload_exact":True,
        "checkpoint_sha256":file_hash(root/"model.pt"),"initial_parameter_sha256":initial,
        "frozen_parameter_sha256_unchanged":frozen,"frozen_codec_sha256_unchanged":codec_hash,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"NCM_success_established":False,
        "final_test_opened":False,"speed_tested":False,"producer_fingerprint":p["fingerprint"]}
    write_json(root/"result.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("stage",choices=["pilot","train"])
    train(parser.parse_args().stage=="pilot")
