"""Fixed-budget source concept pretraining; no word-reconstruction shortcut."""
import argparse
from dataclasses import dataclass,asdict
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_source_hierarchy_core as core
from experiments.structure_net import ncm_word_final_eval as study
from experiments.structure_net import ncm_additional_parent_features as additional
from experiments.structure_net.ncm_memory_data import file_hash,write_json

ROOT_TRAIN=study.data.ROOT/"20261023_word_final_root_capacity_budget1024_train_v1"
ROOT_RESULTS=study.data.ROOT/"20261023_word_final_root_capacity_budget1024_results_v1"
PLAN=Path("docs/07 - Status Reports/2026-10-23_ncm-source-hierarchy-bootstrap-plan.md")
HYPOTHESIS="ncm-source-hierarchy-bootstrap-v1"
PILOT=study.data.ROOT/"20261023_source_hierarchy_bootstrap_pilot_v1"
TRAIN=study.data.ROOT/"20261023_source_hierarchy_bootstrap_train_v1"
EXPECTED="91aece481ae0448126e80b5e29eb65e2c9b1d4950dbae4aa2c4a75e63cb093fb"


@dataclass(frozen=True)
class Config:
    updates:int=16384
    warmup:int=256
    max_lr:float=3e-4
    min_lr:float=3e-5
    model_seed:int=16142
    sampler_seed:int=17142
    noise_seed:int=18142
    corrupt_probability:float=.5
    maximum_uniform_rate:float=.35
    root_weight:float=4.
    padded_weight:float=.25


def rate(step,c=Config()):
    if not 1<=step<=c.updates:raise ValueError("step outside fixed bootstrap budget")
    if step<=c.warmup:return c.max_lr*step/c.warmup
    return c.min_lr+(c.max_lr-c.min_lr)*.5*(1+math.cos(math.pi*(step-c.warmup)/(c.updates-c.warmup)))


def frozen_hash(net):
    return study.recovery.base.tensor_digest({k:v for k,v in net.named_parameters() if not v.requires_grad})


def model():
    study.data.terminal(ROOT_TRAIN)
    receipt=study.q.read(ROOT_RESULTS/"ncm-source-root-capacity-budget1024-v1_registry_readback.json")
    if (receipt["status"]!="registered_and_read_back" or receipt["record"]["campaign_sha256"]!=file_hash(ROOT_RESULTS/"campaign_results.json")
            or not study.q.read(ROOT_TRAIN/"result.json")["capacity_gate"] or file_hash(ROOT_TRAIN/"model.pt")!=EXPECTED):
        raise ValueError("registered successful source-root capacity checkpoint required")
    net,codec,_=study.checkpoint()
    saved=torch.load(ROOT_TRAIN/"model.pt",map_location="cpu",weights_only=True,mmap=True)
    if saved["step"]!=1024:raise ValueError("exact final root checkpoint required")
    study.set_source_frame(net.generator,saved["source_frame"],maximum=8192)
    net.load_state_dict(saved["model"],strict=True);net.eval();core.concept_scope(net)
    return net,codec


def producing(root,pilot,rows,manifest):
    files=[Path(__file__),PLAN,Path(core.__file__)]
    inputs=[ROOT_TRAIN/n for n in ("producer.json","result.json","model.pt","worker_terminal_receipt.json")]
    inputs += [ROOT_RESULTS/n for n in ("campaign_results.json","ncm-source-root-capacity-budget1024-v1_registry_readback.json")]
    inputs += [study.data.DATA/n for n in ("producer.json","result.json","cases.json","selection.json","worker_terminal_receipt.json")]
    inputs += [study.data.FEATURES/n for n in ("producer.json","result.json","manifest.json","worker_terminal_receipt.json")]
    inputs += [additional.DATA/n for n in ("producer.json","result.json","worker_terminal_receipt.json")]
    if not pilot:inputs += [PILOT/n for n in ("producer.json","result.json","worker_terminal_receipt.json")]
    for row in rows:
        entry=manifest[row["id"]]
        if file_hash(entry["path"])!=entry["sha256"]:raise ValueError("frozen old TRAINsource feature changed")
        inputs.append(Path(entry["path"]))
    p=study.producing(root,"source_hierarchy_gradient_reload_pilot" if pilot else "source_hierarchy_fixed_bootstrap_training",
        inputs+files,{**asdict(Config()),"actual_updates":2 if pilot else Config().updates,"batch":1,
            "word_losses":False,"word_modules_frozen":True,"new_teacher_labels_used":False,
            "physical_gpu":0,"matmul_TF32":False,"cuDNN_TF32":False,"precision":"FP32masters/BF16forward/FP32loss_and_reconstruction"})
    p["hypothesis_id"]=HYPOTHESIS
    p["config"].update(hypothesis_id=HYPOTHESIS,coarse_levels=8,word_levels=0,source_prediction=True,
        oracle_target_codes=True,teacher_forced_training=True,task_accuracy=None)
    for i,path in enumerate(files):
        p["source_hashes"][str(path)]=file_hash(path)
        (root/"producer_source"/f"hierarchy_bootstrap_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=study.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(root/"producer.json",p)
    return p


def save(path,net,optimizer,sampler,noise,step,p):
    if path.exists():raise ValueError("checkpoint overwrite prohibited")
    torch.save({"model":net.state_dict(),"optimizer":optimizer.state_dict(),"step":step,
        "source_frame":net.generator.prefix_len,"sampler_rng":sampler.get_state(),"noise_rng":noise.get_state(),
        "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
        "config":asdict(Config()),"producer_fingerprint":p["fingerprint"]},path)


def train(pilot=False):
    study.data.require_gpu();torch.backends.cudnn.allow_tf32=False
    study.data.terminal(study.data.DATA);study.data.terminal(study.data.FEATURES);study.data.terminal(additional.DATA)
    rows=study.q.read(study.data.DATA/"cases.json")["train"]
    manifest=study.q.read(study.data.FEATURES/"manifest.json")
    if len(rows)!=252:raise ValueError("fixed252vetted TRAINsummaries required")
    ids=study.q.read(study.data.DATA/"selection.json")["training_pilot_ids"]
    probes=[next(r for r in rows if r["id"]==identifier) for identifier in ids]
    maximum=study.q.read(additional.DATA/"result.json")["max_source_frame"]
    if maximum!=3328:raise ValueError("declared new maximum frame3328required")
    if not pilot:
        study.data.terminal(PILOT)
        if study.q.read(PILOT/"result.json")["status"]!="systems_source_hierarchy_bootstrap_passed":
            raise ValueError("actual maximum-frame gradient/reload pilot required")
    root=PILOT if pilot else TRAIN;p=producing(root,pilot,probes if pilot else rows,manifest)
    config=Config();torch.manual_seed(config.model_seed);torch.cuda.manual_seed_all(config.model_seed)
    net,codec=model();parameters=[v for v in net.parameters() if v.requires_grad]
    initial=study.recovery.base.tensor_digest(dict(net.named_parameters()));frozen=frozen_hash(net)
    codec_hash=study.core.released.module_hash(codec)
    if not pilot and initial!=study.q.read(PILOT/"result.json")["initial_parameter_sha256"]:
        raise ValueError("full bootstrap must start fresh from the same root checkpoint as pilot")
    optimizer=torch.optim.AdamW(parameters,lr=config.max_lr,betas=(.9,.95),weight_decay=.01)
    sampler=torch.Generator().manual_seed(config.sampler_seed);noise=torch.Generator().manual_seed(config.noise_seed)
    torch.manual_seed(config.model_seed);torch.cuda.manual_seed_all(config.model_seed)
    net.train();torch.cuda.reset_peak_memory_stats();(root/"checkpoints").mkdir()
    updates=2 if pilot else config.updates
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group["lr"]=rate(step+1)
        row=probes[step] if pilot else rows[int(torch.randint(len(rows),(1,),generator=sampler))]
        batch=study.recovery.base.batch_for(row,manifest,frame=maximum if pilot and step==1 else None)
        study.set_source_frame(net.generator,batch["source_ids"].shape[1],maximum=8192)
        corrupt=bool(torch.rand((),generator=noise)<config.corrupt_probability)
        history=core.uniform_history(batch["codes"],net.generator.levels,noise,config.maximum_uniform_rate) if corrupt else batch["codes"]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            logits=core.concept_logits(net,batch["features"],batch["mask"],history,batch["source_ids"])
        value,levels=core.concept_loss(logits,batch["codes"],batch["words"],net.generator.levels,
            root_weight=config.root_weight,pad_weight=config.padded_weight)
        value.backward()
        if (any(v.grad is not None for v in net.parameters() if not v.requires_grad)
                or any(v.grad is not None for v in codec.parameters())
                or net.source_head.source_projection.weight.grad is None
                or any(head.proj.weight.grad is None for head in net.generator.lm_heads[:-1])):
            raise ValueError("source/coarse versus frozen gradient routing failed")
        if pilot and (not torch.count_nonzero(net.source_head.source_projection.weight.grad)
                or any(not torch.count_nonzero(head.proj.weight.grad) for head in net.generator.lm_heads[:-1])):
            raise ValueError("pilot must demonstrate positive gradients in every new source/coarse path")
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
        if torch.cuda.max_memory_allocated()>8_000_000_000:raise ValueError("head allocation exceeds8GBplan")
        trace={"step":step+1,"id":row["id"],"source_frame":net.generator.prefix_len,"corrupted":corrupt,
            "history_sha256":study.recovery.base.tensor_digest({"codes":history}),"lr":rate(step+1),
            "loss":float(value.detach()),"gradient_norm":float(norm),"levels":levels}
        with (root/"trace.jsonl").open("a") as handle:handle.write(json.dumps(trace,allow_nan=False)+"\n")
        if pilot or (step+1)%32==0:print(json.dumps(trace),flush=True)
        del batch,history,logits,value
        if not pilot and (step+1)%2048==0 and step+1<updates:
            if frozen_hash(net)!=frozen or study.core.released.module_hash(codec)!=codec_hash:
                raise ValueError("frozen word/finest/codec parameters changed")
            save(root/"checkpoints"/f"step_{step+1:06d}.pt",net,optimizer,sampler,noise,step+1,p)
    save(root/"model.pt",net,optimizer,sampler,noise,updates,p)
    study.recovery.configure()
    reloaded=study.recovery.base.reload_check(root/"model.pt",net,optimizer,sampler,noise,probes,manifest)
    if (frozen_hash(net)!=frozen or study.core.released.module_hash(codec)!=codec_hash
            or study.recovery.base.tensor_digest(dict(net.named_parameters()))==initial):
        raise ValueError("source concepts must change while frozen words/finest/codec remain exact")
    study.data.verify(p);write_json(root/"source_only_reload_probe.json",reloaded)
    result={"status":"systems_source_hierarchy_bootstrap_passed" if pilot else "source_hierarchy_bootstrap_pending_generated_history_word_training",
        "updates":updates,"cases":252,"initial_parameter_sha256":initial,
        "frozen_parameter_sha256_unchanged":frozen,"frozen_codec_sha256_unchanged":codec_hash,
        "word_modules_finest_head_and_codec_frozen":True,"parent_loaded":False,
        "reload_exact":True,"source_only_reload_probe_exact":True,"checkpoint_sha256":file_hash(root/"model.pt"),
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"task_accuracy":None,
        "NCM_success_established":False,"final_test_opened":False,"speed_tested":False,"producer_fingerprint":p["fingerprint"]}
    write_json(root/"result.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("stage",choices=["pilot","train"])
    train(parser.parse_args().stage=="pilot")
