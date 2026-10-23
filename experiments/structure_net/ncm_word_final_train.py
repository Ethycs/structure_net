"""Fixed-budget training of the new word-final NCM attachment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_word_final_data as data
from experiments.structure_net import ncm_word_final_training as core
from experiments.structure_net import ncm_word_final_initialize as initialization
from experiments.structure_net.ncm_memory_data import file_hash,write_json
from experiments.structure_net.ncm_native_word_mechanism import tensor_digest
from experiments.structure_net.ncm_prefix_geometry import set_source_frame

PILOT=data.ROOT/"20261023_word_final_train_pilot_v1"
TRAIN=data.ROOT/"20261023_word_final_train_v1"


def producing(root,pilot,inputs):
    files=[Path(__file__),Path(core.__file__),Path(initialization.__file__),
        Path("experiments/structure_net/ncm_word_final_head.py"),
        Path("experiments/structure_net/ncm_blockwise_codec.py"),
        Path("experiments/structure_net/ncm_native_word_mechanism.py")]
    files += sorted(data.VENDOR.rglob("*.py"))
    p=data.producing(root,"word_final_gradient_reload_pilot" if pilot else "word_final_fixed_budget_training",
        list(inputs)+files,{**asdict(core.Config()),"pilot":pilot,"actual_updates":2 if pilot else 512,
            "actual_accumulation":1 if pilot else 4,"physical_gpu":0,
            "precision":"FP32masters/BF16autocast/FP32loss_and_quantization","TF32":False,
            "parent_loaded":False,"original_parent_weights_frozen":True})
    for i,path in enumerate(files):
        p["source_hashes"][str(path)]=file_hash(path)
        (root/"producer_source"/f"word_train_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=data.prior.run.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(root/"producer.json",p)
    return p


def batch_for(row,manifest,device="cuda",frame=None):
    entry=manifest[row["id"]]
    cached=torch.load(entry["path"],map_location="cpu",weights_only=True,mmap=True)
    if (cached["id"]!=row["id"] or cached["source_sha256"]!=row["source_sha256"]
            or cached["native_prompt_sha256"]!=row["native_prompt_sha256"]
            or cached["alignment_sha256"]!=row["alignment_sha256"]):
        raise ValueError("source feature/metadata identity mismatch")
    width=len(row["source_ids"])
    frame=row["source_frame"] if frame is None else frame
    if width>frame:raise ValueError("complete source exceeds batch frame; clipping forbidden")
    features=torch.zeros(1,frame,3072,dtype=torch.float32)
    features[0,:width]=cached["features"]
    source_ids=torch.full((1,frame),50257,dtype=torch.long)
    source_ids[0,:width]=torch.tensor(row["source_ids"])
    mask=torch.zeros(1,frame,dtype=torch.long);mask[:,:width]=1
    result={"features":features.to(device),"source_ids":source_ids.to(device),"mask":mask.to(device)}
    if "concept_ids" in row:
        result.update(codes=torch.tensor([row["concept_ids"]],dtype=torch.long,device=device),
            words=torch.tensor([row["word_ids"]],dtype=torch.long,device=device))
    return result


@torch.no_grad()
def probe(net,rows,manifest):
    before=net.training;frame=net.generator.prefix_len
    net.eval();outputs=[]
    try:
        for row in rows:
            batch=batch_for(row,manifest)
            set_source_frame(net.generator,batch["features"].shape[1],maximum=8192)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                result=net.generate(batch["features"],batch["mask"],batch["source_ids"])
            outputs.append({"id":row["id"],"raw_word_ids":result["raw_word_ids"].cpu().tolist(),
                "coarse_codes":[v.cpu().tolist() for v in result["coarse_codes"]]})
    finally:
        set_source_frame(net.generator,frame,maximum=8192)
        net.train(before)
    return outputs


def save(path,net,optimizer,sampler,noise,step,producer):
    if path.exists():raise ValueError("checkpoint overwrite prohibited")
    torch.save({"model":net.state_dict(),"optimizer":optimizer.state_dict(),"step":step,
        "source_frame":net.generator.prefix_len,"sampler_rng":sampler.get_state(),"noise_rng":noise.get_state(),
        "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
        "config":asdict(core.Config()),"producer_fingerprint":producer["fingerprint"]},path)


def reload_check(path,net,optimizer,sampler,noise,rows,manifest):
    before=probe(net,rows,manifest)
    state=torch.load(path,map_location="cpu",weights_only=True)
    with torch.no_grad():
        for value in net.state_dict().values():value.zero_()
    optimizer.state.clear();sampler.manual_seed(0);noise.manual_seed(0);torch.manual_seed(0)
    set_source_frame(net.generator,state["source_frame"],maximum=8192)
    net.load_state_dict(state["model"],strict=True);optimizer.load_state_dict(state["optimizer"])
    sampler.set_state(state["sampler_rng"]);noise.set_state(state["noise_rng"])
    torch.set_rng_state(state["torch_rng"]);torch.cuda.set_rng_state_all(state["cuda_rng"])
    if any(not torch.equal(v.cpu(),state["model"][k]) for k,v in net.state_dict().items()):
        raise ValueError("model/geometry reload differs")
    actual=optimizer.state_dict()
    if actual["param_groups"]!=state["optimizer"]["param_groups"]:
        raise ValueError("optimizer groups differ")
    for index,values in state["optimizer"]["state"].items():
        for key,value in values.items():
            if not torch.equal(value.cpu(),actual["state"][index][key].cpu()):
                raise ValueError("optimizer moment reload differs")
    if (not torch.equal(sampler.get_state(),state["sampler_rng"]) or not torch.equal(noise.get_state(),state["noise_rng"])
            or not torch.equal(torch.get_rng_state(),state["torch_rng"])
            or any(not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),state["cuda_rng"]))):
        raise ValueError("model/sampler/noise RNG reload differs")
    if probe(net,rows,manifest)!=before:raise ValueError("source-only inference reload differs")
    return before


def train(pilot=False):
    data.require_gpu();data.terminal(data.DATA);data.terminal(data.FEATURES);data.terminal(initialization.DEST)
    initialized=data.prior.run.q.read(initialization.DEST/"result.json")
    if initialized["status"]!="full_size_initialization_passed_not_quality":
        raise ValueError("full-size released initialization check required")
    rows=data.prior.run.q.read(data.DATA/"cases.json")["train"]
    manifest=data.prior.run.q.read(data.FEATURES/"manifest.json")
    if len(rows)!=252 or len(manifest)!=380 or file_hash(data.FEATURES/"manifest.json")!=data.prior.run.q.read(data.FEATURES/"result.json")["manifest_sha256"]:
        raise ValueError("complete fixed source feature set required")
    inputs=[data.DATA/n for n in ("producer.json","result.json","cases.json","selection.json","worker_terminal_receipt.json")]
    inputs += [data.FEATURES/n for n in ("producer.json","result.json","manifest.json","worker_terminal_receipt.json")]
    inputs += [initialization.DEST/n for n in ("producer.json","result.json","worker_terminal_receipt.json")]
    inputs += [data.INITIAL/n for n in ("tokenizer.pt","generator.pt","config.yaml")]
    ids=data.prior.run.q.read(data.DATA/"selection.json")["training_pilot_ids"]
    probes=[next(r for r in rows if r["id"]==identifier) for identifier in ids]
    used=probes if pilot else rows
    for row in used:
        entry=manifest[row["id"]]
        if file_hash(entry["path"])!=entry["sha256"]:raise ValueError("frozen feature cache changed")
        inputs.append(Path(entry["path"]))
    if not pilot:
        data.terminal(PILOT)
        if data.prior.run.q.read(PILOT/"result.json")["status"]!="systems_word_final_training_reload_passed":
            raise ValueError("real full-size gradient/reload pilot required")
        inputs += [PILOT/n for n in ("producer.json","result.json","neighbors.pt","worker_terminal_receipt.json")]
    root=PILOT if pilot else TRAIN
    p=producing(root,pilot,inputs)
    config=core.Config();torch.manual_seed(config.model_seed);torch.cuda.manual_seed_all(config.model_seed)
    net,codec,flags=core.make_model()
    initial=tensor_digest(dict(net.named_parameters()));frozen=core.released.module_hash(codec)
    if initial!=initialized["all_new_parameter_sha256"] or frozen!=initialized["frozen_codec_sha256"]:
        raise ValueError("training did not start from verified fresh released initialization")
    neighbors=core.neighbor_table(codec.vq.embeddings[:-1],k=config.neighbors) if pilot else torch.load(PILOT/"neighbors.pt",map_location="cpu",weights_only=True)
    torch.save(neighbors,root/"neighbors.pt")
    parameters=[v for v in net.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=config.max_lr,betas=(.9,.95),weight_decay=.01)
    sampler=torch.Generator().manual_seed(config.sampler_seed);noise=torch.Generator().manual_seed(config.noise_seed)
    torch.manual_seed(config.model_seed);torch.cuda.manual_seed_all(config.model_seed)
    initial_probe=probe(net,probes,manifest);write_json(root/"initial_probe.json",initial_probe)
    net.train();torch.cuda.reset_peak_memory_stats();(root/"checkpoints").mkdir()
    updates,accumulation=(2,1) if pilot else (config.updates,config.accumulation)
    maximum=data.prior.run.q.read(data.DATA/"result.json")["max_source_frame"]
    for step in range(updates):
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group["lr"]=core.rate(step+1,config)
        samples=[]
        for micro in range(accumulation):
            record=probes[step] if pilot else rows[int(torch.randint(len(rows),(1,),generator=sampler))]
            batch=batch_for(record,manifest,frame=maximum if pilot and step==1 else None)
            set_source_frame(net.generator,batch["features"].shape[1],maximum=8192)
            losses,corrupted=core.paired_backward(net,batch,neighbors,noise,accumulation,config)
            samples.append({"id":record["id"],"source_frame":net.generator.prefix_len,
                "noise_sha256":tensor_digest({"codes":corrupted}),**losses})
            del batch,corrupted
        if (net.word_projection.weight.grad is None or not torch.count_nonzero(net.word_projection.weight.grad)
                or net.source_head.source_projection.weight.grad is None
                or not torch.count_nonzero(net.source_head.source_projection.weight.grad)
                or any(v.grad is not None for v in codec.parameters())):
            raise ValueError("new/frozen module gradient routing failed")
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        optimizer.step()
        if torch.cuda.max_memory_allocated()>8_000_000_000:raise ValueError("head allocation exceeds planned pilot budget")
        row={"step":step+1,"samples":samples,"gradient_norm":float(norm),"lr":core.rate(step+1)}
        with (root/"trace.jsonl").open("a") as handle:handle.write(json.dumps(row,allow_nan=False)+"\n")
        if pilot or (step+1)%8==0:print(json.dumps(row),flush=True)
        if not pilot and (step+1)%128==0 and step+1<updates:
            if core.released.module_hash(codec)!=frozen:raise ValueError("frozen codec changed")
            save(root/"checkpoints"/f"step_{step+1:06d}.pt",net,optimizer,sampler,noise,step+1,p)
    save(root/"model.pt",net,optimizer,sampler,noise,updates,p)
    reloaded=reload_check(root/"model.pt",net,optimizer,sampler,noise,probes,manifest)
    if core.released.module_hash(codec)!=frozen or tensor_digest(dict(net.named_parameters()))==initial:
        raise ValueError("new head must change while original codec remains frozen")
    data.verify(p);write_json(root/"reload_probe.json",reloaded)
    result={"status":"systems_word_final_training_reload_passed" if pilot else "word_final_trained_pending_source_only_task_evaluation",
        "updates":updates,"accumulation":accumulation,"initial_parameter_sha256":initial,
        "frozen_codec_sha256_unchanged":frozen,"checkpoint_sha256":file_hash(root/"model.pt"),
        "reload_exact":True,"source_only_reload_probe_exact":True,"parent_loaded":False,
        "original_parent_weights_frozen":True,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "initialization":flags,"task_accuracy":None,"NCM_success_established":False,"final_test_opened":False,
        "producer_fingerprint":p["fingerprint"]}
    write_json(root/"result.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("stage",choices=["pilot","train"])
    train(parser.parse_args().stage=="pilot")
