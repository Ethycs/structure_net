"""Bounded NCM codec and generator adaptation on actual Llama summaries."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
from transformers import AutoModelForCausalLM

from experiments.structure_net.ncm_llama import (
    BASE, INITIAL, FIXTURES, LABELS, MODEL_PATH, Config, autocast, code_loss,
    decode_ids, digest, encode_codes, final_latent, freeze, generator_input, infer_ncm, label_ids,
    load_ncm, seal, sha256, storage_bytes, text_check, tokenizers,
    training_examples, write_json)
from experiments.structure_net.llama_summary_throughput import make_fragment
from experiments.structure_net.llama_tensor_runtime import TensorRuntime, CHECKPOINT

C = Config()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    tmp=path.with_suffix(".tmp");torch.save(obj,tmp);tmp.replace(path)


def prepare(args):
    producer=freeze(args.output,"data_preparation_not_quality_evidence",asdict(C),
        [INITIAL/"config.yaml",INITIAL.parent/"download_receipt.json"])
    gpt,llama=tokenizers();examples=[]
    specs=(("raw256",256,False),("raw384",384,False),("raw480",480,False),("hierarchy384",384,True))
    for split,count,start in (("train",C.train_examples,610000),("development",C.development_examples,620000),
                               ("heldout",C.heldout_examples,630000)):
        for i in range(count):
            family,length,hierarchy=specs[i%4];seed=start+i
            fragment=make_fragment(seed,0,length,gpt,hierarchy)
            source=fragment["source"]
            while len(gpt.encode(source,add_special_tokens=False))<length or len(llama.encode(source,add_special_tokens=False))<192:
                source="Assistant: Background notes about meeting formatting do not change final decisions.\n"+source
            prefix=gpt.encode(source,add_special_tokens=False)
            required={**fragment["required"],"project":fragment["reference_fact"].split(":")[0]}
            instruction=("Write one short summary of the final project decision. Preserve the project, final owner, "
                "approved budget and final deadline. Ignore superseded proposals and routine discussion. "
                "Use at most 32 words. Output only the summary.")
            prompt=llama.apply_chat_template([{"role":"system","content":instruction},
                {"role":"user","content":source}],add_generation_prompt=True)
            if len(prefix)>C.prefix or len(llama.encode(source,add_special_tokens=False))<192 or len(prompt)+137>C.context:
                raise ValueError("source exceeds capacity or bypass boundary; no truncation")
            examples.append({"id":f"{split}_{seed}","split":split,"seed":seed,"family":family,
                "source":source,"source_sha256":digest(source),"prefix_ids":prefix+[50257]*(C.prefix-len(prefix)),
                "source_gpt2_tokens":len(prefix),"prompt_ids":prompt,"required":required,
                "superseded":fragment["superseded"],"reference_fact":fragment["reference_fact"]})
    if len({e["source_sha256"] for e in examples})!=len(examples):raise ValueError("data overlap")
    manifest=read("data/models/Llama-3.2-3B-cotenancy/manifest.json")
    physical={}
    for name,record in manifest["files"].items():
        p=MODEL_PATH/name
        if not p.is_file():continue
        actual=sha256(p)
        expected=record if isinstance(record,str) else record["sha256"]
        if actual!=expected:raise ValueError("original Llama physical readback differs")
        physical[str(p)]=actual
    if len(physical)<7:raise ValueError("full original Llama manifest required")
    write_json(args.output/"input_readback.json",{"status":"verified","teacher_files":physical,
        "gpt2_files":{str(p):sha256(p) for p in Path(gpt.name_or_path).iterdir() if p.is_file()}})
    write_json(args.output/"examples.json",{"schema":"nal.ncm-llama.v1","examples":examples,
        "producer_fingerprint":producer["fingerprint"]})
    seal(args.output,producer,{"status":"prepared","examples":len(examples),
        "fixtures_sha256":sha256(args.output/"examples.json"),"max_source_gpt2_tokens":max(e["source_gpt2_tokens"] for e in examples)})


def load_llama(factorized=False):
    if factorized:
        from experiments.structure_net.llama_noise_tensor_continuation import assembled
        return assembled(CHECKPOINT).eval().requires_grad_(False)
    return AutoModelForCausalLM.from_pretrained(MODEL_PATH,local_files_only=True,
        torch_dtype=torch.float16,attn_implementation="flash_attention_2").to("cuda").eval().requires_grad_(False)


def pilot(args):
    producer=freeze(args.output,"systems_lifecycle_only_not_quality_evidence",asdict(C),
        [INITIAL/"config.yaml",INITIAL/"tokenizer.pt",INITIAL/"generator.pt"])
    gpt,_=tokenizers();vae,net,_=load_ncm(adaptation=True)
    torch.manual_seed(740000);torch.cuda.reset_peak_memory_stats()
    target=torch.tensor([label_ids("Project Birch: Maya Chen owns the launch, the budget is $4,600, and the final deadline is November 18.",gpt)],device="cuda")
    codes=encode_codes(vae,target)
    vae.decoder.train().requires_grad_(True)
    opt=torch.optim.AdamW(vae.decoder.parameters(),lr=2e-4,betas=(.9,.95),weight_decay=.01)
    with autocast():
        logits=vae.decoder(final_latent(vae,codes.repeat(C.codec_batch,1)))
        loss=torch.nn.functional.cross_entropy(logits.float().flatten(0,1),target.repeat(C.codec_batch,1).flatten())
    loss.backward();torch.nn.utils.clip_grad_norm_(vae.decoder.parameters(),1.,error_if_nonfinite=True);opt.step()
    if any(p.grad is not None for p in vae.encoder.parameters()):raise RuntimeError("codec gradient escaped decoder")
    codec_peak=torch.cuda.max_memory_allocated()
    save(args.output/"codec.pt",{"model":vae.state_dict()})
    vae.load_state_dict(torch.load(args.output/"codec.pt",map_location="cuda",weights_only=True)["model"],strict=True)
    vae.eval().requires_grad_(False);del opt,logits,loss
    prefix=torch.full((C.generator_batch,C.prefix),50257,device="cuda",dtype=torch.long)
    prefix[:,:4]=torch.tensor(gpt.encode("A project decision.")[:4],device="cuda")
    net.train();opt=torch.optim.AdamW(net.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.01)
    with autocast():loss=code_loss(vae,net,prefix,codes.repeat(C.generator_batch,1))
    loss.backward()
    if net.l0_queries.grad is None or not net.l0_queries.grad.abs().sum()>0:raise RuntimeError("level0 queries did not learn")
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
    net.eval()
    with torch.inference_mode(),autocast():
        before=net(generator_input(vae,net,prefix[:1],codes),prefix_pad_mask=prefix[:1].eq(50257)).float()
    save(args.output/"generator.pt",{"model":net.state_dict()})
    net.load_state_dict(torch.load(args.output/"generator.pt",map_location="cuda",weights_only=True)["model"],strict=True)
    with torch.inference_mode(),autocast():
        after=net(generator_input(vae,net,prefix[:1],codes),prefix_pad_mask=prefix[:1].eq(50257)).float()
    if not torch.equal(before,after):raise RuntimeError("checkpoint reload changed logits")
    output=infer_ncm(vae,net,prefix[:1],gpt)
    seal(args.output,producer,{"status":"passed","codec_batch":C.codec_batch,"generator_batch":C.generator_batch,
        "codec_peak_allocated_bytes":codec_peak,"unique_weight_buffer_bytes":storage_bytes(vae,net),
        "reload_max_absolute_error":float((before-after).abs().max()),"generator_loss":float(loss),
        "levels":list(vae.vq.levels),"output_termination":output[0]["termination"]})


@torch.inference_mode()
def teacher(args):
    producer=freeze(args.output,"teacher_sequence_targets",asdict(C),
        [FIXTURES,FIXTURES.parent/"input_readback.json"])
    ex=read(FIXTURES)["examples"];gpt,llama=tokenizers();model=load_llama();runtime=TensorRuntime(model,C.context)
    # A systems-only original-weight graph/reference comparison, not a scored example.
    system_prompt=llama.apply_chat_template([{"role":"user","content":"Write one brief sentence about a garden."}],add_generation_prompt=True)
    expected=model.generate(torch.tensor([system_prompt],device="cuda"),max_new_tokens=7,do_sample=False,
        pad_token_id=128009).tolist()[0][len(system_prompt):]
    actual=runtime.generate(system_prompt,7,"graph_lookup8")["output_ids"]
    if expected!=actual:raise RuntimeError("original teacher runtime control differs from HF greedy")
    write_json(args.output/"runtime_control.json",{"label":"systems_only_not_quality_evidence","token_parity":True})
    rows=[]
    for i,e in enumerate(ex):
        r=runtime.generate(e["prompt_ids"],128,"graph_lookup8")
        text=llama.decode(r["output_ids"],skip_special_tokens=True).strip();check=text_check(text,e,llama)
        try:target=label_ids(text,gpt)
        except ValueError:target=None
        rows.append({"id":e["id"],"source_sha256":e["source_sha256"],"text":text,"output_ids":r["output_ids"],
            "target_ids":target,"usable_label":check["diagnostic_pass"] and target is not None,"diagnostics":check})
        if (i+1)%16==0 or i+1==len(ex):
            write_json(args.output/"labels.json",{"status":"running","rows":rows,"fixtures_sha256":sha256(FIXTURES)})
            print(json.dumps({"teacher_sequences":i+1,"total":len(ex)}),flush=True)
    train=training_examples(ex,rows)
    write_json(args.output/"labels.json",{"status":"completed","rows":rows,"fixtures_sha256":sha256(FIXTURES),"producer_fingerprint":producer["fingerprint"]})
    seal(args.output,producer,{"status":"completed","sequences":len(rows),"usable_training_labels":len(train),
        "labels_sha256":sha256(args.output/"labels.json"),"teacher_runtime_control_passed":True})


def pack(vae,examples,labels,split):
    lookup={r["id"]:r for r in labels};chosen=[e for e in examples if e["split"]==split and lookup[e["id"]]["usable_label"]]
    targets=torch.tensor([lookup[e["id"]]["target_ids"] for e in chosen],device="cuda",dtype=torch.long)
    prefixes=torch.tensor([e["prefix_ids"] for e in chosen],device="cuda",dtype=torch.long)
    codes=torch.cat([encode_codes(vae,x) for x in targets.split(16)])
    return chosen,targets,prefixes,codes


@torch.inference_mode()
def codec_validation(vae,examples,targets,codes,gpt,llama):
    outputs=[];total=0.
    for start in range(0,len(examples),16):
        with autocast():logits=vae.decoder(final_latent(vae,codes[start:start+16]))
        ce=torch.nn.functional.cross_entropy(logits.float().flatten(0,1),targets[start:start+16].flatten())
        total+=float(ce)*len(logits)
        for e,ids in zip(examples[start:start+16],logits.argmax(-1).tolist()):
            text,termination=decode_ids(ids,gpt)
            outputs.append({"id":e["id"],"text":text,"termination":termination,"diagnostics":text_check(text,e,llama)})
    return {"passes":sum(r["diagnostics"]["diagnostic_pass"] for r in outputs),"ce":total/len(examples),"outputs":outputs}


@torch.inference_mode()
def generator_validation(vae,net,all_examples,packed,gpt,llama):
    net.eval();outputs=[];total=0.
    for start in range(0,len(all_examples),8):
        part=all_examples[start:start+8];p=torch.tensor([e["prefix_ids"] for e in part],device="cuda")
        out=infer_ncm(vae,net,p,gpt)
        outputs += [{"id":e["id"],**r,"diagnostics":text_check(r["text"],e,llama)} for e,r in zip(part,out)]
    _,targets,prefixes,codes=packed
    for start in range(0,len(targets),8):
        with autocast():loss=code_loss(vae,net,prefixes[start:start+8],codes[start:start+8])
        total+=float(loss)*len(targets[start:start+8])
    return {"passes":sum(r["diagnostics"]["diagnostic_pass"] for r in outputs),"ce":total/len(targets),"outputs":outputs}


def checkpoint(path,model,opt,step,sampler,producer):
    save(path,{"model":model.state_dict(),"optimizer":opt.state_dict(),"step":step,
        "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state(),
        "sampler_rng":sampler.get_state(),"producer_fingerprint":producer["fingerprint"]})


def adapt(args):
    kind=args.command
    inputs=[FIXTURES,LABELS,INITIAL/"config.yaml",INITIAL/"tokenizer.pt",INITIAL/"generator.pt"]
    if args.codec:inputs.append(args.codec)
    producer=freeze(args.output,f"{kind}_adaptation",asdict(C),inputs)
    label_record=read(LABELS)
    if label_record["status"]!="completed" or label_record["fixtures_sha256"]!=sha256(FIXTURES):
        raise ValueError("completed labels bound to these fixtures required")
    examples=read(FIXTURES)["examples"];labels=label_record["rows"];training_examples(examples,labels)
    gpt,llama=tokenizers();vae,net,cfg=load_ncm(codec=args.codec,adaptation=True)
    torch.manual_seed(C.seed);random.seed(C.seed);torch.cuda.reset_peak_memory_stats()
    train=pack(vae,examples,labels,"train");dev=pack(vae,examples,labels,"development")
    all_dev=[e for e in examples if e["split"]=="development"]
    trace=[];best=None;best_file=None;sampler=torch.Generator().manual_seed(C.seed+(1000 if kind=="codec" else 2000))
    if kind=="codec":
        del net
        vae.eval();initial=codec_validation(vae,dev[0],dev[1],dev[3],gpt,llama)
        write_json(args.output/"development_initial.json",initial)
        if initial["passes"]>=.95*C.development_examples:
            shutil.copyfile(INITIAL/"tokenizer.pt",args.output/"model.pt")
            seal(args.output,producer,{"status":"codec_unchanged","development_passes":initial["passes"],
                "selected_step":0,"selected_checkpoint":str(args.output/"model.pt"),"model_sha256":sha256(args.output/"model.pt"),"updates":0})
            return
        model=vae;params=list(vae.decoder.parameters());vae.decoder.train().requires_grad_(True)
        lr=2e-4;batch=C.codec_batch;steps=C.codec_steps;marks=(256,512)
        with torch.no_grad():latents=torch.cat([final_latent(vae,x) for x in train[3].split(16)])
    else:
        if args.codec is None:raise ValueError("selected frozen codec required")
        codec_result=read(args.codec.parent/"result.json")
        if codec_result["status"] not in ("codec_unchanged","codec_adapted"):raise ValueError("codec fidelity gate required")
        model=net;params=list(net.parameters());lr=1e-4;batch=C.generator_batch;steps=C.generator_steps;marks=(256,512,1024)
        initial=generator_validation(vae,net,all_dev,dev,gpt,llama)
        write_json(args.output/"development_initial.json",initial)
    opt=torch.optim.AdamW(params,lr=lr,betas=(.9,.95),weight_decay=.01)
    frozen_before=digest({n:sha256_tensor(t) for n,t in vae.state_dict().items() if not n.startswith("decoder.")})
    for step in range(1,steps+1):
        pick=torch.randint(len(train[0]),(batch,),generator=sampler).to("cuda")
        opt.zero_grad(set_to_none=True)
        if kind=="generator":
            factor=step/50 if step<=50 else .1+.9*.5*(1+math.cos(math.pi*(step-50)/(steps-50)))
            for group in opt.param_groups:group["lr"]=lr*factor
            net.train()
            with autocast():loss=code_loss(vae,net,train[2][pick],train[3][pick])
        else:
            vae.decoder.train()
            with autocast():
                logits=vae.decoder(latents[pick]);loss=torch.nn.functional.cross_entropy(logits.float().flatten(0,1),train[1][pick].flatten())
        if not torch.isfinite(loss):raise RuntimeError("nonfinite training objective")
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
        trace.append({"step":step,"loss":float(loss),"gradient_norm":float(norm)})
        if step%32==0:print(json.dumps({"stage":kind,**trace[-1]}),flush=True);write_json(args.output/"trace.json",{"rows":trace})
        if step in marks:
            if kind=="codec":
                vae.eval();validation=codec_validation(vae,dev[0],dev[1],dev[3],gpt,llama)
            else:validation=generator_validation(vae,net,all_dev,dev,gpt,llama)
            path=args.output/f"checkpoint_{step}.pt";checkpoint(path,model,opt,step,sampler,producer)
            write_json(args.output/f"development_{step}.json",validation)
            score=(-validation["passes"],validation["ce"],step)
            if best is None or score<best:best,best_file=score,path
            print(json.dumps({"stage":kind,"step":step,"development_passes":validation["passes"],"ce":validation["ce"]}),flush=True)
    frozen_after=digest({n:sha256_tensor(t) for n,t in vae.state_dict().items() if not n.startswith("decoder.")})
    if frozen_before!=frozen_after:raise RuntimeError("encoder or quantizer changed during adaptation")
    selected=torch.load(best_file,map_location="cpu",weights_only=True)
    save(args.output/"model.pt",{"model":selected["model"],"step":selected["step"],"producer_fingerprint":producer["fingerprint"]})
    status=("codec_adapted" if -best[0]>=.95*C.development_examples else "codec_fidelity_failed") if kind=="codec" else "generator_adapted"
    write_json(args.output/"trace.json",{"rows":trace})
    seal(args.output,producer,{"status":status,"updates":steps,"selected_step":best[2],"development_passes":-best[0],
        "development_ce":best[1],"selected_checkpoint":str(args.output/"model.pt"),"model_sha256":sha256(args.output/"model.pt"),
        "continuation_checkpoint":str(best_file),"frozen_encoder_quantizer_sha256":frozen_after,
        "trace_sha256":sha256(args.output/"trace.json"),"usable_training_labels":len(train[0])})


def sha256_tensor(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def evaluate(args):
    inputs=[FIXTURES,INITIAL/"config.yaml"]
    if args.condition in ("selected","released"):
        inputs += [args.codec or INITIAL/"tokenizer.pt",args.model or INITIAL/"generator.pt"]
    else:inputs += [FIXTURES.parent/"input_readback.json",CHECKPOINT/"training.json"]
    producer=freeze(args.output,"heldout_isolated_timing",{**asdict(C),"condition":args.condition,"repeats":3},inputs)
    ex=[e for e in read(FIXTURES)["examples"] if e["split"]=="heldout"]
    gpt,llama=tokenizers();warm=[];rows=[]
    if args.condition in ("selected","released"):
        vae,net,cfg=load_ncm(args.codec,args.model,adaptation=args.condition=="selected")
        payload=storage_bytes(vae,net)
        def generate(e):
            prefix=torch.tensor([e["prefix_ids"]],device="cuda")
            return infer_ncm(vae,net,prefix,gpt)[0]
        setup=0.
    else:
        model=load_llama(args.condition=="fp16");runtime=TensorRuntime(model,C.context);payload=storage_bytes(model);setup=runtime.setup_seconds
        def generate(e):
            r=runtime.generate(e["prompt_ids"],128,"graph_lookup8")
            return {**r,"text":llama.decode(r["output_ids"],skip_special_tokens=True).strip(),
                "termination":"eos" if r["output_ids"][-1] in {128001,128008,128009} else "generation_cap"}
    torch.cuda.reset_peak_memory_stats()
    for e in ex:
        for rep in (-1,0,1,2):
            torch.cuda.synchronize();start=time.perf_counter();r=generate(e);torch.cuda.synchronize();wall=time.perf_counter()-start
            r.update(id=e["id"],condition=args.condition,repeat=rep,wall_seconds=wall,
                common_output_tokens=len(llama.encode(r["text"],add_special_tokens=False)),source_sha256=e["source_sha256"])
            (warm if rep<0 else rows).append(r)
        write_json(args.output/"rows.json",{"status":"running","rows":rows})
        write_json(args.output/"excluded_warmups.json",{"rows":warm})
        print(json.dumps({"condition":args.condition,"cases":len(warm),"total":len(ex)}),flush=True)
    write_json(args.output/"rows.json",{"status":"completed","rows":rows})
    seal(args.output,producer,{"status":"completed_pending_source_review","condition":args.condition,"rows":len(rows),
        "rows_sha256":sha256(args.output/"rows.json"),"warmups_sha256":sha256(args.output/"excluded_warmups.json"),
        "unique_runtime_weight_buffer_bytes":payload,"graph_setup_seconds":setup})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("prepare","pilot","teacher","codec","generator","evaluate"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--codec",type=Path);p.add_argument("--model",type=Path)
    p.add_argument("--condition",choices=("selected","teacher","fp16","released"));args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.command!="prepare" and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):raise RuntimeError("select exactly physical GPU0 before Python starts")
    try:
        if args.command in ("codec","generator"):adapt(args)
        else:globals()[args.command](args)
    except BaseException as exc:
        if args.output.is_dir() and not (args.output/"result.json").exists():
            write_json(args.output/"failure.json",{"status":"failed","error_type":type(exc).__name__,"error":str(exc)})
        raise


if __name__=="__main__":main()
