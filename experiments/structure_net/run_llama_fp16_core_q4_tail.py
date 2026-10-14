"""Pack, reload and evaluate a frozen FP16-core/Q4-tail Llama student."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import shutil

import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from accelerate import init_empty_weights

from experiments.structure_net.llama_fp16_core_q4_tail import (
    Config,CoreResidualLinear,HYPOTHESIS_ID,SCHEMA_VERSION,PLAN,replace,seal,unique_bytes)
from experiments.structure_net.llama_tensor_runtime import CHECKPOINT,FIXTURES,TensorRuntime,pilot
from experiments.structure_net.llama_noise_tensor_continuation import assembled
from experiments.structure_net.llama_neighbourhood_tensorization import digest
from experiments.structure_net.llama_summary_throughput import make_fixture
from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH,sha256,write_json

BASE=Path("data/experiments/llama_fp16_core_q4_tail")
TUNING=BASE/"20261014_operator_tuning_v2/result.json"
AMENDMENT=Path("docs/07 - Status Reports/2026-10-14_llama32-q4-tail-operator-amendment.md")


def fixtures(output):
    if output.exists():raise ValueError("fresh fixtures required")
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    specs=((1,500,"raw"),(1,1980,"raw"),(8,6220,"raw"),(1,1000,"hierarchy"))
    examples=[]
    for i in range(20):
        e=make_fixture(tokenizer,25000+i,*specs[i%4],"heldout")
        labels=", ".join(f["id"] for f in e["fragments"])
        body="\n\n".join(f"[{f['id']}]\n{f['source']}" for f in e["fragments"])
        instruction=(f"\n\nSummarize these {len(e['fragments'])} fragments in exactly {len(e['fragments'])} lines labeled {labels}, then stop. "
            "Include each project's final owner, approved budget and deadline. Ignore superseded proposals and background. "
            "Maximum 128 tokens per line. Output only summaries.")
        ids=tokenizer.apply_chat_template([
            {"role":"system","content":"Summarize the supplied evidence accurately and concisely."},
            {"role":"user","content":body+instruction}],add_generation_prompt=True)
        e.update(prompt_ids=ids,prompt_tokens=len(ids))
        if len(ids)>6300 or any(f["source_tokens"]>2048 for f in e["fragments"]):raise ValueError("workload exceeded")
        examples.append(e)
    old=json.loads(FIXTURES.read_text())["examples"]
    old+=json.loads(Path("data/experiments/llama_tensor_runtime/20261014_fixtures_v1.json").read_text())["examples"]
    hashes=[digest(e["prompt_ids"]) for e in examples]
    if len(set(hashes))!=20 or set(hashes)&{digest(e["prompt_ids"]) for e in old}:raise ValueError("heldout overlap")
    snapshot=output.with_suffix(".producer.py")
    if snapshot.exists():raise ValueError("fresh generator snapshot required")
    snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(__file__,snapshot)
    write_json(output,{"schema":SCHEMA_VERSION,"examples":examples,"created_utc":datetime.now(timezone.utc).isoformat(),
        "generator_sha256":sha256(__file__),"generator_snapshot":str(snapshot),"parent_generator_sha256":sha256("experiments/structure_net/llama_summary_throughput.py")})


def model_config(rank,stage,repeats=0):
    tune=json.loads(TUNING.read_text())
    if tune["status"]!="passed":raise ValueError("operator gate required")
    return {"core":asdict(Config(rank)) if rank else None,"stage":stage,"repeats":repeats,"selected_tuning":tune["selected"],
        "parent_checkpoint":str(CHECKPOINT),"parent_training_sha256":sha256(CHECKPOINT/"training.json"),
        "lookup":{"max_ngram":4,"max_tokens":8,"source":"current_prompt_only"},"context":8192,
        "summary_limit":128,"raw_cap":768,"hierarchy_cap":512}


def extras(*paths):
    return [Path(__file__),AMENDMENT,TUNING,Path("experiments/structure_net/tune_llama_q4_tail.py"),
            Path("experiments/structure_net/llama_summary_throughput.py"),
            Path("data/models/Llama-3.2-3B-cotenancy/manifest.json")]+list(paths)


def storage_bytes(state):
    seen=set();total=0
    for x in state.values():
        key=(x.device,x.untyped_storage().data_ptr())
        if key not in seen:seen.add(key);total+=x.untyped_storage().nbytes()
    return total


def cpu_state(model):
    state={};memo={}
    for name,x in model.state_dict().items():
        key=(x.device,x.untyped_storage().data_ptr())
        if key not in memo:memo[key]=x.detach().cpu()
        state[name]=memo[key]
    return state


def load_model(root):
    record=json.loads((root/"result.json").read_text())
    if record["status"]!="packed_checkpoint_validated" or sha256(root/"model.pt")!=record["model_sha256"]:
        raise ValueError("packed checkpoint identity failed")
    state=torch.load(root/"model.pt",map_location="cpu",weights_only=True)
    if storage_bytes(state)!=record["serialized_unique_payload_bytes"]:raise ValueError("physical payload readback mismatch")
    config=AutoConfig.from_pretrained(MODEL_PATH,local_files_only=True)
    config._attn_implementation="flash_attention_2"
    with init_empty_weights(include_buffers=False):
        model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.float16)
    for i,layer in enumerate(model.model.layers):
        for name in ("gate_proj","up_proj","down_proj"):
            prefix=f"model.layers.{i}.mlp.{name}."
            setattr(layer.mlp,name,CoreResidualLinear(*(state[prefix+s] for s in ("core.left","core.right","tail.packed","tail.scales"))))
    model.load_state_dict(state,strict=True,assign=True)
    model.tie_weights()
    model.generation_config=GenerationConfig.from_pretrained(MODEL_PATH,local_files_only=True)
    del state;gc.collect()
    return model.cuda().eval().requires_grad_(False),record


@torch.inference_mode()
def prepare(output,rank):
    config=model_config(rank,"pack")
    producer=seal(output,"systems_checkpoint_only_not_quality_evidence",config,extras())
    model=assembled(CHECKPOINT)
    records=replace(model,Config(rank))
    canary=torch.linspace(-1,1,9*3072,device="cuda",dtype=torch.float16).view(1,9,3072)
    expected=model.model.layers[0].mlp.gate_proj(canary).cpu()
    payload=unique_bytes(model);state=cpu_state(model);serialized=storage_bytes(state)
    torch.save(state,output/"model.pt");del state
    write_json(output/"canary.pt.json",{"ids":"linspace(-1,1,9*3072)","expected_sha256":digest(expected.tolist())})
    torch.save(expected,output/"canary.pt")
    record={"status":"packed_checkpoint_validated","schema":SCHEMA_VERSION,"hypothesis_id":HYPOTHESIS_ID,
        "rank":rank,"producer_fingerprint":producer["fingerprint"],"model_sha256":sha256(output/"model.pt"),
        "file_bytes":(output/"model.pt").stat().st_size,"unique_runtime_weight_buffer_bytes":payload,
        "serialized_unique_payload_bytes":serialized,"projections":records,
        "dense_tail_cached":False,"training_updates":0,"selected_tuning":config["selected_tuning"]}
    write_json(output/"result.json",record)
    del model;gc.collect();torch.cuda.empty_cache()
    restored,_=load_model(output)
    actual=restored.model.layers[0].mlp.gate_proj(canary).cpu()
    if not torch.equal(actual,expected):raise RuntimeError("packed reload canary mismatch")
    record.update(reload_canary_exact=True,tied_embedding_head=restored.model.embed_tokens.weight.data_ptr()==restored.lm_head.weight.data_ptr(),
                  peak_cuda_bytes=torch.cuda.max_memory_allocated())
    write_json(output/"result.json",record);print(json.dumps({k:v for k,v in record.items() if k!="projections"}),flush=True)


@torch.inference_mode()
def evaluate(args):
    fixture=args.fixtures or FIXTURES
    examples=[e for e in json.loads(fixture.read_text())["examples"] if e["split"]==args.split]
    if args.command=="run" and args.repeats!=(3 if args.split=="development" else 5):raise ValueError("fixed repetitions required")
    if args.command=="run" and args.rank and not args.checkpoint:raise ValueError("packed checkpoint required")
    config=model_config(args.rank,args.command+"_"+args.split,args.repeats)
    if args.checkpoint:config["packed_result_sha256"]=sha256(args.checkpoint/"result.json")
    producer=seal(args.output,config["stage"],config,extras(fixture,*([args.checkpoint/"result.json"] if args.checkpoint else [])))
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    model,checkpoint=load_model(args.checkpoint) if args.rank else (assembled(CHECKPOINT),None)
    runtime=TensorRuntime(model)
    longest=max(examples,key=lambda e:len(e["prompt_ids"]))
    if args.command=="pilot":
        result=pilot(runtime,longest);result.update(producer_fingerprint=producer["fingerprint"],checkpoint=checkpoint)
        write_json(args.output/"result.json",result);print(json.dumps({k:v for k,v in result.items() if k not in ("rows","checkpoint")}),flush=True);return
    rows=[];warm=[];arm=f"core{args.rank}_q4" if args.rank else "fp16"
    for e in examples:
        # Excluded per-case compilation/warmup, fresh request reset for every timed call.
        w=runtime.generate(e["prompt_ids"],e["cap"],"graph_lookup8")
        warm.append({"id":e["id"],"wall_seconds":w["wall_seconds"],"output_ids":w["output_ids"]})
        write_json(args.output/"excluded_warmups.json",{"rows":warm})
        for rep in range(args.repeats):
            row=runtime.generate(e["prompt_ids"],e["cap"],"graph_lookup8")
            row.update(id=e["id"],arm=arm,repeat=rep,text=tokenizer.decode(row["output_ids"],skip_special_tokens=True))
            rows.append(row);write_json(args.output/"rows.json",{"status":"running","rows":rows})
            print(json.dumps({"arm":arm,"id":e["id"],"repeat":rep,"seconds":row["wall_seconds"],"tokens":len(row["output_ids"])}),flush=True)
    write_json(args.output/"rows.json",{"status":"completed","rows":rows})
    for p,expected in producer["source_hashes"].items():
        if sha256(p)!=expected:raise RuntimeError("producer changed during campaign")
    write_json(args.output/"result.json",{"status":"completed_pending_source_review","producer_fingerprint":producer["fingerprint"],
        "rows":len(rows),"rows_sha256":sha256(args.output/"rows.json"),"warmups_sha256":sha256(args.output/"excluded_warmups.json"),
        "graph_setup_seconds":runtime.setup_seconds,"peak_cuda_bytes":torch.cuda.max_memory_allocated(),
        "unique_runtime_weight_buffer_bytes":unique_bytes(model),"checkpoint":checkpoint,"arm":arm})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("fixtures","prepare","pilot","run"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--rank",type=int,choices=(0,64,128),default=64)
    p.add_argument("--checkpoint",type=Path);p.add_argument("--fixtures",type=Path)
    p.add_argument("--split",choices=("development","heldout"),default="development");p.add_argument("--repeats",type=int,default=3)
    args=p.parse_args()
    if args.command=="fixtures":fixtures(args.output);return
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("one selected CUDA GPU required")
    if args.command=="prepare":prepare(args.output,args.rank)
    else:evaluate(args)


if __name__=="__main__":main()
