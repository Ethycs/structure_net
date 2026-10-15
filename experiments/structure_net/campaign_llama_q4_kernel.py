"""Frozen-model Q4 kernel tuning, lifecycle controls and paired summary timing."""
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time

import torch
from transformers import AutoTokenizer

from experiments.structure_net.llama_q4_kernel_v3 import Tile,fused_projection,transient_projection,install
from experiments.structure_net.llama_fp16_core_q4_tail import quantize,dequantize,CoreResidualLinear,unique_bytes
from experiments.structure_net.run_llama_fp16_core_q4_tail import load_model
from experiments.structure_net.llama_noise_tensor_continuation import assembled,FIXTURES
from experiments.structure_net.llama_tensor_runtime import TensorRuntime,pilot,CHECKPOINT,digest
from experiments.structure_net.llama_summary_throughput import make_fixture
from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH,sha256,write_json
from experiments.structure_net.tune_llama_q4_tail import measure

HYPOTHESIS_ID="llama32-3b-q4-kernel-runtime-v1"
SCHEMA_VERSION="nal.llama-q4-kernel.v1"
PLAN=Path("docs/07 - Status Reports/2026-10-14_llama32-q4-kernel-plan.md")
AMENDMENT=Path("docs/07 - Status Reports/2026-10-14_llama32-q4-kernel-layout-amendment.md")
EPILOGUE_AMENDMENT=Path("docs/07 - Status Reports/2026-10-14_llama32-q4-kernel-epilogue-amendment.md")
BASE=Path("data/experiments/llama_q4_kernel")
PACKED=Path("data/experiments/llama_fp16_core_q4_tail/20261014_core64_checkpoint_v1")
MODEL_SHA="8937d83adfa9c77e065f1c4918bea90a20ee9e1695ffc281561646e0f949a8ff"
FAMILIES=("packed","staged","transient")
DECODE=[Tile(bn=bn,split=split) for bn in (64,128) for split in (4,8)]
PREFILL=[Tile(128,128,32,1,8,3),Tile(128,128,32,1,4,2),Tile(64,128,64,1,4,3),
         Tile(64,64,128,1,4,2),Tile(64,128,128,1,8,2),Tile(128,128,64,1,8,2)]


def freeze(output,stage,config,extra=()):
    if output.exists():raise ValueError("fresh evidence root required")
    output.mkdir(parents=True)
    paths=[Path(__file__),PLAN,AMENDMENT,EPILOGUE_AMENDMENT,Path("experiments/structure_net/llama_q4_kernel.py"),
        Path("experiments/structure_net/llama_q4_kernel_v2.py"),Path("experiments/structure_net/llama_q4_kernel_v3.py"),
        Path("experiments/structure_net/llama_fp16_core_q4_tail.py"),Path("experiments/structure_net/run_llama_fp16_core_q4_tail.py"),
        Path("experiments/structure_net/llama_tensor_runtime.py"),Path("experiments/structure_net/llama_noise_tensor_continuation.py"),
        Path("experiments/structure_net/llama_neighbourhood_tensorization.py"),Path("experiments/structure_net/llama_summary_throughput.py"),
        Path("experiments/structure_net/tune_llama_q4_tail.py"),Path("experiments/structure_net/prepare_llama_cotenancy.py"),
        PACKED/"result.json",CHECKPOINT/"training.json",Path("data/experiments/llama_fp16_core_q4_tail/20261014_inputs_readback_v1.json")]+list(extra)
    paths=list(dict.fromkeys(paths));snapshot=output/"producer_source";snapshot.mkdir()
    for i,p in enumerate(paths):shutil.copyfile(p,snapshot/f"{i:03d}_{p.name}")
    import triton
    p={"hypothesis_id":HYPOTHESIS_ID,"schema":SCHEMA_VERSION,"stage":stage,"config":config,
       "source_hashes":{str(p):sha256(p) for p in paths},"model_sha256":MODEL_SHA,"command":sys.argv,
       "created_utc":datetime.now(timezone.utc).isoformat(),"torch":torch.__version__,"cuda":torch.version.cuda,
       "triton":triton.__version__,"gpu":torch.cuda.get_device_name(),"physical_gpu":0,"slots":1}
    p["fingerprint"]=digest(p);write_json(output/"producer.json",p)
    return p


def verify_unchanged(producer):
    for p,h in producer["source_hashes"].items():
        if sha256(p)!=h:raise RuntimeError("producer changed during execution")


def numerical(actual,expected):
    rms=float((actual.float()-expected.float()).square().mean().sqrt()/expected.float().square().mean().sqrt().clamp_min(1e-10))
    cos=float(torch.nn.functional.cosine_similarity(actual.float().flatten(),expected.float().flatten(),dim=0))
    if not torch.isfinite(actual).all() or rms>.002 or cos<.9999:raise RuntimeError(f"numerical gate failed: {rms}, {cos}")
    return {"relative_rms":rms,"cosine":cos}


@torch.inference_mode()
def operators(output):
    cfg={"decode":[asdict(t) for t in DECODE],"prefill":[asdict(t) for t in PREFILL],"seed":51015,
         "widths":[1,9,512,1980,6300],"repetitions":3,"replays":20,"implementation_version":3,
         "effective_decode_accumulation":"fp32","old_decode_fallback":True}
    producer=freeze(output,"systems_operator_only_not_quality_evidence",cfg)
    torch.manual_seed(51015);rows=[];excluded=[]
    for n,k in ((8192,3072),(3072,8192)):
        p,s=quantize(torch.randn((n,k),device="cuda",dtype=torch.float16)*.03)
        dense=dequantize(p,s)
        old=CoreResidualLinear(torch.randn((n,64),device="cuda",dtype=torch.float16)*.03,
                               torch.randn((64,k),device="cuda",dtype=torch.float16)*.03,p,s)
        for m in cfg["widths"]:
            x=torch.randn((1,m,k),device="cuda",dtype=torch.float16)
            expected=old.core(x)+torch.nn.functional.linear(x,dense)
            prior=old(x);prior_numerical=numerical(prior,expected)
            old_time,old_reps=measure(lambda:old(x))
            dense_time,_=measure(lambda:old.core(x)+torch.nn.functional.linear(x,dense))
            choices=[]
            for family in (("packed",) if m<=9 else ("packed","staged")):
                for i,t in enumerate(DECODE if m<=9 else PREFILL):
                    choices.append((family,i,replace(t,staged=family=="staged")))
            if m>9:choices.append(("transient",0,None))
            for family,index,t in choices:
                fn=(lambda:transient_projection(x,old.core,old.tail)) if family=="transient" else (lambda:fused_projection(x,old.core,old.tail,t))
                try:
                    torch.cuda.synchronize();start=time.perf_counter()
                    actual=fn();torch.cuda.synchronize();first_call=time.perf_counter()-start
                    control=numerical(actual,expected);old_control=numerical(actual,prior)
                    seconds,reps=measure(fn)
                except Exception as exc:
                    if type(exc).__name__!="OutOfResources":raise
                    excluded.append({"family":family,"tile_index":index,"m":m,"n":n,"k":k,"reason":str(exc)})
                    write_json(output/"excluded.json",{"rows":excluded});continue
                row={"family":family,"tile_index":index,"tile":asdict(t) if t else None,"m":m,"n":n,"k":k,
                     "seconds":seconds,"repeats":reps,"old_seconds":old_time,"old_repeats":old_reps,
                     "dense_seconds":dense_time,"first_call_seconds":first_call,
                     "reference_error":control,"old_error":old_control,"old_reference_error":prior_numerical}
                rows.append(row);write_json(output/"rows.json",{"status":"running","rows":rows})
                print(json.dumps(row),flush=True)
            del x,actual,expected,prior
        del old,dense,p,s
    selected={}
    for family in ("packed","staged","transient"):
        selected[family]={}
        for width,choices,count in (("decode",DECODE,4),("prefill",PREFILL,6)):
            target="packed" if width=="decode" else family
            if target=="transient":selected[family][width]=None;continue
            eligible=[]
            for index,t in enumerate(choices):
                part=[r for r in rows if r["family"]==target and r["tile_index"]==index and (r["m"]<=9)==(width=="decode")]
                if len(part)==count:eligible.append((sum(r["seconds"] for r in part),index,asdict(replace(t,staged=target=="staged"))))
            if not eligible:raise RuntimeError("no complete numerical/resource-valid layout")
            best=min(eligible,key=lambda x:x[:2])
            old_decode={ (r["n"],r["k"],r["m"]):r["old_seconds"] for r in rows if r["m"]<=9 }
            selected[family][width]=None if width=="decode" and sum(old_decode.values())<=best[0] else best[2]
    # Held-out shapes validate masks and shape-independent M without affecting selection.
    masks=[]
    for n,k in ((8192,3072),(3072,8192)):
        p,s=quantize(torch.randn((n,k),device="cuda",dtype=torch.float16)*.03);dense=dequantize(p,s)
        old=CoreResidualLinear(torch.randn((n,64),device="cuda",dtype=torch.float16)*.03,
                               torch.randn((64,k),device="cuda",dtype=torch.float16)*.03,p,s)
        for m in (513,6234):
            x=torch.randn((1,m,k),device="cuda",dtype=torch.float16);expected=old.core(x)+torch.nn.functional.linear(x,dense)
            for family,layouts in selected.items():
                actual=transient_projection(x,old.core,old.tail) if family=="transient" else fused_projection(x,old.core,old.tail,Tile(**layouts["prefill"]))
                masks.append({"m":m,"n":n,"k":k,"family":family,**numerical(actual,expected)})
        del p,s,dense,old,x,expected,actual
    verify_unchanged(producer)
    write_json(output/"rows.json",{"status":"completed","rows":rows})
    write_json(output/"result.json",{"status":"passed","label":"systems_operator_only_not_quality_evidence",
        "producer_fingerprint":producer["fingerprint"],"selected":selected,"rows":len(rows),"excluded":excluded,
        "mask_controls":masks,"rows_sha256":sha256(output/"rows.json"),"peak_cuda_bytes":torch.cuda.max_memory_allocated()})
    print(json.dumps(selected),flush=True)


def fixtures(output):
    if output.exists():raise ValueError("fresh fixtures required")
    tok=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    specs=((1,500,"raw"),(1,1980,"raw"),(8,6220,"raw"),(1,1000,"hierarchy"));examples=[]
    for i in range(20):
        e=make_fixture(tok,26000+i,*specs[i%4],"heldout")
        labels=", ".join(f["id"] for f in e["fragments"])
        body="\n\n".join(f"[{f['id']}]\n{f['source']}" for f in e["fragments"])
        instruction=(f"\n\nSummarize these {len(e['fragments'])} fragments in exactly {len(e['fragments'])} lines labeled {labels}, then stop. "
            "Include each project's final owner, approved budget and deadline. Ignore superseded proposals and background. "
            "Maximum 128 tokens per line. Output only summaries.")
        ids=tok.apply_chat_template([{"role":"system","content":"Summarize the supplied evidence accurately and concisely."},
            {"role":"user","content":body+instruction}],add_generation_prompt=True)
        e.update(prompt_ids=ids,prompt_tokens=len(ids))
        if len(ids)>6300 or any(f["source_tokens"]>2048 for f in e["fragments"]):raise ValueError("workload limits exceeded")
        examples.append(e)
    old=json.loads(FIXTURES.read_text())["examples"]
    for p in ("data/experiments/llama_tensor_runtime/20261014_fixtures_v1.json", "data/experiments/llama_fp16_core_q4_tail/20261014_fixtures_v1.json"):
        old+=json.loads(Path(p).read_text())["examples"]
    ids=[digest(e["prompt_ids"]) for e in examples]
    if len(set(ids))!=20 or set(ids)&{digest(e["prompt_ids"]) for e in old}:raise ValueError("test overlap")
    snap=output.with_suffix(".producer.py");snap.parent.mkdir(parents=True,exist_ok=True)
    if snap.exists():raise ValueError("fresh generator snapshot required")
    shutil.copyfile(__file__,snap)
    write_json(output,{"schema":SCHEMA_VERSION,"examples":examples,"generator_sha256":sha256(__file__),
        "generator_snapshot":str(snap),"parent_generator_sha256":sha256("experiments/structure_net/llama_summary_throughput.py"),
        "created_utc":datetime.now(timezone.utc).isoformat()})


@torch.inference_mode()
def evaluate(args):
    fixture=args.fixtures or FIXTURES
    ex=[e for e in json.loads(fixture.read_text())["examples"] if e["split"]==args.split]
    if args.repeats!=(3 if args.split=="development" else 5):raise ValueError("fixed repetitions required")
    layouts=json.loads((args.operators/"result.json").read_text())
    if layouts["status"]!="passed":raise ValueError("operator gate required")
    cfg={"family":args.family,"repeats":args.repeats,"split":args.split,"context":8192,
        "lookup":{"max_ngram":4,"max_tokens":8,"source":"current_prompt_only"},"selected_layouts":layouts["selected"]}
    producer=freeze(args.output,args.command+"_"+args.split,cfg,[fixture,args.operators/"result.json"])
    if args.family=="fp16":model=assembled(CHECKPOINT)
    else:
        model,record=load_model(PACKED)
        if record["model_sha256"]!=MODEL_SHA:raise RuntimeError("model identity changed")
        if args.family in FAMILIES:
            l=layouts["selected"][args.family]
            install(model,args.family,Tile(**l["decode"]) if l["decode"] else None,Tile(**l["prefill"]) if l["prefill"] else None)
    if args.family!="fp16" and unique_bytes(model)!=3441776896:raise RuntimeError("persistent state footprint changed")
    runtime=TensorRuntime(model);tok=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    longest=max(ex,key=lambda e:len(e["prompt_ids"]))
    if args.command=="pilot":
        r=pilot(runtime,longest);r.update(producer_fingerprint=producer["fingerprint"],family=args.family,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),unique_runtime_weight_buffer_bytes=unique_bytes(model))
        write_json(args.output/"result.json",r);print(json.dumps({k:v for k,v in r.items() if k!="rows"}),flush=True);return
    rows=[];warm=[]
    for e in ex:
        w=runtime.generate(e["prompt_ids"],e["cap"],"graph_lookup8")
        warm.append(dict(id=e["id"],**w));write_json(args.output/"excluded_warmups.json",{"rows":warm})
        for rep in range(args.repeats):
            r=runtime.generate(e["prompt_ids"],e["cap"],"graph_lookup8")
            r.update(id=e["id"],arm=args.family,repeat=rep,text=tok.decode(r["output_ids"],skip_special_tokens=True))
            rows.append(r);write_json(args.output/"rows.json",{"status":"running","rows":rows})
            print(json.dumps({"arm":args.family,"id":e["id"],"repeat":rep,"seconds":r["wall_seconds"],"tokens":len(r["output_ids"])}),flush=True)
    verify_unchanged(producer);write_json(args.output/"rows.json",{"status":"completed","rows":rows})
    write_json(args.output/"result.json",{"status":"completed_pending_source_review","producer_fingerprint":producer["fingerprint"],
        "rows":len(rows),"arm":args.family,"rows_sha256":sha256(args.output/"rows.json"),
        "warmups_sha256":sha256(args.output/"excluded_warmups.json"),"peak_cuda_bytes":torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"graph_setup_seconds":runtime.setup_seconds,
        "unique_runtime_weight_buffer_bytes":unique_bytes(model)})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("fixtures","operators","pilot","run"))
    p.add_argument("--output",type=Path,required=True);p.add_argument("--operators",type=Path)
    p.add_argument("--family",choices=FAMILIES+("old","fp16"),default="packed")
    p.add_argument("--fixtures",type=Path);p.add_argument("--split",choices=("development","heldout"),default="development")
    p.add_argument("--repeats",type=int,default=3);args=p.parse_args()
    if args.command=="fixtures":fixtures(args.output);return
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("one selected CUDA GPU required")
    try:
        if args.command=="operators":operators(args.output)
        else:evaluate(args)
    except BaseException as exc:
        if args.output.is_dir() and not (args.output/"result.json").exists():
            write_json(args.output/"failure.json",{"status":"failed","error_type":type(exc).__name__,"error":str(exc)})
        raise


if __name__=="__main__":main()
