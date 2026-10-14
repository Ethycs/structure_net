"""Audit paired summary cells, apply frozen gates and verify mixed-precision registry."""
import argparse
import json
import math
from pathlib import Path
import shutil

from transformers import AutoTokenizer

from experiments.structure_net.llama_fp16_core_q4_tail import HYPOTHESIS_ID,PLAN
from experiments.structure_net.run_llama_fp16_core_q4_tail import BASE,TUNING,AMENDMENT,storage_bytes
from experiments.structure_net.llama_tensor_runtime import FIXTURES,CHECKPOINT,digest
from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH,sha256,write_json
from experiments.neural_architecture_lab.analyze_llama_tensor_runtime import summarize,make_review_aid
from experiments.neural_architecture_lab.finalize_llama_tensor_runtime import review_counts
from experiments.neural_architecture_lab.analyze_llama_summary_throughput import audit as audit_native
from neural_architecture_lab.llama_fp16_core_q4_tail_hypothesis import BASELINE_BYTES,CoreQ4Evidence,classification


def verify_producer(root):
    p=json.loads((root/"producer.json").read_text())
    if p["hypothesis_id"]!=HYPOTHESIS_ID or digest({k:v for k,v in p.items() if k!="fingerprint"})!=p["fingerprint"]:
        raise ValueError("producer fingerprint mismatch")
    snapshots={sha256(x) for x in (root/"producer_source").iterdir()}
    # Earlier operator versions remain valid through their archived producers.
    for path,expected in p["source_hashes"].items():
        if expected not in snapshots:raise ValueError("producer snapshot absent")
    return p


def audit(root,examples,arm,repeats):
    p=verify_producer(root);r=json.loads((root/"result.json").read_text())
    if r["status"]!="completed_pending_source_review" or r["producer_fingerprint"]!=p["fingerprint"] or r["rows_sha256"]!=sha256(root/"rows.json"):
        raise ValueError("terminal integrity mismatch")
    raw=json.loads((root/"rows.json").read_text())
    if raw["status"]!="completed" or p["config"]["repeats"]!=repeats or r["arm"]!=arm:raise ValueError("fixed cell design mismatch")
    rows=raw["rows"];actual=[(x["id"],x["repeat"],x["arm"]) for x in rows]
    if len(actual)!=len(set(actual)) or set(actual)!={(i,n,arm) for i in examples for n in range(repeats)}:
        raise ValueError("missing or duplicated cells")
    warm=root/"excluded_warmups.json"
    if sha256(warm)!=r["warmups_sha256"] or [x["id"] for x in json.loads(warm.read_text())["rows"]]!=list(examples):
        raise ValueError("excluded per-case warmup mismatch")
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    for x in rows:
        if tokenizer.decode(x["output_ids"],skip_special_tokens=True)!=x["text"]:raise ValueError("decoded text mismatch")
        if not all(math.isfinite(x[k]) and x[k]>0 for k in ("wall_seconds","ttft_seconds")) or x["ttft_seconds"]>x["wall_seconds"]:
            raise ValueError("invalid timing")
        if not math.isclose(x["wall_seconds"],x["ttft_seconds"]+x["decode_seconds"],rel_tol=1e-9):raise ValueError("timing components mismatch")
        if x["cap"]!=examples[x["id"]]["cap"] or not 0<len(x["output_ids"])<=x["cap"]:
            raise ValueError("output cap mismatch")
        if x["eos"]!=(x["output_ids"][-1] in {128001,128008,128009}):raise ValueError("EOS flag mismatch")
    return rows,p,r


def checkpoint_audit(root):
    import torch
    p=verify_producer(root);r=json.loads((root/"result.json").read_text())
    if r["status"]!="packed_checkpoint_validated" or not r["reload_canary_exact"] or not r["tied_embedding_head"]:
        raise ValueError("checkpoint lifecycle gate failed")
    if sha256(root/"model.pt")!=r["model_sha256"] or (root/"model.pt").stat().st_size!=r["file_bytes"]:
        raise ValueError("packed checkpoint changed")
    state=torch.load(root/"model.pt",map_location="cpu",weights_only=True)
    if storage_bytes(state)!=r["serialized_unique_payload_bytes"]:raise ValueError("physical storage mismatch")
    tail_keys=[k for k in state if ".tail." in k]
    if len(tail_keys)!=168 or any(k.endswith(".weight") for k in tail_keys):raise ValueError("dense tail or missing Q4 buffers")
    if state["model.embed_tokens.weight"].data_ptr()!=state["lm_head.weight"].data_ptr():raise ValueError("serialized tie lost")
    training=json.loads((CHECKPOINT/"training.json").read_text())
    for group in training["groups"]:
        if sha256(group["selected_checkpoint"])!=group["selected_sha256"]:raise ValueError("trained factors changed")
        parent=torch.load(group["selected_checkpoint"],map_location="cpu",weights_only=True)
        for i in range(2):
            for name in ("gate_proj","up_proj","down_proj"):
                prefix=f"model.layers.{2*group['group']+i}.mlp.{name}."
                left,right=parent[f"{i}.{name}.left"],parent[f"{i}.{name}.right"]
                if not torch.equal(state[prefix+"core.left"],left[:,:r["rank"]]) or not torch.equal(state[prefix+"core.right"],right[:r["rank"],:]):
                    raise ValueError("core differs from trained factor slice")
                packed,scales=state[prefix+"tail.packed"],state[prefix+"tail.scales"]
                if packed.dtype!=torch.uint8 or packed.shape!=(right.shape[1]//2,left.shape[0]) or scales.dtype!=torch.float16 or scales.shape!=(right.shape[1]//64,left.shape[0]):
                    raise ValueError("Q4 physical shape/precision mismatch")
    return {"path":str(root),"model_sha256":r["model_sha256"],"payload_bytes":storage_bytes(state),
            "runtime_weight_buffer_bytes":r["unique_runtime_weight_buffer_bytes"],"file_bytes":r["file_bytes"],
            "rank":r["rank"],"projection_count":84,"core_parent_match":True,"tied_storage_readback":True}


def examples(path,split):
    return {e["id"]:e for e in json.loads(path.read_text())["examples"] if e["split"]==split}


def preflight(args):
    if args.output.exists():raise ValueError("new input readback required")
    import torch,triton,transformers
    import flash_attn.flash_attn_interface as interface
    import flash_attn_2_cuda
    import transformers.models.llama.modeling_llama as llama
    manifest=json.loads(Path("data/models/Llama-3.2-3B-cotenancy/manifest.json").read_text())
    hashes={}
    for name,item in manifest["files"].items():
        path=MODEL_PATH/name;actual=sha256(path)
        if actual!=item["sha256"]:raise ValueError("original HF model changed")
        hashes[str(path)]=actual
    training=json.loads((CHECKPOINT/"training.json").read_text())
    for group in training["groups"]:
        actual=sha256(group["selected_checkpoint"])
        if actual!=group["selected_sha256"]:raise ValueError("trained factor changed")
        hashes[group["selected_checkpoint"]]=actual
    libraries={str(Path(x.__file__)):sha256(x.__file__) for x in (interface,flash_attn_2_cuda,llama)}
    write_json(args.output,{"status":"inputs_verified_before_development","model_hashes":hashes,
        "library_hashes":libraries,"torch":torch.__version__,"triton":triton.__version__,"transformers":transformers.__version__,
        "source_sha256":sha256(__file__)})
    print(json.dumps({"status":"inputs_verified_before_development","model_files":len(hashes),"library_files":len(libraries)}),flush=True)


def analyze(args):
    if args.output.exists():raise ValueError("new analysis required")
    ex=examples(args.fixtures,args.split)
    rows,p,r=audit(args.root,ex,args.arm,3 if args.split=="development" else 5)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    write_json(args.output,{"status":"integrity_passed_source_review_pending","producer_fingerprint":p["fingerprint"],
        "rows_sha256":sha256(args.root/"rows.json"),"summary":summarize(rows,ex),"review_aid":make_review_aid(rows,ex,tokenizer)})


def development(base):
    ex=examples(FIXTURES,"development");rows=[];counts={};summaries={};roots=[]
    for rank in (64,128):
        root=base/f"20261014_core{rank}_development_v1";arm=f"core{rank}_q4"
        part,_,_=audit(root,ex,arm,3)
        counts[rank]=review_counts(part,ex,base/f"20261014_core{rank}_development_reviews_v1.json")[arm]
        rows+=part;summaries.update(summarize(part,ex));roots.append(root)
    sizes={rank:json.loads((base/f"20261014_core{rank}_checkpoint_v1/result.json").read_text())["serialized_unique_payload_bytes"] for rank in (64,128)}
    eligible=[rank for rank in (64,128) if counts[rank]==8]
    chosen=min(eligible,key=lambda rank:(summaries[f"core{rank}_q4"]["sum_median_seconds"],sizes[rank],rank)) if eligible else None
    return chosen,counts,summaries,rows,roots


def select(args):
    if args.output.exists():raise ValueError("new selection required")
    chosen,counts,summary,rows,roots=development(args.base)
    write_json(args.output,{"hypothesis_id":HYPOTHESIS_ID,"selected_rank":chosen,"development_pass_counts":counts,
        "development_summary":summary,"review_hashes":{str(args.base/f"20261014_core{rank}_development_reviews_v1.json"):sha256(args.base/f"20261014_core{rank}_development_reviews_v1.json") for rank in (64,128)},
        "cell_count":len(rows),"rule":"8/8 source quality; lowest median complete latency sum, then payload, then rank"})
    print(json.dumps({"selected_rank":chosen,"development_pass_counts":counts}),flush=True)


def finalize(args):
    if args.output.exists():raise ValueError("new aggregate required")
    chosen,counts,summary,rows,roots=development(args.base)
    selection=args.base/"20261014_selection_v1.json";selected=json.loads(selection.read_text())
    if selected["selected_rank"]!=chosen:raise ValueError("selection changed")
    for p,h in selected["review_hashes"].items():
        if sha256(p)!=h:raise ValueError("selection reviews changed")
    fixture=args.base/"20261014_fixtures_v1.json";f=json.loads(fixture.read_text())
    if sha256(f["generator_snapshot"])!=f["generator_sha256"] or {e["seed"] for e in f["examples"]}!=set(range(25000,25020)):
        raise ValueError("fresh generator provenance failed")
    pre=args.base/"20261014_inputs_readback_v1.json";inputs=json.loads(pre.read_text())
    if inputs["status"]!="inputs_verified_before_development":raise ValueError("preflight absent")
    pre_source=args.base/"20261014_inputs_readback_v1.producer.py"
    if sha256(pre_source)!=inputs["source_sha256"]:raise ValueError("preflight producer missing")
    for path,expected in {**inputs["model_hashes"],**inputs["library_hashes"]}.items():
        if sha256(path)!=expected:raise ValueError("inputs/libraries changed during campaign")
    audits={};pilots={};paths=[PLAN,AMENDMENT,TUNING,fixture,Path(f["generator_snapshot"]),FIXTURES,selection,pre,pre_source]
    for name in ("20261014_operator_pilot_v1","20261014_operator_tuning_v2"):
        root=args.base/name;verify_producer(root)
        operator=json.loads((root/"result.json").read_text())
        if operator["status"]!="passed":raise ValueError("operator gate failed")
        paths.extend([root/"result.json",root/"producer.json"])
        if (root/"rows.json").exists():paths.append(root/"rows.json")
    for rank in (64,128):
        root=args.base/f"20261014_core{rank}_checkpoint_v1";audits[rank]=checkpoint_audit(root)
        pr=args.base/f"20261014_core{rank}_pilot_v1";p=verify_producer(pr);pilot=json.loads((pr/"result.json").read_text())
        if pilot["status"]!="passed" or pilot["producer_fingerprint"]!=p["fingerprint"]:raise ValueError("full-model CUDA gate failed")
        pilots[rank]={k:v for k,v in pilot.items() if k not in ("rows","checkpoint")}
        paths += [root/"model.pt",root/"result.json",root/"producer.json",pr/"result.json",pr/"producer.json",
                  args.base/f"20261014_core{rank}_development_reviews_v1.json"]
    result={"schema":"nal.llama-fp16-core-q4-tail-campaign.v1","hypothesis_id":HYPOTHESIS_ID,"status":"completed_exploratory",
        "selected_rank":chosen,"development_pass_counts":counts,"development_summary":summary,"checkpoints":audits,"pilots":pilots,
        "scheduler":{"physical_gpu":0,"slots":1,"isolated":True},"training_updates":0,"dense_tail_cached":False,
        "random_inference_noise":False,"application_accuracy_tested":False,"independent_training_seeds":1,
        "scientific_boundary":"one frozen trained model; synthetic summary requests; technical timing repetitions; Codex source review without independent judge",
        "failed_cells":0,"retried_cells":0,"reused_cells":0,"baseline_weight_bytes":BASELINE_BYTES,
        "block_order":["core64","core128"]}
    if chosen is None:
        result.update(classification=classification(counts),heldout_tested=False)
    else:
        ex=examples(fixture,"heldout");final=[];final_summary={};passes={}
        for arm in (f"core{chosen}_q4","fp16"):
            root=args.base/f"20261014_{arm}_heldout_v1";part,_,_=audit(root,ex,arm,5)
            final+=part;roots.append(root);final_summary.update(summarize(part,ex))
        native=args.base/"20261014_native_heldout_v1";raw,np,ne=audit_native(native)
        if np["model_sha256"]!="6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff" or raw["config"]["drafts"]!=[8] or raw["config"]["repeats"]!=5:
            raise ValueError("native design mismatch")
        for i,e in ex.items():
            if ne[i]["prompt_ids"]!=e["prompt_ids"] or ne[i]["cap"]!=e["cap"]:raise ValueError("native prompt mismatch")
        final += [dict(x,arm="q4_lookup8",output_ids=x["generated_ids"],eos=x["stopped_eos"]) for x in raw["rows"] if x["id"]!="excluded_warmup_longest"]
        passes=review_counts(final,ex,args.base/"20261014_heldout_reviews_v1.json");final_summary=summarize(final,ex)
        candidate=final_summary[f"core{chosen}_q4"];reference=final_summary["fp16"];nc=final_summary["q4_lookup8"]
        latency=reference["sum_median_seconds"]/candidate["sum_median_seconds"]
        tps=candidate["end_to_end_tokens_per_second"]/reference["end_to_end_tokens_per_second"]
        result.update(classification=classification(counts,passes[f"core{chosen}_q4"],passes["fp16"],audits[chosen]["payload_bytes"],latency,tps),
            heldout_tested=True,heldout_pass_counts=passes,heldout_summary=final_summary,fp16_latency_speedup=latency,fp16_tps_ratio=tps,
            native_latency_speedup=nc["sum_median_seconds"]/candidate["sum_median_seconds"],
            native_tps_ratio=candidate["end_to_end_tokens_per_second"]/nc["end_to_end_tokens_per_second"],
            excluded_native_warmups=5,final_block_order=[f"core{chosen}_q4","fp16","q4_lookup8"])
        rows+=final;paths += [native/"raw.json",native/"result.json",native/"producer.json",args.base/"20261014_heldout_reviews_v1.json"]
    for root in roots:paths += [root/"producer.json",root/"result.json",root/"rows.json",root/"excluded_warmups.json"]
    result["direct_experiment_count"]=len(rows)
    result["artifacts"]={str(p):{"path":str(p),"sha256":sha256(p)} for p in paths}
    source=[Path(__file__),Path("src/neural_architecture_lab/llama_fp16_core_q4_tail_hypothesis.py"),
            Path("experiments/neural_architecture_lab/analyze_llama_tensor_runtime.py"),
            Path("experiments/neural_architecture_lab/finalize_llama_tensor_runtime.py")]
    snap=args.output.parent/"analysis_source";snap.mkdir(parents=True)
    for p in source:shutil.copyfile(p,snap/p.name)
    result["analysis_source_hashes"]={str(p):sha256(p) for p in source}
    write_json(args.output,result);print(json.dumps({k:result[k] for k in ("classification","direct_experiment_count","selected_rank")}),flush=True)


def register(args):
    c=json.loads(args.campaign.read_text())
    for a in c["artifacts"].values():
        if sha256(a["path"])!=a["sha256"]:raise ValueError("aggregate evidence changed")
    record=CoreQ4Evidence(str(args.campaign),sha256(args.campaign),str(args.report),sha256(args.report),c["classification"],c["direct_experiment_count"]).record()
    from structure_net.logging.standardized_logging import LoggingConfig,StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",queue_dir="data/experiment_queue",
        sent_dir="data/experiment_sent",rejected_dir="data/experiment_rejected",enable_wandb=False,auto_upload=False,
        enable_chromadb=True,chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    exact=logger.hypotheses_collection.get(ids=[HYPOTHESIS_ID],include=["metadatas"])
    if exact["ids"]!=[HYPOTHESIS_ID] or any(exact["metadatas"][0][k]!=record[k] for k in ("tested","confirmed","confirmation_status","direct_experiment_count")):
        raise ValueError("exact registry readback failed")
    semantic=logger.hypotheses_collection.query(query_texts=[record["description"]],n_results=10)
    if HYPOTHESIS_ID not in semantic["ids"][0]:raise ValueError("semantic readback failed")
    receipt={"status":"registered_and_read_back","record":record,"exact_readback":exact["metadatas"][0],"semantic_readback_ids":semantic["ids"][0]}
    for path in (args.campaign.parent/"registry_readback.json",Path("data/hypothesis_registry")/(HYPOTHESIS_ID+".json")):
        if path.exists():raise ValueError("new registry receipt required")
        write_json(path,receipt)
        if json.loads(path.read_text())["record"]!=record:raise ValueError("full JSON readback failed")
    print(json.dumps(record),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest="command",required=True)
    a=s.add_parser("analyze")
    for name in ("root","fixtures","output"):a.add_argument("--"+name,type=Path,required=True)
    a.add_argument("--split",choices=("development","heldout"),required=True);a.add_argument("--arm",required=True)
    for cmd in ("select","finalize"):
        a=s.add_parser(cmd);a.add_argument("--base",type=Path,default=BASE);a.add_argument("--output",type=Path,required=True)
    a=s.add_parser("register")
    for name in ("campaign","report"):a.add_argument("--"+name,type=Path,required=True)
    a=s.add_parser("preflight");a.add_argument("--output",type=Path,required=True)
    args=p.parse_args();globals()[args.command](args)


if __name__=="__main__":main()
