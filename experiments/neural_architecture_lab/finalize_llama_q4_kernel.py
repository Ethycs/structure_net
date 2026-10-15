"""Independent source/cell audit, quality-first kernel selection and registry."""
import argparse
import json
import math
from pathlib import Path
import shutil

from transformers import AutoTokenizer
from experiments.structure_net.campaign_llama_q4_kernel import (
    BASE,PLAN,AMENDMENT,EPILOGUE_AMENDMENT,PACKED,MODEL_SHA,HYPOTHESIS_ID,FAMILIES)
from experiments.structure_net.llama_tensor_runtime import FIXTURES,digest
from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH,sha256,write_json
from experiments.neural_architecture_lab.analyze_llama_tensor_runtime import summarize,make_review_aid
from experiments.neural_architecture_lab.finalize_llama_tensor_runtime import review_counts
from neural_architecture_lab.llama_q4_kernel_hypothesis import KernelEvidence,classification,PAYLOAD_BYTES


def verify_producer(root):
    p=json.loads((root/"producer.json").read_text())
    if p["hypothesis_id"]!=HYPOTHESIS_ID or p["model_sha256"]!=MODEL_SHA or digest({k:v for k,v in p.items() if k!="fingerprint"})!=p["fingerprint"]:
        raise ValueError("producer fingerprint/model identity mismatch")
    copies={sha256(x) for x in (root/"producer_source").iterdir()}
    if any(h not in copies for h in p["source_hashes"].values()):raise ValueError("frozen producer source missing")
    return p


def examples(path,split):
    return {e["id"]:e for e in json.loads(path.read_text())["examples"] if e["split"]==split}


def audit(root,ex,arm,repeats):
    p=verify_producer(root);r=json.loads((root/"result.json").read_text());raw=json.loads((root/"rows.json").read_text())
    if r["status"]!="completed_pending_source_review" or r["producer_fingerprint"]!=p["fingerprint"] or r["rows_sha256"]!=sha256(root/"rows.json"):
        raise ValueError("terminal integrity mismatch")
    if p["config"]["family"]!=arm or p["config"]["repeats"]!=repeats or r["arm"]!=arm or raw["status"]!="completed":
        raise ValueError("fixed design mismatch")
    rows=raw["rows"];keys=[(x["id"],x["repeat"],x["arm"]) for x in rows]
    if len(keys)!=len(set(keys)) or set(keys)!={(i,n,arm) for i in ex for n in range(repeats)}:raise ValueError("missing or duplicate cells")
    warm=root/"excluded_warmups.json"
    if sha256(warm)!=r["warmups_sha256"] or [x["id"] for x in json.loads(warm.read_text())["rows"]]!=list(ex):raise ValueError("per-case warmup mismatch")
    if arm!="fp16" and r["unique_runtime_weight_buffer_bytes"]!=PAYLOAD_BYTES+256:raise ValueError("persistent model footprint changed")
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    for x in rows:
        if tokenizer.decode(x["output_ids"],skip_special_tokens=True)!=x["text"]:raise ValueError("decoded text differs")
        if not all(math.isfinite(x[k]) and x[k]>0 for k in ("wall_seconds","ttft_seconds")) or x["wall_seconds"]<x["ttft_seconds"]:
            raise ValueError("invalid timings")
        if not math.isclose(x["wall_seconds"],x["ttft_seconds"]+x["decode_seconds"],rel_tol=1e-9):raise ValueError("time components differ")
        if not 0<len(x["output_ids"])<=ex[x["id"]]["cap"] or x["cap"]!=ex[x["id"]]["cap"]:raise ValueError("output bounds differ")
        if x["eos"]!=(x["output_ids"][-1] in {128001,128008,128009}):raise ValueError("EOS differs")
    return rows,p,r


def analyze(args):
    if args.output.exists():raise ValueError("new analysis required")
    ex=examples(args.fixtures,args.split);rows,p,_=audit(args.root,ex,args.arm,3 if args.split=="development" else 5)
    tok=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    write_json(args.output,{"status":"integrity_passed_source_review_pending","producer_fingerprint":p["fingerprint"],
        "rows_sha256":sha256(args.root/"rows.json"),"summary":summarize(rows,ex),"review_aid":make_review_aid(rows,ex,tok)})


def development(base):
    ex=examples(FIXTURES,"development");rows=[];counts={};summary={};roots=[]
    for arm in FAMILIES:
        root=base/f"20261014_{arm}_development_v1";part,_,_=audit(root,ex,arm,3)
        counts[arm]=review_counts(part,ex,base/f"20261014_{arm}_development_reviews_v1.json")[arm]
        rows+=part;summary.update(summarize(part,ex));roots.append(root)
    eligible=[a for a in FAMILIES if counts[a]==8]
    chosen=min(eligible,key=lambda a:(summary[a]["sum_median_seconds"],FAMILIES.index(a))) if eligible else None
    return chosen,counts,summary,rows,roots


def select(args):
    if args.output.exists():raise ValueError("new selection required")
    chosen,counts,summary,rows,roots=development(args.base)
    write_json(args.output,{"hypothesis_id":HYPOTHESIS_ID,"selected_arm":chosen,"development_pass_counts":counts,
        "development_summary":summary,"primary_cells":len(rows),"review_hashes":{
            str(args.base/f"20261014_{a}_development_reviews_v1.json"):sha256(args.base/f"20261014_{a}_development_reviews_v1.json") for a in FAMILIES}})
    print(json.dumps({"selected_arm":chosen,"pass_counts":counts}),flush=True)


def finalize(args):
    if args.output.exists():raise ValueError("new aggregate required")
    chosen,counts,summary,rows,roots=development(args.base)
    selection=args.base/"20261014_selection_v1.json";s=json.loads(selection.read_text())
    if s["selected_arm"]!=chosen:raise ValueError("selection differs")
    for p,h in s["review_hashes"].items():
        if sha256(p)!=h:raise ValueError("selection review changed")
    fixture=args.base/"20261014_fixtures_v1.json";f=json.loads(fixture.read_text())
    if {e["seed"] for e in f["examples"]}!=set(range(26000,26020)) or sha256(f["generator_snapshot"])!=f["generator_sha256"]:
        raise ValueError("fresh fixture provenance differs")
    if sha256(PACKED/"model.pt")!=MODEL_SHA:raise ValueError("packed weights changed")
    input_readback=Path("data/experiments/llama_fp16_core_q4_tail/20261014_inputs_readback_v1.json")
    original=json.loads(input_readback.read_text())
    for p,h in {**original["model_hashes"],**original["library_hashes"]}.items():
        if sha256(p)!=h:raise ValueError("parent inputs/libraries changed")
    paths=[PLAN,AMENDMENT,EPILOGUE_AMENDMENT,selection,fixture,Path(f["generator_snapshot"]),FIXTURES,input_readback,PACKED/"model.pt",PACKED/"result.json"]
    compiler=args.base/"20261014_compiler_inspection_v1";inspection=json.loads((compiler/"result.json").read_text())
    if inspection["status"]!="completed" or inspection["operator_result_sha256"]!=sha256(args.base/"20261014_operators_v3/result.json"):
        raise ValueError("compiler inspection provenance differs")
    if sha256(compiler/"producer.py")!=inspection["source_sha256"]:raise ValueError("compiler inspector source differs")
    for row in inspection["rows"]:
        ptx=compiler/(row["family"]+".ptx")
        if sha256(ptx)!=row["ptx_sha256"]:raise ValueError("compiler output changed")
        paths.append(ptx)
    coordinator=args.base/"20261014_development_campaign_v1";execution=json.loads((coordinator/"execution.json").read_text())
    if execution["status"]!="completed_pending_source_review" or sha256(coordinator/"producer.py")!=execution["producer_sha256"]:
        raise ValueError("development coordinator provenance differs")
    if [x["family"] for x in execution["completed"]]!=list(FAMILIES):raise ValueError("development block order differs")
    for child in execution["completed"]:
        if sha256(Path(child["root"])/"result.json")!=child["result_sha256"]:raise ValueError("coordinator child changed")
    paths += [compiler/"result.json",compiler/"producer.py",coordinator/"execution.json",coordinator/"producer.py"]
    operators={};pilots={}
    for version in (1,2,3):
        root=args.base/f"20261014_operators_v{version}";p=verify_producer(root);r=json.loads((root/"result.json").read_text())
        if r["status"]!="passed" or r["producer_fingerprint"]!=p["fingerprint"] or r["rows_sha256"]!=sha256(root/"rows.json"):
            raise ValueError("operator terminal integrity failed")
        operators[version]={"rows":r["rows"],"excluded":r["excluded"],"selected":r["selected"]}
        paths += [root/"producer.json",root/"result.json",root/"rows.json"]
    for arm in FAMILIES:
        root=args.base/f"20261014_{arm}_pilot_v1";p=verify_producer(root);r=json.loads((root/"result.json").read_text())
        if r["status"]!="passed" or r["producer_fingerprint"]!=p["fingerprint"]:raise ValueError("full-model CUDA pilot failed")
        pilots[arm]={k:v for k,v in r.items() if k!="rows"}
        paths += [root/"producer.json",root/"result.json",args.base/f"20261014_{arm}_development_reviews_v1.json"]
    result={"schema":"nal.llama-q4-kernel-campaign.v1","hypothesis_id":HYPOTHESIS_ID,"status":"completed_exploratory",
        "selected_arm":chosen,"development_pass_counts":counts,"development_summary":summary,"operators":operators,"pilots":pilots,
        "compiler_inspection":inspection,
        "model_sha256":MODEL_SHA,"payload_bytes":PAYLOAD_BYTES,"persistent_dense_tail_cached":False,"training_updates":0,
        "scheduler":{"physical_gpu":0,"slots":1,"isolated":True},"failed_cells":0,"retried_cells":0,"reused_cells":0,
        "application_accuracy_tested":False,"scientific_boundary":"one frozen trained checkpoint, synthetic inputs, technical repeats, Codex source review without independent judge",
        "broader_double_native_target_tested":False,"development_block_order":list(FAMILIES)}
    if chosen is None:result.update(classification=classification(counts),heldout_tested=False)
    else:
        ex=examples(fixture,"heldout");final=[];terminals={}
        for arm in (chosen,"old","fp16"):
            root=args.base/f"20261014_{arm}_heldout_v1";part,_,terminal=audit(root,ex,arm,5)
            final+=part;roots.append(root);terminals[arm]=terminal
        reviews=args.base/"20261014_heldout_reviews_v1.json";passes=review_counts(final,ex,reviews);sums=summarize(final,ex)
        candidate=sums[chosen];old=sums["old"];fp16=sums["fp16"]
        lr=old["sum_median_seconds"]/candidate["sum_median_seconds"];tr=candidate["end_to_end_tokens_per_second"]/old["end_to_end_tokens_per_second"]
        fl=fp16["sum_median_seconds"]/candidate["sum_median_seconds"];ft=candidate["end_to_end_tokens_per_second"]/fp16["end_to_end_tokens_per_second"]
        review_items=json.loads(reviews.read_text())["items"]
        factual={a:sum(x.get("factual_passed",x["passed"]) for x in review_items if x["arm"]==a) for a in (chosen,"old","fp16")}
        result.update(classification=classification(counts,passes[chosen],passes["old"],PAYLOAD_BYTES,lr,tr),heldout_tested=True,
            heldout_pass_counts=passes,heldout_factual_pass_counts=factual,heldout_summary=sums,heldout_memory=terminals,
            old_latency_speedup=lr,old_tps_ratio=tr,fp16_latency_speedup=fl,fp16_tps_ratio=ft,
            fp16_speed_recovered=min(fl,ft)>=1,fp16_quality_retained=passes[chosen]>=passes["fp16"],final_block_order=[chosen,"old","fp16"])
        rows+=final;paths.append(reviews)
    for root in roots:paths += [root/"producer.json",root/"result.json",root/"rows.json",root/"excluded_warmups.json"]
    result["direct_experiment_count"]=len(rows)
    result["artifacts"]={str(p):{"path":str(p),"sha256":sha256(p)} for p in paths}
    source=[Path(__file__),Path("src/neural_architecture_lab/llama_q4_kernel_hypothesis.py"),
        Path("experiments/neural_architecture_lab/analyze_llama_tensor_runtime.py"),Path("experiments/neural_architecture_lab/finalize_llama_tensor_runtime.py")]
    snapshot=args.output.parent/"analysis_source";snapshot.mkdir(parents=True)
    for p in source:shutil.copyfile(p,snapshot/p.name)
    result["analysis_source_hashes"]={str(p):sha256(p) for p in source}
    write_json(args.output,result)
    print(json.dumps({k:result.get(k) for k in ("classification","old_latency_speedup","old_tps_ratio","fp16_latency_speedup","fp16_tps_ratio","direct_experiment_count")}),flush=True)


def register(args):
    c=json.loads(args.campaign.read_text())
    for x in c["artifacts"].values():
        if sha256(x["path"])!=x["sha256"]:raise ValueError("aggregate evidence changed")
    record=KernelEvidence(str(args.campaign),sha256(args.campaign),str(args.report),sha256(args.report),c["classification"],c["direct_experiment_count"]).record()
    from structure_net.logging.standardized_logging import LoggingConfig,StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",queue_dir="data/experiment_queue",sent_dir="data/experiment_sent",
        rejected_dir="data/experiment_rejected",enable_wandb=False,auto_upload=False,enable_chromadb=True,chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    exact=logger.hypotheses_collection.get(ids=[HYPOTHESIS_ID],include=["metadatas"])
    if exact["ids"]!=[HYPOTHESIS_ID] or any(exact["metadatas"][0][k]!=record[k] for k in ("tested","confirmed","confirmation_status","direct_experiment_count")):
        raise ValueError("exact registry readback failed")
    semantic=logger.hypotheses_collection.query(query_texts=[record["description"]],n_results=10)
    if HYPOTHESIS_ID not in semantic["ids"][0]:raise ValueError("semantic readback failed")
    receipt={"status":"registered_and_read_back","record":record,"exact_readback":exact["metadatas"][0],"semantic_readback_ids":semantic["ids"][0]}
    for p in (args.campaign.parent/"registry_readback.json",Path("data/hypothesis_registry")/(HYPOTHESIS_ID+".json")):
        if p.exists():raise ValueError("new registry receipt required")
        write_json(p,receipt)
        if json.loads(p.read_text())["record"]!=record:raise ValueError("full JSON readback failed")
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
    args=p.parse_args();globals()[args.command](args)


if __name__=="__main__":main()
