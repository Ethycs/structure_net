"""Independent integrity audit and conservative registration of the NCM pilot."""
import argparse
import json
import math
from pathlib import Path
import shutil
import statistics

from experiments.structure_net.ncm_llama import (
    BASE,FIXTURES,LABELS,Config,HYPOTHESIS_ID,SCHEMA_VERSION,digest,sha256,
    text_check,tokenizers,training_examples,write_json)
from neural_architecture_lab.ncm_llama_hypothesis import NcmEvidence,classification

C=Config()
CONDITIONS=("selected","teacher","fp16","released")


def read(path):return json.loads(Path(path).read_text())


def verify(root):
    p=read(root/"producer.json");r=read(root/"result.json")
    if p["hypothesis_id"]!=HYPOTHESIS_ID or p["schema"]!=SCHEMA_VERSION or digest({k:v for k,v in p.items() if k!="fingerprint"})!=p["fingerprint"]:
        raise ValueError("producer identity differs")
    if r["producer_fingerprint"]!=p["fingerprint"]:raise ValueError("terminal fingerprint differs")
    copies={sha256(x) for x in (root/"producer_source").iterdir()}
    if any(h not in copies for h in p["source_hashes"].values()):raise ValueError("source snapshot missing")
    for path,h in p["input_hashes"].items():
        if sha256(path)!=h:raise ValueError("producing input changed")
    return p,r


def audit(root,condition):
    p,r=verify(root)
    if r["status"]!="completed_pending_source_review" or p["stage"]!="heldout_isolated_timing" or p["config"]["condition"]!=condition or p["config"]["repeats"]!=3:
        raise ValueError("final design differs")
    if p["input_hashes"][str(FIXTURES)]!=sha256(FIXTURES):raise ValueError("final source binding differs")
    raw=read(root/"rows.json");rows=raw["rows"]
    if raw["status"]!="completed" or sha256(root/"rows.json")!=r["rows_sha256"]:raise ValueError("timing file integrity failed")
    ex={e["id"]:e for e in read(FIXTURES)["examples"] if e["split"]=="heldout"}
    keys=[(x["id"],x["repeat"],x["condition"]) for x in rows]
    if len(keys)!=len(set(keys)) or set(keys)!={(i,n,condition) for i in ex for n in range(3)}:raise ValueError("missing or duplicate cells")
    warm=root/"excluded_warmups.json"
    if sha256(warm)!=r["warmups_sha256"] or [x["id"] for x in read(warm)["rows"]]!=list(ex):raise ValueError("warmup binding differs")
    gpt,llama=tokenizers()
    for x in rows:
        if x["source_sha256"]!=ex[x["id"]]["source_sha256"] or digest(ex[x["id"]]["source"])!=x["source_sha256"]:
            raise ValueError("source identity differs")
        if not math.isfinite(x["wall_seconds"]) or x["wall_seconds"]<=0:raise ValueError("invalid request timing")
        if x["common_output_tokens"]!=len(llama.encode(x["text"],add_special_tokens=False)):raise ValueError("output token counting differs")
        if condition in ("teacher","fp16"):
            if llama.decode(x["output_ids"],skip_special_tokens=True).strip()!=x["text"]:raise ValueError("Llama text decode differs")
        else:
            from experiments.structure_net.ncm_llama import decode_ids
            if len(x["native_output_ids"])!=48 or decode_ids(x["native_output_ids"],gpt)!=(x["text"],x["termination"]):
                raise ValueError("NCM text decode differs")
    return rows,ex,r,p


def summarize(rows):
    ids=list(dict.fromkeys(x["id"] for x in rows));cases=[]
    for i in ids:
        part=[x for x in rows if x["id"]==i]
        cases.append({"id":i,"seconds":statistics.median(x["wall_seconds"] for x in part),
            "common_output_tokens":statistics.median(x["common_output_tokens"] for x in part),
            "unique_outputs":len({x["text"] for x in part})})
    seconds=sum(x["seconds"] for x in cases);tokens=sum(x["common_output_tokens"] for x in cases)
    return {"sum_median_seconds":seconds,"mean_request_seconds":seconds/len(cases),"common_output_tokens":tokens,
        "common_tokens_per_second":tokens/seconds,"all_repetitions_same_output":all(x["unique_outputs"]==1 for x in cases),"per_case":cases}


def analyze(args):
    if args.output.exists():raise ValueError("fresh analysis required")
    rows,ex,r,p=audit(args.root,args.condition);_,llama=tokenizers();aid=[];seen=set()
    for x in rows:
        key=(x["id"],digest(x["text"]))
        if key in seen:continue
        seen.add(key);e=ex[x["id"]]
        aid.append({"condition":args.condition,"id":x["id"],"text":x["text"],"text_sha256":digest(x["text"]),
            "source_sha256":e["source_sha256"],"final_source_quote":e["reference_fact"],
            "required":e["required"],"superseded":e["superseded"],"diagnostics":text_check(x["text"],e,llama)})
    write_json(args.output,{"status":"integrity_passed_source_review_pending","producer_fingerprint":p["fingerprint"],
        "summary":summarize(rows),"memory":r,"review_aid":aid})


def review_counts(rows,ex,path):
    items=read(path)["items"];expected={(x["condition"],x["id"],digest(x["text"])) for x in rows}
    index={}
    for x in items:
        key=(x["condition"],x["id"],x["text_sha256"])
        if key in index or key not in expected or x["source_sha256"]!=ex[x["id"]]["source_sha256"]:
            raise ValueError("source-review cell identity differs")
        if type(x["passed"]) is not bool or len(x["rationale"])<20 or x["verdict"] not in ("supported","contradicted","insufficient evidence"):
            raise ValueError("explicit source verdict required")
        if x["passed"] and (x["verdict"]!="supported" or not x["source_quote"] or x["source_quote"] not in ex[x["id"]]["source"]):
            raise ValueError("passing source support missing")
        index[key]=x
    if set(index)!=expected:raise ValueError("incomplete source reviews")
    return {a:sum(all(x["passed"] for k,x in index.items() if k[0]==a and k[1]==i) for i in ex) for a in CONDITIONS}


def audit_adaptation(root,kind):
    r=read(root/"result.json")
    if sha256(r["selected_checkpoint"])!=r["model_sha256"]:raise ValueError("selected model identity differs")
    if kind=="codec" and r["status"]=="codec_unchanged":
        initial=read(root/"development_initial.json")
        if r["selected_step"]!=0 or r["updates"]!=0 or initial["passes"]!=r["development_passes"] or initial["passes"]<.95*64:
            raise ValueError("unchanged codec gate differs")
        return r
    marks=(256,512) if kind=="codec" else (256,512,1024)
    trace=read(root/"trace.json")["rows"]
    if sha256(root/"trace.json")!=r["trace_sha256"] or [x["step"] for x in trace]!=list(range(1,marks[-1]+1)):
        raise ValueError("adaptation update accounting differs")
    if any(not math.isfinite(x[k]) or x[k]<0 for x in trace for k in ("loss","gradient_norm")):
        raise ValueError("invalid adaptation trace")
    candidates=[]
    for step in marks:
        d=read(root/f"development_{step}.json")
        if len({x["id"] for x in d["outputs"]})!=len(d["outputs"]) or d["passes"]!=sum(x["diagnostics"]["diagnostic_pass"] for x in d["outputs"]):
            raise ValueError("development decision differs")
        if not (0<=d["passes"]<=64 and math.isfinite(d["ce"]) and d["ce"]>=0):raise ValueError("invalid selection metric")
        candidates.append((-d["passes"],d["ce"],step))
    best=min(candidates)
    if r["selected_step"]!=best[2] or r["development_passes"]!=-best[0] or r["development_ce"]!=best[1] or r["updates"]!=marks[-1]:
        raise ValueError("quality-first checkpoint selection differs")
    return r


def finalize(args):
    if args.output.exists():raise ValueError("fresh aggregate required")
    roots=[BASE/"20261021_fixtures_v1",BASE/"20261021_teacher_v1",BASE/"20261021_codec_v1"]
    artifacts=[]
    for root in roots:verify(root)
    pilot=args.pilot;p,pr=verify(pilot)
    if pr["status"]!="passed" or pr["codec_batch"]!=16 or pr["generator_batch"]!=8:raise ValueError("full CUDA lifecycle required")
    examples=read(FIXTURES)["examples"];label_record=read(LABELS)
    if label_record["status"]!="completed" or label_record["fixtures_sha256"]!=sha256(FIXTURES):raise ValueError("teacher target binding differs")
    labels=label_record["rows"];usable=len(training_examples(examples,labels));codec=audit_adaptation(roots[-1],"codec")
    readback=read(FIXTURES.parent/"input_readback.json")
    for path,h in {**readback["teacher_files"],**readback["gpt2_files"]}.items():
        if sha256(path)!=h:raise ValueError("physical pretrained input changed")
    parent_readback=Path("data/experiments/llama_fp16_core_q4_tail/20261014_inputs_readback_v1.json")
    parent=read(parent_readback)
    for path,h in {**parent["model_hashes"],**parent["library_hashes"]}.items():
        if sha256(path)!=h:raise ValueError("factorized control or pinned library changed")
    artifacts.append(parent_readback)
    fact=lambda e:tuple(e["required"][k] for k in ("project","owner","budget","deadline"))
    train_facts={fact(e) for e in examples if e["split"]=="train"}
    data_diagnostics={s:{"sources":sum(e["split"]==s for e in examples),
        "unique_fact_tuples":len({fact(e) for e in examples if e["split"]==s}),
        "fact_tuples_seen_in_train":sum(fact(e) in train_facts for e in examples if e["split"]==s)}
        for s in ("train","development","heldout")}
    result={"schema":"nal.ncm-llama-campaign.v1","hypothesis_id":HYPOTHESIS_ID,"status":"completed_exploratory",
        "underpowered":True,"independent_adaptation_seeds":1,"application_accuracy_tested":False,
        "full_application_context_tested":False,"usable_training_labels":usable,"codec":codec,
        "pilot":pr,"teacher_sequences":len(labels),"source_prefix_gpt2_tokens":512,"output_canvas_gpt2_tokens":48,
        "failed_timing_cells":0,"retried_timing_cells":0,"reused_timing_cells":0,
        "dataset_diagnostics":data_diagnostics,"unique_source_hashes":len({e["source_sha256"] for e in examples})}
    if codec["status"]=="codec_fidelity_failed":
        result.update(classification=classification(usable,codec["development_passes"]),direct_timing_cells=0,conditional_generator_stage_ran=False)
    else:
        generator=BASE/"20261021_generator_v1";verify(generator);gr=audit_adaptation(generator,"generator")
        if gr["status"]!="generator_adapted" or gr["updates"]!=1024:raise ValueError("full fixed adaptation budget required")
        roots.append(generator);all_rows=[];summaries={};memory={}
        for condition in CONDITIONS:
            root=BASE/f"20261021_{condition}_heldout_v1";rows,ex,r,p=audit(root,condition)
            roots.append(root);all_rows+=rows;summaries[condition]=summarize(rows);memory[condition]=r
        review=BASE/"20261021_reviews_v1.json";passes=review_counts(all_rows,ex,review);artifacts.append(review)
        candidate,teacher,fp16=(summaries[x] for x in CONDITIONS[:3])
        lr=teacher["sum_median_seconds"]/candidate["sum_median_seconds"]
        tr=candidate["common_tokens_per_second"]/teacher["common_tokens_per_second"]
        result.update(classification=classification(usable,codec["development_passes"],passes["teacher"],passes["selected"],lr,tr),
            direct_timing_cells=len(all_rows),conditional_generator_stage_ran=True,generator=gr,source_pass_counts=passes,
            heldout_summary=summaries,heldout_memory=memory,teacher_latency_speedup=lr,teacher_token_rate_ratio=tr,
            fp16_latency_speedup=fp16["sum_median_seconds"]/candidate["sum_median_seconds"],
            fp16_token_rate_ratio=candidate["common_tokens_per_second"]/fp16["common_tokens_per_second"],
            final_block_order=list(CONDITIONS))
        coordinator=BASE/"20261021_evaluation_campaign_v1";execution=read(coordinator/"execution.json")
        if execution["status"]!="completed_pending_source_review" or execution["order"]!=list(CONDITIONS) or sha256(coordinator/"producer.py")!=execution["source_sha256"]:
            raise ValueError("isolated evaluation coordinator differs")
        if [x["condition"] for x in execution["completed"]]!=list(CONDITIONS):raise ValueError("evaluation order differs")
        for child in execution["completed"]:
            if sha256(Path(child["root"])/"result.json")!=child["result_sha256"]:raise ValueError("coordinator child changed")
        artifacts += [coordinator/"execution.json",coordinator/"producer.py"]
    for root in roots+[pilot]:
        artifacts+=list(root.glob("*.json"))
        r=read(root/"result.json")
        if r.get("selected_checkpoint"):artifacts.append(Path(r["selected_checkpoint"]))
    result["artifacts"]={str(p):sha256(p) for p in dict.fromkeys(artifacts)}
    copies=args.output.parent/"analysis_source";copies.mkdir(parents=True)
    sources=[Path(__file__),Path("src/neural_architecture_lab/ncm_llama_hypothesis.py")]
    for path in sources:shutil.copyfile(path,copies/path.name)
    result["analysis_source_hashes"]={str(p):sha256(p) for p in sources}
    write_json(args.output,result)
    print(json.dumps({k:result.get(k) for k in ("classification","source_pass_counts","teacher_latency_speedup","teacher_token_rate_ratio","direct_timing_cells")}))


def register(args):
    c=read(args.campaign)
    for path,h in c["artifacts"].items():
        if sha256(path)!=h:raise ValueError("aggregate evidence changed")
    record=NcmEvidence(c["classification"],str(args.campaign),sha256(args.campaign),str(args.report),sha256(args.report),c["direct_timing_cells"]).record()
    from structure_net.logging.standardized_logging import LoggingConfig,StandardizedLogger
    logger=StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",queue_dir="data/experiment_queue",
        sent_dir="data/experiment_sent",rejected_dir="data/experiment_rejected",enable_wandb=False,auto_upload=False,
        enable_chromadb=True,chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    exact=logger.hypotheses_collection.get(ids=[HYPOTHESIS_ID],include=["metadatas"])
    if exact["ids"]!=[HYPOTHESIS_ID] or any(exact["metadatas"][0][k]!=record[k] for k in ("tested","confirmed","confirmation_status","direct_experiment_count")):
        raise ValueError("exact registry readback differs")
    semantic=logger.hypotheses_collection.query(query_texts=[record["description"]],n_results=10)
    if HYPOTHESIS_ID not in semantic["ids"][0]:raise ValueError("semantic readback differs")
    receipt={"status":"registered_and_read_back","record":record,"exact_readback":exact["metadatas"][0],"semantic_readback_ids":semantic["ids"][0]}
    for path in (args.campaign.parent/"registry_readback.json",Path("data/hypothesis_registry")/(HYPOTHESIS_ID+".json")):
        if path.exists():raise ValueError("fresh registry receipt required")
        write_json(path,receipt)
        if read(path)["record"]!=record:raise ValueError("full JSON readback differs")
    print(json.dumps(record))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest="command",required=True)
    a=s.add_parser("analyze");a.add_argument("--root",type=Path,required=True);a.add_argument("--condition",choices=CONDITIONS,required=True);a.add_argument("--output",type=Path,required=True)
    a=s.add_parser("finalize");a.add_argument("--output",type=Path,required=True);a.add_argument("--pilot",type=Path,default=BASE/"20261021_pilot_v1")
    a=s.add_parser("register");a.add_argument("--campaign",type=Path,required=True);a.add_argument("--report",type=Path,required=True)
    args=p.parse_args();globals()[args.command](args)


if __name__=="__main__":main()
