"""Source-only development evaluation of the fixed final word-final NCM model."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_word_final_cuda_recovery as recovery
from experiments.structure_net import ncm_oracle_codec_semantic_recovery as previous
from experiments.structure_net import ncm_budget128_profile as profile
from experiments.structure_net.ncm_device_geometry import set_source_frame
from experiments.structure_net.ncm_memory_data import file_hash,text_hash,write_json
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.ncm_frozen_body_public_data import PUBLIC_FILE,PUBLIC_SHA256

data=recovery.base.data
core=recovery.base.core
q=previous.study.q
first=previous.study.first
SOURCES=data.ROOT/"20261023_word_final_evaluation_sources_v1"
PILOT=data.ROOT/"20261023_word_final_inference_pilot_v1"
OUTPUTS=data.ROOT/"20261023_word_final_outputs_v1"
PACKETS=data.ROOT/"20261023_word_final_public_packets_v1"
GRADE=data.ROOT/"20261023_word_final_grade_v1"
COHORTS=("short_development","long_diagnostic")
IDENTITY=("id","source_id","group_id","source_sha256","source","spans")


def producing(root,stage,inputs,config):
    files=[Path(__file__),Path(recovery.__file__),recovery.AMENDMENT,
        Path("experiments/structure_net/ncm_device_geometry.py"),
        Path("experiments/structure_net/ncm_word_final_head.py"),Path(core.__file__),
        Path(first.__file__),first.AMENDMENT,Path(profile.__file__),profile.PLAN]
    p=data.producing(root,stage,list(inputs)+files,{**config,
        "source_only":True,"oracle_target_codes":False,"source_truncated":False,
        "coarse_levels":8,"word_levels":1,"AR_fallback":False,"output_repair":False,
        "final_test_opened":False,"speed_tested":False})
    p["config"]["source_prediction"]=stage in (
        "word_final_source_only_inference_pilot","word_final_source_only_development_outputs",
        "source_disjoint_word_final_NCM_development_quality")
    for i,path in enumerate(files):
        p["source_hashes"][str(path)]=file_hash(path)
        (root/"producer_source"/f"word_eval_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(root/"producer.json",p)
    return p


def prepare_sources():
    data.terminal(data.DATA)
    prepared=q.read(data.DATA/"cases.json")
    if file_hash(data.DATA/"cases.json")!=q.read(data.DATA/"result.json")["cases_sha256"]:
        raise ValueError("complete prepared source identity changed")
    fields=(*IDENTITY,"split","source_ids","source_frame","alignment_sha256","native_prompt_sha256")
    cases={split:[{key:r[key] for key in fields} for r in prepared[split]] for split in COHORTS}
    for split,rows in cases.items():
        if len(rows)!=64:raise ValueError("fixed64cases per development cohort required")
        data.prior.check_disjoint(prepared["train"],rows)
    data.prior.check_disjoint(*cases.values())
    proof=verify_public([r for rows in cases.values() for r in rows])
    p=producing(SOURCES,"source_only_evaluation_metadata",[data.DATA/n for n in
        ("producer.json","result.json","cases.json","worker_terminal_receipt.json")]+[PUBLIC_FILE],
        {"cases":128,"TRAIN_labels_in_evaluation_file":False,"both_final_cohorts_closed":True})
    write_json(SOURCES/"cases.json",cases)
    write_json(SOURCES/"public_proof.json",{"exact_sources":proof,"public_file_sha256":PUBLIC_SHA256})
    data.verify(p)
    result={"status":"source_only_development_metadata_prepared","counts":{s:len(r) for s,r in cases.items()},
        "source_disjoint":True,"TRAIN_labels_in_evaluation_file":False,
        "cases_sha256":file_hash(SOURCES/"cases.json"),"producer_fingerprint":p["fingerprint"]}
    write_json(SOURCES/"result.json",result);print(json.dumps(result),flush=True)


def sources():
    data.terminal(SOURCES)
    if file_hash(SOURCES/"cases.json")!=q.read(SOURCES/"result.json")["cases_sha256"]:
        raise ValueError("source-only metadata changed")
    cases=q.read(SOURCES/"cases.json")
    if tuple(cases)!=COHORTS or any(len(rows)!=64 for rows in cases.values()):
        raise ValueError("fixed ordered64+64development cohorts required")
    for split,rows in cases.items():
        if any(r["split"]!=split or any(k in r for k in ("word_ids","concept_ids","target_text")) for r in rows):
            raise ValueError("source-only cohort identity or no-label contract violated")
    return cases


def checkpoint():
    data.terminal(recovery.TRAIN)
    result=q.read(recovery.TRAIN/"result.json")
    if (result["status"]!="word_final_trained_pending_source_only_task_evaluation" or result["updates"]!=512
            or result["accumulation"]!=4 or not result["reload_exact"]
            or file_hash(recovery.TRAIN/"model.pt")!=result["checkpoint_sha256"]):
        raise ValueError("terminal fixed final512checkpoint required")
    state=torch.load(recovery.TRAIN/"model.pt",map_location="cpu",weights_only=True,mmap=True)
    if (state["step"]!=512 or state["config"]!=asdict(core.Config())
            or state["producer_fingerprint"]!=result["producer_fingerprint"]):
        raise ValueError("checkpoint selection/configuration identity differs")
    net,codec,_=core.make_model()
    set_source_frame(net.generator,state["source_frame"],maximum=8192)
    net.load_state_dict(state["model"],strict=True);net.eval()
    if core.released.module_hash(codec)!=result["frozen_codec_sha256_unchanged"]:
        raise ValueError("original codec identity changed")
    return net,codec,state


def decode_words(raw_ids,tokenizer):
    if len(raw_ids)!=192 or any(type(t) is not int or not 0<=t<50257 for t in raw_ids):
        raise ValueError("complete word frame with valid non-PAD IDs required")
    end=next((i for i,t in enumerate(raw_ids) if t==50256),len(raw_ids))
    # Exactly one decode of the concatenated frame, stopped at its global EOS.
    return tokenizer.decode(raw_ids[:end],skip_special_tokens=False),end


@torch.no_grad()
def predict(net,row,manifest,tokenizer):
    if any(k in row for k in ("word_ids","concept_ids","target_text")):
        raise ValueError("evaluation prediction must not receive target labels")
    batch=recovery.base.batch_for(row,manifest)
    set_source_frame(net.generator,batch["source_ids"].shape[1],maximum=8192)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        generated=net.generate(batch["features"],batch["mask"],batch["source_ids"])
    raw=generated["raw_word_ids"][0].cpu().tolist()
    coarse=[v[0].cpu().tolist() for v in generated["coarse_codes"]]
    if len(coarse)!=8 or any(len(v)!=width or any(not 0<=c<4096 for c in v)
            for v,width in zip(coarse,net.generator.levels[:-1])):
        raise ValueError("full predicted coarse hierarchy required")
    text,end=decode_words(raw,tokenizer)
    return {**{k:row[k] for k in IDENTITY},"split":row["split"],"text":text,
        "raw_word_ids":raw,"coarse_codes":coarse,"global_EOS_position":end if end<len(raw) else None,
        "source_tokens":len(row["source_ids"]),"source_frame":row["source_frame"],
        "source_truncated":False,"fixed_generator_passes":9,"AR_fallback":False,
        "oracle_target_codes":False,"output_repair":False}


def require_outputs(rows,expected):
    if len(rows)!=len(expected):raise ValueError("complete prescribed source set required")
    for row,case in zip(rows,expected):
        if (any(row[k]!=case[k] for k in (*IDENTITY,"split")) or row["source_truncated"]
                or row["AR_fallback"] or row["oracle_target_codes"] or row["output_repair"]
                or row["fixed_generator_passes"]!=9):
            raise ValueError("source/order/decoding contract changed")


def generate(pilot=False):
    data.require_gpu();data.terminal(data.FEATURES)
    cases=sources();all_cases=[r for rows in cases.values() for r in rows]
    manifest=q.read(data.FEATURES/"manifest.json")
    if file_hash(data.FEATURES/"manifest.json")!=q.read(data.FEATURES/"result.json")["manifest_sha256"]:
        raise ValueError("frozen feature manifest changed")
    selected=all_cases[:2] if pilot else all_cases
    inputs=[SOURCES/n for n in ("producer.json","result.json","cases.json","worker_terminal_receipt.json")]
    inputs += [recovery.TRAIN/n for n in ("producer.json","result.json","model.pt","worker_terminal_receipt.json")]
    inputs += [data.FEATURES/n for n in ("producer.json","result.json","manifest.json","worker_terminal_receipt.json")]
    for row in selected:
        entry=manifest[row["id"]]
        if file_hash(entry["path"])!=entry["sha256"]:raise ValueError("frozen feature cache changed")
        inputs.append(Path(entry["path"]))
    if not pilot:
        data.terminal(PILOT)
        if q.read(PILOT/"result.json")["status"]!="systems_source_only_inference_reload_passed":
            raise ValueError("actual deterministic inference pilot required")
        inputs += [PILOT/n for n in ("producer.json","result.json","outputs.json","worker_terminal_receipt.json")]
    root=PILOT if pilot else OUTPUTS
    p=producing(root,"word_final_source_only_inference_pilot" if pilot else "word_final_source_only_development_outputs",
        inputs,{"pilot":pilot,"cases":len(selected),"retained_pilot_cases":0 if pilot else 2,
            "physical_gpu":0,"precision":"FP32weights/BF16autocast","output_selection":"single_deterministic_argmax"})
    net,codec,state=checkpoint()
    data.prior.dependencies()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(data.GPT2,local_files_only=True)
    before=recovery.base.tensor_digest(dict(net.named_parameters()))
    frozen=core.released.module_hash(codec)
    outputs=[] if pilot else q.read(PILOT/"outputs.json")
    if not pilot:require_outputs(outputs,selected[:2])
    for row in selected[len(outputs):]:
        outputs.append(predict(net,row,manifest,tokenizer))
        write_json(root/"outputs.json",outputs)
        if pilot or len(outputs)%8==0 or len(outputs)==len(selected):
            print(json.dumps({"generated":len(outputs),"total":len(selected)}),flush=True)
    require_outputs(outputs,selected)
    if pilot:
        with torch.no_grad():
            for value in net.state_dict().values():value.zero_()
        set_source_frame(net.generator,state["source_frame"],maximum=8192)
        net.load_state_dict(state["model"],strict=True)
        if [predict(net,row,manifest,tokenizer) for row in selected]!=outputs:
            raise ValueError("source-only raw codes/IDs/text differ after checkpoint reload")
    if (recovery.base.tensor_digest(dict(net.named_parameters()))!=before
            or core.released.module_hash(codec)!=frozen):
        raise ValueError("source-only inference changed model/codec weights")
    data.verify(p)
    result={"status":"systems_source_only_inference_reload_passed" if pilot else "source_only_outputs_pending_task_grade",
        "cases":len(outputs),"counts":{s:sum(r["split"]==s for r in outputs) for s in COHORTS},
        "pilot_outputs_retained_exactly":not pilot,"all_parameters_and_codec_unchanged":True,
        "output_sha256":file_hash(root/"outputs.json"),"checkpoint_sha256":file_hash(recovery.TRAIN/"model.pt"),
        "task_accuracy":None,"NCM_success_established":False,"final_test_opened":False,"speed_tested":False,
        "producer_fingerprint":p["fingerprint"]}
    write_json(root/"result.json",result);print(json.dumps(result),flush=True)


def prepare_packets():
    data.terminal(OUTPUTS)
    if file_hash(OUTPUTS/"outputs.json")!=q.read(OUTPUTS/"result.json")["output_sha256"]:
        raise ValueError("fixed source-only outputs changed")
    expected=sources();rows=q.read(OUTPUTS/"outputs.json")
    require_outputs(rows,[r for cohort in expected.values() for r in cohort])
    proof=verify_public(rows)
    packets={s:[q.packet(r["id"],r["source"],[{"candidate_id":"summary","text":r["text"]}])
        for r in rows if r["split"]==s] for s in COHORTS}
    p=producing(PACKETS,"exact_public_source_only_summary_judge_payloads",
        [OUTPUTS/n for n in ("producer.json","result.json","outputs.json","worker_terminal_receipt.json")]+[PUBLIC_FILE],
        {"cases":128,"gateway":"https://central-dev.zt:4000/v1/chat/completions","rubric":profile.RUBRIC,"judges":q.JUDGES})
    write_json(PACKETS/"exact_packets.json",packets)
    write_json(PACKETS/"public_payload_proof.json",{
        "public_dataset":"https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        "public_file_sha256":PUBLIC_SHA256,"exact_sources":proof,"source_fields_public_exact":True,
        "candidate_texts_locally_derived_from_public_sources":True,"private_data_in_packets":False,
        "gateway":"https://central-dev.zt:4000/v1/chat/completions","judges":q.JUDGES,
        "rubric_sha256":text_hash(profile.RUBRIC),"exact_packets_sha256":file_hash(PACKETS/"exact_packets.json")})
    data.verify(p)
    result={"status":"public_source_only_summary_payloads_prepared","cases":128,
        "hashes":{n:file_hash(PACKETS/n) for n in ("exact_packets.json","public_payload_proof.json")},
        "task_accuracy":None,"producer_fingerprint":p["fingerprint"]}
    write_json(PACKETS/"result.json",result);print(json.dumps(result),flush=True)


def prior_cache():
    prior=previous.GRADE
    data.terminal(prior)
    previous.study.registered(data.ROOT/"20261023_oracle_codec_semantic_recovery_results_v1",
        "ncm-oracle-codec-semantic-quality-recovery-v1")
    cache=q.read(prior/"judgment_cache.json")
    if len(cache)!=802:raise ValueError("complete802canonical first judgments required")
    for key,entry in cache.items():
        if key!=first.packet_key(entry["packet"],entry["role"]):
            raise ValueError("prior packet/rubric/model identity differs")
        first.audit_row(entry["packet"],entry["role"],entry["row"])
    return cache


def grade_cohort(split,role,rows,packets,cache):
    import tiktoken
    missing={first.packet_key(packet,role):packet for packet in packets
        if first.packet_key(packet,role) not in cache}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={}
        for key,packet in missing.items():
            request_root=GRADE/"requests"/role/key
            (request_root/"attempts").mkdir(parents=True,exist_ok=False)
            jobs[pool.submit(q.request,packet,role,request_root)]=(key,packet,request_root)
        for i,future in enumerate(as_completed(jobs),1):
            key,packet,request_root=jobs[future]
            row=future.result();first.audit_row(packet,role,row)
            write_json(request_root/"row.json",row)
            first.remember(cache,key,{"packet":packet,"role":role,"row":row,"origin":str(request_root)})
            write_json(GRADE/"judgment_cache.json",cache)
            if i%8==0 or i==len(missing):
                print(json.dumps({"split":split,"role":role,"new_completed":i,"new_total":len(missing)}),flush=True)
    encoding=tiktoken.get_encoding("cl100k_base");scores=[]
    for row,packet in zip(rows,packets):
        entry=cache[first.packet_key(packet,role)];first.audit_row(packet,role,entry["row"])
        tokens=len(encoding.encode(row["text"],disallowed_special=()))
        valid=entry["row"]["status"]=="valid"
        scores.append({"id":row["id"],"source_id":row["source_id"],"text":row["text"],"tokens":tokens,
            "valid":valid,"passed":q.passes(entry["row"]["evaluations"][0],row["text"],tokens) if valid else None,
            "evaluations":entry["row"].get("evaluations"),"attempts":entry["row"]["attempts"],
            "packet_key":first.packet_key(packet,role),"first_origin":entry["origin"]})
    summary=previous.study.teacher.old.summarize_scores(scores)
    result={**summary,"judge":q.JUDGES[role],"new_requests":len(missing),
        "reused_first_judgments":len(rows)-len(missing)}
    write_json(GRADE/f"{split}_{role}_case_scores.json",scores)
    write_json(GRADE/f"{split}_{role}_result.json",result)
    print(json.dumps({"split":split,"role":role,**result}),flush=True)
    return result


def grade():
    data.prior.dependencies();profile.install();data.terminal(PACKETS);data.terminal(OUTPUTS)
    for name,expected in q.read(PACKETS/"result.json")["hashes"].items():
        if file_hash(PACKETS/name)!=expected:raise ValueError("exact public payload proof changed")
    rows=q.read(OUTPUTS/"outputs.json");expected=sources()
    require_outputs(rows,[r for cohort in expected.values() for r in cohort])
    verify_public(rows)
    packets=q.read(PACKETS/"exact_packets.json")
    by_split={s:[r for r in rows if r["split"]==s] for s in COHORTS}
    if packets!={s:[q.packet(r["id"],r["source"],[{"candidate_id":"summary","text":r["text"]}])
            for r in by_split[s]] for s in COHORTS}:
        raise ValueError("outgoing packets differ from public payload proof")
    inputs=[PACKETS/n for n in ("producer.json","result.json","exact_packets.json","public_payload_proof.json","worker_terminal_receipt.json")]
    inputs += [OUTPUTS/n for n in ("producer.json","result.json","outputs.json","worker_terminal_receipt.json")]
    inputs += [previous.GRADE/"judgment_cache.json"]
    for role in ("secondary","primary"):
        root=data.ROOT/("20261023_budget128_judge_"+role+"_v1")
        data.terminal(root)
        if q.read(root/"result.json")["status"]!="passed":raise ValueError("qualified128-token judges required")
        inputs += [root/n for n in ("producer.json","result.json","worker_terminal_receipt.json")]
    cache=prior_cache();prior=dict(cache)
    p=producing(GRADE,"source_disjoint_word_final_NCM_development_quality",inputs,
        {"cases":128,"cohorts":COHORTS,"joint_minimum_per_cohort":61,"valid_required_per_cohort":64,
            "judges":q.JUDGES,"rubric":profile.RUBRIC,"workers":4,"max_attempts":3,
            "prior_first_judgments":len(prior),"semantic_rerolls":False})
    write_json(GRADE/"judgment_cache.json",cache);results={}
    for split in COHORTS:
        secondary=grade_cohort(split,"secondary",by_split[split],packets[split],cache)
        primary=grade_cohort(split,"primary",by_split[split],packets[split],cache) if previous.study.qualifies(secondary) else None
        results[split]={"secondary":secondary,"primary":primary,
            "passed":primary is not None and previous.study.qualifies(primary)}
    if any(cache[key]!=entry for key,entry in prior.items()):raise ValueError("prior first judgments changed")
    data.verify(p);passed=all(r["passed"] for r in results.values())
    result={"status":"completed","classification":"development_gates_passed_final_evaluation_required" if passed else "word_final_NCM_development_quality_failed",
        "results":results,"development_gate":passed,"prior_first_judgments_preserved":len(prior),
        "NCM_model":True,"NCM_adaptation":True,"NCM_success_established":False,"confirmed":False,
        "final_test_opened":False,"speed_tested":False,"producer_fingerprint":p["fingerprint"]}
    write_json(GRADE/"result.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=["sources","pilot","generate","packets","grade"])
    stage=parser.parse_args().stage
    if stage=="sources":prepare_sources()
    elif stage in ("pilot","generate"):generate(stage=="pilot")
    elif stage=="packets":prepare_packets()
    else:grade()
