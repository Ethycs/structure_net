"""New whole-summary labels over disjoint public training sources."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_additional_summary_sources_v2 as availability
from experiments.structure_net import ncm_wide_source_labels as wide
from experiments.structure_net.ncm_memory_data import file_hash,text_hash,write_json

prior=availability.prior
PLAN=Path("docs/07 - Status Reports/2026-10-23_ncm-additional-whole-summary-label-plan.md")
PREP=prior.data.ROOT/"20261023_additional_whole_labels_sources_v1"
CANONICAL=prior.GRADE
QUALIFIED=prior.data.ROOT/"20261023_word_final_results_v1"
QUOTAS={0:212,1:212,2:24,3:64}
SYSTEM=wide.teacher.SYSTEM+"\nPrefer40-60words within the128stored-token ceiling. Preserve material qualifications."
_producing=wide.producing
_gpt=None


def root(pilot,stage):
    return prior.data.ROOT/("20261023_additional_whole_labels_"+("pilot" if pilot else "main")+"_"+stage+"_v1")


def select(index):
    used={k:set() for k in ("id","source_id","group_id","source_sha256")};chosen=[]
    for bucket in (2,3,1,0):
        rows=sorted((r for r in index if r["band"]==bucket),key=lambda r:text_hash("ncm-additional-whole-label-v1:"+r["id"]))
        count=0
        for row in rows:
            if any(row[k] in used[k] for k in used):continue
            chosen.append(row)
            for key in used:used[key].add(row[key])
            count+=1
            if count==QUOTAS[bucket]:break
        if count!=QUOTAS[bucket]:raise ValueError("insufficient disjoint sources for fixed band quota: "+str(bucket))
    return sorted(chosen,key=lambda r:(r["band"],text_hash("ncm-additional-whole-label-v1:"+r["id"])))


def payload(cases):
    passages=[{"id":c["id"],"source_lines":[{"line_id":f"L{i}","text":line} for i,line in enumerate(c["source"].split("\n"))]} for c in cases]
    return {"model":wide.q.JUDGES["primary"],"temperature":0,"max_tokens":2048,
        "messages":[{"role":"system","content":SYSTEM},
                    {"role":"user","content":json.dumps({"passages":passages},ensure_ascii=False)}]}


def prepare():
    prior.data.prior.dependencies();prior.data.terminal(availability.DEST)
    result=prior.q.read(availability.DEST/"result.json")
    if file_hash(availability.DEST/"eligible_index.json")!=result["index_sha256"]:
        raise ValueError("broader eligible source index changed")
    selected=select(prior.q.read(availability.DEST/"eligible_index.json"))
    needed={path:{r["line_number"]:r for r in selected if r["store"]==path} for path in {r["store"] for r in selected}}
    recovered={}
    import tiktoken
    enc=tiktoken.get_encoding("cl100k_base")
    for path,positions in needed.items():
        for number,line in enumerate(Path(path).open()):
            if number not in positions:continue
            row=json.loads(line);entry=positions[number]
            if (row["split"]!="train" or any(row[k]!=entry[k] for k in ("id","source_id","group_id","source_sha256"))
                    or text_hash(row["source"])!=row["source_sha256"]
                    or len(enc.encode(row["source"],disallowed_special=()))!=entry["source_cl100k_tokens"]):
                raise ValueError("selected complete training source identity changed")
            recovered[row["id"]]={**{k:row[k] for k in ("id","source_id","group_id","source_sha256","source","spans")},
                "bin":entry["band"],"source_cl100k_tokens":entry["source_cl100k_tokens"],"pool_split":"train"}
    cases=[recovered[r["id"]] for r in selected]
    if len(cases)!=512:raise ValueError("fixed512selected training sources required")
    known=prior.q.read(prior.data.DATA/"cases.json")
    for rows in known.values():prior.data.prior.check_disjoint(cases,rows)
    proof=wide.verify_public(cases)
    inputs=[availability.DEST/n for n in ("producer.json","result.json","eligible_index.json","worker_terminal_receipt.json")]
    inputs += [Path(path) for path in needed]+[availability.old.PUBLIC_FILE,Path(__file__),PLAN,Path(wide.__file__)]
    p=prior.data.producing(PREP,"additional_complete_training_sources_and_initial_teacher_payloads",inputs,
        {"cases":512,"quotas":QUOTAS,"SYSTEM":SYSTEM,"teacher":wide.q.JUDGES["primary"],"student_training_admitted":False})
    for i,path in enumerate((Path(__file__),PLAN)):
        p["source_hashes"][str(path)]=file_hash(path)
        (PREP/"producer_source"/f"additional_labels_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=prior.q.digest({k:v for k,v in p.items() if k!="fingerprint"});write_json(PREP/"producer.json",p)
    write_json(PREP/"train_sources.json",cases)
    pilot=[next(c for c in cases if c["bin"]==b) for b in range(4)]
    write_json(PREP/"pilot_draft_payloads.json",[payload(pilot[i:i+2]) for i in range(0,4,2)])
    pilot_ids={c["id"] for c in pilot};pending=[c for c in cases if c["id"] not in pilot_ids]
    write_json(PREP/"main_draft_payloads.json",[payload(pending[i:i+2]) for i in range(0,len(pending),2)])
    write_json(PREP/"public_payload_proof.json",{"public_dataset":"https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        "public_file_sha256":availability.old.PUBLIC_SHA256,"exact_sources":proof,"source_fields_public_exact":True,
        "old_targets_used":False,"private_data_in_packets":False,"gateway":"https://central-dev.zt:4000/v1/chat/completions",
        "teacher":wide.q.JUDGES["primary"],"auditor":wide.q.JUDGES["secondary"],
        "pilot_payloads_sha256":file_hash(PREP/"pilot_draft_payloads.json"),"main_payloads_sha256":file_hash(PREP/"main_draft_payloads.json")})
    prior.data.verify(p)
    result={"status":"additional_whole_training_sources_selected","cases":512,"quotas":QUOTAS,
        "source_disjoint":True,"student_training_admitted":False,"task_accuracy":None,
        "files":{str(PREP/n):file_hash(PREP/n) for n in ("train_sources.json","pilot_draft_payloads.json","main_draft_payloads.json","public_payload_proof.json")},
        "producer_fingerprint":p["fingerprint"]}
    write_json(PREP/"result.json",result);print(json.dumps(result),flush=True)


def setup(pilot):
    global _gpt
    wide.dependencies();wide.profile.install();wide.teacher.SYSTEM=SYSTEM
    prior.data.terminal(PREP)
    result=prior.q.read(PREP/"result.json")
    if any(file_hash(path)!=expected for path,expected in result["files"].items()):raise ValueError("fixed source/payload proof changed")
    if any(v["messages"][0]["content"]!=SYSTEM for v in prior.q.read(PREP/"pilot_draft_payloads.json")):
        raise ValueError("teacher system differs from frozen exact payloads")
    cases=prior.q.read(PREP/"train_sources.json")
    if len(cases)!=512 or any(c["pool_split"]!="train" for c in cases):raise ValueError("fixed512training sources required")
    wide.verify_public(cases)
    for role in ("primary","secondary"):
        qual=prior.data.ROOT/("20261023_budget128_judge_"+role+"_v1")
        wide.terminal(qual)
        if wide.q.read(qual/"result.json")["status"]!="passed":raise ValueError("qualified128-token judges required")
    if not pilot:
        wide.terminal(root(True,"audit"))
        if not wide.q.read(root(True,"audit")/"result.json")["pilot_gate"]:
            raise ValueError("unchanged four-of-four final teacher pilot gate required")
    from transformers import AutoTokenizer
    _gpt=AutoTokenizer.from_pretrained(prior.data.GPT2,local_files_only=True)
    return [next(c for c in cases if c["bin"]==b) for b in range(4)] if pilot else cases


def label_fits(row):
    text=row["text"];ids=_gpt.encode(text,add_special_tokens=False)
    return bool(row["teacher_valid"] and text.strip() and row["claims"] and row["stored_tokens"]<=128
        and all(c["claim"].strip() and c["evidence_ids"] for c in row["claims"])
        and len(ids)+1<=192 and _gpt.decode(ids,skip_special_tokens=False)==text
        and not set(ids)&set(_gpt.all_special_ids) and row["native_roundtrip_exact"] and not row["native_special_tokens_in_text"])


def capacity_critique(row,evaluation):
    return {"semantic_review":evaluation,"capacity":{"stored_cl100k_tokens":row["stored_tokens"],"stored_limit":128,
        "GPT2_tokens_including_EOS":len(_gpt.encode(row["text"],add_special_tokens=False))+1,"GPT2_limit":192},
        "instruction":"Review exactly once against the full source. Preserve the main point and material qualifications; fix unsupported content and shorten over-cap text without inventing facts."}


def cache_from(paths):
    prior.data.terminal(CANONICAL)
    record=prior.q.read(QUALIFIED/"ncm-word-final-frozen-parent-v1_registry_readback.json")
    if record["status"]!="registered_and_read_back" or record["record"]["campaign_sha256"]!=file_hash(QUALIFIED/"campaign_results.json"):
        raise ValueError("archived preceding task judgments required")
    cache=prior.q.read(CANONICAL/"judgment_cache.json")
    if len(cache)!=930:raise ValueError("complete930canonical first task judgments required")
    for path in paths:
        wide.terminal(path)
        for key,entry in prior.q.read(path/"judgment_cache.json").items():wide.first.remember(cache,key,entry)
    for key,entry in cache.items():
        if key!=wide.first.packet_key(entry["packet"],entry["role"]):raise ValueError("cached packet/profile/model identity changed")
        wide.first.audit_row(entry["packet"],entry["role"],entry["row"])
    return cache


def producing(destination,stage,inputs,config):
    extra=[Path(__file__),PLAN,PREP/"public_payload_proof.json",CANONICAL/"judgment_cache.json",
        QUALIFIED/"campaign_results.json",QUALIFIED/"ncm-word-final-frozen-parent-v1_registry_readback.json"]
    p=_producing(destination,stage,list(inputs)+extra,{**config,"prior_canonical_judgments":930,
        "word_frame_GPT2":192,"stored_cl100k_limit":128,"old_native128_ceiling_removed":True})
    for path in (Path(__file__),PLAN):
        p["source_hashes"][str(path)]=file_hash(path)
        (destination/"producer_source"/("additional_whole_"+path.name)).write_bytes(path.read_bytes())
    p["fingerprint"]=prior.q.digest({k:v for k,v in p.items() if k!="fingerprint"});write_json(destination/"producer.json",p)
    return p


def configure():
    wide.root=root;wide.setup=setup;wide.cache_from=cache_from;wide.producing=producing
    wide.label_fits=label_fits;wide.capacity.capacity_critique=capacity_critique
    wide.sources.POOL=PREP;wide.SYSTEM=SYSTEM
    return wide


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("stage",choices=["prepare","draft","critique","revision","audit"])
    parser.add_argument("--main",action="store_true");args=parser.parse_args()
    if args.stage=="prepare":prepare()
    elif args.stage in ("draft","revision"):configure().generate(not args.main,args.stage=="revision")
    else:configure().grade(not args.main,args.stage=="audit")
