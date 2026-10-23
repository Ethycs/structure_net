"""Complete-source feature preparation, independent of teacher label outcomes."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_word_final_data as base
from experiments.structure_net import ncm_additional_whole_labels as labels
from experiments.structure_net.ncm_memory_data import file_hash,write_json

PLAN=Path("docs/07 - Status Reports/2026-10-23_ncm-additional-parent-features-plan.md")
ORIGINAL_DATA=base.DATA
DATA=base.ROOT/"20261023_additional_parent_data_v1"
PILOT=base.ROOT/"20261023_additional_parent_pilot_v1"
FEATURES=base.ROOT/"20261023_additional_parent_features_v1"
HYPOTHESIS="ncm-additional-whole-source-features-v1"
_producing=base.producing


def producing(root,stage,inputs,config):
    files=[Path(__file__),PLAN]
    extra=[labels.PREP/n for n in ("producer.json","result.json","train_sources.json","public_payload_proof.json","worker_terminal_receipt.json")]
    p=_producing(root,stage,list(inputs)+files+extra,{**config,"teacher_labels_read":False,
        "additional_training_sources":512,"original_parent_body_hash":base.EXPECTED_BODY})
    p["hypothesis_id"]=HYPOTHESIS;p["config"]["hypothesis_id"]=HYPOTHESIS
    for i,path in enumerate(files):
        p["source_hashes"][str(path)]=file_hash(path)
        (root/"producer_source"/f"additional_parent_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=base.prior.run.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(root/"producer.json",p)
    return p


def prepare():
    base.prior.dependencies();base.terminal(labels.PREP);base.terminal(ORIGINAL_DATA)
    from transformers import AutoTokenizer
    rows=base.prior.run.q.read(labels.PREP/"train_sources.json")
    expected=base.prior.run.q.read(labels.PREP/"result.json")["files"][str(labels.PREP/"train_sources.json")]
    if len(rows)!=512 or file_hash(labels.PREP/"train_sources.json")!=expected:
        raise ValueError("fixed512complete additional sources required")
    known=base.prior.run.q.read(ORIGINAL_DATA/"cases.json")
    for cohort in known.values():base.prior.check_disjoint(rows,cohort)
    sealed=base.prior.run.q.read(base.prior.POOL_META)["sealed_final"]
    for key in base.prior.IDENTITY:
        if {r[key] for r in rows}&{r[key] for r in sealed}:raise ValueError("sealed source metadata overlap")
    proof=base.verify_public(rows)
    native=AutoTokenizer.from_pretrained(base.MODEL_PATH,local_files_only=True)
    gpt=AutoTokenizer.from_pretrained(base.GPT2,local_files_only=True)
    gpt.add_special_tokens({"pad_token":"<|pad|>"});gpt.model_max_length=1_000_000
    inputs=[ORIGINAL_DATA/n for n in ("producer.json","result.json","cases.json","worker_terminal_receipt.json")]
    inputs += [base.PUBLIC_FILE,base.prior.POOL_META]
    inputs += [base.MODEL_PATH/n for n in ("tokenizer.json","tokenizer_config.json","config.json")]
    inputs += [base.GPT2/n for n in ("tokenizer.json","tokenizer_config.json") if (base.GPT2/n).exists()]
    p=producing(DATA,"additional_complete_source_token_offsets",inputs,{"cases":512,"model_execution":False})
    prepared=[]
    for row in rows:
        messages=[{"role":"system","content":base.SYSTEM},
            {"role":"user","content":"<passage>\n"+row["source"]+"\n</passage>"}]
        context=native.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
        encoded=native(context,add_special_tokens=False,return_offsets_mapping=True)
        direct=native.apply_chat_template(messages,add_generation_prompt=True)
        if encoded["input_ids"]!=direct:raise ValueError("canonical source prompt IDs differ")
        start=context.index("<passage>\n")+len("<passage>\n")
        if context[start:start+len(row["source"])]!=row["source"]:raise ValueError("literal full source boundary differs")
        raw=gpt(row["source"],add_special_tokens=False,return_offsets_mapping=True)
        if gpt.decode(raw["input_ids"],skip_special_tokens=False)!=row["source"]:
            raise ValueError("complete GPT2 source round trip failed")
        alignment=base.overlaps(raw["offset_mapping"],encoded["offset_mapping"],start)
        prepared.append({**{k:row[k] for k in (*base.prior.IDENTITY,"source","spans","bin","source_cl100k_tokens")},
            "split":"additional_train_sources","context_ids":direct,"source_ids":raw["input_ids"],
            "alignment":alignment,"alignment_sha256":base.prior.run.q.digest(alignment),
            "native_prompt_sha256":base.prior.run.q.digest(direct),
            "source_frame":base.source_frame(len(raw["input_ids"]),maximum=8192)})
    write_json(DATA/"cases.json",{"additional_train_sources":prepared})
    first=prepared[0];longest=sorted(prepared,key=lambda r:(-len(r["context_ids"]),r["id"]))[0]
    if first["id"]==longest["id"]:raise ValueError("distinct first/maximum native prompt pilot cases required")
    write_json(DATA/"selection.json",{"feature_pilot_ids":[first["id"],longest["id"]]})
    write_json(DATA/"public_proof.json",{"public_file_sha256":base.PUBLIC_SHA256,"exact_sources":proof})
    base.verify(p)
    result={"status":"additional_full_source_metadata_prepared","cases":512,"teacher_labels_read":False,
        "max_native_prompt":max(len(r["context_ids"]) for r in prepared),
        "max_GPT2_source":max(len(r["source_ids"]) for r in prepared),
        "max_source_frame":max(r["source_frame"] for r in prepared),"all_sources_complete":True,
        "cases_sha256":file_hash(DATA/"cases.json"),"task_accuracy":None,"producer_fingerprint":p["fingerprint"]}
    write_json(DATA/"result.json",result);print(json.dumps(result),flush=True)


def configure():
    base.DATA=DATA;base.PILOT=PILOT;base.FEATURES=FEATURES;base.HYPOTHESIS=HYPOTHESIS
    base.producing=producing
    return base


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("stage",choices=["prepare","pilot","features"])
    stage=parser.parse_args().stage
    prepare() if stage=="prepare" else configure().feature_cache(stage=="pilot")
