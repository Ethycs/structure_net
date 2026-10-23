"""Expand long development sources without consuming either sealed cohort."""
import json
from pathlib import Path

from experiments.structure_net import ncm_application_range_sources as pool
from experiments.structure_net import ncm_isolated_quality_data as prior
from experiments.structure_net.ncm_memory_data import file_hash,text_hash,write_json
from experiments.structure_net.ncm_paper_data import ROOT,PREP,DATA,freeze,verify
from experiments.structure_net.ncm_recipe_teacher import verify_public

PLAN=Path("docs/07 - Status Reports/2026-10-23_ncm-word-final-diagnostic-expansion.md")
DEST=ROOT/"20261023_word_final_long_diagnostic_v1"
FAILED=ROOT/"20261023_word_final_preflight_count_failure_v1"


def select_additions(index,blocked,per_bin=12):
    used={key:set(values) for key,values in blocked.items()}
    selected=[]
    for band in (2,3,1,0):
        candidates=sorted((r for r in index if r["bin"]==band),
            key=lambda r:text_hash("ncm-word-final-long-diagnostic-v1:"+r["id"]))
        count=0
        for row in candidates:
            if any(row[key] in used[key] for key in used):continue
            selected.append(row)
            for key in used:used[key].add(row[key])
            count+=1
            if count==per_bin:break
        if count!=per_bin:raise ValueError("insufficient unused disjoint groups in band"+str(band))
    return selected


def main():
    prior.dependencies()
    import tiktoken
    from transformers import AutoTokenizer
    from experiments.structure_net.ncm_frozen_body import MODEL_PATH
    from experiments.structure_net.ncm_native_parent_control import SYSTEM
    prior.terminal(pool.POOL)
    audit=prior.run.q.read(pool.AUDIT/"result.json")
    if file_hash(pool.AUDIT/"eligible_index.json")!=audit["index_sha256"]:
        raise ValueError("original source eligibility index changed")
    metadata=prior.run.q.read(prior.POOL_META)
    earlier=[r for rows in metadata.values() for r in rows]
    old=prior.run.q.read(pool.POOL/"diagnostic_sources.json")
    train=prior.run.q.read(prior.DEST/"cases.json")
    short=prior.run.q.read(PREP/"development_sources.json")
    if len(old)!=16:raise ValueError("inherited16diagnostic sources required")
    origins=prior.run.q.read(ROOT/"20261023_learned_codec_corpus_v1/fit_origins.json")
    fit=[o for r in origins for o in r["origins"]]
    blocked={key:{r[key] for r in [*earlier,*train,*short,*fit] if key in r} for key in pool.IDENTITY}
    chosen=select_additions(prior.run.q.read(pool.AUDIT/"eligible_index.json"),blocked)
    manifest=prior.run.q.read(DATA/"manifest.json")
    if file_hash(DATA/"train.jsonl")!=manifest["files"]["train.jsonl"]["sha256"]:
        raise ValueError("original training source store changed")
    inputs=[pool.AUDIT/n for n in ("producer.json","result.json","eligible_index.json")]
    inputs += [pool.POOL/"diagnostic_sources.json",prior.POOL_META,prior.DEST/"cases.json",PREP/"development_sources.json",
        ROOT/"20261023_learned_codec_corpus_v1/fit_origins.json",DATA/"manifest.json",DATA/"train.jsonl",FAILED/"result.json",Path(__file__),PLAN]
    p=freeze(DEST,"CPU_expand_unlabeled_long_diagnostic_before_model_outputs",inputs,
        {"hypothesis_id":"ncm-word-final-frozen-parent-v1","inherited":16,"new":48,"total":64,
         "new_per_band":12,"band_order":[2,3,1,0],"selection":"SHA256(ncm-word-final-long-diagnostic-v1:+id)",
         "final_texts_opened":False,"task_accuracy":None})
    p["hypothesis_id"]="ncm-word-final-frozen-parent-v1"
    for path in (Path(__file__),PLAN):
        p["source_hashes"][str(path)]=file_hash(path)
        (DEST/"producer_source"/path.name).write_bytes(path.read_bytes())
    p["fingerprint"]=prior.run.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(DEST/"producer.json",p)
    wanted={r["line_number"]:r for r in chosen};recovered={}
    encoding=tiktoken.get_encoding("cl100k_base")
    tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
    with (DATA/"train.jsonl").open() as handle:
        for line_number,line in enumerate(handle):
            if line_number not in wanted:continue
            row=json.loads(line);entry=wanted[line_number]
            tokens=len(encoding.encode(row["source"],disallowed_special=()))
            if (row["split"]!="train" or any(row[k]!=entry[k] for k in pool.IDENTITY)
                    or text_hash(row["source"])!=row["source_sha256"] or pool.source_bin(tokens)!=entry["bin"]):
                raise ValueError("new complete source or band differs")
            prompt=tokenizer.apply_chat_template([{"role":"system","content":SYSTEM},
                {"role":"user","content":"<passage>\n"+row["source"]+"\n</passage>"}],add_generation_prompt=True)
            recovered[row["id"]]={**pool.clean_source(row),"source_cl100k_tokens":tokens,"bin":entry["bin"],
                "native_prompt_tokens":len(prompt),"pool_split":"word_final_long_diagnostic"}
    if len(recovered)!=48:raise ValueError("missing expanded source")
    rows=old+[recovered[r["id"]] for r in chosen]
    proof=verify_public(rows)
    prior.check_disjoint(rows,train+short)
    for key in pool.IDENTITY:
        if {r[key] for r in rows}&{r[key] for r in metadata["sealed_final"]}:
            raise ValueError("sealed cohort overlap")
    if any(sum(r["bin"]==b for r in rows)!=16 for b in range(4)):
        raise ValueError("all four bands require16diagnostic cases")
    write_json(DEST/"sources.json",rows);write_json(DEST/"public_proof.json",proof)
    verify(p)
    result={"status":"expanded_long_diagnostic_sources_prepared","cases":64,"existing_retained":16,
        "new_sources":48,"cases_per_band":[16]*4,"no_targets_created":True,"final_test_opened":False,
        "sources_sha256":file_hash(DEST/"sources.json"),"task_accuracy":None,"producer_fingerprint":p["fingerprint"]}
    write_json(DEST/"result.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
