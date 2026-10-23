"""Expand only the training-group whitelist; retain all held-out exclusions."""
from collections import Counter
import json
from pathlib import Path

from experiments.structure_net import ncm_additional_summary_sources as old
from experiments.structure_net.ncm_memory_data import file_hash,text_hash,write_json

prior=old.prior
PLAN=Path("docs/07 - Status Reports/2026-10-23_ncm-additional-training-source-audit-v2.md")
DEST=prior.data.ROOT/"20261023_additional_summary_source_availability_v2"


def main():
    prior.data.prior.dependencies()
    import tiktoken
    prior.data.terminal(old.DEST);prior.data.terminal(prior.data.DATA)
    previous=prior.q.read(old.DEST/"result.json")
    if previous["eligible_sources"]!=4898 or previous["eligible_groups"]!=1024:
        raise ValueError("preserved restricted availability audit required")
    manifest=prior.q.read(old.corpus.ORIGINAL_DATA/"manifest.json")
    if manifest["source_session_overlap"] or manifest["eligible_input_overlap"]:
        raise ValueError("original training/test source partition proof changed")
    files=[old.corpus.ORIGINAL_DATA/name for name in ("native_train.jsonl","train.jsonl")]
    for path in files:
        if file_hash(path)!=manifest["files"][path.name]["sha256"]:raise ValueError("original training store changed")
    if file_hash(old.PUBLIC_FILE)!=old.PUBLIC_SHA256:raise ValueError("public source corpus changed")
    known=[r for rows in prior.q.read(prior.data.DATA/"cases.json").values() for r in rows]
    sealed=prior.q.read(prior.data.prior.POOL_META)["sealed_final"]
    forbidden={k:{r[k] for r in [*known,*sealed]} for k in ("id","source_id","group_id","source_sha256")}
    source_files=[Path(__file__),PLAN,Path(old.__file__),old.PLAN,
        Path("experiments/structure_net/ncm_expanded_pairs.py")]
    inputs=files+[old.PUBLIC_FILE,old.corpus.ORIGINAL_DATA/"manifest.json",prior.data.DATA/"cases.json",
        prior.data.DATA/"producer.json",prior.data.DATA/"worker_terminal_receipt.json",prior.data.prior.POOL_META]
    inputs += [old.DEST/n for n in ("producer.json","result.json","eligible_index.json","worker_terminal_receipt.json")]
    p=prior.data.producing(DEST,"CPU_all_original_training_group_summary_source_availability",inputs+source_files,
        {"candidate_partition":"all_original_training_groups","model_execution":False,"old_targets_used":False,
            "whitelist_change":"remove_previous_codec_fit_group_restriction","final_source_files_read":False})
    for i,path in enumerate(source_files):
        p["source_hashes"][str(path)]=file_hash(path)
        (DEST/"producer_source"/f"broader_source_audit_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=prior.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(DEST/"producer.json",p)
    public=old.public_index(prior.q.read(old.PUBLIC_FILE));encoding=tiktoken.get_encoding("cl100k_base")
    index={};counts=Counter()
    for path in files:
        for line_number,line in enumerate(path.open()):
            row=json.loads(line);counts["original_training_rows"]+=1
            if row["split"]!="train":raise ValueError("nontraining record in original store")
            if any(row[k] in forbidden[k] for k in forbidden):counts["blocked_identity"]+=1;continue
            key=row["source_sha256"]
            if key in index:counts["duplicate_source"]+=1;continue
            tokens=len(encoding.encode(row["source"],disallowed_special=()));bucket=old.band(tokens)
            if bucket is None:counts["outside_requested_length"]+=1;continue
            if text_hash(row["source"])!=key or not old.matches_public(row,public):
                counts["not_exact_public_source"]+=1;continue
            index[key]={**{k:row[k] for k in ("id","source_id","group_id","source_sha256")},
                "store":str(path),"line_number":line_number,"source_cl100k_tokens":tokens,"band":bucket}
    eligible=sorted(index.values(),key=lambda r:(r["group_id"],r["id"]))
    write_json(DEST/"eligible_index.json",eligible);prior.data.verify(p)
    result={"status":"additional_training_source_availability_audited","eligible_sources":len(eligible),
        "eligible_groups":len({r["group_id"] for r in eligible}),"eligible_sessions":len({r["source_id"] for r in eligible}),
        "counts":dict(counts),"sources_per_band":{str(b):sum(r["band"]==b for r in eligible) for b in range(4)},
        "groups_per_band":{str(b):len({r["group_id"] for r in eligible if r["band"]==b}) for b in range(4)},
        "index_sha256":file_hash(DEST/"eligible_index.json"),"task_accuracy":None,"model_execution":False,
        "final_test_opened":False,"producer_fingerprint":p["fingerprint"]}
    write_json(DEST/"result.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
