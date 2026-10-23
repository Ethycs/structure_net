"""CPU-only eligibility audit for new whole-summary training data."""
from collections import Counter
import json
from pathlib import Path

from experiments.structure_net import ncm_learned_codec_corpus as corpus
from experiments.structure_net import ncm_word_final_eval as prior
from experiments.structure_net.ncm_expanded_pairs import public_index,matches_public
from experiments.structure_net.ncm_frozen_body_public_data import PUBLIC_FILE,PUBLIC_SHA256
from experiments.structure_net.ncm_memory_data import file_hash,text_hash,write_json

PLAN=Path("docs/07 - Status Reports/2026-10-23_ncm-additional-training-source-audit.md")
DEST=prior.data.ROOT/"20261023_additional_summary_source_availability_v1"


def band(tokens):
    if not 192<=tokens<=2048:return None
    return min(tokens//512,3)


def main():
    prior.data.prior.dependencies()
    import tiktoken
    prior.data.terminal(prior.data.DATA);prior.data.terminal(corpus.DEST)
    manifest=prior.q.read(corpus.ORIGINAL_DATA/"manifest.json")
    if manifest["source_session_overlap"] or manifest["eligible_input_overlap"]:
        raise ValueError("original training/test partition overlap")
    files=[corpus.ORIGINAL_DATA/name for name in ("native_train.jsonl","train.jsonl")]
    for path in files:
        if file_hash(path)!=manifest["files"][path.name]["sha256"]:
            raise ValueError("original training store identity changed")
    if file_hash(PUBLIC_FILE)!=PUBLIC_SHA256:raise ValueError("public release identity changed")
    origins=prior.q.read(corpus.DEST/"fit_origins.json")
    allowed={o["group_id"] for r in origins for o in r["origins"]}
    if len(allowed)!=1024:raise ValueError("fixed previously admitted1024training groups required")
    prepared=prior.q.read(prior.data.DATA/"cases.json")
    known=[r for rows in prepared.values() for r in rows]
    sealed=prior.q.read(prior.data.prior.POOL_META)["sealed_final"]
    forbidden={k:{r[k] for r in [*known,*sealed]} for k in ("id","source_id","group_id","source_sha256")}
    allowed-=forbidden["group_id"]
    inputs=files+[PUBLIC_FILE,corpus.ORIGINAL_DATA/"manifest.json",corpus.DEST/"fit_origins.json",
        corpus.DEST/"producer.json",corpus.DEST/"result.json",corpus.DEST/"worker_terminal_receipt.json",
        prior.data.DATA/"cases.json",prior.data.DATA/"producer.json",prior.data.DATA/"worker_terminal_receipt.json",
        prior.data.prior.POOL_META,Path(__file__),PLAN]
    p=prior.data.producing(DEST,"CPU_additional_summary_training_source_availability",inputs,
        {"candidate_partition":"original_training_only","prior_codec_fit_groups":1024,
            "remaining_allowed_groups":len(allowed),"new_teacher_calls":0,"model_execution":False,
            "old_targets_used":False,"final_source_files_read":False,"minimum_cl100k_tokens":192})
    for i,path in enumerate((Path(__file__),PLAN)):
        p["source_hashes"][str(path)]=file_hash(path)
        (DEST/"producer_source"/f"additional_source_audit_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=prior.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(DEST/"producer.json",p)
    public=public_index(prior.q.read(PUBLIC_FILE));encoding=tiktoken.get_encoding("cl100k_base")
    index={};counts=Counter()
    for path in files:
        for line_number,line in enumerate(path.open()):
            row=json.loads(line);counts["original_training_rows"]+=1
            if row["split"]!="train":raise ValueError("nontraining record in original training store")
            if row["group_id"] not in allowed or any(row[k] in forbidden[k] for k in forbidden):
                counts["blocked_group_or_identity"]+=1;continue
            key=row["source_sha256"]
            if key in index:counts["duplicate_source"]+=1;continue
            tokens=len(encoding.encode(row["source"],disallowed_special=()))
            bucket=band(tokens)
            if bucket is None:counts["outside_requested_length"]+=1;continue
            if text_hash(row["source"])!=key or not matches_public(row,public):
                counts["not_exact_public_source"]+=1;continue
            index[key]={**{k:row[k] for k in ("id","source_id","group_id","source_sha256")},
                "store":str(path),"line_number":line_number,"source_cl100k_tokens":tokens,"band":bucket}
    eligible=sorted(index.values(),key=lambda r:(r["group_id"],r["id"]))
    write_json(DEST/"eligible_index.json",eligible)
    prior.data.verify(p)
    result={"status":"additional_training_source_availability_audited","eligible_sources":len(eligible),
        "eligible_groups":len({r["group_id"] for r in eligible}),"eligible_sessions":len({r["source_id"] for r in eligible}),
        "counts":dict(counts),"sources_per_band":{str(b):sum(r["band"]==b for r in eligible) for b in range(4)},
        "groups_per_band":{str(b):len({r["group_id"] for r in eligible if r["band"]==b}) for b in range(4)},
        "index_sha256":file_hash(DEST/"eligible_index.json"),"task_accuracy":None,"model_execution":False,
        "final_test_opened":False,"producer_fingerprint":p["fingerprint"]}
    write_json(DEST/"result.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
