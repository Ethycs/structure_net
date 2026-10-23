"""TRAIN-only component measurements at gold versus generated concept histories."""
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_word_final_eval as study
from experiments.structure_net.ncm_memory_data import file_hash,text_hash,write_json

PLAN=Path("docs/07 - Status Reports/2026-10-23_ncm-word-final-generation-gap-plan.md")
DEST=study.data.ROOT/"20261023_word_final_generation_gap_v1"


def selection(rows):
    if len(rows)!=252:raise ValueError("fixed252TRAINcases required")
    selected=[]
    for band in range(4):
        part=[r for r in rows if min(r["source_cl100k_tokens"]//512,3)==band]
        if len(part)<4:raise ValueError("four TRAINcases in every fixed length band required")
        selected.extend(sorted(part,key=lambda r:text_hash("ncm-word-final-generation-gap-v1:"+r["id"]))[:4])
    if len({r["id"] for r in selected})!=16:raise ValueError("sixteen unique TRAINcases required")
    return selected


def count_words(actual,target):
    mask=target!=50257
    return {"correct":int(((actual==target)&mask).sum()),"total":int(mask.sum())}


def merge_counts(rows):
    output={key:{"correct":sum(r[key]["correct"] for r in rows),"total":sum(r[key]["total"] for r in rows)}
        for key in ("gold_conditioned_words","source_only_words","own_conditioning_forward_words","generation_forward_agreement")}
    for value in output.values():value["fraction"]=value["correct"]/value["total"] if value["total"] else None
    output["concepts"]=[]
    for level in range(8):
        columns=sorted(rows[0]["concepts"][level].keys()-{"level"})
        output["concepts"].append({"level":level,**{k:sum(r["concepts"][level][k] for r in rows) for k in columns}})
    return output


@torch.no_grad()
def measure(net,row,manifest,neighbors):
    batch=study.recovery.base.batch_for(row,manifest)
    study.set_source_frame(net.generator,row["source_frame"],maximum=8192)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        generated=net.generate(batch["features"],batch["mask"],batch["source_ids"])
        gold=net(batch["features"],batch["mask"],batch["codes"],batch["source_ids"])
    if not torch.isfinite(gold["word_logits"]).all() or any(not torch.isfinite(v).all() for v in gold["coarse_logits"]):
        raise ValueError("nonfinite gold-conditioned output")
    gold_words=gold["word_logits"][...,:50257].argmax(-1).cpu()
    gold_coarse=[v.argmax(-1).cpu() for v in gold["coarse_logits"]]
    del gold
    own=torch.cat((*generated["coarse_codes"],torch.zeros(1,192,dtype=torch.long,device="cuda")),dim=1)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        forwarded=net(batch["features"],batch["mask"],own,batch["source_ids"])
    if not torch.isfinite(forwarded["word_logits"]).all():raise ValueError("nonfinite own-conditioned output")
    own_words=forwarded["word_logits"][...,:50257].argmax(-1).cpu()
    raw=generated["raw_word_ids"].cpu();target=batch["words"].cpu()
    own_coarse=[v.cpu() for v in generated["coarse_codes"]]
    concepts=[];position=0
    active=(target.reshape(4,48)!=50257).any(-1)
    for level,width in enumerate(net.generator.levels[:-1]):
        labels=batch["codes"][:,position:position+width].cpu()
        mask=active.repeat_interleave(width//4)[None]
        actual=own_coarse[level];teacher=gold_coarse[level]
        wrong=(actual!=labels)&mask
        covered=(neighbors[level][labels]==actual[...,None]).any(-1)&wrong
        concepts.append({"level":level,"total":labels.numel(),"active_total":int(mask.sum()),
            "gold_conditioned_correct":int((teacher==labels).sum()),
            "source_only_correct":int((actual==labels).sum()),
            "gold_conditioned_active_correct":int(((teacher==labels)&mask).sum()),
            "source_only_active_correct":int(((actual==labels)&mask).sum()),
            "source_only_active_wrong":int(wrong.sum()),"wrong_within_eight_neighbors":int(covered.sum())})
        position+=width
    return {"id":row["id"],"source_id":row["source_id"],"source_sha256":row["source_sha256"],
        "source_cl100k_tokens":row["source_cl100k_tokens"],"source_frame":row["source_frame"],
        "gold_conditioned_words":count_words(gold_words,target),"source_only_words":count_words(raw,target),
        "own_conditioning_forward_words":count_words(own_words,target),
        "generation_forward_agreement":{"correct":int((raw==own_words).sum()),"total":192},
        "raw_word_ids":raw[0].tolist(),"gold_conditioned_word_ids":gold_words[0].tolist(),
        "own_conditioning_forward_word_ids":own_words[0].tolist(),"target_word_ids":target[0].tolist(),
        "source_only_codes":[v[0].tolist() for v in own_coarse],
        "gold_conditioned_codes":[v[0].tolist() for v in gold_coarse],"concepts":concepts}


def main():
    study.data.require_gpu();study.data.terminal(study.OUTPUTS);study.data.terminal(study.data.DATA)
    selected=selection(study.q.read(study.data.DATA/"cases.json")["train"])
    manifest=study.q.read(study.data.FEATURES/"manifest.json")
    inputs=[study.data.DATA/n for n in ("producer.json","result.json","cases.json","worker_terminal_receipt.json")]
    inputs += [study.recovery.TRAIN/n for n in ("producer.json","result.json","model.pt","neighbors.pt","worker_terminal_receipt.json")]
    inputs += [study.OUTPUTS/n for n in ("producer.json","result.json","outputs.json","worker_terminal_receipt.json")]
    files=[Path(__file__),PLAN]
    for row in selected:
        entry=manifest[row["id"]]
        if file_hash(entry["path"])!=entry["sha256"]:raise ValueError("frozen TRAINsource feature changed")
        inputs.append(Path(entry["path"]))
    p=study.producing(DEST,"TRAIN_only_gold_vs_generated_concept_diagnostic",inputs+files,
        {"diagnostic_id":"ncm-word-final-generation-gap-v1","cases":16,"training_updates":0,
            "TRAIN_only":True,"model_quality_measured":False,"task_accuracy":None})
    p["config"].update(source_only=False,source_prediction=True,oracle_target_codes=True,
        arms=["gold_conditioning","source_only_generation","forward_at_generated_conditioning"])
    for i,path in enumerate(files):
        p["source_hashes"][str(path)]=file_hash(path)
        (DEST/"producer_source"/f"generation_gap_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"]=study.q.digest({k:v for k,v in p.items() if k!="fingerprint"})
    write_json(DEST/"producer.json",p);write_json(DEST/"selection.json",[{k:r[k] for k in
        ("id","source_id","source_sha256","source_cl100k_tokens")} for r in selected])
    net,codec,_=study.checkpoint()
    before=study.recovery.base.tensor_digest(dict(net.named_parameters()));frozen=study.core.released.module_hash(codec)
    neighbors=torch.load(study.recovery.TRAIN/"neighbors.pt",map_location="cpu",weights_only=True)
    rows=[]
    for row in selected:
        rows.append(measure(net,row,manifest,neighbors));write_json(DEST/"case_measurements.json",rows)
        print(json.dumps({"measured":len(rows),"total":16}),flush=True)
    if study.recovery.base.tensor_digest(dict(net.named_parameters()))!=before or study.core.released.module_hash(codec)!=frozen:
        raise ValueError("diagnostic must not change model or codec")
    study.data.verify(p)
    result={"status":"TRAIN_only_generation_gap_measured","cases":16,"components":merge_counts(rows),
        "original_model_and_codec_unchanged":True,"task_accuracy":None,"complete_summary_quality_measured":False,
        "final_test_opened":False,"speed_tested":False,"producer_fingerprint":p["fingerprint"],
        "case_measurements_sha256":file_hash(DEST/"case_measurements.json")}
    write_json(DEST/"result.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":main()
