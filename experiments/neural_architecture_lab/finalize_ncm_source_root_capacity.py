"""Archive isolated root fitting without calling TRAIN recovery task accuracy."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_source_root_capacity as trial
from experiments.structure_net.ncm_memory_data import file_hash,write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records

study=trial.study
DEST=study.data.ROOT/"20261023_word_final_root_capacity_results_v1"
REPORT=Path("docs/08 - Analysis/2026-10-23_ncm-source-root-capacity.md")


def recount(rows,measurement):
    outputs=measurement["outputs"]
    if len(outputs)!=16 or [r["id"] for r in rows]!=[r["id"] for r in outputs]:
        raise ValueError("fixed16TRAINsource identities required")
    counts={"correct":0,"total":0,"active_correct":0,"active_total":0,"shuffled_active_correct":0}
    for i,(case,output) in enumerate(zip(rows,outputs)):
        if len(output["codes"])!=16 or any(type(c) is not int or not 0<=c<4096 for c in output["codes"]):
            raise ValueError("all16 valid root IDs required")
        labels=case["concept_ids"][:16]
        block_active=[any(t!=50257 for t in case["word_ids"][start:start+48]) for start in range(0,192,48)]
        active=[flag for flag in block_active for _ in range(4)]
        shuffled=outputs[(i+1)%16]["codes"]
        counts["correct"]+=sum(a==b for a,b in zip(output["codes"],labels));counts["total"]+=16
        counts["active_correct"]+=sum(flag and a==b for a,b,flag in zip(output["codes"],labels,active))
        counts["active_total"]+=sum(active)
        counts["shuffled_active_correct"]+=sum(flag and a==b for a,b,flag in zip(shuffled,labels,active))
    counts.update(all_root_recovery=counts["correct"]/counts["total"],
        active_root_recovery=counts["active_correct"]/counts["active_total"],
        shuffled_active_recovery=counts["shuffled_active_correct"]/counts["active_total"])
    if measurement["task_accuracy"] is not None or counts!=measurement["counts"]:
        raise ValueError("TRAIN-only literal root counts differ")
    return counts


def finalize():
    if DEST.exists() or REPORT.exists():raise ValueError("fresh append-only root capacity finalization required")
    for root in (trial.PILOT,trial.TRAIN):study.data.terminal(root)
    pilot=study.q.read(trial.PILOT/"result.json");trained=study.q.read(trial.TRAIN/"result.json")
    if (pilot["status"]!="systems_source_root_pilot_passed" or pilot["updates"]!=2
            or trained["status"]!="TRAIN_only_source_root_capacity_completed" or trained["updates"]!=256
            or pilot["initial_parameter_sha256"]!=trained["initial_parameter_sha256"]
            or not all(r["frozen_words_other_heads_and_codec_unchanged"] and r["reload_exact"] for r in (pilot,trained))
            or trained["peak_allocated_bytes"]>8_000_000_000 or trained["task_accuracy"] is not None
            or file_hash(trial.TRAIN/"model.pt")!=trained["checkpoint_sha256"]):
        raise ValueError("fixed256fresh initialization/frozen/reload/memory controls required")
    rows=trial.gap.selection(study.q.read(study.data.DATA/"cases.json")["train"])
    initial=study.q.read(trial.TRAIN/"initial_root_measurements.json")
    final=study.q.read(trial.TRAIN/"final_root_measurements.json")
    if initial!=study.q.read(trial.PILOT/"initial_root_measurements.json"):
        raise ValueError("full capacity run must start from the same pre-pilot state")
    baseline=recount(rows,initial);endpoint=recount(rows,final)
    original=study.q.read(trial.gap.DEST/"case_measurements.json")
    if initial["outputs"]!=[{"id":r["id"],"codes":r["source_only_codes"][0]} for r in original]:
        raise ValueError("root baseline must reproduce failed word-final checkpoint exactly")
    trace=[json.loads(line) for line in (trial.TRAIN/"trace.jsonl").read_text().splitlines()]
    sampler=torch.Generator().manual_seed(trial.Config().sampler_seed)
    if len(trace)!=256:raise ValueError("all256capacity updates required")
    for step,row in enumerate(trace,1):
        case=rows[int(torch.randint(16,(1,),generator=sampler))]
        if row["step"]!=step or row["id"]!=case["id"] or row["source_frame"]!=case["source_frame"] or row["lr"]!=trial.rate(step):
            raise ValueError("source/frame/sampling/LR replay differs")
    passed=(endpoint["active_root_recovery"]>=.95 and endpoint["all_root_recovery"]>=.95
        and endpoint["active_root_recovery"]-endpoint["shuffled_active_recovery"]>=.5)
    if trained["capacity_gate"]!=passed or trained["initial_root_counts"]!=baseline or trained["final_root_counts"]!=endpoint:
        raise ValueError("root capacity gate differs")
    classification="source_root_TRAIN_capacity_passed_full_hierarchy_unproven" if passed else "source_root_TRAIN_capacity_failed"
    files=[Path(__file__),Path(trial.__file__),trial.PLAN,
        Path("experiments/neural_architecture_lab/ncm_result_registry.py"),
        trial.COMPLETED/"campaign_results.json",trial.COMPLETED/"ncm-word-final-frozen-parent-v1_registry_readback.json"]
    for root in (trial.PILOT,trial.TRAIN):
        files += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    aggregate={"schema":"nal.ncm-source-root-capacity.v1","hypothesis_id":trial.HYPOTHESIS,
        "classification":classification,"pilot":pilot,"training":trained,"baseline":baseline,"endpoint":endpoint,
        "sampling_frames_LRs_and_counts_replayed":True,"TRAIN_only":True,"task_accuracy_measured":False,
        "NCM_success_established":False,"confirmed":False,"final_test_opened":False,"speed_tested":False,
        "artifacts":{str(p):file_hash(p) for p in dict.fromkeys(files)}}
    DEST.mkdir();campaign=DEST/"campaign_results.json";write_json(campaign,aggregate)
    table=[]
    for name,m in (("initial",baseline),("final",endpoint)):
        table.append(f"| {name} | {m['active_correct']}/{m['active_total']} | {m['correct']}/{m['total']} | {m['shuffled_active_correct']}/{m['active_total']} |")
    REPORT.write_text(f'''# Source-conditioned root learning on16TRAINcases

Date:2026-10-23. Hypothesis `{trial.HYPOTHESIS}`.
Classification **{classification}**. Aggregate `{campaign}`,
SHA256 `{file_hash(campaign)}`.

This is a TRAIN-only root-classification capacity check. It does not measure
summary accuracy. The preceding word-final model remains0/64short and0/64long
complete-summary passes. The95%task goal remains unmet; both final cohorts
remain closed and no speedup is established.

| State | Active-block root labels | All root labels | Active labels with wrong sources |
| --- | --- | --- | --- |
'''+'\n'.join(table)+f'''

The fixed capacity gate {'passed' if passed else 'failed'}: at least95%active
and95%all-root recovery, with at least50percentage points lost when sources are
cyclically shuffled. Only the final256-update checkpoint was selected. Source
shuffling scores each case's labels against the next complete source's
predictions; it is a negative control, not held-out generalization.

The model started from the same failed final512checkpoint as its two-update
pilot. Both reproduce all16initial source-only root sequences exactly. The
fixed selection contains four TRAINcases in each of four source-length bands.
Only the new source bridge, shared generator, root queries and root head trained;
original Llama weights/MLPs, codec, word projection/decoder and other concept
heads stayed frozen. The original parent was not loaded. Full source features
were retained with explicit masks and no source or target clipping.

Training used256updates, batch1, active-root CE+.25padded-root CE, FP32master
weights and AdamW with BF16autocast/FP32loss. The16-update warmup reached3e-4,
then cosine decay ended at3e-5. Scope, weighting, learning rate and data size all
differ from the earlier joint run; this does not isolate one causal change.
No word loss or target history enters root prediction. All256source samples,
frames and learning rates, full raw root IDs, counts and shuffled controls
replayed. Frozen parameter/codec hashes and model/optimizer/RNG/source-only
reload passed. Peak training allocation was{trained['peak_allocated_bytes']:,}
bytes. The pilot exercised the2,816-position maximum source frame.

A capacity pass can inform a separately frozen full-hierarchy training
experiment. It does not qualify the later concept levels, word decoder,
source-to-summary quality, task speed or model generalization. Preserve the
earlier full128-case failure and all its first judgments unchanged.
''')
    print(json.dumps({"status":"finalized","classification":classification,"endpoint":endpoint}),flush=True)


def register():
    aggregate=study.q.read(DEST/"campaign_results.json")
    spec={"name":"Isolated frozen-parent source-to-root learning on16TRAINcases",
        "description":"After archived0/128word-final NCM quality, directly fit the source/root path on16fixedTRAINcases for256updates; word modules, other heads, originalparent andcodec frozen. Full-source predictions with cyclic-source negativecontrol; componentcapacity only, no taskaccuracy/speed claim.",
        "confirmation_status":aggregate["classification"],"TRAIN_only":True,"task_accuracy_measured":False,
        "NCM_success_established":False,"training_updates":256,"capacity_gate":aggregate["training"]["capacity_gate"]}
    spec.update({prefix+"_"+key:value for prefix in ("baseline","endpoint") for key,value in aggregate[prefix].items()})
    register_records(DEST/"campaign_results.json",REPORT,DEST,{trial.HYPOTHESIS:spec})
    print(json.dumps({"status":"registered_and_read_back"}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("stage",choices=["finalize","register"])
    globals()[parser.parse_args().stage]()
