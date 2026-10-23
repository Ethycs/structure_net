"""Audit the fresh longer-budget root control; task accuracy stays separate."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_source_root_budget1024 as trial
from experiments.neural_architecture_lab.finalize_ncm_source_root_capacity import recount
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash,write_json

study=trial.base.study
DEST=study.data.ROOT/"20261023_word_final_root_capacity_budget1024_results_v1"
REPORT=Path("docs/08 - Analysis/2026-10-23_ncm-source-root-budget1024.md")


def finalize():
    if DEST.exists() or REPORT.exists():raise ValueError("fresh append-only follow-up finalization required")
    study.data.terminal(trial.TRAIN);study.data.terminal(trial.PILOT)
    trained=study.q.read(trial.TRAIN/"result.json")
    pilot=study.q.read(trial.PILOT/"result.json")
    producer=study.q.read(trial.TRAIN/"producer.json")
    parent=study.q.read(trial.PARENT/"campaign_results.json")
    receipt=study.q.read(trial.PARENT/"ncm-word-final-source-root-capacity-v1_registry_readback.json")
    if (receipt["status"]!="registered_and_read_back"
            or receipt["record"]["campaign_sha256"]!=file_hash(trial.PARENT/"campaign_results.json")
            or parent["classification"]!="source_root_TRAIN_capacity_failed"
            or trained["updates"]!=1024 or trained["status"]!="TRAIN_only_source_root_capacity_completed"
            or producer["hypothesis_id"]!=trial.HYPOTHESIS or producer["config"]["actual_updates"]!=1024
            or producer["config"]["updates"]!=1024 or trained["producer_fingerprint"]!=producer["fingerprint"]
            or not trained["reload_exact"] or not trained["frozen_words_other_heads_and_codec_unchanged"]
            or trained["peak_allocated_bytes"]>8_000_000_000 or trained["task_accuracy"] is not None
            or file_hash(trial.TRAIN/"model.pt")!=trained["checkpoint_sha256"]):
        raise ValueError("actual fixed1024 controls and preserved parent failure required")
    if (trained["initial_parameter_sha256"]!=pilot["initial_parameter_sha256"]
            or trained["frozen_parameter_sha256_unchanged"]!=pilot["frozen_parameter_sha256_unchanged"]
            or any(trial.rate(i)!=trial._rate(i) for i in range(1,17))):
        raise ValueError("fresh shared initialization and unchanged pilot operations/warmup required")
    saved=torch.load(trial.TRAIN/"model.pt",map_location="cpu",weights_only=True,mmap=True)
    if (saved["step"]!=1024 or saved["config"]!=asdict(trial.Config())
            or saved["producer_fingerprint"]!=producer["fingerprint"]):
        raise ValueError("selected checkpoint must be exactly final1024")
    del saved
    rows=trial.base.gap.selection(study.q.read(study.data.DATA/"cases.json")["train"])
    initial=study.q.read(trial.TRAIN/"initial_root_measurements.json")
    final=study.q.read(trial.TRAIN/"final_root_measurements.json")
    previous_initial=study.q.read(trial.base.TRAIN/"initial_root_measurements.json")
    if initial!=previous_initial:raise ValueError("follow-up must start fresh, not from256 or pilot weights")
    baseline=recount(rows,initial);endpoint=recount(rows,final)
    trace=[json.loads(line) for line in (trial.TRAIN/"trace.jsonl").read_text().splitlines()]
    if len(trace)!=1024:raise ValueError("all1024 scheduled updates required")
    sampler=torch.Generator().manual_seed(trial.Config().sampler_seed)
    for step,actual in enumerate(trace,1):
        case=rows[int(torch.randint(16,(1,),generator=sampler))]
        if (actual["step"]!=step or actual["id"]!=case["id"]
                or actual["source_frame"]!=case["source_frame"] or actual["lr"]!=trial.rate(step)
                or any(not math.isfinite(actual[k]) for k in ("loss","gradient_norm","active_CE","padded_CE"))):
            raise ValueError("source/schedule or finite-loss replay differs")
    passed=(endpoint["active_root_recovery"]>=.95 and endpoint["all_root_recovery"]>=.95
        and endpoint["active_root_recovery"]-endpoint["shuffled_active_recovery"]>=.5)
    if (trained["capacity_gate"]!=passed or trained["initial_root_counts"]!=baseline
            or trained["final_root_counts"]!=endpoint):raise ValueError("capacity gate/counts differ")
    classification="source_root_TRAIN_capacity_passed_full_hierarchy_unproven" if passed else "source_root_TRAIN_capacity_failed"
    files=[Path(__file__),Path(trial.__file__),trial.PLAN,trial.ORIGINAL_PLAN,
        Path("experiments/neural_architecture_lab/finalize_ncm_source_root_capacity.py"),
        Path("experiments/neural_architecture_lab/ncm_result_registry.py"),
        trial.PARENT/"campaign_results.json",trial.PARENT/"ncm-word-final-source-root-capacity-v1_registry_readback.json"]
    for root in (trial.TRAIN,trial.PILOT):
        files += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    aggregate={"schema":"nal.ncm-source-root-budget1024.v1","hypothesis_id":trial.HYPOTHESIS,
        "classification":classification,"training":trained,"baseline":baseline,"endpoint":endpoint,
        "parent256_endpoint":parent["endpoint"],"all1024_samples_frames_rates_and_counts_replayed":True,
        "parent_failure_preserved":True,"TRAIN_only":True,"task_accuracy_measured":False,
        "NCM_success_established":False,"confirmed":False,"final_test_opened":False,"speed_tested":False,
        "artifacts":{str(p):file_hash(p) for p in dict.fromkeys(files)}}
    DEST.mkdir();campaign=DEST/"campaign_results.json";write_json(campaign,aggregate)
    table=[]
    for name,m in (("Initial",baseline),("Previous 256",parent["endpoint"]),("Fresh 1,024",endpoint)):
        table.append(f"| {name} | {m['active_correct']}/{m['active_total']} | {m['correct']}/{m['total']} | {m['shuffled_active_correct']}/{m['active_total']} |")
    REPORT.write_text(f'''# Longer-budget source concept capacity check

Date: 2026-10-23. Hypothesis `{trial.HYPOTHESIS}`.
Classification **{classification}**. Aggregate `{campaign}`,
SHA256 `{file_hash(campaign)}`.

These measurements concern fitting the first concept level on 16 training
cases. They are not summary accuracy. The latest complete summarizer remains
at 0/64 short and 0/64 longer development passes. The 95% task goal is unmet.

| State | Active root labels | All root labels | Active labels with shuffled sources |
| --- | --- | --- | --- |
'''+'\n'.join(table)+f'''

The fixed training-capacity gate **{'passed' if passed else 'failed'}**. It
requires at least 95% active and all-root recovery, with an active recovery
drop of at least 50 percentage points under cyclic source shuffling. A pass
does not qualify later concepts, text generation or generalization.

This run started fresh from the same original word-final checkpoint as the
failed 256-update control. Its initial predictions and parameter hash match
exactly. The data, objective, precision, optimizer, dropout and gradient scope
stayed fixed. The budget increased to 1,024 updates and the cosine schedule
stretched after the same 16-update warmup; this changes both budget and schedule.
Only final 1,024 was selected. The existing maximum-frame two-update CUDA pilot
was reused because operations, state shapes and initial warmup rates were exact.
No pilot or 256-update weights were used as initialization.

All 1,024 sampled source identities, full frames, rates, finite loss records
and final root counts replayed. Frozen word modules, other heads and codec
remained exact, as did model/optimizer/RNG/source-only checkpoint reload.
Original Llama weights and MLPs were frozen; the parent was not loaded for this
training. Peak allocation was {trained['peak_allocated_bytes']:,} bytes.

The failed 256-update control and all 930 canonical task judgments remain
unchanged. Both final task cohorts remain closed. No compression or speedup
has been established. Further hierarchy training and source-disjoint complete
summary evaluation require their own fixed protocol.
''')
    print(json.dumps({"status":"finalized","classification":classification,"endpoint":endpoint}),flush=True)


def register():
    aggregate=study.q.read(DEST/"campaign_results.json")
    spec={"name":"Fresh1024-update source concept capacity follow-up on16TRAINcases",
        "description":"Preserved256-update failure; same original initialization and16trainingcases, longer budget/cosine horizon, frozen parent/codec/word modules/nonroot heads. Root fitting and cyclic-source control only; no summaryaccuracy, finalquality or speedclaim.",
        "confirmation_status":aggregate["classification"],"training_updates":1024,"TRAIN_only":True,
        "task_accuracy_measured":False,"NCM_success_established":False,
        "capacity_gate":aggregate["training"]["capacity_gate"]}
    spec.update({prefix+"_"+key:value for prefix in ("baseline","endpoint") for key,value in aggregate[prefix].items()})
    register_records(DEST/"campaign_results.json",REPORT,DEST,{trial.HYPOTHESIS:spec})
    print(json.dumps({"status":"registered_and_read_back"}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__);parser.add_argument("stage",choices=["finalize","register"])
    globals()[parser.parse_args().stage]()
