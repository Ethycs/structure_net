"""Finalize native representation/capacity controls without task-success claims."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_native_concept_control as concepts
from experiments.structure_net import ncm_native_readout_control as readout
from experiments.structure_net import ncm_native_readout_transfer as transfer
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def checked(root):
    if q.read(root/"worker_terminal_receipt.json")["exit_code"]!=0:
        raise ValueError("completed worker termination required")
    verify(q.read(root/"producer.json"))
    return q.read(root/"result.json")


def artifact_map(roots):
    paths = [Path(__file__),Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    return {str(p):file_hash(p) for p in dict.fromkeys(paths)}


def concept_result():
    result = checked(concepts.FIT); checked(concepts.PILOT)
    if result["status"]!="representation_invariants_passed":
        raise ValueError("completed native concept endpoint required")
    config = torch.load(concepts.FIT/"concepts.pt",weights_only=True)
    for book in config["books"]:
        if not torch.isfinite(book).all() or not torch.equal(book[0],torch.zeros_like(book[0])):
            raise ValueError("native zero code or finite-center identity failed")
    assignments = torch.load(concepts.FIT/"assignments.pt",weights_only=True)
    for split,count in (("train",512),("diagnostic",64)):
        values = assignments[split]
        if len(values)!=5 or any(ids.shape!=(count,width) or not bool(((ids>=0)&(ids<256)).all())
            for ids,width in zip(values,(4,8,16,32,64))):
            raise ValueError("native concept assignment coverage differs")
    dest = ROOT/"20261023_native_concepts_results_v1"
    report = Path("docs/08 - Analysis/2026-10-23_ncm-native-concept-interface.md")
    if dest.exists() or report.exists(): raise ValueError("fresh native concept finalization required")
    dest.mkdir()
    aggregate = {"schema":"nal.ncm-native-concepts.v1","classification":"native_residual_representation_invariants_passed",
        "result":result,"NCM_student_trained":False,"NCM_success_established":False,"task_accuracy_measured":False,
        "student_training_admitted":False,"confirmed":False,"underpowered":True,"final_test_opened":False,"speed_tested":False,
        "artifacts":artifact_map([concepts.PREP,concepts.PILOT,concepts.FIT])}
    campaign = dest/"campaign_results.json"; write_json(campaign,aggregate)
    table = '\n'.join(f"|{width}|{a:.10f}|{b:.10f}|{entries}|" for width,a,b,entries in zip((4,8,16,32,64),
        result['train']['mse_by_level'],result['diagnostic']['mse_by_level'],result['diagnostic']['active_entries_by_level']))
    report.write_text(f'''# Native residual concepts pass representation controls

Date: 2026-10-23. `ncm-native-concept-interface-v1` is complete,
**native_residual_representation_invariants_passed**. This is not a trained
summarizer. No task accuracy,95%NCM result, decoder quality or speed is measured.

Frozen native word embeddings are represented at4,8,16,32,64 cells using256
FP32 residual centers per level, with an exact zero entry. Native word prediction
is a later language stage; these books do not decode text. This is a new
NCM-inspired interface, not the paper's unchanged learned VQ-VAE.

| Concept cells | Raw training embedding MSE | Diagnostic embedding MSE | Diagnostic active entries |
| --- | --- | --- | --- |
{table}

These are squared embedding errors, not token or summary accuracies. The
initial diagnostic error is{result['diagnostic']['initial_mse']:.10f}. Public
training sources:512 for fit,64 separate diagnostic groups. Raw windows are
the first128 native tokens with EOS padding; task examples are never truncated
or evaluated here. No summary labels/dev/final-test cases enter fit.

Zero-code error monotonicity, static/clean-history label parity, experts no
worse than static labels at the same corrupted history, finite centers and
exact center/assignment save/reload passed. Diagnostic same-history repair
gain is{result['diagnostic']['mean_same_history_repair_gain']:.10f} squared
embedding units. This does not establish better generated summaries. K-means
ran32 iterations per book with fixed initialization indices; GPU index-add
floating reductions are not claimed bitwise reproducible across fits.

FP32, physicalGPU2, fixed source/distance chunks. Representative pilot and fit
terminated successfully; own peak allocation{result['peak_allocated_bytes']}bytes.
The original embedding file stayed unchanged. No student run is admitted by
representation MSE alone. Broad training still needs qualified labels and its
own prospective protocol; final256 task cases remain sealed.

Aggregate `{campaign}`; SHA256`{file_hash(campaign)}`. Actual centers, raw ID
windows, assignments, source snapshots and terminal receipts remain append-only.
Registry confirmed=false, task_accuracy_measured=false.
''')
    return dest,report,"ncm-native-concept-interface-v1"


def capacity_gate(free, shuffled):
    return free["cases"]==32 and free["token_accuracy"]>=.98 and free["exact"]>=31 and free["token_accuracy"]-shuffled["token_accuracy"]>=.05


def count_outputs(outputs):
    correct,total = sum(r["correct"] for r in outputs),sum(r["total"] for r in outputs)
    return {"cases":len(outputs),"correct":correct,"total":total,"token_accuracy":correct/total,
        "exact":sum(r["exact"] for r in outputs),"task_accuracy":None,"training_only":True}


def readout_result():
    result = checked(readout.CAPACITY); checked(readout.PILOT)
    replay = checked(transfer.PILOT)
    if replay["status"]!="passed" or not replay["capacity_replay_exact"]:
        raise ValueError("independent frozen-checkpoint replay required")
    free = q.read(transfer.PILOT/"capacity_free_replay.json")
    shuffled = q.read(transfer.PILOT/"capacity_shuffled_replay.json")
    if free!=q.read(readout.CAPACITY/"outputs.json") or count_outputs(free)!=result["free"] or count_outputs(shuffled)!=result["shuffled"]:
        raise ValueError("capacity raw replay and original endpoint disagree")
    passed = capacity_gate(result["free"],result["shuffled"])
    if passed!=(result["status"]=="capacity_gate_passed"):
        raise ValueError("capacity decision differs")
    dest = ROOT/"20261023_native_readout_results_v1"
    report = Path("docs/08 - Analysis/2026-10-23_ncm-native-readout-capacity.md")
    if dest.exists() or report.exists(): raise ValueError("fresh readout capacity finalization required")
    dest.mkdir()
    aggregate = {"schema":"nal.ncm-native-readout-capacity.v1","classification":result["status"],"result":result,
        "capacity_raw_replay_verified":True,"NCM_model":False,"NCM_success_established":False,
        "task_accuracy_measured":False,"confirmed":False,"underpowered":True,
        "final_test_opened":False,"speed_tested":False,"artifacts":artifact_map([readout.PILOT,readout.CAPACITY,transfer.PILOT])}
    campaign = dest/"campaign_results.json"; write_json(campaign,aggregate)
    a,b = result["free"],result["shuffled"]
    report.write_text(f'''# Native readout fits accepted training summaries; transfer remains separate

Date: 2026-10-23. `ncm-native-readout-capacity-v1` is complete,
**{result['status']}**. The fixed checkpoint reproduces{a['exact']}/32 training
texts and{a['correct']}/{a['total']} native target tokens including EOS.
Source shuffling gives{b['correct']}/{b['total']}tokens and{b['exact']}/32 exact
texts. This is training memorization and source dependence, not held-out
summary accuracy or an NCM result. The95%task goal remains unestablished.

The readout retains native tied word embeddings/head, final norm and five
frozen Llama blocks0,7,14,21,27. Per-token original-body states1,14,28 enter a
trainable fusion projection. Only32,656,384 fusion/LoRA parameters trained;
897,352,704 native readout base parameters and all original body/FC weights
stayed unchanged. There is no learned text codec and no parent verifier.

Training used the same32 previously independently accepted summaries for1,024
updates, batch2 and accumulation4, AdamW1e-4. The final checkpoint alone is the
endpoint. Native token round trips, causal masking, gradient routing, real
GPU/BF16 memory, cached/full checks and exact trainable reload passed. Peak
allocation{result['peak_allocated_bytes']}bytes includes full-body extraction.
Training timing on shared hardware is not a speed benchmark.

The producing runner retained only scalar shuffled metrics. A separately
declared no-training replay restored the frozen final checkpoint, matched all
original free token sequences and both metric dictionaries exactly, and saved
the missing raw shuffled predictions. Original artifacts were not rewritten.
This permits audit of the counterfactual without changing the original gate.

The four unseen transfer-pilot outputs are systems checks, not quality evidence.
The separate64-case development evaluation must establish any generalization;
the native AR readout is not itself a coarse-to-fine NCM. Broader training still
requires newly qualified labels. Final256 cases remain sealed.

Aggregate `{campaign}`; SHA256`{file_hash(campaign)}`. Models, source features,
first positive label judgments, raw free/shuffled replay and producer snapshots
remain append-only. Registry confirmed=false, task_accuracy_measured=false.
''')
    return dest,report,"ncm-native-readout-capacity-v1"


def register(kind):
    dest = ROOT/("20261023_native_concepts_results_v1" if kind=="concepts" else "20261023_native_readout_results_v1")
    report = Path("docs/08 - Analysis")/("2026-10-23_ncm-native-concept-interface.md" if kind=="concepts" else "2026-10-23_ncm-native-readout-capacity.md")
    hypothesis = "ncm-native-concept-interface-v1" if kind=="concepts" else "ncm-native-readout-capacity-v1"
    result = q.read(dest/"campaign_results.json")
    register_records(dest/"campaign_results.json",report,dest,{hypothesis:{
        "name":"Native residual concept representation" if kind=="concepts" else "Native-word readout training capacity",
        "description":"Native embedding residual invariants passed on512 raw train and64diagnostic sources; no student accuracy." if kind=="concepts" else
            "Native readout reproduced32/32 accepted training summaries,995/995tokens; source-shuffle25/995, raw replay exact. Training-only, not NCM accuracy.",
        "confirmation_status":result["classification"],"task_accuracy_measured":False,"NCM_success_established":False}})
    print(json.dumps({"status":"registered_and_read_back","hypothesis":hypothesis}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("kind",choices=["concepts","readout"])
    parser.add_argument("command",choices=["finalize","register"]); args = parser.parse_args()
    if args.command=="register": register(args.kind)
    else:
        dest,report,hypothesis = concept_result() if args.kind=="concepts" else readout_result()
        print(json.dumps({"status":"finalized","hypothesis":hypothesis,"report":str(report)}),flush=True)
