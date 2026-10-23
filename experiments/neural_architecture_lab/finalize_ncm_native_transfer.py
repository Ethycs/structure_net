"""Record the native readout's complete held-out language/generalization failure."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_native_readout_transfer as t
from experiments.structure_net import ncm_native_readout_control as c
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_source_readout_grade import audit_row, packet_key
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT/"20261023_native_readout_transfer_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-native-readout-transfer.md")


def finalize():
    if DEST.exists() or REPORT.exists(): raise ValueError("fresh transfer finalization required")
    for root in (t.PILOT,t.OUTPUTS,t.JUDGE_PILOT,t.GRADING):
        if q.read(root/"worker_terminal_receipt.json")["exit_code"]!=0: raise ValueError("worker must be terminal")
        verify(q.read(root/"producer.json"))
    result = q.read(t.GRADING/"result.json")["endpoint"]
    outputs = q.read(t.OUTPUTS/"outputs.json")
    scores = q.read(t.GRADING/"secondary_native_readout/case_scores.json")
    packets = q.read(t.GRADING/"secondary_native_readout/packets.json")
    if len(outputs)!=64 or len(scores)!=64 or [r["id"] for r in outputs]!=[r["id"] for r in scores]:
        raise ValueError("fixed64-case denominator/order changed")
    training,_ = c.selected_cases()
    for key in ("id","source_id","group_id","source_sha256"):
        if {r[key] for r in training}&{r[key] for r in outputs}: raise ValueError("training/development overlap")
    cache = q.read(t.GRADING/"judgment_cache.json")
    flags = Counter(); passed = invalid = 0
    for score,packet,output in zip(scores,packets,outputs):
        key = packet_key(packet,"secondary"); entry = cache[key]; row = entry["row"]
        audit_row(packet,"secondary",row)
        if score["packet_key"]!=key or entry["origin"]!=score["first_origin"] or score["text"]!=output["text"]:
            raise ValueError("first judgment or output changed")
        value = q.passes(row["evaluations"][0],score["text"],score["tokens"]) if row["status"]=="valid" else None
        if value!=score["passed"]: raise ValueError("whole-summary score changed")
        passed += value is True; invalid += value is None
        if value is not None: flags.update(k for k in ("faithful","salient","readable") if row["evaluations"][0][k])
    if passed!=result["passed"] or invalid!=result["invalid"] or result["cases"]!=64:
        raise ValueError("endpoint reconstruction changed")
    for packet in q.read(t.JUDGE_PILOT/"secondary_native_readout/packets.json"):
        key = packet_key(packet,"secondary")
        previous = q.read(t.JUDGE_PILOT/"judgment_cache.json")[key]
        if cache[key]!=previous: raise ValueError("pilot negative/first judgment replaced")
    classification = "native_AR_readout_transfer_failed" if passed<61 or invalid else "native_AR_secondary_gate_passed_primary_required"
    roots = [t.PILOT,t.OUTPUTS,t.JUDGE_PILOT,t.GRADING]
    paths = [Path(__file__),Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    DEST.mkdir()
    aggregate = {"schema":"nal.ncm-native-readout-transfer.v1","classification":classification,"endpoint":result,
        "component_passing_counts":dict(flags),"source_group_disjointness_verified":True,"first_judgments_replayed":True,
        "training_exact":32,"training_cases":32,"NCM_model":False,"NCM_success_established":False,
        "confirmed":False,"underpowered":True,"final_test_opened":False,"speed_tested":False,
        "artifacts":{str(p):file_hash(p) for p in dict.fromkeys(paths)}}
    campaign = DEST/"campaign_results.json"; write_json(campaign,aggregate)
    REPORT.write_text(f'''# Five-block native readout fails all64 unseen summaries

Date: 2026-10-23. `ncm-native-readout-transfer-v1` is complete,
**{classification}**. Whole-summary accuracy is **{passed}/64={100*result['task_accuracy']:.1f}%**
under qualified Terra; all64 judgments are valid. Component pass counts:
faithful{flags['faithful']}/64, salient{flags['salient']}/64, readable{flags['readable']}/64.
The95%task objective remains unmet. This is an AR reference, not NCM success.

The same fixed final checkpoint reproduced32/32 accepted training summaries
and995/995 native target tokens. Source shuffling reduced recovery to25/995;
the separate replay matched original outputs/metrics exactly. These controls
establish fit and source use, not generalization. The development outputs are
nonsensical fragments, such as “The products, moving.” for Badlands travel tips.
This is a substantive application failure, not exact-reference disagreement.

All64 development sources are disjoint from training by ID, source session,
group and source hash. Same source prompt, no truncation, greedy seed42,
BF16/SDPA/GPU0, batch2,128-native-token generation ceiling and128-cl100k storage
cap. No further updates or output repair. All four pilot judgments were valid
negatives and were reused;60 new judgments completed the64-case endpoint.
No Sonnet stage ran after the failed61/64 secondary gate.

The original3B body/FC weights and five native readout base blocks stayed frozen.
Copying the vocabulary and selected blocks did not preserve usable language
generation in this trained configuration. These results do not isolate pruning,
bridge initialization and limited data as separate causes. They do reject a
claim that the training fit solves the summarization task.

The new native-parallel NCM pilot passed numerical systems checks, but its
planned full training-only capacity run is deferred by the explicit intact-
language-control amendment. It has no held-out quality result. The next language
component control keeps an entire pretrained small decoder and requires an
initial no-op correction path; it is not a rescue of this0/64 model.

Aggregate `{campaign}`; SHA256`{file_hash(campaign)}`. Fixed checkpoint, source
features, raw64outputs, first requests/responses and source snapshots remain
append-only. Final256task cases stay sealed, registry confirmed=false, no speed
or NCM-success claim.
''')
    print(json.dumps({"status":classification,"passed":passed,"cases":64,"component_passing_counts":dict(flags)}),flush=True)


def register():
    a = q.read(DEST/"campaign_results.json")
    register_records(DEST/"campaign_results.json",REPORT,DEST,{"ncm-native-readout-transfer-v1":{
        "name":"Native AR readout held-out transfer",
        "description":"Native readout fits32/32training texts but0/64unseen summaries pass; faithful/salient/readable all0/64. First pilot negatives preserved. No NCM, final-test or speed success.",
        "confirmation_status":a["classification"],"task_accuracy_measured":True,"NCM_model":False,
        "NCM_success_established":False}})
    print(json.dumps({"status":"registered_and_read_back"}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("command",choices=["finalize","register"])
    globals()[parser.parse_args().command]()
