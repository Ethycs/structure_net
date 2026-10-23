"""Finalize a failed native-parent competence control without NCM success claims."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net import ncm_native_parent_control as n
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_source_readout_grade import audit_row, packet_key
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_native_parent_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-native-parent-competence.md")


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh native-parent finalization required")
    receipt = q.read(n.GRADING/"worker_terminal_receipt.json")
    if receipt["exit_code"] != 0 or receipt["status"] != "observed_worker_terminal":
        raise ValueError("grading worker must be terminal")
    result = q.read(n.GRADING/"result.json")
    outputs = q.read(n.CONTROL/"outputs.json")
    cache = q.read(n.GRADING/"judgment_cache.json")
    for key,entry in cache.items():
        if key != packet_key(entry["packet"],entry["role"]):
            raise ValueError("cache identity changed")
        audit_row(entry["packet"],entry["role"],entry["row"])
    flags, roles = {}, []
    for role in ("secondary","primary"):
        endpoint = result[role]
        if endpoint is None:
            if role != "primary" or result["secondary"]["passed"] >= 61:
                raise ValueError("required endpoint absent")
            continue
        if role == "primary" and result["secondary"]["passed"] < 61:
            raise ValueError("primary executed after failed gate")
        root = n.GRADING/(role+"_native_parent"); roles.append(root)
        scores, packets = q.read(root/"case_scores.json"), q.read(root/"packets.json")
        if len(scores) != 64 or [r["id"] for r in scores] != [r["id"] for r in outputs]:
            raise ValueError("case denominator/order changed")
        counts = Counter(); failures = Counter()
        for score,packet,output in zip(scores,packets,outputs):
            entry = cache[packet_key(packet,role)]
            row = entry["row"]
            passed = q.passes(row["evaluations"][0],score["text"],score["tokens"]) if row["status"] == "valid" else None
            if passed != score["passed"] or score["text"] != output["text"] or entry["origin"] != score["first_origin"]:
                raise ValueError("score or first judgment changed")
            counts["passed"] += passed is True; counts["invalid"] += passed is None
            if passed is not None:
                failures.update(k for k in ("faithful","salient","readable") if not row["evaluations"][0][k])
        if counts["passed"] != endpoint["passed"] or counts["invalid"] != endpoint["invalid"]:
            raise ValueError("endpoint reconstruction changed")
        flags[role] = dict(failures)
    roots = [n.PILOT,n.CONTROL,n.GRADING,*roles]
    for root in roots:
        verify(q.read(root/"producer.json"))
    if file_hash(n.CONTROL/"outputs.json") != q.read(n.CONTROL/"result.json")["output_sha256"]:
        raise ValueError("native output changed")
    classification = "native_parent_reference_passed" if result["native_reference_gate_passed"] else "native_parent_task_gate_failed"
    paths = [Path(__file__),Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    DEST.mkdir()
    aggregate = {"schema":"nal.ncm-native-parent.v1","classification":classification,"grading":result,
        "failed_flags":flags,"NCM_model":False,"NCM_success_established":False,"confirmed":False,
        "student_training_admitted":False,"final_test_opened":False,"speed_tested":False,
        "first_judgments_replayed":True,"artifacts":{str(p):file_hash(p) for p in dict.fromkeys(paths)}}
    campaign = DEST/"campaign_results.json"; write_json(campaign,aggregate)
    e = result["secondary"]
    REPORT.write_text(f'''# Native parent still fails the summary competence gate

Date: 2026-10-23. `{n.HYPOTHESIS}` is complete, **{classification}**.
The fixed native Llama configuration passes **{e['passed']}/64={100*e['task_accuracy']:.4f}%**
under qualified Terra, all64 judgments valid. Wilson95% interval:
{100*e['wilson_95'][0]:.2f}–{100*e['wilson_95'][1]:.2f}%. The61/64 gate fails;
independent Sonnet grading did not run. This is the original full28-layer
Llama-3.2-3B-Instruct, not a trained NCM model. NCM95% and speed remain unproved.

The stricter source-attribution/qualifier instruction and128-native-token
generation allowance were fixed before inference. The same64 source-disjoint
development passages, unchanged faithful/salient/readable rubric and128-cl100k
storage cap remain. No clipping, case exclusion or semantic retries. Failure
flags are `{flags}`. Exact request/parser/first-judgment origins were replayed.

The original generic parent passed50/64. This combined prompt/budget change
passes{e['passed']}/64 and does not establish improvement or isolate either
change. It also does not prove the frozen representations cannot support a
separately trained readout. Parent outputs are not qualified as95% teacher
labels. A stronger teacher-label intervention needs an independent audit.

BF16/SDPA, physicalGPU0, batch4, greedy, seed42. Original body hash and tied
native vocabulary stayed unchanged; real-memory pilot, repeated greedy output
and serialization passed. Main peak allocation:
{q.read(n.CONTROL/'result.json')['peak_allocated_bytes']}bytes. The generator's
original execution handle was lost in a truncated tool response; a separate
receipt records observed process disappearance and completed result, without
inventing an exit code. The grading worker's exit0 was directly observed.

Final256 task cases remain sealed. No model training, NCM success or matched
speed claim is admitted. Aggregate `{campaign}`;
SHA256 `{file_hash(campaign)}`. All raw outputs, first responses and source
snapshots remain append-only. Registry confirmed=false.
''')
    print(json.dumps({"classification":classification,"passed":e["passed"],"cases":64,"report":str(REPORT)}),flush=True)


def register():
    a = q.read(DEST/"campaign_results.json")
    register_records(DEST/"campaign_results.json",REPORT,DEST,{n.HYPOTHESIS:{
        "name":"Native-word Llama summary competence control",
        "description":"Stricter native Llama parent49/64 under Terra; fails95% whole-summary gate. Frozen full28-layer parent, not NCM; no speed or final-test claim.",
        "confirmation_status":a["classification"],"task_accuracy_measured":True,"NCM_trained":False,
        "NCM_success_established":False}})
    print(json.dumps({"status":"registered_and_read_back"}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__); parser.add_argument("command",choices=["finalize","register"])
    globals()[parser.parse_args().command]()
