"""Record the observed whole-turn allocation failure without rerunning selection."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from experiments.structure_net import ncm_clean_source_partitions as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = run.ROOT / "20261024_clean_source_partition_failure_results_v1"
WORKFLOW = run.ROOT / "20261024_clean_source_partition_failure_workflow_v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-clean-source-failure-audit-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-clean-whole-turn-source-availability.md")
MODULE = "experiments.neural_architecture_lab.finalize_ncm_clean_source_partition_failure"
LOG = run.ROOT / "20261023_clean_source_partitions_workflow_v1.worker.log"


def logged_counts(text):
    matches = re.findall(r"ValueError: insufficient disjoint source components for fixed quotas: (\{[^\n]+\})", text)
    if len(matches) != 1:
        raise ValueError("one original complete allocation failure required")
    counts = ast.literal_eval(matches[0])
    if (set(counts) != set(run.FINAL) or any(type(v["available"]) is not int or v["available"] < 0
            or v["required"] != run.FINAL[b] + run.DEVELOPMENT[b] for b, v in counts.items())
            or not any(v["available"] < v["required"] for v in counts.values())):
        raise ValueError("unchanged full quotas and actual component deficit required")
    return counts


def audit():
    run.configure()
    q = run.overlap.study.reference.q
    producer = q.read(run.WORKFLOW / "producer.json")
    run.verify(producer)
    failure = q.read(run.WORKFLOW / "failure.json")
    if (failure["status"] != "stopped_failure_preserved"
            or [(r["stage"], r["exit_code"]) for r in failure["stages"]] != [("pilot", 0), ("prepare", 1)]
            or failure["producer_fingerprint"] != producer["fingerprint"]):
        raise ValueError("actual successful pilot and failed preparation receipts required")
    counts = logged_counts(LOG.read_text())
    if DEST.exists() or REPORT.exists() or any((run.DEST / (s + "_sources.json")).exists()
            for s in ("final_short", "final_long", "development_short", "development_long")):
        raise ValueError("fresh failure archive and no completed source reservations required")
    paths = [LOG, run.WORKFLOW / "failure.json", run.WORKFLOW / "producer.json",
        run.WORKFLOW / "stage_receipts/pilot.json", run.WORKFLOW / "stage_receipts/prepare.json",
        run.PILOT / "result.json", run.PILOT / "worker_terminal_receipt.json",
        run.DEST / "producer.json", run.DEST / "worker_terminal_receipt.json", run.DEST / "progress.json"]
    files = [Path(__file__), PLAN, Path(run.__file__), Path(run.core.__file__), run.PLAN]
    p = run.freeze(DEST, "posthoc_observed_whole_turn_source_availability_failure", [*paths, *files],
        {"original_hypothesis": run.HYPOTHESIS, "recomputed_source_graph": False,
         "measurements": "original frozen worker logged allocation counts and actual terminal receipts",
         "source_counts_independently_recomputed": False, "quotas_changed": False,
         "model_execution": False, "final_test_opened": False})
    p["hypothesis_id"] = run.HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (DEST / "producer_source" / f"availability_failure_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(DEST / "producer.json", p)
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": "insufficient_component_disjoint_whole_turn_sources",
        "logged_available_and_required_components": counts, "original_pilot_exit": 0, "original_prepare_exit": 1,
        "source_reservation_success": False, "source_graph_recomputed_by_audit": False,
        "source_graph_nodes": q.read(run.DEST / "progress.json")["total_nodes"],
        "quotas_unchanged": True, "original_failure_preserved": True, "model_execution": False,
        "task_accuracy": None, "confirmed": False, "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(path): file_hash(path) for path in [*paths, *files, DEST / "producer.json"]}}
    path = DEST / "campaign_results.json"
    write_json(path, campaign)
    table = "\n".join(f"| {b} | {v['available']} | {v['required']} |" for b, v in counts.items())
    REPORT.write_text(f"""# Whole public turns could not fill the clean NCM source quotas

Date: 2026-10-24. Original hypothesis `{run.HYPOTHESIS}`.
Classification **{campaign['classification']}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`.

| Source length band | Available components logged by worker | Required for development and final |
| --- | --- | --- |
{table}

Bands0–4 are192–511,512–767,768–1023,1024–1535 and1536–2048 cl100k tokens.
The actual public-data pilot passed. Full preparation built the76169-node
source graph, then exited1 at fixed-quota allocation. Each of the two longest
bands had16 eligible components, below the required32. No final or development
source exports were created and quotas were not reduced.

This posthoc audit binds the original producing source, immutable inputs,
complete log and actual subprocess receipts. Counts come from the frozen
worker's allocation exception; the graph was not independently recomputed by
this audit. The failure remains unchanged. A separately specified revision may
define exact bounded fragments from complete contiguous public exchanges before
model use, retaining the full128 development and320 final sizes, all input
bands and the95% semantic target. Generic short-turn hashes need special care
to avoid joining unrelated conversations through repeated greetings.

No model training, inference, labels or accuracy was measured. Both original
sealed source files remain closed. This result establishes a source-availability
failure for the whole-turn-only construction, not a student quality result.
""")
    run.verify(p)


def register():
    run.configure()
    register_records(DEST / "campaign_results.json", REPORT, DEST, {run.HYPOTHESIS: {
        "name": "Whole turn source reservation failed full long-context quotas",
        "description": "Frozen76169-node allocation logged16 available versus32 required in both longest bands; original failure preserved and full quotas unchanged.",
        "classification": "insufficient_component_disjoint_whole_turn_sources"}})


def workflow():
    if WORKFLOW.exists() or DEST.exists():
        raise ValueError("fresh observed-failure audit required")
    WORKFLOW.mkdir()
    receipts = []
    for stage in ("audit", "register"):
        child = subprocess.Popen([sys.executable, "-u", "-m", MODULE, stage], env=os.environ)
        code = child.wait()
        receipt = {"stage": stage, "exit_code": code, "child_pid": child.pid,
            "observer_pid": os.getpid(), "observer": "subprocess.wait"}
        receipts.append(receipt)
        write_json(WORKFLOW / (stage + ".json"), receipt)
        if DEST.exists():
            write_json(DEST / ("worker_terminal_receipt.json" if stage == "audit" else "registry_terminal_receipt.json"), receipt)
        if code:
            write_json(WORKFLOW / "failure.json", {"status": "stopped_failure_preserved", "stages": receipts})
            raise RuntimeError("availability failure audit stage failed: " + stage)
    write_json(WORKFLOW / "result.json", {"status": "observed_availability_failure_registered", "stages": receipts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("audit", "register", "workflow"))
    args = parser.parse_args()
    {"audit": audit, "register": register, "workflow": workflow}[args.stage]()
