"""Reserve additional TRAIN metadata without opening selected final sources."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

from experiments.structure_net import ncm_clean_source_graph as graph
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT = Path("data/experiments/ncm_paper_regime")
PARENT = ROOT / "20261024_clean_exchange_partitions_v2"
ARCHIVE = ROOT / "20261024_clean_exchange_partitions_results_v2"
PARENT_HYPOTHESIS = "ncm-clean-exchange-source-partitions-v2"
PARENT_SHA = "96f602f2f687a589f573a6e5966f07d0acee9a3b43d52e9184ec186359e17538"
HYPOTHESIS = "ncm-additional10000-training-source-reservation-v1"
MODULE = "experiments.structure_net.ncm_additional_training_reservation"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-additional-training-reservation-plan.md")
TEST = Path("tests/structure_net/test_ncm_additional_training_reservation.py")
AUDITOR = Path("experiments/neural_architecture_lab/finalize_ncm_additional_training_reservation.py")
PILOT = ROOT / "20261024_additional10000_training_reservation_pilot_v1"
DEST = ROOT / "20261024_additional10000_training_reservation_v1"
WORKFLOW = ROOT / "20261024_additional10000_training_reservation_workflow_v1"
QUOTAS = {0: 5000, 1: 1250, 2: 1250, 3: 1250, 4: 1250}
MAX_RSS = 8_000_000_000


def read(path):
    return json.loads(Path(path).read_text())


def parent_metadata():
    campaign = ARCHIVE / "campaign_results.json"
    receipt = ARCHIVE / (PARENT_HYPOTHESIS + "_registry_readback.json")
    if file_hash(campaign) != PARENT_SHA:
        raise ValueError("registered unchanged parent reservation required")
    data, proof = read(campaign), read(receipt)
    if (data["classification"] != "clean_exchange_sources_reserved_quality_unmeasured"
            or proof["record"]["campaign_sha256"] != PARENT_SHA or proof["exact"]["campaign_sha256"] != PARENT_SHA):
        raise ValueError("registered parent identity differs")
    paths = [PARENT / name for name in ("component_index.json", "selection_metadata.json", "result.json", "worker_terminal_receipt.json")]
    for path in paths:
        if data["artifacts"].get(str(path)) != file_hash(path):
            raise ValueError("audited metadata or original worker receipt changed")
    if read(paths[-1])["exit_code"] != 0:
        raise ValueError("actual original reservation exit required")
    index, selected = read(paths[0]), read(paths[1])
    if len(index) != 131762 or {k: len(v) for k, v in selected.items()} != {
            "final_short": 256, "final_long": 64, "development_short": 64, "development_long": 64}:
        raise ValueError("full original graph metadata and448 reserved components required")
    reserved = {row["source_component"] for rows in selected.values() for row in rows}
    if len(reserved) != 448:
        raise ValueError("all reserved components must be distinct")
    return index, reserved, [campaign, receipt, *paths]


def choose(index, reserved, quotas):
    components = defaultdict(lambda: {"blocked": False, "rows": []})
    ids, hashes, blocked = set(), set(), set()
    for row in index:
        if "source" in row or "target_text" in row or "text" in row:
            raise ValueError("metadata only; no source or target text")
        if row["id"] in ids or row["source_sha256"] in hashes or type(row["blocked"]) is not bool:
            raise ValueError("unique exact graph row metadata required")
        ids.add(row["id"])
        hashes.add(row["source_sha256"])
        if row["blocked"]:
            blocked.add(row["source_component"])
    for row in index:
        component = row["source_component"]
        if (component not in blocked and component not in reserved and row["candidate"]
                and row["bin"] in quotas and graph.bucket(row["tokens"]) == row["bin"] and row["public_origins"]):
            components[component]["rows"].append(row)
    selected, availability = graph.assign_components(dict(components), quotas, HYPOTHESIS)
    if (set(selected) & (reserved | blocked) or len(selected) != sum(quotas.values())
            or Counter(row["bin"] for row in selected.values()) != Counter(quotas)):
        raise ValueError("complete distinct unblocked unreserved training quotas required")
    values = [selected[key] for key in sorted(selected)]
    return values, {"eligible_components": len(components), "available_components_by_bin": availability,
        "all_reserved_components_excluded": len(reserved), "blocked_components_excluded": len(blocked)}


def produce(root, pilot, paths):
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(graph.__file__), Path(graph.guard.__file__),
             Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    root.mkdir(exist_ok=False)
    (root / "producer_source").mkdir()
    for i, path in enumerate(files):
        (root / "producer_source" / f"{i}_{path.name}").write_bytes(path.read_bytes())
    p = {"hypothesis_id": HYPOTHESIS, "pilot": pilot,
        "source_hashes": {str(path): file_hash(path) for path in files},
        "input_hashes": {str(path): file_hash(path) for path in paths},
        "config": {"metadata_only": True, "selected_final_sources_read_or_hashed": False,
            "new_labels_generated": False, "model_execution": False, "task_accuracy": None,
            "quotas": {k: 2 for k in QUOTAS} if pilot else QUOTAS}}
    write_json(root / "producer.json", p)
    return p


def verify(producer):
    for key in ("source_hashes", "input_hashes"):
        if any(file_hash(path) != expected for path, expected in producer[key].items()):
            raise ValueError("immutable reservation producer or metadata input changed")


def run(pilot=False):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU metadata reservation only")
    index, reserved, paths = parent_metadata()
    root = PILOT if pilot else DEST
    if not pilot:
        if read(PILOT / "worker_terminal_receipt.json")["exit_code"] != 0 or read(PILOT / "result.json")["cases"] != 10:
            raise ValueError("actual representative metadata pilot required")
        paths += [PILOT / n for n in ("producer.json", "result.json", "training_metadata.json", "worker_terminal_receipt.json")]
    p = produce(root, pilot, paths)
    quotas = {k: 2 for k in QUOTAS} if pilot else QUOTAS
    values, availability = choose(index, reserved, quotas)
    write_json(root / "training_metadata.json", values)
    verify(p)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if peak > MAX_RSS:
        raise ValueError("metadata control exceeds8GB host memory")
    result = {"status": "additional_training_metadata_reserved_no_labels_or_quality_claim", "cases": len(values),
        "counts_by_bin": dict(Counter(row["bin"] for row in values)), **availability,
        "source_components_distinct": len({row["source_component"] for row in values}),
        "metadata_sha256": file_hash(root / "training_metadata.json"), "selected_final_sources_read_or_hashed": False,
        "new_labels_generated": False, "new_model_trained": False, "task_accuracy": None, "speed_tested": False,
        "peak_RSS_bytes": peak}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def workflow():
    if any(p.exists() for p in (WORKFLOW, PILOT, DEST)):
        raise ValueError("fresh append-only reservation roots required")
    WORKFLOW.mkdir()
    receipts = []
    for stage, root in (("pilot", PILOT), ("reserve", DEST)):
        command = [sys.executable, "-u", "-m", MODULE, stage]
        child = subprocess.Popen(command)
        code = child.wait()
        receipt = {"stage": stage, "command": command, "child_pid": child.pid, "observer_pid": os.getpid(),
            "observer": "subprocess.wait", "exit_code": code}
        write_json(WORKFLOW / (stage + "_receipt.json"), receipt)
        if root.exists():
            write_json(root / "worker_terminal_receipt.json", receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError("reservation failed:" + stage)
    write_json(WORKFLOW / "result.json", {"status": "metadata_reservation_completed", "stages": receipts})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("pilot", "reserve", "workflow"))
    args = parser.parse_args()
    {"pilot": lambda: run(True), "reserve": run, "workflow": workflow}[args.stage]()
