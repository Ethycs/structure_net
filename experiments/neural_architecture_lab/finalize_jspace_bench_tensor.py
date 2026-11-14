"""Append-only final provenance for the shared-basis and dense-stacking pilot."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from experiments.structure_net import jspace_bench_tensor as base
from experiments.structure_net.jspace_bench_tensor_v2 import DEFAULT_ROOT as V2
from experiments.structure_net.jspace_bench_covered import DEFAULT_ROOT as COVERED
from experiments.structure_net.jspace_bench_stacked import DEFAULT_ROOT as STACKED
from experiments.structure_net.jspace_bench_stacked_timing import DEFAULT_ROOT as TIMING
from experiments.structure_net.jspace_stacked_graph_decode import DEFAULT_ROOT as PREPARED_GRAPH
from experiments.structure_net.jspace_stacked_graph_decode_v2 import DEFAULT_ROOT as GRAPH_V2
from experiments.structure_net.jspace_stacked_graph_decode_v3 import DEFAULT_ROOT as GRAPH
from experiments.structure_net.jspace_bench_data import sha, write


def main():
    base.cpu_only()
    roots = [(base.DEFAULT_ROOT, ("shakedown", "capture")),
             (V2, ("shakedown", "capture", "fit", "evaluation")),
             (COVERED, ("covered",)), (STACKED, ("stacked",)), (TIMING, ("timing",)),
             (PREPARED_GRAPH, ()), (GRAPH_V2, ("graph",)), (GRAPH, ("graph",))]
    campaigns = []
    for root, stages in roots:
        plan = base.verify(root)
        checked = {stage: base.checked(root, stage) for stage in stages}
        if (root / "result.json").exists():
            result = json.loads((root / "result.json").read_text())
            aggregate = json.loads((root / "campaign_results.json").read_text())
            if aggregate.pop("result_sha256") != sha(root / "result.json") or aggregate != result:
                raise ValueError("Aggregate mismatch")
        campaigns.append({"root": str(root), "fingerprint": plan["scientific_fingerprint"],
                          "stages": checked})
    if (base.DEFAULT_ROOT / "fit_manifest.json").exists():
        raise ValueError("The recorded original failed fit was unexpectedly completed")
    registry = base.ROOT / "data/hypothesis_registry/jspace_bench_tensor_20261113.json"
    readback = json.loads((STACKED / "registry_readback.json").read_text())
    record = json.loads(registry.read_text())
    if readback.get("readback_verified") is not True or readback["result_hash"] != record["storage"]["result_hash"]:
        raise ValueError("Fresh-process registry readback missing")
    for name, expected in record["source_sha256"].items():
        if sha(name) != expected:
            raise ValueError("Registry provenance changed")
    extra = [Path(__file__), base.ROOT / "experiments/neural_architecture_lab/store_jspace_bench_tensor.py",
             base.ROOT / "experiments/structure_net/jspace_bench_artifacts.py",
             base.ROOT / "tests/structure_net/test_jspace_bench_tensor_registry.py",
             base.ROOT / "docs/08 - Analysis/jspace/2026-11-13_jspace-bench-tensor.md"]
    archive = STACKED / "postprocessing_source"
    archive.mkdir(exist_ok=False)
    for path in extra:
        target = archive / path.relative_to(base.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    artifacts = {str(registry): sha(registry)}
    for root, _ in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.startswith("finaliz"):
                artifacts[str(path)] = sha(path)
    value = {"study_id": base.STUDY_ID, "status": "completed", "confirmed": False,
             "created_at": datetime.now(timezone.utc).isoformat(),
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=base.ROOT, text=True).strip(),
             "campaigns": campaigns, "artifact_sha256": artifacts, "registry_readback": readback,
             "integrity": {"gpu_fit_failures": 1, "successful_fit_retries": 1,
                           "reused_capture_executions": 1, "reused_shakedown_executions": 1,
                           "newly_executed_gpu_stages": 10,
                           "prepared_not_executed_graph_versions": 1,
                           "distinct_task_questions": 448, "held_out_questions": 256,
                           "independent_model_checkpoints": 1, "independent_training_replicates": 0,
                           "no_threshold_or_channel_changes_for_speed_optimization": True,
                           "gpu_stages_reservation_supervised": True},
             "scope": "One exploratory campaign, with named implementation extensions and counterbalanced timing; no independent confirmations"}
    write(STACKED / "finalization.json", value)
    print(json.dumps({"status": "finalized", "artifacts": len(artifacts), "source_commit": value["source_commit"],
                      "finalization_sha256": sha(STACKED / "finalization.json")}, allow_nan=False))


if __name__ == "__main__":
    main()
