"""Record a failed noise-capacity gate without treating it as summary accuracy."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.neural_architecture_lab.ncm_result_registry import register_records

CAPACITY = ROOT / "20261022_noise_capacity_measure_v1"
PILOT = ROOT / "20261022_residual_learning_pilot_v1"
STOP = ROOT / "20261022_residual_learning_stop_v1/result.json"
DEST = ROOT / "20261022_noise_capacity_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-22_ncm-conditioning-noise-capacity.md")
RUNNER = Path("experiments/structure_net/train_ncm_conditional_residual.py")
AMENDMENT = Path("docs/07 - Status Reports/2026-10-22_ncm-conditional-residual-mild-amendment.md")


def verify_old_pilot():
    producer = q.read(PILOT / "producer.json")
    for path, expected in producer["input_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("v1 pilot input changed: "+path)
    for path, expected in producer["source_hashes"].items():
        if file_hash(path) == expected:
            continue
        # This is the one explicitly versioned producer, retained at commit13fe31e.
        if Path(path).resolve() != RUNNER.resolve():
            raise ValueError("unexpected v1 source change: "+path)
        snapshot = PILOT / "producer_source/residual_train_ncm_conditional_residual.py"
        if file_hash(snapshot) != expected:
            raise ValueError("original runner snapshot changed")
    return {"status": "original_v1_producer_snapshot_verified", "versioned_source": str(RUNNER),
            "amendment_sha256": file_hash(AMENDMENT), "original_source_commit": "13fe31e"}


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh append-only result required")
    capacity = q.read(CAPACITY / "result.json")
    stopped = q.read(STOP)
    if (capacity["status"] != "completed_noise_capacity_measurement" or capacity["cases"] != 128
            or not capacity["vendor_parity"] or not capacity["reload_equal"]):
        raise ValueError("complete capacity measurement required")
    metrics = capacity["metrics"]
    if metrics["all_levels"]["capacity_qualified"] or not metrics["mild_fine"]["capacity_qualified"]:
        raise ValueError("recorded qualification differs")
    if stopped["status"] != "full_training_stopped_after_noise_capacity_failure" or stopped["full_training_started"]:
        raise ValueError("preserved v1 early stop required")
    if (ROOT / "20261022_residual_learning_train_v1").exists():
        raise ValueError("unplanned full v1 training exists")
    verify(q.read(CAPACITY / "producer.json"))
    old_snapshot = verify_old_pilot()
    paths = [Path(__file__), STOP, AMENDMENT,
             Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    paths.extend(p for p in CAPACITY.glob("*") if p.is_file())
    paths.extend(p for p in PILOT.rglob("*") if p.is_file() and "producer_source" not in p.parts)
    paths.append(PILOT / "producer_source/residual_train_ncm_conditional_residual.py")
    DEST.mkdir()
    result = {"schema": "nal.ncm-noise-capacity.v1", "classification": "all_level_noise_capacity_failed",
        "capacity": capacity, "original_learning_stop": stopped, "original_source_snapshot": old_snapshot,
        "task_accuracy": None, "main_v1_training_executed": False, "new_v2_training_results_included": False,
        "final_test_opened": False, "speed_tested": False, "underpowered": True, "confirmed": False,
        "artifacts": {str(p): file_hash(p) for p in paths}}
    write_json(DEST / "campaign_results.json", result)
    table = "\n".join(f"|{name}|{row['nonpad_correct']}/4,452={100*row['nonpad_reconstruction']:.4f}%|"
                      f"{row['exact_texts']}/128|{'yes' if row['capacity_qualified'] else 'no'}|"
                      for name, row in metrics.items())
    REPORT.write_text(f'''# Conditional NCM noise: capacity gate failure before full training

Date: 2026-10-22. `ncm-conditioning-noise-capacity-v1` is complete, classified
**all_level_noise_capacity_failed**. The proposed original full residual-learning
comparison is stopped before training. The95% complete-summary goal remains
unmet; no task accuracy or speedup is established by this diagnostic.

| Noise recipe | Correct non-PAD target tokens | Exact target texts | Capacity gate >=95% |
| --- | --- | --- | --- |
{table}

These are **target-aware reconstruction** measurements on128 hash-selected
existing training summaries, using the frozen polished codec. Each finer expert
knows the correct target latent and requantizes against actual corrupted prior
codes. They are not generated-summary pass rates or a held-out model evaluation.

The root-only recipe replaces half the coarse codes in expectation with one of
eight nearest non-identical vectors. All-level mode additionally replaces finer
codes uniformly, with probabilities linearly.5to0. Mild mode keeps the same
root replacements and thins the fine masks to probability.05 at the first fine
level, falling linearly to0. Last-level codes remain uncorrupted. Common draws,
nested masks and identical replacement IDs control the comparison. Every
declared mode is reported; none is chosen using development task outcomes.

No-noise labels exactly match vendor quantization. Codec state, code reload,
finiteness and GPU1 lifecycle checks passed. Quantization is FP32, neighborhood
distances FP64, decoder FP16, batch32. Minimum distinct-neighbor squared distance
was1.30769. Own peak allocation was855,412,736bytes. One seed and128 training
targets limit the interpretation; no statistical guarantee about unseen text
or student quality is claimed.

The original v1 conditional-residual learning pilot completed two real updates
per arm with matched initialization/noise, nonzero query and adapter gradients,
causal coarse inputs, no same/future fine-label leakage, unchanged codec and
exact checkpoint reload. Those are systems checks, not the1,024-update mechanism
endpoint. The128-target capacity measurement then failed its predeclared95%
gate for v1's all-level noise, so full v1 training was never launched. Its passed
pilot and original source copies from commit13fe31e remain intact.

The explicit amendment creates a fresh v2 study with mild fine noise, qualified
at99.2812% oracle reconstruction, and requires a matching new pilot. Its full
comparison is running separately; **no v2 training result is included here**.
Even its eventual mechanism pass would only support a separately declared
broader training run with actual source-disjoint summary grading. Original-body
and codec weights remain frozen; inference receives no reference text or codes.

The failed original producer's runner path has intentionally advanced to v2.
Finalization checks its preserved source bytes against the original producer
hash, and checks every unchanged source/input normally. It does not rewrite
the old hash or confuse new code with the old run. The amendment and verification
receipt are part of the aggregate.

Aggregate: `{DEST / 'campaign_results.json'}`; SHA256
`{file_hash(DEST / 'campaign_results.json')}`. Reproduce the capacity measurement
with `experiments.structure_net.ncm_conditioning_noise_capacity`, following its
immutable plan and recorded moba runtime/GPU mapping. Registry records retain
confirmed=false, underpowered=true and explicit unexecuted main-training status.
The final256-case task test stays sealed.
''')
    print(json.dumps({"status": result["classification"], "report": str(REPORT)}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {
        "ncm-conditioning-noise-capacity-v1": {"name": "NCM conditioning-noise codec capacity",
            "description": "Target-aware oracle reconstruction on128 training summaries is94.9236percent for all-level noise and99.2812percent for mild fine noise; all-level noise fails the95percent capacity gate. These are not summary task scores.",
            "confirmation_status": "all_level_noise_capacity_failed", "task_accuracy_measured": False},
        "ncm-conditional-residual-learning-v1": {"name": "Original NCM conditional residual learning early stop",
            "description": "The two-update systems pilot passed but full1024-update training never ran because the separately declared codec-noise capacity gate failed. The original pilot and source are preserved; a fresh mild-noise v2 is separate.",
            "confirmation_status": "stopped_before_training_noise_capacity_failed", "task_accuracy_measured": False,
            "main_training_executed": False},
    })
    print(json.dumps({"status": "registered_and_read_back", "studies": 2}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
