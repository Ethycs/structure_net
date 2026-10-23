"""Record unseen codec failure and the separate training-only repair mechanism."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.neural_architecture_lab.ncm_result_registry import register_records

CODEC = ROOT / "20261022_unseen_codec_measure_v1"
REPAIR = ROOT / "20261022_residual_learning_train_v2"
DEST = ROOT / "20261022_transfer_controls_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-22_ncm-unseen-codec-and-repair.md")


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh result required")
    codec, repair = q.read(CODEC / "result.json"), q.read(REPAIR / "result.json")
    if codec["status"] != "completed" or repair["status"] not in ("mechanism_gate_passed", "mechanism_gate_failed"):
        raise ValueError("terminal completed controls required")
    roots = [CODEC, REPAIR, REPAIR / "static", REPAIR / "repair",
             ROOT / "20261022_unseen_codec_pilot_v1", ROOT / "20261022_residual_learning_pilot_v2"]
    for root in roots:
        verify(q.read(root / "producer.json"))
    for path, expected in codec["output_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("codec output changed")
    a, b = repair["results"]["static"], repair["results"]["repair"]
    if a["baseline"] != b["baseline"] or a["initial_hash"] != b["initial_hash"] or a["sample_noise_schedule_hash"] != b["sample_noise_schedule_hash"]:
        raise ValueError("unmatched mechanism comparison")
    gain = b["final"]["all_wrong"]["nonpad_reconstruction"]-a["final"]["all_wrong"]["nonpad_reconstruction"]
    delta = b["final"]["free"]["nonpad_reconstruction"]-b["baseline"]["free"]["nonpad_reconstruction"]
    if gain != repair["all_wrong_repair_advantage"] or delta != repair["free_reconstruction_delta"]:
        raise ValueError("mechanism endpoint changed")
    gate = gain >= .05 and delta >= -.02
    if (repair["status"] == "mechanism_gate_passed") != gate:
        raise ValueError("mechanism classification changed")
    paths = [Path(__file__), Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    DEST.mkdir()
    result = {"schema": "nal.ncm-transfer-controls.v1", "codec": codec, "repair_mechanism": repair,
        "classification": "unseen_codec_capacity_failed" if not codec["adapted_novel_codec_qualified"] else "unseen_codec_capacity_passed",
        "task_accuracy": None, "trained_student_admitted": False, "final_test_opened": False,
        "speed_tested": False, "underpowered": True, "confirmed": False,
        "artifacts": {str(path): file_hash(path) for path in paths}}
    campaign = DEST / "campaign_results.json"
    write_json(campaign, result)
    table = "\n".join(f"|{model}|{cohort}|{m['nonpad_correct']}/{m['nonpad_total']}={100*m['nonpad_reconstruction']:.4f}%|{m['exact_texts']}/{m['cases']}|"
        for model, cohorts in codec["metrics"].items() for cohort, m in cohorts.items())
    mechanism = "\n".join(f"|{arm}|{cond}|{m['nonpad_correct']}/{m['nonpad_total']}={100*m['nonpad_reconstruction']:.4f}%|{m['exact_texts']}/32|"
        for arm, rr in repair["results"].items() for cond, m in rr["final"].items())
    REPORT.write_text(f'''# The adapted codec fails on unseen text, before summary-code prediction

Date: 2026-10-22. `ncm-unseen-codec-control-v1` is complete, classified
**{result['classification']}**. The current NCM attachment remains a failed
summarizer; its prior development result is0/64 under Terra. No new task accuracy
or speedup is established here, and no broader student training is admitted.

| Frozen codec | Cohort | Correct non-PAD target tokens | Exact target texts |
| --- | --- | --- | --- |
{table}

Both codecs receive their own full **correct target codes**, produced by encoding
the reference text. The adapted codec fits seen text but misses its predeclared
98% novel-token gate. This is a defect before source-to-code prediction. One
adapted reconstruction changes Luxembourg to Denmark and Grand Duke Henri to
Grand Commun Basil. Thus even correct generated codes would not reliably recover
novel names or text with this codec. This does not establish that codec error
alone explains the much worse generated summaries. The original release also
falls below98% on these texts, so merely restoring it is not a demonstrated fix.

The20 novel targets are the completed, format-qualified public teacher-pilot
labels, disjoint by ID, session, group and source hash from all1,999 existing
training pairs. Their summary quality was not audited; the experiment asks only
whether their exact text can be encoded and reconstructed. The20 seen cases were
hash-selected before observing outcomes. Quantization and decoding are FP32,
batch4, on GPU1. Model states stayed unchanged, full code tensors reloaded
exactly, public sources and untruncated target-token identities verified. Own
peak GPU allocation was{codec['peak_allocated_bytes']:,}bytes. The sample is a
diagnostic, not a95% complete-summary success estimate or a confidence guarantee.

## Residual learning works on its separate training-only mechanism endpoint

`ncm-conditional-residual-learning-v2` completed1,024 updates in each matched arm
and **{repair['status']}**. Both arms start from the same failed attachment,
use identical sample/noise/dropout schedules, freeze the codec and original body,
and train on the same32 old training targets. Static uses clean fixed finer
labels; repair recomputes finer expert labels after actual corrupted history.
Both use root noise0.5 and mild fine noise0.05 falling to0. The all-wrong-root
evaluation deliberately supplies target-derived nearby wrong roots; it is not
ordinary summary inference.

| Arm | Final training diagnostic | Correct target tokens | Exact target texts |
| --- | --- | --- | --- |
{mechanism}

Repair's advantage after four wrong roots is{100*gain:.4f}percentage points,
above the5point mechanism threshold. Its free-training reconstruction change
from the shared baseline is{100*delta:.4f}points, within the declared tolerance.
The final free-training output is identical in accuracy between the two arms.
This supports learning conditional finer-code repair on known examples. It
does not demonstrate unseen source understanding, better summary accuracy or
95% performance. It cannot rescue the codec's unseen-text failure or admit a
broader student run. No new development summaries were graded in this experiment.

No-noise quantization parity, causal coarse inputs, no same/future fine-label
leakage, nonzero query/bridge gradients, frozen codec and exact prediction
save/reload all passed. Root-neighbor minimum squared distance is1.30769.
The original all-level-noise v1 stop and all prior failures remain preserved.

Aggregate: `{campaign}`; SHA256 `{file_hash(campaign)}`. Exact raw outputs,
producer/source snapshots, schedules, optimizer checkpoints, lifecycle pilots
and input seals remain in their append-only roots. Both registry records retain
confirmed=false and task_accuracy_measured=false. The final task test is sealed.
''')
    print(json.dumps({"status": result["classification"], "report": str(REPORT)}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {
        "ncm-unseen-codec-control-v1": {"name": "NCM unseen-text correct-code reconstruction",
            "description": "With correct target codes the adapted codec reconstructs99.8512percent of seen tokens but92.6081percent of novel tokens,4of20 novel texts exact. Released codec novel96.9317percent. Unseen codec capacity fails before source prediction; not summary accuracy.",
            "confirmation_status": "unseen_codec_capacity_failed", "task_accuracy_measured": False},
        "ncm-conditional-residual-learning-v2": {"name": "NCM conditional finer-code repair mechanism",
            "description": "Matched1024-update32-training-case experiment improves reconstruction after four target-derived wrong roots from93.7075to99.4048percent,5.6973points. Training-only mechanism gate passes, unseen summary quality unmeasured and broader training not admitted because the separate novel codec gate fails.",
            "confirmation_status": "training_mechanism_passed_task_quality_unestablished", "task_accuracy_measured": False},
    })
    print(json.dumps({"status": "registered_and_read_back", "records": 2}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
