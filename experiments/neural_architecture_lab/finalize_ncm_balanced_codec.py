"""Finalize the fixed vocabulary-balanced codec remedy without a summary claim."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_broad_decoder_transfer import qualify
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_vocabulary_balanced_codec import DATA, PILOT, TRAIN
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261022_balanced_codec_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-22_ncm-vocabulary-balanced-codec.md")


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh completed result required")
    result = q.read(TRAIN / "result.json")
    if result["status"] not in ("codec_transfer_gate_passed", "codec_transfer_gate_failed") or result["steps"] != 2048:
        raise ValueError("terminal fixed2048-update run required")
    passed = qualify(result["final"])
    if result["codec_transfer_qualified"] != passed or not all(result[k] for k in ("nondecoder_unchanged", "novel_codes_unchanged", "reload_exact")):
        raise ValueError("codec validity or gate classification changed")
    for root in (DATA, PILOT, TRAIN):
        verify(q.read(root / "producer.json"))
        for path, expected in q.read(root / "result.json")["output_hashes"].items():
            if file_hash(path) != expected:
                raise ValueError("completed artifact changed")
    exposure = q.read(DATA / "exposure_summary.json")
    if exposure["actual_positive_content_ids"] != 50256 or exposure["minimum_actual_content_count"] < 1:
        raise ValueError("actual schedule lacks complete vocabulary supervision")
    if [r["step"] for r in q.read(TRAIN / "trace.json")] != list(range(1, 2049)):
        raise ValueError("optimizer steps missing or repeated")
    paths = [Path(__file__), Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in (DATA, PILOT, TRAIN):
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    DEST.mkdir()
    aggregate = {"schema": "nal.ncm-balanced-codec.v1", "classification": result["status"],
        "training": result, "actual_vocabulary_exposure": exposure,
        "frozen_decoder_data_sweeps_closed": not passed,
        "task_accuracy": None, "student_training_admitted": False, "confirmed": False, "underpowered": True,
        "speed_tested": False, "final_test_opened": False, "artifacts": {str(p): file_hash(p) for p in paths}}
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    table = "\n".join(f"|{stage}|{cohort}|{m['nonpad_correct']}/{m['nonpad_total']}={100*m['nonpad_reconstruction']:.4f}%|{m['exact_texts']}/{m['cases']}|"
        for stage in ("baseline", "final") for cohort,m in result[stage].items())
    interpretation = ("The codec passes its reconstruction endpoint, but source-to-code prediction and complete-summary quality still require independent demonstration."
        if passed else "The codec misses its reconstruction endpoint despite complete content-token supervision. Per the prospective plan, further frozen-XSum-decoder data sweeps stop. The next study must change the representation or final text readout rather than extend this training recipe.")
    REPORT.write_text(f'''# Vocabulary-balanced codec reconstruction

Date: 2026-10-22. `ncm-vocabulary-balanced-codec-v1` is complete, classified
**{result['status']}**. The95% full-summary goal remains unmet. No new summary
accuracy or speed result is measured; the existing full attachment remains
at its earlier0/64 Terra development outcome.

| Checkpoint | Cohort | Correct non-PAD target tokens | Exact target texts |
| --- | --- | --- | --- |
{table}

{interpretation}

The unchanged released encoder and codebooks supply correct target codes.
Only the decoder trains, starting from the original release, for2,048 fixed
updates. The augmented raw bank contains57,258 public natural text windows and
114,516 synthetic native-token sequences, interleaved1:2. Each update samples48
entries from that bank plus16 old summary anchors: expected32 synthetic,
16 natural and16 summary targets. Synthetic sequences are random token-space
exercises, not natural summaries; semantic evaluation never uses them.

Their content follows balanced shuffled permutations of all50,256 nonreserved
GPT-2 IDs, length8..47, followed by EOS and PAD to48. The exact main sampler
schedule is replayed before training: all50,256 content IDs receive positive
target exposure, with minimum{exposure['minimum_actual_content_count']} occurrences.
Seeds5042/6042 separate lengths and content; sampling2042 and dropout52 remain
the same as the natural-only study. No seed/LR/outcome retry or alternate
checkpoint selection occurred. Final-step2,048 is the fixed endpoint.

Raw256 validation and novel20 targets/source groups stay excluded from natural
training and are byte-identical to the earlier broad study. The gate requires
99.5% non-PAD tokens and90% exact texts separately on both cohorts. These are
known development diagnostics and target-aware reconstruction, not a new blind
task test. Complete vocabulary exposure is a validity measurement, not proof
of faithful text recovery or semantic transfer. The20 seen cases are training
diagnostics; high seen reconstruction cannot rescue a failed novel endpoint.

The wrapper changes only dataset/root/plan configuration in its process and
preserves/restores the shared training engine. Both source versions are frozen;
the original engine file is unchanged. FP32 quantization, BF16 decoder, batch64,
GPU0; real-batch lifecycle, nonzero decoder gradients, all frozen state, novel
code identities and exact checkpoint/prediction reload passed. Own peak GPU
allocation was{result['peak_allocated_bytes']:,}bytes. Original Llama and all
previous generators/codecs remain untouched. No new student training is admitted
by this report. The final task test remains sealed.

Aggregate: `{campaign}`; SHA256 `{file_hash(campaign)}`. Every fixed checkpoint,
full target/output, synthetic ID bank, actual exposure counts, cached codes,
producer/source seal and lifecycle pilot remain in append-only roots. Registry
confirmed=false and task_accuracy_measured=false. Reproduction follows the
immutable vocabulary-balanced plan and `ncm_vocabulary_balanced_codec` commands.
''')
    print(json.dumps({"status": result["status"], "report": str(REPORT)}), flush=True)


def register():
    result = q.read(DEST / "campaign_results.json")
    final = result["training"]["final"]
    register_records(DEST / "campaign_results.json", REPORT, DEST, {
        "ncm-vocabulary-balanced-codec-v1": {"name": "Vocabulary-balanced frozen-language NCM codec reconstruction",
            "description": f"All50256 content token IDs receive positive exposure in fixed2048-update decoder training. Final novel reconstruction{100*final['novel']['nonpad_reconstruction']:.4f}percent, raw256{100*final['raw_validation']['nonpad_reconstruction']:.4f}percent. Classification{result['classification']}. Target-aware codec metrics, no summary accuracy or speed claim.",
            "confirmation_status": result["classification"], "task_accuracy_measured": False,
            "frozen_decoder_data_sweeps_closed": result["frozen_decoder_data_sweeps_closed"]},
    })
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
