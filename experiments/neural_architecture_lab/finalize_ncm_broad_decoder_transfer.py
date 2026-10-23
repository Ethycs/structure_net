"""Preserve a failed broad-text decoder remedy and describe lexical error coverage."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_broad_decoder_transfer import DATA, PILOT, TRAIN, qualify
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_unseen_codec_control import MEASURE, NOVEL, SEEN
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261022_broad_decoder_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-22_ncm-broad-decoder-transfer.md")


def coverage():
    old = q.read(SEEN)
    new = q.read(DATA / "raw_train.json")
    old_counts = Counter(t for r in old for t in r["native_target_ids"] if t not in (50256, 50257))
    broad_counts = old_counts+Counter(t for r in new for t in r["native_target_ids"] if t not in (50256, 50257))
    novel = {r["id"]: r for r in q.read(NOVEL)}
    outputs, result = q.read(MEASURE / "outputs.json"), {}
    for model in ("released", "adapted"):
        strata = {k: {"total": 0, "errors": 0} for k in ("absent_old", "old_1_to_5", "old_over_5")}
        absent_broad = 0
        for row in outputs[model]["novel"]:
            for target, prediction in zip(novel[row["id"]]["native_target_ids"], row["native_output_ids"]):
                if target in (50256, 50257):
                    continue
                n = old_counts[target]
                bucket = "absent_old" if n == 0 else "old_1_to_5" if n <= 5 else "old_over_5"
                strata[bucket]["total"] += 1
                if prediction != target:
                    strata[bucket]["errors"] += 1
                    absent_broad += broad_counts[target] == 0
        result[model] = {"strata": strata, "error_positions_absent_broad": absent_broad}
    return {"status": "posthoc_descriptive_not_causal", "old_distinct_content_tokens": len(old_counts),
            "broad_distinct_content_tokens": len(broad_counts), "models": result}


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh completed result required")
    result = q.read(TRAIN / "result.json")
    if result["status"] not in ("codec_transfer_gate_passed", "codec_transfer_gate_failed") or result["steps"] != 2048:
        raise ValueError("terminal fixed2048-update experiment required")
    if (result["codec_transfer_qualified"] != qualify(result["final"]) or not result["nondecoder_unchanged"] or
            not result["novel_codes_unchanged"] or not result["reload_exact"]):
        raise ValueError("codec classification or validity changed")
    for root in (DATA, PILOT, TRAIN, MEASURE):
        verify(q.read(root / "producer.json"))
        for path, expected in q.read(root / "result.json").get("output_hashes", {}).items():
            if file_hash(path) != expected:
                raise ValueError("completed output changed: "+path)
    trace = q.read(TRAIN / "trace.json")
    if [r["step"] for r in trace] != list(range(1, 2049)):
        raise ValueError("missing or duplicated optimizer steps")
    lexical = coverage()
    paths = [Path(__file__), SEEN, NOVEL, MEASURE / "result.json", MEASURE / "outputs.json",
             Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in (DATA, PILOT, TRAIN):
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    DEST.mkdir()
    write_json(DEST / "lexical_coverage.json", lexical)
    paths.append(DEST / "lexical_coverage.json")
    aggregate = {"schema": "nal.ncm-broad-decoder-transfer.v1", "classification": result["status"],
        "training": result, "data": q.read(DATA / "result.json"), "lexical_coverage": lexical,
        "task_accuracy": None, "student_training_admitted": False, "speed_tested": False,
        "final_test_opened": False, "confirmed": False, "underpowered": True,
        "artifacts": {str(p): file_hash(p) for p in paths}}
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    table = "\n".join(f"|{stage}|{cohort}|{m['nonpad_correct']}/{m['nonpad_total']}={100*m['nonpad_reconstruction']:.4f}%|{m['exact_texts']}/{m['cases']}|"
        for stage in ("baseline", "final") for cohort,m in result[stage].items())
    REPORT.write_text(f'''# Broader decoder-only training fails to restore unseen reconstruction

Date: 2026-10-22. `ncm-broad-decoder-transfer-v1` is complete, classified
**{result['status']}**. Original body, generator and previously failed codec
remain untouched. The current summary attachment still has its earlier0/64
Terra result; no new summary accuracy or speedup is established here.

| Checkpoint | Cohort | Correct non-PAD tokens | Exact target texts |
| --- | --- | --- | --- |
{table}

This study starts from the original released codec, freezes its encoder,
codebooks and all nondecoder state, and trains only its decoder for2,048 fixed
updates on57,258 public text windows plus1,999 old summary anchors. Each batch
contains48 raw windows and16 summary anchors. The256 raw validation sources are
group/session-disjoint from codec training, and all20 novel summary texts and
their source groups are excluded from training. Data are exact untruncated
8/16/24/32-word public substrings, selected and split before training.

The joint diagnostic requires99.5% token reconstruction and90% exact texts
separately on raw256 and novel20. Both fail. Training-target reconstruction
improves while novel-summary reconstruction degrades. The fixed final checkpoint
is reported even though an earlier checkpoint had better raw validation. That
earlier checkpoint also missed the declared gates. More decoder training and
broader text alone are not a demonstrated cure; no generator run is admitted.

The matched baseline in this study is BF16 decoding at batch16, with FP32
quantization. It differs by one novel token from the earlier FP32/batch4 control;
the table uses this study's own baseline for the before/after comparison.
Encoder/codebook hashes and novel code assignments remain exact, no frozen
parameter receives gradients, cache and checkpoint/prediction reload are exact,
all2,048 optimizer steps are present and own peak allocation is
{result['peak_allocated_bytes']:,}bytes. This is one exploratory training seed,
not a statistical guarantee or a fresh blind task test. The20 novel targets were
already known development diagnostics motivating the remedy.

## Descriptive lexical coverage of the earlier codec errors

In the original adapted-codec novel control,44/53 wrong content-token positions
involve token IDs absent from the1,999 old training summaries. Among69 such
positions44 are wrong, versus7/134 positions with old token frequency1–5 and
2/494 positions with frequency above5. The original released codec makes
11/69,4/134 and7/494 errors in the same strata. These are post-hoc associations,
not proof that vocabulary coverage alone causes the failure. EOS/PAD are
excluded from this descriptive breakdown. The broader training bank expands
distinct content-token coverage from13,151 to33,635, but8 of the adapted model's
original error positions still involve IDs absent from that broader bank.
Natural windows alone do not cover the full GPT-2 vocabulary.

The next independent information control checks whether correct quantized
latents can be inverted using the encoder's original token dictionary. It
separates target-blind estimated normalization from target-aware oracle
normalization, with exact unquantized inversion as an implementation control.
No result from that new study is included here. Neither codec oracle metrics
nor known-example residual repair can rescue the failed summary task.

Aggregate: `{campaign}`; SHA256 `{file_hash(campaign)}`. Raw outputs, all fixed
checkpoints, cached codes, source splits, exclusions, lifecycle pilot and source
snapshots remain in append-only roots. Registry confirmed=false,
task_accuracy_measured=false, final task test sealed. Reproduction follows the
immutable broad-decoder protocol and `ncm_broad_decoder_transfer` commands.
''')
    print(json.dumps({"status": result["status"], "report": str(REPORT)}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {
        "ncm-broad-decoder-transfer-v1": {"name": "Broad reconstruction with released NCM code language frozen",
            "description": "Decoder-only2048-update training on57258 public windows and1999 summary anchors fails unseen codec gates. Novel reconstruction drops96.7922to94.8396percent; raw256validation96.4953percent; seen99.8512percent. Encoder/codebooks stay fixed. Not summary accuracy; no student training admitted.",
            "confirmation_status": "codec_transfer_gate_failed", "task_accuracy_measured": False},
    })
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
