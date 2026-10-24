"""Replay the fixed group-invariant readout and retain its negative comparison."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_lm1b_dictionary_readout as run
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json

RESULTS = run.ROOT / "20261024_lm1b_dictionary_readout_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-lm1b-group-readout.md")


def finalize():
    if RESULTS.exists() or REPORT.exists():
        raise ValueError("fresh readout aggregate/report required")
    q = run.q
    done = q.read(run.WORKFLOW / "result.json")
    if [(s["stage"], s["exit_code"]) for s in done["stages"]] != [("pilot", 0), ("measure", 0)]:
        raise ValueError("both actual successful readout children required")
    for root in (run.PILOT, run.MEASURE):
        run.native.vendor.terminal(root)
    rows, baseline = run.inputs()
    values = q.read(run.MEASURE / "outputs.json")
    if len(values) != 690 or [r["id"] for r in rows] != [v["id"] for v in values]:
        raise ValueError("complete ordered690 comparison required")
    t = run.native.tokenizer()
    totals = {"cases": 690, "correct": 0, "total": 0, "exact_native_ids": 0,
        "all_window_EOS_correct": 0, "all_frame_BOS_correct": 0, "neural_decoder_correct": 0,
        "clean_encoder_exact_positions": 0}
    for row, v in zip(rows, values):
        native = baseline[row["id"]]
        raw = v["raw_ids"]
        if len(raw) != len(row["packed_ids"]) * 128 or any(not 0 <= x < run.VOCAB for x in raw):
            raise ValueError("full valid native prediction required")
        frames = [raw[i:i + 128] for i in range(0, len(raw), 128)]
        text, ends, bos = run.framing.render(frames, t)
        correct = sum(a == b for a, b in zip(raw, row["target_ids"]))
        expected_ends = [sum(x != 0 for x in p) - 1 for p in row["packed_ids"]]
        exact = raw[:len(row["target_ids"])] == row["target_ids"]
        if (v["text"] != text or v["correct"] != correct or v["total"] != len(row["target_ids"])
                or v["source_code_sha256"] != q.digest(native["codes"]) or v["neural_decoder_correct"] != native["correct"]
                or v["exact_native_ids"] != exact or v["all_window_EOS_correct"] != (ends == expected_ends)
                or v["all_frame_BOS_correct"] != all(bos) or not v["clean_control_exact_all_positions"]
                or v["clean_control_positions"] != len(raw)):
            raise ValueError("raw matched-code readout evidence differs")
        for key in ("correct", "total", "exact_native_ids", "all_window_EOS_correct", "all_frame_BOS_correct", "neural_decoder_correct"):
            totals[key] += v[key]
        totals["clean_encoder_exact_positions"] += v["clean_control_positions"]
    result = q.read(run.MEASURE / "result.json")
    if any(result[k] != v for k, v in totals.items()) or file_hash(run.MEASURE / "outputs.json") != result["outputs_sha256"]:
        raise ValueError("replayed aggregate differs")
    by_id = {v["id"]: v for v in values}
    if any(by_id[v["id"]] != v for v in q.read(run.PILOT / "outputs.json")):
        raise ValueError("retained fresh-process pilot differs")
    if result["codec_parameter_hash_unchanged"] != q.read(run.framing.RECON / "result.json")["codec_parameter_hash"]:
        raise ValueError("native model identity differs")
    classification = ("group_invariant_dictionary_readout_worse_than_native_decoder" if totals["correct"] < totals["neural_decoder_correct"]
        else "group_invariant_dictionary_readout_numeric_improvement_semantics_unmeasured")
    hashes = {str(p): file_hash(p) for p in (Path(__file__), Path(run.__file__), run.PLAN,
        run.WORKFLOW / "producer.json", run.WORKFLOW / "result.json")}
    for root in (run.PILOT, run.MEASURE):
        hashes.update(q.read(root / "producer.json")["input_hashes"])
        hashes.update({str(p): file_hash(p) for p in root.iterdir() if p.is_file()})
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": classification, "numeric_results": totals,
        "token_fidelity": totals["correct"] / totals["total"],
        "matched_native_decoder_fidelity": totals["neural_decoder_correct"] / totals["total"],
        "correct_token_delta": totals["correct"] - totals["neural_decoder_correct"],
        "no_target_statistics_in_prediction": True, "model_training_executed": False,
        "source_prediction": False, "oracle_target_codes": True, "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False, "confirmed": False, "artifacts": hashes}
    RESULTS.mkdir()
    path = RESULTS / "campaign_results.json"
    write_json(path, campaign)
    REPORT.write_text(f"""# Group-invariant dictionary readout failed to improve the native codec

Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`.
Classification **{classification}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`. Exploratory reconstruction control, no training.

| Same saved native codes, all690 TRAIN targets | Native token agreement | Exact native sequences |
| --- | ---: | ---: |
| Learned native decoder | {totals['neural_decoder_correct']}/{totals['total']} = {campaign['matched_native_decoder_fidelity']:.4%} | 271/690 |
| Group-invariant dictionary | {totals['correct']}/{totals['total']} = {campaign['token_fidelity']:.4%} | {totals['exact_native_ids']}/690 |

The fixed dictionary rule loses{-campaign['correct_token_delta']} correct tokens.
The independent continuous-encoder control recovered all{totals['clean_encoder_exact_positions']}
raw positions, including padding. Thus the normalization invariance held for
clean encoder vectors, but the fixed readout did not sufficiently tolerate the
quantized vectors. This does not prove an information-theoretic loss bound.
The learned decoder remains the better measured readout. No normalization
variant was selected afterward, and no weights were changed.

Prediction received only fixed-code latents and pretrained embeddings, with no
correct token IDs or true group statistics. Both pilot cases reloaded exactly;
all690 comparisons and raw predictions were retained and independently replayed.
FP32, TF32 off, physical GPU2; peak{result['peak_allocated_bytes']} bytes.

These are target-code reconstruction measurements, not source-only summaries.
The frozen-body student's separate date-controlled development score is0/128.
No95% task success, final-test result or speed gain is established. All2051
canonical semantic first decisions and every sealed final source remain intact.
""")
    print(json.dumps({"classification": classification, "numeric_results": totals,
        "correct_token_delta": campaign["correct_token_delta"], "campaign_sha256": file_hash(path)}), flush=True)


def register():
    campaign = run.q.read(RESULTS / "campaign_results.json")
    register_records(RESULTS / "campaign_results.json", REPORT, RESULTS, {run.HYPOTHESIS: {
        "name": "Native NCM group-invariant fixed dictionary readout",
        "description": "Fixed group normalization cancels unknown affine statistics on continuous encoder controls but compares worse on exact saved quantized codes across all690 TRAIN targets; source-only quality unmeasured.",
        "classification": campaign["classification"], "source_only_task_accuracy_measured": False,
        "all690_preserved": True, "model_training_executed": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    run.native.vendor.corpus.quality.dependencies()
    (finalize if args.stage == "finalize" else register)()
