"""Independently replay native LM1B codec controls; no source-only quality claim."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_lm1b_codec as original
from experiments.structure_net import ncm_lm1b_framed_codec as framed
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def locations(use_framing):
    r = framed if use_framing else original
    stem = "20261024_lm1b_framed_codec" if use_framing else "20261024_lm1b_codec"
    return r, original.ROOT / (stem + "_results_v1"), Path("docs/08 - Analysis/2026-10-24_ncm-" + stem[9:].replace("_", "-") + ".md")


def finalize(use_framing=False):
    run, dest, report = locations(use_framing)
    if dest.exists() or report.exists():
        raise ValueError("fresh codec audit/report required")
    q = original.q
    done = q.read(run.WORKFLOW / "result.json")
    expected = ("prepare", "native", "pilot", "measure") if use_framing else ("download", "prepare", "pilot", "measure")
    if [(s["stage"], s["exit_code"]) for s in done["stages"]] != [(s, 0) for s in expected]:
        raise ValueError("all four actual native codec children must complete")
    roots = [run.DATA, run.PILOT, run.RECON, framed.NATIVE if use_framing else original.RELEASE]
    for root in roots:
        original.vendor.terminal(root)
    original.vendor.verify(q.read(run.WORKFLOW / "producer.json"))
    if file_hash(original.CASES) != original.CASES_SHA:
        raise ValueError("all690 original targets changed")
    all_rows = q.read(original.CASES)["train"]
    data, outputs = q.read(run.DATA / "cases.json"), q.read(run.RECON / "outputs.json")
    if [r["id"] for r in all_rows] != [r["id"] for r in data] or [r["id"] for r in data] != [r["id"] for r in outputs] or len(data) != 690:
        raise ValueError("complete ordered690 denominator required")
    t = original.tokenizer()
    totals = {"cases": 690, "correct": 0, "total": 0, "exact_native_ids": 0, "boundary_correct": 0,
        "active_invalid_ids": 0, "empty": 0, "fits_one_frame": 0, "overlength": 0}
    import tiktoken
    storage = tiktoken.get_encoding("cl100k_base")
    totals["stored_overflow"] = 0
    for original_row, row, output in zip(all_rows, data, outputs):
        lexical = t.encode(original_row["target_text"], add_special_tokens=False)
        gold, packed = (framed.pack(lexical) if use_framing else original.pack(lexical))
        if (row["packed_ids"] != packed or row["target_ids"] != gold or output["target_ids"] != gold
                or row["target_text"] != original_row["target_text"] or row["source"] != original_row["source"]):
            raise ValueError("complete source/target/tokenizer pairing differs")
        raw = output["raw_ids"]
        if len(raw) != 128 * len(packed) or any(not 0 <= v < 30528 for v in raw):
            raise ValueError("full native decoder frame differs")
        if len(output["codes"]) != len(packed) or any(len(c) != sum(original.LEVELS)
                or any(not 0 <= v < 8192 for v in c) for c in output["codes"]):
            raise ValueError("complete native hierarchical codes required")
        if use_framing:
            raw_frames = [raw[i:i + 128] for i in range(0, len(raw), 128)]
            # Independent boundary replay: a genuine predicted BOS is consumed.
            joined, ends, bos = [], [], []
            for frame in raw_frames:
                start = int(frame[0] == 101)
                end = next((i for i in range(start, 128) if frame[i] == 101), 128)
                joined += frame[start:end]
                ends.append(end)
                bos.append(frame[0] == 101)
            text = t.decode(joined, skip_special_tokens=True).strip()
            boundary = ends == [sum(v != 0 for v in p) - 1 for p in packed]
            invalid = sum(v >= len(t) for p, end in zip(raw_frames, ends) for v in p[:end])
            if output["frame_EOS_positions"] != ends or output["all_frame_BOS_correct"] != all(bos):
                raise ValueError("predicted boundary evidence differs")
        else:
            end = next((i for i, v in enumerate(raw) if v == 101), len(raw))
            text = t.decode(raw[:end], skip_special_tokens=True).strip()
            boundary = end == len(gold) - 1
            invalid = sum(v >= len(t) for v in raw[:end])
        correct = sum(a == b for a, b in zip(raw, gold))
        exact = raw[:len(gold)] == gold
        if (output["text"] != text or output["correct"] != correct or output["total"] != len(gold)
                or output["exact_native_ids"] != exact or output["first_EOS_correct"] != boundary
                or output["active_invalid_ids"] != invalid or output["fits_one_native_frame"] != (len(packed) == 1)):
            raise ValueError("raw IDs do not replay stored metrics or text")
        totals["correct"] += correct
        totals["total"] += len(gold)
        totals["exact_native_ids"] += exact
        totals["boundary_correct"] += boundary
        totals["active_invalid_ids"] += invalid
        totals["empty"] += not text
        totals["stored_overflow"] += len(storage.encode(text, disallowed_special=())) > 128
        totals["fits_one_frame"] += len(packed) == 1
        totals["overlength"] += len(packed) > 1
    by_id = {v["id"]: v for v in outputs}
    if any(by_id[v["id"]] != v for v in q.read(run.PILOT / "outputs.json")):
        raise ValueError("full fresh-process outputs changed a retained pilot")
    result = q.read(run.RECON / "result.json")
    key_map = {"boundary_correct": "all_window_EOS_correct" if use_framing else "first_EOS_correct",
        "overlength": "overlength_in_denominator"}
    for k, value in totals.items():
        if k in ("empty", "stored_overflow"):
            continue
        if result[key_map.get(k, k)] != value:
            raise ValueError("aggregate differs: " + k)
    if (file_hash(run.RECON / "outputs.json") != result["outputs_sha256"] or not result["codec_unchanged"]
            or result["peak_allocated_bytes"] > 4700000000):
        raise ValueError("native model identity or memory evidence differs")
    for name, sha in original.FILES.items():
        if file_hash(original.RELEASE / "files" / name) != sha:
            raise ValueError("official checkpoint binary changed")
    artifacts = [Path(__file__), Path(run.__file__), run.PLAN, original.CASES,
        run.WORKFLOW / "producer.json", run.WORKFLOW / "result.json"]
    artifacts += [run.WORKFLOW / "stage_receipts" / (s + ".json") for s in expected]
    hashes = {str(p): file_hash(p) for p in artifacts}
    for root in roots:
        hashes.update(q.read(root / "producer.json")["input_hashes"])
        for p in root.iterdir():
            if p.is_file():
                hashes[str(p)] = file_hash(p)
    classification = ("native_CLS_codec_numeric_control_completed_semantics_unmeasured" if use_framing else
        "native_codec_boundary_format_mismatch_preserved")
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": classification, "numeric_results": totals,
        "native_token_fidelity": totals["correct"] / totals["total"], "codec_parameter_hash": result["codec_parameter_hash"],
        "model_training_executed": False, "oracle_target_codes": True, "source_prediction": False,
        "task_accuracy": None, "semantic_quality_measured": False, "all690_preserved": True,
        "final_test_opened": False, "speed_tested": False, "confirmed": False, "artifacts": hashes}
    if use_framing:
        native = q.read(framed.NATIVE / "result.json")
        if native["correct"] != sum(a == b for x, y in zip(native["raw_ids"], native["input_ids"]) for a, b in zip(x, y)):
            raise ValueError("native four-frame control differs")
        campaign["native_four_frame_control"] = {k: native[k] for k in ("status", "cases", "correct", "total", "codec_hash")}
    dest.mkdir()
    path = dest / "campaign_results.json"
    write_json(path, campaign)
    report.write_text(f"""# Released LM1B codec {'with corrected CLS framing' if use_framing else 'boundary-format failure'}

Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`.
Classification **{classification}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`. Conformance: EXPLORATORY target-aware control.

| Representation measurement | Result |
| --- | ---: |
| Complete TRAIN targets | 690/690 |
| Native token agreement | {totals['correct']}/{totals['total']} = {campaign['native_token_fidelity']:.4%} |
| Exact native token sequences | {totals['exact_native_ids']}/690 |
| {'All window boundaries' if use_framing else 'First global EOS'} correct | {totals['boundary_correct']}/690 |
| Empty reconstructed texts | {totals['empty']}/690 |
| Above128 storage tokens | {totals['stored_overflow']}/690 |
| Fit one native output frame | {totals['fits_one_frame']}/690 |
| Overlength targets retained with complete windows | {totals['overlength']}/690 |

These are correct-target-code reconstructions, not source-only summary accuracy.
No source-conditioned generator was trained. BERT lowercases and changes spacing;
token agreement uses its native representation, not exact cased summary matches.
The128-position frame includes native boundaries. Multi-window oracle coverage
does not show that one generated frame can express an overlength reference.

{'The initial control omitted a leading CLS and stopped at the first predicted CLS, which was usually the native BOS. Its empty results are preserved; they do not establish failure of a correctly framed model.' if not use_framing else 'Every frame has an explicit CLS boundary at each end. The parser consumes a predicted initial CLS and stops at the next predicted CLS without access to correct target length. Full raw IDs and every overlength window remain available. The separate four-frame native-data control is descriptive, not a full benchmark replication.'}

All four worker exits were observed successfully. The full fresh codec load
reproduced both retained pilot outputs; all raw counts, tokenization and texts
replayed independently. Released weights remain unchanged, FP32 with TF32 off,
on physical GPU2. Peak allocation {result['peak_allocated_bytes']} bytes.
Original Llama and its MLPs were absent and remain frozen. No final source was
opened. The95% source-disjoint task goal and inference speed remain unproven.
A source-grounded semantic audit is required before conditional retraining.
""")
    print(json.dumps({"campaign": str(path), "classification": classification, "numeric_results": totals,
        "native_token_fidelity": campaign["native_token_fidelity"], "task_accuracy": None}), flush=True)


def register(use_framing=False):
    run, dest, report = locations(use_framing)
    campaign = original.q.read(dest / "campaign_results.json")
    register_records(dest / "campaign_results.json", report, dest, {run.HYPOTHESIS: {
        "name": "Released LM1B target-code reconstruction with " + ("native CLS framing" if use_framing else "preserved boundary mismatch"),
        "description": "All690 existing TRAIN targets, unchanged author weights, exact native codec reload and raw-ID replay; target-aware representation only, no conditional summarization or speed claim.",
        "classification": campaign["classification"], "source_only_task_accuracy_measured": False,
        "semantic_quality_measured": False, "all690_preserved": True}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    parser.add_argument("--framed", action="store_true")
    args = parser.parse_args()
    # Install the same dependency paths as the existing registry workflows.
    original.vendor.corpus.quality.dependencies()
    (finalize if args.stage == "finalize" else register)(args.framed)
