"""Replay fixed-representation decoder transfer and register its limited scope."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_frozen_representation_decoder as trial
from experiments.structure_net import ncm_broad_codec_reconstruction as measurement
from experiments.structure_net.ncm_sample_scaled_codec_ema import FixedVocabularySize
from experiments.neural_architecture_lab.finalize_ncm_broad_learned_codec import replay
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json

DEST = trial.core.ROOT / "20261023_frozen_representation_decoder_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-frozen-representation-decoder.md")


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("append-only finalization requires a fresh destination")
    trial.configure()
    roots = (trial.core.corpus.DEST, trial.PILOT, trial.TRAIN, trial.RECON)
    for root in roots:
        trial.core.terminal(root)
    load = lambda p: json.loads(p.read_text())
    trained = load(trial.TRAIN / "result.json")
    pilot = load(trial.PILOT / "result.json")
    measured = load(trial.RECON / "result.json")
    controls = load(trial.RECON / "representation_equivalence.json")
    if (trained["updates"] != 2048 or not trained["only_decoder_trained"]
            or not trained["reload_exact"] or not pilot["reload_exact"]
            or trained["initial_model_sha256"] != pilot["initial_model_sha256"]
            or trained["representation_sha256_unchanged"] != pilot["representation_sha256_unchanged"]
            or not trained["initial_inference_matches_original_exactly"]
            or not trained["all_sampled_windows_and_LRs_match_previous"]
            or not all(controls.values())
            or file_hash(trial.TRAIN / "model.pt") != trained["checkpoint_sha256"]):
        raise ValueError("fixed-budget decoder-only/reload/frozen-representation controls required")
    previous = [json.loads(s) for s in (trial.ORIGINAL_TRAIN / "trace.jsonl").read_text().splitlines()]
    actual = [json.loads(s) for s in (trial.TRAIN / "trace.jsonl").read_text().splitlines()]
    if len(previous) != 2048 or len(actual) != 2048 or any(
            (a["window_indices"], a["lr"]) != (b["window_indices"], b["lr"])
            for a, b in zip(previous, actual)):
        raise ValueError("fixed sample lists or LR schedules differ")
    cases = load(trial.core.corpus.DEST / "summary_diagnostic.json")
    origins = load(trial.core.corpus.DEST / "fit_origins.json")
    for key in ("source_id", "group_id", "source_sha256"):
        if {c[key] for c in cases} & {o[key] for r in origins for o in r["origins"]}:
            raise ValueError("codec-fit/diagnostic source overlap")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(trial.core.corpus.GPT2, local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer = FixedVocabularySize(tokenizer)
    components, rows = {}, {}
    for arm in ("original", "trained"):
        path = trial.RECON / (arm + "_outputs.json")
        if file_hash(path) != measured["output_hashes"][arm]:
            raise ValueError("raw outputs changed")
        rows[arm] = load(path)
        components[arm] = replay(cases, rows[arm], tokenizer)
        if components[arm] != measured[arm]:
            raise ValueError("replayed reconstruction aggregate differs")
    if rows["original"] != load(trial.REFERENCE / "original_outputs.json") or any(
            a["codes"] != b["codes"] for a,b in zip(rows["original"], rows["trained"])):
        raise ValueError("reference outputs or frozen codes changed")
    passed = measurement.gate(components["original"], components["trained"])
    classification = "codec_reconstruction_gate_passed_semantic_audit_required" if passed else "codec_reconstruction_fidelity_failed"
    if (measured["classification"] != classification or measured["representation_gate"] != passed
            or measured["task_accuracy"] is not None or measured["source_prediction"]):
        raise ValueError("representation gate or task-accuracy scope changed")
    paths = [Path(__file__), Path(trial.__file__), trial.PLAN,
        Path("experiments/neural_architecture_lab/ncm_result_registry.py"),
        trial.ORIGINAL_TRAIN / "trace.jsonl", trial.PREVIOUS / "campaign_results.json"]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    aggregate = {"schema":"nal.ncm-frozen-representation-decoder.v1", "classification":classification,
        "training":trained,"pilot":pilot,"reconstruction":measured,"components":components,
        "all252_raw_outputs_replayed":True,"all252_codes_unchanged":True,"all2048_sample_lists_LRs_match":True,
        "representation_frozen":True,"oracle_target_codes":True,"source_prediction":False,
        "task_accuracy_measured":False,"confirmed":False,"NCM_success_established":False,
        "final_test_opened":False,"speed_tested":False,"artifacts":{str(p):file_hash(p) for p in dict.fromkeys(paths)}}
    DEST.mkdir()
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    table = []
    for arm, m in components.items():
        table.append(f"| {arm} | {m['correct']}/{m['total']} ({100*m['token_recovery']:.3f}%) | {m['exact_texts']}/252 | {m['first_EOS_correct']}/252 | {m['changed_number_cases']} | {m['active_invalid_ids']} |")
    REPORT.write_text(f"""# Codec decoder transfer with the representation frozen

Date: 2026-10-23. Hypothesis `{trial.HYPOTHESIS}`.
Classification **{classification}**. Aggregate `{campaign}`,
SHA256 `{file_hash(campaign)}`.

| Codec | Correct non-PAD tokens | Exact texts | Correct first EOS | Changed number sequences | Active invalid IDs |
| --- | --- | --- | --- | --- | --- |
""" + "\n".join(table) + f"""

These measurements supply complete correct target-derived codes; they are
oracle reconstruction, not source-to-summary accuracy. All 252 final trained
code arrays equal their released-reference counterparts. The encoder, EMA
codebooks, normalization and resampling state stayed exact. Only the new codec
copy's {trained['trainable_parameters']:,} decoder parameters trained. The
parent Llama was not loaded. No conditional NCM generator was trained here.

Training used exactly 2,048 FP32 updates on GPU2, microbatch four and accumulation
eight. The 31,109 fit windows from 7,088 unpaired texts exclude every diagnostic
source group. The sampler lists and learning-rate schedules match the preceding
joint-transfer runs. Full correct-code latents were computed in eval/no-grad mode;
only decoder CE was optimized, retaining its .1 training dropout. This recipe
changes both gradient scope and the latent/dropout training distribution, so it
does not isolate every difference from earlier joint transfer.

The original reference reproduces all earlier outputs exactly. Raw IDs, codes,
complete texts and counts replayed for all 252 cases. Pilot/full save-reload
controls and frozen-state checks passed. Peak allocation was
{trained['peak_allocated_bytes']:,} bytes. Only the final checkpoint was selected.

The unchanged gate requires at least 99.5% token recovery, 99% first-EOS agreement,
zero active invalid IDs and no increase in changed-number cases. Exact-text
agreement and numeric-sequence changes are descriptive; a separate semantic
audit is required before a task-quality claim. The actual source-only native
teacher remains 56/64 (87.5%), below the goal. Both final task cohorts remain
closed; no speed or compression benefit was measured.
""")
    print(json.dumps({"status":"finalized","classification":classification,"components":components}),flush=True)


def register():
    x = json.loads((DEST / "campaign_results.json").read_text())
    spec = {"name":"Released-codec decoder transfer with frozen target representations",
        "description":"2048 full-code decoder-only FP32 updates on excluded-source unpaired text;252 complete oracle reconstructions. Encoder, EMA books and resampling state frozen, all252 target codes identical. This measures reconstruction, not source-to-summary accuracy or speed.",
        "confirmation_status":x["classification"],"task_accuracy_measured":False,
        "oracle_target_codes":True,"source_prediction":False,"NCM_success_established":False,
        "representation_gate":x["reconstruction"]["representation_gate"],"representation_frozen":True}
    spec.update({arm+"_"+key:value for arm,m in x["components"].items() for key,value in m.items()})
    register_records(DEST / "campaign_results.json", REPORT, DEST, {trial.HYPOTHESIS:spec})
    print(json.dumps({"status":"registered_and_read_back"}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("command",choices=["finalize","register"])
    globals()[parser.parse_args().command]()
