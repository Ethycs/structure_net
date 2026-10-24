"""Replay all byte-tokenizer outputs and record the limited component finding."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_tinystories_pretoken_capacity as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records

HYPOTHESIS = "ncm-tinystories-pretoken-capacity-control-v1"
DEST = run.ROOT / "audit"
CLASSIFICATION = "lossless_text_encoding_but_unchanged128_frame_capacity_failed"


def finalize():
    if DEST.exists() or run.REPORT.exists():
        raise ValueError("fresh append-only component report required")
    run.native.vendor.corpus.quality.dependencies()
    run.native.vendor.terminal(run.ROOT)
    q = run.q
    result = q.read(run.ROOT / "result.json")
    for name, expected in run.ASSETS.items():
        value = result["downloads"][name]
        if file_hash(value["path"]) != expected or value["sha256"] != expected:
            raise ValueError("pinned author byte-tokenizer asset changed")
    vocab = run.plain_load(Path(result["downloads"]["tinystories/bpe_vocab_8192.pickle"]["path"]).read_bytes())
    merges = run.plain_load(Path(result["downloads"]["tinystories/bpe_merges_8192.pickle"]["path"]).read_bytes())
    tokenizer, bert = run.bpe(vocab, merges), run.native.tokenizer()
    source = q.read(run.training.DATA / "cases.json")
    actual = q.read(run.ROOT / "cases.json")
    if len(source) != 690 or [r["id"] for r in actual] != [r["id"] for r in source]:
        raise ValueError("same complete ordered690 targets required")
    expected = []
    for case in source:
        text = case["target_text"]
        ids, bert_ids = tokenizer.encode(text), bert.encode(text, add_special_tokens=False)
        expected.append({"id": case["id"], "target_sha256": q.digest(text), "TinyStories_lexical_tokens": len(ids),
            "TinyStories_text_exact": tokenizer.decode(ids, skip_special_tokens=True) == text,
            "TinyStories_ids": ids, "fits128_no_frame_overhead": len(ids) <= 128,
            "fits127_plus_one_boundary": len(ids) <= 127, "fits126_plus_two_boundaries": len(ids) <= 126,
            "BERT_lexical_tokens": len(bert_ids), "BERT_text_exact": bert.decode(bert_ids, skip_special_tokens=True) == text})
    if actual != expected:
        raise ValueError("native tokenizer output or inclusive capacity replay differs")
    replay = {"cases": len(expected), "TinyStories_exact_text_roundtrips": sum(r["TinyStories_text_exact"] for r in expected),
        "BERT_exact_text_roundtrips": sum(r["BERT_text_exact"] for r in expected),
        "fits128_no_frame_overhead": sum(r["TinyStories_lexical_tokens"] <= 128 for r in expected),
        "fits127_plus_one_boundary": sum(r["TinyStories_lexical_tokens"] <= 127 for r in expected),
        "fits126_plus_two_boundaries": sum(r["TinyStories_lexical_tokens"] <= 126 for r in expected),
        "TinyStories_max_lexical_tokens": max(r["TinyStories_lexical_tokens"] for r in expected),
        "TinyStories_total_lexical_tokens": sum(r["TinyStories_lexical_tokens"] for r in expected),
        "BERT_total_lexical_tokens": sum(r["BERT_lexical_tokens"] for r in expected)}
    if any(result[k] != v for k, v in replay.items()):
        raise ValueError("component aggregates differ")
    if (result["task_accuracy"] is not None or result["neural_codec_reconstruction"] is not None
            or result["source_only_generation"] or result["native_framing_qualified"]
            or result["final_test_opened"] or result["speed_tested"]):
        raise ValueError("encoding control cannot claim model quality or speed")
    for probe in result["case_and_unicode_probes"]:
        if tokenizer.decode(tokenizer.encode(probe["text"]), skip_special_tokens=True) != probe["restored"]:
            raise ValueError("case or Unicode fixture replay differs")
    files = [run.ROOT / n for n in ("producer.json", "cases.json", "result.json", "worker_terminal_receipt.json")]
    files += [Path(v["path"]) for v in result["downloads"].values()]
    files += [Path(__file__), Path(run.__file__), run.PLAN, run.TEST, run.training.DATA / "cases.json"]
    campaign = {"schema": "nal.ncm-tinystories-pretoken-capacity.v1", "hypothesis_id": HYPOTHESIS,
        "status": "completed", "classification": CLASSIFICATION, "tested": True, "confirmed": False,
        "underpowered": True, "component_metrics": replay, "all690_cases_replayed": True,
        "task_accuracy": None, "neural_codec_or_generator_tested": False, "native_framing_qualified": False,
        "source_prediction": False, "final_test_opened": False, "speed_tested": False,
        "canonical_semantic_cache_changed": False, "artifacts": {str(p): file_hash(p) for p in files}}
    DEST.mkdir()
    path = DEST / "campaign_results.json"
    write_json(path, campaign)
    run.REPORT.write_text(
        "# TinyStories byte-tokenizer capacity control\n\n"
        f"Date: 2026-10-24. Hypothesis `{HYPOTHESIS}`. Classification **{CLASSIFICATION}**.\n"
        f"Campaign `{path}`, SHA256 `{file_hash(path)}`.\n\n"
        "The pinned author byte vocabulary and merge files were downloaded (203,399bytes),\n"
        "SHA-verified, safely loaded as plain data and evaluated with the unchanged author\n"
        "BPETokenizer. All690 current TRAIN targets remain unchanged and in order.\n"
        "The independent CPU finalizer replayed every encoded sequence and boundary.\n\n"
        "| Component measurement | Count |\n| --- | ---: |\n"
        f"| TinyStories exact text roundtrips | {replay['TinyStories_exact_text_roundtrips']}/690 |\n"
        f"| BERT literal text roundtrips | {replay['BERT_exact_text_roundtrips']}/690 |\n"
        f"| Fits128 lexical positions, no boundaries | {replay['fits128_no_frame_overhead']}/690 |\n"
        f"| Fits127 lexical positions plus one boundary | {replay['fits127_plus_one_boundary']}/690 |\n"
        f"| Fits126 lexical positions plus two boundaries | {replay['fits126_plus_two_boundaries']}/690 |\n\n"
        f"TinyStories uses {replay['TinyStories_total_lexical_tokens']} lexical tokens versus "
        f"{replay['BERT_total_lexical_tokens']} BERT wordpieces; the longest TinyStories target "
        f"has {replay['TinyStories_max_lexical_tokens']}tokens. Case and Unicode fixtures roundtrip exactly.\n\n"
        "Literal BERT text differences reflect normalization and formatting; this does\n"
        "not imply zero semantic accuracy. Likewise, lossless byte encoding does not\n"
        "establish neural codec reconstruction or source-only summarization. The byte\n"
        "encoder is not a direct128-position replacement for these unchanged labels.\n"
        "Testing it would require a new framing/root-policy study, shorter targets or\n"
        "a larger native model; no claim about those untested variants is made.\n\n"
        "No neural checkpoint, DEV labels or final sources were used by this control.\n"
        "No teacher or judge requests were issued and the canonical semantic cache was\n"
        "unchanged. The ongoing LM1B training and fixed source-only evaluation retain\n"
        "their original producers and gates. The95%task goal and speed remain unproven.\n")
    print(json.dumps({"classification": CLASSIFICATION, "component_metrics": replay,
        "campaign_sha256": file_hash(path), "task_accuracy": None}), flush=True)


def register():
    path = DEST / "campaign_results.json"
    register_records(path, run.REPORT, DEST, {HYPOTHESIS: {
        "name": "TinyStories byte-tokenizer roundtrip and128-position capacity",
        "description": "All690 current training targets byte-roundtrip exactly, but only181fit128lexical positions before boundaries. CPU encoding/capacity control only; source-only task accuracy and neural codec quality remain unmeasured.",
        "classification": CLASSIFICATION, "all690_cases_retained": True,
        "task_accuracy_measured": False, "neural_codec_tested": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    {"finalize": finalize, "register": register}[args.stage]()
