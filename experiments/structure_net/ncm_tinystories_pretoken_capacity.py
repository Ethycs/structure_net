"""Byte round-trip and frame-capacity control; never a summarization score."""
import argparse
import io
import json
from pathlib import Path
import pickle
import sys

from experiments.structure_net import ncm_lm1b_grounded_training as training
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q = training.native, training.q
ROOT = native.ROOT / "20261024_tinystories_pretoken_capacity_v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-tinystories-pretoken-capacity-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-tinystories-pretoken-capacity.md")
TEST = Path("tests/structure_net/test_ncm_tinystories_pretoken_capacity.py")
VENDOR_CONFIG = native.vendor.VENDOR / "config/eval/tinystories.yaml"
VENDOR_BPE = native.vendor.VENDOR / "tokenization/bpe_tokenizer.py"
ASSETS = {
    "tinystories/bpe_vocab_8192.pickle": "fb176b4c68108e4b5ef61b9ff2af3731600124e3e160970a6bd78cd0a5d3ff6c",
    "tinystories/bpe_merges_8192.pickle": "0d4cab13f2e7b684b75eaa290a52e596ba87744ef13d943cf3a50a5a8e080ec1",
}


class PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Only plain vocabulary and merge data are allowed")


def plain_load(data):
    return PlainUnpickler(io.BytesIO(data)).load()


def bpe(vocab, merges):
    sys.path.insert(0, str(native.vendor.VENDOR))
    from tokenization.bpe_tokenizer import BPETokenizer
    return BPETokenizer(vocab, merges, {"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"})


def capacity(n):
    if type(n) is not int or n < 0:
        raise ValueError("nonnegative full lexical token count required")
    return {"fits128_no_frame_overhead": n <= 128, "fits127_plus_one_boundary": n <= 127,
        "fits126_plus_two_boundaries": n <= 126}


def run():
    native.vendor.corpus.quality.dependencies()
    native.vendor.terminal(training.DATA)
    cases = q.read(training.DATA / "cases.json")
    if len(cases) != 690 or len({r["id"] for r in cases}) != 690:
        raise ValueError("all690 current training targets required")
    sources = [Path(__file__), PLAN, TEST, VENDOR_CONFIG, VENDOR_BPE,
        native.vendor.VENDOR / "tokenization/tokenizer.py"]
    inputs = [training.DATA / n for n in ("cases.json", "result.json", "worker_terminal_receipt.json")]
    inputs += [native.INVENTORY / "inventory.json", *sources]
    inputs += [p for p in (native.INVENTORY / "bert-tokenizer").glob("*") if p.is_file()]
    config = {"schema": "nal.ncm-tinystories-pretoken-capacity.v1", "conformance": "EXPLORATORY component control only",
        "repository": "hazyresearch/ncm-checkpoints", "revision": native.REVISION, "assets": ASSETS,
        "all690_current_TRAIN_targets": True, "targets_not_rewritten_or_truncated": True,
        "neural_codec_or_generator_loaded": False, "native_framing_qualified": False,
        "native_model_root_policy_qualified": False, "task_accuracy": None, "source_prediction": False,
        "final_test_opened": False, "speed_tested": False}
    producer = native.vendor.freeze(ROOT, "plain_text_pretoken_roundtrip_and_capacity_only", inputs, config)
    producer["hypothesis_id"] = "ncm-tinystories-pretoken-capacity-control-v1"
    for i, path in enumerate(sources):
        producer["source_hashes"][str(path)] = file_hash(path)
        (ROOT / "producer_source" / f"pretoken_{i}_{path.name}").write_bytes(path.read_bytes())
    producer["fingerprint"] = q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(ROOT / "producer.json", producer)
    inventory = {r["path"]: r for r in q.read(native.INVENTORY / "inventory.json")["files"]}
    from huggingface_hub import hf_hub_download
    downloaded = {}
    for name, expected in ASSETS.items():
        path = Path(hf_hub_download("hazyresearch/ncm-checkpoints", name, revision=native.REVISION,
            local_dir=ROOT / "files", cache_dir=ROOT / "hf-cache"))
        if expected != inventory[name]["sha256"] or file_hash(path) != expected or path.stat().st_size != inventory[name]["size"]:
            raise ValueError("pinned author BPE asset differs")
        downloaded[name] = {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}
    vocab = plain_load(Path(downloaded["tinystories/bpe_vocab_8192.pickle"]["path"]).read_bytes())
    merges = plain_load(Path(downloaded["tinystories/bpe_merges_8192.pickle"]["path"]).read_bytes())
    if (not isinstance(vocab, dict) or set(vocab) != set(range(8192)) or any(not isinstance(v, bytes) for v in vocab.values())
            or not isinstance(merges, (list, tuple, dict)) or any(not isinstance(v, tuple) or len(v) != 2
                or any(not isinstance(x, bytes) for x in v) for v in merges)):
        raise ValueError("plain native8192 byte vocabulary and ordered byte merges required")
    tokenizer, bert = bpe(vocab, merges), native.tokenizer()
    if tokenizer.eos_token_id != 0 or tokenizer.pad_token_id != 1:
        raise ValueError("native special-token configuration changed")
    proof = []
    for case in cases:
        text = case["target_text"]
        ids = tokenizer.encode(text)
        restored = tokenizer.decode(ids, skip_special_tokens=True)
        bert_ids = bert.encode(text, add_special_tokens=False)
        proof.append({"id": case["id"], "target_sha256": q.digest(text), "TinyStories_lexical_tokens": len(ids),
            "TinyStories_text_exact": restored == text, "TinyStories_ids": ids, **capacity(len(ids)),
            "BERT_lexical_tokens": len(bert_ids), "BERT_text_exact": bert.decode(bert_ids, skip_special_tokens=True) == text})
    probes = ["USB File.py 10 MB", "Café and naïve UTF-8", "x != X; A + a = 2"]
    probe_results = [{"text": text, "restored": tokenizer.decode(tokenizer.encode(text), skip_special_tokens=True)} for text in probes]
    result = {"status": "component_pretoken_capacity_measured_not_task_accuracy", "cases": len(proof),
        "TinyStories_exact_text_roundtrips": sum(r["TinyStories_text_exact"] for r in proof),
        "BERT_exact_text_roundtrips": sum(r["BERT_text_exact"] for r in proof),
        **{key: sum(r[key] for r in proof) for key in capacity(0)},
        "TinyStories_max_lexical_tokens": max(r["TinyStories_lexical_tokens"] for r in proof),
        "TinyStories_total_lexical_tokens": sum(r["TinyStories_lexical_tokens"] for r in proof),
        "BERT_total_lexical_tokens": sum(r["BERT_lexical_tokens"] for r in proof),
        "case_and_unicode_probes": probe_results, "downloads": downloaded,
        "neural_codec_reconstruction": None, "native_framing_qualified": False,
        "source_only_generation": False, "task_accuracy": None, "final_test_opened": False,
        "speed_tested": False, "producer_fingerprint": producer["fingerprint"]}
    write_json(ROOT / "cases.json", proof)
    native.vendor.verify(producer)
    write_json(ROOT / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.parse_args()
    run()
