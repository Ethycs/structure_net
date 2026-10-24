"""Unmodified released LM1B codec on complete existing TRAIN summary texts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_broad_learned_codec as vendor
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT = vendor.ROOT
HYPOTHESIS = "ncm-lm1b-native-codec-compatibility-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-native-codec-plan.md")
MODULE = "experiments.structure_net.ncm_lm1b_codec"
RELEASE = ROOT / "20261024_lm1b_native_release_v1"
INVENTORY = ROOT / "20261024_long_context_release_inventory_v1"
DATA = ROOT / "20261024_lm1b_codec_data_v1"
PILOT = ROOT / "20261024_lm1b_codec_pilot_v1"
RECON = ROOT / "20261024_lm1b_codec_reconstruction_v1"
WORKFLOW = ROOT / "20261024_lm1b_codec_workflow_v1"
CASES = ROOT / "20261023_reviewed_draft_data_v1/cases.json"
CASES_SHA = "ac3c56f7d03cf9160de9e719ef042b44a0bf12dececd778949b46e7288aac983"
REVISION = "d342ce8aaea58830eef3b8ad473664673e41f692"
FILES = {"lm1b/tokenizer.pt": "d88aa17b66d730f2cb1fb31c5dbe63888a10da47ed8bea29dcb1f89423c3059c",
    "lm1b/small/generator.pt": "b0789d5a2aad9114d40976d26bd1af09d8cea0b0bec7d385b6a44739f77256fe"}
LEVELS = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)
q = vendor.corpus.quality.run.q


def pack(ids):
    """Cover all words and one genuine EOS; never truncate an overlength label."""
    if not ids or any(not 0 <= x < 30522 for x in ids) or any(x in (0, 101) for x in ids):
        raise ValueError("nonempty native text IDs without embedded PAD/EOS required")
    full = [*ids, 101]
    windows = [full[i:i + 128] for i in range(0, len(full), 128)]
    return full, [w + [0] * (128 - len(w)) for w in windows]


def tokenizer():
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(INVENTORY / "bert-tokenizer", local_files_only=True)
    t.eos_token = "[CLS]"
    if t.eos_token_id != 101 or t.pad_token_id != 0 or len(t) != 30522:
        raise ValueError("released native BERT tokenizer identity differs")
    return t


def producing(root, stage, inputs=(), config=None):
    files = [Path(__file__), PLAN, Path("tests/structure_net/test_ncm_lm1b_codec.py"), Path(vendor.__file__),
        Path("experiments/structure_net/ncm_frozen_body.py"), *sorted(vendor.VENDOR.rglob("*.py"))]
    p = vendor.freeze(root, stage, [*inputs, *files], {**(config or {}),
        "hypothesis_id": HYPOTHESIS, "schema": "nal.ncm-lm1b-native-codec.v1",
        "conformance": "EXPLORATORY target-aware representation control", "revision": REVISION,
        "native_context": 128, "native_levels": LEVELS, "native_tokenizer": "bert-base-uncased",
        "precision": "FP32", "TF32": False, "dropout": "eval", "model_training_executed": False,
        "parent_loaded": False, "all_parent_MLPs_frozen": True, "oracle_target_codes": True,
        "source_only_summary_accuracy": None, "task_accuracy": None, "final_test_opened": False,
        "one_output_frame_covers_all_labels": False, "no_target_truncation": True,
        "overlength_targets": "multiple native128 windows for reconstruction diagnostic only",
        "output_policy": "single_raw_argmax_full_native_vocab30528", "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"lm1b_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def download():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("download must stay CPU-only")
    p = producing(RELEASE, "official_release_download_only", [INVENTORY / "inventory.json"], {"CPU_only": True})
    from huggingface_hub import hf_hub_download
    paths = ["lm1b/README.md", "lm1b/small/config.yaml", *FILES]
    receipts = {}
    inventory = {v["path"]: v for v in q.read(INVENTORY / "inventory.json")["files"]}
    for name in paths:
        path = Path(hf_hub_download("hazyresearch/ncm-checkpoints", name, revision=REVISION,
            local_dir=RELEASE / "files", cache_dir=RELEASE / "hf-cache"))
        actual = file_hash(path)
        if name in FILES and (actual != FILES[name] or actual != inventory[name]["sha256"]):
            raise ValueError("official released binary SHA differs")
        if path.stat().st_size != inventory[name]["size"]:
            raise ValueError("official released file size differs")
        receipts[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
        print(json.dumps({"downloaded": name, "bytes": path.stat().st_size, "sha256": actual}), flush=True)
    if (RELEASE / "files/lm1b/small/config.yaml").read_bytes() != (vendor.VENDOR / "config/eval/lm1b-small.yaml").read_bytes():
        raise ValueError("released config differs from pinned official vendor config")
    write_json(RELEASE / "downloads.json", receipts)
    vendor.verify(p)
    write_json(RELEASE / "result.json", {"status": "official_LM1B_weights_downloaded_hash_verified",
        "revision": REVISION, "downloads": receipts, "task_accuracy": None, "producer_fingerprint": p["fingerprint"]})


def prepare():
    if file_hash(CASES) != CASES_SHA or torch.cuda.is_initialized():
        raise ValueError("unchanged complete690 TRAIN and CPU-only preparation required")
    vendor.terminal(RELEASE)
    t = tokenizer()
    rows = q.read(CASES)["train"]
    if len(rows) != 690:
        raise ValueError("all690 targets required")
    p = producing(DATA, "complete_TRAIN_target_native_tokenization", [CASES, RELEASE / "result.json",
        INVENTORY / "bert-download.json", INVENTORY / "bert_coverage.json"], {"CPU_only": True})
    values = []
    for row in rows:
        ids = t.encode(row["target_text"], add_special_tokens=False)
        full, windows = pack(ids)
        if t.unk_token_id in ids:
            raise ValueError("unrepresentable training target")
        values.append({"id": row["id"], "group_id": row["group_id"], "source_id": row["source_id"],
            "source_sha256": row["source_sha256"], "source": row["source"], "target_text": row["target_text"],
            "native_round_trip": t.decode(ids, skip_special_tokens=True), "target_ids": full,
            "packed_ids": windows, "fits_one_native_frame": len(full) <= 128})
    longest = min(values, key=lambda r: (-len(r["target_ids"]), r["id"]))
    selected = [values[0]["id"], longest["id"]]
    if len(set(selected)) != 2:
        raise ValueError("distinct first and longest lifecycle probes required")
    write_json(DATA / "cases.json", values)
    write_json(DATA / "selection.json", {"pilot_ids": selected})
    vendor.verify(p)
    write_json(DATA / "result.json", {"status": "all690_native_codec_cases_prepared", "cases": 690,
        "fit_one_frame": sum(v["fits_one_native_frame"] for v in values), "overlength": sum(not v["fits_one_native_frame"] for v in values),
        "windows": sum(len(v["packed_ids"]) for v in values), "unknown_tokens": 0,
        "cases_sha256": file_hash(DATA / "cases.json"), "producer_fingerprint": p["fingerprint"], "task_accuracy": None})
    print(json.dumps(q.read(DATA / "result.json")), flush=True)


def load_codec():
    OmegaConf, VQVAE, _ = vendor.vendor_classes()
    cfg = OmegaConf.load(RELEASE / "files/lm1b/small/config.yaml")
    codec = VQVAE(**cfg.tokenizer.vqvae)
    path = RELEASE / "files/lm1b/tokenizer.pt"
    if file_hash(path) != FILES["lm1b/tokenizer.pt"]:
        raise ValueError("native codec checkpoint changed")
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    codec.load_state_dict(saved["model"], strict=True)
    if codec.context_length != 128 or tuple(codec.vq.levels) != LEVELS:
        raise ValueError("native128 architecture must stay intact")
    if any(not torch.equal(v.cpu(), saved["model"][k]) for k, v in codec.state_dict().items()):
        raise ValueError("every released parameter and buffer must match")
    return codec.cuda().float().eval().requires_grad_(False)


@torch.no_grad()
def reconstruct(codec, row, t):
    ids = torch.tensor(row["packed_ids"], device="cuda", dtype=torch.long)
    codes = codec.encode(ids)
    logits = codec.decode(codes)
    if logits.shape != (len(ids), 128, 30528) or not torch.isfinite(logits).all():
        raise ValueError("full native finite decoder logits required")
    raw = logits.argmax(-1).flatten().cpu().tolist()
    end = raw.index(101) if 101 in raw else len(raw)
    text = t.decode(raw[:end], skip_special_tokens=True).strip()
    gold = row["target_ids"]
    return {"id": row["id"], "text": text, "native_round_trip": row["native_round_trip"],
        "raw_ids": raw, "codes": codes.cpu().tolist(), "target_ids": gold,
        "correct": sum(a == b for a, b in zip(raw, gold)), "total": len(gold),
        "first_EOS_correct": end == len(gold) - 1, "exact_native_ids": raw[:len(gold)] == gold,
        "active_invalid_ids": sum(v >= len(t) for v in raw[:end]),
        "fits_one_native_frame": row["fits_one_native_frame"]}


def require_gpu():
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "2" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or not torch.cuda.is_available() or "2060 SUPER" not in torch.cuda.get_device_name()
            or torch.cuda.mem_get_info()[0] < 4_000_000_000):
        raise ValueError("physical GPU2 with4GBfree required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def measure(pilot=False):
    require_gpu()
    vendor.terminal(DATA)
    if file_hash(DATA / "cases.json") != q.read(DATA / "result.json")["cases_sha256"]:
        raise ValueError("complete native target metadata changed")
    all_rows = q.read(DATA / "cases.json")
    ids = q.read(DATA / "selection.json")["pilot_ids"]
    rows = [next(r for r in all_rows if r["id"] == i) for i in ids] if pilot else all_rows
    extra = [DATA / n for n in ("producer.json", "result.json", "cases.json", "selection.json", "worker_terminal_receipt.json")]
    extra += [RELEASE / "files/lm1b/tokenizer.pt", RELEASE / "files/lm1b/small/config.yaml"]
    retained = {}
    if not pilot:
        vendor.terminal(PILOT)
        extra += [PILOT / n for n in ("producer.json", "result.json", "outputs.json", "worker_terminal_receipt.json")]
        retained = {r["id"]: r for r in q.read(PILOT / "outputs.json")}
    dest = PILOT if pilot else RECON
    p = producing(dest, "systems_native128_codec_pilot" if pilot else "all690_native128_codec_reconstruction",
        extra, {"cases": len(rows), "pilot": pilot, "physical_gpu": 2, "peak_limit_bytes": 4700000000})
    codec, t = load_codec(), tokenizer()
    before = vendor.module_hash(codec)
    torch.cuda.reset_peak_memory_stats()
    outputs = []
    for row in rows:
        output = reconstruct(codec, row, t)
        if pilot and output != reconstruct(codec, row, t):
            raise ValueError("native reconstruction must repeat exactly")
        if row["id"] in retained and output != retained[row["id"]]:
            raise ValueError("fresh native reload changed a retained pilot")
        outputs.append(output)
        if pilot or len(outputs) % 16 == 0 or len(outputs) == len(rows):
            write_json(dest / "outputs.json", outputs)
        if pilot or len(outputs) % 32 == 0 or len(outputs) == len(rows):
            print(json.dumps({"reconstructed": len(outputs), "total": len(rows)}), flush=True)
    if vendor.module_hash(codec) != before or any(v.grad is not None for v in codec.parameters()):
        raise ValueError("native codec state changed")
    peak = torch.cuda.max_memory_allocated()
    if peak > 4700000000:
        raise ValueError("native codec exceeds planned peak")
    vendor.verify(p)
    result = {"status": "systems_native_codec_passed_not_quality" if pilot else "all690_native_codec_measured_semantics_pending",
        "cases": len(outputs), "correct": sum(v["correct"] for v in outputs), "total": sum(v["total"] for v in outputs),
        "exact_native_ids": sum(v["exact_native_ids"] for v in outputs), "first_EOS_correct": sum(v["first_EOS_correct"] for v in outputs),
        "active_invalid_ids": sum(v["active_invalid_ids"] for v in outputs), "codec_parameter_hash": before,
        "codec_unchanged": True, "peak_allocated_bytes": peak, "source_only_task_accuracy": None,
        "no_parent_loaded": True, "fits_one_frame": sum(v["fits_one_native_frame"] for v in outputs),
        "overlength_in_denominator": sum(not v["fits_one_native_frame"] for v in outputs),
        "outputs_sha256": file_hash(dest / "outputs.json"), "producer_fingerprint": p["fingerprint"]}
    write_json(dest / "result.json", result)
    print(json.dumps(result), flush=True)


def workflow():
    if any(p.exists() for p in (WORKFLOW, RELEASE, DATA, PILOT, RECON)):
        raise ValueError("fresh complete native codec workflow required")
    p = producing(WORKFLOW, "download_then_native_codec_control", [INVENTORY / "inventory.json", INVENTORY / "bert_coverage.json"])
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    (WORKFLOW / "stage_receipts").mkdir()
    try:
        for stage, dest, gpu in (("download", RELEASE, False), ("prepare", DATA, False),
                ("pilot", PILOT, True), ("measure", RECON, True)):
            vendor.verify(p)
            command = [sys.executable, "-u", "-m", MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "2" if gpu else ""})
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"stage": stage, "exit_code": code, "child_pid": child.pid,
                "observer_pid": os.getpid(), "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if dest.exists():
                write_json(dest / "worker_terminal_receipt.json", receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("native codec stage failed: " + stage)
        vendor.verify(p)
        journal.update(status="native_codec_numeric_control_completed_semantics_unmeasured", active_stage=None, child_pid=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("download", "prepare", "pilot", "measure", "workflow"))
    args = parser.parse_args()
    {"download": download, "prepare": prepare, "pilot": lambda: measure(True), "measure": measure, "workflow": workflow}[args.stage]()
