"""Native CLS framing control after the preserved LM1B boundary-format failure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_lm1b_codec as prior
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT, q = prior.ROOT, prior.q
HYPOTHESIS = "ncm-lm1b-framed-codec-compatibility-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-framed-codec-plan.md")
MODULE = "experiments.structure_net.ncm_lm1b_framed_codec"
DATA = ROOT / "20261024_lm1b_framed_codec_data_v1"
NATIVE = ROOT / "20261024_lm1b_framed_codec_author_control_v1"
PILOT = ROOT / "20261024_lm1b_framed_codec_pilot_v1"
RECON = ROOT / "20261024_lm1b_framed_codec_reconstruction_v1"
WORKFLOW = ROOT / "20261024_lm1b_framed_codec_workflow_v1"
AUTHOR = ROOT / "20261024_lm1b_author_format_audit_v1"
OLD_DATA, OLD_RECON, OLD_WORKFLOW = prior.DATA, prior.RECON, prior.WORKFLOW
_producing = prior.producing


def pack(ids):
    if not ids or any(not 0 < x < 30522 or x == 101 for x in ids):
        raise ValueError("native lexical IDs without PAD or CLS required")
    chunks = [ids[i:i + 126] for i in range(0, len(ids), 126)]
    frames = [[101, *chunk, 101] for chunk in chunks]
    return [v for frame in frames for v in frame], [frame + [0] * (128 - len(frame)) for frame in frames]


def render(raw_frames, tokenizer):
    lexical, ends, bos = [], [], []
    for raw in raw_frames:
        if len(raw) != 128:
            raise ValueError("complete native128 frame required")
        start = 1 if raw[0] == 101 else 0
        end = next((i for i in range(start, 128) if raw[i] == 101), 128)
        lexical.extend(raw[start:end])
        ends.append(end)
        bos.append(raw[0] == 101)
    return tokenizer.decode(lexical, skip_special_tokens=True).strip(), ends, bos


def producing(root, stage, inputs=(), config=None):
    files = [Path(__file__), PLAN, Path("tests/structure_net/test_ncm_lm1b_framed_codec.py")]
    extra = [AUTHOR / "observation.json", OLD_RECON / "result.json", OLD_RECON / "outputs.json",
        OLD_WORKFLOW / "result.json", *files]
    p = _producing(root, stage, [*inputs, *extra], {**(config or {}),
        "schema": "nal.ncm-lm1b-framed-codec.v1", "changed_factors": ["explicit initial CLS", "per-frame boundary parsing"],
        "frame": "CLS101 + atmost126 lexical IDs + CLS101 + PAD0",
        "framing_basis": "released validation frames all begin/end CLS101; input strings have no PAD/CLS",
        "renderer": "consume predicted leading CLS if present; stop at next predicted CLS per frame; join lexical IDs",
        "original_wrong_boundary_control_preserved": True, "first_EOS_correct_field": "all window ends correct",
        "native_author_control": "first4 released128-token validation frames; no training on validation"})
    p["config"]["schema"] = "nal.ncm-lm1b-framed-codec.v1"
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"framed_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def configure():
    for name in ("HYPOTHESIS", "DATA", "PILOT", "RECON"):
        setattr(prior, name, globals()[name])
    prior.producing, prior.reconstruct = producing, reconstruct


def prepare():
    prior.vendor.terminal(OLD_DATA)
    if file_hash(prior.CASES) != prior.CASES_SHA or torch.cuda.is_initialized():
        raise ValueError("CPU-only unchanged full TRAIN required")
    rows = q.read(prior.CASES)["train"]
    t = prior.tokenizer()
    p = producing(DATA, "complete690_CLS_framed_targets", [prior.CASES, OLD_DATA / "cases.json"], {"CPU_only": True})
    values = []
    for row in rows:
        ids = t.encode(row["target_text"], add_special_tokens=False)
        full, frames = pack(ids)
        values.append({"id": row["id"], "source_id": row["source_id"], "group_id": row["group_id"],
            "source_sha256": row["source_sha256"], "source": row["source"], "target_text": row["target_text"],
            "native_round_trip": t.decode(ids, skip_special_tokens=True), "target_ids": full,
            "packed_ids": frames, "lexical_ids": ids, "fits_one_native_frame": len(frames) == 1})
    longest = min(values, key=lambda r: (-len(r["lexical_ids"]), r["id"]))
    write_json(DATA / "cases.json", values)
    write_json(DATA / "selection.json", {"pilot_ids": [values[0]["id"], longest["id"]]})
    prior.vendor.verify(p)
    write_json(DATA / "result.json", {"status": "all690_native_CLS_framed_cases_prepared", "cases": len(values),
        "fit_one_frame": sum(v["fits_one_native_frame"] for v in values),
        "overlength": sum(not v["fits_one_native_frame"] for v in values),
        "windows": sum(len(v["packed_ids"]) for v in values), "target_truncation": False,
        "cases_sha256": file_hash(DATA / "cases.json"), "producer_fingerprint": p["fingerprint"], "task_accuracy": None})
    print(json.dumps(q.read(DATA / "result.json")), flush=True)


@torch.no_grad()
def native_control():
    prior.require_gpu()
    import numpy as np
    proof = q.read(AUTHOR / "observation.json")
    for entry in proof["files"].values():
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("author native validation data changed")
    path = AUTHOR / "files/lm1b/valid_bert_30522.bin"
    p = producing(NATIVE, "first4_author_native_codec_control", [path], {"cases": 4, "physical_gpu": 2})
    inputs = torch.tensor(np.memmap(path, mode="r", dtype=np.uint16)[:512].reshape(4, 128).astype('int64'), device="cuda")
    if not ((inputs[:, 0] == 101).all() and (inputs[:, -1] == 101).all()):
        raise ValueError("author frame boundaries differ")
    codec = prior.load_codec()
    before = prior.vendor.module_hash(codec)
    torch.cuda.reset_peak_memory_stats()
    codes = codec.encode(inputs)
    direct = codec.vq.idx_to_z_q(codes)
    _, levels = codec.vq.idx_to_zhats(codes, return_zlevels=True)
    if not torch.equal(direct, levels[-1]):
        raise ValueError("native codec latent differs from author evaluation helper")
    raw = codec.decode(codes).argmax(-1)
    if not torch.equal(raw, codec.decoder(levels[-1]).argmax(-1)) or not torch.equal(codes, codec.encode(inputs)):
        raise ValueError("author decoder parity or repeated codec inference differs")
    if prior.vendor.module_hash(codec) != before:
        raise ValueError("author-native codec state changed")
    if torch.cuda.max_memory_allocated() > 4700000000 or any(v.grad is not None for v in codec.parameters()):
        raise ValueError("native control memory or frozen-gradient contract failed")
    result = {"status": "author_native_codec_helper_parity_exact", "cases": 4,
        "correct": int((raw == inputs).sum()), "total": inputs.numel(),
        "raw_ids": raw.cpu().tolist(), "input_ids": inputs.cpu().tolist(), "codes": codes.cpu().tolist(),
        "codec_hash": before, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "native_control_not_task_accuracy": True, "task_accuracy": None, "producer_fingerprint": p["fingerprint"]}
    write_json(NATIVE / "result.json", result)
    prior.vendor.verify(p)
    print(json.dumps({k: v for k, v in result.items() if k not in ("raw_ids", "input_ids", "codes")}), flush=True)


@torch.no_grad()
def reconstruct(codec, row, t):
    inputs = torch.tensor(row["packed_ids"], device="cuda", dtype=torch.long)
    codes = codec.encode(inputs)
    logits = codec.decode(codes)
    if logits.shape != (len(inputs), 128, 30528) or not torch.isfinite(logits).all():
        raise ValueError("native framed finite decoder logits required")
    frames = logits.argmax(-1).cpu().tolist()
    raw = [v for frame in frames for v in frame]
    text, ends, bos = render(frames, t)
    gold = row["target_ids"]
    expected_ends = [len([v for v in frame if v != 0]) - 1 for frame in row["packed_ids"]]
    return {"id": row["id"], "text": text, "native_round_trip": row["native_round_trip"],
        "raw_ids": raw, "codes": codes.cpu().tolist(), "target_ids": gold,
        "correct": sum(a == b for a, b in zip(raw, gold)), "total": len(gold),
        "first_EOS_correct": ends == expected_ends, "frame_EOS_positions": ends,
        "all_frame_BOS_correct": all(bos), "exact_native_ids": raw[:len(gold)] == gold,
        "active_invalid_ids": sum(v >= len(t) for frame, end in zip(frames, ends) for v in frame[:end]),
        "fits_one_native_frame": row["fits_one_native_frame"]}


def measure(pilot=False):
    prior.vendor.terminal(NATIVE)
    prior.measure(pilot)
    dest = PILOT if pilot else RECON
    result = q.read(dest / "result.json")
    result["all_window_EOS_correct"] = result.pop("first_EOS_correct")
    result["all_frame_BOS_correct"] = sum(v["all_frame_BOS_correct"] for v in q.read(dest / "outputs.json"))
    result["native_framing_control_passed"] = True
    write_json(dest / "result.json", result)


def workflow():
    if any(p.exists() for p in (WORKFLOW, DATA, NATIVE, PILOT, RECON)):
        raise ValueError("fresh framed codec workflow required")
    if q.read(OLD_WORKFLOW / "result.json")["status"] != "native_codec_numeric_control_completed_semantics_unmeasured":
        raise ValueError("preserved terminal original format control required")
    p = producing(WORKFLOW, "native_author_format_then_full690_framed_codec_control")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    try:
        for stage, dest, gpu in (("prepare", DATA, False), ("native", NATIVE, True), ("pilot", PILOT, True), ("measure", RECON, True)):
            prior.vendor.verify(p)
            command = [sys.executable, "-u", "-m", MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "2" if gpu else ""})
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"stage": stage, "exit_code": code, "child_pid": child.pid, "observer_pid": os.getpid(),
                "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if dest.exists():
                write_json(dest / "worker_terminal_receipt.json", receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("framed native codec stage failed: " + stage)
        prior.vendor.verify(p)
        journal.update(status="framed_native_codec_numeric_control_completed_semantics_unmeasured", active_stage=None, child_pid=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "native", "pilot", "measure", "workflow"))
    args = parser.parse_args()
    configure()
    {"prepare": prepare, "native": native_control, "pilot": lambda: measure(True), "measure": measure, "workflow": workflow}[args.stage]()
