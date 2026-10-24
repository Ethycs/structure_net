"""Evaluate only the fixed final decoder repair; retain canonical first judgments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_lm1b_decoder_repair as repair
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, framed, prior = repair.native, repair.framed, repair.semantics
q, reference, oracle = native.q, prior.reference, prior.oracle
ROOT = native.ROOT
HYPOTHESIS = "ncm-lm1b-fixed1024-decoder-repair-semantic-v1"
MODULE = "experiments.structure_net.ncm_lm1b_decoder_repair_eval"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_lm1b_decoder_repair"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-decoder-repair-evaluation-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-lm1b-decoder-repair.md")
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_lm1b_decoder_repair.py")
WORKFLOW = ROOT / "20261024_lm1b_decoder_repair_eval_workflow_v1"
PILOT = ROOT / "20261024_lm1b_decoder_repair_reconstruction_pilot_v1"
RECON = ROOT / "20261024_lm1b_decoder_repair_reconstruction_v1"
PACKETS = ROOT / "20261024_lm1b_decoder_repair_packets_v1"
JUDGE_PILOT = ROOT / "20261024_lm1b_decoder_repair_judge_pilot_v1"
GRADE = ROOT / "20261024_lm1b_decoder_repair_grade_v1"
RESULTS = ROOT / "20261024_lm1b_decoder_repair_results_v1"
ARMS = ("uncompressed", "bert_roundtrip", "PAD0_codec", "repaired_codec")
BASE_CACHE = prior.GRADE / "judgment_cache.json"
BASE_RESULTS = prior.RESULTS
BASE_PACKETS = prior.PACKETS
BASE_HYPOTHESIS = prior.HYPOTHESIS
BASE_CACHE_SHA = "df7de0055109fccc6fed26aba6fb4f3dbf0df7a29f520da77dcb2460bc60eab9"
_producing, _released_loader = native.producing, native.load_codec


def configure():
    native.vendor.corpus.quality.dependencies()
    reference.profile.install()
    native.HYPOTHESIS, native.DATA, native.PILOT, native.RECON = HYPOTHESIS, framed.DATA, PILOT, RECON
    native.producing, native.load_codec, native.reconstruct = producing, load_repaired, reconstruct
    for name in ("PACKETS", "JUDGE_PILOT", "GRADE", "ARMS"):
        setattr(prior, name, globals()[name])
    prior.producing, prior.base_cache = producing, base_cache


def base_cache():
    campaign = q.read(BASE_RESULTS / "campaign_results.json")
    receipt = q.read(BASE_RESULTS / (BASE_HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(BASE_CACHE) != BASE_CACHE_SHA or len(q.read(BASE_CACHE)) != 2239
            or file_hash(BASE_RESULTS / "campaign_results.json") != repair.PARENT_SHA
            or campaign["artifacts"].get(str(BASE_CACHE)) != BASE_CACHE_SHA
            or campaign["canonical_first_judgments"] != 2239
            or receipt["record"]["campaign_sha256"] != repair.PARENT_SHA
            or receipt["exact"]["campaign_sha256"] != repair.PARENT_SHA
            or receipt["status"] != "registered_and_read_back"):
        raise ValueError("latest2239 registered canonical firsts required")
    cache = q.read(BASE_CACHE)
    prior.audit_cache(cache)
    return cache


def producing(root, stage, inputs=(), config=None):
    sources = list(dict.fromkeys([Path(__file__), PLAN, AUDITOR_FILE,
        Path("tests/structure_net/test_ncm_lm1b_decoder_repair_eval.py"),
        Path(repair.__file__), repair.PLAN, Path(framed.__file__), framed.PLAN,
        *[Path(p) for p in q.read(prior.WORKFLOW / "producer.json")["source_hashes"]]]))
    common = [BASE_CACHE, BASE_RESULTS / "campaign_results.json",
        BASE_RESULTS / (BASE_HYPOTHESIS + "_registry_readback.json"),
        repair.TRAIN / "model.pt", repair.TRAIN / "result.json", repair.TRAIN / "producer.json",
        repair.TRAIN / "worker_terminal_receipt.json", repair.WORKFLOW / "result.json",
        repair.WORKFLOW / "tool_terminal_receipt.json",
        framed.DATA / "cases.json", framed.DATA / "selection.json", framed.RECON / "outputs.json",
        framed.RECON / "result.json", BASE_PACKETS / "cases.json"]
    # Freeze request receipts as well as the inherited first-decision cache.
    for entry in q.read(BASE_CACHE).values():
        common += [Path(p) for p in entry["row"]["attempts"]]
    p = _producing(root, stage, [*inputs, *common, *sources], {**(config or {}),
        "hypothesis_id": HYPOTHESIS, "fixed_final_update": 1024, "reconstruction_cases": 690,
        "semantic_cases": 64, "semantic_arms": ARMS, "canonical_prior_firsts": 2239,
        "frozen_encoder_and_codebooks": True, "checkpoint_selection": "only fixed final1024",
        "source_prediction": False, "oracle_target_codes": True, "task_accuracy": None,
        "model_training_executed_in_this_evaluation": False, "judges": q.JUDGES,
        "rubric": reference.profile.RUBRIC, "same_case_joint_minimum": 61,
        "all64_valid_required": True, "workers": 4, "max_attempts": 3,
        "no_semantic_rerolls": True, "parent_failure_campaign": repair.PARENT_SHA})
    p["hypothesis_id"] = HYPOTHESIS
    p["config"]["schema"] = "nal.ncm-lm1b-decoder-repair-semantic.v1"
    for i, source in enumerate(sources):
        p["source_hashes"][str(source)] = file_hash(source)
        (root / "producer_source" / f"repair_eval_{i}_{source.name}").write_bytes(source.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def load_repaired():
    native.vendor.terminal(repair.TRAIN)
    state = q.read(repair.TRAIN / "result.json")
    if (state["updates"] != 1024 or not state["exact_reload"]
            or file_hash(repair.TRAIN / "model.pt") != state["checkpoint_sha256"]):
        raise ValueError("exact final1024 repair checkpoint required")
    codec = _released_loader()
    saved = torch.load(repair.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    if saved["step"] != 1024 or saved["frozen_state_sha256"] != repair.frozen_hash(codec):
        raise ValueError("native representation changed before decoder load")
    codec.decoder.load_state_dict(saved["decoder"], strict=True)
    codec.eval().requires_grad_(False)
    if (repair.frozen_hash(codec) != state["frozen_state_sha256_unchanged"]
            or native.vendor.module_hash(codec) != state["final_codec_hash"]):
        raise ValueError("fixed final decoder or frozen representation differs")
    return codec


@torch.no_grad()
def reconstruct(codec, row, tokenizer):
    inputs = torch.tensor(row["packed_ids"], device="cuda", dtype=torch.long)
    codes = codec.encode(inputs)
    logits = codec.decode(codes)
    if logits.shape != (len(inputs), 128, 30528) or not torch.isfinite(logits).all():
        raise ValueError("finite full native logits required")
    raw_frames = logits.argmax(-1).cpu().tolist()
    raw = [v for frame in raw_frames for v in frame]
    text, ends, bos = framed.render(raw_frames, tokenizer)
    gold = row["target_ids"]
    return {"id": row["id"], "text": text, "native_round_trip": row["native_round_trip"],
        "raw_ids": raw, "codes": codes.cpu().tolist(), "target_ids": gold,
        "correct": sum(a == b for a, b in zip(raw, gold)), "total": len(gold),
        "first_EOS_correct": ends == [sum(v != 0 for v in f) - 1 for f in row["packed_ids"]],
        "frame_EOS_positions": ends, "all_frame_BOS_correct": all(bos),
        "exact_native_ids": raw[:len(gold)] == gold,
        "active_invalid_ids": sum(v >= 30522 for f, end in zip(raw_frames, ends) for v in f[:end]),
        "fits_one_native_frame": row["fits_one_native_frame"]}


def packets():
    native.vendor.terminal(RECON)
    original = prior.select(q.read(native.CASES)["train"])
    old_cases = q.read(BASE_PACKETS / "cases.json")
    if [r["id"] for r in original] != [r["id"] for r in old_cases]:
        raise ValueError("same prospective64 source IDs required")
    outputs = {r["id"]: r for r in q.read(RECON / "outputs.json")}
    old = {r["id"]: r for r in q.read(framed.RECON / "outputs.json")}
    cases = []
    for r, previous in zip(original, old_cases):
        texts = {arm: previous["texts"][arm] for arm in ARMS if arm != "repaired_codec"}
        texts["repaired_codec"] = outputs[r["id"]]["text"]
        if outputs[r["id"]]["codes"] != old[r["id"]]["codes"]:
            raise ValueError("decoder repair must preserve every native code")
        cases.append({**{k: r[k] for k in reference.IDENTITY}, "texts": texts})
    proof = reference.verify_public(cases)
    payloads = {arm: [q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["texts"][arm]}])
        for r in cases] for arm in ARMS}
    p = producing(PACKETS, "same64_source_grounded_fixed_final_repair_packets", [native.CASES,
        RECON / "outputs.json", reference.PUBLIC_FILE], {"CPU_only": True})
    write_json(PACKETS / "cases.json", cases)
    write_json(PACKETS / "exact_packets.json", payloads)
    write_json(PACKETS / "public_payload_proof.json", {"public_file_sha256": reference.PUBLIC_SHA256,
        "exact_sources": proof, "private_data_in_packets": False,
        "exact_packets_sha256": file_hash(PACKETS / "exact_packets.json")})
    native.vendor.verify(p)
    write_json(PACKETS / "result.json", {"status": "fixed_final_repair_public_packets_prepared", "cases": 64,
        "hashes": {n: file_hash(PACKETS / n) for n in ("cases.json", "exact_packets.json", "public_payload_proof.json")},
        "task_accuracy": None, "producer_fingerprint": p["fingerprint"]})


def workflow():
    native.vendor.terminal(repair.TRAIN)
    base_cache(); prior.require_judges()
    if q.read(repair.WORKFLOW / "result.json")["status"] != "decoder_repair_training_completed_quality_unmeasured":
        raise ValueError("whole repair training workflow must actually finish first")
    if any(p.exists() for p in (WORKFLOW, PILOT, RECON, PACKETS, JUDGE_PILOT, GRADE, RESULTS, REPORT)):
        raise ValueError("fresh fixed-checkpoint evaluation required")
    p = producing(WORKFLOW, "fixed_final_repair_reconstruction_and_semantics")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    try:
        for stage, root, gpu in (("pilot", PILOT, True), ("measure", RECON, True), ("packets", PACKETS, False),
                ("judge-pilot", JUDGE_PILOT, False), ("grade", GRADE, False),
                ("finalize", RESULTS, False), ("register", RESULTS, False)):
            native.vendor.verify(p)
            command = [sys.executable, "-u", "-m", AUDITOR if stage in ("finalize", "register") else MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "2" if gpu else ""})
            journal.update(active_stage=stage, child_pid=child.pid); write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            record = {"stage": stage, "exit_code": code, "child_pid": child.pid, "observer_pid": os.getpid(),
                "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), record)
            if root.exists():
                write_json(root / ("registry_terminal_receipt.json" if stage == "register" else "worker_terminal_receipt.json"), record)
            journal["stages"].append(record)
            if code: raise RuntimeError("fixed repair evaluation failed: " + stage)
        native.vendor.verify(p)
        journal.update(status="fixed_repair_reconstruction_and_semantics_completed", active_stage=None, child_pid=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("pilot", "measure", "packets", "judge-pilot", "grade", "workflow"))
    args = parser.parse_args()
    configure()
    {"pilot": lambda: native.measure(True), "measure": native.measure, "packets": packets,
        "judge-pilot": lambda: prior.grade(True), "grade": prior.grade, "workflow": workflow}[args.stage]()
