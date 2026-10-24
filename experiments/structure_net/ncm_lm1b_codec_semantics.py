"""Matched CLS-tail control and source-grounded semantic codec qualification."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_lm1b_codec as native
from experiments.structure_net import ncm_lm1b_framed_codec as framed
from experiments.structure_net import ncm_oracle_codec_semantic as oracle
from experiments.structure_net import ncm_word_final_eval as reference
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT, q = native.ROOT, native.q
HYPOTHESIS = "ncm-lm1b-CLS-padding-semantic-control-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-padding-semantics-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-lm1b-padding-semantics.md")
MODULE = "experiments.structure_net.ncm_lm1b_codec_semantics"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_lm1b_codec_semantics.py")
WORKFLOW = ROOT / "20261024_lm1b_padding_semantic_workflow_v1"
DATA = ROOT / "20261024_lm1b_CLS_padding_data_v1"
PILOT = ROOT / "20261024_lm1b_CLS_padding_pilot_v1"
RECON = ROOT / "20261024_lm1b_CLS_padding_reconstruction_v1"
PACKETS = ROOT / "20261024_lm1b_padding_semantic_packets_v1"
JUDGE_PILOT = ROOT / "20261024_lm1b_padding_semantic_pilot_v1"
GRADE = ROOT / "20261024_lm1b_padding_semantic_grade_v1"
RESULTS = ROOT / "20261024_lm1b_padding_semantic_results_v1"
ARMS = ("uncompressed", "bert_roundtrip", "PAD0_codec", "CLS_codec")
OLD_CACHE = ROOT / "20261024_pinned_date_development_grade_v1/judgment_cache.json"
CACHE_RESULTS = ROOT / "20261024_pinned_date_development_results_v1"
CACHE_HYPOTHESIS = "ncm-pinned-date-full690-semantic-development-v1"
CACHE_SHA = "ea07f76887bd15c07ffaea8fc70339fee8ffa41731c6ce5572916dea51b2c4a4"
CACHE_CAMPAIGN = "18c00793c40489c0f028d8dce895e5f00ed3f84ebe4e683dc997e9e810bfff47"
_producing = native.producing


def replace_tail_padding(row):
    ends = [sum(v != 0 for v in frame) - 1 for frame in row["packed_ids"]]
    if any(len(frame) != 128 or frame[0] != 101 or frame[end] != 101 or any(v != 0 for v in frame[end + 1:])
            for frame, end in zip(row["packed_ids"], ends)):
        raise ValueError("complete genuine CLS boundaries and PAD0-only tails required")
    frames = [[101 if v == 0 else v for v in frame] for frame in row["packed_ids"]]
    return {**row, "packed_ids": frames, "frame_ends": ends}


def select(rows):
    if len(rows) != 690 or len({r["id"] for r in rows}) != 690:
        raise ValueError("all690 distinct cases required before selection")
    key = lambda r: q.digest("native-codec-semantic-64142:" + r["id"])
    chosen = sorted(rows[:252], key=key)[:32] + sorted(rows[252:], key=key)[:32]
    return sorted(chosen, key=lambda r: q.digest("native-codec-order-65142:" + r["id"]))


def configure():
    native.vendor.corpus.quality.dependencies()
    reference.profile.install()
    if oracle.q is not q or oracle.first is not reference.first:
        raise ValueError("one canonical packet/rubric/judge identity required")
    for name in ("HYPOTHESIS", "DATA", "PILOT", "RECON"):
        setattr(native, name, globals()[name])
    native.producing, native.reconstruct = producing, reconstruct


def producing(root, stage, inputs=(), config=None):
    upstream = q.read(ROOT / "20261024_pinned_date_development_workflow_v1/producer.json")
    files = list(dict.fromkeys([Path(__file__), PLAN, AUDITOR_FILE,
        Path("tests/structure_net/test_ncm_lm1b_codec_semantics.py"), Path(oracle.__file__),
        Path("experiments/neural_architecture_lab/finalize_ncm_dense_semantic_development.py"),
        Path("experiments/neural_architecture_lab/ncm_result_registry.py"),
        Path(framed.__file__), framed.PLAN, *[Path(p) for p in upstream["source_hashes"]]]))
    common = [OLD_CACHE, CACHE_RESULTS / "campaign_results.json",
        CACHE_RESULTS / (CACHE_HYPOTHESIS + "_registry_readback.json"),
        framed.DATA / "cases.json", framed.DATA / "selection.json", framed.RECON / "outputs.json",
        framed.RECON / "result.json", framed.WORKFLOW / "result.json",
        ROOT / "20261024_lm1b_framed_codec_results_v1/campaign_results.json",
        ROOT / "20261024_lm1b_controls_backend_readback_v1.json",
        framed.AUTHOR / "boundary_population.json"]
    for entry in q.read(OLD_CACHE).values():
        common += [Path(p) for p in entry["row"]["attempts"]]
    p = _producing(root, stage, [*inputs, *common, *files], {**(config or {}),
        "hypothesis_id": HYPOTHESIS, "changed_factor": "only target-frame tail PAD0 replaced with CLS101",
        "same_lexical_tokens_and_genuine_EOS": True, "same_native_weights": True,
        "native_frame": 128, "representation_cases": 690, "semantic_cases": 64,
        "semantic_selection": "32 each original252/reviewed438, fixed SHA seeds64142/65142",
        "semantic_arms": ARMS, "judges": q.JUDGES, "rubric": reference.profile.RUBRIC,
        "canonical_prior_firsts": 2051, "workers": 4, "max_attempts": 3,
        "semantic_rerolls": False, "joint_minimum": 61, "all64_valid_required": True,
        "source_prediction": False, "oracle_target_codes": True, "task_accuracy": None})
    p["config"]["schema"] = "nal.ncm-lm1b-padding-semantic-control.v1"
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"padding_semantic_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def prepare():
    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only preparation required")
    for root in (framed.DATA, framed.RECON):
        native.vendor.terminal(root)
    if file_hash(native.CASES) != native.CASES_SHA:
        raise ValueError("complete690 original targets changed")
    old = q.read(framed.DATA / "cases.json")
    rows = [replace_tail_padding(r) for r in old]
    if len(rows) != 690:
        raise ValueError("all690 retained")
    p = producing(DATA, "same690_targets_only_CLS_tail_intervention", [native.CASES], {"CPU_only": True})
    write_json(DATA / "cases.json", rows)
    selection = q.read(framed.DATA / "selection.json")
    selection["semantic_case_ids"] = [r["id"] for r in select(q.read(native.CASES)["train"])]
    write_json(DATA / "selection.json", selection)
    native.vendor.verify(p)
    write_json(DATA / "result.json", {"status": "all690_same_targets_CLS_tail_prepared", "cases": 690,
        "cases_sha256": file_hash(DATA / "cases.json"), "semantic_cases": 64,
        "only_tail_padding_changed": True, "target_truncation": False, "task_accuracy": None,
        "producer_fingerprint": p["fingerprint"]})


@torch.no_grad()
def reconstruct(codec, row, t):
    ids = torch.tensor(row["packed_ids"], device="cuda", dtype=torch.long)
    codes = codec.encode(ids)
    logits = codec.decode(codes)
    if logits.shape != (len(ids), 128, 30528) or not torch.isfinite(logits).all():
        raise ValueError("full finite native decoder required")
    frames = logits.argmax(-1).cpu().tolist()
    raw = [v for frame in frames for v in frame]
    text, ends, bos = framed.render(frames, t)
    gold = row["target_ids"]
    return {"id": row["id"], "text": text, "native_round_trip": row["native_round_trip"],
        "raw_ids": raw, "codes": codes.cpu().tolist(), "target_ids": gold,
        "correct": sum(a == b for a, b in zip(raw, gold)), "total": len(gold),
        "first_EOS_correct": ends == row["frame_ends"], "frame_EOS_positions": ends,
        "all_frame_BOS_correct": all(bos), "exact_native_ids": raw[:len(gold)] == gold,
        "active_invalid_ids": sum(v >= 30522 for frame, end in zip(frames, ends) for v in frame[:end]),
        "fits_one_native_frame": row["fits_one_native_frame"]}


def packets():
    native.vendor.terminal(RECON)
    original = q.read(native.CASES)["train"]
    chosen = select(original)
    if [r["id"] for r in chosen] != q.read(DATA / "selection.json")["semantic_case_ids"]:
        raise ValueError("prospective semantic selection changed")
    padded = {r["id"]: r for r in q.read(RECON / "outputs.json")}
    old = {r["id"]: r for r in q.read(framed.RECON / "outputs.json")}
    cases = []
    for r in chosen:
        texts = {"uncompressed": r["target_text"], "bert_roundtrip": old[r["id"]]["native_round_trip"],
            "PAD0_codec": old[r["id"]]["text"], "CLS_codec": padded[r["id"]]["text"]}
        cases.append({**{k: r[k] for k in reference.IDENTITY}, "texts": texts})
    proof = reference.verify_public(cases)
    payloads = {arm: [q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["texts"][arm]}])
        for r in cases] for arm in ARMS}
    p = producing(PACKETS, "fixed64_public_source_grounded_four_arm_packets", [native.CASES,
        RECON / "outputs.json", reference.PUBLIC_FILE], {"CPU_only": True, "cases": 64, "arm_cases": 256})
    write_json(PACKETS / "cases.json", cases)
    write_json(PACKETS / "exact_packets.json", payloads)
    write_json(PACKETS / "public_payload_proof.json", {"public_file_sha256": reference.PUBLIC_SHA256,
        "exact_sources": proof, "private_data_in_packets": False, "exact_packets_sha256": file_hash(PACKETS / "exact_packets.json")})
    native.vendor.verify(p)
    write_json(PACKETS / "result.json", {"status": "four_arm_public_semantic_packets_prepared", "cases": 64,
        "hashes": {n: file_hash(PACKETS / n) for n in ("cases.json", "exact_packets.json", "public_payload_proof.json")},
        "task_accuracy": None, "producer_fingerprint": p["fingerprint"]})


def base_cache():
    campaign = q.read(CACHE_RESULTS / "campaign_results.json")
    receipt = q.read(CACHE_RESULTS / (CACHE_HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(OLD_CACHE) != CACHE_SHA or file_hash(CACHE_RESULTS / "campaign_results.json") != CACHE_CAMPAIGN
            or campaign["artifacts"].get(str(OLD_CACHE)) != CACHE_SHA
            or campaign["canonical_first_judgments"] != 2051 or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != CACHE_CAMPAIGN or receipt["exact"]["campaign_sha256"] != CACHE_CAMPAIGN):
        raise ValueError("latest2051 registered first decisions required")
    cache = q.read(OLD_CACHE)
    if len(cache) != 2051:
        raise ValueError("all2051 firsts required")
    audit_cache(cache)
    return cache


def audit_cache(cache):
    for key, entry in cache.items():
        if key != reference.first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("first-decision identity changed")
        reference.first.audit_row(entry["packet"], entry["role"], entry["row"])


def require_judges():
    for role in ("secondary", "primary"):
        root = ROOT / ("20261023_budget128_judge_" + role + "_v1")
        native.vendor.terminal(root)
        result = q.read(root / "result.json")
        if result["status"] != "passed" or result["controls"] != 50 or result["correct"] != 50:
            raise ValueError("same qualified50/50 judges required")


def grade(pilot=False):
    native.vendor.terminal(PACKETS)
    require_judges()
    for name, sha in q.read(PACKETS / "result.json")["hashes"].items():
        if file_hash(PACKETS / name) != sha:
            raise ValueError("public packet proof changed")
    all_cases, payloads = q.read(PACKETS / "cases.json"), q.read(PACKETS / "exact_packets.json")
    cache = base_cache()
    if not pilot:
        native.vendor.terminal(JUDGE_PILOT)
        if q.read(JUDGE_PILOT / "result.json")["status"] != "systems_semantic_pilot_passed":
            raise ValueError("actual four-arm schema pilot required")
        for key, entry in q.read(JUDGE_PILOT / "judgment_cache.json").items():
            reference.first.remember(cache, key, entry)
    inherited = dict(cache)
    cases = all_cases[:4] if pilot else all_cases
    root = JUDGE_PILOT if pilot else GRADE
    inputs = [PACKETS / n for n in ("producer.json", "result.json", "cases.json", "exact_packets.json", "public_payload_proof.json", "worker_terminal_receipt.json")]
    if not pilot:
        inputs += [JUDGE_PILOT / n for n in ("result.json", "judgment_cache.json", "worker_terminal_receipt.json")]
    p = producing(root, "semantic_path_pilot" if pilot else "fixed64_four_arm_oracle_semantic_quality", inputs,
        {"pilot": pilot, "cases": len(cases), "prior_first_judgments": len(inherited)})
    write_json(root / "judgment_cache.json", cache)
    results = {arm: {"secondary": oracle.grade_arm(root, arm, "secondary", cases, payloads, cache), "primary": None}
        for arm in ARMS}
    if not pilot and oracle.qualifies(results["uncompressed"]["secondary"]):
        for arm in ARMS:
            if oracle.qualifies(results[arm]["secondary"]):
                results[arm]["primary"] = oracle.grade_arm(root, arm, "primary", cases, payloads, cache)
    if any(cache.get(k) != v for k, v in inherited.items()):
        raise ValueError("a canonical first decision changed")
    audit_cache(cache)
    native.vendor.verify(p)
    valid = all(v["secondary"]["valid"] == len(cases) for v in results.values())
    status = ("systems_semantic_pilot_passed" if valid else "systems_semantic_pilot_invalid") if pilot else "completed"
    write_json(root / "result.json", {"status": status, "results": results,
        "canonical_first_judgments": len(cache), "prior_first_judgments_preserved": len(inherited),
        "oracle_target_codes": True, "source_prediction": False, "task_accuracy": None,
        "final_test_opened": False, "producer_fingerprint": p["fingerprint"]})
    if pilot and not valid:
        raise ValueError("semantic pilot invalid; first failures preserved")


def workflow():
    base_cache()
    require_judges()
    if any(p.exists() for p in (WORKFLOW, DATA, PILOT, RECON, PACKETS, JUDGE_PILOT, GRADE, RESULTS, REPORT)):
        raise ValueError("fresh complete codec semantic workflow required")
    p = producing(WORKFLOW, "matched_padding_then_semantic_qualification")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    try:
        for stage, dest, gpu in (("prepare", DATA, False), ("pilot", PILOT, True), ("measure", RECON, True),
                ("packets", PACKETS, False), ("judge-pilot", JUDGE_PILOT, False), ("grade", GRADE, False),
                ("finalize", RESULTS, False), ("register", RESULTS, False)):
            native.vendor.verify(p)
            command = [sys.executable, "-u", "-m", AUDITOR if stage in ("finalize", "register") else MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "2" if gpu else ""})
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"stage": stage, "exit_code": code, "child_pid": child.pid, "observer_pid": os.getpid(),
                "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if dest.exists():
                write_json(dest / ("registry_terminal_receipt.json" if stage == "register" else "worker_terminal_receipt.json"), receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("codec semantic stage failed: " + stage)
        native.vendor.verify(p)
        journal.update(status="padding_semantic_control_completed", active_stage=None, child_pid=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "measure", "packets", "judge-pilot", "grade", "workflow"))
    args = parser.parse_args()
    configure()
    {"prepare": prepare, "pilot": lambda: native.measure(True), "measure": native.measure,
        "packets": packets, "judge-pilot": lambda: grade(True), "grade": grade, "workflow": workflow}[args.stage]()
