"""CPU-only semantic diagnosis of saved NCM TRAIN outputs, including oracle history."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.structure_net import ncm_matched_clean_word as clean
from experiments.structure_net import ncm_matched_forecaster as forecaster
from experiments.structure_net.ncm_expanded_word_data_workflow import kernel_identity
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_label_quality import dependencies
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.ncm_frozen_body_public_data import PUBLIC_FILE, PUBLIC_SHA256

ROOT = clean.ROOT
HYPOTHESIS = "ncm-clean-word-TRAIN-semantic-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-clean-word-train-semantic-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-clean-word-train-semantic.md")
WORKFLOW = ROOT / "20261023_clean_word_train_semantic_workflow_v1"
PREP = ROOT / "20261023_clean_word_train_semantic_prepare_v1"
PILOT = ROOT / "20261023_clean_word_train_semantic_pilot_v1"
GRADE = ROOT / "20261023_clean_word_train_semantic_grade_v1"
RESULTS = ROOT / "20261023_clean_word_train_semantic_results_v1"
MODULE = "experiments.structure_net.ncm_clean_word_train_semantic"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_clean_word_train_semantic"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_clean_word_train_semantic.py")
OLD_CACHE = ROOT / "20261023_reviewed_draft_history_grade_v1/judgment_cache.json"
MODELS = ("previous_word", "clean_word")
ARMS = ("correct_history", "actual_history")
CONDITIONS = tuple(model + "__" + arm for model in MODELS for arm in ARMS)
reference = clean.recovery.evaluation.reference
q, first, profile = reference.q, reference.first, reference.profile


def configure():
    dependencies()
    clean.recovery.configure()
    profile.install()


def admitted():
    # Both valid fitting outcomes are eligible, without amending either GPU branch.
    return forecaster.admission() is not None


def qualifies(metrics):
    return metrics["cases"] == 32 and metrics["valid"] == 32 and metrics["passed"] >= 31


def build_observations(selected, measurements):
    if (len(selected) != 32 or len({r["id"] for r in selected}) != 32
            or set(measurements) != set(MODELS)):
        raise ValueError("exact32 TRAIN probes and both fixed model measurements required")
    result = {}
    for model in MODELS:
        values = measurements[model]
        if [r["id"] for r in values] != [r["id"] for r in selected]:
            raise ValueError("same ordered TRAIN identities required")
        for arm in ARMS:
            condition = model + "__" + arm
            rows = []
            for source, item in zip(selected, values):
                if (item["source_sha256"] != source["source_sha256"]
                        or item["diagnostic_group"] != source["diagnostic_group"]):
                    raise ValueError("saved source or TRAIN group differs")
                value = item["arms"][arm]
                raw = value["raw_word_ids"]
                if len(raw) != 192 or any(type(word) is not int or not 0 <= word < 50257 for word in raw):
                    raise ValueError("complete raw192-word frames required")
                rows.append({**{key: source[key] for key in
                    ("id", "source_id", "group_id", "source_sha256", "source", "spans", "diagnostic_group")},
                    "condition": condition, "model": model, "history": arm,
                    "oracle_target_codes": arm == "correct_history", "TRAIN_only": True,
                    "text": value["text"], "raw_word_ids": raw,
                    "global_EOS_position": value["global_EOS_position"]})
            result[condition] = rows
    return result


def packets_for(rows):
    # Deliberately exclude targets, model, history, fitting counts and raw codes.
    return [q.packet(row["id"], row["source"], [{"candidate_id": "summary", "text": row["text"]}])
            for row in rows]


def audit_cache(cache):
    for key, entry in cache.items():
        if key != first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("canonical first-judgment key differs")
        first.audit_row(entry["packet"], entry["role"], entry["row"])


def require_prior():
    campaign = q.read(clean.diagnostic.ARCHIVE / "campaign_results.json")
    if campaign["artifacts"].get(str(OLD_CACHE)) != file_hash(OLD_CACHE):
        raise ValueError("registered previous full first-judgment cache required")
    cache = q.read(OLD_CACHE)
    if len(cache) != 1572:
        raise ValueError("all1572 prior first judgments required")
    audit_cache(cache)
    return cache


def inputs_and_observations():
    configure()
    if not admitted():
        raise ValueError("actual completed and exactly registered clean-word component required")
    _, _, _, selected = clean.require_inputs()
    campaign = q.read(clean.RESULTS / "campaign_results.json")
    for name in ("result.json", "case_measurements.json"):
        path = clean.FIT / name
        if campaign["artifacts"].get(str(path)) != file_hash(path):
            raise ValueError("registered final clean-word measurements changed")
    measurements = {
        MODELS[0]: q.read(clean.diagnostic.DEST / "case_measurements.json"),
        MODELS[1]: q.read(clean.FIT / "case_measurements.json")}
    if (file_hash(clean.diagnostic.DEST / "case_measurements.json") !=
            "e24bf39dbf139198a274e1f345862e3f563145a41c7c67e6482257852fdc0f94"):
        raise ValueError("immutable previous32-case raw measurements required")
    observations = build_observations(selected, measurements)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(clean.recovery.study.data.GPT2, local_files_only=True)
    for rows in observations.values():
        for row in rows:
            text, end = reference.decode_words(row["raw_word_ids"], tokenizer)
            if row["text"] != text or row["global_EOS_position"] != (end if end < 192 else None):
                raise ValueError("complete original raw output decoding differs")
    verify_public(selected)
    return selected, observations


def producing(destination, stage, inputs, config, *, pending=False):
    study = clean.recovery.study
    paths = [Path(__file__), PLAN, AUDITOR_FILE, Path(first.__file__), first.AMENDMENT,
        Path(q.__file__), Path(q.original.__file__), Path(profile.__file__), profile.PLAN,
        Path(reference.__file__), Path(clean.__file__), clean.PLAN,
        Path(forecaster.__file__), forecaster.PLAN, Path(clean.diagnostic.__file__)]
    paths += [Path(name) for name in q.read(clean.WORKFLOW / "producer.json")["source_hashes"]]
    files = list({str(path.resolve()): path for path in paths}.values())
    common = [clean.WORKFLOW / "producer.json", clean.diagnostic.ARCHIVE / "campaign_results.json", OLD_CACHE]
    if not pending:
        if not admitted():
            raise ValueError("registered completed clean-word component required")
        common += [clean.WORKFLOW / "result.json", clean.RESULTS / "campaign_results.json",
            clean.RESULTS / (clean.HYPOTHESIS + "_registry_readback.json")]
        common += [location / name for location in (clean.FIT, clean.diagnostic.DEST)
                   for name in ("result.json", "case_measurements.json", "worker_terminal_receipt.json")]
    p = study.data.freeze(destination, stage, list(inputs) + common + files,
        {**config, "hypothesis_id": HYPOTHESIS, "TRAIN_cases": 32, "conditions": CONDITIONS,
         "models": MODELS, "histories": ARMS, "new_generation_or_training": False,
         "CPU_only": True, "rubric": profile.RUBRIC, "judges": q.JUDGES,
         "workers": 4, "max_attempts": 3, "semantic_rerolls": False,
         "task_accuracy": None, "heldout_accuracy_measured": False,
         "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (destination / "producer_source" / f"train_semantic_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(destination / "producer.json", p)
    return p


def prepare():
    selected, observations = inputs_and_observations()
    cache = require_prior()
    qualifications = []
    for role in ("secondary", "primary"):
        location = ROOT / ("20261023_budget128_judge_" + role + "_v1")
        clean.recovery.study.data.terminal(location)
        result = q.read(location / "result.json")
        if result["status"] != "passed" or result["correct"] != 50 or result["judge"] != q.JUDGES[role]:
            raise ValueError("both exact qualified budget128 judges required")
        qualifications += [location / name for name in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    p = producing(PREP, "prepare_saved_TRAIN_summary_semantic_diagnostic", qualifications + [PUBLIC_FILE],
        {"observations": 128, "prior_first_judgments": len(cache), "public_source_sha256": PUBLIC_SHA256})
    write_json(PREP / "observations.json", observations)
    write_json(PREP / "packets.json", {name: packets_for(rows) for name, rows in observations.items()})
    probes = clean.training_probes(selected)
    write_json(PREP / "pilot_case_ids.json", [row["id"] for row in probes])
    write_json(PREP / "judgment_cache.json", cache)
    clean.recovery.study.data.verify(p)
    checkpoints = {MODELS[0]: q.read(clean.diagnostic.DEST / "result.json")["model_checkpoint_sha256"],
        MODELS[1]: q.read(clean.FIT / "result.json")["checkpoint_sha256"]}
    write_json(PREP / "result.json", {"status": "saved_TRAIN_semantic_packets_prepared", "observations": 128,
        "model_checkpoints": checkpoints,
        "prior_first_judgments": len(cache), "hashes": {name: file_hash(PREP / name) for name in
            ("observations.json", "packets.json", "pilot_case_ids.json", "judgment_cache.json")},
        "task_accuracy": None, "final_test_opened": False, "producer_fingerprint": p["fingerprint"]})


def prepared():
    clean.recovery.study.data.terminal(PREP)
    for name, expected in q.read(PREP / "result.json")["hashes"].items():
        if file_hash(PREP / name) != expected:
            raise ValueError("prepared TRAIN observations, packets or first cache changed")
    observations = q.read(PREP / "observations.json")
    packets = q.read(PREP / "packets.json")
    if tuple(observations) != CONDITIONS or packets != {key: packets_for(rows) for key, rows in observations.items()}:
        raise ValueError("exact anonymous packets for all four conditions required")
    return observations, packets


def request_packets(packets, role, destination, cache):
    missing = {first.packet_key(packet, role): packet for packet in packets
               if first.packet_key(packet, role) not in cache}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {}
        for key, packet in missing.items():
            request_root = destination / "requests" / role / key
            (request_root / "attempts").mkdir(parents=True, exist_ok=False)
            jobs[pool.submit(q.request, packet, role, request_root)] = key, packet, request_root
        for future in as_completed(jobs):
            key, packet, request_root = jobs[future]
            row = future.result()
            first.audit_row(packet, role, row)
            write_json(request_root / "row.json", row)
            first.remember(cache, key, {"packet": packet, "role": role, "row": row, "origin": str(request_root)})
            write_json(destination / "judgment_cache.json", cache)
    return len(missing)


def scores_for(rows, packets, role, cache, encoding):
    if len(rows) != len(packets) or packets != packets_for(rows):
        raise ValueError("complete anonymous observation/packet pairing required")
    result = []
    for source, packet in zip(rows, packets):
        key = first.packet_key(packet, role)
        entry = cache[key]
        first.audit_row(packet, role, entry["row"])
        row = entry["row"]
        tokens = len(encoding.encode(source["text"], disallowed_special=()))
        valid = row["status"] == "valid"
        result.append({"id": source["id"], "source_id": source["source_id"], "diagnostic_group": source["diagnostic_group"],
            "text": source["text"], "tokens": tokens, "valid": valid,
            "passed": q.passes(row["evaluations"][0], source["text"], tokens) if valid else None,
            "evaluations": row.get("evaluations"), "packet_key": key, "first_origin": entry["origin"]})
    return result


def metric(scores):
    if not scores or len({row["id"] for row in scores}) != len(scores):
        raise ValueError("nonempty complete unique case scores required")
    valid = sum(row["valid"] for row in scores)
    passed = sum(row["passed"] is True for row in scores)
    count = len(scores)
    return {"cases": count, "valid": valid, "invalid": count - valid, "passed": passed,
        "semantic_pass_rate": passed / count if valid == count else None,
        "all_case_pass_lower_bound": passed / count, "all_case_pass_upper_bound": (passed + count - valid) / count,
        "wilson_95": q.wilson(passed, count) if valid == count else None,
        "components": {key: sum(row["valid"] and row["evaluations"][0][key] for row in scores)
                       for key in ("faithful", "salient", "readable")},
        "stored_overflow": sum(row["tokens"] > 128 for row in scores),
        "empty": sum(not row["text"].strip() for row in scores),
        "groups": {group: {"cases": sum(row["diagnostic_group"] == group for row in scores),
            "valid": sum(row["diagnostic_group"] == group and row["valid"] for row in scores),
            "passed": sum(row["diagnostic_group"] == group and row["passed"] is True for row in scores)}
            for group in ("original_16", "additional_16")}, "task_accuracy": None}


def joint_count(secondary, primary):
    if [row["id"] for row in secondary] != [row["id"] for row in primary]:
        raise ValueError("same ordered source cases required for joint judgments")
    return {"cases": len(secondary), "both_valid": sum(a["valid"] and b["valid"] for a, b in zip(secondary, primary)),
        "both_passed": sum(a["passed"] is True and b["passed"] is True for a, b in zip(secondary, primary))}


def judge(pilot=False):
    configure()
    observations, packets = prepared()
    if not admitted():
        raise ValueError("unchanged registered clean-word model required")
    source = PREP if pilot else PILOT
    clean.recovery.study.data.terminal(source)
    if not pilot and q.read(PILOT / "result.json")["status"] != "systems_TRAIN_semantic_judge_path_passed":
        raise ValueError("actual successful judge lifecycle pilot required")
    destination = PILOT if pilot else GRADE
    inputs = [PREP / name for name in ("producer.json", "result.json", "observations.json", "packets.json", "judgment_cache.json")]
    if not pilot:
        inputs += [PILOT / name for name in ("producer.json", "result.json", "judgment_cache.json", "worker_terminal_receipt.json")]
    cache = q.read(source / "judgment_cache.json")
    prior = dict(cache)
    p = producing(destination, "TRAIN_semantic_judge_lifecycle_pilot" if pilot else "saved_TRAIN_semantic_comparison",
        inputs, {"systems_only": pilot, "full_observations": 128, "prior_first_judgments": len(prior)})
    write_json(destination / "judgment_cache.json", cache)
    if pilot:
        selected = set(q.read(PREP / "pilot_case_ids.json"))
        probes = [packet for values in packets.values() for packet in values if packet["case_id"] in selected]
        new = request_packets(probes, "secondary", destination, cache)
        valid = sum(cache[first.packet_key(packet, "secondary")]["row"]["status"] == "valid" for packet in probes)
        result = {"status": "systems_TRAIN_semantic_judge_path_passed" if valid == len(probes) else "invalid_pilot_judgments",
            "systems_only": True, "observations": len(probes), "valid": valid, "new_requests": new,
            "prior_first_judgments_preserved": len(prior), "task_accuracy": None,
            "producer_fingerprint": p["fingerprint"]}
    else:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        results = {}
        for condition in CONDITIONS:
            rows, values = observations[condition], packets[condition]
            new = request_packets(values, "secondary", destination, cache)
            secondary = scores_for(rows, values, "secondary", cache, encoding)
            write_json(destination / (condition + "_secondary_scores.json"), secondary)
            summary = {"secondary": metric(secondary), "secondary_new_requests": new, "primary": None, "joint": None}
            if qualifies(summary["secondary"]):
                summary["primary_new_requests"] = request_packets(values, "primary", destination, cache)
                primary = scores_for(rows, values, "primary", cache, encoding)
                write_json(destination / (condition + "_primary_scores.json"), primary)
                summary["primary"] = metric(primary)
                summary["joint"] = joint_count(secondary, primary)
            results[condition] = summary
            write_json(destination / "progress.json", results)
            print(json.dumps({"condition": condition, **summary}), flush=True)
        result = {"status": "TRAIN_semantic_diagnostic_completed", "results": results,
            "prior_and_pilot_first_judgments_preserved": len(prior), "unique_first_judgments": len(cache),
            "training_semantic_quality_measured": True, "heldout_accuracy_measured": False,
            "task_accuracy": None, "NCM_success_established": False, "final_test_opened": False,
            "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    if any(cache[key] != value for key, value in prior.items()):
        raise ValueError("prior or pilot first judgments changed")
    clean.recovery.study.data.verify(p)
    write_json(destination / "result.json", result)
    if pilot and result["status"] != "systems_TRAIN_semantic_judge_path_passed":
        raise RuntimeError("pilot invalidity preserved; full semantic requests not admitted")


def run_stage(name, command, destination, journal, producer, existing=False):
    if destination.exists() != existing:
        raise ValueError("fresh stage destination required: " + name)
    clean.recovery.study.data.verify(producer)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    child = subprocess.Popen(command, env=env)
    journal.update(status="running", active_stage=name, child_pid=child.pid)
    write_json(WORKFLOW / "progress.json", journal)
    code = child.wait()
    receipt = {"status": "observed_worker_terminal", "stage": name, "exit_code": code,
        "child_pid": child.pid, "observer_pid": os.getpid(), "observer": "subprocess.wait",
        "command": command, "destination": str(destination)}
    write_json(WORKFLOW / "stage_receipts" / (name + ".json"), receipt)
    if destination.exists():
        path = destination / ("registry_terminal_receipt.json" if name == "register" else "worker_terminal_receipt.json")
        if path.exists():
            raise ValueError("terminal receipt overwrite prohibited")
        write_json(path, receipt)
    journal["stages"].append(receipt)
    write_json(WORKFLOW / "progress.json", journal)
    if code != 0:
        raise RuntimeError("stage failed with observed exit: " + name)
    if name not in ("finalize", "register"):
        clean.recovery.study.data.terminal(destination)
    clean.recovery.study.data.verify(producer)


def workflow():
    configure()
    if any(path.exists() for path in (WORKFLOW, PREP, PILOT, GRADE, RESULTS, REPORT,
            Path("data/hypothesis_registry") / (HYPOTHESIS + ".json"))):
        raise ValueError("fresh semantic diagnostic workflow and all destinations required")
    identity = None
    if not admitted():
        identity = kernel_identity(q.read(clean.WORKFLOW / "progress.json")["observer_pid"])
        if identity is None:
            raise ValueError("unfinished upstream must be authoritatively live")
    p = producing(WORKFLOW, "observe_registered_word_component_then_grade_TRAIN_outputs", [],
        {"upstream_kernel_identity": identity, "poll_seconds": 5, "upstream_worker_restarted": False}, pending=True)
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "waiting_for_registered_clean_word_component", "observer_pid": os.getpid(),
        "upstream_kernel_identity": identity, "stages": [], "final_test_opened": False,
        "producer_fingerprint": p["fingerprint"]}
    write_json(WORKFLOW / "progress.json", journal)
    try:
        while not admitted():
            if (clean.WORKFLOW / "failure.json").exists():
                raise RuntimeError("upstream failure preserved")
            if identity is None or kernel_identity(identity["pid"]) != identity:
                raise RuntimeError("upstream terminal without registered component evidence")
            time.sleep(5)
        for stage, destination in (("prepare", PREP), ("pilot", PILOT), ("grade", GRADE)):
            run_stage(stage, [sys.executable, "-u", "-m", MODULE, stage], destination, journal, p)
        for stage in ("finalize", "register"):
            run_stage(stage, [sys.executable, "-u", "-m", AUDITOR, stage], RESULTS, journal, p, existing=stage == "register")
        path = RESULTS / "campaign_results.json"
        receipt = q.read(RESULTS / (HYPOTHESIS + "_registry_readback.json"))
        if receipt["status"] != "registered_and_read_back" or receipt["record"]["campaign_sha256"] != file_hash(path):
            raise ValueError("exact conservative semantic diagnostic registry readback required")
        journal.update(status="TRAIN_semantic_diagnostic_completed", active_stage=None,
            task_accuracy=None, heldout_accuracy_measured=False, NCM_success_established=False,
            campaign_sha256=file_hash(path))
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "prepare", "pilot", "grade", "workflow"))
    args = parser.parse_args()
    configure()
    if args.stage == "check":
        cache = require_prior()
        print(json.dumps({"prior_first_judgments": len(cache), "upstream_admitted": admitted(),
            "conditions": CONDITIONS, "semantic_accuracy": None}))
    elif args.stage == "workflow":
        workflow()
    elif args.stage == "prepare":
        prepare()
    else:
        judge(args.stage == "pilot")
