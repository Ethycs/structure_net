"""Complete source-only DEV for the entire trained native655 NCM checkpoint."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_lm1b_native_source_training as training
from experiments.structure_net import ncm_lm1b_framed_codec as framed
from experiments.structure_net import ncm_oracle_codec_semantic as oracle
from experiments.structure_net import ncm_dense_semantic_development as semantic
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, head, q, ROOT = training.native, training.lifecycle.head, training.q, training.ROOT
reference = semantic.reference
HYPOTHESIS = "ncm-lm1b-native655-source-only-DEV-v1"
MODULE = "experiments.structure_net.ncm_lm1b_native_development"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_lm1b_native_development"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-native-development-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-lm1b-native-development.md")
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_lm1b_native_development.py")
TEST = Path("tests/structure_net/test_ncm_lm1b_native_development.py")
WORKFLOW = ROOT / "20261024_lm1b_native655_development_workflow_v1"
METADATA = ROOT / "20261024_lm1b_native655_development_metadata_v1"
PILOT = ROOT / "20261024_lm1b_native655_development_inference_pilot_v1"
OUTPUTS = ROOT / "20261024_lm1b_native655_development_outputs_v1"
PACKETS = ROOT / "20261024_lm1b_native655_development_packets_v1"
JUDGE_PILOT = ROOT / "20261024_lm1b_native655_development_judge_pilot_v1"
GRADE = ROOT / "20261024_lm1b_native655_development_grade_v1"
RESULTS = ROOT / "20261024_lm1b_native655_development_results_v1"
SOURCE = ROOT / "20261024_pinned_date_development_sources_v1"
FEATURES = ROOT / "20261024_pinned_date_development_features_v1"
SOURCE_SHA = "1395a35a259ec02073c41f77858a682596e44b66c13ccb505a8cd9da18b2130f"
FEATURE_SHA = "637be63f6917c9d0f8027d2b5dd39f735df41e1ed549a0241658b8df31c4e39b"
OLD_CACHE = ROOT / "20261024_evidence_teacher64_qualification_grade_v1/judgment_cache.json"
CACHE_ARCHIVE = ROOT / "20261024_evidence_teacher64_qualification_results_v1"
CACHE_HYPOTHESIS = "ncm-evidence64-teacher-semantic-qualification-v1"
CACHE_CAMPAIGN = "a09911fa275b89100a7c645f25d6cf6b26d6274ab0d7624f27db4ecd7e125b8d"
CACHE_SHA = "efed1413ddb85d44142af207d393744bc8e65a90e454a206840e1eccbf709829"
CACHE_COUNT = 3094
TRAIN_AUDIT = ROOT / "20261024_lm1b_native655_training_audit_v1"
COHORTS = ("short_development", "long_diagnostic")
FORBIDDEN = frozenset(("target_text", "word_ids", "native_word_ids", "concept_ids", "target_ids", "codes"))


def configure():
    native.vendor.corpus.quality.dependencies()
    reference.profile.install()


def validate_sources(cases):
    if tuple(cases) != COHORTS or any(len(cases[k]) != 64 for k in COHORTS):
        raise ValueError("complete ordered64short+64long source-only populations required")
    rows = [r for cohort in COHORTS for r in cases[cohort]]
    if any(FORBIDDEN & r.keys() for r in rows):
        raise ValueError("targets or concept codes in source-only inputs")
    for key in ("id", "source_id", "group_id", "source_sha256", "source_component"):
        if len({r[key] for r in rows}) != 128:
            raise ValueError("duplicate DEV source identity or family")
    if any(r["split"] != cohort for cohort in COHORTS for r in cases[cohort]):
        raise ValueError("source cohort identity changed")
    return cases


def source_inputs():
    native.vendor.terminal(SOURCE)
    native.vendor.terminal(FEATURES)
    if file_hash(SOURCE / "cases.json") != SOURCE_SHA or file_hash(FEATURES / "manifest.json") != FEATURE_SHA:
        raise ValueError("unchanged date23 full-source DEV metadata and features required")
    cases = validate_sources(q.read(SOURCE / "cases.json"))
    manifest = q.read(FEATURES / "manifest.json")
    state = q.read(FEATURES / "result.json")
    if (state["manifest_sha256"] != FEATURE_SHA or state["parent_body_hash_unchanged"] != reference.data.EXPECTED_BODY
            or not state["all_source_tokens_retained"] or state["cases"] != 128
            or set(manifest) != {r["id"] for rows in cases.values() for r in rows}):
        raise ValueError("complete original-body FP32 features required")
    return cases, manifest


def producing(root, stage, inputs=(), config=None):
    files = [Path(__file__), PLAN, AUDITOR_FILE, TEST, Path(training.__file__),
        Path(head.__file__), Path(head.concepts.__file__), Path(training.lifecycle.objective.__file__),
        Path(framed.__file__), Path(native.__file__), Path(oracle.__file__), Path(semantic.__file__),
        Path(reference.__file__), Path(reference.first.__file__), reference.first.AMENDMENT,
        Path(reference.profile.__file__), reference.profile.PLAN, *sorted(native.vendor.VENDOR.rglob("*.py"))]
    common = [SOURCE / "cases.json", FEATURES / "manifest.json", OLD_CACHE,
        CACHE_ARCHIVE / "campaign_results.json", CACHE_ARCHIVE / (CACHE_HYPOTHESIS + "_registry_readback.json"),
        CACHE_ARCHIVE / "backend_exact_readback.json",
        ROOT / "20261024_evidence_teacher64_qualification_workflow_v1/tool_terminal_receipt.json",
        TRAIN_AUDIT / "result.json", TRAIN_AUDIT / "actual_tool_terminal_receipt.json", training.PLAN,
        training.AUDITOR, Path(training.lifecycle.__file__), Path(training.lifecycle.mechanism.__file__)]
    cfg = {**(config or {}), "hypothesis_id": HYPOTHESIS, "schema": "nal.ncm-lm1b-native655-source-only-dev.v1",
        "conformance": "EXPLORATORY complete source-disjoint DEV; final test unopened",
        "source_only": True, "source_prediction": True, "oracle_target_codes": False,
        "target_fields_in_prediction": False, "all128_cases": True, "date_string": "23 Oct 2026",
        "training_cases": 655, "native_trainable_parameters_during_training": 127732736,
        "native_generator_and_word_decoder_trained": True, "native_encoder_and_VQ_frozen": True,
        "original_Llama_body_and_MLPs_frozen": True,
        "all_source_tokens_retained": True, "checkpoint_selection": "fixed final1024 native655 only",
        "precision": "FP32", "TF32": False, "native_levels": list(native.LEVELS),
        "raw_words": 128, "word_vocabulary": 30528, "full_concepts": 641,
        "output_policy": "native CLS renderer; full-vocabulary argmax; no repair or masking",
        "task_accuracy": None, "final_test_opened": False, "speed_tested": False,
        "rubric": q.RUBRIC, "judges": q.JUDGES, "canonical_prior_firsts": CACHE_COUNT}
    p = native.vendor.freeze(root, stage, list(dict.fromkeys([*inputs, *common, *files])), cfg)
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"grounded_dev_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def prepare():
    configure()
    cases, _ = source_inputs()
    native.vendor.terminal(training.DATA)
    proof = training.split_guard(q.read(training.DATA / "cases.json"), cases)
    public = reference.verify_public([r for rows in cases.values() for r in rows])
    p = producing(METADATA, "complete128_source_only_metadata", [training.DATA / "cases.json", reference.PUBLIC_FILE])
    write_json(METADATA / "cases.json", cases)
    write_json(METADATA / "split_proof.json", proof)
    write_json(METADATA / "public_proof.json", {"public_file_sha256": reference.PUBLIC_SHA256, "exact_sources": public})
    native.vendor.verify(p)
    write_json(METADATA / "result.json", {"status": "complete128_source_only_metadata_prepared",
        "cases_sha256": file_hash(METADATA / "cases.json"), "cases": 128, "task_accuracy": None,
        "final_test_opened": False, "producer_fingerprint": p["fingerprint"]})


def checkpoint_metadata(state, saved):
    if (state["status"] != "native655_training_completed_source_only_DEV_pending"
            or state["updates"] != 1024 or state["train_cases"] != 655
            or state["trainable_parameters"] != 127732736
            or not state["strict_reload_passed"] or not state["root_queries_updated"]
            or state["native16_or_pilot_weights_reused"]
            or state["initial_trainable_sha256"] != training.capacity.INITIAL
            or state["final_trainable_sha256"] == training.capacity.INITIAL
            or state["protected_codec_hashes"] != {"encoder": training.lifecycle.ENCODER, "VQ": training.lifecycle.VQ}
            or state["peak_allocated_bytes"] > 8_000_000_000 or state["source_disjoint_task_accuracy"] is not None
            or state["speed_tested"] or not state["all_final_sources_closed"]
            or saved["step"] != 1024 or saved["config"] != asdict(training.Config())
            or saved["producer_fingerprint"] != state["producer_fingerprint"]):
        raise ValueError("actual fixed1024 full-native655 checkpoint/reset/protected-geometry contract differs")


def checkpoint_contract():
    native.vendor.terminal(training.TRAIN)
    native.vendor.verify(q.read(training.WORKFLOW / "producer.json"))
    whole = q.read(training.WORKFLOW / "result.json")
    state = q.read(training.TRAIN / "result.json")
    if (whole["status"] != "native655_training_workflow_completed_source_only_DEV_pending"
            or q.read(training.WORKFLOW / "tool_terminal_receipt.json")["exit_code"] != 0
            or [r["stage"] for r in whole["stages"]] != ["prepare", "pilot", "train"]
            or any(r["exit_code"] != 0 for r in whole["stages"])
            or file_hash(training.TRAIN / "model.pt") != state["checkpoint_sha256"]
            or q.read(TRAIN_AUDIT / "actual_tool_terminal_receipt.json")["exit_code"] != 0):
        raise ValueError("actual completed whole native655 training and independent CPU audit required")
    audit = q.read(TRAIN_AUDIT / "result.json")
    if (audit["status"] != "native655_actual_training_trace_and_state_replayed_DEV_pending"
            or audit["results"]["full"]["updates"] != 1024
            or audit["results"]["full"]["checkpoint_sha256"] != state["checkpoint_sha256"]):
        raise ValueError("complete native655 trace/state audit differs")
    for path, expected in audit["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("audited native655 evidence changed:" + path)
    saved = torch.load(training.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    checkpoint_metadata(state, saved)
    return state, saved


def model_contract(model, state):
    if (training.lifecycle.trainable_hash(model) != state["final_trainable_sha256"]
            or sum(p.numel() for p in model.parameters() if p.requires_grad) != 127732736
            or training.lifecycle.protected(model) != {"encoder": training.lifecycle.ENCODER, "VQ": training.lifecycle.VQ}
            or native.vendor.module_hash(model.concepts.generator) != state["final_native_hashes"]["generator"]
            or native.vendor.module_hash(model.codec.decoder) != state["final_native_hashes"]["word_decoder"]):
        raise ValueError("strict trained native generator/decoder/attention or protected geometry differs")


def load_selected():
    state, saved = checkpoint_contract()
    model = training.capacity.fresh()
    model.load_state_dict(saved["model"], strict=True)
    model_contract(model, state)
    return model.eval().requires_grad_(False), state


def decode(words, tokenizer):
    if len(words) != 128 or any(type(v) is not int or not 0 <= v < 30528 for v in words):
        raise ValueError("all128 raw full-vocabulary positions required")
    text, ends, bos = framed.render([words], tokenizer)
    start = 1 if bos[0] else 0
    return {"text": text, "predicted_BOS": bos[0], "predicted_EOS_position": ends[0],
        "missing_EOS": ends[0] == 128, "active_unmapped_ids": sum(v >= 30522 for v in words[start:ends[0]])}


def predict(model, row, manifest, tokenizer):
    if FORBIDDEN & row.keys():
        raise ValueError("prediction must receive source metadata only")
    entry = manifest[row["id"]]
    if file_hash(entry["path"]) != entry["sha256"]:
        raise ValueError("source feature bytes changed")
    value = torch.load(entry["path"], map_location="cpu", weights_only=True)
    if (any(value[k] != row[k] for k in ("id", "source_sha256", "alignment_sha256", "native_prompt_sha256"))
            or value["features"].shape != (entry["source_tokens"], 3072)
            or value["features"].dtype != torch.float32 or not torch.isfinite(value["features"]).all()):
        raise ValueError("full source feature identity, shape or precision changed")
    with torch.no_grad():
        features = value["features"].unsqueeze(0).cuda()
        mask = torch.ones(features.shape[:2], dtype=torch.bool, device="cuda")
        generated = model.generate(features, mask)
        if (set(generated) != {"codes", "raw_word_ids"} or generated["codes"].shape != (1, 641)
                or generated["raw_word_ids"].shape != (1, 128)):
            raise ValueError("source-only complete native generation contract differs")
        codes = generated["codes"].cpu().tolist()[0]
        words = generated["raw_word_ids"].cpu().tolist()[0]
    if any(type(v) is not int or not 0 <= v < 8192 for v in codes):
        raise ValueError("complete native concept vocabulary required")
    if model._word_bound is not None or model.concepts._bound is not None:
        raise ValueError("source context leaked across requests")
    return {"id": row["id"], "source_sha256": row["source_sha256"], "split": row["split"],
        "feature_sha256": entry["sha256"], "source_positions": entry["source_tokens"],
        "codes": codes, "raw_word_ids": words, **decode(words, tokenizer)}


def inference_probes(cases, manifest):
    rows = [r for cohort in COHORTS for r in cases[cohort]]
    return [cases[COHORTS[0]][0], min(rows, key=lambda r: (-manifest[r["id"]]["source_tokens"], r["id"]))]


def generate(pilot=False):
    configure()
    training.training.require_gpu()
    native.vendor.terminal(METADATA)
    cases, manifest = source_inputs()
    if q.read(METADATA / "cases.json") != cases:
        raise ValueError("source-only metadata changed")
    _, selected = checkpoint_contract()
    del selected
    chosen = inference_probes(cases, manifest) if pilot else [r for cohort in COHORTS for r in cases[cohort]]
    root = PILOT if pilot else OUTPUTS
    extra = []
    if not pilot:
        native.vendor.terminal(PILOT)
        extra = [PILOT / name for name in ("result.json", "outputs.json", "worker_terminal_receipt.json")]
    p = producing(root, "source_only_inference_pilot" if pilot else "complete128_source_only_predictions",
        [training.TRAIN / n for n in ("producer.json", "model.pt", "result.json", "worker_terminal_receipt.json")]
        + [training.WORKFLOW / n for n in ("result.json", "tool_terminal_receipt.json")]
        + [METADATA / "cases.json", *[Path(manifest[r["id"]]["path"]) for r in chosen], *extra], {"pilot": pilot})
    model, state = load_selected()
    before = native.vendor.module_hash(model)
    tokenizer, outputs = native.tokenizer(), []
    prior = {r["id"]: r for r in q.read(PILOT / "outputs.json")} if not pilot else {}
    torch.cuda.reset_peak_memory_stats()
    try:
        for row in chosen:
            output = predict(model, row, manifest, tokenizer)
            if pilot and predict(model, row, manifest, tokenizer) != output:
                raise ValueError("source-only generation is not deterministic")
            if row["id"] in prior and output != prior[row["id"]]:
                raise ValueError("fresh full inference differs from retained pilot")
            outputs.append(output)
            write_json(root / "outputs.json", outputs)
            if pilot or len(outputs) % 16 == 0:
                print(json.dumps({"generated": len(outputs), "total": len(chosen)}), flush=True)
        if native.vendor.module_hash(model) != before or any(v.grad is not None for v in model.parameters()):
            raise ValueError("frozen complete model changed during inference")
    finally:
        model.close()
    native.vendor.verify(p)
    write_json(root / "result.json", {"status": "source_only_inference_pilot_passed" if pilot else "all128_source_only_outputs_generated",
        "cases": len(outputs), "checkpoint_sha256": state["checkpoint_sha256"], "model_sha256": before,
        "model_unchanged": True, "outputs_sha256": file_hash(root / "outputs.json"),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "task_accuracy": None,
        "final_test_opened": False, "producer_fingerprint": p["fingerprint"]})


def packet_inputs():
    cases, _ = source_inputs()
    outputs = q.read(OUTPUTS / "outputs.json")
    rows = [r for cohort in COHORTS for r in cases[cohort]]
    if [r["id"] for r in outputs] != [r["id"] for r in rows]:
        raise ValueError("every ordered source must have exactly one output")
    combined = [{**row, "texts": {"student": output["text"]}} for row, output in zip(rows, outputs)]
    return {c: [r for r in combined if r["split"] == c] for c in COHORTS}


def packets():
    configure()
    native.vendor.terminal(OUTPUTS)
    cases = packet_inputs()
    payload = {c: {"student": [q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["texts"]["student"]}])
        for r in rows]} for c, rows in cases.items()}
    proof = reference.verify_public([r for rows in cases.values() for r in rows])
    p = producing(PACKETS, "all128_exact_source_semantic_packets", [OUTPUTS / "outputs.json", reference.PUBLIC_FILE])
    write_json(PACKETS / "cases.json", cases)
    write_json(PACKETS / "exact_packets.json", payload)
    write_json(PACKETS / "public_proof.json", {"public_file_sha256": reference.PUBLIC_SHA256, "exact_sources": proof})
    native.vendor.verify(p)
    write_json(PACKETS / "result.json", {"status": "complete128_public_packets_prepared", "cases": 128,
        "exact_packets_sha256": file_hash(PACKETS / "exact_packets.json"), "producer_fingerprint": p["fingerprint"]})


def base_cache():
    if CACHE_COUNT is None or CACHE_SHA is None or CACHE_CAMPAIGN is None:
        raise ValueError("pin completed latest canonical lineage before freezing this experiment")
    campaign, receipt = q.read(CACHE_ARCHIVE / "campaign_results.json"), q.read(CACHE_ARCHIVE / (CACHE_HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(CACHE_ARCHIVE / "campaign_results.json") != CACHE_CAMPAIGN or file_hash(OLD_CACHE) != CACHE_SHA
            or campaign["artifacts"].get(str(OLD_CACHE)) != CACHE_SHA or campaign["canonical_first_judgments"] != CACHE_COUNT
            or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != CACHE_CAMPAIGN or receipt["exact"]["campaign_sha256"] != CACHE_CAMPAIGN
            or q.read(ROOT / "20261024_evidence_teacher64_qualification_workflow_v1/tool_terminal_receipt.json")["exit_code"] != 0
            or q.read(CACHE_ARCHIVE / "backend_exact_readback.json")["status"] != "independent_backend_exact_readback_passed"):
        raise ValueError("latest completed registered canonical firsts and backend readback required")
    cache = q.read(OLD_CACHE)
    if len(cache) != CACHE_COUNT:
        raise ValueError("all inherited canonical first decisions required")
    semantic.audit_cache(cache)
    return cache


def task_metric(scores):
    result = oracle.metric(scores)
    result.pop("oracle_semantic_pass_rate")
    result.update(task_accuracy=result["passed"] / result["cases"] if not result["invalid"] else None,
        source_prediction=True, oracle_target_codes=False)
    return result


def secondary_admitted(metrics):
    return set(metrics) == set(COHORTS) and all(oracle.qualifies(metrics[c]) for c in COHORTS)


def joint_gate(secondary, primary):
    if (len(secondary) != 64 or len(primary) != 64 or len({r["id"] for r in secondary}) != 64
            or {r["id"] for r in secondary} != {r["id"] for r in primary}):
        raise ValueError("same complete64 cases under both judges required")
    other = {r["id"]: r for r in primary}
    count = sum(r["passed"] is True and other[r["id"]]["passed"] is True for r in secondary)
    valid = all(r["valid"] and other[r["id"]]["valid"] for r in secondary)
    return {"cases": 64, "joint_passed": count, "all_valid": valid, "qualified": valid and count >= 61}


def grade(pilot=False):
    configure()
    semantic.require_judges()
    native.vendor.terminal(PACKETS)
    cases, payload = q.read(PACKETS / "cases.json"), q.read(PACKETS / "exact_packets.json")
    root = JUDGE_PILOT if pilot else GRADE
    cache = base_cache()
    extra = []
    if not pilot:
        native.vendor.terminal(JUDGE_PILOT)
        old = cache
        cache = q.read(JUDGE_PILOT / "judgment_cache.json")
        if any(cache.get(k) != v for k, v in old.items()) or q.read(JUDGE_PILOT / "result.json")["status"] != "systems_four_first_judgments_valid":
            raise ValueError("all inherited firsts and valid four-case pilot required")
        extra = [JUDGE_PILOT / n for n in ("result.json", "judgment_cache.json", "worker_terminal_receipt.json")]
    inherited = dict(cache)
    p = producing(root, "four_case_judge_validity_only" if pilot else "complete128_source_only_semantic_quality",
        [PACKETS / n for n in ("cases.json", "exact_packets.json", "public_proof.json", "worker_terminal_receipt.json")] + extra,
        {"pilot": pilot, "pilot_quality_gate": False})
    metrics = {c: {} for c in COHORTS}

    def judge(cohort, role):
        rows = cases[cohort][:2] if pilot else cases[cohort]
        target = root / cohort
        target.mkdir(exist_ok=True)
        original = oracle.grade_arm(target, "student", role, rows, payload[cohort], cache)
        scores = q.read(target / f"{role}_student_case_scores.json")
        value = {**{k: v for k, v in original.items() if k not in oracle.metric(scores)}, **task_metric(scores)}
        write_json(target / f"{role}_student_result.json", value)
        write_json(root / "judgment_cache.json", cache)
        metrics[cohort][role] = value

    for cohort in COHORTS:
        judge(cohort, "secondary")
    admitted = not pilot and secondary_admitted({c: metrics[c]["secondary"] for c in COHORTS})
    joints = None
    if admitted:
        for cohort in COHORTS:
            judge(cohort, "primary")
        joints = {c: joint_gate(q.read(root / c / "secondary_student_case_scores.json"),
            q.read(root / c / "primary_student_case_scores.json")) for c in COHORTS}
    if any(cache.get(k) != v for k, v in inherited.items()):
        raise ValueError("a prior first judgment changed")
    semantic.audit_cache(cache)
    valid_pilot = all(metrics[c]["secondary"]["valid"] == 2 for c in COHORTS)
    qualified = admitted and all(joints[c]["qualified"] for c in COHORTS)
    status = ("systems_four_first_judgments_valid" if valid_pilot else "systems_judge_pilot_invalid") if pilot else (
        "full128_source_only_DEV_qualified" if qualified else "full128_source_only_DEV_quality_failed")
    native.vendor.verify(p)
    write_json(root / "result.json", {"status": status, "metrics": metrics, "primary_admitted": admitted,
        "joint": joints, "DEV_qualified": qualified, "canonical_first_judgments": len(cache),
        "inherited_first_judgments": len(inherited), "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False, "producer_fingerprint": p["fingerprint"]})
    if pilot and not valid_pilot:
        raise ValueError("four-case semantic validity pilot failed; all firsts retained")


def workflow():
    configure()
    checkpoint_contract()
    base_cache()
    if any(p.exists() for p in (WORKFLOW, PILOT, OUTPUTS, PACKETS, JUDGE_PILOT, GRADE, RESULTS)):
        raise ValueError("fresh source-only DEV workflow required")
    p = producing(WORKFLOW, "complete128_source_only_DEV_workflow", [training.TRAIN / "result.json"])
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    stages = ([] if METADATA.exists() else [(MODULE, "prepare", METADATA, "")]) + [
        (MODULE, "inference-pilot", PILOT, "0"), (MODULE, "generate", OUTPUTS, "0"),
        (MODULE, "packets", PACKETS, ""), (MODULE, "judge-pilot", JUDGE_PILOT, ""),
        (MODULE, "grade", GRADE, ""), (AUDITOR, "finalize", RESULTS, ""), (AUDITOR, "register", None, "")]
    if METADATA.exists():
        native.vendor.terminal(METADATA)
    try:
        for module, stage, root, gpu in stages:
            native.vendor.verify(p)
            command = [sys.executable, "-u", "-m", module, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"stage": stage, "exit_code": code, "child_pid": child.pid, "observer_pid": os.getpid(),
                "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if root is not None and root.exists():
                write_json(root / "worker_terminal_receipt.json", receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("source-only DEV stage failed: " + stage)
        native.vendor.verify(p)
        journal.update(status="source_only_DEV_workflow_completed", active_stage=None, child_pid=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "inference-pilot", "generate", "packets", "judge-pilot", "grade", "workflow"))
    args = parser.parse_args()
    {"prepare": prepare, "inference-pilot": lambda: generate(True), "generate": generate,
        "packets": packets, "judge-pilot": lambda: grade(True), "grade": grade, "workflow": workflow}[args.stage]()
