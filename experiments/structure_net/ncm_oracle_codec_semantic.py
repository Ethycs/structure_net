"""Judge meaning in oracle codec reconstructions; never call it model accuracy."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from experiments.structure_net import ncm_isolated_quality_eval as teacher
from experiments.structure_net import ncm_sample_scaled_codec_ema as codec
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, freeze, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.ncm_frozen_body_public_data import PUBLIC_FILE, PUBLIC_SHA256

q = teacher.old.run.q
first = teacher.old.first
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-oracle-codec-semantic-plan.md")
PREP = ROOT / "20261023_oracle_codec_semantic_prepare_v1"
PILOT = ROOT / "20261023_oracle_codec_semantic_pilot_v1"
GRADE = ROOT / "20261023_oracle_codec_semantic_grade_v1"
CODEC_RESULTS = ROOT / "20261023_sample_scaled_codec_ema_results_v1"
TEACHER_RESULTS = ROOT / "20261023_isolated_quality_results_v1"
ARMS = ("uncompressed", "original", "trained")
HYPOTHESIS = "ncm-oracle-codec-semantic-quality-v1"


def select(rows):
    if len(rows) != 252 or len({r["id"] for r in rows}) != 252:
        raise ValueError("all252 unique targets required before selection")
    return sorted(rows, key=lambda r: text_hash("ncm-codec-semantic-v1:" + r["id"]))[:64]


def qualifies(metrics):
    return metrics["cases"] == 64 and metrics["valid"] == 64 and metrics["passed"] >= 61


def metric(scores):
    passed = sum(s["passed"] is True for s in scores)
    invalid = sum(not s["valid"] for s in scores)
    return {"cases": len(scores), "passed": passed, "valid": len(scores) - invalid, "invalid": invalid,
        "oracle_semantic_pass_rate": passed / len(scores) if not invalid else None,
        "all_case_pass_lower_bound": passed / len(scores),
        "all_case_pass_upper_bound": (passed + invalid) / len(scores),
        "wilson_95": q.wilson(passed, len(scores)) if not invalid else None,
        "components": {k: sum(s["valid"] and s["evaluations"][0][k] is True for s in scores)
                       for k in ("faithful", "salient", "readable")},
        "stored_overflow": sum(s["tokens"] > 128 for s in scores),
        "empty": sum(not s["text"].strip() for s in scores),
        "task_accuracy": None, "oracle_target_codes": True, "source_prediction": False}


def producing(root, stage, inputs, config):
    sources = [Path(__file__), PLAN, Path(first.__file__), first.AMENDMENT,
               Path(teacher.__file__), Path(teacher.old.profile.__file__), teacher.old.profile.PLAN]
    p = freeze(root, stage, list(inputs) + sources, {**config, "hypothesis_id": HYPOTHESIS,
        "rubric": teacher.old.profile.RUBRIC, "judges": q.JUDGES, "workers": 4, "max_attempts": 3,
        "oracle_target_codes": True, "source_prediction": False, "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, source in enumerate(sources):
        p["source_hashes"][str(source)] = file_hash(source)
        (root / "producer_source" / f"semantic_{i}_{source.name}").write_bytes(source.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def registered(root, hypothesis):
    campaign = root / "campaign_results.json"
    receipt = q.read(root / (hypothesis + "_registry_readback.json"))
    if receipt["status"] != "registered_and_read_back" or receipt["record"]["campaign_sha256"] != file_hash(campaign):
        raise ValueError("prior completed registry identity changed")
    for path, expected in q.read(campaign)["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("completed evidence changed: " + path)


def prepare():
    registered(CODEC_RESULTS, "ncm-sample-scaled-codec-ema-v1")
    registered(TEACHER_RESULTS, "ncm-isolated-native-quality-teacher-v1")
    sources = q.read(teacher.training.data.DEST / "cases.json")
    diagnostic = q.read(codec.core.corpus.DEST / "summary_diagnostic.json")
    selected = select(diagnostic)
    by_source = {r["id"]: r for r in sources}
    outputs = {arm: {r["id"]: r for r in q.read(codec.RECON / (arm + "_outputs.json"))}
               for arm in ("original", "trained")}
    cases = []
    for target in selected:
        source = by_source[target["id"]]
        if source["target_text"] != target["text"] or any(source[k] != target[k] for k in ("source_id", "group_id", "source_sha256")):
            raise ValueError("complete target/source binding differs")
        texts = {"uncompressed": target["text"]}
        for arm in ("original", "trained"):
            row = outputs[arm][target["id"]]
            if row["target"] != target["text"]:
                raise ValueError("reconstruction target changed")
            texts[arm] = row["text"]
        cases.append({**{k: source[k] for k in ("id", "source_id", "group_id", "source_sha256", "source", "spans")}, "texts": texts})
    proof = verify_public(cases)
    packets = {arm: [q.packet(c["id"], c["source"], [{"candidate_id": "summary", "text": c["texts"][arm]}])
                    for c in cases] for arm in ARMS}
    inputs = [teacher.training.data.DEST / "cases.json", codec.core.corpus.DEST / "summary_diagnostic.json",
              PUBLIC_FILE, teacher.GRADE / "judgment_cache.json"]
    inputs += [codec.RECON / (arm + "_outputs.json") for arm in ("original", "trained")]
    for root, hyp in ((CODEC_RESULTS, "ncm-sample-scaled-codec-ema-v1"), (TEACHER_RESULTS, "ncm-isolated-native-quality-teacher-v1")):
        inputs += [root / "campaign_results.json", root / (hyp + "_registry_readback.json")]
    p = producing(PREP, "fixed_hash_sample_and_exact_public_payload_proof", inputs,
                  {"cases": 64, "arms": ARMS, "selection": "SHA256(ncm-codec-semantic-v1:+id)", "prior_first_judgments": 320})
    write_json(PREP / "cases.json", cases)
    write_json(PREP / "exact_packets.json", packets)
    write_json(PREP / "public_payload_proof.json", {"public_dataset": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        "public_file_sha256": PUBLIC_SHA256, "exact_sources": proof,
        "case_ids": [c["id"] for c in cases], "exact_packets_sha256": file_hash(PREP / "exact_packets.json"),
        "source_fields_public_exact": True, "candidate_texts_locally_derived_from_public_sources": True,
        "private_data_in_packets": False, "gateway": "https://central-dev.zt:4000/v1/chat/completions",
        "judges": q.JUDGES, "rubric_sha256": text_hash(teacher.old.profile.RUBRIC)})
    verify(p)
    result = {"status": "prepared_public_semantic_audit", "cases": 64, "arm_cases": 192,
        "oracle_target_codes": True, "source_prediction": False, "task_accuracy": None,
        "producer_fingerprint": p["fingerprint"],
        "hashes": {name: file_hash(PREP / name) for name in ("cases.json", "exact_packets.json", "public_payload_proof.json")}}
    write_json(PREP / "result.json", result)
    print(json.dumps(result), flush=True)


def prior_cache():
    registered(TEACHER_RESULTS, "ncm-isolated-native-quality-teacher-v1")
    cache = q.read(teacher.GRADE / "judgment_cache.json")
    if len(cache) != 320:
        raise ValueError("all320 previous first judgments required")
    for key, entry in cache.items():
        if key != first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("prior cache identity differs")
        first.audit_row(entry["packet"], entry["role"], entry["row"])
    return cache


def check_prepared():
    teacher.training.data.terminal(PREP)
    r = q.read(PREP / "result.json")
    for name, expected in r["hashes"].items():
        if file_hash(PREP / name) != expected:
            raise ValueError("prepared public payload changed")
    cases, packets = q.read(PREP / "cases.json"), q.read(PREP / "exact_packets.json")
    verify_public(cases)
    for role in ("secondary", "primary"):
        root = ROOT / ("20261023_budget128_judge_" + role + "_v1")
        teacher.training.data.terminal(root)
        if q.read(root / "result.json")["status"] != "passed":
            raise ValueError("both qualified128-token judges required")
    for arm in ARMS:
        expected = [q.packet(c["id"], c["source"], [{"candidate_id": "summary", "text": c["texts"][arm]}]) for c in cases]
        if packets[arm] != expected:
            raise ValueError("packet differs from exact public payload proof")
    return cases, packets


def grade_arm(root, arm, role, cases, packets, cache):
    import tiktoken
    packet_list = packets[arm][:len(cases)]
    # One cache spans every arm. Identical reconstructions use one first judgment.
    missing = {first.packet_key(p, role): p for p in packet_list if first.packet_key(p, role) not in cache}
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {}
        for key, packet in missing.items():
            request_root = root / "requests" / role / key
            (request_root / "attempts").mkdir(parents=True, exist_ok=False)
            jobs[pool.submit(q.request, packet, role, request_root)] = (key, packet, request_root)
        for i, future in enumerate(as_completed(jobs), 1):
            key, packet, request_root = jobs[future]
            row = future.result()
            first.audit_row(packet, role, row)
            write_json(request_root / "row.json", row)
            first.remember(cache, key, {"packet": packet, "role": role, "row": row, "origin": str(request_root)})
            write_json(root / "judgment_cache.json", cache)
            if i % 8 == 0 or i == len(missing):
                print(json.dumps({"arm": arm, "role": role, "new_completed": i, "new_total": len(missing)}), flush=True)
    encoding = tiktoken.get_encoding("cl100k_base")
    scores = []
    for case, packet in zip(cases, packet_list):
        entry = cache[first.packet_key(packet, role)]
        first.audit_row(packet, role, entry["row"])
        text = case["texts"][arm]
        tokens = len(encoding.encode(text, disallowed_special=()))
        valid = entry["row"]["status"] == "valid"
        scores.append({"id": case["id"], "text": text, "tokens": tokens, "valid": valid,
            "passed": q.passes(entry["row"]["evaluations"][0], text, tokens) if valid else None,
            "evaluations": entry["row"].get("evaluations"), "packet_key": first.packet_key(packet, role),
            "attempts": entry["row"]["attempts"], "first_origin": entry["origin"]})
    result = {**metric(scores), "judge": q.JUDGES[role], "new_requests": len(missing),
              "reused_first_judgments": len(cases) - len(missing)}
    write_json(root / f"{role}_{arm}_case_scores.json", scores)
    write_json(root / f"{role}_{arm}_result.json", result)
    print(json.dumps({"role": role, "arm": arm, **result}), flush=True)
    return result


def grade(pilot=False):
    cases, packets = check_prepared()
    root = PILOT if pilot else GRADE
    inputs = [PREP / n for n in ("producer.json", "result.json", "cases.json", "exact_packets.json", "public_payload_proof.json", "worker_terminal_receipt.json")]
    cache = prior_cache()
    if not pilot:
        teacher.training.data.terminal(PILOT)
        if q.read(PILOT / "result.json")["status"] != "systems_pilot_passed_not_quality_evidence":
            raise ValueError("all12 valid arm-case pilot judgments required")
        pilot_cache = q.read(PILOT / "judgment_cache.json")
        for key, entry in pilot_cache.items():
            first.audit_row(entry["packet"], entry["role"], entry["row"])
            first.remember(cache, key, entry)
        inputs += [PILOT / n for n in ("producer.json", "result.json", "judgment_cache.json", "worker_terminal_receipt.json")]
    p = producing(root, "semantic_judge_pilot" if pilot else "oracle_codec_semantic_quality", inputs,
                  {"pilot": pilot, "cases": 4 if pilot else 64, "arms": ARMS, "joint_minimum": 61})
    write_json(root / "judgment_cache.json", cache)
    results = {arm: {"secondary": grade_arm(root, arm, "secondary", cases[:4] if pilot else cases, packets, cache), "primary": None} for arm in ARMS}
    if not pilot and qualifies(results["uncompressed"]["secondary"]):
        for arm in ARMS:
            if qualifies(results[arm]["secondary"]):
                results[arm]["primary"] = grade_arm(root, arm, "primary", cases, packets, cache)
    passed = [arm for arm in ARMS if results[arm]["primary"] is not None and qualifies(results[arm]["primary"])]
    if pilot:
        status = "systems_pilot_passed_not_quality_evidence" if all(v["secondary"]["valid"] == 4 for v in results.values()) else "systems_pilot_failed"
    else:
        status = "completed"
    qualified = [arm for arm in ("original", "trained") if arm in passed and "uncompressed" in passed]
    verify(p)
    result = {"status": status, "classification": "systems_only" if pilot else ("oracle_semantic_codec_qualified_conditional_generation_unproven" if qualified else "oracle_semantic_codec_quality_failed"),
        "results": results, "semantically_qualified_codecs": qualified, "prior_first_judgments_preserved": 320,
        "oracle_target_codes": True, "source_prediction": False, "task_accuracy": None,
        "NCM_success_established": False, "confirmed": False, "final_test_opened": False,
        "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def main():
    teacher.old.dependencies()
    teacher.old.profile.install()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["prepare", "pilot", "grade"])
    stage = parser.parse_args().stage
    prepare() if stage == "prepare" else grade(stage == "pilot")


if __name__ == "__main__":
    main()
