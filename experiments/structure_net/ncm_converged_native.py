"""Measure all-case development quality of fixed-point native NCM decoding."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.metadata
import json
from pathlib import Path
import shutil

import torch

from experiments.structure_net import train_ncm_native_query_neighborhood as run
from experiments.structure_net import ncm_native_query_first_grade as old_grade
from experiments.structure_net import ncm_native_word_mechanism as mechanism
from experiments.structure_net import ncm_native_word_convergence as convergence
from experiments.structure_net import ncm_source_readout_grade as first
from experiments.structure_net import ncm_budget128_profile as profile
from experiments.structure_net.ncm_native_label_quality import dependencies
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_parent_control import native_inputs
from experiments.structure_net.ncm_native_readout import frozen_hash, restore_trainable, trainable_state
from experiments.structure_net.ncm_native_word_neighborhood import initial_state, native_word_logits, predicted_concepts, special_policy
from experiments.structure_net.ncm_paper_data import ROOT, PREP, freeze, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-converged-development-quality-plan.md")
PILOT = ROOT / "20261023_converged_native_pilot_v1"
OUTPUTS = ROOT / "20261023_converged_native_outputs_v1"
GRADE = ROOT / "20261023_converged_native_grade_v1"
PRIOR_CAMPAIGN = ROOT / "20261023_native_query_neighborhood_results_v1/campaign_results.json"
PRIOR_REGISTRY = Path("data/hypothesis_registry/ncm-native-query-neighborhood-v1.json")


def producing(root, stage, inputs, config):
    paths = [Path(p) for p in run.q.read(run.TRAIN / "producer.json")["source_hashes"]]
    paths += [Path(__file__), PLAN, Path(old_grade.__file__), old_grade.PLAN,
        Path(mechanism.__file__), Path(convergence.__file__), convergence.PLAN,
        Path(first.__file__), first.AMENDMENT, Path(profile.__file__), profile.PLAN,
        Path("experiments/structure_net/ncm_native_label_quality.py")]
    paths = list(dict.fromkeys(paths))
    producer = freeze(root, stage, inputs + paths, {**config,
        "hypothesis_id": "ncm-converged-native-summary-quality-v1", "NCM_trained": True,
        "NCM_causal_hybrid": True, "word_refinements": 128, "verification_passes": 1,
        "concept_passes": 5, "fixed_forward_passes": 134, "seed": 42,
        "original_weights_frozen": True, "native_blocks": 28, "allow_tf32": False,
        "precision": "BF16native/FP32new", "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"), "paper_replication": False})
    for path in paths:
        producer["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("converged_" + path.name)).write_bytes(path.read_bytes())
    producer["fingerprint"] = run.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(root / "producer.json", producer)
    return producer


def terminal(root):
    if run.q.read(root / "worker_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("observed terminal success required: " + str(root))
    verify(run.q.read(root / "producer.json"))


def inputs_and_cases():
    terminal(run.TRAIN); terminal(run.OUTPUTS)
    if file_hash(run.TRAIN / "model.pt") != run.q.read(run.TRAIN / "result.json")["checkpoint_sha256"]:
        raise ValueError("original trained checkpoint changed")
    cases = run.q.read(PREP / "development_sources.json")
    verify_public(cases)
    old = run.q.read(run.OUTPUTS / "outputs.json")
    identity = ("id", "source_id", "group_id", "source_sha256", "source")
    training = run.q.read(run.data.DEST / "cases.json")
    groups = {c["group_id"] for rows in training.values() for c in rows}
    if (len(cases) != 64 or len(old) != 64 or groups & {c["group_id"] for c in cases}
            or any(tuple(a[k] for k in identity) != tuple(b[k] for k in identity) for a, b in zip(cases, old))
            or file_hash(run.OUTPUTS / "outputs.json") != run.q.read(run.OUTPUTS / "result.json")["output_sha256"]):
        raise ValueError("fixed complete source-disjoint development cohort changed")
    inputs = [run.TRAIN / n for n in ("model.pt", "producer.json", "result.json", "worker_terminal_receipt.json")]
    inputs += [run.OUTPUTS / n for n in ("outputs.json", "producer.json", "result.json")]
    inputs += [PREP / "development_sources.json", run.data.DEST / "cases.json", run.cache_preparation.BOOKS / "concepts.pt"]
    inputs += [p for p in run.MODEL_PATH.iterdir() if p.is_file()]
    return inputs, cases, old


def invariant_hashes(net):
    return {"original": frozen_hash(net), "new_parameters": mechanism.tensor_digest(trainable_state(net)),
        "books": mechanism.tensor_digest({str(i): b for i, b in enumerate(net.books)})}


@torch.no_grad()
def generate_batch(net, tokenizer, cases, originals):
    x, _ = native_inputs(cases, tokenizer, "cuda")
    memory = net.backbone.source_cache(**x)
    codes, prior = predicted_concepts(net, memory)
    initial = initial_state(net, prior, tokenizer, 2048)
    def step(state):
        logits = native_word_logits(net, memory, prior, state)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite native word logits")
        return logits.argmax(-1)
    state, states, prefixes, verified = convergence.iterate(step, initial, 128)
    stops, _ = special_policy(net, tokenizer)
    outputs, traces = [], []
    for j, case in enumerate(cases):
        actual_codes = [c[j].tolist() for c in codes]
        replay = states[3][j] == originals[j]["raw_parallel_ids"] and actual_codes == originals[j]["codes"]
        full = state[j].tolist()
        end = next((i + 1 for i, token in enumerate(full) if token in stops), len(full))
        outputs.append({**case, "text": tokenizer.decode(full[:end], skip_special_tokens=True).strip(),
            "raw_parallel_ids": full, "native_output_ids": full[:end], "codes": actual_codes,
            "source_truncated": False, "NCM_model": True, "NCM_trained": True,
            "NCM_causal_hybrid": True, "fixed_forward_passes": 134, "word_updates": 128,
            "verification_passes": 1, "AR_fallback": False, "external_verifier": False,
            "four_pass_replay_exact": replay, "full_frame_fixed": verified})
        traces.append({"id": case["id"], "initial_ids": initial[j].tolist(),
            "word_states": [s[j] for s in states], "matching_prefixes": [p[j] for p in prefixes]})
    return outputs, traces


def require_generated(rows, count):
    if len(rows) != count or any(not r["four_pass_replay_exact"] or not r["full_frame_fixed"] for r in rows):
        raise ValueError("fixed convergence or original four-pass replay failed")


def generate(pilot=False):
    run.require_gpu()
    inputs, cases, old = inputs_and_cases()
    if not pilot:
        terminal(PILOT)
        if run.q.read(PILOT / "result.json")["status"] != "systems_lifecycle_passed_not_quality_evidence":
            raise ValueError("exact inference/reload pilot required")
        inputs += [PILOT / n for n in ("producer.json", "result.json", "outputs.json", "worker_terminal_receipt.json")]
        inputs += sorted((PILOT / "word_states").glob("*.json"))
    root = PILOT if pilot else OUTPUTS
    producer = producing(root, "converged_native_inference_reload_pilot" if pilot else "converged_native_development_generation",
        inputs, {"cases": 2 if pilot else 64, "pilot": pilot, "batch": 2, "physical_gpu": 0,
                 "retained_pilot_cases": 0 if pilot else 2, "task_accuracy": None})
    (root / "word_states").mkdir()
    net, tokenizer = run.model_and_tokenizer()
    saved = torch.load(run.TRAIN / "model.pt", map_location="cpu", weights_only=True)
    restore_trainable(net, saved["trainable"])
    before = invariant_hashes(net)
    if before["original"] != run.q.read(run.TRAIN / "result.json")["original_frozen_hash_unchanged"]:
        raise ValueError("original frozen body differs")
    outputs = [] if pilot else run.q.read(PILOT / "outputs.json")
    if not pilot:
        require_generated(outputs, 2)
        for source in (PILOT / "word_states").glob("*.json"):
            shutil.copyfile(source, root / "word_states" / source.name)
    for start in range(0 if pilot else 2, 2 if pilot else 64, 2):
        rows, traces = generate_batch(net, tokenizer, cases[start:start+2], old[start:start+2])
        outputs.extend(rows)
        write_json(root / "word_states" / f"batch_{start//2:03d}.json", traces)
        write_json(root / "outputs.json", outputs)
        require_generated(rows, 2)
        print(json.dumps({"generated": len(outputs), "total": 2 if pilot else 64}), flush=True)
    if pilot:
        with torch.no_grad():
            for parameter in net.parameters():
                if parameter.requires_grad:
                    parameter.zero_()
        restore_trainable(net, saved["trainable"])
        reload_rows, reload_traces = generate_batch(net, tokenizer, cases[:2], old[:2])
        if reload_rows != outputs or reload_traces != run.q.read(root / "word_states/batch_000.json"):
            raise ValueError("native generation reload differs")
        write_json(root / "reload_check.json", {"all_word_states_exact": True, "outputs_exact": True})
    if invariant_hashes(net) != before:
        raise ValueError("body, new parameters or books mutated")
    verify(producer)
    result = {"status": "systems_lifecycle_passed_not_quality_evidence" if pilot else "generated_pending_task_grade",
        "cases": len(outputs), "pilot_outputs_retained_exactly": not pilot, "fixed_forward_passes": 134,
        "all_four_pass_replays_exact": True, "all_final_frames_fixed": True,
        "body_new_parameters_books_unchanged": True, "NCM_trained": True, "NCM_causal_hybrid": True,
        "task_accuracy": None, "NCM_success_established": False, "final_test_opened": False,
        "speed_tested": False, "producer_fingerprint": producer["fingerprint"],
        "output_sha256": file_hash(root / "outputs.json")}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


def load_prior_cache():
    cache = old_grade.load_prior_cache()
    registered = run.q.read(PRIOR_REGISTRY)
    if registered["status"] != "registered_and_read_back" or file_hash(PRIOR_CAMPAIGN) != registered["record"]["campaign_sha256"]:
        raise ValueError("prior failed-student registry changed")
    for path, expected in run.q.read(PRIOR_CAMPAIGN)["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("prior failed-student evidence changed: " + path)
    for key, entry in run.q.read(old_grade.GRADE / "judgment_cache.json").items():
        if key != first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("prior exact packet/profile/model identity differs")
        first.audit_row(entry["packet"], entry["role"], entry["row"])
        first.remember(cache, key, entry)
    if len(cache) != 192:
        raise ValueError("all192 prior first judgments required")
    return cache


def quality_gate(result):
    return result["cases"] == 64 and result["valid"] == 64 and result["passed"] >= 61


def score_case(case, packet, entry, encoding):
    row = entry["row"]
    tokens = len(encoding.encode(case["text"], disallowed_special=()))
    valid = row["status"] == "valid"
    passed = run.q.passes(row["evaluations"][0], case["text"], tokens) if valid else None
    return {"id": case["id"], "source_id": case["source_id"], "text": case["text"],
        "tokens": tokens, "native_tokens_including_stop": len(case["native_output_ids"]),
        "valid": valid, "passed": passed, "evaluations": row.get("evaluations"),
        "attempts": row["attempts"], "packet_key": first.packet_key(packet, entry["role"]), "first_origin": entry["origin"]}


def summarize_scores(scores):
    passed = sum(s["passed"] is True for s in scores)
    invalid = sum(not s["valid"] for s in scores)
    return {"cases": len(scores), "passed": passed, "valid": len(scores) - invalid, "invalid": invalid,
        "task_accuracy": passed / len(scores) if not invalid else None,
        "all_case_pass_lower_bound": passed / len(scores), "all_case_pass_upper_bound": (passed + invalid) / len(scores),
        "wilson_95": run.q.wilson(passed, len(scores)) if not invalid else None,
        "components": {k: sum(s["valid"] and s["evaluations"][0][k] is True for s in scores) for k in ("faithful", "salient", "readable")},
        "stored_overflow": sum(s["tokens"] > 128 for s in scores), "empty": sum(not s["text"].strip() for s in scores)}


def grade_condition(cases, role, cache, inputs):
    import tiktoken
    root = GRADE / (role + "_summary")
    producer = producing(root, "converged_native_whole_summary_first_judgments", inputs,
        {"role": role, "cases": 64, "rubric": profile.RUBRIC, "workers": 4, "max_attempts": 3})
    packets = [run.q.packet(c["id"], c["source"], [{"candidate_id": "summary", "text": c["text"]}]) for c in cases]
    write_json(root / "packets.json", packets)
    missing = [p for p in packets if first.packet_key(p, role) not in cache]
    if len({first.packet_key(p, role) for p in missing}) != len(missing):
        raise ValueError("duplicate uncached packet")
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {}
        for packet in missing:
            key = first.packet_key(packet, role)
            request_root = GRADE / "requests" / role / key
            (request_root / "attempts").mkdir(parents=True, exist_ok=False)
            jobs[pool.submit(run.q.request, packet, role, request_root)] = (key, packet, request_root)
        for i, future in enumerate(as_completed(jobs), 1):
            key, packet, request_root = jobs[future]
            row = future.result(); first.audit_row(packet, role, row)
            write_json(request_root / "row.json", row)
            first.remember(cache, key, {"packet": packet, "role": role, "row": row, "origin": str(request_root)})
            write_json(GRADE / "judgment_cache.json", cache)
            if i % 16 == 0 or i == len(missing):
                print(json.dumps({"role": role, "new_completed": i, "new_total": len(missing)}), flush=True)
    encoding = tiktoken.get_encoding("cl100k_base")
    scores = []
    for case, packet in zip(cases, packets):
        entry = cache[first.packet_key(packet, role)]
        first.audit_row(packet, role, entry["row"])
        scores.append(score_case(case, packet, entry, encoding))
    summary = summarize_scores(scores)
    result = {"status": "completed" if not summary["invalid"] else "invalid_judgments", **summary,
        "new_requests": len(missing), "reused_first_judgments": len(packets) - len(missing),
        "judge": run.q.JUDGES[role], "NCM_trained": True, "NCM_causal_hybrid": True,
        "speed_tested": False, "producer_fingerprint": producer["fingerprint"]}
    verify(producer)
    write_json(root / "case_scores.json", scores); write_json(root / "result.json", result)
    print(json.dumps({"role": role, **result}), flush=True)
    return result


def grade():
    dependencies(); profile.install()
    terminal(OUTPUTS)
    generated = run.q.read(OUTPUTS / "result.json")
    if generated["status"] != "generated_pending_task_grade" or file_hash(OUTPUTS / "outputs.json") != generated["output_sha256"]:
        raise ValueError("complete unchanged generation required")
    cases = run.q.read(OUTPUTS / "outputs.json")
    require_generated(cases, 64)
    original = run.q.read(PREP / "development_sources.json")
    identity = ("id", "source_id", "group_id", "source_sha256", "source")
    if any(tuple(a[k] for k in identity) != tuple(b[k] for k in identity) for a, b in zip(cases, original)):
        raise ValueError("complete original development sources differ")
    verify_public(cases)
    inputs = [OUTPUTS / n for n in ("producer.json", "result.json", "outputs.json", "worker_terminal_receipt.json")]
    for role in ("secondary", "primary"):
        root = ROOT / ("20261023_budget128_judge_" + role + "_v1")
        terminal(root)
        if run.q.read(root / "result.json")["status"] != "passed":
            raise ValueError("qualified128-token judges required")
        inputs += [root / n for n in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    cache = load_prior_cache()
    inputs += [old_grade.GRADE / "judgment_cache.json", PRIOR_CAMPAIGN, PRIOR_REGISTRY]
    producer = producing(GRADE, "converged_native_development_quality", inputs,
        {"cases": 64, "prior_first_judgments": 192, "rubric": profile.RUBRIC, "joint_minimum": 61})
    write_json(GRADE / "judgment_cache.json", cache)
    secondary = grade_condition(cases, "secondary", cache, inputs)
    primary = grade_condition(cases, "primary", cache, inputs) if quality_gate(secondary) else None
    passed = primary is not None and quality_gate(primary)
    verify(producer)
    result = {"status": "completed", "secondary": secondary, "primary": primary,
        "classification": "short_converged_development_gate_passed_final_required" if passed else "converged_native_summary_development_failed",
        "development_gate": passed, "prior_first_judgments_preserved": 192,
        "NCM_trained": True, "NCM_causal_hybrid": True, "NCM_success_established": False,
        "final_test_opened": False, "speed_tested": False, "producer_fingerprint": producer["fingerprint"]}
    write_json(GRADE / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "generate", "grade"])
    command = parser.parse_args().command
    generate(command == "pilot") if command != "grade" else grade()
