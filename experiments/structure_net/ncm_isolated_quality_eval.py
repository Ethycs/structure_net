"""All-case quality evaluation of the isolated native word teacher control."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
import json
from pathlib import Path
import shutil

import torch

from experiments.structure_net import ncm_isolated_quality_train as training
from experiments.structure_net import ncm_converged_native as old
from experiments.structure_net import ncm_quality_word_runtime as runtime
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_native_parent_control import native_inputs
from experiments.structure_net.ncm_native_readout import frozen_hash, restore_trainable, trainable_state
from experiments.structure_net.ncm_native_word_mechanism import tensor_digest
from experiments.structure_net.ncm_native_word_neighborhood import special_policy
from experiments.structure_net.ncm_paper_data import ROOT, PREP, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public

AMENDMENT = Path("docs/07 - Status Reports/2026-10-23_ncm-quality-evaluation-runtime-amendment.md")
PILOT = ROOT / "20261023_isolated_quality_inference_pilot_v1"
OUTPUTS = ROOT / "20261023_isolated_quality_outputs_v1"
GRADE = ROOT / "20261023_isolated_quality_grade_v1"
PRIOR_CAMPAIGN = ROOT / "20261023_converged_native_results_v1/campaign_results.json"
PRIOR_REGISTRY = Path("data/hypothesis_registry/ncm-converged-native-summary-quality-v1.json")


def producing(root, stage, inputs, config):
    files = [Path(__file__), AMENDMENT, Path(runtime.__file__), Path(old.__file__), old.PLAN]
    p = training.producing(root, stage, inputs + files,
        {**config, "inference_only": True, "word_updates": 128, "verification_passes": 1,
         "query_forward_passes": 129, "coarse_queries_used": False, "new_word_hint_or_bias": False})
    for path in files:
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("quality_eval_" + path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = old.run.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def source_cases():
    cases = old.run.q.read(PREP / "development_sources.json")
    rows = old.run.q.read(training.data.DEST / "cases.json")
    if len(cases) != 64:
        raise ValueError("original64development cases required")
    training.data.check_disjoint(rows, cases)
    verify_public(cases)
    return cases


def load_model():
    training.data.terminal(training.TRAIN)
    result = old.run.q.read(training.TRAIN / "result.json")
    if (result["status"] != "quality_teacher_trained_pending_task_evaluation" or result["updates"] != 512
            or not result["only_word_bank_trained"] or not result["original_body_forecaster_books_unchanged"]
            or file_hash(training.TRAIN / "model.pt") != result["checkpoint_sha256"]):
        raise ValueError("terminal final512quality checkpoint required")
    net, tokenizer = training.model_and_tokenizer()
    checkpoint = torch.load(training.TRAIN / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["step"] != 512 or checkpoint["producer_fingerprint"] != result["producer_fingerprint"]:
        raise ValueError("quality checkpoint identity differs")
    restore_trainable(net, checkpoint["word"])
    return net, tokenizer, checkpoint["word"]


def model_hashes(net):
    return {"frozen": frozen_hash(net), "word": tensor_digest(trainable_state(net)),
        "books": tensor_digest({str(i): b for i, b in enumerate(net.books)})}


@torch.no_grad()
def make_memory(net, tokenizer, cases):
    inputs, _ = native_inputs(cases, tokenizer, "cuda")
    return net.backbone.source_cache(**inputs)


@torch.no_grad()
def decode(net, tokenizer, memory, cases, static):
    state = torch.full((len(cases), 128), tokenizer.eos_token_id, device="cuda", dtype=torch.long)
    initial = state.clone()
    states, prefixes = [], []
    context = runtime.static_word_runtime(net.backbone) if static else nullcontext()
    with context:
        for _ in range(128):
            logits = training.bank.word_logits(net, memory, state)
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite isolated word logits")
            updated = logits.argmax(-1)
            prefixes.append(state.eq(updated).long().cumprod(-1).sum(-1).cpu().tolist())
            states.append(updated.cpu().tolist()); state = updated
        verified = torch.equal(training.bank.word_logits(net, memory, state).argmax(-1), state)
    stops, _ = special_policy(net, tokenizer)
    outputs, traces = [], []
    for i, case in enumerate(cases):
        full = state[i].tolist()
        end = next((j + 1 for j, token in enumerate(full) if token in stops), 128)
        outputs.append({**case, "text": tokenizer.decode(full[:end], skip_special_tokens=True).strip(),
            "raw_parallel_ids": full, "native_output_ids": full[:end], "full_frame_fixed": verified,
            "source_truncated": False, "word_updates": 128, "verification_passes": 1,
            "query_forward_passes": 129, "quality_teacher_control": True, "NCM_model": False,
            "AR_fallback": False, "external_verifier": False})
        traces.append({"id": case["id"], "initial_ids": initial[i].tolist(),
            "word_states": [s[i] for s in states], "matching_prefixes": [p[i] for p in prefixes]})
    return outputs, traces


def require_generated(rows, count):
    if len(rows) != count or any(not r["full_frame_fixed"] or r["source_truncated"] for r in rows):
        raise ValueError("all complete teacher outputs must reach verified fixed points")


def generate(pilot=False):
    old.run.require_gpu()
    cases = source_cases()
    inputs = [training.TRAIN / n for n in ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json")]
    inputs += [training.data.DEST / "cases.json", PREP / "development_sources.json"]
    if not pilot:
        training.data.terminal(PILOT)
        if old.run.q.read(PILOT / "result.json")["status"] != "systems_inference_equivalence_passed_not_quality_evidence":
            raise ValueError("plain/static inference/reload pilot required")
        inputs += [PILOT / n for n in ("producer.json", "result.json", "outputs.json", "worker_terminal_receipt.json")]
        inputs += sorted((PILOT / "word_states").glob("*.json"))
    root = PILOT if pilot else OUTPUTS
    p = producing(root, "isolated_word_teacher_inference_pilot" if pilot else "isolated_word_teacher_development_generation",
        inputs, {"pilot": pilot, "cases": 2 if pilot else 64, "retained_pilot": 0 if pilot else 2,
                 "batch": 2, "physical_gpu": 0, "static_word_runtime": not pilot})
    (root / "word_states").mkdir()
    net, tokenizer, saved_word = load_model()
    before = model_hashes(net)
    outputs = [] if pilot else old.run.q.read(PILOT / "outputs.json")
    if not pilot:
        require_generated(outputs, 2)
        for path in (PILOT / "word_states").glob("*.json"):
            shutil.copyfile(path, root / "word_states" / path.name)
    for start in range(0 if pilot else 2, 2 if pilot else 64, 2):
        chunk = cases[start:start+2]
        memory = make_memory(net, tokenizer, chunk)
        rows, traces = decode(net, tokenizer, memory, chunk, static=not pilot)
        outputs.extend(rows)
        write_json(root / "outputs.json", outputs)
        write_json(root / "word_states" / f"batch_{start//2:03d}.json", traces)
        require_generated(rows, 2)
        if pilot:
            initial = torch.full((2, 128), tokenizer.eos_token_id, device="cuda", dtype=torch.long)
            with torch.no_grad():
                plain = training.bank.word_logits(net, memory, initial)
                with runtime.static_word_runtime(net.backbone):
                    fast = training.bank.word_logits(net, memory, initial)
                if not torch.equal(plain, fast):
                    raise ValueError("static runtime full-vocabulary logits differ")
                del plain, fast
            static_rows, static_traces = decode(net, tokenizer, memory, chunk, static=True)
            if (static_rows, static_traces) != (rows, traces):
                raise ValueError("static runtime word histories differ")
            with torch.no_grad():
                for parameter in net.parameters():
                    if parameter.requires_grad:
                        parameter.zero_()
            restore_trainable(net, saved_word)
            reloaded = decode(net, tokenizer, memory, chunk, static=True)
            if reloaded != (rows, traces):
                raise ValueError("reloaded teacher inference differs")
            write_json(root / "equivalence.json", {"initial_logits_exact": True, "all_word_states_exact": True,
                "decoded_outputs_exact": True, "checkpoint_reload_exact": True})
        print(json.dumps({"generated": len(outputs), "total": 2 if pilot else 64}), flush=True)
        del memory
    if model_hashes(net) != before:
        raise ValueError("teacher inference changed frozen or word parameters/books")
    verify(p)
    result = {"status": "systems_inference_equivalence_passed_not_quality_evidence" if pilot else "generated_pending_task_grade",
        "cases": len(outputs), "pilot_outputs_retained_exactly": not pilot, "word_updates": 128,
        "verification_passes": 1, "all_final_frames_fixed": True, "all_parameters_books_unchanged": True,
        "quality_teacher_control": True, "NCM_model": False, "task_accuracy": None,
        "NCM_success_established": False, "final_test_opened": False, "speed_tested": False,
        "producer_fingerprint": p["fingerprint"], "output_sha256": file_hash(root / "outputs.json")}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


def load_prior_cache():
    cache = old.load_prior_cache()
    receipt = old.run.q.read(PRIOR_REGISTRY)
    if receipt["status"] != "registered_and_read_back" or file_hash(PRIOR_CAMPAIGN) != receipt["record"]["campaign_sha256"]:
        raise ValueError("registered converged failure identity changed")
    for path, expected in old.run.q.read(PRIOR_CAMPAIGN)["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("prior converged evidence changed: " + path)
    for key, entry in old.run.q.read(old.GRADE / "judgment_cache.json").items():
        if key != old.first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("prior judgment packet/profile/model differs")
        old.first.audit_row(entry["packet"], entry["role"], entry["row"])
        old.first.remember(cache, key, entry)
    if len(cache) != 256:
        raise ValueError("all256 exact prior first judgments required")
    return cache


def grade_condition(cases, role, cache, inputs):
    import tiktoken
    root = GRADE / (role + "_summary")
    p = producing(root, "isolated_teacher_whole_summary_first_judgments", inputs,
        {"role": role, "cases": 64, "rubric": old.profile.RUBRIC, "workers": 4, "max_attempts": 3})
    packets = [old.run.q.packet(c["id"], c["source"], [{"candidate_id": "summary", "text": c["text"]}]) for c in cases]
    write_json(root / "packets.json", packets)
    missing = [packet for packet in packets if old.first.packet_key(packet, role) not in cache]
    if len({old.first.packet_key(packet, role) for packet in missing}) != len(missing):
        raise ValueError("duplicate new packet")
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {}
        for packet in missing:
            key = old.first.packet_key(packet, role)
            request_root = GRADE / "requests" / role / key
            (request_root / "attempts").mkdir(parents=True, exist_ok=False)
            jobs[pool.submit(old.run.q.request, packet, role, request_root)] = (key, packet, request_root)
        for i, future in enumerate(as_completed(jobs), 1):
            key, packet, request_root = jobs[future]
            row = future.result(); old.first.audit_row(packet, role, row)
            write_json(request_root / "row.json", row)
            old.first.remember(cache, key, {"packet": packet, "role": role, "row": row, "origin": str(request_root)})
            write_json(GRADE / "judgment_cache.json", cache)
            if i % 16 == 0 or i == len(missing):
                print(json.dumps({"role": role, "new_completed": i, "new_total": len(missing)}), flush=True)
    encoding = tiktoken.get_encoding("cl100k_base")
    scores = []
    for case, packet in zip(cases, packets):
        entry = cache[old.first.packet_key(packet, role)]
        old.first.audit_row(packet, role, entry["row"])
        scores.append(old.score_case(case, packet, entry, encoding))
    summary = old.summarize_scores(scores)
    result = {"status": "completed" if not summary["invalid"] else "invalid_judgments", **summary,
        "new_requests": len(missing), "reused_first_judgments": len(cases) - len(missing),
        "judge": old.run.q.JUDGES[role], "quality_teacher_control": True, "NCM_model": False,
        "producer_fingerprint": p["fingerprint"], "speed_tested": False}
    verify(p)
    write_json(root / "case_scores.json", scores); write_json(root / "result.json", result)
    print(json.dumps({"role": role, **result}), flush=True)
    return result


def grade():
    old.dependencies(); old.profile.install()
    training.data.terminal(OUTPUTS)
    generated = old.run.q.read(OUTPUTS / "result.json")
    if generated["status"] != "generated_pending_task_grade" or file_hash(OUTPUTS / "outputs.json") != generated["output_sha256"]:
        raise ValueError("complete unchanged teacher outputs required")
    cases = old.run.q.read(OUTPUTS / "outputs.json")
    require_generated(cases, 64)
    development = old.run.q.read(PREP / "development_sources.json")
    identity = ("id", "source_id", "group_id", "source_sha256", "source")
    if len(development) != 64 or any(tuple(a[k] for k in identity) != tuple(b[k] for k in identity) for a, b in zip(cases, development)):
        raise ValueError("fixed64 source order/text changed")
    verify_public(cases)
    inputs = [OUTPUTS / n for n in ("producer.json", "result.json", "outputs.json", "worker_terminal_receipt.json")]
    for role in ("secondary", "primary"):
        root = ROOT / ("20261023_budget128_judge_" + role + "_v1")
        training.data.terminal(root)
        if old.run.q.read(root / "result.json")["status"] != "passed":
            raise ValueError("both qualified128-token judges required")
        inputs += [root / n for n in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    cache = load_prior_cache()
    inputs += [old.GRADE / "judgment_cache.json", PRIOR_CAMPAIGN, PRIOR_REGISTRY]
    p = producing(GRADE, "isolated_teacher_development_quality", inputs,
        {"cases": 64, "prior_first_judgments": 256, "rubric": old.profile.RUBRIC, "joint_minimum": 61})
    write_json(GRADE / "judgment_cache.json", cache)
    secondary = grade_condition(cases, "secondary", cache, inputs)
    primary = grade_condition(cases, "primary", cache, inputs) if old.quality_gate(secondary) else None
    passed = primary is not None and old.quality_gate(primary)
    verify(p)
    result = {"status": "completed", "secondary": secondary, "primary": primary,
        "classification": "quality_teacher_short_development_passed_NCM_distillation_required" if passed else "isolated_quality_teacher_development_failed",
        "development_teacher_gate": passed, "prior_first_judgments_preserved": 256,
        "quality_teacher_control": True, "NCM_model": False, "NCM_success_established": False,
        "final_test_opened": False, "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    write_json(GRADE / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "generate", "grade"])
    command = parser.parse_args().command
    generate(command == "pilot") if command != "grade" else grade()
