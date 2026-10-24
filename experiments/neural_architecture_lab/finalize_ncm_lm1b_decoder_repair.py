"""Independently replay fixed decoder-repair outputs and first semantic decisions."""
import argparse
import json
import math
from pathlib import Path

from experiments.structure_net import ncm_lm1b_decoder_repair_eval as run
from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
from experiments.neural_architecture_lab.finalize_ncm_dense_semantic_development import joint_counts
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def replay_numeric(rows, outputs, old, tokenizer):
    if [r["id"] for r in rows] != [r["id"] for r in outputs]:
        raise ValueError("complete ordered reconstruction required")
    counts = {"cases": len(rows), "correct": 0, "total": 0, "exact_native_ids": 0,
        "first_EOS_correct": 0, "active_invalid_ids": 0, "fits_one_frame": 0, "overlength_in_denominator": 0}
    for row, value in zip(rows, outputs):
        raw = value["raw_ids"]
        if len(raw) != 128 * len(row["packed_ids"]) or any(not 0 <= v < 30528 for v in raw):
            raise ValueError("all raw decoder positions required")
        if value["codes"] != old[row["id"]]["codes"]:
            raise ValueError("frozen native codes changed")
        if len(value["codes"]) != len(row["packed_ids"]) or any(len(c) != sum(run.native.LEVELS)
                or any(not 0 <= v < 8192 for v in c) for c in value["codes"]):
            raise ValueError("full native code hierarchy required")
        frames = [raw[i:i + 128] for i in range(0, len(raw), 128)]
        text, ends, bos = run.framed.render(frames, tokenizer)
        gold = row["target_ids"]
        expected = {"text": text, "native_round_trip": row["native_round_trip"], "target_ids": gold,
            "correct": sum(a == b for a, b in zip(raw, gold)), "total": len(gold),
            "exact_native_ids": raw[:len(gold)] == gold,
            "first_EOS_correct": ends == [sum(v != 0 for v in f) - 1 for f in row["packed_ids"]],
            "frame_EOS_positions": ends, "all_frame_BOS_correct": all(bos),
            "active_invalid_ids": sum(v >= 30522 for f, end in zip(frames, ends) for v in f[:end]),
            "fits_one_native_frame": row["fits_one_native_frame"]}
        if any(value[k] != v for k, v in expected.items()):
            raise ValueError("raw fixed decoder reconstruction metrics differ")
        for k in ("correct", "total", "exact_native_ids", "first_EOS_correct", "active_invalid_ids"):
            counts[k] += expected[k]
        counts["fits_one_frame"] += row["fits_one_native_frame"]
        counts["overlength_in_denominator"] += not row["fits_one_native_frame"]
    return counts


def audit_training():
    import torch
    repair = run.repair
    for root in (repair.DATA, repair.PILOT, repair.TRAIN):
        run.native.vendor.terminal(root)
    result = run.q.read(repair.TRAIN / "result.json")
    if (result["updates"] != 1024 or not result["exact_reload"] or not result["model_training_executed"]
            or result["application_targets_used_for_gradient"] or result["validation_used_for_training"]
            or result["initial_codec_hash"] != "9aace450a49a7dfb44ba1eb00a19974a50d67062ad9f2202a6a2361ea4344e32"
            or result["final_codec_hash"] == result["initial_codec_hash"] or result["peak_allocated_bytes"] > 4700000000
            or result["checkpoint_sha256"] != file_hash(repair.TRAIN / "model.pt")):
        raise ValueError("actual fixed-budget decoder-only training evidence differs")
    if run.q.read(repair.WORKFLOW / "tool_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("actual whole training tool termination required")
    sampler = torch.Generator().manual_seed(repair.Config().sampler_seed)
    augmentation = torch.Generator().manual_seed(repair.Config().augmentation_seed)
    frames = torch.load(repair.DATA / "frames.pt", map_location="cpu", weights_only=True)
    trace = [json.loads(line) for line in (repair.TRAIN / "trace.jsonl").read_text().splitlines()]
    if len(trace) != 1024:
        raise ValueError("contiguous full1024 update trace required")
    for step, row in enumerate(trace, 1):
        samples = []
        for _ in range(repair.Config().accumulation):
            indices = torch.randint(len(frames), (repair.Config().microbatch,), generator=sampler).tolist()
            samples += [{"row": i, **repair.augment(frames[i], augmentation)[1]} for i in indices]
        if (row["step"] != step or row["lr"] != repair.rate(step) or row["samples"] != samples
                or not math.isfinite(row["loss"]) or not math.isfinite(row["gradient_norm"])):
            raise ValueError("training schedule, augmentation, samples or finite trace differs")
    for stage in ("prepare", "pilot", "train"):
        receipt = run.q.read(repair.WORKFLOW / "stage_receipts" / (stage + ".json"))
        if receipt["exit_code"] != 0 or receipt["observer"] != "subprocess.wait":
            raise ValueError("actual successful training child receipt required")
    return result, {"first32_mean_loss": sum(r["loss"] for r in trace[:32]) / 32,
        "last32_mean_loss": sum(r["loss"] for r in trace[-32:]) / 32,
        "sampled_frames": 1024 * 32, "trace_replayed_exactly": True, "task_accuracy": None}


def finalize():
    run.configure()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh fixed-repair aggregate and report required")
    q = run.q
    training, trace = audit_training()
    roots = (run.PILOT, run.RECON, run.PACKETS, run.JUDGE_PILOT, run.GRADE)
    for root in roots:
        run.native.vendor.terminal(root)
    rows = q.read(run.framed.DATA / "cases.json")
    outputs = q.read(run.RECON / "outputs.json")
    old = {r["id"]: r for r in q.read(run.framed.RECON / "outputs.json")}
    if len(rows) != 690 or len({r["id"] for r in rows}) != 690:
        raise ValueError("all690 unchanged targets required")
    numeric = replay_numeric(rows, outputs, old, run.native.tokenizer())
    state = q.read(run.RECON / "result.json")
    if any(state[k] != v for k, v in numeric.items()) or file_hash(run.RECON / "outputs.json") != state["outputs_sha256"]:
        raise ValueError("all690 numeric aggregate differs")
    for root in (run.PILOT, run.RECON):
        value = q.read(root / "result.json")
        if (value["codec_parameter_hash"] != training["final_codec_hash"] or not value["codec_unchanged"]
                or value["peak_allocated_bytes"] > 4700000000):
            raise ValueError("fixed trained checkpoint or inference memory differs")
    by_id = {r["id"]: r for r in outputs}
    if any(by_id[r["id"]] != r for r in q.read(run.PILOT / "outputs.json")):
        raise ValueError("fresh evaluation reload differs from retained pilot")
    chosen = run.prior.select(q.read(run.native.CASES)["train"])
    cases, packets = q.read(run.PACKETS / "cases.json"), q.read(run.PACKETS / "exact_packets.json")
    previous = q.read(run.BASE_PACKETS / "cases.json")
    if [r["id"] for r in chosen] != [r["id"] for r in cases] or [r["id"] for r in cases] != [r["id"] for r in previous]:
        raise ValueError("same fixed64 source selection required")
    for source, case, old_case in zip(chosen, cases, previous):
        expected = {arm: old_case["texts"][arm] for arm in run.ARMS if arm != "repaired_codec"}
        expected["repaired_codec"] = by_id[case["id"]]["text"]
        if case["texts"] != expected or any(case[k] != source[k] for k in run.reference.IDENTITY):
            raise ValueError("unchanged sources and exact candidate texts required")
    proof = run.reference.verify_public(cases)
    if q.read(run.PACKETS / "public_payload_proof.json") != {"public_file_sha256": run.reference.PUBLIC_SHA256,
            "exact_sources": proof, "private_data_in_packets": False,
            "exact_packets_sha256": file_hash(run.PACKETS / "exact_packets.json")}:
        raise ValueError("exact public source proof differs")
    if packets != {arm: [q.packet(c["id"], c["source"], [{"candidate_id": "summary", "text": c["texts"][arm]}]) for c in cases] for arm in run.ARMS}:
        raise ValueError("anonymous single candidate packet differs")
    baseline, pilot_cache = run.base_cache(), q.read(run.JUDGE_PILOT / "judgment_cache.json")
    cache = q.read(run.GRADE / "judgment_cache.json")
    if any(pilot_cache.get(k) != v for k, v in baseline.items()) or any(cache.get(k) != v for k, v in pilot_cache.items()):
        raise ValueError("canonical prior or pilot first judgment changed")
    run.prior.audit_cache(cache)
    allowed = {run.reference.first.packet_key(p, role) for payloads in packets.values() for p in payloads for role in ("secondary", "primary")}
    if set(cache) - set(baseline) - allowed:
        raise ValueError("extra first judgment outside fixed study appeared")
    grade, pilot = q.read(run.GRADE / "result.json"), q.read(run.JUDGE_PILOT / "result.json")
    results, joint = {}, {}
    for arm in run.ARMS:
        pilot_rows = replay_scores(cases[:4], {arm: packets[arm][:4]}, pilot_cache, arm, "secondary")
        if q.read(run.JUDGE_PILOT / f"secondary_{arm}_case_scores.json") != pilot_rows or not all(r["valid"] for r in pilot_rows):
            raise ValueError("all actual pilot first decisions must be retained")
        if any(pilot["results"][arm]["secondary"][k] != v for k, v in run.oracle.metric(pilot_rows).items()):
            raise ValueError("raw pilot semantic metrics differ")
        primary_expected = (run.oracle.qualifies(grade["results"]["uncompressed"]["secondary"])
            and run.oracle.qualifies(grade["results"][arm]["secondary"]))
        results[arm], scores = {}, {}
        for role in ("secondary", "primary"):
            endpoint = grade["results"][arm][role]
            if role == "primary" and not primary_expected:
                if endpoint is not None: raise ValueError("primary admitted before fixed61/64 gate")
                scores[role], results[arm][role] = None, None
                continue
            scores[role] = replay_scores(cases, packets, cache, arm, role)
            if q.read(run.GRADE / f"{role}_{arm}_case_scores.json") != scores[role]:
                raise ValueError("first-decision semantic scores differ")
            if endpoint is None or any(endpoint[k] != v for k, v in run.oracle.metric(scores[role]).items()):
                raise ValueError("semantic aggregate differs")
            results[arm][role] = endpoint
        joint[arm] = joint_counts(scores["secondary"], scores["primary"])
    qualifies = lambda r: r is not None and r["cases"] == r["valid"] == 64 and r["passed"] >= 61
    qualified = qualifies(joint["uncompressed"]) and qualifies(joint["repaired_codec"])
    invalid = any(v[r]["invalid"] for v in results.values() for r in ("secondary", "primary") if v[r] is not None)
    classification = ("decoder_repair_semantic_judgments_invalid" if invalid else
        "decoder_repair_oracle_semantics_qualified_generation_unproven" if qualified else "decoder_repair_oracle_semantics_quality_failed")
    if (grade["canonical_first_judgments"] != len(cache) or grade["prior_first_judgments_preserved"] != len(pilot_cache)
            or pilot["canonical_first_judgments"] != len(pilot_cache) or pilot["status"] != "systems_semantic_pilot_passed"):
        raise ValueError("complete canonical first-decision counts or pilot status differs")
    receipts = [run.WORKFLOW / "stage_receipts" / (s + ".json") for s in ("pilot", "measure", "packets", "judge-pilot", "grade")]
    if any(q.read(p)["exit_code"] != 0 or q.read(p)["observer"] != "subprocess.wait" for p in receipts):
        raise ValueError("actual successful producing child receipts required")
    artifacts = {str(p): file_hash(p) for p in [Path(__file__), Path(run.__file__), run.PLAN, run.WORKFLOW / "producer.json", *receipts]}
    artifacts.update(q.read(run.WORKFLOW / "producer.json")["input_hashes"])
    for root in (*roots, run.repair.DATA, run.repair.PILOT, run.repair.TRAIN):
        artifacts.update(q.read(root / "producer.json")["input_hashes"])
        artifacts.update({str(p): file_hash(p) for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts})
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": classification,
        "decoder_training": training, "training_trace_audit": trace, "numeric_repaired": numeric,
        "numeric_baseline": q.read(run.framed.RECON / "result.json"), "semantic_results": results,
        "same_case_joint": joint, "repaired_codec_qualified": qualified,
        "canonical_prior_firsts": len(baseline), "canonical_first_judgments": len(cache),
        "all_prior_and_pilot_firsts_preserved": True, "oracle_target_codes": True, "source_prediction": False,
        "model_training_executed": True, "task_accuracy": None, "final_test_opened": False,
        "speed_tested": False, "confirmed": False, "artifacts": artifacts}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    lines = [f"| {arm} | {role} | {v[role]['passed']}/64 | {v[role]['valid']}/64 |" for arm, v in results.items()
             for role in ("secondary", "primary") if v[role] is not None]
    run.REPORT.write_text(f"""# Released NCM decoder fine-tune

Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`.
Classification **{classification}**. Campaign `{path}`, SHA256 `{file_hash(path)}`.

The authors' exact LM1B 111M codec was loaded before updates. Only its 18,870,656
decoder parameters were fine-tuned for 1024 fixed updates. The encoder, codebooks,
EMA and normalization state stayed unchanged. The original Llama and native NCP
generator were absent. All 690 evaluation code hierarchies match the release.
The native TRAIN split supplied 16384 selected frames, 20% full128 and 80% variable
short reconstructions. Application summaries and native validation were not used
for gradients. Decoder/optimizer/RNG reload and all 1024 sampled updates were
audited. Training loss is not task accuracy.

| Same 64 full public sources | Judge | Reconstructed-summary passes | Valid |
| --- | --- | ---: | ---: |
""" + "\n".join(lines) + f"""

Oracle codec qualified: **{qualified}**. Qualification requires 61 of the SAME 64
cases passing both qualified judges, including the uncompressed control. Meaning,
salience, readability, nonempty output and <=128 storage tokens determine passes;
paraphrases are accepted. The inherited 2239 decisions and pilot firsts remain
unchanged; the resulting cache has {len(cache)} firsts. No valid failure was rerolled.

All 690 numeric targets remain, including 25 that need multiple native frames.
Repaired token reconstruction is {numeric['correct']}/{numeric['total']}, versus
63872/65063 before fine-tuning. These are gold-code TRAIN reconstruction probes,
not a source-only summarizer or held-out task accuracy. The previous conditional
student remains 0/128 on its source-disjoint development evaluation. No final
source was opened. The 95% task goal and any speed improvement remain unestablished.
""")
    print(json.dumps({"classification": classification, "numeric_repaired": numeric,
        "semantic_results": results, "same_case_joint": joint, "canonical_firsts": len(cache),
        "campaign_sha256": file_hash(path)}), flush=True)


def register():
    campaign = run.q.read(run.RESULTS / "campaign_results.json")
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Released native NCM fixed1024 decoder fine-tune and semantic audit",
        "description": "Frozen encoder/books; author TRAIN reconstruction gradients only; fixed final1024, all690 numeric targets and same64 source-grounded semantic controls. No conditional-model or speed claim.",
        "classification": campaign["classification"], "semantic_component_quality_measured": True,
        "source_only_task_accuracy_measured": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    run.configure()
    (finalize if args.stage == "finalize" else register)()
