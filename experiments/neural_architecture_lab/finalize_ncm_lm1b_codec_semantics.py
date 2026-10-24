"""Replay padding interventions and every source-grounded first judgment."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_lm1b_codec_semantics as run
from experiments.neural_architecture_lab.finalize_ncm_dense_semantic_development import joint_counts
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def replay_scores(cases, packets, cache, arm, role):
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    rows = []
    for case, packet in zip(cases, packets[arm]):
        key = run.reference.first.packet_key(packet, role)
        entry = cache[key]
        run.reference.first.audit_row(packet, role, entry["row"])
        text = case["texts"][arm]
        tokens = len(encoding.encode(text, disallowed_special=()))
        valid = entry["row"]["status"] == "valid"
        rows.append({"id": case["id"], "text": text, "tokens": tokens, "valid": valid,
            "passed": run.q.passes(entry["row"]["evaluations"][0], text, tokens) if valid else None,
            "evaluations": entry["row"].get("evaluations"), "packet_key": key,
            "attempts": entry["row"]["attempts"], "first_origin": entry["origin"]})
    return rows


def finalize():
    run.configure()
    q = run.q
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh semantic aggregate/report required")
    roots = (run.DATA, run.PILOT, run.RECON, run.PACKETS, run.JUDGE_PILOT, run.GRADE)
    for root in roots:
        run.native.vendor.terminal(root)
    old_rows = q.read(run.framed.DATA / "cases.json")
    rows = q.read(run.DATA / "cases.json")
    if rows != [run.replace_tail_padding(r) for r in old_rows] or len(rows) != 690:
        raise ValueError("same690 cases and only tail-padding intervention required")
    outputs = q.read(run.RECON / "outputs.json")
    old = {r["id"]: r for r in q.read(run.framed.RECON / "outputs.json")}
    if [r["id"] for r in outputs] != [r["id"] for r in rows]:
        raise ValueError("complete ordered native reconstruction required")
    t = run.native.tokenizer()
    counts = {"cases": 690, "correct": 0, "total": 0, "exact_native_ids": 0,
        "first_EOS_correct": 0, "active_invalid_ids": 0, "fits_one_frame": 0, "overlength_in_denominator": 0}
    no_padding_controls = 0
    for r, original_row, value in zip(rows, old_rows, outputs):
        raw = value["raw_ids"]
        if len(raw) != len(r["packed_ids"]) * 128 or any(not 0 <= v < 30528 for v in raw):
            raise ValueError("all native decoder positions required")
        if len(value["codes"]) != len(r["packed_ids"]) or any(len(c) != sum(run.native.LEVELS)
                or any(not 0 <= v < 8192 for v in c) for c in value["codes"]):
            raise ValueError("complete native code hierarchy required")
        frames = [raw[i:i + 128] for i in range(0, len(raw), 128)]
        text, ends, bos = run.framed.render(frames, t)
        correct = sum(a == b for a, b in zip(raw, r["target_ids"]))
        exact = raw[:len(r["target_ids"])] == r["target_ids"]
        invalid = sum(v >= 30522 for frame, end in zip(frames, ends) for v in frame[:end])
        if (value["target_ids"] != r["target_ids"] or value["text"] != text or value["correct"] != correct
                or value["total"] != len(r["target_ids"]) or value["exact_native_ids"] != exact
                or value["first_EOS_correct"] != (ends == r["frame_ends"])
                or value["frame_EOS_positions"] != ends or value["all_frame_BOS_correct"] != all(bos)
                or value["active_invalid_ids"] != invalid):
            raise ValueError("raw native prediction metrics differ")
        if not any(0 in frame for frame in original_row["packed_ids"]):
            if any(value[k] != old[r["id"]][k] for k in ("raw_ids", "codes", "text")):
                raise ValueError("unchanged no-padding case changed prediction")
            no_padding_controls += 1
        for k in ("correct", "total", "exact_native_ids", "first_EOS_correct", "active_invalid_ids"):
            counts[k] += value[k]
        counts["fits_one_frame"] += r["fits_one_native_frame"]
        counts["overlength_in_denominator"] += not r["fits_one_native_frame"]
    numeric = q.read(run.RECON / "result.json")
    if any(numeric[k] != v for k, v in counts.items()) or file_hash(run.RECON / "outputs.json") != numeric["outputs_sha256"]:
        raise ValueError("native numeric aggregate differs")
    expected_codec_hash = q.read(run.framed.RECON / "result.json")["codec_parameter_hash"]
    for root in (run.PILOT, run.RECON):
        state = q.read(root / "result.json")
        if not state["codec_unchanged"] or state["codec_parameter_hash"] != expected_codec_hash or state["peak_allocated_bytes"] > 4700000000:
            raise ValueError("frozen codec identity or memory contract differs")
    by_id = {r["id"]: r for r in outputs}
    if any(by_id[r["id"]] != r for r in q.read(run.PILOT / "outputs.json")):
        raise ValueError("fresh retained numerical pilot differs")
    chosen = run.select(q.read(run.native.CASES)["train"])
    cases = q.read(run.PACKETS / "cases.json")
    if [r["id"] for r in chosen] != [r["id"] for r in cases]:
        raise ValueError("prospective fixed64 selection differs")
    proof = run.reference.verify_public(cases)
    if q.read(run.PACKETS / "public_payload_proof.json") != {"public_file_sha256": run.reference.PUBLIC_SHA256,
            "exact_sources": proof, "private_data_in_packets": False,
            "exact_packets_sha256": file_hash(run.PACKETS / "exact_packets.json")}:
        raise ValueError("complete public source proof differs")
    for source, case in zip(chosen, cases):
        expected_texts = {"uncompressed": source["target_text"], "bert_roundtrip": old[source["id"]]["native_round_trip"],
            "PAD0_codec": old[source["id"]]["text"], "CLS_codec": by_id[source["id"]]["text"]}
        if case["texts"] != expected_texts or any(case[k] != source[k] for k in run.reference.IDENTITY):
            raise ValueError("semantic source/candidate binding differs")
    packets = q.read(run.PACKETS / "exact_packets.json")
    if packets != {arm: [q.packet(c["id"], c["source"], [{"candidate_id": "summary", "text": c["texts"][arm]}]) for c in cases] for arm in run.ARMS}:
        raise ValueError("anonymous exact public packets differ")
    baseline = run.base_cache()
    pilot_cache = q.read(run.JUDGE_PILOT / "judgment_cache.json")
    cache = q.read(run.GRADE / "judgment_cache.json")
    if any(pilot_cache.get(k) != v for k, v in baseline.items()) or any(cache.get(k) != v for k, v in pilot_cache.items()):
        raise ValueError("a canonical baseline or pilot first changed")
    run.audit_cache(cache)
    allowed = {run.reference.first.packet_key(p, role) for payloads in packets.values() for p in payloads for role in ("secondary", "primary")}
    if set(cache) - set(baseline) - allowed:
        raise ValueError("unregistered extra first decisions appeared")
    grade = q.read(run.GRADE / "result.json")
    pilot = q.read(run.JUDGE_PILOT / "result.json")
    if pilot["status"] != "systems_semantic_pilot_passed" or pilot["canonical_first_judgments"] != len(pilot_cache):
        raise ValueError("actual schema-only pilot required")
    results, joint = {}, {}
    for arm in run.ARMS:
        pilot_rows = replay_scores(cases[:4], {arm: packets[arm][:4]}, pilot_cache, arm, "secondary")
        if q.read(run.JUDGE_PILOT / f"secondary_{arm}_case_scores.json") != pilot_rows or not all(r["valid"] for r in pilot_rows):
            raise ValueError("all pilot first decisions must be valid and retained")
        if any(pilot["results"][arm]["secondary"][k] != v for k, v in run.oracle.metric(pilot_rows).items()):
            raise ValueError("pilot raw semantic counts differ")
        results[arm] = {}
        role_scores = {}
        primary_expected = (run.oracle.qualifies(grade["results"]["uncompressed"]["secondary"])
            and run.oracle.qualifies(grade["results"][arm]["secondary"]))
        for role in ("secondary", "primary"):
            endpoint = grade["results"][arm][role]
            if role == "primary" and not primary_expected:
                if endpoint is not None:
                    raise ValueError("primary judge admitted before the fixed gate")
                role_scores[role], results[arm][role] = None, None
                continue
            scores = replay_scores(cases, packets, cache, arm, role)
            if q.read(run.GRADE / f"{role}_{arm}_case_scores.json") != scores:
                raise ValueError("raw first-decision scores differ")
            measured = run.oracle.metric(scores)
            if endpoint is None or any(endpoint[k] != v for k, v in measured.items()):
                raise ValueError("semantic counts/components/intervals differ")
            results[arm][role], role_scores[role] = endpoint, scores
        joint[arm] = joint_counts(role_scores["secondary"], role_scores["primary"])
    qualifies_joint = lambda r: r is not None and r["cases"] == r["valid"] == 64 and r["passed"] >= 61
    qualified = [arm for arm in ("PAD0_codec", "CLS_codec") if qualifies_joint(joint["uncompressed"]) and qualifies_joint(joint[arm])]
    invalid = any(v[role]["invalid"] for v in results.values() for role in ("secondary", "primary") if v[role] is not None)
    classification = ("native_codec_semantic_judgments_invalid" if invalid else
        "native_codec_semantic_training_control_qualified_generation_unproven" if qualified else
        "native_codec_semantic_training_control_quality_failed")
    if grade["canonical_first_judgments"] != len(cache) or grade["prior_first_judgments_preserved"] != len(pilot_cache):
        raise ValueError("complete canonical cache counts differ")
    receipts = [run.WORKFLOW / "stage_receipts" / (s + ".json") for s in ("prepare", "pilot", "measure", "packets", "judge-pilot", "grade")]
    if any(q.read(p)["exit_code"] != 0 or q.read(p)["observer"] != "subprocess.wait" for p in receipts):
        raise ValueError("six actual successful producing children required")
    hashes = {str(p): file_hash(p) for p in [Path(__file__), Path(run.__file__), run.PLAN, run.WORKFLOW / "producer.json", *receipts]}
    hashes.update(q.read(run.WORKFLOW / "producer.json")["input_hashes"])
    for root in roots:
        hashes.update(q.read(root / "producer.json")["input_hashes"])
        hashes.update({str(p): file_hash(p) for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts})
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": classification, "numeric_CLS_results": counts,
        "numeric_PAD_results": q.read(run.framed.RECON / "result.json"), "no_padding_control_cases": no_padding_controls,
        "semantic_results": results, "same_case_joint": joint, "qualified_codecs": qualified,
        "canonical_prior_firsts": len(baseline), "canonical_first_judgments": len(cache),
        "all_prior_and_pilot_firsts_preserved": True, "oracle_target_codes": True, "source_prediction": False,
        "model_training_executed": False, "task_accuracy": None, "final_test_opened": False,
        "speed_tested": False, "confirmed": False, "artifacts": hashes}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    lines = []
    for arm in run.ARMS:
        for role in ("secondary", "primary"):
            e = results[arm][role]
            if e is not None:
                lines.append(f"| {arm} | {role} | {e['passed']}/64 | {e['valid']}/64 |")
    run.REPORT.write_text(f"""# Native NCM padding and semantic reconstruction

Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`.
Classification **{classification}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`. Exploratory target-aware TRAIN representation control.

| Same64 full public sources | Judge | Complete reconstructed-summary passes | Valid |
| --- | --- | ---: | ---: |
""" + "\n".join(lines) + f"""

Qualified codec arms: `{qualified}`. At least61 of the SAME64 cases must pass
both qualified judges for the reference and codec. Paraphrases are accepted;
faithfulness, salience, readability, nonempty text and<=128 storage tokens decide
each pass. These are oracle target-code reconstructions, NOT source-only model
accuracy. The frozen student's separate source-disjoint result remains0/128.

The sample was fixed by hash before outputs:32 original targets and32 reviewed
targets. The uncompressed, BERT roundtrip, PAD0 codec and CLS codec arms isolate
label quality, normalization and neural reconstruction. All prior{len(baseline)}
first decisions and all pilot decisions are preserved; the final cache contains
{len(cache)}. Identical packets shared their first decision. No valid negative
or exhausted invalid was rerolled.

All690 numerical targets remain, including25 overlength targets. The only new
input factor replaces tail PAD0 with CLS101. CLS reconstruction recovers
{counts['correct']}/{counts['total']} native tokens, versus63872/65063 for PAD0.
No-padding controls ({no_padding_controls} cases) retained exact raw predictions.
The native model stayed unchanged, FP32/TF32 off on GPU2, with a repeated memory
pilot and fresh full-load parity. This is not a single-frame claim for overlength
references. Token agreement cannot replace semantic evaluation.

No conditional generator was trained. Original Llama weights and MLPs remain
frozen and absent. No final source was opened, and no95% task or speed claim is
established. The next training decision must address the failed semantic
components rather than select a model by token agreement alone.
""")
    print(json.dumps({"classification": classification, "semantic_results": results, "same_case_joint": joint,
        "qualified_codecs": qualified, "canonical_firsts": len(cache), "campaign_sha256": file_hash(path)}), flush=True)


def register():
    campaign = run.q.read(run.RESULTS / "campaign_results.json")
    register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Native NCM CLS padding and source-grounded semantic reconstruction",
        "description": "Matched CLS-tail intervention on all690 targets and four anonymous reference/BERT/PAD0/CLS arms on fixed64 TRAIN sources; qualified first judges and same-case joint gates; no source-only task or speed claim.",
        "classification": campaign["classification"], "semantic_component_quality_measured": True,
        "source_only_task_accuracy_measured": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    run.configure()
    (finalize if args.stage == "finalize" else register)()
