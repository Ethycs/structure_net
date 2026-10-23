"""Replay complete converged summary evidence and register its actual outcome."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_converged_native as study
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, PREP, verify
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_converged_native_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-converged-native-development-quality.md")
SPLIT_AUDIT = ROOT / "20261023_converged_native_split_audit_v1.json"
SCORE_REPLAY = ROOT / "20261023_converged_native_score_replay_v1.json"


def replay_role(cases, role, cache, expected):
    import tiktoken
    root = study.GRADE / (role + "_summary")
    verify(study.run.q.read(root / "producer.json"))
    scores = study.run.q.read(root / "case_scores.json")
    packets = study.run.q.read(root / "packets.json")
    if len(scores) != 64 or len(packets) != 64 or not expected["NCM_trained"]:
        raise ValueError("complete trained-model endpoint required")
    encoding = tiktoken.get_encoding("cl100k_base")
    for case, score, packet in zip(cases, scores, packets):
        correct_packet = study.run.q.packet(case["id"], case["source"], [{"candidate_id": "summary", "text": case["text"]}])
        if packet != correct_packet:
            raise ValueError("frozen output/source packet changed")
        entry = cache[study.first.packet_key(packet, role)]
        study.first.audit_row(packet, role, entry["row"])
        if score != study.score_case(case, packet, entry, encoding):
            raise ValueError("first judgment decision replay differs")
    actual = study.summarize_scores(scores)
    if any(expected[k] != v for k, v in actual.items()):
        raise ValueError("all-case endpoint aggregate differs")
    return actual


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh finalization required")
    for root in (study.PILOT, study.OUTPUTS, study.GRADE):
        study.terminal(root)
    study.dependencies(); study.profile.install()
    generated = study.run.q.read(study.OUTPUTS / "result.json")
    grading = study.run.q.read(study.GRADE / "result.json")
    cases = study.run.q.read(study.OUTPUTS / "outputs.json")
    original = study.run.q.read(PREP / "development_sources.json")
    failed = study.run.q.read(study.run.OUTPUTS / "outputs.json")
    study.require_generated(cases, 64)
    identity = ("id", "source_id", "group_id", "source_sha256", "source")
    if (file_hash(study.OUTPUTS / "outputs.json") != generated["output_sha256"]
            or len(original) != 64 or len(failed) != 64
            or any(tuple(a[k] for k in identity) != tuple(b[k] for k in identity) for a, b in zip(cases, original))
            or cases[:2] != study.run.q.read(study.PILOT / "outputs.json")):
        raise ValueError("original cohort, frozen outputs or retained pilot changed")
    old = {c["id"]: c for c in failed}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(study.run.MODEL_PATH, local_files_only=True)
    config = study.run.q.read(study.run.MODEL_PATH / "generation_config.json")
    configured = config.get("eos_token_id", [])
    stops = {tokenizer.eos_token_id, *([configured] if isinstance(configured, int) else configured)}
    traces = [row for p in sorted((study.OUTPUTS / "word_states").glob("*.json")) for row in study.run.q.read(p)]
    if [r["id"] for r in traces] != [c["id"] for c in cases]:
        raise ValueError("all64 word histories required in source order")
    for case, trace in zip(cases, traces):
        states = trace["word_states"]
        if (len(states) != 128 or any(len(s) != 128 for s in states)
                or states[3] != old[case["id"]]["raw_parallel_ids"] or case["codes"] != old[case["id"]]["codes"]
                or states[-1] != case["raw_parallel_ids"] or case["word_updates"] != 128
                or case["verification_passes"] != 1 or case["fixed_forward_passes"] != 134
                or case["source_truncated"] or case["AR_fallback"] or case["external_verifier"]):
            raise ValueError("complete source-only128-word convergence history changed")
        previous = trace["initial_ids"]
        prefixes = []
        for state in states:
            prefixes.append(next((i for i, (a, b) in enumerate(zip(previous, state)) if a != b), 128))
            previous = state
        if trace["matching_prefixes"] != prefixes:
            raise ValueError("word prefix trace replay differs")
        full = case["raw_parallel_ids"]
        end = next((i + 1 for i, token in enumerate(full) if token in stops), 128)
        if case["native_output_ids"] != full[:end] or case["text"] != tokenizer.decode(full[:end], skip_special_tokens=True).strip():
            raise ValueError("first stop or final decoded text differs")
    prior = study.load_prior_cache()
    split_audit = study.run.q.read(SPLIT_AUDIT)
    training = [c for rows in study.run.q.read(study.run.data.DEST / "cases.json").values() for c in rows]
    overlaps = {key: len({c[key] for c in training} & {c[key] for c in original})
                for key in ("id", "source_id", "group_id", "source_sha256")}
    if (any(overlaps.values()) or split_audit["overlap_counts"] != overlaps
            or split_audit["training_manifest_sha256"] != file_hash(study.run.data.DEST / "cases.json")
            or split_audit["development_sources_sha256"] != file_hash(PREP / "development_sources.json")):
        raise ValueError("case/session/group/source-hash disjointness audit differs")
    scorer_replay = study.run.q.read(SCORE_REPLAY)
    if (scorer_replay["source_sha256"] != file_hash(study.__file__)
            or scorer_replay["plan_sha256"] != file_hash(study.PLAN)
            or scorer_replay["prior_first_judgments"] != len(prior)
            or scorer_replay["prior_cache_keys_sha256"] != study.run.q.digest(sorted(prior))
            or scorer_replay["API_requests"] != 0
            or scorer_replay["original_failed_outputs_replayed"]["passed"] != 0):
        raise ValueError("new scorer's offline old-endpoint replay differs")
    cache = study.run.q.read(study.GRADE / "judgment_cache.json")
    if any(cache.get(k) != v for k, v in prior.items()) or grading["prior_first_judgments_preserved"] != 192:
        raise ValueError("prior192 first judgments changed")
    components = {"secondary": replay_role(cases, "secondary", cache, grading["secondary"])}
    if (grading["primary"] is not None) != study.quality_gate(grading["secondary"]):
        raise ValueError("independent judge gate order changed")
    if grading["primary"] is not None:
        components["primary"] = replay_role(cases, "primary", cache, grading["primary"])
    passed = grading["primary"] is not None and study.quality_gate(grading["primary"])
    if grading["development_gate"] != passed:
        raise ValueError("joint quality gate differs")
    paths = [Path(__file__), Path(study.__file__), study.PLAN,
        Path("experiments/neural_architecture_lab/ncm_result_registry.py"),
        study.PRIOR_CAMPAIGN, study.PRIOR_REGISTRY, SPLIT_AUDIT, SCORE_REPLAY,
        ROOT / "20261023_budget128_baselines_results_v1/campaign_results.json"]
    for root in (study.PILOT, study.OUTPUTS, study.GRADE):
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    aggregate = {"schema": "nal.ncm-converged-native-results.v1", "classification": grading["classification"],
        "generation": generated, "grading": grading, "components": components,
        "task_accuracy_measured": True, "prior_192_first_judgments_preserved": True,
        "split_audit": split_audit, "old_endpoint_scorer_replay": scorer_replay,
        "all_raw_word_states_and_decisions_replayed": True, "original_nine_pass_score": {"passed": 0, "cases": 64},
        "development_reused_not_final": True, "source_scope": "original short-source64development, not full application range",
        "NCM_trained": True, "NCM_causal_hybrid": True, "NCM_success_established": False,
        "confirmed": False, "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(paths)}}
    DEST.mkdir()
    failures = {}
    for role in components:
        scores = study.run.q.read(study.GRADE / (role + "_summary") / "case_scores.json")
        failures[role] = []
        for case, score in zip(cases, scores):
            if score["passed"] is True:
                continue
            evaluation = score["evaluations"][0] if score["valid"] else None
            failed = [k for k in ("faithful", "salient", "readable") if evaluation and not evaluation[k]]
            failed += (["stored_overflow"] if score["tokens"] > 128 else [])
            failed += (["empty"] if not case["text"].strip() else [])
            failed += (["invalid_judgment"] if not score["valid"] else [])
            failures[role].append({"id": case["id"], "source_id": case["source_id"],
                "source": case["source"], "summary": case["text"], "tokens": score["tokens"],
                "failed_components": failed, "evaluation": evaluation,
                "first_origin": score["first_origin"], "attempts": score["attempts"]})
    failure_path = DEST / "case_failures.json"
    write_json(failure_path, failures)
    aggregate["failure_case_counts"] = {role: len(rows) for role, rows in failures.items()}
    aggregate["artifacts"][str(failure_path)] = file_hash(failure_path)
    campaign = DEST / "campaign_results.json"; write_json(campaign, aggregate)
    secondary = grading["secondary"]
    rows = []
    for role, value in components.items():
        c = value["components"]
        rows.append(f"| {role} | {value['passed']}/64 | {value['valid']}/64 | {c['faithful']}/64 | {c['salient']}/64 | {c['readable']}/64 |")
    accuracy = f"{100*secondary['task_accuracy']:.2f}%" if secondary["task_accuracy"] is not None else "unresolved because some judgments are invalid"
    REPORT.write_text(f'''# Converged frozen-native summary development evaluation

Date: 2026-10-23. Hypothesis `ncm-converged-native-summary-quality-v1`.
Classification **{grading['classification']}**. Aggregate `{campaign}`,
SHA256 `{file_hash(campaign)}`.

The converged hybrid scored **{secondary['passed']}/64 whole-summary passes
({accuracy})** under qualified Terra. All 64 original development cases remain
in the denominator. The original fixed-nine-pass model remains a separate
0/64 failure; its checkpoint and all original judgments are unchanged.

| Judge role | Whole summary | Valid | Faithful | Salient | Readable |
| --- | --- | --- | --- | --- | --- |
''' + '\n'.join(rows) + f'''

Whole-summary success requires faithfulness, salience, readability and at most
128 stored cl100k tokens jointly. The Terra result contains
{secondary['stored_overflow']} stored overflows and {secondary['empty']} empty
summaries. Its Wilson 95% interval is `{secondary['wilson_95']}`. Independent
Sonnet grading was {'executed on the identical frozen texts' if grading['primary'] is not None else 'not admitted because Terra did not reach 64 valid judgments and 61 passing summaries'}.
The joint development gate **{'passed' if passed else 'failed'}**.
Every failed case, source text, component failure, reason and cited claim
evaluation is retained in `{failure_path}`. This includes invalid judgments
and format failures; no failed case is removed from the denominator.

This uses the original 2,048-update model, not the later four-case capacity
control. Original embeddings, all 28 Llama3B body blocks including MLPs, native
word head and concept books stayed frozen. Source-only five-level predictions
conditioned 128 whole-frame word updates and one additional verification pass.
All full word frames reached an exact verified fixed point; every four-update
snapshot exactly reproduced the original failed outputs and concept codes.
There was no oracle plan, gold history, external verifier, AR fallback, new
training, output repair or choice between alternatives.

The two-case inference/reload pilot completed before the remaining 62 cases;
its first outputs and all word states were retained exactly. Generation ended
before judging. Both qualified 128-token judge profiles were checked. All 192
earlier first judgments, including negatives, were audited and preserved.
Raw requests, first-valid response parsing, token counts, complete histories,
source identities and whole-summary decisions were replayed offline. New
endpoints correctly identify NCM_trained=true. All workers terminated normally.
An independent metadata audit found no overlap by case ID, source session,
group or exact source hash against all 1,231 stored training/diagnostic cases.
The new scorer also reproduced the old0/64 endpoint and its component counts
without making API requests. These controls are separate from new model quality.

This is a new causal-decoding hybrid endpoint, not a reproduction of released
NCM training and not a repair of the original nine-pass result. It uses 134
body/query passes per frame including verification. No throughput, latency,
compression or speed benefit is measured. Unchanged AR baselines under the
same 128-token rubric previously scored 52/64 for 3B and 47/64 for 1B.

The cohort is source-group disjoint from training but has been reused for
development; inputs are short fragments. The original 256 final cases and new
64 longer-source final cases remain sealed. Neither a passing development point
estimate nor numerical convergence alone establishes 95% final application
accuracy. Full input-range and final evaluation remain required for the goal.
''')
    print(json.dumps({"status": "finalized", "classification": aggregate["classification"], "secondary": secondary}), flush=True)


def register():
    aggregate = study.run.q.read(DEST / "campaign_results.json")
    endpoint = aggregate["grading"]["secondary"]
    register_records(DEST / "campaign_results.json", REPORT, DEST, {"ncm-converged-native-summary-quality-v1": {
        "name": "Complete converged frozen-native hybrid summary development quality",
        "description": f"Original2048-update frozen28-block3B NCM, five source-only concept levels then128wordupdates plusverification. QualifiedTerra scored{endpoint['passed']}/64 complete summaries with{endpoint['valid']}valid; same original64source-disjoint development cases. Prior192judgments and original0/64 fixed-nine-pass failure preserved. No finaltest or speedclaim.",
        "confirmation_status": aggregate["classification"], "task_accuracy_measured": True,
        "Terra_passed": endpoint["passed"], "Terra_cases": 64, "Terra_valid": endpoint["valid"],
        "development_gate": aggregate["grading"]["development_gate"], "NCM_success_established": False}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
