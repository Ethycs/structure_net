"""Archive the final isolated teacher endpoint without declaring NCM success."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_isolated_quality_eval as study
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, PREP, verify
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_isolated_quality_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-isolated-native-quality-teacher.md")


def replay_role(cases, role, cache, endpoint):
    import tiktoken
    root = study.GRADE / (role + "_summary")
    verify(study.old.run.q.read(root / "producer.json"))
    scores = study.old.run.q.read(root / "case_scores.json")
    packets = study.old.run.q.read(root / "packets.json")
    if len(scores) != 64 or len(packets) != 64 or not endpoint["quality_teacher_control"] or endpoint["NCM_model"]:
        raise ValueError("complete teacher-control endpoint required")
    encoding = tiktoken.get_encoding("cl100k_base")
    for case, score, packet in zip(cases, scores, packets):
        expected = study.old.run.q.packet(case["id"], case["source"], [{"candidate_id": "summary", "text": case["text"]}])
        entry = cache[study.old.first.packet_key(expected, role)]
        study.old.first.audit_row(expected, role, entry["row"])
        if packet != expected or score != study.old.score_case(case, expected, entry, encoding):
            raise ValueError("frozen source/summary/first decision changed")
    measured = study.old.summarize_scores(scores)
    if any(endpoint[k] != v for k, v in measured.items()):
        raise ValueError("whole-summary aggregate differs")
    return measured, scores


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh teacher finalization required")
    roots = (study.training.TRAIN, study.PILOT, study.OUTPUTS, study.GRADE)
    for root in roots:
        study.training.data.terminal(root)
    study.old.dependencies(); study.old.profile.install()
    trained = study.old.run.q.read(study.training.TRAIN / "result.json")
    generated = study.old.run.q.read(study.OUTPUTS / "result.json")
    grading = study.old.run.q.read(study.GRADE / "result.json")
    if (trained["updates"] != 512 or trained["cases"] != 252 or not trained["only_word_bank_trained"]
            or not trained["original_body_forecaster_books_unchanged"]
            or file_hash(study.training.TRAIN / "model.pt") != trained["checkpoint_sha256"]):
        raise ValueError("fixed final512 quality training checkpoint required")
    equivalence = study.old.run.q.read(study.PILOT / "equivalence.json")
    if not all(equivalence[k] for k in ("initial_logits_exact", "all_word_states_exact", "decoded_outputs_exact", "checkpoint_reload_exact")):
        raise ValueError("plain/static/reload equivalence pilot required")
    cases = study.old.run.q.read(study.OUTPUTS / "outputs.json")
    development = study.old.run.q.read(PREP / "development_sources.json")
    study.require_generated(cases, 64)
    identity = ("id", "source_id", "group_id", "source_sha256", "source")
    if (len(development) != 64 or cases[:2] != study.old.run.q.read(study.PILOT / "outputs.json")
            or any(tuple(a[k] for k in identity) != tuple(b[k] for k in identity) for a, b in zip(cases, development))
            or file_hash(study.OUTPUTS / "outputs.json") != generated["output_sha256"]):
        raise ValueError("fixed development source order or retained first outputs changed")
    study.training.data.check_disjoint(study.old.run.q.read(study.training.data.DEST / "cases.json"), cases)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(study.old.run.MODEL_PATH, local_files_only=True)
    config = study.old.run.q.read(study.old.run.MODEL_PATH / "generation_config.json")
    configured = config.get("eos_token_id", [])
    stops = {tokenizer.eos_token_id, *([configured] if isinstance(configured, int) else configured)}
    traces = [r for p in sorted((study.OUTPUTS / "word_states").glob("*.json")) for r in study.old.run.q.read(p)]
    if [r["id"] for r in traces] != [c["id"] for c in cases]:
        raise ValueError("complete teacher word histories required")
    for case, trace in zip(cases, traces):
        states = trace["word_states"]
        if (len(states) != 128 or any(len(s) != 128 for s in states)
                or trace["initial_ids"] != [tokenizer.eos_token_id] * 128
                or states[-1] != case["raw_parallel_ids"] or case["query_forward_passes"] != 129
                or not case["quality_teacher_control"] or case["NCM_model"]
                or case["AR_fallback"] or case["external_verifier"]):
            raise ValueError("teacher word protocol differs")
        previous = trace["initial_ids"]; prefixes = []
        for state in states:
            prefixes.append(next((i for i, (a, b) in enumerate(zip(previous, state)) if a != b), 128))
            previous = state
        full = case["raw_parallel_ids"]
        end = next((i + 1 for i, token in enumerate(full) if token in stops), 128)
        if (prefixes != trace["matching_prefixes"] or case["native_output_ids"] != full[:end]
                or case["text"] != tokenizer.decode(full[:end], skip_special_tokens=True).strip()):
            raise ValueError("raw word prefix/first stop/text replay differs")
    prior = study.load_prior_cache()
    cache = study.old.run.q.read(study.GRADE / "judgment_cache.json")
    if any(cache.get(k) != v for k, v in prior.items()) or grading["prior_first_judgments_preserved"] != 256:
        raise ValueError("prior256 first judgments changed")
    components, scores_by_role = {}, {}
    components["secondary"], scores_by_role["secondary"] = replay_role(cases, "secondary", cache, grading["secondary"])
    if (grading["primary"] is not None) != study.old.quality_gate(grading["secondary"]):
        raise ValueError("independent primary judge gate order changed")
    if grading["primary"] is not None:
        components["primary"], scores_by_role["primary"] = replay_role(cases, "primary", cache, grading["primary"])
    passed = grading["primary"] is not None and study.old.quality_gate(grading["primary"])
    if grading["development_teacher_gate"] != passed or grading["NCM_model"] or grading["NCM_success_established"]:
        raise ValueError("teacher quality gate was mislabeled as NCM success")
    paths = [Path(__file__), Path(study.__file__), Path(study.training.__file__), study.training.data.PLAN,
        study.AMENDMENT, Path("experiments/neural_architecture_lab/ncm_result_registry.py"),
        study.PRIOR_CAMPAIGN, study.PRIOR_REGISTRY]
    for root in roots:
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    aggregate = {"schema": "nal.ncm-isolated-quality-teacher-results.v1", "classification": grading["classification"],
        "training": trained, "generation": generated, "grading": grading, "components": components,
        "raw_word_histories_and_first_judgments_replayed": True, "prior_256_judgments_preserved": True,
        "quality_teacher_control": True, "NCM_model": False, "NCM_success_established": False,
        "task_accuracy_measured": True, "confirmed": False, "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(paths)}}
    DEST.mkdir(); failures = {}
    for role, scores in scores_by_role.items():
        failures[role] = [{"id": c["id"], "source": c["source"], "summary": c["text"],
            "tokens": s["tokens"], "valid": s["valid"], "evaluation": s["evaluations"],
            "first_origin": s["first_origin"]} for c, s in zip(cases, scores) if s["passed"] is not True]
    write_json(DEST / "case_failures.json", failures)
    aggregate["artifacts"][str(DEST / "case_failures.json")] = file_hash(DEST / "case_failures.json")
    campaign = DEST / "campaign_results.json"; write_json(campaign, aggregate)
    secondary = grading["secondary"]
    table = []
    for role, endpoint in components.items():
        c = endpoint["components"]
        table.append(f"| {role} | {endpoint['passed']}/64 | {endpoint['valid']}/64 | {c['faithful']}/64 | {c['salient']}/64 | {c['readable']}/64 |")
    accuracy = f"{100 * secondary['task_accuracy']:.2f}%" if secondary["task_accuracy"] is not None else "unresolved due to invalid judgments"
    REPORT.write_text(f'''# Isolated native quality teacher: complete development evaluation

Date: 2026-10-23. Hypothesis `ncm-isolated-native-quality-teacher-v1`.
Classification **{grading['classification']}**. Aggregate `{campaign}`,
SHA256 `{file_hash(campaign)}`.

The clean native quality teacher scored **{secondary['passed']}/64 whole-summary
passes ({accuracy})** under qualified Terra. This is an isolated native-language
teacher control preparing NCM distillation, not a working NCM replacement or
goal completion. The original nine-pass NCM remains0/64; the converged shared
adapter hybrid remains41/64; unchanged3B and1B baselines remain52/64 and47/64.

| Role | Whole summary | Valid | Faithful | Salient | Readable |
| --- | --- | --- | --- | --- | --- |
''' + '\n'.join(table) + f'''

Whole-summary success jointly requires faithfulness, salience, readability,
nonempty output and at most128stored cl100k tokens. Terra found
{secondary['stored_overflow']} overflows and {secondary['empty']} empty outputs.
Its Wilson95% interval is `{secondary['wilson_95']}`. Independent Sonnet was
{'run on identical frozen texts' if grading['primary'] is not None else 'not admitted after the Terra gate failed'}.
The teacher development gate **{'passed' if passed else 'failed'}**. Every
failed case and cited judgment is retained in `{DEST / 'case_failures.json'}`.

Training used exactly512updates, batch one and accumulation eight, on252vetted
complete summaries:143old short labels plus109longer-source labels. Inputs
were complete, with a maximum2,276native prompt in a2,304-token frame. Saved
single-atom snippets were not used as if they summarized complete passages.
Only4,587,520fresh word-adapter parameters trained; original28blocks/MLPs,
embeddings, native head, old concept bank, heads, hints, biases and books stayed
frozen. Word-quality queries intentionally used no concept hint/bias. No coarse
loss, noisy-word loss, raw-language replay or intermediate checkpoint selection.

The actual longest-input cache/gradient pilots passed before full training.
Their earlier rank2-ID/rank3-embedding validation bug and correctedv2runs
remain separate preserved artifacts. Full training started from fresh zero-B
word adapters, not pilot weights. Frozen parameters/books, all CPU source cache
values and exact optimizer/RNG/checkpoint reload passed integrity checks.

After training, the first two development outputs from the plain implementation
were retained exactly. Static word-adapter inference had identical initial
full-vocabulary logits and all128word histories, including after checkpoint
reload. Its only implementation change avoids repeated GPU zero checks for
unchanged matrices; no speed measurement was made. All64outputs used128fixed
causal word updates and one exact fixed-point check, with EOS initialization.
No oracle plan, gold history, external verifier, repair or alternative selection.

All256prior first judgments, including negatives, were audited and preserved.
Generation terminated before judging. Complete histories, source identities,
token counts and raw first-valid decisions were replayed offline. Development
sources are disjoint from training by ID/session/group/hash, but are reused
development cases and short fragments. Original256and newer64final sources
remain sealed. A teacher gate does not establish final/full-range NCM accuracy.
Later acceleration training must use the qualified native teacher's own fixed
points/histories and retain complete-summary quality checks. No compression,
latency, tokens/s improvement or NCM success is established here.
''')
    print(json.dumps({"status": "finalized", "classification": aggregate["classification"], "secondary": secondary}), flush=True)


def register():
    aggregate = study.old.run.q.read(DEST / "campaign_results.json")
    endpoint = aggregate["grading"]["secondary"]
    register_records(DEST / "campaign_results.json", REPORT, DEST, {"ncm-isolated-native-quality-teacher-v1": {
        "name": "Independent clean native word teacher complete-summary development quality",
        "description": f"Fresh rank8 wordbank over frozen28blockLlama3B and frozenconceptbank,512cleanupdates on252vetted short/longsummaries. Terra{endpoint['passed']}/64wholepasses,{endpoint['valid']}valid. OriginalNCM0/64 andconverged41/64preserved. Teachercontrol, notNCMreplacement; finaltestsclosed, nospeedclaim.",
        "confirmation_status": aggregate["classification"], "task_accuracy_measured": True,
        "Terra_passed": endpoint["passed"], "Terra_cases": 64, "Terra_valid": endpoint["valid"],
        "quality_teacher_control": True, "NCM_model": False, "NCM_success_established": False,
        "development_teacher_gate": aggregate["grading"]["development_teacher_gate"]}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
