"""Independent sealed320 semantic and original-body integration audit."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_clean_sealed_final as run
from experiments.neural_architecture_lab import finalize_ncm_sealed_final_quality as old
from experiments.neural_architecture_lab import finalize_ncm_clean_semantic_development as semantic
from experiments.neural_architecture_lab.finalize_ncm_clean_development_features import check_feature
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def decline():
    if run.admission_state() is not False or run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("only an observed unqualified development outcome can decline final access")
    run.sealed_metadata()
    if (run.WORKFLOW / "final_access_intent.json").exists() or any(p.exists() for p in
            (run.DATA, run.FEATURE_PILOT, run.FEATURES, run.PILOT, run.BODY_REPLAY, run.HEAD_REPLAY, run.OUTPUTS, run.PACKETS, run.GRADE)):
        raise ValueError("no final content or numerical stages may precede admission")
    paths = [run.WORKFLOW / "producer.json", run.development.RESULTS / "campaign_results.json",
        run.development.RESULTS / (run.development.HYPOTHESIS + "_registry_readback.json"),
        run.partitions.DEST / "seal_receipts.json", run.partitions.DEST / "selection_metadata.json"]
    campaign = {"schema": "nal.ncm-clean-sealed-final.v1", "hypothesis_id": run.HYPOTHESIS, "classification": "development_not_qualified_final_sources_unopened",
        "goal_target_met": False, "final_test_opened": False, "task_accuracy": None,
        "final_cases_evaluated": 0, "model_training_executed": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in paths}}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    run.REPORT.write_text(f"""# Final NCM evaluation was not admitted

Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`.
Classification **development_not_qualified_final_sources_unopened**.
Campaign `{path}`; SHA256 `{file_hash(path)}`.

The fixed candidate did not qualify on fresh semantic development. No final
source file was opened or rehashed and no final case was evaluated. The256 short
and64 long sources remain sealed. Final accuracy is unmeasured; the95% goal
remains unmet. This conditional decision does not overwrite earlier failures.
""")


def replay_sources():
    parts = run.final_rows()
    rows = [r for values in parts.values() for r in values]
    run.final_exclusion(rows)
    proof = run.base.verify_public(rows)
    native, gpt, storage = run.data.tokenizers()
    expected = {split: [run.data.prepare_row(r, split, native, gpt, storage) for r in values] for split, values in parts.items()}
    cases = run.sources()
    if cases != expected or run.q.read(run.DATA / "public_proof.json") != {
            "public_file_sha256": run.base.PUBLIC_SHA256, "exact_sources": proof}:
        raise ValueError("literal final sources, provenance, tokenization and exclusion replay differ")
    first = cases["short_final"][0]
    others = [r for values in cases.values() for r in values if r["id"] != first["id"]]
    selected = {"feature_pilot_ids": [first["id"], min(others, key=lambda r: (-len(r["context_ids"]), r["id"]))["id"]],
        "inference_pilot_ids": [first["id"], min(others, key=lambda r: (-len(r["source_ids"]), r["id"]))["id"]]}
    if run.q.read(run.DATA / "selection.json") != selected:
        raise ValueError("fixed first/maximum frame pilot identities changed")
    return cases


def finalize():
    run.require_admission()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh final aggregate/report required")
    roots = (run.DATA, run.FEATURE_PILOT, run.FEATURES, run.PILOT, run.BODY_REPLAY,
        run.HEAD_REPLAY, run.OUTPUTS, run.PACKETS, run.GRADE)
    for root in roots:
        run.base.terminal(root)
    stages = ("prepare", "feature-pilot", "features", "pilot", "body-replay", "head-replay", "generate", "packets", "grade")
    receipts = [run.WORKFLOW / "stage_receipts" / (s + ".json") for s in stages]
    if any(run.q.read(p)["exit_code"] or run.q.read(p)["observer"] != "subprocess.wait" for p in receipts):
        raise ValueError("nine actual successful final numerical/grading children required")
    cases = replay_sources()
    rows = [r for values in cases.values() for r in values]
    by_id = {r["id"]: r for r in rows}
    full = run.q.read(run.FEATURES / "manifest.json")
    pilot = run.q.read(run.FEATURE_PILOT / "manifest.json")
    selection = run.q.read(run.DATA / "selection.json")
    if set(full) != set(by_id) or set(pilot) != set(selection["feature_pilot_ids"]) or any(full[k] != v for k, v in pilot.items()):
        raise ValueError("all320 feature cases and exact retained pilot files required")
    feature_paths = []
    for row in rows:
        root = run.FEATURE_PILOT if row["id"] in pilot else run.FEATURES
        path = root / "features" / (row["id"] + ".pt")
        check_feature(row, full[row["id"]], path)
        feature_paths.append(path)
    for root, count in ((run.FEATURE_PILOT, 2), (run.FEATURES, 320)):
        result = run.q.read(root / "result.json")
        if (result["cases"] != count or result["parent_body_hash_unchanged"] != run.base.EXPECTED_BODY
                or not result["all_source_tokens_retained"] or result["peak_allocated_bytes"] > 8_600_000_000
                or result["manifest_sha256"] != file_hash(root / "manifest.json")):
            raise ValueError("original full-body feature identity or memory evidence differs")
    body = run.q.read(run.BODY_REPLAY / "manifest.json")
    body_result = run.q.read(run.BODY_REPLAY / "result.json")
    if (set(body) != set(selection["inference_pilot_ids"]) or body_result["status"] != "original_body_replay_exact"
            or body_result["original_body_and_MLPs_sha256"] != run.base.EXPECTED_BODY
            or not body_result["cached_features_bit_exact"] or body_result["peak_allocated_bytes"] > 8_600_000_000
            or body_result["manifest_sha256"] != file_hash(run.BODY_REPLAY / "manifest.json")):
        raise ValueError("fresh complete original-body integration evidence differs")
    for key, entry in body.items():
        path = run.BODY_REPLAY / "features" / (key + ".pt")
        check_feature(by_id[key], entry, path)
        fresh = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["features"]
        cached = torch.load(full[key]["path"], map_location="cpu", weights_only=True, mmap=True)["features"]
        if not torch.equal(fresh, cached):
            raise ValueError("actual replayed original-body tensors differ from final cache")
        feature_paths.append(path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(run.base.GPT2, local_files_only=True)
    outputs = run.q.read(run.OUTPUTS / "outputs.json")
    semantic.replay_outputs(outputs, rows, tokenizer)
    pilot_outputs = run.q.read(run.PILOT / "outputs.json")
    semantic.replay_outputs(pilot_outputs, run.selected_rows(cases), tokenizer)
    indexed = {r["id"]: r for r in outputs}
    if any(indexed[r["id"]] != r for r in pilot_outputs):
        raise ValueError("all final inference pilot outputs must be retained exactly")
    train_rows, _, expected_train = run.development.training_parity_inputs()
    train_outputs = run.q.read(run.PILOT / "TRAIN_parity_outputs.json")
    semantic.replay_outputs(train_outputs, train_rows, tokenizer)
    run.development.parity_check(train_outputs, expected_train)
    if run.q.read(run.PILOT / "DEV_parity_outputs.json") != run.q.read(run.development.PILOT / "outputs.json"):
        raise ValueError("same selected model must retain registered DEV predictions")
    integration = run.q.read(run.HEAD_REPLAY / "outputs.json")
    result = run.q.read(run.HEAD_REPLAY / "result.json")
    if (integration != pilot_outputs or result["status"] != "original_body_to_NCM_integration_passed"
            or result["original_body_and_MLPs_sha256"] != run.base.EXPECTED_BODY or not result["outputs_bit_exact"]
            or result["outputs_sha256"] != file_hash(run.HEAD_REPLAY / "outputs.json")
            or result["peak_allocated_bytes"] > 8_000_000_000):
        raise ValueError("full original-body-to-NCM predictions must reproduce exact pilot outputs")
    models = run.q.read(run.WORKFLOW / "admission.json")["models"]
    if models != run.expected_models():
        raise ValueError("all actual selected model identities changed after development")
    for root, count in ((run.PILOT, 2), (run.HEAD_REPLAY, 2), (run.OUTPUTS, 320)):
        value = run.q.read(root / "result.json")
        if value["cases"] != count or value["models"] != models or value["peak_allocated_bytes"] > 8_000_000_000:
            raise ValueError("same fixed model and complete final inference evidence required")
    for root in (run.PILOT, run.OUTPUTS):
        value = run.q.read(root / "result.json")
        if (value["outputs_sha256"] != file_hash(root / "outputs.json") or not value["all_parameters_and_codec_unchanged"]
                or not value["fresh_reload_or_retained_pilot_exact"]):
            raise ValueError("actual unchanged-model complete output proof differs")
    packets = run.q.read(run.PACKETS / "packets.json")
    if packets != {split: [run.q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["text"]}])
            for r in outputs if r["split"] == split] for split in run.COUNTS}:
        raise ValueError("exact public single-candidate final packets differ")
    for name, digest in run.q.read(run.PACKETS / "result.json")["files"].items():
        if file_hash(run.PACKETS / name) != digest:
            raise ValueError("final public payload proof changed")
    old.final = run
    grade, failures, prior_count, total_count = old.replay_judgments(outputs)
    invalid = any(value["invalid"] for results in grade["results"].values()
        for value in (results["secondary"], results["primary"]) if value is not None)
    met = grade["quality_gate"] and not invalid
    classification = "clean_sealed_final_judgments_invalid" if invalid else "clean_sealed_final_target_met" if met else "clean_sealed_final_target_not_met"
    producer = run.q.read(run.WORKFLOW / "producer.json")
    run.base.verify(producer)
    artifacts = {str(p): file_hash(p) for p in [Path(__file__), Path(run.__file__), run.PLAN,
        run.WORKFLOW / "producer.json", run.WORKFLOW / "admission.json", run.WORKFLOW / "final_access_intent.json", *receipts, *feature_paths]}
    artifacts.update(producer["input_hashes"])
    for root in roots:
        artifacts.update(run.q.read(root / "producer.json")["input_hashes"])
        artifacts.update({str(p): file_hash(p) for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts})
    campaign = {"schema": "nal.ncm-clean-sealed-final.v1", "hypothesis_id": run.HYPOTHESIS, "classification": classification, "goal_target_met": met,
        "results": grade["results"], "cohort_counts": run.COUNTS, "source_disjoint": True,
        "all_sources_complete": True, "original_body_to_NCM_integration_passed": True,
        "original_body_and_MLPs_sha256": run.base.EXPECTED_BODY, "models": models,
        "model_weights_unchanged_since_development": True, "all_first_judgments_replayed": True,
        "prior_first_judgments_preserved": prior_count, "canonical_judgments": total_count,
        "final_test_opened": True, "human_rated": False, "confirmed": False, "underpowered": True,
        "paper_replication_established": False, "speed_tested": False, "artifacts": artifacts}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    write_json(run.RESULTS / "case_failures.json", failures)
    lines = []
    for split, results in grade["results"].items():
        for role in ("secondary", "primary"):
            endpoint = results[role]
            if endpoint is not None:
                point = f"{endpoint['passed']/endpoint['cases']:.2%}" if endpoint['valid'] == endpoint['cases'] else "unmeasured"
                lines.append(f"| {split} | {role} | {endpoint['passed']}/{endpoint['cases']} | {endpoint['valid']}/{endpoint['cases']} | {point} |")
        joint = results["consensus"]
        point = f"{joint['task_accuracy']:.2%}" if joint['task_accuracy'] is not None else "unmeasured"
        passes = f"{joint['joint_passed']}/{joint['cases']}" if joint['primary_admitted'] else "not admitted"
        lines.append(f"| {split} | same case joint | {passes} | {joint['valid_both']}/{joint['cases']} | {point} |")
    run.REPORT.write_text(f"""# Sealed final complete summary accuracy and body integration

Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`.
Classification **{classification}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`.

| Final cohort | Judge | Complete summary passes | Valid | Task accuracy |
| --- | --- | --- | --- | --- |
""" + "\n".join(lines) + f"""

The fixed95% task target {'passed' if met else 'did not pass'}. Success requires
the same244/256 short and61/64 long cases to pass both qualified model judges,
with every decision valid. These are held-out model-judge rates with retained
Wilson intervals, not human ratings or a95% population confidence lower bound.

All320 reserved cases retain their original identities, order, full text and
public spans. Literal source, tokenizer and actual known/DEV exclusion checks
replay independently. Final contents were accessed only after the fixed model
passed the complete registered same-case development gate. No model changed
after final admission. All prior failures remain recorded.

Original28-layer Llama body and MLP parameters remained frozen. Complete source
features were checked individually. A fresh body load recomputed both actual
inference-pilot feature tensors exactly; separately reloaded NCM heads reproduced
the same words, predicted coarse codes and latent hashes from those new tensors.
Registered TRAIN/DEV parity and fresh final reload passed. The full320 outputs
retained both inference pilots exactly. No target, repair, clipping, lookup,
autoregressive fallback or best-of choice entered prediction.

Every raw first judgment, source citation, token count, endpoint component and
joint decision replayed, preserving{prior_count} earlier firsts and{total_count}
canonical firsts in total. Negatives and invalids remain in their original
denominators. This is an NCM task adaptation, without a paper-replication or speed
claim. Any goal-completion claim still requires independent review of this
complete evidence and truthful final registry readback.
""")
    print(json.dumps({"classification": classification, "goal_target_met": met, "results": grade["results"],
        "campaign_sha256": file_hash(path)}), flush=True)


def register():
    campaign = run.q.read(run.RESULTS / "campaign_results.json")
    if not campaign["final_test_opened"]:
        register_records(run.RESULTS / "campaign_results.json", run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
            "name": "Conditional final NCM evaluation not admitted",
            "description": "Development quality did not qualify; all320 final sources remain unopened, no final accuracy measured.",
            "classification": campaign["classification"], "goal_target_met": False, "task_accuracy_measured": False}})
    else:
        old.final, old.DEST, old.REPORT = run, run.RESULTS, run.REPORT
        old.register()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("decline", "finalize", "register"))
    args = parser.parse_args()
    run.configure()
    decline() if args.stage == "decline" else finalize() if args.stage == "finalize" else register()
