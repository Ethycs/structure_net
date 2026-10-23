"""Replay first-judgment semantic audit and register only its oracle scope."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_oracle_codec_semantic as study
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = study.ROOT / "20261023_oracle_codec_semantic_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-oracle-codec-semantic-quality.md")


def paired(a, b):
    if [r["id"] for r in a] != [r["id"] for r in b] or len(a) != 64:
        raise ValueError("all64 matched sources required")
    return {"both_pass": sum(x["passed"] is True and y["passed"] is True for x, y in zip(a, b)),
        "fixed": sum(x["passed"] is False and y["passed"] is True for x, y in zip(a, b)),
        "regressed": sum(x["passed"] is True and y["passed"] is False for x, y in zip(a, b)),
        "both_fail": sum(x["passed"] is False and y["passed"] is False for x, y in zip(a, b)),
        "invalid_pairs": sum(not x["valid"] or not y["valid"] for x, y in zip(a, b))}


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("append-only finalization requires a fresh destination")
    study.teacher.old.dependencies()
    study.teacher.old.profile.install()
    cases, packets = study.check_prepared()
    for root in (study.PILOT, study.GRADE):
        study.teacher.training.data.terminal(root)
    result = study.q.read(study.GRADE / "result.json")
    cache = study.q.read(study.GRADE / "judgment_cache.json")
    previous = study.q.read(study.teacher.GRADE / "judgment_cache.json")
    if len(previous) != 320 or any(cache.get(k) != v for k, v in previous.items()):
        raise ValueError("all320 previous judgments must remain exact")
    for key, entry in cache.items():
        if key != study.first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("cached identity differs")
        study.first.audit_row(entry["packet"], entry["role"], entry["row"])
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    scores = {}
    for role in ("secondary", "primary"):
        for arm in study.ARMS:
            observed = result["results"][arm][role]
            admitted = role == "secondary" or (study.qualifies(result["results"]["uncompressed"]["secondary"])
                and study.qualifies(result["results"][arm]["secondary"]))
            if (observed is not None) != admitted:
                raise ValueError("conditional judge admission differs from plan")
            if not admitted:
                continue
            rows = study.q.read(study.GRADE / f"{role}_{arm}_case_scores.json")
            if len(rows) != 64 or [r["id"] for r in rows] != [r["id"] for r in cases]:
                raise ValueError("all64 source identities required")
            for c, packet, r in zip(cases, packets[arm], rows):
                key = study.first.packet_key(packet, role)
                entry = cache[key]
                raw = entry["row"]
                valid = raw["status"] == "valid"
                text = c["texts"][arm]
                tokens = len(encoding.encode(text, disallowed_special=()))
                expected = {"id": c["id"], "text": text, "tokens": tokens, "valid": valid,
                    "passed": study.q.passes(raw["evaluations"][0], text, tokens) if valid else None,
                    "evaluations": raw.get("evaluations"), "packet_key": key,
                    "attempts": raw["attempts"], "first_origin": entry["origin"]}
                if r != expected:
                    raise ValueError("case score differs from raw first judgment")
            replayed = study.metric(rows)
            if any(observed[k] != value for k, value in replayed.items()):
                raise ValueError("all-case semantic aggregate differs")
            scores[role + "_" + arm] = rows
    qualified = [arm for arm in ("original", "trained") if all(
        result["results"][a]["primary"] is not None and study.qualifies(result["results"][a]["primary"])
        for a in (arm, "uncompressed"))]
    expected_class = "oracle_semantic_codec_qualified_conditional_generation_unproven" if qualified else "oracle_semantic_codec_quality_failed"
    if (result["semantically_qualified_codecs"] != qualified or result["classification"] != expected_class
            or result["task_accuracy"] is not None or result["source_prediction"] or result["NCM_success_established"]):
        raise ValueError("oracle scope or classification changed")
    paired_results = {"released_to_trained": paired(scores["secondary_original"], scores["secondary_trained"])}
    for arm in ("original", "trained"):
        paired_results["uncompressed_to_" + arm] = paired(scores["secondary_uncompressed"], scores["secondary_" + arm])
    paths = [Path(__file__), Path(study.__file__), study.PLAN,
        Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in (study.PREP, study.PILOT, study.GRADE):
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    for entry in cache.values():
        paths += [Path(p) for p in entry["row"]["attempts"]]
    paths += [study.CODEC_RESULTS / "campaign_results.json", study.TEACHER_RESULTS / "campaign_results.json"]
    aggregate = {"schema": "nal.ncm-oracle-codec-semantic-quality.v1", "classification": expected_class,
        "result": result, "paired_secondary": paired_results, "all_first_judgments_replayed": True,
        "all320_previous_judgments_preserved": True, "oracle_target_codes": True,
        "source_prediction": False, "task_accuracy_measured": False, "confirmed": False,
        "NCM_success_established": False, "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(paths)}}
    DEST.mkdir()
    failures = {key: [r for r in rows if r["passed"] is not True] for key, rows in scores.items()}
    write_json(DEST / "semantic_failures.json", failures)
    aggregate["artifacts"][str(DEST / "semantic_failures.json")] = file_hash(DEST / "semantic_failures.json")
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    table = []
    for arm in study.ARMS:
        for role in ("secondary", "primary"):
            m = result["results"][arm][role]
            if m is None:
                continue
            table.append(f"| {arm} | {role} | {m['passed']}/64 | {m['valid']}/64 | {m['components']['faithful']} | {m['components']['salient']} | {m['components']['readable']} | {m['stored_overflow']} |")
    REPORT.write_text(f"""# Semantic quality of oracle codec reconstruction

Date: 2026-10-23. Hypothesis `{study.HYPOTHESIS}`.
Classification **{expected_class}**. Aggregate `{campaign}`,
SHA256 `{file_hash(campaign)}`.

| Arm | Judge role | Complete-summary passes | Valid | Faithful | Salient | Readable | Over 128 tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
""" + "\n".join(table) + f"""

These are semantic reconstruction pass rates with correct target-derived codes,
**not source-to-summary model accuracy**. The uncompressed target is a control.
Source-only conditional generation remains unproven. The model-development
teacher result remains 56/64 (87.5%), and the earlier NCM hybrid 41/64. Neither
final cohort was opened. No speedup was measured.

The sample contains 64 of 252 codec-excluded complete summaries, selected by
case-ID hash without output or judgment filtering. Sources span the training
label pool, including longer passages. Both codecs reconstruct complete targets
through four 48-token blocks. Each anonymous single-candidate packet contains
the full SHA-verified public source and one summary. Both qualified judges use
the same 128-token storage rubric. Terra is secondary; Sonnet is primary and
is admitted only when both the arm and the uncompressed control reach 61/64
valid Terra passes. Qualification requires the same threshold on both judges.
Qualified codecs: {', '.join(qualified) if qualified else 'none'}.

On paired Terra cases, changing released to adapted reconstruction fixed
{paired_results['released_to_trained']['fixed']} cases and regressed
{paired_results['released_to_trained']['regressed']}; both passed on
{paired_results['released_to_trained']['both_pass']} and both failed on
{paired_results['released_to_trained']['both_fail']}.
Invalid pairs: {paired_results['released_to_trained']['invalid_pairs']}.

All first valid raw judgments, including negatives, replayed exactly. The
12 arm-case systems pilot contributes its unchanged first judgments. All 320
prior first judgments remain exact. Transport/schema retries alone are allowed,
with a maximum of three attempts. The failed 99.5% literal-reconstruction
gates remain failed; this prospective semantic audit does not amend them.
Raw failures and claim verdicts are retained in `semantic_failures.json`.
""")
    print(json.dumps({"status": "finalized", "classification": expected_class, "qualified": qualified,
                      "paired_secondary": paired_results}), flush=True)


def register():
    x = json.loads((DEST / "campaign_results.json").read_text())
    spec = {"name": "Source-grounded semantic quality of oracle NCM codec reconstructions",
        "description": "Hash-selected64 excluded summary targets; uncompressed, released-codec and sample-scaled-codec arms judged against complete public sources. Correct target codes supplied; this is semantic reconstruction, not source-to-summary accuracy. All prior320 first judgments preserved.",
        "confirmation_status": x["classification"], "task_accuracy_measured": False,
        "oracle_target_codes": True, "source_prediction": False, "NCM_success_established": False,
        "semantically_qualified_codecs": ",".join(x["result"]["semantically_qualified_codecs"]) or "none"}
    for arm, roles in x["result"]["results"].items():
        for role, m in roles.items():
            if m is not None:
                spec.update({arm + "_" + role + "_" + k: m[k] for k in ("cases", "passed", "valid", "invalid")})
    register_records(DEST / "campaign_results.json", REPORT, DEST, {study.HYPOTHESIS: spec})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
