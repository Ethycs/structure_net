"""Rebuild source-grounded task pass rates from raw independent judge replies."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import shutil

from experiments.structure_net import ncm_memory_accuracy as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def audit(root, packets, role):
    producer = q.read(root / "producer.json")
    result = q.read(root / "result.json")
    if q.digest({k: v for k, v in producer.items() if k != "fingerprint"}) != producer["fingerprint"]:
        raise ValueError("producer fingerprint mismatch")
    if result["producer_fingerprint"] != producer["fingerprint"]:
        raise ValueError("terminal fingerprint mismatch")
    for path, expected in {**producer["source_hashes"], **producer["input_hashes"]}.items():
        if file_hash(path) != expected:
            raise ValueError("producer source/input changed")
    copies = {file_hash(p) for p in (root / "producer_source").iterdir()}
    if not set(producer["source_hashes"].values()) <= copies:
        raise ValueError("missing source snapshot")
    rows = q.read(root / "judgments.json")
    if file_hash(root / "judgments.json") != result["judgments_sha256"]:
        raise ValueError("judgment checksum mismatch")
    expected = {p["case_id"]: p for p in packets}
    if len(rows) != len({r["case_id"] for r in rows}) or any(r["case_id"] not in expected for r in rows):
        raise ValueError("duplicate/unknown judge rows")
    for row in rows:
        p = expected[row["case_id"]]
        if row["packet_sha256"] != q.digest(p) or row["judge"] != q.JUDGES[role]:
            raise ValueError("packet/judge binding mismatch")
        payload = {"model": q.JUDGES[role], "temperature": 0, "max_tokens": 4096,
                   "messages": [{"role": "system", "content": q.RUBRIC},
                                {"role": "user", "content": json.dumps(p, ensure_ascii=False)}]}
        if not 1 <= len(row["attempts"]) <= 3:
            raise ValueError("attempt count")
        for i, path in enumerate(row["attempts"]):
            raw = q.read(path)
            if raw["attempt"] != i or raw["request"] != payload or raw["request_sha256"] != q.digest(payload):
                raise ValueError("request changed")
            if i < len(row["attempts"])-1 and raw["valid"]:
                raise ValueError("valid quality outcome retried")
        if row["status"] == "valid":
            if not raw["valid"] or raw["http_status"] != 200 or raw["response"]["model"] != q.JUDGES[role]:
                raise ValueError("transport/returned alias mismatch")
            rebuilt = q.parse(raw["response"]["choices"][0]["message"]["content"], p)
            if rebuilt != row["evaluations"]:
                raise ValueError("raw response reconstruction mismatch")
        elif raw["valid"]:
            raise ValueError("invalid row contradicts raw response")
    return rows, result


def cluster_ci(rows, draws=10000):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_id"]].append(int(row["passed"]))
    if not groups:
        return None
    grouped = [(sum(v), len(v)) for _, v in sorted(groups.items())]
    rng = random.Random(20261022)
    estimates = []
    for _ in range(draws):
        sampled = [grouped[rng.randrange(len(grouped))] for _ in grouped]
        estimates.append(sum(x[0] for x in sampled)/sum(x[1] for x in sampled))
    estimates.sort()
    return [estimates[int(draws*.025)], estimates[int(draws*.975)-1]]


def summary(rows):
    n = len(rows)
    k = sum(r["passed"] for r in rows)
    claims = [c for r in rows for c in r["grade"]["claims"]]
    return {"n": n, "passes": k, "task_pass_rate": k/n if n else None,
            "wilson95": q.wilson(k, n), "source_cluster_bootstrap95": cluster_ci(rows),
            "source_sessions": len({r["source_id"] for r in rows}),
            "fully_supported_summaries": sum(r["grade"]["faithful"] for r in rows),
            "salience_failures": sum(not r["grade"]["salient"] for r in rows),
            "readability_failures": sum(not r["grade"]["readable"] for r in rows),
            "empty_outputs": sum(not r["text"].strip() for r in rows),
            "over128_outputs": sum(r["tokens"] > 128 for r in rows),
            "material_claims": len(claims),
            "supported_claims": sum(c["verdict"] == "supported" for c in claims)}


def finalize(args):
    campaign_path = args.root / "campaign_results.json"
    if campaign_path.exists() or args.report.exists():
        raise ValueError("fresh aggregate/report required")
    expected = q.read(args.root / "calibration_expected.json")
    calibration = {}
    scores = {}
    stage_paths = []
    for role in ("primary", "secondary"):
        cal = args.root / ("calibration_"+role)
        rows, result = audit(cal, q.read(args.root / "calibration_packets.json"), role)
        stage_paths += [cal / "producer.json", cal / "result.json", cal / "judgments.json"]
        positives = negatives = accepted = rejected = 0
        for row in rows:
            if row["status"] != "valid":
                continue
            for grade in row["evaluations"]:
                truth = expected[grade["candidate_id"]]
                passed = q.passes(grade, truth["text"], truth["tokens"])
                if truth["expected_pass"]:
                    positives += 1
                    accepted += passed
                else:
                    negatives += 1
                    rejected += not passed
        qualified = positives == negatives == 24 and accepted >= 23 and rejected >= 23 and len(rows) == 12
        if (result["status"] == "passed") != qualified:
            raise ValueError("calibration gate recomputation mismatch")
        calibration[role] = {"judge": q.JUDGES[role], "qualified": qualified,
                             "positive_controls": positives, "positives_accepted": accepted,
                             "negative_controls": negatives, "negatives_rejected": rejected,
                             "pairs": len(rows), "invalid_pairs": result["invalid_pairs"]}
        if not qualified:
            if role == "primary":
                raise ValueError("primary judge qualification failed")
            continue
        stage = args.root / ("judge_"+role)
        concurrency = args.root / (role+"_concurrency") / "result.json"
        if role == "secondary" and concurrency.exists() and q.read(concurrency)["status"] != "passed":
            calibration[role]["audit_status"] = "not_run_concurrency_gate_failed"
            continue
        packets = q.selected_packets(args.root, role)
        rows, result = audit(stage, packets, role)
        if result["status"] != "completed" or len(rows) != len(packets) or result["invalid_pairs"]:
            raise ValueError("incomplete endpoint; do not exclude missing judgments")
        if q.read(stage / "packets.json") != packets:
            raise ValueError("saved main packets differ")
        lookup = q.read(args.root / "candidate_map.json")
        flat = []
        for row in rows:
            for grade in row["evaluations"]:
                candidate = lookup[grade["candidate_id"]]
                flat.append({**candidate, "candidate_id": grade["candidate_id"], "grade": grade,
                             "passed": q.passes(grade, candidate["text"], candidate["tokens"])})
        if len(flat) != 3*len(packets) or len({r["candidate_id"] for r in flat}) != len(flat):
            raise ValueError("candidate accounting differs")
        scores[role] = flat
        stage_paths += [stage / name for name in ("producer.json", "result.json", "judgments.json", "packets.json")]
    summaries = {model: summary([r for r in scores["primary"] if r["model"] == model]) for model in q.MODELS}
    agreement = {"status": "unavailable", "reason": "secondary judge qualification or concurrency gate failed"}
    if "secondary" in scores:
        first = {r["candidate_id"]: r for r in scores["primary"]}
        agreement = {model: {"cases": len([r for r in scores["secondary"] if r["model"] == model]),
                    "same_pass_fail": sum(r["passed"] == first[r["candidate_id"]]["passed"]
                                          for r in scores["secondary"] if r["model"] == model),
                    "secondary_passes": sum(r["passed"] for r in scores["secondary"] if r["model"] == model)}
                     for model in q.MODELS}
    trained = summaries["trained_ncm"]
    original = summaries["original_ncm"]
    point_met = trained["passes"] >= 244
    write_json(args.root / "case_scores.json", scores)
    artifacts = [p for p in args.root.glob("*.json") if p != campaign_path] + stage_paths
    source = Path(__file__)
    copy = args.root / "analysis_source"
    copy.mkdir()
    shutil.copyfile(source, copy / source.name)
    result = {"schema": "nal.ncm-memory-task-accuracy-campaign.v1", "hypothesis_id": q.HYPOTHESIS,
              "status": "completed_exploratory", "classification": "ncm_memory_task_accuracy_point_target_met_judged_only"
              if point_met else "ncm_memory_task_accuracy_below_95", "models": summaries,
              "measure": "Source-grounded LLM-judged whole-summary task accuracy estimate",
              "primary_judge": q.JUDGES["primary"], "calibration": calibration, "agreement": agreement,
              "target": .95, "point_target_met": point_met,
              "wilson_lower95_target_met": trained["wilson95"][0] >= .95,
              "trained_minus_original_percentage_points": 100*(trained["task_pass_rate"]-original["task_pass_rate"]),
              "human_rated": False, "production_accuracy": None, "speed_tested": False,
              "underpowered": True, "independent_trained_checkpoints": 1,
              "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(artifacts)},
              "analysis_source_sha256": file_hash(source)}
    write_json(campaign_path, result)
    controls = calibration["primary"]
    table = "\n".join(f"| {model} | {value['passes']}/{value['n']} | {value['task_pass_rate']:.2%} | "
                      f"{value['wilson95'][0]:.2%}–{value['wilson95'][1]:.2%} | "
                      f"{value['fully_supported_summaries']}/{value['n']} |"
                      for model, value in summaries.items())
    failures = "\n".join(f"| {m} | {v['salience_failures']} | {v['readability_failures']} | "
                         f"{v['supported_claims']}/{v['material_claims']} |" for m, v in summaries.items())
    audit_description = json.dumps(agreement, indent=2)
    args.report.write_text(f"""# End-to-end accuracy of the memory-trained NCM

**Trained NCM task accuracy estimate: {trained['passes']}/{trained['n']} = {trained['task_pass_rate']:.2%}.**
The 95% target is {'met at the point-estimate level only' if point_met else 'not met'}.
This is source-grounded LLM-judged summary quality on held-out short fragments.
It is not the 95.46% codec reconstruction figure and is not human-rated production accuracy.

Date: 2026-10-22. Hypothesis: `{q.HYPOTHESIS}`.
Classification: `{result['classification']}`.
Aggregate: `{campaign_path}`.
SHA256: `{file_hash(campaign_path)}`.

## What counts as correct

A complete generated summary passes only if it is nonempty, within 128
cl100k_base tokens, faithful to the original source, readable and useful as a
brief summary of the main point. Every material claim must be supported.
Reasonable paraphrases and omission of minor/list details are allowed. Headings,
tangential fragments, unsupported invented facts and incoherent text fail.
Claims carry supported/contradicted/insufficient-evidence labels and source-line
IDs; the exact original evidence text is reconstructed by code.

Primary judge: **`{q.JUDGES['primary']}`**. Model identities were hidden and
candidate order shuffled. All candidates were judged independently against the
complete original text. Saved targets were a third anonymous candidate, not
assumed correct. No outputs were regenerated and no model was retrained or
selected using these grades.

The judge accepted **{controls['positives_accepted']}/{controls['positive_controls']}**
source-reviewed acceptable controls and rejected
**{controls['negatives_rejected']}/{controls['negative_controls']}** incorrect controls.
These controls contain realistic paraphrases from 12 separate source sessions,
not just easy exact-match positives. They were authored/reviewed by Codex, not
independent humans. The previous overly strict 0–4 rubric was not reused.

## Measured pass rates

| Candidate | Correct summaries | Task accuracy estimate | Wilson 95% interval | Fully source-supported |
| --- | ---: | ---: | ---: | ---: |
{table}

`original_ncm` is the NCM before the shared-data adaptation, not the full Llama
3B teacher. `saved_summary` is the supplied training-target style. A supported
snippet can still fail as a summary when it misses the main point.

| Candidate | Salience failures | Readability failures | Supported/material claims |
| --- | ---: | ---: | ---: |
{failures}

The trained-minus-original change is
**{result['trained_minus_original_percentage_points']:.2f} percentage points**.
The point target requires at least 244/256 passes. The stronger condition that
the Wilson lower confidence bound is at least 95% is
`{result['wilson_lower95_target_met']}`. Confidence intervals describe sampling
under these judge labels; they do not include judge error or model-seed variation.
Source-session cluster-bootstrap intervals (10,000 draws, seed 20261022) are
saved in the aggregate to account for multiple fragments from one session.

## Independent audit

Secondary judge: `{q.JUDGES['secondary']}`. Its control result and any completed
64-case audit are retained, including infrastructure failures. Agreement:

```json
{audit_description}
```

## Scope and integrity

The 256 cases are exactly the fixed held-out cases from the completed training
campaign. The train/test source conversations are disjoint. The sources fit
512 GPT-2 input tokens, with a 48-position NCM output canvas. This is not an
evaluation of the full 2,048-token fragments or 6,300-token application batches.
LongMemEval was used as training data, so this is the designated source-disjoint
test partition, not the untouched LongMemEval benchmark.

Raw requests, replies, retries, source citations, candidate mappings, producing
source and hashes are preserved under `{args.root}`. Valid low grades were never
retried. The aggregate was independently rebuilt from the raw responses. The
study remains exploratory with one adapted checkpoint; no speed claim is made.
The earlier codec accuracy and word-overlap report remains unchanged as a
separate measurement of different things.
""")
    print(json.dumps({"classification": result["classification"], "models": summaries,
                      "calibration": calibration, "agreement": agreement}), flush=True)


def register(args):
    path = args.root / "campaign_results.json"
    result = q.read(path)
    for file, expected in result["artifacts"].items():
        if file_hash(file) != expected:
            raise ValueError("completed evidence changed")
    score = result["models"]["trained_ncm"]
    record = {"id": q.HYPOTHESIS, "name": "End-to-end task accuracy of the memory-trained NCM",
              "description": f"The memory-trained NCM passed {score['passes']}/{score['n']} held-out "
              f"summaries ({score['task_pass_rate']:.2%}) under source-grounded blinded LLM judging. "
              "Measures complete summary faithfulness, main-point usefulness and readability. "
              "Codec reconstruction accuracy is a separate metric. One checkpoint, native short-context range.",
              "category": "architecture", "tested": True, "confirmed": False,
              "confirmation_status": result["classification"], "status": "completed_exploratory",
              "underpowered": True, "human_rated": False, "task_pass_rate": score["task_pass_rate"],
              "task_passes": score["passes"], "task_cases": score["n"],
              "judge": result["primary_judge"], "application_accuracy_tested": False,
              "direct_experiment_count": 512, "speed_tested": False,
              "campaign": str(path), "campaign_sha256": file_hash(path),
              "report": str(args.report), "report_sha256": file_hash(args.report)}
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(
        project_name="structure_net_completed_hypotheses", queue_dir="data/experiment_queue",
        sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected", enable_wandb=False,
        auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    exact = logger.hypotheses_collection.get(ids=[q.HYPOTHESIS], include=["metadatas"])
    if exact["ids"] != [q.HYPOTHESIS] or any(exact["metadatas"][0][k] != record[k]
        for k in ("tested", "confirmed", "confirmation_status", "task_passes", "task_cases")):
        raise ValueError("exact registry readback failed")
    semantic = logger.hypotheses_collection.query(query_texts=[record["description"]], n_results=10)
    if q.HYPOTHESIS not in semantic["ids"][0]:
        raise ValueError("semantic registry readback failed")
    receipt = {"status": "registered_and_read_back", "record": record,
               "exact_readback": exact["metadatas"][0], "semantic_ids": semantic["ids"][0]}
    for dest in (args.root / "registry_readback.json", Path("data/hypothesis_registry") / (q.HYPOTHESIS+".json")):
        if dest.exists():
            raise ValueError("fresh registry receipt required")
        write_json(dest, receipt)
        if q.read(dest)["record"] != record:
            raise ValueError("JSON readback failed")
    print(json.dumps({"status": receipt["status"], "passes": score["passes"], "n": score["n"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
