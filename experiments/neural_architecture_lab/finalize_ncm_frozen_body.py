"""Grade frozen-body NCM summaries, audit raw replies, and record the result."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_frozen_body import HYPOTHESIS, PLAN, PRIOR, verify
from experiments.structure_net.ncm_llama import GPT2

CALIBRATION = Path("data/experiments/ncm_memory/20261022_accuracy_v2")
ARMS = ("hybrid", "native", "parent")
SEED_NOTE = Path("docs/07 - Status Reports/2026-10-22_ncm-frozen-body-seed-clarification.md")


def grade(args):
    import tiktoken
    if any(p.exists() for p in (args.root / "accuracy", args.root / "campaign_results.json",
                               args.root / "native_tokenizer_readback.json", args.report)):
        raise ValueError("fresh append-only analysis required")
    training = q.read(args.root / "result.json")
    producer = q.read(args.root / "producer.json")
    verify(producer)
    if training["status"] != "trained_quality_pending" or training["test_cases"] != 64:
        raise ValueError("completed fixed 64-case training run required")
    public_receipt_path = args.root / "public_data_readback.json"
    public_receipt = q.read(public_receipt_path)
    if public_receipt["status"] != "verified_exact_public_benchmark_sources":
        raise ValueError("public-data provenance required before any judge request")
    public_path = Path(public_receipt["local_file"])
    if (file_hash(public_path) != public_receipt["local_sha256"]
            or public_receipt["public_file_metadata"]["lfs"]["oid"] != public_receipt["local_sha256"]):
        raise ValueError("public dataset does not match published Hugging Face SHA256")
    public_turns = defaultdict(dict)
    for sample in q.read(public_path):
        for sid, session in zip(sample["haystack_session_ids"], sample["haystack_sessions"]):
            for turn in session:
                public_turns[str(sid)][(turn["role"], text_hash(turn["content"]))] = turn["content"]
    public_sources = {}
    for case in q.read(args.root / "selection.json")["test"]:
        parts = []
        for span in case["spans"]:
            turn = public_turns[span["source_id"]][(span["role"], span["turn_text_sha256"])]
            part = turn[span["start_char"]:span["end_char"]]
            if text_hash(part) != span["span_text_sha256"]:
                raise ValueError("public source span changed")
            parts.append(part)
        public_sources[case["id"]] = "\n".join(parts)
        if public_sources[case["id"]] != case["source"]:
            raise ValueError("evaluation passage is not the exact public source")
    del public_turns
    # The body producer hashes the parent tokenizer, but omitted a separate
    # entry for the native GPT-2 text tokenizer. Check it against the immutable
    # prior training producer before grading, without rewriting either producer.
    prior_producer = q.read(PRIOR / "producer.json")
    native_tokenizer_hashes = {str(p): file_hash(p) for p in GPT2.iterdir() if p.is_file()}
    if not native_tokenizer_hashes or any(prior_producer["input_hashes"].get(p) != digest
                                        for p, digest in native_tokenizer_hashes.items()):
        raise ValueError("native tokenizer differs from prior producing identity")
    tokenizer_receipt = args.root / "native_tokenizer_readback.json"
    write_json(tokenizer_receipt, {"status": "matches_prior_producer", "hashes": native_tokenizer_hashes,
        "prior_producer": str(PRIOR / "producer.json"), "prior_sha256": file_hash(PRIOR / "producer.json"),
        "timing": "Finalization before quality grading; omitted from the new training producer's explicit inputs. "
        "Original producer and all outputs preserved unchanged."})
    from experiments.neural_architecture_lab import finalize_ncm_memory_accuracy_v2 as calibration_audit
    controls = q.read(CALIBRATION / "calibration_packets.json")
    calibration_rows, calibration = calibration_audit.audit(CALIBRATION / "calibration_primary", controls, "primary")
    if calibration["status"] != "passed" or len(calibration_rows) != 12:
        raise ValueError("qualified unchanged primary judge required")
    expected = q.read(CALIBRATION / "calibration_expected.json")
    accepted = rejected = 0
    for row in calibration_rows:
        for item in row["evaluations"]:
            truth = expected[item["candidate_id"]]
            passed = q.passes(item, truth["text"], truth["tokens"])
            accepted += int(truth["expected_pass"] and passed)
            rejected += int(not truth["expected_pass"] and not passed)
    if accepted != 24 or rejected != 24:
        raise ValueError("reconstructed calibration differs from completed qualification")
    root = args.root / "accuracy"
    root.mkdir(exist_ok=False)
    (root / "attempts").mkdir()
    (root / "rows").mkdir()
    enc = tiktoken.get_encoding("cl100k_base")
    outputs = {arm: q.read(args.root / ("test_"+arm+".json"))["outputs"] for arm in ARMS}
    ids = [r["id"] for r in outputs["hybrid"]]
    if any([r["id"] for r in outputs[arm]] != ids for arm in ARMS):
        raise ValueError("unmatched output cohorts")
    packets, candidates = [], {}
    for i, case in enumerate(outputs["hybrid"]):
        if public_sources.get(case["id"]) != case["source"]:
            raise ValueError("nonpublic passage prohibited from judge payload")
        cs = []
        for arm in ARMS:
            item = outputs[arm][i]
            if any(item[k] != case[k] for k in ("id", "source_id", "source", "target")):
                raise ValueError("unmatched sources")
            cid = text_hash("frozen-body-anonymous:"+case["id"]+arm)[:20]
            cs.append({"candidate_id": cid, "text": item["text"]})
            candidates[cid] = {"arm": arm, "id": case["id"], "source_id": case["source_id"],
                              "text": item["text"], "tokens": len(enc.encode(item["text"], disallowed_special=()))}
        packets.append(q.packet(case["id"], case["source"], cs))
    write_json(root / "packets.json", packets)
    write_json(root / "candidate_map.json", candidates)
    sources = [Path(__file__), PLAN, SEED_NOTE, Path("experiments/structure_net/ncm_memory_accuracy.py"),
               Path("experiments/structure_net/ncm_memory_accuracy_v2.py"),
               Path("experiments/structure_net/ncm_memory_data.py"),
               Path("experiments/structure_net/ncm_frozen_body.py")]
    inputs = [args.root / "result.json", args.root / "producer.json", args.root / "selection.json",
              root / "packets.json", root / "candidate_map.json", tokenizer_receipt]
    inputs += [public_receipt_path, public_path]
    inputs += [Path(p) for p in native_tokenizer_hashes]
    inputs += [args.root / ("test_"+arm+".json") for arm in ARMS]
    inputs += [CALIBRATION / "calibration_primary" / name for name in
               ("producer.json", "result.json", "judgments.json")]
    snapshot = root / "producer_source"
    snapshot.mkdir()
    for i, p in enumerate(sources):
        shutil.copyfile(p, snapshot / f"{i}_{p.name}")
    frozen = {"schema": "nal.ncm-frozen-body-accuracy.v1", "hypothesis_id": HYPOTHESIS,
              "judge": q.JUDGES["primary"], "rubric": q.RUBRIC,
              "source_hashes": {str(p): file_hash(p) for p in sources},
              "input_hashes": {str(p): file_hash(p) for p in inputs},
              "workers": 4, "max_attempts": 3, "created_utc": datetime.now(timezone.utc).isoformat(),
              "calibration_reused": str(CALIBRATION / "calibration_primary"),
              "calibration_counts": calibration["counts"]}
    frozen["fingerprint"] = q.digest(frozen)
    write_json(root / "producer.json", frozen)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = pool.map(lambda p: q.request(p, "primary", root), packets)
        judgments = []
        for row in jobs:
            write_json(root / "rows" / (row["case_id"]+".json"), row)
            judgments.append(row)
            if len(judgments) % 8 == 0:
                print(json.dumps({"stage": "task_accuracy", "completed": len(judgments), "total": 64}), flush=True)
    write_json(root / "judgments.json", judgments)
    verify(frozen)
    verify(producer)
    scores = []
    # Rebuild decisions from the actual raw replies rather than trusting the
    # already parsed row. Never drop an invalid terminal judgment.
    for row, p in zip(judgments, packets):
        attempts = [q.read(path) for path in row["attempts"]]
        if any(a["valid"] for a in attempts[:-1]):
            raise ValueError("valid low grade retried")
        for raw in attempts:
            payload = {"model": q.JUDGES["primary"], "temperature": 0, "max_tokens": 4096,
                       "messages": [{"role": "system", "content": q.RUBRIC},
                                    {"role": "user", "content": json.dumps(p, ensure_ascii=False)}]}
            if raw["request"] != payload or raw["request_sha256"] != q.digest(payload):
                raise ValueError("judge request changed")
        if row["status"] != "valid":
            continue
        raw = attempts[-1]
        if not raw["valid"] or raw["http_status"] != 200 or raw["response"].get("model") != q.JUDGES["primary"]:
            raise ValueError("judge transport/alias mismatch")
        parsed = q.parse(raw["response"]["choices"][0]["message"]["content"], p)
        if parsed != row["evaluations"] or row["packet_sha256"] != q.digest(p):
            raise ValueError("raw-response audit mismatch")
        for grade in parsed:
            candidate = candidates[grade["candidate_id"]]
            scores.append({**candidate, "grade": grade,
                           "passed": q.passes(grade, candidate["text"], candidate["tokens"])})
    valid = len(judgments) == 64 and all(r["status"] == "valid" for r in judgments)
    summary = {}
    for arm in ARMS:
        arm_scores = [r for r in scores if r["arm"] == arm]
        n, passes = len(arm_scores), sum(r["passed"] for r in arm_scores)
        summary[arm] = {"n": n, "passes": passes,
            "task_accuracy": passes/n if valid and n else None,
            "wilson_95": q.wilson(passes, n) if valid else None,
            "salience_failures": sum(not r["grade"]["salient"] for r in arm_scores),
            "readability_failures": sum(not r["grade"]["readable"] for r in arm_scores),
            "faithfulness_failures": sum(not r["grade"]["faithful"] for r in arm_scores)}
    quality = valid and summary["hybrid"]["task_accuracy"] >= .95
    classification = ("frozen_body_accuracy_endpoint_invalid" if not valid else
                      "frozen_body_training_succeeded_quality_point_target_met" if quality else
                      "frozen_body_training_succeeded_quality_below_95")
    write_json(root / "case_scores.json", scores)
    features = {str(p): file_hash(p) for split in ("train", "test")
                for p in sorted((args.root / (split+"_features")).glob("*.pt"))}
    if len(features) != 2112:
        raise ValueError("frozen feature cache incomplete")
    write_json(args.root / "feature_hashes.json", features)
    artifacts = [args.root / "result.json", args.root / "producer.json", args.root / "selection.json",
                 root / "producer.json", root / "judgments.json", root / "case_scores.json",
                 args.root / "feature_hashes.json", tokenizer_receipt]
    artifacts += [public_receipt_path]
    artifacts += [args.root / ("test_"+arm+".json") for arm in ARMS]
    artifacts += sorted((root / "attempts").glob("*.json"))
    for arm in ("hybrid", "native"):
        item = training["arms"][arm]
        if file_hash(item["checkpoint"]) != item["sha256"]:
            raise ValueError("final checkpoint changed")
        artifacts += [Path(item["checkpoint"]), args.root / (arm+"_trace.json")]
    result = {"schema": "nal.ncm-frozen-body-campaign.v1", "hypothesis_id": HYPOTHESIS,
        "status": "completed_exploratory", "classification": classification,
        "config": producer["config"], "training_fingerprint": producer["fingerprint"],
        "seed_streams": {"adapter_initialization": 42, "dropout": 52, "minibatches": 2042},
        "environment": {k: producer[k] for k in ("device", "torch", "physical_gpu", "workers")},
        "models": summary, "primary_judge": q.JUDGES["primary"], "judge_qualified": True,
        "human_rated": False, "quality_point_target_met": quality,
        "improved_over_matched_native": valid and summary["hybrid"]["passes"] > summary["native"]["passes"],
        "body_unchanged": training["body_unchanged"], "codec_unchanged": training["codec_unchanged"],
        "training": training, "invalid_judgments": sum(r["status"] != "valid" for r in judgments),
        "retried_judgments": sum(len(r["attempts"]) > 1 for r in judgments),
        "speed_tested": False, "underpowered": True,
        "artifacts": {**{str(p): file_hash(p) for p in artifacts}, **features}}
    write_json(args.root / "campaign_results.json", result)
    if args.report.exists():
        raise ValueError("fresh report required")
    table = "\n".join(f"| {arm} | {s['passes']}/{s['n']} | "
        f"{s['task_accuracy']:.2%} | {s['wilson_95'][0]:.2%}–{s['wilson_95'][1]:.2%} |"
        if s["task_accuracy"] is not None else
        f"| {arm} | {s['passes']}/{s['n']} valid cases | endpoint invalid | unavailable |"
        for arm, s in summary.items())
    accuracy_text = (f"{summary['hybrid']['task_accuracy']:.2%}" if valid else "invalid endpoint")
    args.report.write_text(f"""# NCM trained on a frozen Llama body

**Hybrid summary task accuracy: {summary['hybrid']['passes']}/{summary['hybrid']['n']} = {accuracy_text}.**
The parent body and codec remained unchanged; the attached NCM module trained.

Date: 2026-10-22. Hypothesis: `{HYPOTHESIS}`. One seed, exploratory.
Classification: `{classification}`.
Aggregate: `{args.root / 'campaign_results.json'}`.
SHA256: `{file_hash(args.root / 'campaign_results.json')}`.

| Model | Passing complete summaries | LLM-judged task accuracy | Wilson 95% interval |
| --- | ---: | ---: | ---: |
{table}

The hybrid is the retained full Llama-3.2-3B-Instruct body followed by a newly
initialized LayerNorm/linear adapter and the existing NCM generator. Only the
adapter and generator trained. The codec was frozen. The native control used
the same initialized generator/codec, 2,048 eligible training pairs, minibatches,
512 updates and optimizer schedule, with its ordinary prompt representation.
The parent baseline uses greedy generation from the same rendered prompt.
Adapter initialization seed is 42, dropout 52 and minibatches 2042. The dated
[seed clarification](../07%20-%20Status%20Reports/2026-10-22_ncm-frozen-body-seed-clarification.md)
records the already-frozen implementation's derived dropout seed; no run changed.
The new training producer omitted explicit GPT-2 tokenizer hashes. Before
grading, finalization checked every cached file against the immutable prior
training producer and retained `native_tokenizer_readback.json`; no tokenizer,
producer or generated output was changed. This is a late provenance check.

The 64 cases were selected by fixed ID hash from the source-conversation-disjoint
test partition before training. Both source tokenizers must fit their 512-token
limits; no source or target truncation. The codec has a 48-position output canvas.
These are short fragments, not the full application input/output range.

The unchanged source-grounded rubric requires faithfulness, main-point usefulness,
readability, nonempty output and <=128 cl100k_base tokens. Model identities were
hidden and candidate order shuffled. The primary judge's hash-verified calibration
was reused: 24/24 acceptable controls and 24/24 bad controls correctly classified.
Controls are agent-authored/source-reviewed, not independent human gold labels.
All 64 passages are exact spans of the public LongMemEval-S dataset. Its local
file SHA256 matches Hugging Face's published LFS SHA256; the provenance receipt
is retained. No training targets, credentials, private corpus or internal paths
are included in judge payloads. The original attempted grading launch was
rejected before execution; public-data verification preceded the reviewed retry.
Rates are LLM-judged task estimates; Wilson intervals and every raw response,
retry, source citation and per-case decision are retained in the aggregate/data.

The point target requires at least 61/64 passes. Target met: `{quality}`.
Hybrid improves over matched native control: `{result['improved_over_matched_native']}`.
No checkpoint selection or additional training used these final grades.

Frozen-body feasibility is separate from the task-quality result. The body was
run in inference mode and its entire state hash matched before/after extraction;
it was unloaded before optimization. The entire codec hash also matched after
both arms. Adapter gradients were nonzero, the external-context wrapper passed
native logits/code parity, minibatches matched and saved weights reloaded with
identical logits. CPU contracts and a separate CUDA lifecycle/memory pilot passed.

Training used cached frozen features to avoid recomputing them on every update.
This does not assume application prefix reuse and does not measure a deployment
speedup. No throughput claim is established. No neighborhood noise was added in
this first attached-head test. Prior saved targets include low-salience snippets;
their quality remains a limitation on this training objective.

## Reproduction

Use the moba environment Python from the repository root, `LD_LIBRARY_PATH=''`,
`HF_HUB_OFFLINE=1`, `OMP_NUM_THREADS=2`, `PYTHONPATH=src:.`, and physical GPU 0:

```text
python -m experiments.structure_net.ncm_frozen_body pilot --output NEW_PILOT_ROOT
python -m experiments.structure_net.ncm_frozen_body train --pilot NEW_PILOT_ROOT --output NEW_TRAIN_ROOT
```

Set `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0` before either GPU
command starts. For CPU/network grading, add
`data/experiments/ncm_memory/dependencies` to `PYTHONPATH` and set
`TIKTOKEN_CACHE_DIR=data/experiments/ncm_memory/tokenizer_cache`:

```text
python -m experiments.neural_architecture_lab.finalize_ncm_frozen_body grade --root NEW_TRAIN_ROOT --report NEW_REPORT
python -m experiments.neural_architecture_lab.finalize_ncm_frozen_body register --root NEW_TRAIN_ROOT --report NEW_REPORT
```
""")
    print(json.dumps({"classification": classification, "models": summary}), flush=True)


@dataclass(frozen=True)
class Evidence:
    campaign: Path
    report: Path

    def record(self):
        result = q.read(self.campaign)
        for path, expected in result["artifacts"].items():
            if file_hash(path) != expected:
                raise ValueError("completed evidence changed")
        score = result["models"]["hybrid"]
        return {"id": HYPOTHESIS, "name": "Train an NCM module on a frozen Llama 3B body",
            "description": f"Frozen-body attached NCM trained with unchanged body/codec; "
            f"hybrid whole-summary task passes {score['passes']}/{score['n']}. "
            "Source-grounded blinded LLM judging; one seed, short native context. No speed claim.",
            "category": "architecture", "tested": True, "confirmed": False,
            "confirmation_status": result["classification"], "underpowered": True,
            "direct_experiment_count": 192, "independent_training_seeds": 1,
            "body_unchanged": result["body_unchanged"], "human_rated": False,
            "task_passes": score["passes"], "task_cases": score["n"], "speed_tested": False,
            "campaign": str(self.campaign), "campaign_sha256": file_hash(self.campaign),
            "report": str(self.report), "report_sha256": file_hash(self.report)}


def register(args):
    evidence = Evidence(args.root / "campaign_results.json", args.report)
    record = evidence.record()
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(
        project_name="structure_net_completed_hypotheses", queue_dir="data/experiment_queue",
        sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected", enable_wandb=False,
        auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    collection = logger.hypotheses_collection
    if collection is None:
        raise ValueError("registry unavailable")
    exact = collection.get(ids=[HYPOTHESIS], include=["metadatas"])
    collection.update(ids=[HYPOTHESIS], metadatas=[{**exact["metadatas"][0], **record}])
    exact = collection.get(ids=[HYPOTHESIS], include=["metadatas"])
    if exact["ids"] != [HYPOTHESIS] or any(exact["metadatas"][0].get(k) != v for k,v in record.items()):
        raise ValueError("exact readback failed")
    semantic = collection.query(query_texts=[record["description"]], n_results=10)
    if HYPOTHESIS not in semantic["ids"][0]:
        raise ValueError("semantic readback failed")
    receipt = {"status": "registered_and_read_back", "record": record,
               "exact": exact["metadatas"][0], "semantic_ids": semantic["ids"][0]}
    for p in (args.root / "registry_readback.json", Path("data/hypothesis_registry") / (HYPOTHESIS+".json")):
        if p.exists():
            raise ValueError("fresh receipt required")
        write_json(p, receipt)
        if q.read(p) != receipt:
            raise ValueError("JSON readback failed")
    print(json.dumps({"status": receipt["status"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["grade", "register"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)
