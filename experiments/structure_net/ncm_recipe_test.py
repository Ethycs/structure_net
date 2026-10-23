"""Source-disjoint test accuracy of the fixed updated NCM checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import shutil

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.infer_ncm_recipe_update import paired_codec
from experiments.structure_net.ncm_frozen_body import bridge, external_codes
from experiments.structure_net.ncm_llama import autocast, decode_ids, load_ncm, tokenizers
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_recipe_control import ROOT
from experiments.structure_net.ncm_recipe_data import PARENT
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.ncm_training_recipe import cumulative_latent
from experiments.structure_net.ncm_vocabulary import CachedVocabulary

HYPOTHESIS = "ncm-recipe-update-test-v1"
TRAINING = ROOT / "20261022_training_v1"
TEST = ROOT / "20261022_test_v1"
NOTE = Path("docs/07 - Status Reports/2026-10-22_ncm-recipe-test-amendment.md")
ARMS = ("legacy", "updated", "parent")


def verify(producer):
    for path, digest in {**producer["source_hashes"], **producer["input_hashes"]}.items():
        if file_hash(path) != digest:
            raise ValueError("test producer changed")


def generate():
    if TEST.exists() or not torch.cuda.is_available():
        raise ValueError("fresh test root and CUDA required")
    campaign = q.read(PARENT / "campaign_results.json")
    from experiments.structure_net.run_ncm_recipe_update import verify as verify_training
    training_producer = q.read(TRAINING / "producer.json")
    verify_training(training_producer)
    if file_hash(PARENT / "campaign_results.json") != "b514e093a892cd2dd4cdf46cdd018d900d41371fa5b233f623c6764b98e406b3":
        raise ValueError("original test campaign changed")
    cases = q.read(PARENT / "selection.json")["test"]
    trained = q.read(TRAINING / "selection.json")["train"]
    if (len(cases) != 64 or {r["source_id"] for r in cases} & {r["source_id"] for r in trained}
            or {r["source_sha256"] for r in cases} & {r["source_sha256"] for r in trained}):
        raise ValueError("test source overlap or cohort mismatch")
    verify_public(cases)
    paths = [PARENT / "test_parent.json", PARENT / "selection.json"]
    paths += [PARENT / "test_features" / (r["id"]+".pt") for r in cases]
    for path in paths:
        if file_hash(path) != campaign["artifacts"][str(path)]:
            raise ValueError("original test input changed")
    sources = [Path(__file__), NOTE]
    sources += [Path("experiments/structure_net") / n for n in
        ("ncm_training_recipe.py", "ncm_frozen_body.py", "ncm_llama.py", "infer_ncm_recipe_update.py",
         "ncm_recipe_teacher.py", "ncm_memory_accuracy.py", "ncm_memory_accuracy_v2.py")]
    inputs = [*paths, PARENT / "campaign_results.json", TRAINING / "producer.json", TRAINING / "result.json"]
    inputs += [TRAINING / (arm+suffix) for arm in ARMS[:2] for suffix in ("_codec.pt", "_generator.pt")]
    producer = {"schema": "nal.ncm-recipe-test.v1", "hypothesis_id": HYPOTHESIS, "cases": 64,
        "source_hashes": {str(p): file_hash(p) for p in sources},
        "input_hashes": {str(p): file_hash(p) for p in inputs},
        "judge": q.JUDGES["primary"], "rubric": q.RUBRIC,
        "physical_gpu": 0, "precision": "FP32 parameters and BF16 autocast", "updates": 0,
        "previously_inspected_test_cohort": True, "source_disjoint": True}
    producer["fingerprint"] = q.digest(producer)
    TEST.mkdir(parents=True)
    (TEST / "producer_source").mkdir()
    for i, path in enumerate(sources):
        shutil.copyfile(path, TEST / "producer_source" / f"{i:03d}_{path.name}")
    write_json(TEST / "producer.json", producer)
    write_json(TEST / "selection.json", {"test": cases})
    gpt, _ = tokenizers()
    gpt = CachedVocabulary(gpt)
    for arm in ARMS[:2]:
        state = torch.load(TRAINING / (arm+"_generator.pt"), map_location="cpu", weights_only=True)
        if state["producer_fingerprint"] != training_producer["fingerprint"] or state["step"] != 512:
            raise ValueError("wrong producer or incomplete test checkpoint")
        codec = paired_codec(state)
        vae, net, _ = load_ncm(codec=codec)
        net.load_state_dict(state["generator"], strict=True)
        net.eval().requires_grad_(False)
        adapter = bridge().cuda().eval().requires_grad_(False)
        adapter.load_state_dict(state["adapter"], strict=True)
        outputs = []
        for i, case in enumerate(cases):
            feature = torch.load(PARENT / "test_features" / (case["id"]+".pt"), weights_only=True)
            with torch.inference_mode(), autocast():
                context = adapter(feature["hidden"][None].cuda().float())
                codes = external_codes(net, context, feature["pad_mask"][None].cuda())
                logits = vae.decoder(cumulative_latent(vae, codes))
            if not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite test generation")
            ids = logits.argmax(-1)[0].tolist()
            text, termination = decode_ids(ids, gpt)
            outputs.append({"id": case["id"], "source": case["source"], "text": text,
                            "native_output_ids": ids, "termination": termination})
            if (i+1) % 16 == 0:
                print(json.dumps({"stage": "test_generation", "arm": arm, "completed": i+1}), flush=True)
        write_json(TEST / (arm+"_outputs.json"), {"outputs": outputs})
        del state, vae, net, adapter
        gc.collect()
        torch.cuda.empty_cache()
    shutil.copyfile(PARENT / "test_parent.json", TEST / "parent_outputs.json")
    verify(producer)
    write_json(TEST / "generation_result.json", {"status": "completed", "cases": 64,
        "producer_fingerprint": producer["fingerprint"], "trained": False})


def grade(args):
    import tiktoken
    if (TEST / "accuracy").exists() or args.report.exists():
        raise ValueError("fresh test judging/report required")
    producer = q.read(TEST / "producer.json")
    verify(producer)
    cases = q.read(TEST / "selection.json")["test"]
    verify_public(cases)
    outputs = {arm: q.read(TEST / (arm+"_outputs.json"))["outputs"] for arm in ARMS}
    packets, candidates = [], {}
    enc = tiktoken.get_encoding("cl100k_base")
    for i, case in enumerate(cases):
        cs = []
        for arm in ARMS:
            row = outputs[arm][i]
            if row["id"] != case["id"] or row["source"] != case["source"]:
                raise ValueError("unmatched test outputs")
            cid = q.digest(["test-blind", case["id"], arm])[:20]
            cs.append({"candidate_id": cid, "text": row["text"]})
            candidates[cid] = {"arm": arm, "id": case["id"], "text": row["text"],
                "tokens": len(enc.encode(row["text"], disallowed_special=()))}
        packets.append(q.packet(case["id"], case["source"], cs))
    root = TEST / "accuracy"
    root.mkdir()
    for name in ("attempts", "rows"):
        (root / name).mkdir()
    write_json(root / "packets.json", packets)
    write_json(root / "candidate_map.json", candidates)
    with ThreadPoolExecutor(max_workers=4) as pool:
        judgments = []
        for row in pool.map(lambda p: q.request(p, "primary", root), packets):
            write_json(root / "rows" / (row["case_id"]+".json"), row)
            judgments.append(row)
            if len(judgments) % 16 == 0:
                print(json.dumps({"stage": "test_accuracy", "completed": len(judgments)}), flush=True)
    scores = []
    for packet, row in zip(packets, judgments):
        if row["status"] != "valid":
            continue
        raws = [q.read(p) for p in row["attempts"]]
        if any(r["valid"] for r in raws[:-1]):
            raise ValueError("valid test judgment retried")
        parsed = q.parse(raws[-1]["response"]["choices"][0]["message"]["content"], packet)
        if parsed != row["evaluations"] or row["packet_sha256"] != q.digest(packet):
            raise ValueError("test raw audit mismatch")
        for decision in parsed:
            candidate = candidates[decision["candidate_id"]]
            scores.append({**candidate, "grade": decision,
                "passed": q.passes(decision, candidate["text"], candidate["tokens"])})
    summary = {}
    for arm in ARMS:
        rs = [r for r in scores if r["arm"] == arm]
        k = sum(r["passed"] for r in rs)
        summary[arm] = {"n": len(rs), "passes": k, "task_accuracy": k/64 if len(rs) == 64 else None,
                        "wilson_95": q.wilson(k, len(rs))}
    valid = len(judgments) == 64 and all(r["status"] == "valid" for r in judgments)
    classification = ("updated_test_accuracy_invalid" if not valid else
        "updated_test_point_target_met" if summary["updated"]["passes"] >= 61 else "updated_test_accuracy_below_95")
    write_json(root / "judgments.json", judgments)
    write_json(root / "case_scores.json", scores)
    verify(producer)
    result = {"schema": "nal.ncm-recipe-test-campaign.v1", "hypothesis_id": HYPOTHESIS,
        "status": "completed", "classification": classification, "models": summary,
        "valid": valid, "retried": sum(len(r["attempts"]) > 1 for r in judgments),
        "source_disjoint": True, "previously_inspected_cohort": True,
        "underpowered": True, "human_rated": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in TEST.rglob("*.json") if "producer_source" not in p.parts}}
    write_json(TEST / "campaign_results.json", result)
    table = "\n".join(f"| {arm} | {summary[arm]['passes']}/64 | {100*summary[arm]['task_accuracy']:.2f}% |"
                       if summary[arm]["task_accuracy"] is not None else f"| {arm} | invalid | invalid |" for arm in ARMS)
    args.report.write_text(f'''# Test accuracy after the NCM recipe update

**Updated summary task accuracy: {summary['updated']['passes']}/64.**
Classification: `{classification}`. Date: 2026-10-22.
ID: `{HYPOTHESIS}`. EXPLORATORY; one training seed.
[Fixed follow-up](../07%20-%20Status%20Reports/2026-10-22_ncm-recipe-test-amendment.md).
Aggregate: `{TEST / 'campaign_results.json'}`.
SHA256: `{file_hash(TEST / 'campaign_results.json')}`.

| Candidate | Passing complete summaries | Task accuracy estimate |
| --- | ---: | ---: |
{table}

The test source conversations and exact source texts are disjoint from the 32
quality-checked training pairs. These are the original 64 test cases, previously
inspected during diagnosis; this is not an untouched benchmark or independent
confirmation. No checkpoint selection, training, regeneration or output repair
used the test scores. The updated and matched legacy recipes share initialization,
data, steps and frozen body features. The full-parent output was reused after
checking its immutable campaign hash. The parent used the original 32-word
instruction; regenerated training targets requested 24 words to fit the codec.

The judge is `{q.JUDGES['primary']}`, with unchanged source-grounded faithfulness,
salience, readability, nonempty and <=128-token checks. It previously accepted
24/24 good controls and rejected 24/24 bad controls, authored/source-reviewed by
Codex rather than independent humans. Candidates were anonymous and shuffled.
Public LongMemEval source file SHA and exact spans were verified before every
grading stage. Only public passages and generated summaries were uploaded.
All raw replies and claim evidence are retained. Valid low grades were not
retried. All 64 packets valid: `{valid}`. Transport/schema retries: `{result['retried']}`.

Point target95% requires 61/64. Wilson intervals for every model are in the
aggregate; they exclude judge error and model-seed variation. This small,
LLM-judged test does not establish human-rated production accuracy. It cannot
rescue a failed training-competence gate. No throughput or full application
input/output-range claim is made: source prefix512, codec canvas48 GPT2 tokens.

Reproduction uses the moba Python, offline model cache, physical GPU0 selected
before CUDA, `LD_LIBRARY_PATH='' OMP_NUM_THREADS=2 PYTHONPATH=src:.`:

```text
python -m experiments.structure_net.ncm_recipe_test generate
python -m experiments.structure_net.ncm_recipe_test grade --report NEW_REPORT
python -m experiments.structure_net.ncm_recipe_test register --report NEW_REPORT
```

Add the memory dependencies and tiktoken cache for CPU/network grading. Pinned
artifact roots are explicit in source; new runs require a new producing version.
''')
    print(json.dumps({"classification": classification, "models": summary}), flush=True)


def register(args):
    result = q.read(TEST / "campaign_results.json")
    for path, digest in result["artifacts"].items():
        if file_hash(path) != digest:
            raise ValueError("test evidence changed")
    record = {"id": HYPOTHESIS, "name": "Source-disjoint accuracy after NCM recipe update",
        "description": "Fixed 64-case reused source-disjoint test of updated and matched legacy "
        "NCM heads against the original full Llama parent; source-grounded LLM judging, one seed.",
        "category": "architecture", "tested": True, "confirmed": False, "underpowered": True,
        "confirmation_status": result["classification"], "task_passes": result["models"]["updated"]["passes"],
        "task_cases": 64, "speed_tested": False, "human_rated": False,
        "campaign": str(TEST / "campaign_results.json"), "campaign_sha256": file_hash(TEST / "campaign_results.json"),
        "report": str(args.report), "report_sha256": file_hash(args.report)}
    from structure_net.logging.standardized_logging import LoggingConfig, StandardizedLogger
    logger = StandardizedLogger(LoggingConfig(project_name="structure_net_completed_hypotheses",
        queue_dir="data/experiment_queue", sent_dir="data/experiment_sent", rejected_dir="data/experiment_rejected",
        enable_wandb=False, auto_upload=False, enable_chromadb=True, chromadb_path="data/chroma_db"))
    logger.log_hypothesis(record)
    collection = logger.hypotheses_collection
    if collection is None:
        raise ValueError("registry unavailable")
    exact = collection.get(ids=[HYPOTHESIS], include=["metadatas"])
    collection.update(ids=[HYPOTHESIS], metadatas=[{**exact["metadatas"][0], **record}])
    exact = collection.get(ids=[HYPOTHESIS], include=["metadatas"])
    semantic = collection.query(query_texts=[record["description"]], n_results=10)
    if any(exact["metadatas"][0].get(k) != v for k,v in record.items()) or HYPOTHESIS not in semantic["ids"][0]:
        raise ValueError("test registry readback failed")
    receipt = {"status": "registered_and_read_back", "record": record, "exact": exact["metadatas"][0], "semantic_ids": semantic["ids"][0]}
    for p in (TEST / "registry_readback.json", Path("data/hypothesis_registry") / (HYPOTHESIS+".json")):
        if p.exists():
            raise ValueError("fresh test registry receipt required")
        write_json(p, receipt)
        if q.read(p) != receipt:
            raise ValueError("test JSON readback failed")
    print(json.dumps({"status": receipt["status"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["generate", "grade", "register"])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    generate() if args.command == "generate" else globals()[args.command](args)
