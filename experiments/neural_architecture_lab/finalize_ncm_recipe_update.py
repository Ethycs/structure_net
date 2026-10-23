"""Grade training competence and preserve the bounded NCM recipe comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_recipe_control import PLAN, ROOT
from experiments.structure_net.ncm_recipe_teacher import QUALITY, TEACHER, verify_public

HYPOTHESIS = "ncm-recipe-update-v1"
ARMS = ("legacy", "updated", "target")


def classify(summary, valid):
    if not valid:
        return "recipe_update_training_judgment_invalid"
    if summary["target"]["passes"] < 31:
        return "recipe_update_training_positive_control_failed"
    if summary["updated"]["passes"] >= 31:
        return "recipe_update_train_competence_passed_needs_heldout"
    return "recipe_update_implemented_train_competence_failed"


def grade(args):
    import tiktoken
    destination = args.root / "accuracy"
    if destination.exists() or args.report.exists():
        raise ValueError("fresh grading/report required")
    training = q.read(args.root / "result.json")
    if training["status"] != "trained_pending_train_competence_judging":
        raise ValueError("completed fixed training required")
    cases = q.read(args.root / "selection.json")["train"]
    if len(cases) != 32:
        raise ValueError("fixed 32 training cases required")
    verify_public(cases)
    models = {arm: q.read(args.root / (arm+"_outputs.json"))["outputs"] for arm in ARMS[:2]}
    candidates, packets = {}, []
    enc = tiktoken.get_encoding("cl100k_base")
    for i, case in enumerate(cases):
        cs = []
        for arm in ARMS:
            row = {"text": case["target"]} if arm == "target" else models[arm][i]
            if arm != "target" and any(row[k] != case[k] for k in ("id", "source", "target")):
                raise ValueError("model/source pairing changed")
            cid = q.digest(["recipe-blind", case["id"], arm])[:20]
            cs.append({"candidate_id": cid, "text": row["text"]})
            candidates[cid] = {"arm": arm, "id": case["id"], "text": row["text"],
                "tokens": len(enc.encode(row["text"], disallowed_special=()))}
        packets.append(q.packet(case["id"], case["source"], cs))
    destination.mkdir()
    for name in ("attempts", "rows", "producer_source"):
        (destination / name).mkdir()
    write_json(destination / "packets.json", packets)
    write_json(destination / "candidate_map.json", candidates)
    sources = [Path(__file__), PLAN, Path("experiments/structure_net/ncm_recipe_data.py"),
        Path("experiments/structure_net/ncm_memory_accuracy.py"),
        Path("experiments/structure_net/ncm_memory_accuracy_v2.py"),
        Path("experiments/structure_net/ncm_recipe_teacher.py"),
        Path("docs/07 - Status Reports/2026-10-22_ncm-recipe-teacher-target-amendment.md")]
    inputs = [args.root / n for n in ("result.json", "producer.json", "selection.json",
                                      "legacy_outputs.json", "updated_outputs.json")]
    inputs += [QUALITY / "result.json", QUALITY / "selected.json"]
    producer = {"schema": "nal.ncm-recipe-train-judging.v1", "hypothesis_id": HYPOTHESIS,
        "judge": q.JUDGES["primary"], "rubric": q.RUBRIC, "workers": 4,
        "source_hashes": {str(p): file_hash(p) for p in sources},
        "input_hashes": {str(p): file_hash(p) for p in inputs},
        "scope": "training competence only; not held-out model accuracy"}
    producer["fingerprint"] = q.digest(producer)
    for i, path in enumerate(sources):
        shutil.copyfile(path, destination / "producer_source" / f"{i:03d}_{path.name}")
    write_json(destination / "producer.json", producer)
    with ThreadPoolExecutor(max_workers=4) as pool:
        judgments = []
        for row in pool.map(lambda packet: q.request(packet, "primary", destination), packets):
            write_json(destination / "rows" / (row["case_id"]+".json"), row)
            judgments.append(row)
            if len(judgments) % 8 == 0:
                print(json.dumps({"stage": "train_competence_judging", "completed": len(judgments)}), flush=True)
    scores = []
    for packet, row in zip(packets, judgments):
        if row["status"] != "valid":
            continue
        raws = [q.read(p) for p in row["attempts"]]
        if any(r["valid"] for r in raws[:-1]):
            raise ValueError("valid judgment retried")
        parsed = q.parse(raws[-1]["response"]["choices"][0]["message"]["content"], packet)
        if parsed != row["evaluations"] or row["packet_sha256"] != q.digest(packet):
            raise ValueError("raw judgment audit mismatch")
        for decision in parsed:
            candidate = candidates[decision["candidate_id"]]
            scores.append({**candidate, "grade": decision,
                           "passed": q.passes(decision, candidate["text"], candidate["tokens"])})
    write_json(destination / "judgments.json", judgments)
    write_json(destination / "case_scores.json", scores)
    summary = {}
    for arm in ARMS:
        rs = [r for r in scores if r["arm"] == arm]
        passes = sum(r["passed"] for r in rs)
        summary[arm] = {"n": len(rs), "passes": passes,
            "training_pass_fraction": passes/len(rs) if rs else None,
            "salience_failures": sum(not r["grade"]["salient"] for r in rs),
            "readability_failures": sum(not r["grade"]["readable"] for r in rs),
            "faithfulness_failures": sum(not r["grade"]["faithful"] for r in rs)}
    valid = len(judgments) == 32 and all(r["status"] == "valid" for r in judgments)
    classification = classify(summary, valid)
    for path, digest in {**producer["source_hashes"], **producer["input_hashes"]}.items():
        if file_hash(path) != digest:
            raise ValueError("judging producer changed")
    artifacts = [p for p in args.root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    artifacts += [ROOT / "20261022_control_v2" / n for n in
                  ("result.json", "producer.json", "outputs.json", "generation_reuse.json")]
    artifacts += [p for p in QUALITY.rglob("*.json") if "producer_source" not in p.parts]
    artifacts += [p for p in TEACHER.glob("*.json")]
    artifacts += [p for p in (ROOT / "20261022_quality_v1").rglob("*.json") if "producer_source" not in p.parts]
    campaign = {"schema": "nal.ncm-recipe-update-campaign.v1", "hypothesis_id": HYPOTHESIS,
        "status": "completed", "classification": classification, "conformance_profile": "EXPLORATORY",
        "original_control": q.read(ROOT / "20261022_control_v2/result.json"),
        "target_quality": q.read(QUALITY / "result.json"), "training": training,
        "models": summary, "valid_judgments": valid,
        "retried_packets": sum(len(r["attempts"]) > 1 for r in judgments),
        "requested_training_arms": 2, "completed_training_arms": 2,
        "independent_training_seeds": 1, "heldout_accuracy_measured": False,
        "human_rated": False, "speed_tested": False, "underpowered": True,
        "artifacts": {str(p): file_hash(p) for p in artifacts}}
    campaign_path = args.root / "campaign_results.json"
    write_json(campaign_path, campaign)
    ctrl = campaign["original_control"]
    table = "\n".join(f"| {arm} | {summary[arm]['passes']}/{summary[arm]['n']} |" for arm in ARMS)
    args.report.write_text(f'''# NCM recipe update

**Updated recipe training-summary passes: {summary['updated']['passes']}/32.**
These are training cases, not held-out accuracy. No 95% application-quality or
throughput claim follows from this bounded experiment.

Date: 2026-10-22. ID: `{HYPOTHESIS}`. Conformance: EXPLORATORY, one seed.
Classification: `{classification}`.
Aggregate: `{campaign_path}`.
SHA256: `{file_hash(campaign_path)}`.
[Plan](../07%20-%20Status%20Reports/2026-10-22_ncm-recipe-update-plan.md).

The update adds actual cumulative-latent cutoffs for decoder adaptation,
conditioning-code corruption with clean generator targets, equal per-level code
loss, independent saved augmentation RNG, and explicit generator-to-codec SHA256
pairing. Encoder/codebooks and Llama body remain frozen; both arms reuse the same
saved body features and initialized hybrid. This adapts pretrained weights; it
does not reproduce full codec/codebook training. No old producer was edited.

The original released NCM passes the fixed 128-case XSum control: ROUGE-L
{ctrl['rouge']['rougeL']:.2f}, reconstruction {ctrl['codec_correct']}/{ctrl['codec_total']}
= {100*ctrl['codec_accuracy']:.2f}%. The unmodified released evaluation shard and
decoder settings produced these outputs. This is a small original-domain sanity
check, not the full 11,334-example paper evaluation or semantic task accuracy.
The local codec scorer initially rejected long references; the
[windowing correction](../07%20-%20Status%20Reports/2026-10-22_ncm-recipe-control-windowing-fix.md)
preserves that failure and reuses unchanged original outputs with SHA receipts.
All reference tokens are covered in 48-token windows without truncation.

The original saved-target audit accepted only 5/128 candidates, with one invalid
judgment. That failed audit is preserved and those snippets were not used for
the quality-training run. The
[teacher-target amendment](../07%20-%20Status%20Reports/2026-10-22_ncm-recipe-teacher-target-amendment.md)
then fixed the first 64 public training passages and generated one greedy local
Llama3B summary each, requesting approximately 24 words to fit the codec. No
quality-based generation retry or prompt tuning occurred. Of
{campaign['target_quality']['candidates']} native-compatible generated targets,
{campaign['target_quality']['accepted']} passed the unchanged source-grounded
judge, selecting the first 32 accepted in fixed source order. Only public source
passages and newly generated summaries were sent. Cached body features retain
their original approximately-32-word prompt; both arms share that condition.
The judge controls were previously qualified and are agent-authored, not
independent human gold labels. Selection grades are not held-out task accuracy.

Each matched arm trained 128 decoder updates and 512 generator/adapter updates,
batch eight, AdamW lr1e-4, warmup32/cosine floor .1, clipping1. Legacy uses full
latents and clean conditioning with equal position weights. Updated uses .5
nested dropout, uniform retained cutoff 1..K-1 when dropped, corruption decreasing
from .30 to zero by level, and alpha=1 equal-level loss. Those rates and weighting
are experimental choices, not known optimal XSum settings. Loss values between
these differently weighted objectives must not be directly ranked.

| Blinded candidate | Passing complete training summaries |
| --- | ---: |
{table}

The primary judge is `{q.JUDGES['primary']}`, using the unchanged source-grounded
faithfulness, salience, readability, nonempty and <=128-token rubric. Candidates
were anonymous and shuffled. Raw requests, replies, evidence citations and every
retry are retained. Valid failures were not retried. The train competence gate
requires at least 31/32 updated outputs and 31/32 reference controls to pass;
classification above applies that gate. There is no held-out endpoint in this
study and no model was selected using a held-out score.

Frozen encoder/codebook identities, unchanged codec during generator training,
actual removal of finer latent levels, nonzero adapter/coarse-query gradients,
matching batches and exact checkpoint reload were checked in a separate CUDA
pilot and the full run. The vendor cumulative-history helper aliases entries;
the new helper explicitly accumulates only retained levels and is tested against
the full vendor latent. Physical GPU0 only; FP32 parameters, BF16 compute, one
worker. Body feature caching is a training convenience, not an application reuse
assumption or a deployment speed measurement. The native output canvas remains
48 GPT2 tokens and source prefix512, below the application's full range.

Reproduce from the repository root with the moba environment Python,
`LD_LIBRARY_PATH='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 PYTHONPATH=src:.`, selecting
physical GPU0 before CUDA commands. Add the memory dependencies and tokenizer
cache for the CPU/network grading stages:

```text
python -m experiments.structure_net.ncm_recipe_control pilot --output NEW_CONTROL_PILOT
python -m experiments.structure_net.ncm_recipe_control control --pilot NEW_CONTROL_PILOT --output NEW_CONTROL
python -m experiments.structure_net.run_ncm_recipe_update pilot --output NEW_TRAIN_PILOT
python -m experiments.structure_net.run_ncm_recipe_update train --pilot NEW_TRAIN_PILOT --output NEW_TRAIN
python -m experiments.neural_architecture_lab.finalize_ncm_recipe_update grade --root NEW_TRAIN --report NEW_REPORT
```

The direct scripts' pinned input roots are declared in source and fingerprinted.
Changing those roots or parameters defines a new producer. The registry must
retain `confirmed: false`, explicit training-only evidence and no speed claim.
''')
    print(json.dumps({"classification": classification, "models": summary}), flush=True)


@dataclass(frozen=True)
class Evidence:
    campaign: Path
    report: Path

    def record(self):
        value = q.read(self.campaign)
        for path, digest in value["artifacts"].items():
            if file_hash(path) != digest:
                raise ValueError("completed recipe evidence changed")
        return {"id": HYPOTHESIS, "name": "Update NCM hierarchical adaptation recipe",
            "description": "Original XSum sanity control and matched 32-example training competence "
            "comparison of clean/full-latent versus nested/corrupted/equal-level adaptation. "
            "No held-out task accuracy or speed measurement.", "category": "architecture",
            "tested": True, "confirmed": False, "underpowered": True,
            "confirmation_status": value["classification"], "independent_training_seeds": 1,
            "direct_experiment_count": 32, "heldout_accuracy_measured": False,
            "training_passes": value["models"]["updated"]["passes"], "speed_tested": False,
            "campaign": str(self.campaign), "campaign_sha256": file_hash(self.campaign),
            "report": str(self.report), "report_sha256": file_hash(self.report)}


def register(args):
    record = Evidence(args.root / "campaign_results.json", args.report).record()
    paths = [args.root / "registry_readback.json", Path("data/hypothesis_registry") / (HYPOTHESIS+".json")]
    if any(p.exists() for p in paths):
        raise ValueError("fresh registration required")
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
    if exact["ids"] != [HYPOTHESIS] or any(exact["metadatas"][0].get(k) != v for k,v in record.items()):
        raise ValueError("exact registry readback failed")
    semantic = collection.query(query_texts=[record["description"]], n_results=10)
    if HYPOTHESIS not in semantic["ids"][0]:
        raise ValueError("semantic registry readback failed")
    receipt = {"status": "registered_and_read_back", "record": record,
               "exact": exact["metadatas"][0], "semantic_ids": semantic["ids"][0]}
    for path in paths:
        write_json(path, receipt)
        if q.read(path) != receipt:
            raise ValueError("JSON registry readback failed")
    print(json.dumps({"status": receipt["status"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["grade", "register"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args)
