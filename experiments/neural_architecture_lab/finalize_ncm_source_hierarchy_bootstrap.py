"""Audit the fixed concept bootstrap only after its observed terminal success."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_source_hierarchy_supervised as supervised
from experiments.structure_net import ncm_source_hierarchy_bootstrap as worker
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = worker.study.data.ROOT / "20261023_source_hierarchy_bootstrap_results_terminal_v2"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-source-hierarchy-bootstrap.md")
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-hierarchy-bootstrap-finalization-plan.md")
LEVELS = (16, 24, 40, 56, 80, 104, 128, 160, 192)
CLASSIFICATION = "source_hierarchy_bootstrap_completed_word_training_and_task_quality_unproven"


def replay_trace(rows, trace, config):
    if len(trace) != config.updates:
        raise ValueError("only the complete fixed training budget can be finalized")
    sampler = torch.Generator().manual_seed(config.sampler_seed)
    noise = torch.Generator().manual_seed(config.noise_seed)
    counts = Counter()
    corruptions = 0
    for step, entry in enumerate(trace, 1):
        row = rows[int(torch.randint(len(rows), (1,), generator=sampler))]
        corrupted = bool(torch.rand((), generator=noise) < config.corrupt_probability)
        codes = torch.tensor([row["concept_ids"]], dtype=torch.long)
        history = worker.core.uniform_history(codes, LEVELS, noise, config.maximum_uniform_rate) if corrupted else codes
        digest = worker.study.recovery.base.tensor_digest({"codes": history})
        if (entry["step"] != step or entry["id"] != row["id"] or entry["source_frame"] != row["source_frame"]
                or entry["corrupted"] is not corrupted or entry["history_sha256"] != digest
                or entry["lr"] != worker.rate(step, config)):
            raise ValueError("sample, full source, corruption history or learning rate replay differs")
        values = [entry["loss"], entry["gradient_norm"]]
        if len(entry["levels"]) != 8 or [r["level"] for r in entry["levels"]] != list(range(8)):
            raise ValueError("all eight concept losses required for every update")
        values += [r[key] for r in entry["levels"] for key in ("active_CE", "padded_CE")]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("nonfinite or negative loss/norm evidence")
        counts[row["id"]] += 1
        corruptions += corrupted
    return sampler.get_state(), noise.get_state(), {
        "updates": len(trace), "distinct_TRAIN_cases_sampled": len(counts),
        "sample_counts": dict(sorted(counts.items())), "corrupted_updates": corruptions,
        "all_sample_histories_and_rates_replayed": True}


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh append-only finalization required")
    for root in (supervised.PILOT, supervised.TRAIN, worker.study.data.DATA):
        worker.study.data.terminal(root)
    workflow = worker.study.q.read(supervised.WORKFLOW / "result.json")
    worker.study.data.verify(worker.study.q.read(supervised.WORKFLOW / "producer.json"))
    if (workflow["status"] != "completed_concept_bootstrap_only"
            or [r["stage"] for r in workflow["stages"]] != ["pilot", "train"]
            or any(r["exit_code"] != 0 for r in workflow["stages"])):
        raise ValueError("complete durable supervisor result required before hashing its artifacts")
    result = worker.study.q.read(supervised.TRAIN / "result.json")
    pilot = worker.study.q.read(supervised.PILOT / "result.json")
    config = worker.Config()
    if (result["status"] != "source_hierarchy_bootstrap_pending_generated_history_word_training"
            or result["updates"] != config.updates or result["cases"] != 252
            or result["task_accuracy"] is not None or result["NCM_success_established"]
            or result["final_test_opened"] or result["speed_tested"]
            or not result["reload_exact"] or not result["source_only_reload_probe_exact"]
            or result["parent_loaded"] or not result["word_modules_finest_head_and_codec_frozen"]
            or result["peak_allocated_bytes"] > 8_000_000_000):
        raise ValueError("unchanged completed concept-only bootstrap contract required")
    for key in ("initial_parameter_sha256", "frozen_parameter_sha256_unchanged", "frozen_codec_sha256_unchanged"):
        if result[key] != pilot[key]:
            raise ValueError("fresh initialization or frozen-state identity differs")
    rows = worker.study.q.read(worker.study.data.DATA / "cases.json")["train"]
    trace = [json.loads(line) for line in (supervised.TRAIN / "trace.jsonl").open()]
    sampler, noise, replay = replay_trace(rows, trace, config)
    model_path = supervised.TRAIN / "model.pt"
    if file_hash(model_path) != result["checkpoint_sha256"]:
        raise ValueError("fixed final checkpoint hash differs")
    state = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
    if (state["step"] != config.updates or state["config"] != asdict(config)
            or state["source_frame"] != trace[-1]["source_frame"]
            or state["producer_fingerprint"] != result["producer_fingerprint"]
            or not torch.equal(sampler, state["sampler_rng"]) or not torch.equal(noise, state["noise_rng"])):
        raise ValueError("selected checkpoint, sampler or noise RNG differs")
    if not state["optimizer"]["state"] or any(int(value["step"]) != config.updates for value in state["optimizer"]["state"].values()):
        raise ValueError("optimizer states did not complete the declared updates")
    initial = torch.load(worker.ROOT_TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)["model"]
    frozen = [name for name in state["model"] if name.startswith(("word_projection.", "word_decoder.",
        "source_head.generator.lm_heads.8."))]
    if not frozen or any(not torch.equal(state["model"][name], initial[name]) for name in frozen):
        raise ValueError("copied word modules or unused finest head changed")
    artifacts = [Path(__file__), PLAN, worker.PLAN, supervised.PLAN]
    for root in (supervised.PILOT, supervised.TRAIN, supervised.WORKFLOW):
        artifacts += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    aggregate = {"schema": "nal.ncm-source-hierarchy-bootstrap.v1", "hypothesis_id": worker.HYPOTHESIS,
        "classification": CLASSIFICATION, "config": asdict(config), "result": result,
        "replay": replay, "frozen_state_tensors_checked": len(frozen),
        "complete_summary_accuracy": None, "original_parent_weights_frozen": True,
        "word_training_required": True, "confirmed": False, "final_test_opened": False,
        "speed_tested": False, "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(artifacts)}}
    DEST.mkdir()
    campaign = DEST / "campaign_results.json"
    write_json(campaign, aggregate)
    REPORT.write_text(f'''# Complete concept hierarchy pretrained; summary quality remains unproven

Date: 2026-10-23. Hypothesis `{worker.HYPOTHESIS}`.
Classification `{CLASSIFICATION}`.
Aggregate `{campaign}`, SHA256 `{file_hash(campaign)}`.

The fixed 16,384-update bootstrap completed on 252 vetted training summaries.
All eight coarse concept levels trained with source features, balanced active
and padded-block losses, root emphasis and uniform conditioning corruption.
The original Llama body was absent; its original weights and MLPs remain frozen.
The original codec, copied word projection/decoder and unused finest head stayed
exact. Peak allocation was {result['peak_allocated_bytes']:,} bytes; no latency
comparison was performed.

Every sampled case, full source frame, learning rate and corruption-history hash
replayed. Final model step, optimizer steps, sampler/noise RNG and checkpoint
digest matched. The maximum-frame systems pilot and full training have actual
zero-exit child receipts and exact checkpoint/source-only reload checks. Only
the fixed final checkpoint was selected; intermediate checkpoints are recovery
artifacts. The earlier terminal-unobserved pilot remains unchanged, with its
exit unknown. The repeated pilot used identical initial parameters and supplied
the required observed terminal evidence.

This stage has no complete-summary accuracy measurement. The latest graded
word-final summarizer still failed 0/64 short and 0/64 longer development cases.
All prior failures remain failures. Generated-history word training and fresh
source-only task evaluation remain necessary. Both final cohorts stay closed;
the 95% task goal is unmet and no compression or speedup is established.
''')
    print(json.dumps({"status": "finalized_concept_bootstrap_only", "classification": CLASSIFICATION,
        "campaign_sha256": file_hash(campaign)}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {worker.HYPOTHESIS: {
        "name": "Frozen-parent source concept hierarchy bootstrap",
        "description": "Fixed16384updates on252vetted TRAINsummaries; all8coarseheads trained with uniform history corruption. Originalcodec, word modules and originalLlamabody frozen. Exact trace/RNG/checkpoint replay. No complete-summary accuracy or speed measurement.",
        "confirmation_status": CLASSIFICATION, "task_accuracy_measured": False,
        "NCM_success_established": False, "word_training_required": True}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
