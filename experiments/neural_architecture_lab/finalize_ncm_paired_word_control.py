"""Report paired-word capacity without substituting it for task accuracy."""
import argparse
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_paired_word_control as control
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_paired_word_control_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-paired-word-training-control.md")


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh result root required")
    for root in (control.PILOT, control.TRAIN):
        if control.run.q.read(root / "worker_terminal_receipt.json")["exit_code"] != 0:
            raise ValueError("both workers must be observed terminal")
        verify(control.run.q.read(root / "producer.json"))
    result = control.run.q.read(control.TRAIN / "result.json")
    config = control.Config()
    if (result["updates"] != config.updates or result["status"] != "training_capacity_control_completed"
            or not result["original_body_books_selected_cache_unchanged"]
            or not result["checkpoint_optimizer_RNG_reload_exact"]
            or file_hash(control.TRAIN / "model.pt") != result["checkpoint_sha256"]):
        raise ValueError("fixed complete training control required")
    manifest = control.run.q.read(control.run.cache_preparation.CACHE / "manifest.json")
    ids = manifest["cohorts"]["summary_train"][:4]
    targets = {r["id"]: r["target_ids"] for r in control.run.q.read(control.run.data.DEST / "cases.json")["summary_train"]}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(control.run.MODEL_PATH, local_files_only=True)
    stops = {tokenizer.eos_token_id}
    generation = control.run.q.read(control.run.MODEL_PATH / "generation_config.json")
    configured = generation.get("eos_token_id", [])
    stops.update([configured] if isinstance(configured, int) else configured)
    phases = {}
    for phase in ("before", "after"):
        rows = control.run.q.read(control.TRAIN / (phase + "_outputs.json"))
        if [r["id"] for r in rows] != ids or control.metrics(rows) != result[phase]:
            raise ValueError("fixed control cohort or metric differs")
        for row in rows:
            target = targets[row["id"]]
            padded = target + [-100] * (128 - len(target))
            for arm in ("A", "B"):
                saved = row["arms"][arm]
                replayed = control.mechanism.score(saved["raw_parallel_ids"], padded, stops, tokenizer)
                if any(saved[k] != v for k, v in replayed.items()) or saved["oracle_plan"] != (arm == "B"):
                    raise ValueError("raw token recovery differs")
        phases[phase] = rows
    trace = [json.loads(line) for line in (control.TRAIN / "trace.jsonl").read_text().splitlines()]
    sampler = torch.Generator().manual_seed(7042); trajectory = torch.Generator().manual_seed(8042)
    if len(trace) != 128:
        raise ValueError("all128 updates must be retained")
    for step, row in enumerate(trace):
        expected_ids = [ids[i] for i in torch.randint(4, (2,), generator=sampler).tolist()]
        passes = int(torch.randint(4, (1,), generator=trajectory))
        if (row["step"] != step + 1 or row["ids"] != expected_ids or row["trajectory_passes"] != passes
                or row["oracle_plan"] != (step < 16 or step % 2 == 0) or row["task"] != "summary_train"
                or not all(math.isfinite(row[k]) for k in ("clean_CE", "noisy_soft_CE", "gradient_norm"))):
            raise ValueError("fixed training seed/curriculum trace differs")
    gate = result["after"]["B"]["token_recovery"] >= .9
    if result["oracle_training_capacity_gate"] != gate:
        raise ValueError("declared90percent capacity gate differs")
    classification = "oracle_word_training_capacity_passed_generalization_unmeasured" if gate else "oracle_word_training_capacity_failed"
    paths = [Path(__file__), Path(control.__file__), control.PLAN,
        Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    for root in (control.PILOT, control.TRAIN):
        paths += [p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    aggregate = {"schema": "nal.ncm-paired-word-control-results.v1", "classification": classification,
        "control": result, "sample_and_trajectory_streams_replayed": True,
        "raw_token_scores_replayed": True, "task_accuracy_measured": False,
        "NCM_success_established": False, "confirmed": False, "final_test_opened": False,
        "speed_tested": False, "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(paths)}}
    DEST.mkdir(); campaign = DEST / "campaign_results.json"; write_json(campaign, aggregate)
    rows = []
    for arm in ("A", "B"):
        before, after = result["before"][arm], result["after"][arm]
        rows.append(f"| {arm}: {'oracle target-derived' if arm == 'B' else 'source-predicted'} plan | {before['correct']}/{before['total']} ({100*before['token_recovery']:.2f}%) | {after['correct']}/{after['total']} ({100*after['token_recovery']:.2f}%) |")
    REPORT.write_text(f'''# Paired word training: capacity control, not summary accuracy

Date: 2026-10-23. Hypothesis `ncm-paired-word-control-v1`.
Classification `{classification}`. Aggregate `{campaign}`,
SHA256 `{file_hash(campaign)}`.

This control continued the failed 2,048-update checkpoint for exactly 128 word
updates on four fixed TRAIN summaries. It retained the frozen original
28-block Llama 3B body, MLPs, embeddings, word head and concept books.
Every update paired clean hard-target CE×10 with full-vocabulary consistency
on imperfect word states against the detached current clean branch. Correct
coarse plans were used during 16 warmup updates and half of later updates;
remaining updates used source-predicted plans. No new labels or API requests.

All results below are **training token recovery, not task accuracy**. Both
conditions use the same four parallel word refinements. Arm B reveals target
information and is unavailable to source-only inference.

| Plan condition | Before | After 128 updates |
| --- | --- | --- |
''' + '\n'.join(rows) + f'''

The fixed 90% oracle capacity gate **{'passed' if gate else 'failed'}**.
{'This admits a separate generalization study, not a successful summarizer.' if gate else 'Do not scale this recipe on the strength of lower losses; its decoding control failed.'}
The complete source-disjoint student task endpoint remains 0/64 whole-summary
passes. No development or final summaries were generated in this control,
and neither target-conditioned recovery nor a four-case fit can establish 95%.

Focused CPU checks verified causal divergence masking, full soft CE, detached
teacher gradients and sequential/joint gradient equivalence. The separate
four-update CUDA pilot covered summary/language cases and both plan types.
Both workers terminated successfully. Original body, books and selected cache
remained unchanged. Reload checks deliberately cleared optimizer state and
perturbed weights and RNGs before restoration; parameter values, optimizer
moments, RNG state and all probe outputs matched exactly. Peak allocated GPU
memory was {result['peak_allocated_bytes']:,} bytes. Sampling and trajectory RNG
streams and raw output token matches were replayed offline at finalization.

This is a combined training change inspired by JacobiForcing, not its original
full-model recipe or a replication of the unreleased NCM trainer. New query
adapters are shared with coarse prediction, so word training can also move
source-predicted plans; this control does not isolate those effects. No
generalization, compression, latency or throughput advantage is established.
''')
    print(json.dumps({"status": "finalized", "classification": classification, "after": result["after"]}), flush=True)


def register():
    aggregate = control.run.q.read(DEST / "campaign_results.json")
    result = aggregate["control"]
    register_records(DEST / "campaign_results.json", REPORT, DEST, {"ncm-paired-word-control-v1": {
        "name": "Paired clean/noisy frozen-native word training capacity control",
        "description": f"128 fixed updates on four TRAIN summaries; paired cleanCEtimes10 and full-vocabulary detached-clean consistency, mixed oracle/source plans. Oracle token recovery {result['after']['B']['token_recovery']:.4f}; source recovery {result['after']['A']['token_recovery']:.4f}. Training-only capacity; no taskaccuracy. Studentdevelopment remains0/64.",
        "confirmation_status": aggregate["classification"], "task_accuracy_measured": False,
        "NCM_success_established": False}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
