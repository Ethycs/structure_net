"""Preserve word-bank identity and gradient routing without quality claims."""
import argparse
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_separated_word_bank as study
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.neural_architecture_lab.ncm_result_registry import register_records

DEST = ROOT / "20261023_separated_word_bank_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-separated-word-bank-identity.md")


def finalize():
    if DEST.exists() or REPORT.exists():
        raise ValueError("fresh finalization required")
    if study.run.q.read(study.PILOT / "worker_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("observed terminal pilot required")
    verify(study.run.q.read(study.PILOT / "producer.json"))
    result = study.run.q.read(study.PILOT / "result.json")
    required = ("zero_word_bank_parent_logits_exact", "old_concept_generation_unchanged",
                "original_body_forecaster_books_cache_unchanged", "only_new_word_bank_trained",
                "checkpoint_optimizer_RNG_output_reload_exact")
    if result["status"] != "systems_identity_and_word_training_passed_not_quality_evidence" or not all(result[k] for k in required):
        raise ValueError("all word identity/routing gates required")
    checkpoint = study.PILOT / "model.pt"
    if file_hash(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("word checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    count = sum(v.numel() for v in saved["word"].values())
    if count != result["word_parameters"] or count != 4587520 or any(not k.endswith(("word_a", "word_b")) for k in saved["word"]):
        raise ValueError("checkpoint contains a different parameter scope")
    trace = study.run.q.read(study.PILOT / "trace.json")
    if len(trace) != 2 or trace != result["trace"] or any(r["step"] != i + 1 or not math.isfinite(r["loss"]) or not math.isfinite(r["gradient_norm"]) or r["gradient_norm"] <= 0 for i, r in enumerate(trace)):
        raise ValueError("fixed two-update gradient trace differs")
    paths = [Path(__file__), Path(study.__file__), study.PLAN,
        Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    paths += [p for p in study.PILOT.rglob("*") if p.is_file() and "producer_source" not in p.parts]
    aggregate = {"schema": "nal.ncm-separated-word-bank-results.v1",
        "classification": "separated_word_bank_identity_passed_quality_unmeasured", "pilot": result,
        "task_accuracy_measured": False, "NCM_success_established": False, "confirmed": False,
        "new_full_training_admitted": False, "final_test_opened": False, "speed_tested": False,
        "artifacts": {str(p): file_hash(p) for p in dict.fromkeys(paths)}}
    DEST.mkdir(); campaign = DEST / "campaign_results.json"; write_json(campaign, aggregate)
    REPORT.write_text(f'''# Independent word adapters preserve the native parent path

Date: 2026-10-23. Hypothesis `ncm-separated-word-bank-identity-v1`.
Classification `separated_word_bank_identity_passed_quality_unmeasured`.
Aggregate `{campaign}`, SHA256 `{file_hash(campaign)}`.

The separate word bank passed an exact identity and training-lifecycle control.
Before training, its zero-output rank-eight adapters produced exactly the same
logits as the original cached native language path with all adapters disabled.
Replacing wrappers preserved the trained five-level concept predictions and
their prior vectors exactly. This comparison uses the cached query path; it
does not claim identity to every full-prompt or incremental numerical kernel.

Two clean hard-target updates on the first two fixed summary TRAIN examples
trained only 4,587,520 new word-adapter parameters. All original 28 blocks,
MLPs, embeddings, native head, trained concept adapters/heads, old hint and
biases remained frozen. Word queries intentionally omit the old concept hint
and level bias. The shared original body is retained, with no extra full model
copy. Only word-bank A/B tensors appear in the saved trainable checkpoint.

CPU tests checked old-bank preservation, zero-word identity, gradient routing
and context restoration after exceptions. The real CUDA pilot passed finite
loss and nonzero word-gradient checks, unchanged old concept predictions,
body/books/cache hashes, and exact parameter/optimizer/RNG/output reload after
deliberate perturbation. Peak allocated GPU memory was
{result['peak_allocated_bytes']:,} bytes. The worker exited normally.

This prepares an isolated language-quality teacher; it does not demonstrate
new summary accuracy, use of NCM concepts in word generation, compression or
speed. The measured converged hybrid remains **41/64 whole-summary passes**,
below the unchanged 3B baseline's 52/64. The original nine-pass model remains
0/64. No development or sealed cases were used by this two-case pilot.

The observed regression motivates separating clean language-quality learning
from concept/noisy-state gradients, but this control does not identify sharing
as the sole cause. A new fixed training protocol and representative long-input
memory pilot are still required. Learn and measure a faithful quality teacher
before distilling its own native fixed-point trajectories for acceleration.
All prior results remain unchanged; confirmed=false.
''')
    print(json.dumps({"status": "finalized", "classification": aggregate["classification"], "word_parameters": count}), flush=True)


def register():
    register_records(DEST / "campaign_results.json", REPORT, DEST, {"ncm-separated-word-bank-identity-v1": {
        "name": "Separate native word bank preserves original language and frozen NCM forecast",
        "description": "Rank8 word-only QKVO bank over shared frozen28-block3B body. Exact zero-bank cached parent logits and unchanged old concept predictions; two clean updates touch only4587520new word parameters. Exact optimizer/RNG/output reload. Systems preparation only; no new taskaccuracy, old converged41/64 remainsfailed.",
        "confirmation_status": "separated_word_bank_identity_passed_quality_unmeasured",
        "task_accuracy_measured": False, "new_full_training_admitted": False,
        "NCM_success_established": False}})
    print(json.dumps({"status": "registered_and_read_back"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["finalize", "register"])
    globals()[parser.parse_args().command]()
