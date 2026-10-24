"""Locked source-only evaluation of the fixed1024 word-gradient continuation."""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from experiments.structure_net import ncm_lm1b_grounded_development as base
from experiments.structure_net import ncm_lm1b_word_gradient_training as continuation
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, head, q, ROOT = base.native, base.head, base.q, base.ROOT
training, semantic, reference = continuation.engine, base.semantic, base.reference
HYPOTHESIS = "ncm-lm1b-word-gradient-source-only-DEV-v1"
MODULE = "experiments.structure_net.ncm_lm1b_word_gradient_development"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_lm1b_word_gradient_development"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-lm1b-word-gradient-development-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-lm1b-word-gradient-development.md")
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_lm1b_word_gradient_development.py")
TEST = Path("tests/structure_net/test_ncm_lm1b_word_gradient_development.py")
WORKFLOW = ROOT / "20261024_lm1b_word_gradient_development_workflow_v1"
METADATA = ROOT / "20261024_lm1b_word_gradient_development_metadata_v1"
PILOT = ROOT / "20261024_lm1b_word_gradient_development_inference_pilot_v1"
OUTPUTS = ROOT / "20261024_lm1b_word_gradient_development_outputs_v1"
PACKETS = ROOT / "20261024_lm1b_word_gradient_development_packets_v1"
JUDGE_PILOT = ROOT / "20261024_lm1b_word_gradient_development_judge_pilot_v1"
GRADE = ROOT / "20261024_lm1b_word_gradient_development_grade_v1"
RESULTS = ROOT / "20261024_lm1b_word_gradient_development_results_v1"
OLD_CACHE = ROOT / "20261024_lm1b_native_word_readout_grade_v1/judgment_cache.json"
CACHE_ARCHIVE = ROOT / "20261024_lm1b_native_word_readout_results_v1"
CACHE_HYPOTHESIS = "ncm-lm1b-native-word-readout-source-only-DEV-v1"
CACHE_CAMPAIGN = "c8164656fc0d434c0584a8a8e0f9c16a8de8f9d1e3b356433fbf9847bf0aa9b7"
CACHE_SHA = "d27baaf96d563d2528263d002861754ddd98f65bc458d64fc80bc3659a171b28"
SOURCE, FEATURES = base.SOURCE, base.FEATURES
SOURCE_SHA, FEATURE_SHA = base.SOURCE_SHA, base.FEATURE_SHA
COHORTS, FORBIDDEN, GENERATOR_HASH = base.COHORTS, base.FORBIDDEN, base.GENERATOR_HASH
_configure, _producing = base.configure, base.producing


def configure():
    _configure()
    continuation.configure()
    for name in ("HYPOTHESIS", "MODULE", "AUDITOR", "PLAN", "REPORT", "AUDITOR_FILE", "TEST", "WORKFLOW",
                 "METADATA", "PILOT", "OUTPUTS", "PACKETS", "JUDGE_PILOT", "GRADE", "RESULTS", "OLD_CACHE",
                 "CACHE_ARCHIVE", "CACHE_HYPOTHESIS", "CACHE_CAMPAIGN", "CACHE_SHA"):
        setattr(base, name, globals()[name])
    base.configure, base.producing = configure, producing
    base.checkpoint_contract, base.base_cache = checkpoint_contract, base_cache


def producing(root, stage, inputs=(), config=None):
    files = [Path(__file__), Path(continuation.__file__), Path(continuation.objective.__file__),
             Path(continuation._original_objective.__file__)]
    p = _producing(root, stage, [*inputs, *files], config)
    p["hypothesis_id"] = HYPOTHESIS
    p["config"].update(hypothesis_id=HYPOTHESIS, schema="nal.ncm-lm1b-word-gradient-source-only-dev.v1",
        checkpoint_selection="fixed final1024 continuation of original4096 only", canonical_prior_firsts=2728,
        warm_start_checkpoint_sha256=continuation.BASE_SHA, new_random_trainable_weights=False,
        inference="unchanged hard native generation; surrogate backward absent", training_word_gradient_into_concept_attention=True)
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"word_gradient_dev_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def checkpoint_contract():
    native.vendor.terminal(continuation.TRAIN)
    native.vendor.verify(q.read(continuation.WORKFLOW / "producer.json"))
    whole = q.read(continuation.WORKFLOW / "result.json")
    tool = q.read(continuation.WORKFLOW / "tool_terminal_receipt.json")
    state = q.read(continuation.TRAIN / "result.json")
    if (whole["status"] != "word_gradient_fixed1024_completed_task_accuracy_pending" or tool["exit_code"] != 0
            or [r["stage"] for r in whole["stages"]] != ["pilot", "train"]
            or any(r["exit_code"] != 0 for r in whole["stages"])
            or state["updates"] != 1024 or state["final_fixed_updates"] != 1024
            or state["status"] != "source_grounded_NCM_trained_source_only_task_accuracy_pending"
            or not state["exact_attention_optimizer_sampler_neighbor_CPU_CUDA_reload"]
            or not state["source_only_reload_exact"] or not state["new_attention_updated"]
            or not state["word_gradient_objective"] or state["original_body_updated"]
            or state["initial_attention_sha256"] != continuation.BASE_ATTENTION
            or state["warm_start_checkpoint_sha256"] != continuation.BASE_SHA
            or state["frozen_native_states_unchanged"] != {"generator": GENERATOR_HASH, "codec": head.DOMAIN_CODEC_HASH}
            or file_hash(continuation.TRAIN / "model.pt") != state["checkpoint_sha256"]):
        raise ValueError("actual fixed1024 continuation, frozen states and whole-tool exit required")
    saved = torch.load(continuation.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    if (saved["step"] != 1024 or saved["config"] != asdict(continuation.Config())
            or saved["producer_fingerprint"] != state["producer_fingerprint"]):
        raise ValueError("strict final1024 continuation checkpoint configuration differs")
    return state, saved


def base_cache():
    campaign = q.read(CACHE_ARCHIVE / "campaign_results.json")
    receipt = q.read(CACHE_ARCHIVE / (CACHE_HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(CACHE_ARCHIVE / "campaign_results.json") != CACHE_CAMPAIGN or file_hash(OLD_CACHE) != CACHE_SHA
            or campaign["artifacts"].get(str(OLD_CACHE)) != CACHE_SHA or campaign["canonical_first_judgments"] != 2728
            or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != CACHE_CAMPAIGN or receipt["exact"]["campaign_sha256"] != CACHE_CAMPAIGN):
        raise ValueError("registered latest2728 canonical first judgments required")
    cache = q.read(OLD_CACHE)
    if len(cache) != 2728:
        raise ValueError("all2728 canonical first judgments required")
    semantic.audit_cache(cache)
    return cache


source_inputs, decode, predict = base.source_inputs, base.decode, base.predict
inference_probes, task_metric, joint_gate = base.inference_probes, base.task_metric, base.joint_gate
validate_sources, secondary_admitted = base.validate_sources, base.secondary_admitted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "inference-pilot", "generate", "packets", "judge-pilot", "grade", "workflow"))
    args = parser.parse_args()
    configure()
    {"prepare": base.prepare, "inference-pilot": lambda: base.generate(True), "generate": base.generate,
        "packets": base.packets, "judge-pilot": lambda: base.grade(True), "grade": base.grade, "workflow": base.workflow}[args.stage]()
