"""Independent CPU replay of hard-forward parity and saved word-gradient tensors."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_lm1b_word_gradient_pilot as run
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json

RESULTS = run.ROOT / "20261024_lm1b_word_gradient_cuda_pilot_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-lm1b-word-gradient-pilot.md")


def active(values, prefix):
    return any(n.startswith(prefix) and v is not None and bool(v.abs().sum() > 0) for n, v in values.items())


def attention_parameters(saved):
    result = {}
    for name, values in saved["concept_attention"].items():
        result.update({"concepts." + name + "." + k: v for k, v in values.items()})
    for name in ("word_memory", "word_adapters"):
        result.update({name + "." + k: v for k, v in saved[name].items()})
    return result


def finalize():
    if RESULTS.exists() or REPORT.exists():
        raise ValueError("fresh systems aggregate required")
    if torch.cuda.is_initialized():
        raise ValueError("CPU audit only")
    run.native.vendor.terminal(run.DEST)
    q = run.q
    result, producer = q.read(run.DEST / "result.json"), q.read(run.DEST / "producer.json")
    frozen = {"generator": "60316ea14c56e6276be581112a2a1d81437e3ad8bb4ea3ddf8a98b946805e26a", "codec": run.head.DOMAIN_CODEC_HASH}
    if (result["status"] != "systems_word_gradient_hard_forward_and_reload_passed_not_task_quality"
            or result["updates"] != 2 or result["training_microbatch"] != 4
            or result["new_trainable_parameters"] != 11796480 or result["old_frozen_states_unchanged"] != frozen
            or result["peak_allocated_bytes"] > 7_500_000_000 or result["initial_attention_sha256"] != run.BASE_ATTENTION
            or not result["exact_hard_generation_optimizer_and_allRNG_reload"]
            or not result["baseline_word_loss_concept_gradient_absent"]
            or not result["bridged_word_loss_concept_gradient_nonzero"]
            or not result["hard_forward_word_logits_bitwise_equal"]
            or result["task_accuracy"] is not None or result["final_test_opened"] or result["speed_tested"]):
        raise ValueError("preregistered systems gate differs")
    parity = torch.load(run.DEST / "forward_parity.pt", map_location="cpu", weights_only=True)
    if (parity["hard_word_logits"].shape != (2, 128, 30528)
            or not torch.equal(parity["hard_word_logits"], parity["bridged_word_logits"])
            or not torch.isfinite(parity["hard_word_logits"]).all()
            or parity["source_only_codes"].shape != (2, 641)):
        raise ValueError("complete native hard word-logit parity differs")
    gradients = torch.load(run.DEST / "word_only_gradients.pt", map_location="cpu", weights_only=True)
    proof = q.read(run.DEST / "gradient_proof.json")
    actual = {name: {p: active(values, p) for p in run.GROUPS} for name, values in gradients.items()}
    if (set(gradients["hard"]) != set(gradients["bridged"])
            or any(actual["hard"][p] for p in run.GROUPS[:2]) or not all(actual["hard"][p] for p in run.GROUPS[2:])
            or not all(actual["bridged"].values())
            or proof["hard_active_groups"] != actual["hard"] or proof["bridged_active_groups"] != actual["bridged"]
            or not proof["word_logits_bitwise_equal"] or not proof["old_frozen_gradients_absent"]
            or proof["diagnostic_optimizer_steps"] != 0):
        raise ValueError("word-only gradient routing proof differs")
    for values in gradients.values():
        if any(v is not None and not torch.isfinite(v).all() for v in values.values()):
            raise ValueError("nonfinite saved gradient")
    saved = torch.load(run.DEST / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    parameters = attention_parameters(saved["attention"])
    digest = hashlib.sha256()
    for name, value in sorted(parameters.items()):
        digest.update(name.encode())
        digest.update(value.contiguous().numpy().tobytes())
    if (sum(v.numel() for v in parameters.values()) != 11796480 or set(parameters) != set(gradients["bridged"])
            or digest.hexdigest() != result["final_attention_sha256"] or digest.hexdigest() == run.BASE_ATTENTION
            or file_hash(run.DEST / "model.pt") != result["checkpoint_sha256"] or saved["updates"] != 2
            or saved["producer_fingerprint"] != producer["fingerprint"]
            or any(int(v["step"]) != 2 for v in saved["optimizer"]["state"].values())):
        raise ValueError("actual two-update attention/optimizer checkpoint differs")
    identities = producer["config"]["probes"]
    codes = {r["id"]: r["codes"] for r in q.read(run.training.CACHE / "outputs.json")}
    clean = torch.tensor([codes[identities[i % 2]] for i in range(4)], dtype=torch.long)
    neighbor = torch.Generator().manual_seed(105142)
    trace = q.read(run.DEST / "trace.json")
    if len(trace) != 2:
        raise ValueError("two trace rows required")
    for step, row in enumerate(trace, 1):
        _, expected = run.training.objective.corrupt_history(clean, 1024, neighbor)
        detail = row["losses"]
        if (row["step"] != step or not math.isfinite(row["gradient_norm"]) or not all(row["active_groups"].values())
                or detail["corruption"] != expected or detail["temperature"] != 1.
                or detail["auxiliary_concept_weight"] != .25 or detail["auxiliary_word_weight"] != 1.
                or not detail["word_gradient_into_concept_attention"]
                or any(not math.isfinite(v) or v < 0 for k, v in detail.items() if k.endswith("_loss"))
                or len(detail["source_only_history_codes"]) != 4
                or any(len(v) != 641 or any(type(x) is not int or not 0 <= x < 8192 for x in v)
                    for v in detail["source_only_history_codes"])):
            raise ValueError("combined clean/noisy/surrogate-history objective trace differs")
    if not torch.equal(neighbor.get_state(), saved["neighbor_rng"]):
        raise ValueError("neighborhood RNG replay differs")
    artifacts = {str(p): file_hash(p) for p in run.DEST.rglob("*") if p.is_file() and "producer_source" not in p.parts}
    artifacts[str(Path(__file__))] = file_hash(__file__)
    campaign = {"schema": "nal.ncm-lm1b-word-gradient-pilot.v1", "hypothesis_id": run.HYPOTHESIS,
        "status": "completed", "classification": "systems_hard_forward_exact_word_gradient_restored",
        "systems_only_not_task_quality": True, "word_only_gradient_groups": actual,
        "hard_forward_word_logits_bitwise_equal": True, "old_frozen_states_unchanged": frozen,
        "strict_reload_passed": True, "updates": 2, "training_microbatch": 4,
        "peak_allocated_bytes": result["peak_allocated_bytes"], "task_accuracy": None,
        "confirmed": False, "final_test_opened": False, "speed_tested": False, "artifacts": artifacts}
    RESULTS.mkdir()
    path = RESULTS / "campaign_results.json"
    write_json(path, campaign)
    REPORT.write_text(
        "# Word-loss gradient through native NCM codebooks: systems result\n\n"
        f"Hypothesis `{run.HYPOTHESIS}`. Classification **{campaign['classification']}**.\n"
        f"Campaign `{path}`, SHA256 `{file_hash(path)}`.\n\n"
        "On two complete TRAIN sources, the hard and surrogate paths produced\n"
        "bitwise identical FP32 word logits at every position. Saved gradient tensors\n"
        "show that ordinary hard-code word loss reaches only word-source attention;\n"
        "the surrogate also reaches concept-source memory and adapters. The CPU\n"
        "auditor replayed those tensors, complete logits, both noise batches and\n"
        "the actual two-update optimizer checkpoint. No diagnostic optimizer step\n"
        "preceded the forward/gradient comparison.\n\n"
        "The combined clean/noisy/source-only-history objective completed two real\n"
        "BF16 microbatch4 updates, with exact fresh attention/optimizer/allRNG/hard\n"
        "generation reload. Frozen generator and codec fingerprints stayed unchanged.\n"
        f"Peak allocated CUDA memory was {result['peak_allocated_bytes']} bytes.\n\n"
        "This is a biased surrogate backward rule with an unchanged hard forward.\n"
        "It proves a trainable gradient path, not high-quality summarization. Only\n"
        "the existing 11,796,480 attention parameters updated. The original Llama\n"
        "body stayed unloaded; all pretrained NCM weights stayed frozen. The\n"
        "two-update checkpoint is not a full-training seed. Both readout policies\n"
        "on the evaluated fixed model scored 0/128 under the qualified semantic\n"
        "judge. New task accuracy is unmeasured, final sources stay closed,\n"
        "speed is unmeasured and the95%objective remains unmet.\n")
    print(json.dumps({"classification": campaign["classification"], "campaign_sha256": file_hash(path),
        "task_accuracy": None}), flush=True)


def register():
    path = RESULTS / "campaign_results.json"
    register_records(path, REPORT, RESULTS, {run.HYPOTHESIS: {
        "name": "Exact hard NCM forward with word gradient into concept attention",
        "description": "Systems-only native-codebook surrogate restores generated-word loss gradients into concept-source attention while preserving complete hard forward logits; two BF16 updates and exact reload, no semantic task qualification.",
        "classification": "systems_hard_forward_exact_word_gradient_restored", "systems_only": True,
        "hard_forward_word_logits_bitwise_equal": True, "task_accuracy_measured": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    {"finalize": finalize, "register": register}[args.stage]()
