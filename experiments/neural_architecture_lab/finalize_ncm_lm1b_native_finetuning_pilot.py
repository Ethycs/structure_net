"""Independent CPU replay of native fine-tuning parity, gradients and actual states."""
import argparse
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_lm1b_native_finetuning_pilot as run
from experiments.neural_architecture_lab.ncm_result_registry import register_records

RESULTS = run.ROOT / "20261024_lm1b_native_finetuning_cuda_pilot_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-lm1b-native-finetuning-pilot.md")


def require_generation(values, batch):
    shapes = {"codes": (batch, 641), "raw_word_ids": (batch, 128)}
    if set(values) != set(shapes):
        raise ValueError("complete hard-generation fields required")
    for key, shape in shapes.items():
        value = torch.as_tensor(values[key])
        maximum = 8192 if key == "codes" else 30528
        if value.dtype != torch.long or value.shape != shape or bool((value < 0).any()) or bool((value >= maximum).any()):
            raise ValueError("complete native hard-generation shape/vocabulary required")


def finalize():
    if RESULTS.exists() or REPORT.exists() or torch.cuda.is_initialized():
        raise ValueError("fresh CPU native lifecycle audit required")
    run.native.vendor.terminal(run.DEST)
    q = run.q
    result, producer = q.read(run.DEST / "result.json"), q.read(run.DEST / "producer.json")
    if (result["status"] != "systems_native_NCM_finetuning_and_reload_passed_task_accuracy_unmeasured"
            or result["updates"] != 2 or result["trainable_parameters"] != 127732736
            or result["protected_codec_hashes"] != {"encoder": run.ENCODER, "VQ": run.VQ}
            or not result["native_root_query_updated"] or not result["exact_initial_native_forward_and_hard_generation"]
            or not result["strict_fullstate_optimizer_allRNG_hard_generation_reload"]
            or result["peak_allocated_bytes"] > 8_000_000_000 or result["Llama_body_loaded"]
            or result["task_accuracy"] is not None or result["final_test_opened"] or result["speed_tested"]):
        raise ValueError("preregistered native lifecycle result differs")
    parity = torch.load(run.DEST / "parity.pt", map_location="cpu", weights_only=True, mmap=True)
    for key in ("generated_before", "generated_after"):
        require_generation(parity[key], 2)
    shapes = {"concept_logits": (2, 641, 8192), "word_logits": (2, 128, 30528)}
    if (set(parity["before"]) != set(shapes) or set(parity["after"]) != set(shapes)
            or any(parity["before"][k].shape != shape or not torch.equal(parity["before"][k], parity["after"][k])
                   or not torch.isfinite(parity["before"][k]).all() for k, shape in shapes.items())
            or not run.same(parity["generated_before"], parity["generated_after"])):
        raise ValueError("complete bitwise initial native forward/hard-generation parity differs")
    consistency = torch.load(run.DEST / "teacher_stepwise_parity.pt", map_location="cpu", weights_only=True, mmap=True)
    expected_consistency = {
        "same_history_max_abs_logit_difference": float((consistency["teacher_logits"] - consistency["step_logits"]).abs().max()),
        "same_history_argmax_disagreements": int((consistency["teacher_logits"].argmax(-1) != consistency["step_logits"].argmax(-1)).sum()),
        "same_history_logits_allclose": torch.allclose(consistency["teacher_logits"], consistency["step_logits"], rtol=1e-4, atol=1e-4)}
    if (consistency["teacher_logits"].shape != shapes["concept_logits"]
            or not torch.equal(consistency["teacher_logits"], parity["before"]["concept_logits"])
            or not torch.equal(consistency["forced_codes"], consistency["target_codes"])
            or not expected_consistency["same_history_logits_allclose"]
            or q.read(run.DEST / "teacher_stepwise_parity.json") != expected_consistency
            or any(result.get(k) != v for k, v in expected_consistency.items())):
        raise ValueError("actual identical-history joint/stepwise consistency differs")
    saved = torch.load(run.DEST / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    gradients = torch.load(run.DEST / "full_native_gradients.pt", map_location="cpu", weights_only=True, mmap=True)
    model = run.head.from_domain(device="cpu")
    warm = torch.load(run.WARM / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    model.load_attention_state(warm["attention"])
    if run.native.vendor.module_hash(model) != run.WARM_MODEL:
        raise ValueError("actual fixed warm source model differs")
    run.mechanism.enable(model).eval()
    before = run.trainable_hash(model)
    root = model.concepts.generator.l0_queries.detach().clone()
    initial_native = {"generator": run.native.vendor.module_hash(model.concepts.generator), "word_decoder": run.native.vendor.module_hash(model.codec.decoder)}
    names = {n: v for n, v in model.named_parameters() if v.requires_grad}
    if (set(names) != set(gradients) or sum(v.numel() for v in names.values()) != 127732736
            or any(v is not None and (v.shape != names[n].shape or not torch.isfinite(v).all()) for n, v in gradients.items())):
        raise ValueError("all exact native trainable gradient tensors required")
    active = {prefix: any(n.startswith(prefix) and v is not None and bool(v.abs().sum() > 0)
                         for n, v in gradients.items()) for prefix in run.GROUPS}
    if not all(active.values()):
        raise ValueError("native/root/word/source-attention saved gradients inactive")
    model.load_state_dict(saved["model"], strict=True)
    after = run.trainable_hash(model)
    final_native = {"generator": run.native.vendor.module_hash(model.concepts.generator), "word_decoder": run.native.vendor.module_hash(model.codec.decoder)}
    if (before != result["initial_trainable_sha256"] or after != result["final_trainable_sha256"] or before == after
            or initial_native != result["initial_native_hashes"] or final_native != result["final_native_hashes"]
            or any(initial_native[k] == final_native[k] for k in initial_native)
            or torch.equal(root, model.concepts.generator.l0_queries.detach())
            or run.protected(model) != {"encoder": run.ENCODER, "VQ": run.VQ}
            or saved["step"] != 2 or saved["producer_fingerprint"] != producer["fingerprint"]
            or result["checkpoint_sha256"] != run.file_hash(run.DEST / "model.pt")
            or any(int(v["step"]) != 2 for v in saved["optimizer"]["state"].values())):
        raise ValueError("actual two-update native parameters/protected states/optimizer differ")
    if not saved["optimizer"]["state"]:
        raise ValueError("actual optimizer updates required")
    probes = q.read(run.DEST / "reload_probes.json")
    if len(probes) != 2:
        raise ValueError("both complete representative reload probes required")
    for value in probes:
        require_generation(value, 1)
    neighbor = torch.Generator().manual_seed(111142)
    codes = {r["id"]: r["codes"] for r in q.read(run.training.CACHE / "outputs.json")}
    ids = producer["config"]["source_ids"]
    clean = torch.tensor([codes[ids[i % 2]] for i in range(4)], dtype=torch.long)
    trace = q.read(run.DEST / "trace.json")
    if len(trace) != 2:
        raise ValueError("two actual update traces required")
    for step, row in enumerate(trace, 1):
        _, proof = run.objective.original.corrupt_history(clean, 5120 + step, neighbor)
        detail = row["losses"]
        if (row["step"] != step or not math.isfinite(row["gradient_norm"]) or not all(row["active_groups"].values())
                or detail["corruption"] != proof or detail["temperature"] != 1.
                or detail["auxiliary_concept_weight"] != .25 or detail["auxiliary_word_weight"] != 1.
                or not detail["word_gradient_into_concept_attention"]
                or len(detail["source_only_history_codes"]) != 4
                or any(len(h) != 641 or any(type(v) is not int or not 0 <= v < 8192 for v in h) for h in detail["source_only_history_codes"])
                or any(not math.isfinite(v) or v < 0 for k, v in detail.items() if k.endswith("_loss"))):
            raise ValueError("native objective/noise/history trace differs")
    if not torch.equal(neighbor.get_state(), saved["neighbor_rng"]):
        raise ValueError("saved native neighborhood RNG replay differs")
    model.close()
    artifacts = {str(p): run.file_hash(p) for p in run.DEST.rglob("*") if p.is_file() and "producer_source" not in p.parts}
    artifacts[str(Path(__file__))] = run.file_hash(__file__)
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": "systems_native_NCM_finetuning_and_reload_passed_not_task_quality",
        "systems_only": True, "updates": 2, "trainable_parameters": 127732736, "saved_gradient_groups": active,
        "native_generator_and_word_decoder_actually_updated": True, "native_root_query_actually_updated": True,
        "protected_codec_hashes": result["protected_codec_hashes"], "exact_initial_forward_and_hard_generation": True,
        "strict_fullstate_optimizer_allRNG_hard_generation_reload": True, "peak_allocated_bytes": result["peak_allocated_bytes"],
        **expected_consistency,
        "task_accuracy": None, "confirmed": False, "final_test_opened": False, "speed_tested": False, "artifacts": artifacts}
    RESULTS.mkdir()
    path = RESULTS / "campaign_results.json"
    run.write_json(path, campaign)
    REPORT.write_text(
        "# Native NCM fine-tuning CUDA lifecycle\n\n"
        f"Hypothesis `{run.HYPOTHESIS}`. Campaign `{path}`, SHA256 `{run.file_hash(path)}`.\n\n"
        "The released native generator, domain word decoder and existing source-attention\n"
        "weights completed two BF16 microbatch4 optimizer updates,127,732,736trainable\n"
        "parameters in total. Both native generator and decoder actually changed; the\n"
        "learned root query received nonzero gradients and changed. The CPU auditor\n"
        "checked the full saved gradient tensors, exact initial concept/word-logit and\n"
        "hard-generation parity, both noise traces, actual optimizer states and native\n"
        "parameter hashes. Encoder/VQ target geometry stayed frozen.\n\n"
        f"Exact fullstate/optimizer/allRNG/hard-generation reload passed. Peak CUDA allocation: {result['peak_allocated_bytes']} bytes.\n\n"
        "The original Llama body/MLPs stayed unloaded and unchanged. No weights were\n"
        "randomly reinitialized. This is a systems lifecycle, not semantic task quality\n"
        "or speed evidence. The two-update checkpoint is not a full-training seed.\n"
        "Larger-data training and the95%source-disjoint complete-summary objective\n"
        "remain pending; final sources stay unopened.\n")
    print(json.dumps({"classification": campaign["classification"], "campaign_sha256": run.file_hash(path), "task_accuracy": None}), flush=True)


def register():
    register_records(RESULTS / "campaign_results.json", REPORT, RESULTS, {run.HYPOTHESIS: {
        "name": "Native pretrained NCM fine-tuning with frozen Llama body and target codec",
        "description": "Systems-only two-update BF16 native generator, learned root, word decoder and source-attention fine-tuning; frozen encoder/VQ and original Llama body, complete hard-forward parity and exact reload, no task-quality claim.",
        "classification": "systems_native_NCM_finetuning_and_reload_passed_not_task_quality",
        "systems_only": True, "native_weights_updated": True, "task_accuracy_measured": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    {"finalize": finalize, "register": register}[args.stage]()
