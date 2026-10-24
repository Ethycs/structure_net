"""Audit/register the real native NCM adapter pilot as systems evidence only."""
import argparse
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_lm1b_source_attention_pilot as run
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json

RESULTS = run.ROOT / "20261024_lm1b_source_attention_cuda_pilot_results_v1"
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-lm1b-source-attention-cuda-pilot.md")


def finalize():
    if torch.cuda.is_initialized(): raise ValueError("CPU audit only")
    run.native.vendor.terminal(run.DEST)
    if RESULTS.exists() or REPORT.exists(): raise ValueError("fresh pilot audit required")
    result, producer = run.q.read(run.DEST / "result.json"), run.q.read(run.DEST / "producer.json")
    expected_hashes = {"generator": "60316ea14c56e6276be581112a2a1d81437e3ad8bb4ea3ddf8a98b946805e26a",
        "codec": "9aace450a49a7dfb44ba1eb00a19974a50d67062ad9f2202a6a2361ea4344e32"}
    if (result["status"] != "systems_native_source_attention_gradients_and_reload_passed_not_quality"
            or result["released_states_unchanged"] != expected_hashes or result["trainable_new_parameters"] != 8257536
            or result["real_backward_updates"] != 2 or result["peak_allocated_bytes"] > 8_000_000_000
            or not result["zero_output_full641_logit_parity_exact"]
            or not result["teacher_logits_and_source_only_generation_reload_exact"]
            or result["task_accuracy"] is not None or result["speed_tested"] or result["final_test_opened"]
            or file_hash(run.DEST / "adapter.pt") != result["checkpoint_sha256"]):
        raise ValueError("native generator lifecycle evidence differs")
    traces = run.q.read(run.DEST / "trace.json")
    if len(traces) != 2: raise ValueError("two actual gradient updates required")
    for step, trace in enumerate(traces, 1):
        if (trace["step"] != step or trace["case_id"] != producer["config"]["probes"][step-1]
                or not math.isfinite(trace["loss"]) or not math.isfinite(trace["gradient_norm"])
                or not any("out.weight" in n for n in trace["nonzero_gradient_names"])
                or (step == 2 and not any(n.startswith("source_memory.") for n in trace["nonzero_gradient_names"]))):
            raise ValueError("actual new attention gradient path or finite trace differs")
    saved = torch.load(run.DEST / "adapter.pt", map_location="cpu", weights_only=True)
    if set(saved["adapter"]) != {"source_memory", "adapters"}:
        raise ValueError("adapter-only checkpoint required")
    if sum(v.numel() for s in saved["adapter"].values() for v in s.values()) != 8257536:
        raise ValueError("unexpected saved trainable parameter count")
    if any(int(v["step"]) != 2 for v in saved["optimizer"]["state"].values()):
        raise ValueError("two-step AdamW state required")
    outputs = torch.load(run.DEST / "reload_outputs.pt", map_location="cpu", weights_only=True)
    if (outputs["teacher_logits"].shape != (1, 641, 8192) or not torch.isfinite(outputs["teacher_logits"]).all()
            or outputs["source_only_generated_codes"].shape != (1, 641)
            or (outputs["source_only_generated_codes"] < 0).any() or (outputs["source_only_generated_codes"] >= 8192).any()):
        raise ValueError("full native teacher logits and generated hierarchy required")
    paths = [Path(__file__), run.PLAN, REPORT]
    hashes = {**producer["input_hashes"], **producer["source_hashes"]}
    hashes.update({str(p): file_hash(p) for p in run.DEST.rglob("*") if p.is_file() and "producer_source" not in p.parts})
    hashes[str(Path(__file__))] = file_hash(__file__)
    campaign = {"hypothesis_id": run.HYPOTHESIS, "classification": "systems_native_NCM_source_attention_lifecycle_passed_not_quality",
        "result": result, "trace": traces, "native_generator_qualified_for_task": False,
        "source_only_task_accuracy": None, "task_accuracy": None, "confirmed": False,
        "underpowered": True, "final_test_opened": False, "speed_tested": False, "artifacts": hashes}
    RESULTS.mkdir()
    path = RESULTS / "campaign_results.json"
    write_json(path, campaign)
    REPORT.write_text(f"""# Native NCM source-attention attachment: CUDA lifecycle

Date: 2026-10-24. Hypothesis `{run.HYPOTHESIS}`.
Classification **systems_native_NCM_source_attention_lifecycle_passed_not_quality**.
Campaign `{path}`, SHA256 `{file_hash(path)}`.

The real released 97M LM1B generator and 111M codec accepted the new source
attention. Zero-initialized output adapters preserved full 641-position concept
logits exactly on two TRAIN cases with 486 and 2744 full source-feature positions.
Two real gradient updates reached only the 8,257,536 new attention parameters;
released generator/codec states stayed unchanged. Native prefix, learned root,
post-upsampling convolutions and shared output heads remained intact.

A fresh released-model load plus adapter/optimizer/RNG restore exactly reproduced
teacher logits and deterministic source-only concept generation. Source memory
cleared between requests. Peak allocation was {result['peak_allocated_bytes']} bytes
on physical GPU0 RTX3060, FP32/TF32 off. The original Llama body was absent and
its SHA-verified cached TRAIN features supplied the source memory.

This is systems evidence only: two cases, two updates, no semantic judging, no
new source-only task accuracy, no trained summarizer or speed claim. The failed
codec branches and previous conditional student's 0/128 result remain unchanged.
All final sources stayed closed. The full 95% task goal remains unmet.
""")
    print(json.dumps({"classification": campaign["classification"], "campaign_sha256": file_hash(path),
        "task_accuracy": None}), flush=True)


def register():
    register_records(RESULTS / "campaign_results.json", REPORT, RESULTS, {run.HYPOTHESIS: {
        "name": "Native released NCM source-attention CUDA lifecycle",
        "description": "Two TRAIN systems cases: native full-logit zero-output parity, new-only gradients, unchanged released generator/codec and exact fresh reload/source-only generation. Systems evidence, no semantic qualification or speed claim.",
        "classification": "systems_native_NCM_source_attention_lifecycle_passed_not_quality",
        "source_only_task_accuracy_measured": False, "systems_lifecycle_only": True}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    (finalize if args.stage == "finalize" else register)()
