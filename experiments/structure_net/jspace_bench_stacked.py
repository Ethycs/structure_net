"""Benchmark dense stacking of the frozen 73-channel polynomial core."""
import argparse
import gc
import json
from pathlib import Path
import shutil

import torch

from . import jspace_bench_tensor as base
from .jspace_bench_tensor_v2 import DEFAULT_ROOT as FIT_ROOT
from .jspace_bench_covered import DEFAULT_ROOT as COVERED_ROOT, PackedResidualMLP
from .jspace_packed_core import StackedCoreMLP
from .jspace_bench_data import sha, write


DEFAULT_ROOT = base.ROOT / "data/experiments/jspace_bench_tensor/20261113_stacked_v1"
PROTOCOL = base.ROOT / "docs/07 - Status Reports/jspace/2026-11-13_jspace-bench-stacked-plan.md"
STACKING = {
    "covered_channels": "Frozen 20/34/19 channels from strict coverage follow-up",
    "core": "Stack compact polynomial outputs with nonlinear residual into one dense down projection",
    "gate_up": "Stack projections and pad retained rows to a multiple of 16",
    "kernel": "Triton fused BF16 SwiGLU and FP32 polynomial contraction; BF16 packed output",
    "controls": ["original", "previous_factorized", "stacked_dense_original", "stacked_polynomial_core"],
    "generic_optimization": "All 36 MLPs stack gate/up and fuse activation in both stacked controls",
    "attention": "Unchanged eager attention",
    "interpretation": "Engineering optimization with accepted small task loss; no coverage or coefficient search",
    "known_before_freeze": "Original 210/256; previous factorized core 209/256 and about 2.5% slower",
    "independent_confirmation": False,
}


def prepare(root):
    base.cpu_only()
    if root.exists():
        raise FileExistsError(root)
    parent = base.verify(COVERED_ROOT)
    base.checked(COVERED_ROOT, "covered")
    extras = [Path(__file__), PROTOCOL, base.ROOT / "experiments/structure_net/jspace_packed_core.py",
              base.ROOT / "tests/structure_net/test_jspace_packed_core.py"]
    value = {**parent, "stacking_config": STACKING,
             "sources": {**parent["sources"], **{str(p): sha(p) for p in extras}},
             "input_sha256": {**parent["input_sha256"], **{str(COVERED_ROOT/n): sha(COVERED_ROOT/n)
                                  for n in ("plan.json", "covered_manifest.json", "coverage.json", "result.json")}}}
    value.pop("scientific_fingerprint")
    value["scientific_fingerprint"] = base.digest(value)
    root.mkdir(parents=True)
    for name in value["sources"]:
        p = Path(name)
        target = root / "producer" / p.relative_to(base.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    shutil.copy2(COVERED_ROOT / "tasks.json", root / "tasks.json")
    write(root / "plan.json", value)
    return {"fingerprint": value["scientific_fingerprint"], "stacking": STACKING}


@torch.inference_mode()
def kernel_check(stage, original, fit, removed):
    rows = []
    for selected in ([], removed):
        packed = StackedCoreMLP(original, fit, selected).cuda()
        for tokens in (1, 37):
            x = torch.randn(1, tokens, original.gate_proj.in_features, device="cuda:0", dtype=torch.bfloat16)*.3
            expected, actual = packed.reference(x), packed(x)
            error = (actual.float()-expected.float()).square().mean().sqrt()
            relative = float(error/expected.float().square().mean().sqrt().clamp_min(1e-8))
            if not torch.isfinite(actual).all() or relative > .01:
                raise ValueError(f"Fused CUDA packing failed its numerical shakedown: {relative}")
            rows.append({"removed": len(selected), "tokens": tokens, "relative_rms_error": relative,
                         "maximum_absolute_error": float((actual.float()-expected.float()).abs().max())})
        del packed
    write(stage.path / "kernel_shakedown.json", {"scope": "systems_numerics_only", "cases": rows})


@torch.inference_mode()
def microseconds(module, x):
    for _ in range(10):
        module(x)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        module(x)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)*1000/100


def run(root):
    if base.verify(root)["stacking_config"] != STACKING:
        raise ValueError("Stacking configuration changed")
    stage = base.Stage(root, "stacked")
    fit = torch.load(FIT_ROOT / "fit.pt", map_location="cpu", weights_only=True)
    coverage = json.loads((COVERED_ROOT / "coverage.json").read_text())
    removed = {int(k): v["indices"] for k, v in coverage.items()}
    originals = {k: block.mlp for k, block in enumerate(stage.model.model.layers)}
    kernel_check(stage, originals[8], fit["8"], removed[8])
    metrics, timings, micro = {}, {}, {}

    def install(arm):
        for layer in originals:
            stage.model.model.layers[layer].mlp.to("cpu")
            original = originals[layer].cpu()
            if arm == "previous_factorized" and layer in removed:
                replacement = PackedResidualMLP(original, fit[str(layer)], removed[layer], compiled=False)
            elif arm.startswith("stacked"):
                covered = arm == "stacked_polynomial_core" and layer in removed
                replacement = StackedCoreMLP(original, fit[str(layer)] if covered else None,
                                             removed[layer] if covered else [])
            else:
                replacement = original
            stage.model.model.layers[layer].mlp = replacement.cuda()
        gc.collect()
        torch.cuda.empty_cache()

    baseline = base.run_cases(stage, stage.tasks["evaluation"], "original_evaluation")
    metrics["original"] = base.accuracy(baseline)
    timings["original_before"] = base.timing(stage, "original_before")
    for arm in ("previous_factorized", "stacked_dense_original", "stacked_polynomial_core"):
        install(arm)
        rows = base.run_cases(stage, stage.tasks["evaluation"], arm+"_evaluation")
        metrics[arm] = {"accuracy": base.accuracy(rows), "comparison": base.comparison(baseline, rows),
                        "resident_model_tensor_bytes": base.model_bytes(stage.model)}
        timings[arm] = base.timing(stage, arm)
        micro[arm] = {}
        for layer in base.CONFIG["layers"]:
            module = stage.model.model.layers[layer].mlp
            micro[arm][str(layer)] = {str(tokens): microseconds(module, torch.randn(
                1, tokens, 2048, device="cuda:0", dtype=torch.bfloat16)*.3) for tokens in (1, 128)}
        if arm == "stacked_polynomial_core":
            base.save(root / "compressed_modules.pt", {
                str(k): {name: value.cpu() for name, value in stage.model.model.layers[k].mlp.state_dict().items()}
                for k in base.CONFIG["layers"]})
    install("original")
    timings["original_after"] = base.timing(stage, "original_after")
    micro["original"] = {str(layer): {str(tokens): microseconds(originals[layer], torch.randn(
        1, tokens, 2048, device="cuda:0", dtype=torch.bfloat16)*.3) for tokens in (1, 128)}
        for layer in base.CONFIG["layers"]}
    value = {"metrics": metrics, "timings": timings, "module_microseconds": micro,
             "removed_counts": {str(k): len(v) for k, v in removed.items()}, "stacking_config": STACKING}
    write(root / "evaluation.json", value)
    return stage.finish(["evaluation.json", "compressed_modules.pt"], stacking_config=STACKING)


def analyze(root):
    base.cpu_only()
    plan = base.verify(root)
    base.checked(root, "stacked")
    evidence = json.loads((root / "evaluation.json").read_text())
    timing = evidence["timings"]
    packed = timing["stacked_polynomial_core"]["tokens_per_second"]
    original = (timing["original_before"]["tokens_per_second"]+timing["original_after"]["tokens_per_second"])/2
    value = {"study_id": base.STUDY_ID, "status": "completed", "confirmed": False,
             "classification": "dense_stacked_core_engineering_measurement",
             "scientific_fingerprint": plan["scientific_fingerprint"], "result": evidence,
             "speedup_vs_previous_factorized": packed/timing["previous_factorized"]["tokens_per_second"],
             "speedup_vs_original": packed/original,
             "speedup_vs_matched_stacked_original": packed/timing["stacked_dense_original"]["tokens_per_second"],
             "scope": "Same four-task held-out pool; accepted small approximation loss; no independent confirmation"}
    write(root / "result.json", value)
    write(root / "campaign_results.json", {**value, "result_sha256": sha(root / "result.json")})
    return {k: v for k, v in value.items() if k != "result"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "analyze"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "run": run, "analyze": analyze}[args.stage](args.root), allow_nan=False), flush=True)
