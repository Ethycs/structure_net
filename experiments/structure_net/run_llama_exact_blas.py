"""Backend preference follow-up, reusing frozen exactness and timing contracts."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.structure_net import llama_cotenancy_common as common
from experiments.structure_net import llama_exact_blas as variants
from experiments.structure_net.llama_tensorized_model import TensorizedLlama
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json
from experiments.structure_net.run_llama_exact_speed import (
    ORIGINAL, ORIGINAL_SHA256, FIXTURES, GRID, benchmark_pair, cpu_tree,
    memory, generation_trace, exact_controls, measure,
)

HYPOTHESIS_ID = "llama32-3b-exact-blas-iteration-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-10_llama32-exact-blas-plan.md")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    common.setup()
    if sha256(ORIGINAL) != ORIGINAL_SHA256:
        raise ValueError("original checkpoint changed")
    torch.backends.cuda.preferred_blas_library("cublas")
    common.HYPOTHESIS_ID = HYPOTHESIS_ID
    sources = [Path(__file__), Path(variants.__file__), PLAN, FIXTURES,
               Path("experiments/structure_net/run_llama_exact_speed.py"),
               Path("experiments/structure_net/llama_exact_projections.py"),
               Path("experiments/structure_net/llama_cuda_graph.py"),
               Path("experiments/structure_net/run_llama_exact_deployment.py"),
               Path("experiments/structure_net/campaign_llama_cotenancy.py"),
               Path(torch.__file__).parent / "backends/cuda/__init__.py"]
    producing = common.producer(args.output, "exact_blas_pilot" if args.pilot else "exact_blas_full", sources,
        {"checkpoint": str(ORIGINAL), "checkpoint_sha256": ORIGINAL_SHA256, "fixture_sha256": sha256(FIXTURES),
         "candidates": variants.CANDIDATES, "grid": GRID, "pilot": args.pilot, "samples": 1 if args.pilot else 10,
         "baseline": "original cuBLAS CUDA graph", "intervention": "cuBLASLt preference for projections", "physical_models": 1})
    try:
        tiny = []
        if args.pilot:
            torch.manual_seed(779)
            cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                              num_attention_heads=4, num_key_value_heads=2, head_dim=16, eos_token_id=2)
            hf = LlamaForCausalLM(cfg).eval().half()
            reference = TensorizedLlama.from_hf(hf, device="cuda", attention="flash")
            for name in variants.CANDIDATES:
                candidate = variants.BlasLlama(reference, name)
                for steps in (0, 4):
                    row, baseline, captured, _ = exact_controls(reference, candidate, [list(range(16)), list(range(20, 36))], steps, args.output / f"tiny_{name}_{steps}.pt")
                    tiny.append({"candidate": name, "steps": steps, **row})
                    del baseline, captured
                del candidate
            del reference, hf
            gc.collect()
            torch.cuda.empty_cache()
            write_json(args.output / "tiny_lifecycle.json", {"status": "systems_lifecycle_only_not_quality_evidence", "rows": tiny})
        payload = torch.load(ORIGINAL, map_location="cpu", weights_only=False, mmap=True)
        reference = TensorizedLlama(payload)
        sequences = json.loads(FIXTURES.read_text())["benchmark_ids"]
        anchors = {}
        for length, steps in GRID:
            records = []
            for sequence in benchmark_pair(sequences, length):
                ids = torch.tensor([sequence], device="cuda")
                expected = generation_trace(reference, ids, steps) if steps else reference(ids, use_cache=True)
                records.append({"ids": ids.cpu(), "expected": cpu_tree(expected)})
                del expected, ids
            anchors[f"{length}_{steps}"] = records
        anchor_path = args.output / "original_reference_anchors.pt"
        torch.save(anchors, anchor_path)
        torch.cuda.empty_cache()
        rows, setup = [], {}
        for name in variants.CANDIDATES:
            start = time.perf_counter()
            candidate = variants.BlasLlama(reference, name)
            setup[name] = {"packing_seconds": time.perf_counter()-start, "memory": memory(), "weight_bits_verified": True}
            for length, steps in GRID:
                rows.append(measure(reference, candidate, sequences, args.output, length, steps, 1 if args.pilot else 10, anchors[f"{length}_{steps}"]))
            del candidate
            gc.collect()
            torch.cuda.empty_cache()
        after = common.gpu_state()
        if not common.uncontended(producing["gpu_before"], after):
            raise RuntimeError("GPU contention changed")
        eligible = [n for n in variants.CANDIDATES if all(r["all_bitwise"] for r in rows if r["candidate"] == n)]
        promoted = [{"candidate": r["candidate"], "length": r["length"], "decode_steps": r["decode_steps"]}
                    for r in rows if r["candidate"] in eligible and r["speed_gate"]] if not args.pilot else []
        write_json(args.output / "result.json", {"schema": "nal.llama-exact-blas.v1", "hypothesis_id": HYPOTHESIS_ID,
            "status": "systems_lifecycle_only_not_quality_evidence" if args.pilot else "completed",
            "producer_fingerprint": producing["fingerprint"], "workloads": rows, "setup": setup,
            "tiny_lifecycle": tiny, "eligible_candidates": eligible, "promoted": promoted,
            "anchor_artifact": str(anchor_path), "anchor_sha256": sha256(anchor_path),
            "parameter_bytes": reference.storage_bytes(), "gpu_after": after, "uncontended": True,
            "application_accuracy_tested": False, "weight_compression_used": False, "two_for_one_supported": False})
    except Exception as error:
        write_json(args.output / "failure.json", {"status": "failed", "type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
