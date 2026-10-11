"""Compare candidate graph execution with the current unchanged graph baseline."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.structure_net import llama_cotenancy_common as common
from experiments.structure_net import llama_exact_projections as projections
from experiments.structure_net.llama_cuda_graph import CapturedCall, bitwise_tree, clone_tree
from experiments.structure_net.llama_tensorized_model import TensorizedLlama, generate
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json
from experiments.structure_net.run_llama_exact_deployment import (
    ORIGINAL, ORIGINAL_SHA256, FIXTURES, benchmark_pair, cpu_tree, memory, paired,
)

HYPOTHESIS_ID = "llama32-3b-exact-speed-iteration-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-10_llama32-exact-speed-plan.md")
AMENDMENT = Path("docs/07 - Status Reports/2026-10-10_llama32-exact-speed-amendment.md")
STORAGE_AMENDMENT = Path("docs/07 - Status Reports/2026-10-10_llama32-exact-speed-storage-amendment.md")
GRID = ((128, 0), (512, 0), (128, 32))


def generation_trace(model, ids, steps):
    """Same arithmetic as generate; keep every logit and final KV for auditing."""
    logits, past = model(ids, use_cache=True)
    tokens, outputs = [], []
    for index in range(steps):
        outputs.append(logits)
        token = logits[:, -1].argmax(-1)
        tokens.append(token)
        if index + 1 < steps:
            logits, past = model(token[:, None], past=past, use_cache=True)
    return torch.stack(tokens, dim=1), tuple(outputs), past


def exact_controls(reference, candidate, sequences, steps, artifact, anchors=None):
    def operation(model):
        return (lambda ids: generation_trace(model, ids, steps)) if steps else (lambda ids: model(ids, use_cache=True))
    base_fn, candidate_fn = operation(reference), operation(candidate)
    ids = torch.tensor([sequences[0]], device="cuda")
    base = CapturedCall(base_fn, ids)
    optimized = CapturedCall(candidate_fn, ids)
    torch.cuda.empty_cache()
    records = []
    previous, previous_copy = None, None
    owned = True
    for index, sequence in enumerate(sequences):
        changed = torch.tensor([sequence], device="cuda")
        baseline_eager = cpu_tree(base_fn(changed))
        expected = baseline_eager if anchors is None else anchors[index]["expected"]
        if anchors is not None and (not bitwise_tree(changed.cpu(), anchors[index]["ids"]) or not bitwise_tree(expected, baseline_eager)):
            raise RuntimeError("reference layout differs from original pre-packing anchor")
        eager = cpu_tree(candidate_fn(changed))
        baseline = cpu_tree(base(changed))
        actual = optimized(changed)
        if previous is not None:
            owned &= bitwise_tree(cpu_tree(previous), previous_copy)
        previous, previous_copy = actual, cpu_tree(actual)
        records.append({"ids": changed.cpu(), "expected": expected,
                        "baseline_eager": baseline_eager, "baseline_graph": baseline, "candidate_eager": eager,
                        "actual": previous_copy,
                        "bitwise_equal": all(bitwise_tree(expected, v) for v in (baseline, eager, previous_copy))})
        del expected, eager, baseline, actual, changed
    torch.save(records, artifact)
    valid = owned and all(r["bitwise_equal"] for r in records)
    tokens = [(r["ids"], r["expected"][0]) for r in records] if steps else []
    result = {"all_bitwise": valid, "owned_outputs": owned, "control_artifact": str(artifact),
              "control_sha256": sha256(artifact), "control_count": len(records),
              "trace_capture_seconds": {"baseline": base.capture_and_warmup_seconds,
                                        "candidate": optimized.capture_and_warmup_seconds}}
    return result, base, optimized, tokens


def measure(reference, candidate, sequences, output, length, steps, samples, anchors):
    pair = benchmark_pair(sequences, length)
    artifact = output / f"controls_{candidate.variant}_{length}_{steps}.pt"
    torch.cuda.reset_peak_memory_stats()
    controls, baseline, optimized, tokens = exact_controls(reference, candidate, pair, steps, artifact, anchors)
    row = {**controls, "candidate": candidate.variant, "length": length, "decode_steps": steps,
           "parameter_bytes": candidate.storage_bytes(), "packing_verified": candidate.packing_verified,
           "control_memory": memory()}
    if not controls["all_bitwise"]:
        row.update(status="rejected_bit_mismatch", timings=None, orders=[], speedup=None, speed_gate=False)
    else:
        if steps:
            del baseline, optimized
            gc.collect()
            torch.cuda.empty_cache()
            ids = torch.tensor([pair[0]], device="cuda")
            baseline = CapturedCall(lambda x: generate(reference, x, steps=steps, stop_eos=False), ids)
            optimized = CapturedCall(lambda x: generate(candidate, x, steps=steps, stop_eos=False), ids)
            token_records = []
            for input_ids, expected in tokens:
                input_ids = input_ids.cuda()
                a, b = baseline(input_ids), optimized(input_ids)
                token_records.append({"ids": input_ids.cpu(), "expected": expected,
                                      "baseline": a.cpu(), "actual": b.cpu()})
            token_path = output / f"tokens_{candidate.variant}_{length}_{steps}.pt"
            torch.save(token_records, token_path)
            row.update(token_artifact=str(token_path), token_sha256=sha256(token_path))
            if not all(bitwise_tree(r["expected"], r["baseline"]) and bitwise_tree(r["expected"], r["actual"]) for r in token_records):
                raise RuntimeError("generation token graph disagrees with its exact trace")
        ids = torch.tensor([pair[0]], device="cuda")
        torch.cuda.reset_peak_memory_stats()
        timings, orders = paired({"baseline_graph": lambda: baseline(ids), "candidate_graph": lambda: optimized(ids)}, samples)
        ratio = timings["baseline_graph"]["wall_median_ms"] / timings["candidate_graph"]["wall_median_ms"]
        row.update(status="exact_measured", timings=timings, orders=orders, speedup=ratio, speed_gate=ratio >= 1.10,
                   paired_memory=memory(), samples=samples,
                   timing_capture_seconds={"baseline": baseline.capture_and_warmup_seconds, "candidate": optimized.capture_and_warmup_seconds})
    write_json(output / f"workload_{candidate.variant}_{length}_{steps}.json", row)
    print(json.dumps({k: row[k] for k in ("candidate", "length", "decode_steps", "status", "speedup")}), flush=True)
    del baseline, optimized
    gc.collect()
    torch.cuda.empty_cache()
    return row


def tiny_pilot(output):
    torch.manual_seed(779)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, eos_token_id=2)
    hf = LlamaForCausalLM(cfg).eval().half()
    model = TensorizedLlama.from_hf(hf, device="cuda", attention="flash")
    rows = []
    sequences = [list(range(16)), list(range(20, 36))]
    for name in projections.CANDIDATES:
        candidate = projections.ProjectionLlama(model, name)
        for steps in (0, 4):
            r, base, optimized, _ = exact_controls(model, candidate, sequences, steps, output / f"tiny_{name}_{steps}.pt")
            rows.append({"candidate": name, "steps": steps, **r})
            del base, optimized
        del candidate
    del model, hf
    gc.collect()
    torch.cuda.empty_cache()
    return rows


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    common.setup()
    if sha256(ORIGINAL) != ORIGINAL_SHA256:
        raise ValueError("original checkpoint changed")
    common.HYPOTHESIS_ID = HYPOTHESIS_ID
    sources = [Path(__file__), Path(projections.__file__), PLAN, AMENDMENT, STORAGE_AMENDMENT, FIXTURES,
               Path("experiments/structure_net/llama_cuda_graph.py"),
               Path("experiments/structure_net/run_llama_exact_deployment.py"),
               Path("experiments/structure_net/campaign_llama_cotenancy.py")]
    producing = common.producer(args.output, "exact_speed_pilot" if args.pilot else "exact_speed_full", sources,
        {"checkpoint": str(ORIGINAL), "checkpoint_sha256": ORIGINAL_SHA256, "fixture_sha256": sha256(FIXTURES),
         "candidates": projections.CANDIDATES, "grid": GRID, "pilot": args.pilot, "samples": 1 if args.pilot else 10,
         "baseline": "original CUDA graph", "physical_models": 1, "weight_bits_unchanged": True})
    try:
        tiny = tiny_pilot(args.output) if args.pilot else None
        if tiny:
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
        for name in projections.CANDIDATES:
            start = time.perf_counter()
            candidate = projections.ProjectionLlama(reference, name, share_packed_reference=True)
            torch.cuda.synchronize()
            setup[name] = {"packing_seconds": time.perf_counter() - start, "memory": memory(),
                           "weight_bits_verified": candidate.packing_verified}
            for length, steps in GRID:
                rows.append(measure(reference, candidate, sequences, args.output, length, steps, 1 if args.pilot else 10, anchors[f"{length}_{steps}"]))
            del candidate
            gc.collect()
            torch.cuda.empty_cache()
        after = common.gpu_state()
        if not common.uncontended(producing["gpu_before"], after):
            raise RuntimeError("GPU contention changed")
        eligible = [name for name in projections.CANDIDATES if all(r["all_bitwise"] for r in rows if r["candidate"] == name)]
        promoted = [{"candidate": r["candidate"], "length": r["length"], "decode_steps": r["decode_steps"]}
                    for r in rows if r["candidate"] in eligible and r["speed_gate"]] if not args.pilot else []
        result = {"schema": "nal.llama-exact-speed.v1", "hypothesis_id": HYPOTHESIS_ID,
                  "status": "systems_lifecycle_only_not_quality_evidence" if args.pilot else "completed",
                  "producer_fingerprint": producing["fingerprint"], "workloads": rows, "setup": setup,
                  "tiny_lifecycle": tiny, "eligible_candidates": eligible, "promoted": promoted,
                  "anchor_artifact": str(anchor_path), "anchor_sha256": sha256(anchor_path),
                  "parameter_bytes": reference.storage_bytes(), "gpu_after": after, "uncontended": True,
                  "application_accuracy_tested": False, "weight_compression_used": False, "two_for_one_supported": False}
        write_json(args.output / "result.json", result)
    except Exception as error:
        write_json(args.output / "failure.json", {"status": "failed", "type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
