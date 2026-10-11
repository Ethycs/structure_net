"""Complete-model exact execution benchmark against the unchanged eager model."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from experiments.structure_net import llama_cotenancy_common as common
from experiments.structure_net import llama_cuda_graph as graph_api
from experiments.structure_net.llama_tensorized_model import TensorizedLlama, generate
from experiments.structure_net.prepare_llama_cotenancy import sha256, write_json

HYPOTHESIS_ID = "llama32-3b-lossless-deployment-efficiency-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-10_llama32-lossless-deployment-plan.md")
ORIGINAL = common.ROOT / "20261010_control_v1/model.pt"
ORIGINAL_SHA256 = "f01cc0653ec6c9bd7026b8676864612b91b6c79d2dc4df42599cbe4cf99eb9a9"
FIXTURES = common.ROOT / "inputs_v2/fixtures.json"
AMENDMENT = Path("docs/07 - Status Reports/2026-10-10_llama32-deployment-input-amendment.md")


def benchmark_pair(rows, length):
    if len(rows) < 2 or min(map(len, rows[:2])) < length:
        raise ValueError("benchmark sequences are too short")
    pair = [rows[0][:length], rows[1][-length:]]
    if pair[0] == pair[1]:
        raise ValueError("correctness inputs must differ at the measured length")
    return pair


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.cpu()
    if value is None:
        return None
    return type(value)(cpu_tree(v) for v in value)


def paired(arms, samples):
    for fn in arms.values():
        value = fn()
        del value
    names = list(arms)
    records, orders = {name: [] for name in names}, []
    for index in range(samples):
        order = names.copy() if index % 2 == 0 else list(reversed(names))
        orders.append(order)
        for name in order:
            timing, value = common.timed(arms[name])
            records[name].append(timing)
            del value
    return {name: {"samples": rows, "wall_median_ms": statistics.median(v["wall_ms"] for v in rows),
                   "cuda_median_ms": statistics.median(v["cuda_ms"] for v in rows)} for name, rows in records.items()}, orders


def memory():
    return {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def tiny_pilot():
    torch.manual_seed(779)
    config = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=16, eos_token_id=2)
    hf = LlamaForCausalLM(config).eval().half()
    model = TensorizedLlama.from_hf(hf, device="cuda", attention="flash")
    ids = torch.arange(16, device="cuda")[None, :]
    checks = []
    for kind in ("prefill", "generation", "cached_step"):
        past = model(ids, use_cache=True)[1] if kind == "cached_step" else None
        example = ids[:, -1:] if past is not None else ids
        fn = (lambda x: generate(model, x, steps=4, stop_eos=False)) if kind == "generation" else (lambda x: model(x, past=past, use_cache=True))
        captured = graph_api.CapturedCall(fn, example)
        first = captured(example)
        saved = graph_api.clone_tree(first)
        second_input = (example + 3) % config.vocab_size
        changed = captured(second_input)
        valid = graph_api.bitwise_tree(first, fn(example)) and graph_api.bitwise_tree(changed, fn(second_input)) and graph_api.bitwise_tree(first, saved)
        checks.append({"kind": kind, "bitwise_equal_and_owned": valid})
        if not valid:
            raise RuntimeError("tiny graph bitwise or output ownership check failed")
        del captured, first, saved, changed, past
    del model, hf
    gc.collect()
    torch.cuda.empty_cache()
    return checks


def measure(model, rows, output, length, steps, samples):
    pair = benchmark_pair(rows, length)
    ids = torch.tensor([pair[0]], device="cuda")
    fn = (lambda x: generate(model, x, steps=steps, stop_eos=False)) if steps else (lambda x: model(x, use_cache=True))
    for _ in range(2):
        value = fn(ids)
        del value
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    _, value = common.timed(lambda: fn(ids))
    eager_memory = memory()
    del value
    torch.cuda.reset_peak_memory_stats()
    captured = graph_api.CapturedCall(fn, ids)
    capture_memory = memory()
    controls = []
    for sequence in pair:
        changed_ids = torch.tensor([sequence], device="cuda")
        expected = fn(changed_ids)
        actual = captured(changed_ids)
        exact = graph_api.bitwise_tree(actual, expected)
        controls.append({"ids": changed_ids.cpu(), "expected": cpu_tree(expected), "actual": cpu_tree(actual), "bitwise_equal": exact})
        del expected, actual, changed_ids
    control_path = output / f"controls_{length}_{steps}.pt"
    torch.save(controls, control_path)
    if not all(c["bitwise_equal"] for c in controls):
        raise RuntimeError(f"full graph bit mismatch: {length}/{steps}")
    torch.cuda.reset_peak_memory_stats()
    _, value = common.timed(lambda: captured(ids))
    graph_memory = memory()
    del value
    timings, orders = paired({"eager": lambda: fn(ids), "cuda_graph": lambda: captured(ids)}, samples)
    speedup = timings["eager"]["wall_median_ms"] / timings["cuda_graph"]["wall_median_ms"]
    result = {"length": length, "decode_steps": steps, "samples": samples, "timings": timings, "orders": orders,
              "speedup": speedup, "all_bitwise": True, "control_artifact": str(control_path), "control_sha256": sha256(control_path),
              "eager_memory": eager_memory, "capture_memory": capture_memory, "graph_memory": graph_memory,
              "capture_and_warmup_seconds": captured.capture_and_warmup_seconds,
              "faster_single_model_gate": speedup >= 1.10, "parameter_bytes": model.storage_bytes()}
    write_json(output / f"workload_{length}_{steps}.json", result)
    print(json.dumps({"workload": [length, steps], "speedup": speedup, "all_bitwise": True,
                      "timings_ms": {name: r["wall_median_ms"] for name, r in timings.items()},
                      "peak_bytes": {"eager": eager_memory["peak_allocated_bytes"], "graph": graph_memory["peak_allocated_bytes"]}}), flush=True)
    del controls, captured, ids, fn
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    common.setup()
    if sha256(ORIGINAL) != ORIGINAL_SHA256:
        raise ValueError("original checkpoint hash changed")
    common.HYPOTHESIS_ID = HYPOTHESIS_ID
    producing = common.producer(args.output, "exact_deployment_pilot" if args.pilot else "exact_deployment_benchmark",
        [Path(__file__), Path(graph_api.__file__), PLAN, AMENDMENT, FIXTURES, Path("experiments/structure_net/campaign_llama_cotenancy.py")],
        {"checkpoint": str(ORIGINAL), "checkpoint_sha256": ORIGINAL_SHA256, "fixture_sha256": sha256(FIXTURES),
         "pilot": args.pilot, "samples": 1 if args.pilot else 10, "physical_models": 1,
         "intervention": "CUDA graph replay of unchanged dense arithmetic, with owned output clones"})
    try:
        lifecycle = tiny_pilot() if args.pilot else None
        if lifecycle:
            write_json(args.output / "tiny_lifecycle.json", {"status": "systems_lifecycle_only_not_quality_evidence", "checks": lifecycle})
        payload = torch.load(ORIGINAL, map_location="cpu", weights_only=False, mmap=True)
        model = TensorizedLlama(payload)
        fixtures = json.loads(FIXTURES.read_text())
        sequences = fixtures["benchmark_ids"]
        if len(sequences) < 2 or sequences[0] == sequences[1] or min(map(len, sequences[:2])) < 512:
            raise ValueError("two unequal fixed benchmark sequences required")
        workloads = ((512, 0), (128, 32)) if args.pilot else ((128, 0), (512, 0), (128, 32))
        results = [measure(model, sequences, args.output, length, steps, 1 if args.pilot else 10) for length, steps in workloads]
        after = common.gpu_state()
        if not common.uncontended(producing["gpu_before"], after):
            raise RuntimeError("GPU contention changed during run")
        result = {"schema": "nal.llama-exact-deployment.v1", "hypothesis_id": HYPOTHESIS_ID,
            "status": "systems_lifecycle_only_not_quality_evidence" if args.pilot else "completed",
            "producer_fingerprint": producing["fingerprint"], "workloads": results, "tiny_lifecycle": lifecycle,
            "all_bitwise": all(r["all_bitwise"] for r in results), "parameter_bytes": model.storage_bytes(),
            "faster_single_model_supported": not args.pilot and any(r["faster_single_model_gate"] for r in results),
            "two_for_one_supported": False, "two_for_one_reason": "not measured: tested storage representations exceed the one-model budget for two independent copies",
            "inference_arithmetic_changed": False, "weight_compression_used": False,
            "gpu_after": after, "uncontended": True, "application_accuracy_tested": False}
        write_json(args.output / "result.json", result)
    except Exception as error:
        write_json(args.output / "failure.json", {"status": "failed", "error": str(error), "type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
