"""Frozen, independently retryable CUDA cells for whole-model tensor packing."""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import statistics
import subprocess
import time
import traceback

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from structure_net.components.layers.model_major_tensor import CONFIGURATIONS
from experiments.structure_net import llama_cotenancy_common as common
from experiments.structure_net.llama_cotenancy_data import prepare, score
from experiments.structure_net.llama_tensorized_model import TensorizedLlama, convert, generate
from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH, OUTPUT, GGUF_FILE, sha256, write_json

FIXTURES = common.ROOT / "inputs_v1/fixtures.json"
BINARY = common.ROOT / "gguf_assay"
CPP = Path("experiments/structure_net/llama_cotenancy_gguf.cpp")


def aggregate_quality(rows):
    fragments = sum(r["metrics"]["covered_fragments"] for r in rows)
    total = sum(r["metrics"]["total_fragments"] for r in rows)
    supported = sum(r["metrics"]["supported_lines"] for r in rows)
    lines = sum(r["metrics"]["output_lines"] for r in rows)
    return {"examples": len(rows), "covered_fragments": fragments, "total_fragments": total,
            "fragment_coverage": fragments/max(1, total), "supported_lines": supported, "output_lines": lines,
            "supported_line_rate": supported/max(1, lines),
            "complete_summaries": sum(r["metrics"]["complete_summary"] for r in rows),
            "reference_nll_mean": statistics.mean(r["reference_nll"] for r in rows) if rows else None}


def promotion(candidate, baseline):
    if not candidate or len({r["id"] for r in candidate}) != len(candidate) or {r["id"] for r in candidate} != {r["id"] for r in baseline} or len(candidate) != len(baseline):
        raise ValueError("quality gates require identical nonempty, unique example IDs")
    a, b = aggregate_quality(candidate), aggregate_quality(baseline)
    matched = {r["id"]: r for r in baseline}
    competence = b["fragment_coverage"] >= .90 and b["supported_line_rate"] >= .95
    examples = {r["id"]: r["metrics"]["covered_fragments"] >= matched[r["id"]]["metrics"]["covered_fragments"]-1 for r in candidate}
    passed = (competence and a["fragment_coverage"] >= b["fragment_coverage"]
              and a["supported_line_rate"] >= b["supported_line_rate"]
              and a["complete_summaries"] >= b["complete_summaries"] and all(examples.values()))
    return {"baseline_competent": competence, "per_example": examples, "quality_pass": passed}


@torch.inference_mode()
def quality(model, fixtures, split):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    rows = []
    for row in fixtures["examples"]:
        if row["split"] != split:
            continue
        ids = torch.tensor([row["prompt_ids"]], device="cuda")
        started = time.perf_counter()
        generated = generate(model, ids)[0].tolist()
        text = tokenizer.decode(generated, skip_special_tokens=True)
        reference = torch.tensor([row["reference_ids"]], device="cuda")
        full = torch.cat((ids, reference[:, :-1]), 1)
        logits, _ = model(full, all_logits=True)
        loss = F.cross_entropy(logits[:, ids.shape[1]-1:].flatten(0, 1), reference.flatten())
        record = {"id": row["id"], "family": row["family"], "seed": row["seed"], "generated_ids": generated,
                  "text": text, "reference_nll": float(loss), "metrics": score(text, row), "seconds": time.perf_counter()-started}
        rows.append(record)
        print(json.dumps({"quality": row["id"], "metrics": record["metrics"], "nll": float(loss)}), flush=True)
        del logits, full, reference, ids, loss
    return rows


@torch.inference_mode()
def control(output, fixtures):
    start = time.perf_counter()
    hf = LlamaForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, torch_dtype=torch.float16,
                                        attn_implementation="flash_attention_2", low_cpu_mem_usage=True).eval().cuda()
    ids = torch.tensor([fixtures["benchmark_ids"][0][:128]], device="cuda")
    expected = hf(ids, use_cache=True, logits_to_keep=1)
    target = expected.logits.cpu()
    next_ids = torch.tensor([[1475]], device="cuda")
    next_target = hf(next_ids, past_key_values=expected.past_key_values, use_cache=True, logits_to_keep=1).logits.cpu()
    del expected
    hf.cpu()
    torch.cuda.empty_cache()
    matrices, vectors = {}, {}
    for name, value in hf.state_dict().items():
        if name == "lm_head.weight" and hf.config.tie_word_embeddings:
            continue
        if value.ndim == 2:
            matrices[name] = {"kind": "dense", "shape": list(value.shape), "cores": [value.detach()]}
        else:
            vectors[name] = value.detach()
    payload = {"schema": "structure-net.tensorized-llama.v1", "config": hf.config.to_dict(),
               "setting": {"name": "dense_fp16", "kind": "dense"}, "matrices": matrices, "vectors": vectors}
    checkpoint = output / "model.pt"
    torch.save(payload, checkpoint)
    del payload, matrices, vectors, hf
    gc.collect()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model = TensorizedLlama(payload)
    actual, past = model(ids, use_cache=True)
    next_actual, _ = model(next_ids, past=past, use_cache=True)
    rows = {"prefill": common.parity(actual.cpu(), target), "cached_decode": common.parity(next_actual.cpu(), next_target)}
    # Same graph/weights/attention backend: the primary numerical gates also apply.
    result = {"status": "completed", "control_pass": all(r["pass"] for r in rows.values()), "checks": rows,
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "parameter_bytes": model.storage_bytes(),
              "seconds": time.perf_counter()-start, "layers": model.config.num_hidden_layers,
              "systems_lifecycle_only_not_quality_evidence": True}
    return result


def memory():
    return {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "device": common.gpu_state()}


@torch.inference_mode()
def benchmark(model, fixtures, samples):
    count = model.replicas
    separate = [model.slice(i) for i in range(count)]
    results = []
    for length, steps in ((128, 0), (512, 0), (128, 32)):
        ids = torch.tensor([r[:length] for r in fixtures["benchmark_ids"][:count]], device="cuda")
        # Reserve unused KV capacity too: every physical replica has a 1024-token
        # budget, matching the private llama.cpp contexts. Dynamic live K/V is
        # additionally allocated by the forward; no context reduction at high M.
        kv_per_token = count * model.config.num_hidden_layers * 2 * model.config.num_key_value_heads * model.config.head_dim * 2
        slack_bytes = (1024-length-max(0, steps-1))*kv_per_token
        context_slack = torch.empty(slack_bytes, dtype=torch.uint8, device="cuda")
        check = {}
        if not steps:
            actual, past = model(ids, use_cache=True)
            singles = [m(ids[i:i+1], use_cache=True) for i, m in enumerate(separate)]
            check["prefill_logits"] = common.parity(actual, torch.cat([r[0] for r in singles]))
            kv_errors = [float((past[layer][kv]-torch.cat([r[1][layer][kv] for r in singles])).abs().max())
                         for layer in range(model.config.num_hidden_layers) for kv in (0, 1)]
            token = ids[:, -1:]
            decoded, _ = model(token, past=past, use_cache=True)
            expected = torch.cat([m(token[i:i+1], past=singles[i][1], use_cache=True)[0] for i, m in enumerate(separate)])
            check["cached_decode_logits"] = common.parity(decoded, expected)
            check["kv_max_abs"] = max(kv_errors)
            check["kv_bytes"] = sum(t.numel()*t.element_size() for layer in past for t in layer)
            del actual, past, singles, token, decoded, expected, _
        if steps:
            arms = {"packed": lambda: generate(model, ids, steps, False),
                    "separate": lambda: [generate(m, ids[i:i+1], steps, False) for i, m in enumerate(separate)]}
        else:
            arms = {"packed": lambda: model(ids, use_cache=True),
                    "separate": lambda: [m(ids[i:i+1], use_cache=True) for i, m in enumerate(separate)]}
        measurements = common.paired(arms, samples=samples)
        measured_memory = {}
        for name, fn in arms.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            resident = torch.cuda.memory_allocated()
            value = fn()
            torch.cuda.synchronize()
            measured_memory[name] = {**memory(), "resident_allocated_before_bytes": resident,
                                    "peak_inference_increment_bytes": torch.cuda.max_memory_allocated()-resident}
            del value
        record = {"length": length, "decode_steps": steps, "timings": measurements, "memory": measured_memory, "parity": check,
                  "context_per_model": 1024, "reserved_unused_kv_bytes": slack_bytes, "kv_capacity_bytes": 1024*kv_per_token,
                  "packing_wall_speedup": measurements["separate"]["wall_median_ms"]/measurements["packed"]["wall_median_ms"]}
        results.append(record)
        print(json.dumps({"benchmark": [count, length, steps], "speedup": record["packing_wall_speedup"], "checks": check}), flush=True)
        del context_slack
    return results


def gguf(output, fixtures, count, split, pilot):
    extra = [CPP, BINARY, common.ROOT / "llama.cpp/build/CMakeCache.txt",
             *[common.ROOT / "llama.cpp/build/bin" / name for name in ("libllama.so.0.6.0", "libggml.so.0.26.0", "libggml-base.so.0.26.0", "libggml-cpu.so.0.26.0", "libggml-cuda.so.0.26.0")]]
    producer = common.producer(output, "gguf", [Path(__file__), FIXTURES, *extra],
                               {"replicas": count, "split": split, "pilot": pilot, "threads_per_model": 2})
    result_path = output / "raw.json"
    command = [str(BINARY.resolve()), str((OUTPUT/GGUF_FILE).resolve()), str(FIXTURES.resolve()), str(count), str(result_path.resolve()), "pilot" if pilot else split]
    with (output / "runtime.log").open("w") as log:
        run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "failed", "reason": "runtime exited without JSON; inspect runtime.log"}
    result["returncode"] = run.returncode
    if run.returncode:
        log = (output/"runtime.log").read_text(errors="replace")
        result["status"] = "capacity_failure" if "out of memory" in log.lower() or "failed to allocate" in log.lower() else "failed"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    by_id = {r["id"]: r for r in fixtures["examples"]}
    for row in result.get("quality", []):
        fixture = by_id[row["id"]]
        row["text"] = tokenizer.decode(row["generated_ids"], skip_special_tokens=True)
        row["metrics"] = score(row["text"], fixture)
        row["family"], row["seed"] = fixture["family"], fixture["seed"]
    result["quality_summary"] = aggregate_quality(result.get("quality", []))
    result["producer_fingerprint"] = producer["fingerprint"]
    result["gpu_after"] = common.gpu_state()
    result["uncontended"] = common.uncontended(producer["gpu_before"], result["gpu_after"])
    result["systems_lifecycle_only_not_quality_evidence"] = pilot
    write_json(output/"result.json", result)
    print(json.dumps({"completed": str(output), "status": result["status"], "quality": result["quality_summary"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fixtures", "control", "convert", "evaluate", "benchmark", "gguf"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--setting", choices=[s.name for s in CONFIGURATIONS])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--replicas", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.stage == "fixtures":
        if args.output.exists():
            raise ValueError("immutable fixtures already exist")
        prepare(args.output)
        print(json.dumps({"fixtures": str(args.output), "sha256": sha256(args.output)}))
        return
    common.setup()
    fixtures = json.loads(FIXTURES.read_text())
    if args.stage == "gguf":
        gguf(args.output, fixtures, args.replicas, args.split, args.pilot)
        return
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.checkpoint:
        config["checkpoint_sha256"] = sha256(args.checkpoint)
    producer = common.producer(args.output, args.stage, [Path(__file__), FIXTURES], config)
    try:
        if args.stage == "control":
            result = control(args.output, fixtures)
        elif args.stage == "convert":
            setting = next(s for s in CONFIGURATIONS if s.name == args.setting)
            result = convert(MODEL_PATH, LlamaConfig.from_pretrained(MODEL_PATH), setting, args.output/"checkpoint")
        else:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
            started = time.perf_counter()
            model = TensorizedLlama(payload, args.replicas)
            torch.cuda.synchronize()
            result = {"status": "completed", "replicas": args.replicas, "parameter_bytes": model.storage_bytes(),
                      "load_seconds": time.perf_counter()-started, "resident_memory": memory(), "setting": payload["setting"]}
            if args.stage == "evaluate":
                if args.replicas != 1:
                    raise ValueError("quality is per model")
                result["quality"] = quality(model, fixtures, args.split)
                result["quality_summary"] = aggregate_quality(result["quality"])
            else:
                result["benchmarks"] = benchmark(model, fixtures, 1 if args.pilot else 5)
                result["systems_lifecycle_only_not_quality_evidence"] = args.pilot
        result["producer_fingerprint"] = producer["fingerprint"]
    except torch.cuda.OutOfMemoryError as error:
        result = {"status": "capacity_failure", "stage": args.stage, "error": str(error), "traceback": traceback.format_exc()}
    except Exception as error:
        result = {"status": "failed", "stage": args.stage, "error": str(error), "traceback": traceback.format_exc()}
    result["gpu_after"] = common.gpu_state()
    result["uncontended"] = common.uncontended(producer["gpu_before"], result["gpu_after"])
    result["producer_fingerprint"] = producer["fingerprint"]
    write_json(args.output/"result.json", result)
    print(json.dumps({"completed": str(args.output), "status": result["status"], "control_pass": result.get("control_pass"),
                      "quality": result.get("quality_summary"), "error": result.get("error")}), flush=True)
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
