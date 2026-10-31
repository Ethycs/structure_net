"""Isolated CUDA benchmark worker using the existing serving Pixi environment."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import subprocess
import time

import llama_cpp
from llama_cpp import Llama

from .common import digest, read, sha, write
from .speculative_draft import CrossTokenizerDraft

TARGET = "/model-weights/Phi-4-GGUF/phi-4-Q4_K_M.gguf"
DRAFT = "/model-weights/Phi-4-mini-instruct-GGUF/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
PILOT_IDS = ["metaphor", "cases/normal_reserve/sender/turn_01", "cases/cancel/receiver"]


class MeasuredLlama(Llama):
    measurement = None

    def eval(self, tokens):
        m = self.measurement
        if m is not None and not m["evaluations"] and self.draft_model is not None:
            self.draft_model.prompt_ids = list(tokens)
        start = time.perf_counter()
        super().eval(tokens)
        if m is not None:
            m["evaluations"].append({"tokens": len(tokens), "seconds": time.perf_counter() - start})

    def sample(self, *args, **kwargs):
        token = super().sample(*args, **kwargs)
        m = self.measurement
        if m is not None:
            if not m["target_tokens"]:
                m["first_token_seconds"] = time.perf_counter() - m["start"]
            idx = kwargs.get("idx", self.n_tokens - 1)
            if idx is not None and idx + 1 < self.n_tokens and token == self.input_ids[idx + 1]:
                m["accepted_draft_tokens"] += 1
            m["target_tokens"].append(int(token))
        return token


class WrongDraft(CrossTokenizerDraft):
    """Negative control: candidates deliberately unrelated to the response."""
    def __call__(self, input_ids):
        import numpy as np
        self.calls += 1
        self.proposed += self.length
        return np.full(self.length, self.target.tokenize(b" WRONG", add_bos=False)[0], dtype=np.intc)


def complete(target, draft, request, length, out, adapter_type=CrossTokenizerDraft):
    out.mkdir(parents=True, exist_ok=False)
    write(out / "request.json", request)
    start = time.perf_counter()
    target.reset()
    target._logits_all = bool(length)
    adapter = adapter_type(target, draft, length) if length else None
    target.draft_model = adapter
    if adapter:
        adapter.begin(request["prompt"])
    m = {"start": start, "evaluations": [], "target_tokens": [], "accepted_draft_tokens": 0}
    target.measurement = m
    response = target.create_chat_completion(messages=[{"role": "user", "content": request["prompt"]}],
        temperature=0.0, seed=73, max_tokens=request["max_tokens"], response_format={"type": "json_object"})
    elapsed = time.perf_counter() - m.pop("start")
    target.measurement = None
    result = {"request_id": request["id"], "draft_length": length, "elapsed_seconds": elapsed,
        "text": response["choices"][0]["message"]["content"], "usage": response["usage"],
        "finish_reason": response["choices"][0]["finish_reason"], **m,
        "draft_seconds": adapter.seconds if adapter else 0,
        "draft_calls": adapter.calls if adapter else 0,
        "proposed_target_tokens": adapter.proposed if adapter else 0,
        "generated_native_draft_tokens": adapter.native_tokens if adapter else 0}
    write(out / "result.json", result)
    print(json.dumps({k: result[k] for k in ("request_id", "draft_length", "elapsed_seconds", "accepted_draft_tokens", "proposed_target_tokens")}), flush=True)
    return result


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    inputs = read(args.requests)
    if len(inputs) != 36:
        raise ValueError("complete frozen request set required")
    source_paths = [Path(__file__), Path(__file__).with_name("speculative_draft.py"), Path(__file__).with_name("common.py"),
                    Path("docs/07 - Status Reports/2026-10-31_word-games-phi4-speculative-plan.md")]
    bindings = {str(p): sha(p) for p in source_paths}
    config = {"hypothesis_id": "word-games-phi4-mini-speculation-v1", "stage": args.stage,
        "draft_length": args.length, "requests_sha256": sha(args.requests), "source_hashes": bindings,
        "target": TARGET, "draft": DRAFT, "target_context": 8192, "draft_context": 8192,
        "target_gpu": "RTX 3060", "draft_gpu": "RTX 2060 SUPER", "quantization": "Q4_K_M", "kv": "Q8_0",
        "python": platform.python_version(), "llama_cpp_python": llama_cpp.__version__, "seed": 73,
        "cross_request_kv_reuse": False, "two_gpus_required": True}
    prepared = read(Path(args.requests).with_name("prepared.json"))
    config.update(target_sha256=prepared["target"]["weights_sha256"], draft_sha256=prepared["draft_sha256"],
                  draft_revision=prepared["draft_revision"], container_image="sha256:ea2c80aa8efcf2b1578db2ab869ea1978a243b8327d5e4df9e95cabaaf74cc4f")
    if sha(args.requests) != prepared["requests_sha256"]:
        raise ValueError("frozen requests changed")
    write(out / "producer.json", {**config, "fingerprint": digest(config)})
    for path in source_paths:
        (out / path.name).write_bytes(path.read_bytes())
    load_start = time.perf_counter()
    options = dict(n_gpu_layers=-1, split_mode=0, n_ctx=8192, type_k=8, type_v=8,
                   flash_attn=True, verbose=False, seed=73)
    # llama-cpp-python allocates only n_batch score rows unless logits_all is
    # true at construction. Speculation can address rows across the context.
    target = MeasuredLlama(TARGET, main_gpu=0, logits_all=True, **options)
    target._logits_all = False
    draft = Llama(DRAFT, main_gpu=1, **options)
    write(out / "runtime.json", {"load_seconds": time.perf_counter() - load_start,
        "target_vocab": target.n_vocab(), "draft_vocab": draft.n_vocab(),
        "target_chat_template": target.metadata.get("tokenizer.chat_template"),
        "draft_chat_template": draft.metadata.get("tokenizer.chat_template"),
        "gpu_memory": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.used,memory.free", "--format=csv"], text=True)})
    warmup_start = time.perf_counter()
    # Warm both models on an unrelated formatting request, excluded from timing.
    for model in (target, draft):
        model.create_chat_completion(messages=[{"role": "user", "content": 'Return {"ready":true} as JSON.'}],
            temperature=0.0, max_tokens=8, response_format={"type": "json_object"})
        model.reset()
    warmup = {"id": "warmup", "prompt": "Background: " + "A blue circle and a yellow square. " * 100 + '\nReturn exactly {"ready":true} as JSON.', "max_tokens": 24}
    for length in (2, 4, 8):
        complete(target, draft, warmup, length, out / "warmup" / f"draft{length}")
    write(out / "warmup.json", {"elapsed_seconds": time.perf_counter() - warmup_start, "included_in_primary_timing": False})
    if args.stage == "pilot":
        control = {"id": "wrong_draft_control", "prompt": 'Return exactly {"ready":true} as JSON.', "max_tokens": 24}
        baseline = complete(target, draft, control, 0, out / "controls/baseline")
        wrong = complete(target, draft, control, 4, out / "controls/wrong_draft", WrongDraft)
        if baseline["target_tokens"] != wrong["target_tokens"]:
            raise RuntimeError("wrong candidates escaped target verification")
        write(out / "verification_control.json", {"wrong_candidates_rejected": True,
              "same_target_tokens": True, "proposed": wrong["proposed_target_tokens"], "accepted": wrong["accepted_draft_tokens"]})
    selected = [next(r for r in inputs if r["id"] == name) for name in PILOT_IDS] if args.stage == "pilot" else inputs
    lengths = [0, 2, 4, 8] if args.stage == "pilot" else [0, args.length]
    results, pairs = [], []
    try:
        for i, request in enumerate(selected):
            order = lengths if i % 2 == 0 else list(reversed(lengths))
            current = {}
            for length in order:
                current[length] = complete(target, draft, request, length, out / "runs" / f"{i:02d}_draft{length}")
                results.append(current[length])
            for length in lengths[1:]:
                b, s = current[0], current[length]
                pairs.append({"request_id": request["id"], "draft_length": length,
                    "tokens_equal": b["target_tokens"] == s["target_tokens"], "text_equal": b["text"] == s["text"],
                    "baseline_seconds": b["elapsed_seconds"], "speculative_seconds": s["elapsed_seconds"]})
    except Exception as exc:
        write(out / "interruption.json", {"error": repr(exc), "completed_calls": len(results)})
        raise
    finally:
        target.close(); draft.close()
    if any(sha(path) != expected for path, expected in bindings.items()):
        raise ValueError("producing source changed")
    totals = {}
    for length in lengths:
        rows = [r for r in results if r["draft_length"] == length]
        seconds = sum(r["elapsed_seconds"] for r in rows)
        output = sum(r["usage"]["completion_tokens"] for r in rows)
        totals[str(length)] = {"calls": len(rows), "request_seconds": seconds, "output_tokens": output,
            "output_tokens_per_second": output / seconds,
            "draft_seconds": sum(r["draft_seconds"] for r in rows),
            "accepted": sum(r["accepted_draft_tokens"] for r in rows),
            "proposed": sum(r["proposed_target_tokens"] for r in rows),
            "target_evaluation_calls": sum(len(r["evaluations"]) for r in rows)}
    eligible = [k for k in lengths[1:] if all(p["tokens_equal"] and p["text_equal"] for p in pairs if p["draft_length"] == k)]
    chosen = min(eligible, key=lambda k: totals[str(k)]["request_seconds"]) if eligible else None
    speedup = totals["0"]["request_seconds"] / totals[str(chosen)]["request_seconds"] if chosen else None
    classification = "output_mismatch" if not chosen else "observed_faster" if speedup > 1 else "observed_slower"
    result = {"hypothesis_id": config["hypothesis_id"], "stage": args.stage,
        "status": "systems_lifecycle_only_not_quality_evidence" if args.stage == "pilot" else "completed_exploratory",
        "classification": classification, "confirmed": False, "selected_draft_length": chosen,
        "speedup": speedup, "totals": totals, "pairs": pairs,
        "fingerprint": digest(config), "requested_calls": len(selected) * len(lengths), "completed_calls": len(results),
        "artifacts": {str(p): sha(p) for p in (out / "runs").rglob("result.json")}}
    write(out / "campaign_results.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("pairs", "artifacts")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["pilot", "full"], required=True)
    parser.add_argument("--length", type=int, default=4, choices=[2, 4, 8])
    run(parser.parse_args())
