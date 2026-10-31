"""Pinned native Phi-4 request replay and live WORD_GAMES quality evaluation.

Run from the repository root with ``pixi run --locked -e word-games python -m``.
The controller owns only its temporary server and never unloads other models.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import requests

from .common import digest, read, sha, write
from .transport import MODEL, TransportFailure

GPU = "GPU-0370fdb0-6490-d871-15e5-c14a1fa671ff"
WEIGHTS = Path("/home/rabbit/models/Phi-4-GGUF/phi-4-Q4_K_M.gguf")
WEIGHTS_SHA = "009aba717c09d4a35890c7d35eb59d54e1dba884c7c526e7197d9c13ab5911d9"
BINARY = Path(".pixi/word-games-native-b11960/llama-b11960")
PLAN = Path("docs/07 - Status Reports/2026-10-31_word-games-phi4-40tps-plan.md")
PILOT = ["metaphor", "cases/normal_reserve/sender/turn_01", "cases/cancel/receiver"]
IMAGE = "sha256:24cbda07ad69c186ebaa9927292448d82e705d0d5d68c68e6c0f8eab420b64dd"


def output_tokens(response):
    # Native n_gen includes an EOS token; the previous Python completion usage
    # excludes it. Keep accepted, user-visible token accounting comparable.
    return response["tokens_predicted"] - int(response.get("stop_type") == "eos")


def aggregate(rows):
    seconds = sum(r["elapsed_seconds"] for r in rows)
    tokens = sum(output_tokens(r["response"]) for r in rows)
    decode_ms = sum(r["response"]["timings"]["predicted_ms"] for r in rows)
    # Server eval timing excludes the first token produced by prompt eval.
    decode_tokens = sum(r["response"]["timings"]["predicted_per_second"] *
                        r["response"]["timings"]["predicted_ms"] / 1000 for r in rows)
    return {"calls": len(rows), "request_seconds": seconds, "output_tokens": tokens,
            "output_tokens_per_second": tokens / seconds,
            "decode_only_tokens_per_second": decode_tokens / (decode_ms / 1000),
            "json_objects": sum(isinstance(r["parsed"], dict) for r in rows),
            "truncated": sum(r["response"].get("stop_type") == "limit" or r["response"].get("truncated", False) for r in rows),
            "drafted": sum(r["response"]["timings"].get("draft_n", 0) for r in rows),
            "draft_accepted": sum(r["response"]["timings"].get("draft_n_accepted", 0) for r in rows)}


class NativePhi4:
    def __init__(self, port, *, compact_json=False, greedy_only=False, no_grammar=False):
        self.url = f"http://127.0.0.1:{port}"
        self.http = requests.Session()
        self.rows = []
        self.compact_json = compact_json
        self.greedy_only, self.no_grammar = greedy_only, no_grammar

    def complete(self, prompt, path, *, max_tokens=1024):
        # Exact template embedded in the frozen target GGUF; no source text is
        # inserted beyond the current agent's authorized input.
        formatted = "<|im_start|>user<|im_sep|>" + prompt + "<|im_end|><|im_start|>assistant<|im_sep|>"
        body = {"prompt": formatted, "n_predict": max_tokens, "temperature": 0,
                "seed": 73, "repeat_penalty": 1.0, "cache_prompt": False,
                "json_schema": {"type": "object"}, "stream": False, "return_tokens": True}
        if self.compact_json:
            del body["json_schema"]
            body["grammar"] = Path(__file__).with_name("json_compact.gbnf").read_text()
        if self.greedy_only:
            body["samplers"] = ["temperature"]
        if self.no_grammar:
            if self.compact_json:
                raise ValueError("compact grammar and no grammar are incompatible")
            del body["json_schema"]
        write(path / "request.json", body)
        start = time.perf_counter()
        try:
            r = self.http.post(self.url + "/completion", json=body, timeout=240)
            elapsed = time.perf_counter() - start
            r.raise_for_status()
            value = r.json()
            text = value["content"]
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
            row = {"elapsed_seconds": elapsed, "response": value, "parsed": parsed}
            write(path / "response.json", row)
            self.rows.append(row)
            return text
        except Exception as exc:
            if not (path / "response.json").exists():
                write(path / "response.json", {"error": repr(exc), "elapsed_seconds": time.perf_counter() - start})
            raise TransportFailure(str(exc)) from exc


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    inputs = read(args.requests)
    if len(inputs) != 36 or sha(args.requests) != "f9830c9480f8a4694630a06102cfb33e9e16517faec22e4f9f59888137fd53cb":
        raise ValueError("frozen 36-request corpus required")
    if sha(WEIGHTS) != WEIGHTS_SHA:
        raise ValueError("target weights changed")
    name = f"word-games-native-{os.getpid()}"
    cmd = ["docker", "run", "--rm", "--name", name, "--gpus", f"device={GPU}",
           "--user", f"{os.getuid()}:{os.getgid()}",
           "-p", f"127.0.0.1:{args.port}:8080", "-v", f"{BINARY.resolve()}:/llama:ro",
           "-v", "/usr/local/cuda/lib64:/cuda:ro", "-v", f"{WEIGHTS.resolve()}:/model.gguf:ro",
           "-e", f"CUDA_VISIBLE_DEVICES={GPU}", IMAGE,
           "-m", "/model.gguf", "--host", "0.0.0.0", "--port", "8080",
           "--ctx-size", "8192", "--parallel", "1", "--fit", "off",
           "--n-gpu-layers", "99", "--flash-attn", "on", "--cache-type-k", args.kv, "--cache-type-v", args.kv,
           "--batch-size", str(args.batch), "--ubatch-size", str(args.batch), "--threads", str(args.threads),
           "--cache-ram", "0", "--no-cache-prompt", "--spec-type", args.spec]
    if args.spec != "none":
        cmd += [f"--spec-{args.spec}-size-n", str(args.ngram), f"--spec-{args.spec}-size-m", str(args.draft)]
    if args.backend_sampling:
        cmd += ["--backend-sampling"]
    sources = [Path(__file__), Path(__file__).with_name("common.py"), Path(__file__).with_name("Dockerfile.native"), PLAN]
    if args.compact_json:
        sources.append(Path(__file__).with_name("json_compact.gbnf"))
    if args.stage == "live":
        sources.append(Path(__file__).with_name("native_quality.py"))
    fixture_hashes = {}
    if args.paired_fixture_root:
        if args.stage != "live":
            raise ValueError("paired fixtures apply only to live quality")
        fixture_hashes = {str(p): sha(p) for p in sorted((args.paired_fixture_root / "quality/cases").glob("*/context.json"))}
        if len(fixture_hashes) != 12:
            raise ValueError("twelve paired fixture contexts required")
    config = {"hypothesis_id": "word-games-phi4-native-40tps-v1", "stage": args.stage,
              "command": cmd, "weights_sha256": WEIGHTS_SHA, "requests_sha256": sha(args.requests),
              "binary_hashes": {p.name: sha(p) for p in sorted(BINARY.glob("*")) if p.is_file() and not p.is_symlink()},
              "source_hashes": {str(p): sha(p) for p in sources}, "cross_request_cache": False,
              "spec": args.spec, "ngram": args.ngram, "draft": args.draft, "backend_sampling": args.backend_sampling,
              "compact_json": args.compact_json,
              "kv": args.kv, "batch": args.batch,
              "greedy_only": args.greedy_only, "no_grammar": args.no_grammar,
              "paired_fixture_hashes": fixture_hashes,
              "gpu_before": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.used", "--format=csv"], text=True)}
    write(out / "producer.json", {**config, "fingerprint": digest(config)})
    for p in sources:
        (out / p.name).write_bytes(p.read_bytes())
    client = NativePhi4(args.port, compact_json=args.compact_json,
                        greedy_only=args.greedy_only, no_grammar=args.no_grammar)
    process = None
    start = time.perf_counter()
    try:
        with (out / "server.log").open("x") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 240
            while True:
                if process.poll() is not None:
                    raise RuntimeError("native server exited during startup; see server.log")
                try:
                    if client.http.get(client.url + "/health", timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup")
                time.sleep(0.5)
            write(out / "runtime.json", {"load_seconds": time.perf_counter() - start,
                  "properties": client.http.get(client.url + "/props", timeout=10).json(),
                  "gpu_loaded": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.used", "--format=csv"], text=True)})
            client.complete('Return exactly {"ready":true} as JSON.', out / "warmup", max_tokens=24)
            client.rows.clear()
            if args.stage == "live":
                from .native_quality import run_quality
                identity = {"model": MODEL, "ready": True, "weights_sha256": WEIGHTS_SHA,
                            "native_fingerprint": digest(config), "backend": "llama.cpp b11960 native"}
                write(out / "identity.json", identity)
                quality = run_quality(out / "quality", out / "identity.json", client, args.paired_fixture_root)
                records = [read(p) for p in (out / "quality/cases").glob("*/result.json")]
                write(out / "quality_summary.json", {"metrics": quality["metrics"],
                      "joint_case_success": sum(all(r[k] for k in ("sender_operation_match", "boundary_outcome_match", "receiver_state_correct")) for r in records),
                      "cases": len(records), "semantic_review_completed": False})
            else:
                chosen = [next(r for r in inputs if r["id"] == name) for name in PILOT] if args.stage == "pilot" else inputs
                for i, request in enumerate(chosen):
                    client.complete(request["prompt"], out / "runs" / request["id"], max_tokens=request["max_tokens"])
                    print(json.dumps({"id": request["id"], "seconds": client.rows[-1]["elapsed_seconds"],
                                      "tokens": output_tokens(client.rows[-1]["response"])}), flush=True)
            if any(sha(p) != h for p, h in config["source_hashes"].items()):
                raise RuntimeError("producing source changed")
            if any(sha(p) != h for p, h in fixture_hashes.items()):
                raise RuntimeError("paired fixture changed")
            result = {"hypothesis_id": config["hypothesis_id"], "stage": args.stage, "confirmed": False,
                      "fingerprint": digest(config), "totals": aggregate(client.rows),
                      "artifacts": {str(p): sha(p) for p in sorted(out.rglob("response.json")) if p.parent.name != "warmup"}}
            write(out / "campaign_results.json", result)
            print(json.dumps(result["totals"]), flush=True)
    except Exception as exc:
        write(out / "interruption.json", {"error": repr(exc), "completed_calls": len(client.rows)})
        raise
    finally:
        if process:
            subprocess.run(["docker", "stop", "--time", "5", name], capture_output=True)
            process.wait(timeout=30)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--requests", type=Path, default=Path("data/experiments/word_games/20261031_speculative_inputs_v1/requests.json"))
    p.add_argument("--stage", choices=["pilot", "full", "live"], default="pilot")
    p.add_argument("--spec", choices=["none", "ngram-simple", "ngram-map-k", "ngram-map-k4v"], default="none")
    p.add_argument("--ngram", type=int, default=3)
    p.add_argument("--draft", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch", type=int, choices=[256, 512, 1024, 2048], default=512)
    p.add_argument("--kv", choices=["q8_0", "f16"], default="q8_0")
    p.add_argument("--backend-sampling", action="store_true")
    p.add_argument("--compact-json", action="store_true")
    p.add_argument("--greedy-only", action="store_true")
    p.add_argument("--no-grammar", action="store_true")
    p.add_argument("--paired-fixture-root", type=Path)
    p.add_argument("--port", type=int, default=18084)
    run(p.parse_args())
