"""Audit the packed WORD_GAMES pathway from immutable responses and native logs.

This is a retrospective systems audit, not a new model or speed experiment.
Run from the repository root using Pixi. Outputs are append-only.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics


STAMP = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")
TOKEN = re.compile(r"slot process_toke:.*task\s+(\d+).*n_gen = (\d+)")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def timestamp_us(line):
    match = STAMP.match(line)
    if not match:
        return None
    minutes, seconds, millis, micros = map(int, match.groups())
    return ((minutes * 60 + seconds) * 1000 + millis) * 1000 + micros


def token_intervals(log):
    """Exclude each request's first output and the first (warmup) task.

    Decode-region time includes scheduling, CPU experts, GPU work, transfers,
    synchronization and the yielding worker join. The post-decode interval
    includes sampling and bookkeeping; it is not a pure sampler measurement.
    """
    previous, token_counts, records = {}, Counter(), []
    decode_start = decode_done = None
    for line in log.splitlines():
        timestamp = timestamp_us(line)
        if timestamp is None:
            continue
        if "srv        decode:" in line:
            decode_start = timestamp
        if "done yielding to queue" in line:
            decode_done = timestamp
        match = TOKEN.search(line)
        if not match:
            continue
        task, generated = map(int, match.groups())
        token_counts[task] += 1
        if (task in previous and decode_start is not None and decode_done is not None
                and previous[task] < decode_start < decode_done <= timestamp):
            records.append({
                "task": task, "generated": generated,
                "cycle_us": timestamp - previous[task],
                "before_decode_us": decode_start - previous[task],
                "decode_region_us": decode_done - decode_start,
                "after_decode_us": timestamp - decode_done,
            })
        previous[task] = timestamp
    if not token_counts:
        raise ValueError("no token timestamps")
    warmup_task = next(iter(token_counts))
    return [r for r in records if r["task"] != warmup_task], token_counts, warmup_task


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty interval series")
    return {"count": len(values), "mean_us": statistics.mean(values),
            "median_us": statistics.median(values),
            "p95_us": ordered[int(.95 * (len(ordered) - 1))],
            "sum_seconds": sum(values) / 1e6}


def audit(run_root, output):
    output.mkdir(parents=True, exist_ok=False)
    result = read(run_root / "result.json")
    if not result["completed"] or result["phase"] != "full":
        raise ValueError("completed full reference required")
    response_paths = sorted((run_root / "runs").rglob("response.json"))
    if len(response_paths) != 36:
        raise ValueError("expected 36 frozen requests")
    for path in response_paths:
        if sha(path) != result["response_hashes"][str(path)]:
            raise ValueError(f"response hash mismatch: {path}")
    parent = read(run_root.parent / "producer.json")
    binary_root = Path(".pixi/word-games-native-b11960/llama-b11960")
    binary_names = ["libllama.so.0.8.0", "libllama-server-impl.so", "libllama-common.so.0.8.0",
                    "libggml-base.so.0.28.0", "libggml-cuda.so", "libggml-cpu-ivybridge.so"]
    for name in binary_names:
        if sha(binary_root / name) != parent["binary_hashes"][name]:
            raise ValueError(f"native binary changed: {name}")
    rows = [read(path) for path in response_paths]
    if any(r["response"]["generation_settings"]["backend_sampling"] for r in rows):
        raise ValueError("this audit assumes the recorded CPU sampling path")
    wall = sum(r["elapsed_seconds"] for r in rows)
    prefill = sum(r["response"]["timings"]["prompt_ms"] for r in rows) / 1000
    decode = sum(r["response"]["timings"]["predicted_ms"] for r in rows) / 1000
    log = (run_root / "server.log").read_text()
    intervals, counts, warmup = token_intervals(log)
    if len(counts) != 37:
        raise ValueError("expected one warmup and 36 request tasks")
    expected_intervals = sum(n - 1 for task, n in counts.items() if task != warmup)
    if len(intervals) != expected_intervals:
        raise ValueError("incomplete token interval coverage")
    distributions = {key: distribution([r[key] for r in intervals]) for key in
                     ["cycle_us", "before_decode_us", "decode_region_us", "after_decode_us"]}
    reconciliation = distributions["cycle_us"]["sum_seconds"] - decode
    if abs(reconciliation) > .01:
        raise ValueError("log intervals do not reconcile with reported decode time")
    prefixes = Counter()
    for line in log.splitlines():
        match = re.match(r"\S+ [DIWE] ([^:]+):", line)
        if match:
            prefixes[match.group(1)] += 1
    vocab = int(re.search(r"n_vocab\s+= (\d+)", log).group(1))
    reserve = re.search(r"graph \(pp bs=(\d+), tg bs=(\d+)\): nodes = (\d+) / (\d+), splits = (\d+) / (\d+)", log)
    if reserve is None:
        raise ValueError("missing graph reserve record")
    audited_sources = [
        Path(__file__), Path("experiments/word_games/native_speed.py"),
        Path("experiments/word_games/qwen38_moe_runtime.py"),
        Path("experiments/word_games/three_gpu_experts.py"),
        Path("experiments/neural_architecture_lab/run_word_games_packed_experts.py"),
    ]
    native = Path(".pixi/word-games-hybrid/source")
    audited_sources += [native / p for p in [
        "tools/server/server-context.cpp", "tools/server/server-queue.cpp", "tools/server/server-http.cpp",
        "common/sampling.cpp", "common/log.cpp", "src/llama-context.cpp", "src/llama-graph.cpp",
        "src/models/qwen35moe.cpp", "ggml/src/ggml-backend.cpp", "ggml/src/ggml-cuda/ggml-cuda.cu",
    ]]
    evidence_paths = [run_root / n for n in ["result.json", "producer.json", "runtime.json", "placement.json", "server.log"]]
    summary = {
        "schema_version": 1, "classification": "retrospective_systems_path_audit",
        "new_model_execution": False, "confirmed_speedup": False, "reference_run": str(run_root),
        "source_hashes": {str(p): sha(p) for p in audited_sources},
        "evidence_hashes": {str(p): sha(p) for p in evidence_paths + response_paths},
        "native_binary_hashes_verified": {name: parent["binary_hashes"][name] for name in binary_names},
        "request_timing": {"calls": len(rows), "completion_wall_seconds": wall,
            "server_prefill_seconds": prefill, "server_decode_seconds": decode,
            "unattributed_residual_seconds": wall - prefill - decode,
            "unattributed_residual_percent": 100 * (wall - prefill - decode) / wall,
            "excluded_from_completion_timer": ["apply-template HTTP request", "request artifact write",
                "response JSON parse and artifact write", "startup and shutdown"],
            "residual_is_not_an_isolated_http_measurement": True},
        "token_timing": {"warmup_task_excluded": warmup, "warmup_outputs": counts[warmup],
            "request_first_outputs_excluded": len(counts) - 1,
            "intervals": len(intervals), "distributions": distributions,
            "reconciliation_error_seconds": reconciliation,
            "post_decode_includes": ["sampling", "redundant context synchronization calls",
                                     "grammar acceptance", "detokenization", "bookkeeping"],
            "decode_includes": ["CPU scheduling", "GPU kernels", "CPU expert computation",
                                "tensor copies", "backend synchronization", "queue worker join"]},
        "logging": {"bytes": len(log.encode()), "lines": len(log.splitlines()),
            "verbosity": 5, "includes_startup_and_warmup": True,
            "prefix_counts": dict(prefixes.most_common()),
            "next_response_posts": log.count("posting NEXT_RESPONSE"),
            "declined_tasks": log.count("task declined, id =")},
        "sampling": {"backend_sampling": False, "vocabulary_size": vocab,
            "raw_fp32_logit_bytes_per_output": vocab * 4,
            "source_derived_minimum_context_sync_calls_per_standard_output": 7,
            "sync_call_count_is_not_a_measured_cuda_api_count": True},
        "graph_reserve": dict(zip(["prefill_batch", "decode_batch", "prefill_nodes", "decode_nodes",
                                    "prefill_splits", "decode_splits"], map(int, reserve.groups()))),
        "placement": read(run_root / "placement.json"),
        "limits": ["Log timestamps perturb timing and aggregate server work; no kernel or syscall trace.",
                   "Static branches and synchronization counts are not measured hot-path attribution.",
                   "Native source tree includes inactive hybrid hooks; verified runtime is original b11960.",
                   "This maps the latest benchmark configuration, not a continuously deployed bridge."]}
    write(output / "audit.json", summary)
    with (output / "token_intervals.jsonl").open("x") as stream:
        for row in intervals:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "request_timing": summary["request_timing"],
                      "token_timing": summary["token_timing"], "audit_sha256": sha(output / "audit.json")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("data/experiments/word_games/20261031_packed_experts_v1/full_packed38"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run_root, args.output)
