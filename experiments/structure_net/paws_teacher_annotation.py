#!/usr/bin/env python3
"""Experiment 03: resumable Qwen annotation of eligible PAWS training groups."""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any

from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner

try:
    from experiments.structure_net.paws_dataset_contract import EXPECTED_HASHES, read_split, render_prompt
except ModuleNotFoundError:
    from paws_dataset_contract import EXPECTED_HASHES, read_split, render_prompt

SCHEMA_VERSION = "nal.paws-qwen-annotation.v1"
HYPOTHESIS_ID = "paws-qwen-teacher-annotation-v1"
PROMPT_VERSION = "paws-exact-label-v1"
_HTTP_LOCAL = threading.local()


def http_session():
    import requests
    if not hasattr(_HTTP_LOCAL, "session"):
        _HTTP_LOCAL.session = requests.Session()
    return _HTTP_LOCAL.session


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def key() -> str:
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("LITELLM_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("LITELLM_KEY missing")


def eligible_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_split(path)
    labels: dict[str, set[int]] = {}
    for row in rows:
        labels.setdefault(row["group_id"], set()).add(row["label"])
    seen: set[str] = set()
    result = []
    for row in rows:
        if len(labels[row["group_id"]]) != 1 or row["group_id"] in seen:
            continue
        seen.add(row["group_id"])
        result.append(row)
    return result


def audit_rows(path: Path, samples: int, seed: int = 23) -> list[dict[str, Any]]:
    if samples < 2 or samples % 2:
        raise ValueError("audit samples must be even and >= 2")
    rows = eligible_rows(path)
    chosen = []
    for label in (0, 1):
        pool = [row for row in rows if row["label"] == label]
        pool.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['group_id']}".encode()).hexdigest())
        chosen.extend(pool[: samples // 2])
    return sorted(chosen, key=lambda row: row["group_id"])


def request_fingerprint(model: str, prompt: str) -> str:
    payload = {"model": model, "prompt": prompt, "prompt_version": PROMPT_VERSION, "temperature": 0, "max_tokens": 16}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def annotate(row: dict[str, Any], *, base_url: str, model: str, ca_path: str, api_key: str) -> dict[str, Any]:
    import requests

    prompt = render_prompt(row["sentence1"], row["sentence2"])
    started = time.perf_counter()
    error = None
    for attempt in range(1, 5):
        try:
            response = http_session().post(
                base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 16},
                verify=ca_path,
                timeout=60,
            )
            response.raise_for_status()
            raw = response.json()["choices"][0]["message"]["content"].strip()
            teacher_label = {"DIFFERENT": 0, "PARAPHRASE": 1}.get(raw)
            return {
                "schema_version": SCHEMA_VERSION, "source_id": row["source_id"], "group_id": row["group_id"],
                "human_label": row["label"], "teacher_label": teacher_label, "teacher_correct": teacher_label == row["label"],
                "raw": raw, "rationale": None, "status": "success" if teacher_label is not None else "malformed",
                "latency_seconds": time.perf_counter() - started, "attempts": attempt,
                "request_fingerprint": request_fingerprint(model, prompt),
            }
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < 4:
                time.sleep(2 ** (attempt - 1))
    return {
        "schema_version": SCHEMA_VERSION, "source_id": row["source_id"], "group_id": row["group_id"],
        "human_label": row["label"], "teacher_label": None, "teacher_correct": False, "raw": None, "rationale": None,
        "status": "failed", "error": error, "latency_seconds": time.perf_counter() - started, "attempts": 4,
        "request_fingerprint": request_fingerprint(model, prompt),
    }


def worker(experiment: Experiment, _device_id: int) -> ExperimentResult:
    started = time.perf_counter()
    parameters = experiment.parameters
    train_path = Path(parameters["train_path"])
    output = Path(parameters["annotations"])
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = audit_rows(train_path, parameters["audit_samples"])
    expected = {row["group_id"] for row in rows}
    existing: dict[str, dict[str, Any]] = {}
    reuse_path = output if output.exists() else output.with_name("annotations_superseded_full_attempt.jsonl")
    if reuse_path.exists():
        for line in reuse_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["group_id"] in expected:
                existing[record["group_id"]] = record
    pending = [row for row in rows if existing.get(row["group_id"], {}).get("status") != "success"]
    api_key = key()
    batch_size = parameters["checkpoint_rows"]
    with ThreadPoolExecutor(max_workers=parameters["concurrency"]) as pool:
      for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        if batch:
            records = list(pool.map(lambda row: annotate(row, base_url=parameters["base_url"], model=parameters["model"], ca_path=parameters["ca_path"], api_key=api_key), batch))
            existing.update((record["group_id"], record) for record in records)
            temporary = output.with_suffix(".jsonl.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for group_id in sorted(existing):
                    handle.write(json.dumps(existing[group_id], sort_keys=True, allow_nan=False) + "\n")
            temporary.replace(output)
    records = list(existing.values())
    observed = {record["group_id"] for record in records}
    failures = sum(record["status"] != "success" for record in records)
    accuracy = sum(record["teacher_correct"] for record in records) / len(rows)
    complete = observed == expected and len(records) == len(rows) and failures == 0
    summary = {
        "schema_version": SCHEMA_VERSION, "hypothesis_id": HYPOTHESIS_ID, "complete": complete,
        "audit_groups": len(rows), "annotation_rows": len(records), "failures": failures, "teacher_accuracy": accuracy,
        "dataset_sha256": sha256(train_path), "expected_dataset_sha256": EXPECTED_HASHES["train"],
        "selection_sha256": hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest(),
        "prompt_version": PROMPT_VERSION, "model": parameters["model"], "annotations_sha256": sha256(output),
        "code_sha256": sha256(Path(__file__)), "runtime_seconds": time.perf_counter() - started,
    }
    summary_path = Path(parameters["summary"])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    error = None if complete else f"annotation gates failed: failures={failures}, rows={len(records)}/{len(rows)}"
    return ExperimentResult(experiment_id=experiment.id, hypothesis_id=HYPOTHESIS_ID, metrics={"teacher_accuracy": accuracy, "failure_rate": failures / len(rows)}, primary_metric=accuracy, model_architecture=[], model_parameters=0, training_time=time.perf_counter() - started, observations=[f"summary={summary_path}", f"annotations={output}"], error=error)


async def run(output: Path, concurrency: int, batch_size: int, audit_samples: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    parameters = {
        "train_path": "data/datasets/paws-wiki/labeled/train.csv", "annotations": str(output / "annotations.jsonl"),
        "summary": str(output / "summary.json"), "base_url": "https://central-dev.zt:4000/v1", "model": "qwen3-8b",
        "ca_path": "/home/rabbit/.config/lakefs/caddy-root.crt", "concurrency": concurrency, "checkpoint_rows": batch_size, "audit_samples": audit_samples,
    }
    experiment = Experiment(id="paws-qwen-teacher-annotation", hypothesis_id=HYPOTHESIS_ID, name="PAWS Qwen teacher annotation", parameters=parameters)
    runner = AsyncExperimentRunner(LabConfig(project_name="paws_qwen_teacher_annotation", results_dir=str(output / "nal"), device_ids=[-1], max_parallel_experiments=1, min_experiments_per_hypothesis=1, require_statistical_significance=False, enable_wandb=False), worker)
    result = (await runner.run_experiments([experiment]))[0]
    return {"complete": result.error is None, "metrics": result.metrics, "error": result.error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/experiments/paws_abc_routing/2026-08-16_experiment_03"))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--audit-samples", type=int, default=2048)
    args = parser.parse_args()
    result = asyncio.run(run(args.output, args.concurrency, args.batch_size, args.audit_samples))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__":
    main()
