#!/usr/bin/env python3
"""Experiment 01: freeze the PAWS-Wiki dataset and prompt contract."""

from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from typing import Any, Iterable, Mapping

import numpy as np
from tokenizers import Tokenizer

from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner


SCHEMA_VERSION = "nal.paws-dataset-contract.v1"
HYPOTHESIS_ID = "paws-abc-routing-dataset-contract-v1"
SPLITS = ("train", "dev", "test")
EXPECTED_ROWS = {"train": 49_401, "dev": 8_000, "test": 8_000}
EXPECTED_HASHES = {
    "train": "f8ac90c04483a5b4b2c3583aad5e355122702f6066dcdf18787051f2ac2e7c98",
    "dev": "069c5c604db421f5d3513e932e9b1025456f1ffa7321ddb8336189bbc627bd67",
    "test": "51fc5f45bf33e7991a0c09b38bb4ff1ee482afd170128ae5ae5d8c4ba1d250cf",
}
PROMPT_TEMPLATE = (
    "Sentence A: {sentence1}\n"
    "Sentence B: {sentence2}\n\n"
    "Do these sentences have the same meaning?\n"
    "Answer only PARAPHRASE or DIFFERENT."
)
OUTPUT_LABELS = {"DIFFERENT": 0, "PARAPHRASE": 1}


@dataclass(frozen=True)
class Config:
    dataset_root: str = "data/datasets/paws-wiki/labeled"
    tiny_tokenizer: str = "data/corpora/babylm_10M_bpe16k.tokenizer.json"
    smol_tokenizer: str = "/data/models/SmolLM2-360M-Instruct/tokenizer.json"
    tiny_context: int = 256
    smol_context: int = 8_192
    qwen_context: int = 8_192
    validate_qwen_endpoint: bool = False
    qwen_base_url: str = "https://central-dev.zt:4000/v1"
    qwen_model: str = "qwen3-8b"
    qwen_ca_path: str = "/home/rabbit/.config/lakefs/caddy-root.crt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_sentence(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def pair_group_id(sentence1: str, sentence2: str) -> str:
    pair = sorted((normalize_sentence(sentence1), normalize_sentence(sentence2)))
    return hashlib.sha256(json.dumps(pair, ensure_ascii=False).encode()).hexdigest()


def render_prompt(sentence1: str, sentence2: str) -> str:
    return PROMPT_TEMPLATE.format(sentence1=sentence1, sentence2=sentence2)


def parse_output(value: str) -> int | None:
    return OUTPUT_LABELS.get(value.strip())


def read_split(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["id", "sentence1", "sentence2", "label"]:
            raise ValueError(f"unexpected fields in {path}: {reader.fieldnames}")
        records = []
        for row in reader:
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"invalid label {label} in {path}")
            records.append(
                {
                    "source_id": int(row["id"]),
                    "sentence1": row["sentence1"],
                    "sentence2": row["sentence2"],
                    "label": label,
                    "group_id": pair_group_id(row["sentence1"], row["sentence2"]),
                }
            )
    return records


def _token_lengths(tokenizer: Tokenizer, prompts: Iterable[str]) -> np.ndarray:
    lengths: list[int] = []
    batch: list[str] = []
    for prompt in prompts:
        batch.append(prompt)
        if len(batch) == 512:
            lengths.extend(len(value.ids) for value in tokenizer.encode_batch(batch))
            batch.clear()
    if batch:
        lengths.extend(len(value.ids) for value in tokenizer.encode_batch(batch))
    return np.asarray(lengths, dtype=np.int64)


def _length_record(lengths: np.ndarray, context: int) -> dict[str, Any]:
    return {
        "count": int(len(lengths)),
        "minimum": int(lengths.min()),
        "median": float(np.median(lengths)),
        "p95": float(np.quantile(lengths, 0.95)),
        "p99": float(np.quantile(lengths, 0.99)),
        "maximum": int(lengths.max()),
        "context_limit": context,
        "over_context_count": int((lengths > context).sum()),
    }


def _manifest_digest(records: list[dict[str, Any]]) -> str:
    payload = [
        (record["source_id"], record["group_id"], record["label"])
        for record in records
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def write_manifests(config: Config, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    for split in SPLITS:
        values = read_split(Path(config.dataset_root) / f"{split}.csv")
        labels_by_group: dict[str, set[int]] = {}
        for value in values:
            labels_by_group.setdefault(value["group_id"], set()).add(value["label"])
        path = destination / f"{split}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for value in values:
                record = {
                    "example_id": hashlib.sha256(
                        f"{split}:{value['source_id']}:{value['group_id']}".encode()
                    ).hexdigest(),
                    "source_id": value["source_id"],
                    "group_id": value["group_id"],
                    "label": value["label"],
                    "eligible": len(labels_by_group[value["group_id"]]) == 1,
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        temporary.replace(path)
        result[split] = {"path": str(path), "sha256": _sha256(path), "rows": len(values)}
    return result


def build_contract(config: Config = Config()) -> dict[str, Any]:
    root = Path(config.dataset_root)
    records = {split: read_split(root / f"{split}.csv") for split in SPLITS}
    split_results: dict[str, Any] = {}
    group_sets: dict[str, set[str]] = {}
    sentence_sets: dict[str, set[str]] = {}
    all_prompts: list[str] = []
    for split, values in records.items():
        groups: dict[str, list[dict[str, Any]]] = {}
        for value in values:
            groups.setdefault(value["group_id"], []).append(value)
        conflicts = sum(
            len({item["label"] for item in group}) > 1 for group in groups.values()
        )
        conflict_rows = sum(
            len(group)
            for group in groups.values()
            if len({item["label"] for item in group}) > 1
        )
        group_sets[split] = set(groups)
        sentence_sets[split] = {
            normalize_sentence(value[field])
            for value in values
            for field in ("sentence1", "sentence2")
        }
        prompts = [render_prompt(v["sentence1"], v["sentence2"]) for v in values]
        all_prompts.extend(prompts)
        split_results[split] = {
            "path": str(root / f"{split}.csv"),
            "sha256": _sha256(root / f"{split}.csv"),
            "rows": len(values),
            "label_counts": {
                str(label): sum(v["label"] == label for v in values)
                for label in (0, 1)
            },
            "unique_groups": len(groups),
            "duplicate_group_rows": len(values) - len(groups),
            "conflicting_label_groups": conflicts,
            "quarantined_conflicting_rows": conflict_rows,
            "eligible_rows": len(values) - conflict_rows,
            "manifest_sha256": _manifest_digest(values),
        }
    cross_split: dict[str, Any] = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            cross_split[f"{left}:{right}"] = {
                "group_overlap": len(group_sets[left] & group_sets[right]),
                "sentence_overlap": len(sentence_sets[left] & sentence_sets[right]),
            }

    tiny = Tokenizer.from_file(config.tiny_tokenizer)
    smol = Tokenizer.from_file(config.smol_tokenizer)
    tiny_lengths = _token_lengths(tiny, all_prompts)
    smol_lengths = _token_lengths(smol, all_prompts)
    tokenizer_results = {
        "tinyllm": {
            "path": config.tiny_tokenizer,
            "sha256": _sha256(Path(config.tiny_tokenizer)),
            "vocabulary_size": tiny.get_vocab_size(),
            **_length_record(tiny_lengths, config.tiny_context),
        },
        "smollm": {
            "path": config.smol_tokenizer,
            "sha256": _sha256(Path(config.smol_tokenizer)),
            "vocabulary_size": smol.get_vocab_size(),
            **_length_record(smol_lengths, config.smol_context),
        },
        "qwen": {
            "interface": "LiteLLM qwen3-8b; tokenizer is server-side",
            "context_limit": config.qwen_context,
            "token_counts_available": False,
            "validation": "longest-prompt endpoint acceptance required before annotation",
        },
    }
    if config.validate_qwen_endpoint:
        import requests

        key = None
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("LITELLM_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
        if not key:
            raise RuntimeError("LITELLM_KEY is not configured")
        longest = all_prompts[int(np.argmax(tiny_lengths))]
        response = requests.post(
            f"{config.qwen_base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": config.qwen_model,
                "messages": [{"role": "user", "content": longest}],
                "temperature": 0,
                "max_tokens": 8,
            },
            timeout=60,
            verify=config.qwen_ca_path,
        )
        response.raise_for_status()
        body = response.json()
        tokenizer_results["qwen"].update(
            {
                "longest_tinyllm_prompt_accepted": True,
                "response_model": body.get("model"),
                "finish_reason": body["choices"][0].get("finish_reason"),
            }
        )
    gates = {
        "expected_file_hashes": all(
            split_results[s]["sha256"] == EXPECTED_HASHES[s] for s in SPLITS
        ),
        "expected_row_counts": all(
            split_results[s]["rows"] == EXPECTED_ROWS[s] for s in SPLITS
        ),
        "no_missing_or_invalid_labels": all(
            sum(split_results[s]["label_counts"].values()) == EXPECTED_ROWS[s]
            for s in SPLITS
        ),
        "no_cross_split_groups": all(v["group_overlap"] == 0 for v in cross_split.values()),
        "no_cross_split_sentences": all(v["sentence_overlap"] == 0 for v in cross_split.values()),
        "conflicting_groups_identified_for_quarantine": all(
            split_results[s]["eligible_rows"]
            + split_results[s]["quarantined_conflicting_rows"]
            == split_results[s]["rows"]
            for s in SPLITS
        ),
        "tinyllm_context_fit": tokenizer_results["tinyllm"]["over_context_count"] == 0,
        "smollm_context_fit": tokenizer_results["smollm"]["over_context_count"] == 0,
        "parser_rejects_non_contract_output": parse_output("paraphrase") is None,
    }
    if config.validate_qwen_endpoint:
        gates["qwen_longest_prompt_accepted"] = bool(
            tokenizer_results["qwen"].get("longest_tinyllm_prompt_accepted")
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_role": "dataset_and_interface_contract_not_model_quality_evidence",
        "configuration": asdict(config),
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
        "output_labels": OUTPUT_LABELS,
        "splits": split_results,
        "cross_split": cross_split,
        "tokenizers": tokenizer_results,
        "gates": gates,
        "valid": all(gates.values()),
    }


def contract_worker(experiment: Experiment, device_id: int) -> ExperimentResult:
    """NAL worker for the one independently retryable dataset-contract cell."""
    started = time.perf_counter()
    config = Config(**experiment.parameters["configuration"])
    payload = build_contract(config)
    output = Path(experiment.parameters["output"])
    payload["manifests"] = write_manifests(config, output.parent / "manifests")
    _write_json(output, payload)
    return ExperimentResult(
        experiment_id=experiment.id,
        hypothesis_id=experiment.hypothesis_id,
        metrics={
            "contract_valid": float(payload["valid"]),
            "rows": float(sum(value["rows"] for value in payload["splits"].values())),
            "tinyllm_max_tokens": float(payload["tokenizers"]["tinyllm"]["maximum"]),
            "smollm_max_tokens": float(payload["tokenizers"]["smollm"]["maximum"]),
        },
        primary_metric=float(payload["valid"]),
        model_architecture=[],
        model_parameters=0,
        training_time=time.perf_counter() - started,
        observations=[
            "PAWS dataset and tokenizer contract; no model training performed.",
            f"detail={output}",
            f"logical_device_id={device_id}",
        ],
    )


async def run_nal(output: Path, config: Config = Config()) -> dict[str, Any]:
    experiment = Experiment(
        id="paws-abc-routing-experiment-01-dataset-contract",
        hypothesis_id=HYPOTHESIS_ID,
        name="PAWS A/B/C dataset and tokenizer contract",
        parameters={"configuration": asdict(config), "output": str(output)},
        device_id=-1,
        seed=0,
    )
    runner = AsyncExperimentRunner(
        LabConfig(
            project_name="paws_abc_routing_experiment_01",
            results_dir=str(output.parent / "nal"),
            device_ids=[-1],
            max_parallel_experiments=1,
            min_experiments_per_hypothesis=1,
            require_statistical_significance=False,
            enable_wandb=False,
            verbose=True,
        ),
        contract_worker,
    )
    results = await runner.run_experiments([experiment])
    if len(results) != 1 or results[0].error is not None:
        error = None if not results else results[0].error
        raise RuntimeError(f"NAL dataset-contract cell failed: {error}")
    result = results[0]
    _write_json(
        output.parent / "nal_result.json",
        {
            "experiment_id": result.experiment_id,
            "hypothesis_id": result.hypothesis_id,
            "status": result.status.value,
            "metrics": result.metrics,
            "primary_metric": result.primary_metric,
            "training_time": result.training_time,
            "observations": result.observations,
            "timestamp": result.timestamp.isoformat(),
        },
    )
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/experiments/paws_abc_routing/2026-08-16_experiment_01/contract.json"
        ),
    )
    args = parser.parse_args()
    payload = asyncio.run(
        run_nal(args.output, Config(validate_qwen_endpoint=True))
    )
    print(
        json.dumps(
            {"output": str(args.output), "valid": payload["valid"], "gates": payload["gates"], "tokenizers": payload["tokenizers"]},
            indent=2,
        )
    )
    raise SystemExit(0 if payload["valid"] else 1)


if __name__ == "__main__":
    main()
