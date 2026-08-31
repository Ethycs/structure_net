#!/usr/bin/env python3
"""Experiments 04/05: frozen-representation PAWS distillation heads under NAL."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_architecture_lab.core import Experiment, ExperimentResult, LabConfig
from neural_architecture_lab.runners import AsyncExperimentRunner
from structure_net.components.models import TinyLLMConfig, TinyLLMModel

try:
    from experiments.structure_net.paws_dataset_contract import read_split, render_prompt
    from experiments.structure_net.paws_teacher_annotation import eligible_rows, sha256
except ModuleNotFoundError:
    from paws_dataset_contract import read_split, render_prompt
    from paws_teacher_annotation import eligible_rows, sha256

SCHEMA_VERSION = "nal.paws-local-distillation.v1"
HYPOTHESIS_ID = "paws-local-distillation-v1"
PARTITION_SALT = "paws-dev-partition-v1:"
LABELS = ("DIFFERENT", "PARAPHRASE")


def partition(group_id: str) -> int:
    return hashlib.sha256((PARTITION_SALT + group_id).encode()).digest()[0] % 4


def annotations(path: Path) -> dict[str, int]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["status"] == "success":
            result[row["group_id"]] = int(row["teacher_label"])
    return result


def load_tiny(checkpoint: str, device: torch.device) -> TinyLLMModel:
    model = TinyLLMModel(TinyLLMConfig.from_preset("d8", block_size=256, vocab_size=50257, initialization_seed=7))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def encode_rows(model_name: str, rows: list[dict[str, Any]], device: torch.device, batch_size: int) -> np.ndarray:
    values = []
    if model_name == "tinyllm":
        tokenizer = Tokenizer.from_file("data/corpora/babylm_10M_bpe16k.tokenizer.json")
        model = load_tiny("data/experiments/tinyllm_babylm_pretrain/20260812_d8_seed7/checkpoint_step12000.pt", device)
        for start in range(0, len(rows), batch_size):
            encoded = [tokenizer.encode(render_prompt(row["sentence1"], row["sentence2"])).ids for row in rows[start:start + batch_size]]
            lengths = [len(item) for item in encoded]
            ids = torch.zeros((len(encoded), max(lengths)), dtype=torch.long, device=device)
            for index, item in enumerate(encoded): ids[index, :len(item)] = torch.tensor(item, device=device)
            output = model(ids, output_hidden_states=True, return_dict=True)
            hidden = model.transformer["ln_f"](output.hidden_states[-1])
            values.append(hidden[torch.arange(len(encoded), device=device), torch.tensor(lengths, device=device) - 1].float().cpu().numpy())
    else:
        tokenizer = AutoTokenizer.from_pretrained("/data/models/SmolLM2-360M-Instruct", local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained("/data/models/SmolLM2-360M-Instruct", local_files_only=True, torch_dtype=torch.float16).to(device).eval()
        for start in range(0, len(rows), batch_size):
            prompts = [render_prompt(row["sentence1"], row["sentence2"]) for row in rows[start:start + batch_size]]
            batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=8192)
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch, output_hidden_states=True, use_cache=False)
            lengths = batch["attention_mask"].sum(1) - 1
            values.append(output.hidden_states[-1][torch.arange(len(prompts), device=device), lengths].float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(values).astype(np.float16)


def build_cache(model_name: str, destination: Path, device: torch.device) -> None:
    train = eligible_rows(Path("data/datasets/paws-wiki/labeled/train.csv"))
    dev = eligible_rows(Path("data/datasets/paws-wiki/labeled/dev.csv"))
    batch_size = 64 if model_name == "tinyllm" else 16
    train_x = encode_rows(model_name, train, device, batch_size)
    dev_x = encode_rows(model_name, dev, device, batch_size)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, train_x=train_x, train_y=np.asarray([row["label"] for row in train], dtype=np.int8), train_groups=np.asarray([row["group_id"] for row in train]), dev_x=dev_x, dev_y=np.asarray([row["label"] for row in dev], dtype=np.int8), dev_groups=np.asarray([row["group_id"] for row in dev]), dev_partition=np.asarray([partition(row["group_id"]) for row in dev], dtype=np.int8))
    temporary.replace(destination)


def model_digest(model_name: str) -> str:
    paths = [Path("data/experiments/tinyllm_babylm_pretrain/20260812_d8_seed7/checkpoint_step12000.pt"), Path("data/corpora/babylm_10M_bpe16k.tokenizer.json")] if model_name == "tinyllm" else sorted(Path("/data/models/SmolLM2-360M-Instruct").glob("*.safetensors")) + [Path("/data/models/SmolLM2-360M-Instruct/config.json"), Path("/data/models/SmolLM2-360M-Instruct/tokenizer.json")]
    digest=hashlib.sha256()
    for path in paths: digest.update(str(path).encode()); digest.update(sha256(path).encode())
    return digest.hexdigest()


def write_cache_manifest(model_name: str, cache: Path) -> Path:
    manifest={"schema_version":SCHEMA_VERSION,"model":model_name,"cache_sha256":sha256(cache),"model_digest":model_digest(model_name),"train_sha256":sha256(Path("data/datasets/paws-wiki/labeled/train.csv")),"dev_sha256":sha256(Path("data/datasets/paws-wiki/labeled/dev.csv")),"code_sha256":sha256(Path(__file__))}
    path=cache.with_suffix(".manifest.json");path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8");return path


def balanced_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(sum((prediction[target == label] == label).float().mean().item() for label in (0, 1)) / 2)


def worker(experiment: Experiment, device_id: int) -> ExperimentResult:
    started = time.perf_counter()
    p = experiment.parameters
    device = torch.device(f"cuda:{device_id}")
    cache = Path(p["cache"])
    if not cache.exists(): build_cache(p["model"], cache, device)
    manifest_path=cache.with_suffix(".manifest.json")
    if not manifest_path.exists() or json.loads(manifest_path.read_text())["cache_sha256"]!=sha256(cache): raise RuntimeError("feature cache provenance gate failed")
    data = np.load(cache)
    teacher_by_group = annotations(Path(p["annotations"]))
    train_groups = data["train_groups"].tolist()
    train_x = torch.from_numpy(data["train_x"].astype(np.float32)).to(device)
    train_y = torch.from_numpy(data["train_y"].astype(np.int64)).to(device)
    teacher = torch.tensor([teacher_by_group.get(group, -1) for group in train_groups], dtype=torch.long, device=device)
    dev_mask = data["dev_partition"] == 0
    dev_x = torch.from_numpy(data["dev_x"][dev_mask].astype(np.float32)).to(device)
    dev_y = torch.from_numpy(data["dev_y"][dev_mask].astype(np.int64)).to(device)
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-5)
    train_x, dev_x = (train_x - mean) / std, (dev_x - mean) / std
    torch.manual_seed(p["seed"])
    head = torch.nn.Linear(train_x.shape[1], 2).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(p["seed"])
    history, best = [], None
    for epoch in range(50):
        order = torch.randperm(len(train_x), generator=generator).to(device)
        head.train()
        total = 0.0
        for start in range(0, len(order), 512):
            index = order[start:start + 512]
            logits = head(train_x[index])
            human_loss = F.cross_entropy(logits, train_y[index])
            valid_teacher = teacher[index] >= 0
            if p["arm"] == "label_only" or not bool(valid_teacher.any()):
                loss = human_loss
            else:
                loss = 0.75 * human_loss + 0.25 * F.cross_entropy(logits[valid_teacher], teacher[index][valid_teacher])
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += float(loss) * len(index)
        head.eval()
        with torch.no_grad():
            prediction = head(dev_x).argmax(1)
            accuracy = float((prediction == dev_y).float().mean())
            balanced = balanced_accuracy(prediction, dev_y)
        history.append({"epoch": epoch + 1, "loss": total / len(train_x), "accuracy": accuracy, "balanced_accuracy": balanced})
        if best is None or (balanced, accuracy) > (best[0], best[1]): best = (balanced, accuracy, epoch + 1, {key: value.detach().cpu() for key, value in head.state_dict().items()})
    checkpoint = Path(p["checkpoint"]); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "model": p["model"], "arm": p["arm"], "seed": p["seed"], "epoch": best[2], "head_state": best[3], "mean": mean.cpu(), "std": std.cpu(), "labels": LABELS, "cache_sha256": sha256(cache)}, checkpoint)
    return ExperimentResult(experiment_id=experiment.id, hypothesis_id=HYPOTHESIS_ID, metrics={"accuracy": best[1], "balanced_accuracy": best[0]}, primary_metric=best[0], model_architecture=[train_x.shape[1], 2], model_parameters=train_x.shape[1] * 2 + 2, training_time=time.perf_counter() - started, training_history=history, model_checkpoint=str(checkpoint), observations=[f"cache={cache}"])


async def run(model: str, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    cache = output / f"{model}_features.npz"
    experiments = []
    for arm in ("label_only", "teacher_assisted"):
        for seed in (7, 17, 29):
            experiment_id = f"paws-{model}-{arm}-seed{seed}"
            experiments.append(Experiment(id=experiment_id, hypothesis_id=HYPOTHESIS_ID, name=experiment_id, seed=seed, parameters={"model": model, "arm": arm, "seed": seed, "cache": str(cache), "annotations": "data/experiments/paws_abc_routing/2026-08-16_experiment_03/annotations.jsonl", "checkpoint": str(output / "checkpoints" / f"{experiment_id}.pt")}))
    runner = AsyncExperimentRunner(LabConfig(project_name=f"paws_{model}_distillation", results_dir=str(output / "nal"), device_ids=[0], max_parallel_experiments=1, min_experiments_per_hypothesis=1, require_statistical_significance=False, enable_wandb=False), worker)
    results = await runner.run_experiments(experiments)
    payload = {"schema_version": SCHEMA_VERSION, "model": model, "complete": len(results) == 6 and all(result.error is None for result in results), "results": {result.experiment_id: {"metrics": result.metrics, "checkpoint": result.model_checkpoint, "error": result.error} for result in results}}
    (output / "campaign_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def prepare_worker(experiment: Experiment, device_id: int) -> ExperimentResult:
    started = time.perf_counter(); cache = Path(experiment.parameters["cache"]); model = experiment.parameters["model"]
    if not cache.exists(): build_cache(model, cache, torch.device(f"cuda:{device_id}"))
    manifest=write_cache_manifest(model,cache)
    data = np.load(cache)
    complete = len(data["train_groups"]) > 49000 and len(data["dev_groups"]) > 7900
    return ExperimentResult(experiment_id=experiment.id, hypothesis_id=HYPOTHESIS_ID, metrics={"train_groups": float(len(data["train_groups"])), "dev_groups": float(len(data["dev_groups"]))}, primary_metric=float(complete), model_architecture=[], model_parameters=0, training_time=time.perf_counter()-started, observations=[f"cache={cache}", f"sha256={sha256(cache)}",f"manifest={manifest}"], error=None if complete else "feature cache row gate failed")


async def prepare(model: str, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True); cache=output/f"{model}_features.npz"
    experiment=Experiment(id=f"paws-{model}-feature-cache",hypothesis_id=HYPOTHESIS_ID,name=f"PAWS {model} feature cache",parameters={"model":model,"cache":str(cache)})
    runner=AsyncExperimentRunner(LabConfig(project_name=f"paws_{model}_feature_cache",results_dir=str(output/"nal_prepare"),device_ids=[0],max_parallel_experiments=1,min_experiments_per_hypothesis=1,require_statistical_significance=False,enable_wandb=False),prepare_worker)
    result=(await runner.run_experiments([experiment]))[0]
    return {"complete":result.error is None,"metrics":result.metrics,"error":result.error,"cache":str(cache)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", choices=("tinyllm", "smollm"), required=True); parser.add_argument("--output", type=Path); parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(); output = args.output or Path(f"data/experiments/paws_abc_routing/2026-08-16_experiment_{'05' if args.model == 'tinyllm' else '04'}")
    result = asyncio.run(prepare(args.model, output) if args.prepare_only else run(args.model, output)); print(json.dumps(result, indent=2)); raise SystemExit(0 if result["complete"] else 1)


if __name__ == "__main__": main()
