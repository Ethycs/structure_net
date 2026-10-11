"""Prospectively corrected extractive assay, separate from runtime campaigns."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
from transformers import AutoTokenizer

from experiments.structure_net import llama_cotenancy_common as common
from experiments.structure_net.llama_cotenancy_data import example, score
from experiments.structure_net.llama_tensorized_model import TensorizedLlama
from experiments.structure_net.prepare_llama_cotenancy import MODEL_PATH, OUTPUT, GGUF_FILE, sha256, write_json
from experiments.structure_net.run_llama_cotenancy import quality, aggregate_quality, BINARY, CPP

AMENDMENT = Path("docs/07 - Status Reports/2026-10-10_llama32-cotenancy-quality-amendment.md")


def fixtures(path):
    if path.exists():
        raise ValueError("preserve immutable fixtures")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    demonstration = example(9988, "work", "demonstration")
    rows = [example(seed, family, "validation") for seed in (2963, 2964) for family in ("work", "personal")]
    rows += [example(seed, family, "test") for seed in (7, 17, 29, 41, 53) for family in ("work", "personal")]
    for row in rows:
        messages = [{"role": "system", "content": "You extract faithful summaries. Copy whole source sentences verbatim, including their subjects. Output only the three final-decision source sentences, one per line. Follow the example's format."},
                    {"role": "user", "content": demonstration["prompt"]},
                    {"role": "assistant", "content": "\n".join(demonstration["gold_fragments"])},
                    {"role": "user", "content": row["prompt"]}]
        row["prompt_ids"] = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        row["reference_ids"] = tokenizer.encode("\n".join(row["gold_fragments"]), add_special_tokens=False)+[128009]
        if len(row["prompt_ids"])+96 > 1024:
            raise ValueError("quality prompt exceeds fixed context budget")
    value = {"schema": "nal.llama-cotenancy-extractive-fixtures.v2", "demonstration": demonstration, "examples": rows,
             "benchmark_ids": [(r["prompt_ids"]*4)[:512] for r in rows[:4]], "eos_ids": [128001, 128008, 128009]}
    write_json(path, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fixtures", "gguf", "evaluate"))
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    if args.stage == "fixtures":
        fixtures(args.fixtures)
        print(json.dumps({"fixtures": str(args.fixtures), "sha256": sha256(args.fixtures)}))
        return
    common.setup()
    data = json.loads(args.fixtures.read_text())
    extra = [Path(__file__), AMENDMENT, args.fixtures, Path("experiments/structure_net/run_llama_cotenancy.py")]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.checkpoint:
        config["checkpoint_sha256"] = sha256(args.checkpoint)
    if args.stage == "gguf":
        extra += [CPP, BINARY, common.ROOT/"llama.cpp/build/CMakeCache.txt"]
        extra += [common.ROOT/"llama.cpp/build/bin"/n for n in ("libllama.so.0.6.0", "libggml.so.0.26.0", "libggml-base.so.0.26.0", "libggml-cpu.so.0.26.0", "libggml-cuda.so.0.26.0")]
    producer = common.producer(args.output, "quality_v2_"+args.stage, extra, config)
    if args.stage == "gguf":
        raw = args.output/"raw.json"
        with (args.output/"runtime.log").open("w") as log:
            result = subprocess.run([str(BINARY.resolve()), str((OUTPUT/GGUF_FILE).resolve()), str(args.fixtures.resolve()), "1", str(raw.resolve()), args.split], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("GGUF quality runtime failed; preserve log")
        result = json.loads(raw.read_text())
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        by_id = {r["id"]: r for r in data["examples"]}
        for row in result["quality"]:
            fixture = by_id[row["id"]]
            row["text"] = tokenizer.decode(row["generated_ids"], skip_special_tokens=True)
            row["metrics"] = score(row["text"], fixture)
            row["family"], row["seed"] = fixture["family"], fixture["seed"]
    else:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
        model = TensorizedLlama(payload)
        result = {"status": "completed", "quality": quality(model, data, args.split)}
    result.update(quality_summary=aggregate_quality(result["quality"]), producer_fingerprint=producer["fingerprint"], gpu_after=common.gpu_state())
    result["uncontended"] = common.uncontended(producer["gpu_before"], result["gpu_after"])
    write_json(args.output/"result.json", result)
    print(json.dumps({"completed": str(args.output), "quality": result["quality_summary"]}), flush=True)


if __name__ == "__main__":
    main()
