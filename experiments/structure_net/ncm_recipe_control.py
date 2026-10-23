"""Original-checkpoint XSum control before adapting the NCM training recipe."""
import argparse
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import urllib.request

from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

ROOT = Path("data/experiments/ncm_recipe_update")
INPUTS = ROOT / "20261022_inputs_v1"
PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-recipe-update-plan.md")
EXPECTED = {"passages.pickle": "39b820f0a71ddd6ad178b2700eeae6d035729053b1b2f115d3a92810c6ebf4f8",
            "reference_summaries.pickle": "db624e065b6685dfea25db9106752e837f06f26b799763593759efd150fe554a"}


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("dataset contains executable/custom pickle objects")


def safe_load(path):
    return PrimitiveUnpickler(io.BytesIO(Path(path).read_bytes())).load()


def codec_windows(ids, canvas=48, pad=50257):
    if not ids or canvas < 1:
        raise ValueError("nonempty reference and positive canvas required")
    return [ids[i:i+canvas] + [pad]*max(0, canvas-len(ids[i:i+canvas]))
            for i in range(0, len(ids), canvas)]


def prepare():
    if INPUTS.exists():
        raise ValueError("fresh inputs directory required")
    INPUTS.mkdir(parents=True)
    api = "https://huggingface.co/api/datasets/hazyresearch/ncm-datasets/tree/main/xsum"
    with urllib.request.urlopen(api, timeout=60) as response:
        metadata = json.load(response)
    entries = {Path(row["path"]).name: row for row in metadata}
    for name, expected in EXPECTED.items():
        if entries[name]["lfs"]["oid"] != expected:
            raise ValueError("published dataset identity changed")
        url = "https://huggingface.co/datasets/hazyresearch/ncm-datasets/resolve/main/xsum/" + name
        with urllib.request.urlopen(url, timeout=90) as response, (INPUTS / name).open("xb") as output:
            shutil.copyfileobj(response, output)
        if file_hash(INPUTS / name) != expected:
            raise ValueError("dataset download SHA256 mismatch")
        values = safe_load(INPUTS / name)
        if not isinstance(values, list) or len(values) != 11334:
            raise ValueError("unexpected released XSum dataset")
    write_json(INPUTS / "receipt.json", {"status": "public_sha256_verified", "metadata_url": api,
        "metadata": metadata, "files": EXPECTED, "created_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "prepared", "examples": 11334}), flush=True)


def run(args):
    import torch
    from experiments.structure_net.ncm_llama import (
        DEPENDENCIES, GPT2, INITIAL, VENDOR, autocast, encode_codes, load_ncm, tokenizers, vendor_classes)
    from experiments.structure_net.ncm_training_recipe import cumulative_latent
    vendor_classes()
    sys.path.insert(0, str((ROOT / "dependencies").resolve()))
    from rouge_score import rouge_scorer
    if args.output.exists() or not torch.cuda.is_available():
        raise ValueError("fresh output and CUDA required")
    for name, expected in EXPECTED.items():
        if file_hash(INPUTS / name) != expected:
            raise ValueError("released data changed")
    if args.command == "control":
        pilot = json.loads((args.pilot / "result.json").read_text())
        if pilot["status"] != "passed":
            raise ValueError("successful original-checkpoint CUDA pilot required")
    args.output.mkdir(parents=True)
    source = [Path(__file__), PLAN, Path("experiments/structure_net/ncm_training_recipe.py"),
              Path("experiments/structure_net/ncm_llama.py"),
              Path("experiments/structure_net/ncm_memory_data.py"),
              Path("docs/07 - Status Reports/2026-10-22_ncm-recipe-control-windowing-fix.md")]
    source += sorted(VENDOR.rglob("*.py"))
    source += sorted((ROOT / "dependencies/rouge_score").glob("*.py"))
    source += [ROOT / "dependencies/nltk/stem/porter.py"]
    inputs = [INITIAL / n for n in ("config.yaml", "tokenizer.pt", "generator.pt")]
    inputs += [INPUTS / n for n in (*EXPECTED, "receipt.json")]
    inputs += [p for p in GPT2.iterdir() if p.is_file()]
    copies = args.output / "producer_source"
    copies.mkdir()
    for i, p in enumerate(source):
        shutil.copyfile(p, copies / f"{i:03d}_{p.name}")
    count = 2 if args.command == "pilot" else 128
    producer = {"schema": "nal.ncm-original-control.v1", "hypothesis_id": "ncm-recipe-update-v1",
        "source_hashes": {str(p): file_hash(p) for p in source},
        "input_hashes": {str(p): file_hash(p) for p in inputs}, "cases": count,
        "batch": 2 if args.command == "pilot" else 8, "physical_gpu": 0,
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "precision": "unmodified released shard FP32", "updates": 0,
        "scope": "limited original-domain control; not full paper replication or task accuracy"}
    producer["fingerprint"] = text_hash(json.dumps(producer, sort_keys=True))
    write_json(args.output / "producer.json", producer)
    env = {**os.environ, "NCM_ROOT": str(VENDOR.resolve()),
           "PYTHONPATH": os.pathsep.join([str(DEPENDENCIES.resolve()), str(VENDOR.resolve()), env_path])
           if (env_path := os.environ.get("PYTHONPATH", "")) else
           os.pathsep.join([str(DEPENDENCIES.resolve()), str(VENDOR.resolve())])}
    command = [sys.executable, str(VENDOR / "scripts/eval_shard.py"), "--config", str(INITIAL / "config.yaml"),
        "--checkpoint-dir", str(INITIAL), "--data-dir", str(INPUTS), "--n", str(count),
        "--bs", str(producer["batch"]), "--out", str(args.output / "official_outputs.pickle")]
    write_json(args.output / "command.json", command)
    if args.reuse_shard is not None:
        old = json.loads((args.reuse_shard / "producer.json").read_text())
        if old["cases"] != count or old["input_hashes"] != producer["input_hashes"]:
            raise ValueError("reused official shard input/cohort mismatch")
        for p, digest in old["source_hashes"].items():
            if str(VENDOR) in p and file_hash(p) != digest:
                raise ValueError("official generating source changed")
        for name in ("official_outputs.pickle", "official_shard.log"):
            shutil.copyfile(args.reuse_shard / name, args.output / name)
        write_json(args.output / "generation_reuse.json", {"source": str(args.reuse_shard),
            "producer_sha256": file_hash(args.reuse_shard / "producer.json"),
            "outputs_sha256": file_hash(args.reuse_shard / "official_outputs.pickle"),
            "reason": "Original generation completed; only reference codec windowing changes."})
    else:
        with (args.output / "official_shard.log").open("x") as output:
            subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    outputs = safe_load(args.output / "official_outputs.pickle")
    if outputs["idxs"] != list(range(count)):
        raise ValueError("original evaluation cohort mismatch")
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [sc.score(ref, hyp) for hyp, ref in zip(outputs["hyps"], outputs["refs"])]
    rouge = {key: 100*sum(row[key].fmeasure for row in scores)/count for key in scores[0]}
    # Measure the released codec separately, without modifying its generator.
    vae, net, _ = load_ncm()
    del net
    gpt, _ = tokenizers()
    refs = safe_load(INPUTS / "reference_summaries.pickle")[:count]
    targets = []
    for ref in refs:
        ids = gpt.encode(ref, add_special_tokens=False) if isinstance(ref, str) else list(ref)
        targets.extend(codec_windows(ids))
    targets = torch.tensor(targets)
    correct = total = 0
    with torch.inference_mode():
        for start in range(0, len(targets), 8):
            target = targets[start:start+8].cuda()
            codes = encode_codes(vae, target)
            # Same FP32 inference as the released shard.
            logits = vae.decoder(cumulative_latent(vae, codes))
            valid = target.ne(50257)
            correct += int((logits.argmax(-1).eq(target) & valid).sum())
            total += int(valid.sum())
    for p, expected in {**producer["source_hashes"], **producer["input_hashes"]}.items():
        if file_hash(p) != expected:
            raise ValueError("control producer changed")
    passed = all(outputs["hyps"]) and (args.command == "pilot" or (rouge["rougeL"] >= 15 and correct/total >= .98))
    write_json(args.output / "outputs.json", outputs)
    result = {"status": "passed" if passed else "original_checkpoint_control_failed",
        "cases": count, "rouge": rouge, "codec_correct": correct, "codec_total": total,
        "codec_accuracy": correct/total, "reference_codec_windows": len(targets),
        "producer_fingerprint": producer["fingerprint"],
        "task_accuracy": None, "speed_tested": False,
        "label": "systems_lifecycle_only_not_quality_evidence" if args.command == "pilot" else "original_domain_sanity_control"}
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "pilot", "control"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--reuse-shard", type=Path)
    args = parser.parse_args()
    prepare() if args.command == "prepare" else run(args)
