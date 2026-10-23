"""Released-checkpoint and codec controls for the paper-regime recovery."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from experiments.structure_net.ncm_llama import (
    INITIAL, VENDOR, autocast, decode_ids, encode_codes, infer_ncm, load_ncm, tokenizers)
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import (
    PLAN, PREP, ROOT, TEACHER, freeze, verify)
from experiments.structure_net.ncm_training_recipe import cumulative_latent
from experiments.structure_net.ncm_vocabulary import CachedVocabulary


def producer(root, stage, inputs, config):
    result = freeze(root, stage, inputs, config)
    paths = [Path(__file__), Path("experiments/structure_net/ncm_training_recipe.py")]
    paths += sorted(VENDOR.rglob("*.py"))
    for i, path in enumerate(paths):
        result["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"control_{i:03d}_{path.name}").write_bytes(path.read_bytes())
    from experiments.structure_net import ncm_memory_accuracy_v2 as q
    result["fingerprint"] = q.digest({k: v for k, v in result.items() if k != "fingerprint"})
    write_json(root / "producer.json", result)
    return result


def baseline(pilot):
    from experiments.structure_net import ncm_memory_accuracy_v2 as q
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    root = ROOT / ("20261022_baseline_pilot_v1" if pilot else "20261022_baseline_v1")
    cases = q.read(PREP / "development_sources.json")[:2 if pilot else 64]
    if not pilot and q.read(ROOT / "20261022_baseline_pilot_v1/result.json")["status"] != "passed":
        raise ValueError("baseline lifecycle pilot required")
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "released_native_development_control",
        [PREP / "development_sources.json", *[INITIAL / n for n in ("config.yaml", "tokenizer.pt", "generator.pt")]],
        {"cases": len(cases), "batch": 4, "seed": 42, "decoding": "greedy raw logits, EOS-or-PAD stop"})
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    gpt, _ = tokenizers()
    gpt = CachedVocabulary(gpt)
    vae, net, _ = load_ncm()
    outputs = []
    start = time.perf_counter()
    for i in range(0, len(cases), 4):
        chunk = cases[i:i+4]
        prefix = [gpt.encode(r["source"], add_special_tokens=False) for r in chunk]
        if any(len(ids) > 512 for ids in prefix):
            raise ValueError("native prefix truncation prohibited")
        prefix = torch.tensor([ids+[50257]*(512-len(ids)) for ids in prefix], device="cuda")
        outputs.extend({**case, **output} for case, output in zip(chunk, infer_ncm(vae, net, prefix, gpt)))
    write_json(root / "outputs.json", outputs)
    verify(p)
    result = {"status": "passed", "label": p["stage"], "cases": len(outputs),
        "producer_fingerprint": p["fingerprint"], "elapsed_seconds": time.perf_counter()-start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "task_accuracy": None,
        "trained": False, "speed_tested": False}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def codec():
    from experiments.structure_net import ncm_memory_accuracy_v2 as q
    from experiments.structure_net.ncm_memory_data import text_hash
    if not torch.cuda.is_available() or q.read(TEACHER / "result.json")["status"] != "passed":
        raise ValueError("CUDA and complete teacher data required")
    root = ROOT / "20261022_codec_control_v1"
    cases = q.read(TEACHER / "train.json")
    p = producer(root, "released_codec_on_new_training_labels", [TEACHER / "train.json",
        *[INITIAL / n for n in ("config.yaml", "tokenizer.pt", "generator.pt")]], {"batch": 32, "required_nonpad": .98})
    gpt, _ = tokenizers()
    gpt = CachedVocabulary(gpt)
    vae, net, _ = load_ncm()
    del net
    gc.collect()
    torch.cuda.empty_cache()
    correct = total = content_correct = content_total = exact = 0
    outputs = []
    for i in range(0, len(cases), 32):
        chunk = cases[i:i+32]
        targets = torch.tensor([r["native_target_ids"] for r in chunk], device="cuda")
        codes = encode_codes(vae, targets)
        with torch.inference_mode(), autocast():
            ids = vae.decoder(cumulative_latent(vae, codes)).argmax(-1)
        nonpad, content = targets.ne(50257), targets.ne(50257) & targets.ne(50256)
        correct += int((ids.eq(targets) & nonpad).sum())
        total += int(nonpad.sum())
        content_correct += int((ids.eq(targets) & content).sum())
        content_total += int(content.sum())
        for row, generated in zip(chunk, ids.tolist()):
            text, termination = decode_ids(generated, gpt)
            exact += text == row["target"]
            outputs.append({**row, "text": text, "native_output_ids": generated, "termination": termination})
    write_json(root / "outputs.json", outputs)
    verify(p)
    result = {"status": "passed_reconstruction_pending_semantic_control" if correct/total >= .98 else "codec_retraining_required",
        "cases": len(cases), "nonpad_correct": correct, "nonpad_total": total,
        "nonpad_reconstruction": correct/total, "content_reconstruction": content_correct/content_total,
        "exact_text_cases": exact, "producer_fingerprint": p["fingerprint"],
        "test_accuracy": None, "trained": False}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["baseline-pilot", "baseline", "codec"])
    args = parser.parse_args()
    if args.command.startswith("baseline"):
        baseline(args.command.endswith("pilot"))
    else:
        codec()
