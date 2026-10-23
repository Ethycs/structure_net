"""Broaden reconstruction training while keeping the released code language fixed."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
import os
from pathlib import Path
import re
import time

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_llama import GPT2, INITIAL, decode_ids, encode_codes, label_ids, tokenizers, vendor_classes
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.ncm_training_recipe import cumulative_latent
from experiments.structure_net.ncm_unseen_codec_control import CODEC, MEASURE, NOVEL, SEEN, cohorts, validate_targets
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.train_ncm_decoder_polish import frozen_hash
from experiments.structure_net.train_ncm_memory import save

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-broad-decoder-transfer-plan.md")
SOURCES = ROOT / "20261022_expanded_data_v2"
DATA = ROOT / "20261022_broad_decoder_data_v1"
PILOT = ROOT / "20261022_broad_decoder_pilot_v1"
TRAIN = ROOT / "20261022_broad_decoder_train_v1"


@dataclass(frozen=True)
class Config:
    steps: int = 2048
    batch: int = 64
    raw_per_batch: int = 48
    anchor_per_batch: int = 16
    quantization_batch: int = 128
    eval_batch: int = 16
    lr: float = 1e-4
    warmup: int = 64
    weight_decay: float = .01
    seed: int = 42
    dropout_seed: int = 52
    sampler_seed: int = 2042
    minimum_raw_training: int = 30000


def split_sources(rows):
    if len(rows) != 8192:
        raise ValueError("exact8192 source cohort required")
    for key in ("id", "source_id", "group_id", "source_sha256"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError("repeated source unit: "+key)
    ordered = sorted(rows, key=lambda r: text_hash("ncm-broad-decoder-split-v1:"+r["id"]))
    return ordered[1024:], ordered[:1024]


def source_windows(row, tokenizer):
    words = list(re.finditer(r"\S+", row["source"]))
    candidates = {}
    rejected = 0
    for length in (8, 16, 24, 32):
        for start in range(0, len(words)-length+1, 16):
            a, b = words[start].start(), words[start+length-1].end()
            text = row["source"][a:b]
            try:
                ids = label_ids(text, tokenizer)
            except ValueError:
                rejected += 1
                continue
            digest = text_hash(text)
            candidates.setdefault(digest, {"id": text_hash(row["id"]+":"+str(a)+":"+str(b)),
                "source_id": row["source_id"], "group_id": row["group_id"], "source_sha256": row["source_sha256"],
                "source_record_id": row["id"], "char_start": a, "char_end": b,
                "target": text, "target_sha256": digest, "native_target_ids": ids})
    ordered = sorted(candidates.values(), key=lambda r: text_hash("ncm-broad-decoder-window-v1:"+r["id"]))
    return ordered[:8], rejected


def qualify(metrics):
    return all(metrics[k]["nonpad_reconstruction"] >= .995 and
               metrics[k]["exact_texts"]/metrics[k]["cases"] >= .9 for k in ("raw_validation", "novel"))


def freeze(root, stage, inputs, config):
    p = producer(root, stage, inputs, config)
    for path in (Path(__file__), PLAN, Path("experiments/structure_net/ncm_unseen_codec_control.py"),
                 Path("experiments/structure_net/train_ncm_decoder_polish.py")):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("broad_decoder_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-broad-decoder-transfer-v1"
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def prepare():
    cfg = Config()
    if q.read(MEASURE / "result.json")["adapted_novel_codec_qualified"]:
        raise ValueError("this remedy requires the preserved failed novel-codec gate")
    verify(q.read(MEASURE / "producer.json"))
    inputs = [SOURCES/n for n in ("train_sources.json", "producer.json", "public_proof.json", "result.json")]
    inputs += [SEEN, NOVEL, MEASURE / "result.json"]
    inputs += sorted(p for p in GPT2.iterdir() if p.is_file())
    p = freeze(DATA, "public_unpaired_codec_data_before_training", inputs, asdict(cfg))
    all_sources, old, novel = q.read(SOURCES / "train_sources.json"), q.read(SEEN), q.read(NOVEL)
    train, validation = split_sources(all_sources)
    for key in ("id", "source_id", "group_id", "source_sha256"):
        groups = [{r[key] for r in xs} for xs in (all_sources, old, novel)]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("source groups overlap old/new data")
    proof = verify_public(all_sources)
    gpt, _ = tokenizers(); gpt = CachedVocabulary(gpt)
    validation_rows, train_rows, excluded = [], [], {"oversize_candidates": 0, "training_duplicates_or_forbidden": 0}
    seen_hashes = set()
    for row in validation:
        windows, rejected = source_windows(row, gpt)
        excluded["oversize_candidates"] += rejected
        candidate = next((r for r in windows if r["target_sha256"] not in seen_hashes), None)
        if candidate:
            validation_rows.append(candidate); seen_hashes.add(candidate["target_sha256"])
        if len(validation_rows) == 256:
            break
    forbidden = {r["target_sha256"] for r in validation_rows+novel}
    unique_training = set()
    for i, row in enumerate(train):
        windows, rejected = source_windows(row, gpt)
        excluded["oversize_candidates"] += rejected
        for candidate in windows:
            key = candidate["target_sha256"]
            if key in forbidden or key in unique_training:
                excluded["training_duplicates_or_forbidden"] += 1
                continue
            train_rows.append(candidate); unique_training.add(key)
        if (i+1) % 1024 == 0:
            print(json.dumps({"prepared_sources": i+1, "raw_windows": len(train_rows)}), flush=True)
    anchors = []
    for row in old:
        if row["target_sha256"] in forbidden or row["target_sha256"] in unique_training:
            excluded["training_duplicates_or_forbidden"] += 1
            continue
        anchors.append(row); unique_training.add(row["target_sha256"])
    old_sample = cohorts(old, novel)["seen"]
    validate_targets(validation_rows+train_rows+anchors+novel, gpt)
    for name, rows in (("raw_train", train_rows), ("raw_validation", validation_rows),
                       ("anchors", anchors), ("novel", novel), ("seen", old_sample)):
        write_json(DATA / (name+".json"), rows)
    write_json(DATA / "source_split.json", {"train": [r["id"] for r in train], "validation": [r["id"] for r in validation]})
    write_json(DATA / "exclusions.json", excluded)
    verify(p)
    passed = len(train_rows) >= cfg.minimum_raw_training and len(validation_rows) == 256 and len(anchors) >= 1900
    result = {"status": "passed" if passed else "failed_data_gate", "raw_training": len(train_rows),
        "anchors": len(anchors), "validation": len(validation_rows), "novel": len(novel),
        "public_source_count": len(proof), "source_disjoint": True, "producer_fingerprint": p["fingerprint"],
        "output_hashes": {str(path): file_hash(path) for path in DATA.glob("*.json") if path.name != "producer.json"},
        "trained": False, "task_accuracy": None, "final_test_opened": False}
    write_json(DATA / "result.json", result); print(json.dumps(result), flush=True)


@torch.inference_mode()
def evaluate(vae, codes, targets, rows, cfg):
    vae.decoder.eval()
    correct = total = exact = 0
    output = []
    gpt, _ = tokenizers(); gpt = CachedVocabulary(gpt)
    for start in range(0, len(rows), cfg.eval_batch):
        sl = slice(start, start+cfg.eval_batch)
        latent = cumulative_latent(vae, codes[sl].cuda())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = vae.decoder(latent)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite decoder logits")
        raw = logits.argmax(-1).cpu()
        valid = targets[sl].ne(50257)
        correct += int((raw.eq(targets[sl]) & valid).sum()); total += int(valid.sum())
        for row, ids in zip(rows[sl], raw.tolist()):
            text, termination = decode_ids(ids, gpt)
            exact += text == row["target"]
            output.append({"id": row["id"], "target": row["target"], "text": text,
                           "native_output_ids": ids, "termination": termination})
    return {"cases": len(rows), "nonpad_correct": correct, "nonpad_total": total,
            "nonpad_reconstruction": correct/total, "exact_texts": exact}, output


def run(pilot):
    cfg = Config()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("select physicalGPU0 before startup")
    if not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name():
        raise ValueError("planned CUDA device unavailable")
    if torch.cuda.mem_get_info()[0] < 6_000_000_000:
        raise ValueError("insufficient GPU0 headroom")
    data = q.read(DATA / "result.json")
    if data["status"] != "passed":
        raise ValueError("qualified unpaired data required")
    verify(q.read(DATA / "producer.json"))
    for path, expected in data["output_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("prepared data changed")
    root = PILOT if pilot else TRAIN
    inputs = [DATA/n for n in ("result.json", "producer.json", "raw_train.json", "raw_validation.json",
                               "anchors.json", "seen.json", "novel.json")]
    inputs += [INITIAL / "config.yaml", INITIAL / "tokenizer.pt", MEASURE / "result.json"]
    p = freeze(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "fixed_language_broad_decoder_training",
        inputs, {**asdict(cfg), "actual_steps": 2 if pilot else cfg.steps, "precision": "FP32 quantization, BF16 decoder"})
    if not pilot:
        prior, receipt = q.read(PILOT / "producer.json"), q.read(PILOT / "result.json")
        if receipt["status"] != "passed" or any(prior[k] != p[k] for k in ("source_hashes", "input_hashes")):
            raise ValueError("matching real-batch CUDA lifecycle pilot required")
        verify(prior)
    torch.manual_seed(cfg.seed); torch.cuda.reset_peak_memory_stats()
    OmegaConf, VQVAE, _ = vendor_classes()
    config = OmegaConf.load(INITIAL / "config.yaml")
    vae = VQVAE(**config.tokenizer.vqvae)
    initial = torch.load(INITIAL / "tokenizer.pt", map_location="cpu", weights_only=True)
    vae.load_state_dict(initial["model"], strict=True); del initial; gc.collect()
    vae.cuda().float().eval().requires_grad_(False)
    original_frozen, original_decoder = frozen_hash(vae), module_hash(vae.decoder)
    banks, targets, codes = {}, {}, {}
    sizes = {"raw_train": 128, "anchors": 128, "raw_validation": 8, "novel": 4, "seen": 4}
    for name in sizes:
        rows = q.read(DATA / (name+".json")); banks[name] = rows[:sizes[name]] if pilot else rows
        targets[name] = torch.tensor([r["native_target_ids"] for r in banks[name]])
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(banks[name]), cfg.quantization_batch):
                chunks.append(encode_codes(vae, targets[name][start:start+cfg.quantization_batch].cuda()).cpu())
                if start % 4096 == 0:
                    print(json.dumps({"cache": name, "completed": min(start+cfg.quantization_batch,len(banks[name])),
                                      "total": len(banks[name])}), flush=True)
        codes[name] = torch.cat(chunks)
    save(root / "code_cache.pt", {"codes": codes, "targets": targets})
    reload = torch.load(root / "code_cache.pt", weights_only=True)
    if any(not torch.equal(reload["codes"][n], codes[n]) or not torch.equal(reload["targets"][n], targets[n]) for n in banks):
        raise ValueError("code cache reload mismatch")
    del reload; gc.collect()
    eval_names = ("raw_validation", "novel", "seen")
    def measure(label):
        measurements, outputs = {}, {}
        with torch.inference_mode():
            for name in eval_names:
                measurements[name], outputs[name] = evaluate(vae, codes[name], targets[name], banks[name], cfg)
        write_json(root / (label+"_metrics.json"), measurements)
        write_json(root / (label+"_outputs.json"), outputs)
        return measurements, outputs
    baseline, _ = measure("baseline")
    vae.decoder.requires_grad_(True)
    parameters = list(vae.decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, betas=(.9, .95), weight_decay=cfg.weight_decay)
    rng = torch.Generator().manual_seed(cfg.sampler_seed)
    torch.manual_seed(cfg.dropout_seed); torch.cuda.manual_seed_all(cfg.dropout_seed)
    steps, trace, max_gradient = (2 if pilot else cfg.steps), [], 0.
    start_time = time.perf_counter()
    for step in range(1, steps+1):
        vae.decoder.train(); optimizer.zero_grad(set_to_none=True)
        rate = min(step/cfg.warmup, 1.) if step <= cfg.warmup else .5*(1+math.cos(math.pi*(step-cfg.warmup)/max(1, steps-cfg.warmup)))
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr*rate
        idx = {"raw_train": torch.randint(len(banks["raw_train"]), (cfg.raw_per_batch,), generator=rng),
               "anchors": torch.randint(len(banks["anchors"]), (cfg.anchor_per_batch,), generator=rng)}
        clean = torch.cat([codes[n][i] for n, i in idx.items()]).cuda()
        ids = torch.cat([targets[n][i] for n, i in idx.items()]).cuda()
        with torch.no_grad():
            latent = cumulative_latent(vae, clean)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = vae.decoder(latent)
            loss = F.cross_entropy(logits.float().flatten(0, 1), ids.flatten(), ignore_index=50257)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite reconstruction objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        max_gradient = max(max_gradient, float(norm)); optimizer.step()
        trace.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm), "lr": cfg.lr*rate})
        if step % 64 == 0 or step == steps:
            write_json(root / "trace.json", trace); print(json.dumps(trace[-1]), flush=True)
        if step % 512 == 0 or step == steps:
            measure(f"step{step}")
            if frozen_hash(vae) != original_frozen:
                raise ValueError("released code language changed")
            save(root / f"step{step}.pt", {"model": vae.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "sampler_rng": rng.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "config": asdict(cfg),
                "producer_fingerprint": p["fingerprint"], "frozen_nondecode_hash": original_frozen})
    vae.requires_grad_(False).eval()
    if (max_gradient <= 0 or frozen_hash(vae) != original_frozen or module_hash(vae.decoder) == original_decoder or
            any(x.grad is not None for name, x in vae.named_parameters() if not name.startswith("decoder."))):
        raise ValueError("decoder-only update invariant failed")
    with torch.inference_mode():
        if not torch.equal(encode_codes(vae, targets["novel"].cuda()).cpu(), codes["novel"]):
            raise ValueError("novel target code assignments changed")
    final, outputs = measure("final")
    model_hash = module_hash(vae)
    reload = torch.load(root / f"step{steps}.pt", map_location="cpu", weights_only=True)
    vae.load_state_dict(reload["model"], strict=True); del reload; gc.collect(); torch.clear_autocast_cache()
    repeated, texts = measure("reload")
    if model_hash != module_hash(vae) or final != repeated or outputs != texts:
        raise ValueError("checkpoint reload changed codec predictions")
    verify(p)
    result = {"status": "passed" if pilot else ("codec_transfer_gate_passed" if qualify(final) else "codec_transfer_gate_failed"),
        "hypothesis_id": p["hypothesis_id"], "steps": steps, "baseline": baseline, "final": final,
        "codec_transfer_qualified": None if pilot else qualify(final), "nondecoder_unchanged": True,
        "frozen_nondecode_hash": original_frozen, "novel_codes_unchanged": True, "reload_exact": True,
        "model_hash": model_hash, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter()-start_time, "producer_fingerprint": p["fingerprint"],
        "output_hashes": {str(path): file_hash(path) for path in root.glob("*") if path.is_file() and path.name != "producer.json"},
        "target_aware": True, "task_accuracy": None, "student_training_admitted": False,
        "speed_tested": False, "final_test_opened": False, "confirmed": False, "underpowered": True}
    write_json(root / "result.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "output_hashes"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "pilot", "train"])
    command = parser.parse_args().command
    prepare() if command == "prepare" else run(command == "pilot")
