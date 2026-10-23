"""Check unseen-text codec reconstruction before blaming summary-code prediction."""
import argparse
import gc
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_llama import GPT2, INITIAL, decode_ids, encode_codes, label_ids, tokenizers, vendor_classes
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.ncm_training_recipe import cumulative_latent
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.train_ncm_memory import save

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-unseen-codec-control-plan.md")
SEEN = ROOT / "20261022_teacher_v1/train.json"
NOVEL = ROOT / "20261022_expanded_teacher_pilot_v2/train.json"
CODEC = ROOT / "20261022_decoder_train_v1/model.pt"
EXPECTED_CODEC = "d8c61b5a96ba0e151a827924e4a17e415664baa5d23f6a8d47b736373a14dcbe"
PILOT = ROOT / "20261022_unseen_codec_pilot_v1"
MEASURE = ROOT / "20261022_unseen_codec_measure_v1"


def cohorts(seen, novel):
    if len(seen) != 1999 or len(novel) != 20:
        raise ValueError("exact frozen cohort sizes required")
    for key in ("id", "source_id", "group_id", "source_sha256"):
        a, b = {r[key] for r in seen}, {r[key] for r in novel}
        if a & b or len(b) != len(novel):
            raise ValueError("novel codec cohort overlaps training or repeats "+key)
    return {"seen": sorted(seen, key=lambda r: text_hash("ncm-unseen-codec-control-v1:"+r["id"]))[:20],
            "novel": list(novel)}


def validate_targets(cases, tokenizer):
    for case in cases:
        if text_hash(case["target"]) != case["target_sha256"]:
            raise ValueError("target text hash mismatch")
        if label_ids(case["target"], tokenizer) != case["native_target_ids"]:
            raise ValueError("target IDs differ or were truncated")


@torch.inference_mode()
def main(pilot):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("select physicalGPU1 before startup")
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 2060":
        raise ValueError("planned CUDA device unavailable")
    if file_hash(CODEC) != EXPECTED_CODEC or q.read(NOVEL.parent / "result.json")["status"] != "passed":
        raise ValueError("frozen codec and complete novel-label pilot required")
    selected = cohorts(q.read(SEEN), q.read(NOVEL))
    root = PILOT if pilot else MEASURE
    gpt, _ = tokenizers(); gpt = CachedVocabulary(gpt)
    verify_public([r for rows in selected.values() for r in rows])
    validate_targets([r for rows in selected.values() for r in rows], gpt)
    inputs = [SEEN, NOVEL, NOVEL.parent / "producer.json", NOVEL.parent / "result.json",
              CODEC, INITIAL / "config.yaml", INITIAL / "tokenizer.pt"]
    inputs += sorted(p for p in GPT2.iterdir() if p.is_file())
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "unseen_correct_code_codec_control",
        inputs, {"cases_per_cohort": 2 if pilot else 20, "batch": 4, "precision": "FP32",
                 "required_novel_adapted_nonpad": .98, "seed": 42, "physical_gpu": 1})
    for path in (Path(__file__), PLAN):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("unseen_codec_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-unseen-codec-control-v1"
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    if not pilot:
        prior, receipt = q.read(PILOT / "producer.json"), q.read(PILOT / "result.json")
        if receipt["status"] != "passed" or any(prior[k] != p[k] for k in ("source_hashes", "input_hashes")):
            raise ValueError("matching GPU lifecycle pilot required")
        verify(prior)
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
    OmegaConf, VQVAE, _ = vendor_classes()
    config = OmegaConf.load(INITIAL / "config.yaml")
    outputs, metrics, codes, identities = {}, {}, {}, {}
    for name, checkpoint in (("released", INITIAL / "tokenizer.pt"), ("adapted", CODEC)):
        vae = VQVAE(**config.tokenizer.vqvae)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        vae.load_state_dict(state["model"], strict=True); del state; gc.collect()
        vae.cuda().float().eval().requires_grad_(False)
        identities[name] = module_hash(vae)
        outputs[name], metrics[name], codes[name] = {}, {}, {}
        for cohort, rows in selected.items():
            cases = rows[:2] if pilot else rows
            correct = total = exact = 0
            result, tensors = [], []
            for start in range(0, len(cases), 4):
                chunk = cases[start:start+4]
                targets = torch.tensor([r["native_target_ids"] for r in chunk], device="cuda")
                encoded = encode_codes(vae, targets)
                logits = vae.decoder(cumulative_latent(vae, encoded))
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("nonfinite codec logits")
                ids = logits.argmax(-1)
                valid = targets.ne(50257)
                correct += int((ids.eq(targets) & valid).sum()); total += int(valid.sum())
                tensors.append(encoded.cpu())
                for case, code, raw in zip(chunk, encoded.tolist(), ids.tolist()):
                    text, termination = decode_ids(raw, gpt)
                    exact += text == case["target"]
                    result.append({"id": case["id"], "target": case["target"], "text": text,
                        "native_output_ids": raw, "gold_codes": code, "termination": termination})
            metrics[name][cohort] = {"cases": len(cases), "nonpad_correct": correct, "nonpad_total": total,
                "nonpad_reconstruction": correct/total, "exact_texts": exact}
            outputs[name][cohort] = result; codes[name][cohort] = torch.cat(tensors)
        if module_hash(vae) != identities[name] or any(p.grad is not None for p in vae.parameters()):
            raise ValueError("codec changed")
        del vae; gc.collect(); torch.cuda.empty_cache()
    write_json(root / "outputs.json", outputs)
    save(root / "codes.pt", codes)
    reload = torch.load(root / "codes.pt", weights_only=True)
    if any(not torch.equal(reload[a][b], codes[a][b]) for a in codes for b in codes[a]):
        raise ValueError("codec codes changed on reload")
    verify(p)
    result = {"status": "passed" if pilot else "completed", "hypothesis_id": p["hypothesis_id"],
        "metrics": metrics, "adapted_novel_codec_qualified": metrics["adapted"]["novel"]["nonpad_reconstruction"] >= .98,
        "models_frozen_hashes": identities, "reload_exact": True, "precision": "FP32",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "producer_fingerprint": p["fingerprint"],
        "output_hashes": {str(root/n): file_hash(root/n) for n in ("outputs.json", "codes.pt")},
        "target_aware": True, "task_accuracy": None, "trained": False, "confirmed": False,
        "underpowered": True, "final_test_opened": False, "speed_tested": False}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "measure"])
    main(parser.parse_args().command == "pilot")
