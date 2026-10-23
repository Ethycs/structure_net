"""Measure reconstructability before teaching error-conditioned hierarchy targets."""
import argparse
import gc
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_conditional_residual_targets import Corruption, sample_corruption, training_targets
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_llama import INITIAL, decode_ids, tokenizers, vendor_classes
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_residual_oracle import nearest_distinct
from experiments.structure_net.ncm_training_recipe import cumulative_latent
from experiments.structure_net.train_ncm_memory import save

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-conditioning-noise-capacity-plan.md")
CODEC = ROOT / "20261022_decoder_train_v1/model.pt"
DATA = ROOT / "20261022_teacher_v1/train.json"
MODES = ("clean", "root_only", "all_levels", "mild_fine")


@torch.inference_mode()
def main(pilot):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("physicalGPU1 must be selected before torch startup")
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 2060":
        raise ValueError("planned GPU1 required")
    cases = sorted(q.read(DATA), key=lambda r: text_hash("ncm-noise-capacity-v1:"+r["id"]))[:4 if pilot else 128]
    root = ROOT / ("20261022_noise_capacity_pilot_v1" if pilot else "20261022_noise_capacity_measure_v1")
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "conditional_noise_codec_capacity",
        [CODEC, DATA, INITIAL / "config.yaml"], {"batch": 32, "cases": len(cases), "seed": 42,
            "augmentation_seed": 3042, "modes": MODES, "quantization": "FP32", "decoder": "FP16", "neighbors": 8})
    for path in (Path(__file__), PLAN, Path("experiments/structure_net/ncm_conditional_residual_targets.py"),
                 Path("experiments/structure_net/ncm_residual_oracle.py")):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("noise_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-conditioning-noise-capacity-v1"
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    if not pilot:
        pr = ROOT / "20261022_noise_capacity_pilot_v1"
        if q.read(pr / "result.json")["status"] != "passed" or q.read(pr / "producer.json")["source_hashes"] != p["source_hashes"]:
            raise ValueError("matching GPU1 lifecycle pilot required")
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
    gpt, _ = tokenizers()
    OmegaConf, VQVAE, _ = vendor_classes()
    config = OmegaConf.load(INITIAL / "config.yaml")
    vae = VQVAE(**config.tokenizer.vqvae)
    state = torch.load(CODEC, map_location="cpu", weights_only=True)
    vae.load_state_dict(state["model"], strict=True); del state; gc.collect()
    vae.cuda().eval().requires_grad_(False)
    frozen = module_hash(vae)
    neighbors, distances = nearest_distinct(vae.vq.embeddings[0].weight, 8)
    rng = torch.Generator().manual_seed(3042)
    counts = {mode: {"nonpad_correct": 0, "nonpad_total": 0, "exact_texts": 0} for mode in MODES}
    outputs, coded = {mode: [] for mode in MODES}, {mode: [] for mode in MODES}
    for start in range(0, len(cases), 32):
        chunk = cases[start:start+32]
        targets = torch.tensor([r["native_target_ids"] for r in chunk], device="cuda")
        z = vae.encode_z(targets)
        clean = vae.vq.z_to_idx(z)
        noise = sample_corruption(clean, vae.vq.levels, vae.vq.vqvae_vocab_sizes, neighbors, rng, .5, 0.)
        root_mask = noise.mask.clone(); root_mask[:, 4:] = False
        thin = (torch.rand(clean.shape, generator=rng) < (.05/.4375)).cuda()
        thin[:, :4] = True
        masks = {"clean": torch.zeros_like(noise.mask), "root_only": root_mask,
                 "all_levels": noise.mask, "mild_fine": noise.mask & thin}
        for mode, mask in masks.items():
            expert, history, _ = training_targets(vae.vq, z, clean, Corruption(mask, noise.replacements), repair=True)
            if mode == "clean" and (not torch.equal(expert, clean) or not torch.equal(history, clean)):
                raise ValueError("clean vendor-code parity failed")
            if bool(mask[:, -48:].any()):
                raise ValueError("last-level noise should be zero")
            with torch.autocast("cuda", dtype=torch.float16):
                logits = vae.decoder(cumulative_latent(vae, history))
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite oracle logits")
            raw = logits.argmax(-1)
            nonpad = targets.ne(50257)
            counts[mode]["nonpad_correct"] += int((raw.eq(targets) & nonpad).sum())
            counts[mode]["nonpad_total"] += int(nonpad.sum())
            coded[mode].append(history.cpu())
            for case, ids in zip(chunk, raw.tolist()):
                text, termination = decode_ids(ids, gpt)
                counts[mode]["exact_texts"] += text == case["target"]
                outputs[mode].append({"id": case["id"], "target": case["target"], "text": text,
                                      "native_output_ids": ids, "termination": termination})
    if module_hash(vae) != frozen:
        raise ValueError("frozen codec changed")
    metrics = {mode: {**value, "nonpad_reconstruction": value["nonpad_correct"]/value["nonpad_total"],
                      "capacity_qualified": value["nonpad_correct"]/value["nonpad_total"] >= .95}
               for mode, value in counts.items()}
    for mode in MODES:
        write_json(root / (mode+".json"), outputs[mode])
    tensors = {mode: torch.cat(value) for mode, value in coded.items()}
    save(root / "codes.pt", tensors)
    reloaded = torch.load(root / "codes.pt", weights_only=True)
    if any(not torch.equal(tensors[mode], reloaded[mode]) for mode in MODES):
        raise ValueError("discrete history reload changed")
    verify(p)
    result = {"status": "passed" if pilot else "completed_noise_capacity_measurement", "cases": len(cases),
        "metrics": metrics, "codec_frozen_hash": frozen, "vendor_parity": True, "reload_equal": True,
        "minimum_neighbor_squared_distance": float(distances.min()), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "producer_fingerprint": p["fingerprint"], "target_aware": True, "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False, "underpowered": True, "confirmed": False}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "measure"])
    main(parser.parse_args().command == "pilot")
