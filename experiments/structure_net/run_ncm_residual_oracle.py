"""Measure discrete finer-code repair capacity; never a summarizer accuracy test."""
import argparse
import gc
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_llama import INITIAL, decode_ids, tokenizers, vendor_classes
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.ncm_residual_oracle import nearest_distinct, requantize
from experiments.structure_net.ncm_training_recipe import cumulative_latent
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.train_ncm_memory import save

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-residual-oracle-plan.md")
CODEC = ROOT / "20261022_decoder_train_v1/model.pt"
DATA = ROOT / "20261022_teacher_v1/train.json"
ARMS = ("clean", "one_replay", "one_repair", "all_replay", "all_repair")


@torch.inference_mode()
def main(pilot):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("physicalGPU2 must be selected before torch startup")
    if not torch.cuda.is_available() or "2060 SUPER" not in torch.cuda.get_device_name():
        raise ValueError("planned GPU required")
    cases = q.read(DATA)[:32] if pilot else q.read(DATA)
    verify_public(cases)
    root = ROOT / ("20261022_residual_oracle_pilot_v1" if pilot else "20261022_residual_oracle_v1")
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "target_latent_residual_repair_oracle",
        [CODEC, DATA, INITIAL / "config.yaml"], {"batch": 32, "neighbors": 8, "seed": 42,
            "augmentation_seed": 3042, "quantization": "FP32", "neighbors_precision": "FP64",
            "decoder": "FP16", "physical_gpu": 2, "arms": list(ARMS)})
    p["hypothesis_id"] = "ncm-residual-oracle-v1"
    for path in (Path(__file__), PLAN, Path("experiments/structure_net/ncm_residual_oracle.py")):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("residual_"+path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    if not pilot:
        pr = ROOT / "20261022_residual_oracle_pilot_v1"
        if q.read(pr / "result.json")["status"] != "passed" or q.read(pr / "producer.json")["source_hashes"] != p["source_hashes"]:
            raise ValueError("matching oracle lifecycle pilot required")
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
    gpt, _ = tokenizers(); gpt = CachedVocabulary(gpt)
    OmegaConf, VQVAE, _ = vendor_classes()
    cfg = OmegaConf.load(INITIAL / "config.yaml")
    vae = VQVAE(**cfg.tokenizer.vqvae)
    state = torch.load(CODEC, map_location="cpu", weights_only=True)
    vae.load_state_dict(state["model"], strict=True)
    del state; gc.collect()
    vae.cuda().eval().requires_grad_(False)
    frozen = module_hash(vae)
    neighbors, distances = nearest_distinct(vae.vq.embeddings[0].weight, 8)
    neighbors_cpu = neighbors.cpu()
    sampler = torch.Generator().manual_seed(3042)
    outputs = {arm: [] for arm in ARMS}
    code_banks = {arm: [] for arm in ARMS}
    counts = {arm: {"nonpad_correct": 0, "nonpad_total": 0, "exact_text_cases": 0} for arm in ARMS}
    for start in range(0, len(cases), 32):
        chunk = cases[start:start+32]
        ids = torch.tensor([row["native_target_ids"] for row in chunk], device="cuda")
        with torch.autocast("cuda", enabled=False):
            z = vae.encode_z(ids)
            reference = vae.vq.z_to_idx(z)
            clean, latent = requantize(vae.vq, z)
            if not torch.equal(reference, clean) or not torch.allclose(latent, cumulative_latent(vae, clean), atol=1e-5, rtol=1e-5):
                raise ValueError("clean code/latent parity with vendor failed")
            root_width = vae.vq.levels[0]
            root_codes = clean[:, :root_width].cpu()
            offsets = torch.randint(8, root_codes.shape, generator=sampler)
            all_root = neighbors_cpu[root_codes, offsets]
            position = torch.randint(root_width, (len(chunk),), generator=sampler)
            one_root = root_codes.clone()
            one_root[torch.arange(len(chunk)), position] = all_root[torch.arange(len(chunk)), position]
            one_root, all_root = one_root.cuda(), all_root.cuda()
            replay_one, replay_all = clean.clone(), clean.clone()
            replay_one[:, :root_width] = one_root; replay_all[:, :root_width] = all_root
            repair_one, latent_one = requantize(vae.vq, z, one_root)
            repair_all, latent_all = requantize(vae.vq, z, all_root)
            for forced, repaired, actual_latent in ((one_root, repair_one, latent_one), (all_root, repair_all, latent_all)):
                if not torch.equal(forced, repaired[:, :root_width]) or not torch.allclose(actual_latent, cumulative_latent(vae, repaired), atol=1e-5, rtol=1e-5):
                    raise ValueError("forced root or discrete reconstructed latent changed")
        coded = dict(zip(ARMS, (clean, replay_one, repair_one, replay_all, repair_all)))
        nonpad = ids.ne(50257)
        for arm, codes in coded.items():
            latent = cumulative_latent(vae, codes)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = vae.decoder(latent)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite oracle decode logits")
            generated = logits.argmax(-1)
            counts[arm]["nonpad_correct"] += int((generated.eq(ids) & nonpad).sum())
            counts[arm]["nonpad_total"] += int(nonpad.sum())
            code_banks[arm].append(codes.cpu())
            for case, raw in zip(chunk, generated.tolist()):
                text, termination = decode_ids(raw, gpt)
                counts[arm]["exact_text_cases"] += text == case["target"]
                outputs[arm].append({"id": case["id"], "target": case["target"], "text": text,
                    "native_output_ids": raw, "termination": termination})
        if start % 256 == 0:
            print(json.dumps({"cases": start+len(chunk), "total": len(cases)}), flush=True)
    if module_hash(vae) != frozen or any(value.grad is not None for value in vae.parameters()):
        raise ValueError("frozen codec changed")
    save(root / "codes.pt", {arm: torch.cat(values) for arm, values in code_banks.items()})
    save(root / "frozen_codec.pt", {"model": vae.state_dict(), "sampler_rng": sampler.get_state(),
        "producer_fingerprint": p["fingerprint"], "codec_sha256": file_hash(CODEC)})
    loaded = torch.load(root / "frozen_codec.pt", map_location="cpu", weights_only=True)
    vae.load_state_dict(loaded["model"], strict=True); torch.clear_autocast_cache()
    if module_hash(vae) != frozen:
        raise ValueError("frozen codec reload changed state")
    metrics = {}
    for arm, values in counts.items():
        write_json(root / (arm+".json"), outputs[arm])
        metrics[arm] = {**values, "nonpad_reconstruction": values["nonpad_correct"]/values["nonpad_total"]}
    recovered = {}
    for prefix in ("one", "all"):
        lost = counts["clean"]["nonpad_correct"]-counts[prefix+"_replay"]["nonpad_correct"]
        restored = counts[prefix+"_repair"]["nonpad_correct"]-counts[prefix+"_replay"]["nonpad_correct"]
        recovered[prefix] = {"lost_relative_to_clean": lost, "restored_by_requantizing": restored,
            "fraction_lost_restored": restored/lost if lost > 0 else None}
    verify(p)
    result = {"status": "passed" if pilot else "completed_representation_oracle", "cases": len(cases),
        "metrics": metrics, "paired_recovery": recovered, "vendor_code_and_latent_parity": True,
        "forced_roots_preserved": True, "frozen_codec_hash": frozen, "reload_equal": True,
        "minimum_neighbor_squared_distance": float(distances.min()), "neighbors_exclude_identical_vectors": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "producer_fingerprint": p["fingerprint"],
        "task_accuracy": None, "oracle_knows_target_text": True, "trained": False,
        "final_test_opened": False, "speed_tested": False, "underpowered": True, "confirmed": False}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "measure"])
    main(parser.parse_args().command == "pilot")
