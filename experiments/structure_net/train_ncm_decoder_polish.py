"""Improve full-cutoff text reconstruction while preserving the NCM code language."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_llama import autocast, encode_codes, load_ncm, tokenizers
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_context import unpaired_windows
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import ROOT, TEACHER, verify
from experiments.structure_net.ncm_paper_train_generic import lr_factor
from experiments.structure_net.ncm_training_recipe import cumulative_latent
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.train_ncm_memory import save
from experiments.structure_net.train_ncm_paper_codec_v2 import evaluate

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-decoder-polish-plan.md")
CODEC = ROOT / "20261022_codec_training_v2/model.pt"


@dataclass(frozen=True)
class Config:
    steps: int = 1024
    batch: int = 32
    lr: float = 1e-4
    weight_decay: float = .01
    warmup: int = 64
    seed: int = 42
    dropout_seed: int = 52
    sampler_seed: int = 2042


def frozen_hash(vae):
    class FrozenView(torch.nn.Module):
        def state_dict(self):
            return {key: value for key, value in vae.state_dict().items() if not key.startswith("decoder.")}
    return module_hash(FrozenView())


def main(args):
    cfg = Config(); pilot = args.command == "pilot"
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise ValueError("physical GPU0 required")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or "3060" not in torch.cuda.get_device_name():
        raise ValueError("planned GPU profile required")
    root = ROOT / ("20261022_decoder_pilot_v1" if pilot else "20261022_decoder_train_v1")
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "fixed_assignment_decoder_reconstruction",
        [CODEC, TEACHER / "train.json", ROOT / "20261022_codec_training_v2/result.json"], asdict(cfg))
    paths = [Path(__file__), PLAN, Path("experiments/structure_net/ncm_paper_context.py"),
        Path("experiments/structure_net/ncm_paper_train_generic.py"),
        Path("experiments/structure_net/train_ncm_paper_codec_v2.py")]
    for path in paths:
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("decoder_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-decoder-polish-v1"
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    if not pilot:
        pr = ROOT / "20261022_decoder_pilot_v1"
        if q.read(pr / "result.json")["status"] != "passed" or q.read(pr / "producer.json")["source_hashes"] != p["source_hashes"]:
            raise ValueError("matching decoder lifecycle pilot required")
    cases = q.read(TEACHER / "train.json")[:32] if pilot else q.read(TEACHER / "train.json")
    torch.manual_seed(cfg.seed); torch.cuda.reset_peak_memory_stats()
    gpt, _ = tokenizers(); gpt = CachedVocabulary(gpt)
    vae, net, _ = load_ncm(codec=CODEC)
    del net; gc.collect(); torch.cuda.empty_cache()
    original_frozen, original_decoder = frozen_hash(vae), module_hash(vae.decoder)
    targets = torch.tensor([r["native_target_ids"] for r in cases])
    windows = unpaired_windows(cases, gpt)
    def encode_bank(bank):
        return torch.cat([encode_codes(vae, bank[i:i+32].cuda()).cpu() for i in range(0, len(bank), 32)])
    target_codes, window_codes = encode_bank(targets), encode_bank(windows)
    baseline, baseline_outputs = evaluate(vae, cases, gpt)
    write_json(root / "baseline.json", baseline); write_json(root / "baseline_outputs.json", baseline_outputs)
    vae.requires_grad_(False).eval(); vae.decoder.requires_grad_(True).train()
    parameters = list(vae.decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, betas=(.9, .95), weight_decay=cfg.weight_decay)
    sampler = torch.Generator().manual_seed(cfg.sampler_seed)
    torch.manual_seed(cfg.dropout_seed); torch.cuda.manual_seed_all(cfg.dropout_seed)
    steps = 2 if pilot else cfg.steps
    trace = []; max_gradient = 0.; start = time.perf_counter()
    for step in range(1, steps+1):
        selected = [torch.randint(len(bank), (cfg.batch//2,), generator=sampler) for bank in (targets, windows)]
        ids = torch.cat([bank[idx] for bank, idx in zip((targets, windows), selected)]).cuda()
        codes = torch.cat([bank[idx] for bank, idx in zip((target_codes, window_codes), selected)]).cuda()
        latent = cumulative_latent(vae, codes)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr*lr_factor(step, steps, cfg.warmup)
        with autocast():
            logits = vae.decoder(latent)
            loss = F.cross_entropy(logits.float().flatten(0, 1), ids.flatten(), ignore_index=50257)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite decoder objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        max_gradient = max(max_gradient, float(norm)); optimizer.step()
        trace.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm)})
        if step % 64 == 0 or step == steps:
            write_json(root / "trace.json", trace)
            print(json.dumps(trace[-1]), flush=True)
    if (max_gradient <= 0 or frozen_hash(vae) != original_frozen or
        module_hash(vae.decoder) == original_decoder or
        any(x.grad is not None for name, x in vae.named_parameters() if not name.startswith("decoder."))):
        raise ValueError("decoder-only training contract failed")
    vae.requires_grad_(False).eval()
    codes_after = encode_bank(targets)
    if not torch.equal(target_codes, codes_after):
        raise ValueError("code language changed")
    save(root / "model.pt", {"model": vae.state_dict(), "optimizer": optimizer.state_dict(),
        "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "step": steps, "config": asdict(cfg), "producer_fingerprint": p["fingerprint"],
        "original_codec_sha256": file_hash(CODEC), "frozen_nondecode_hash": original_frozen})
    before = module_hash(vae)
    restored = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    vae.load_state_dict(restored["model"], strict=True); torch.clear_autocast_cache()
    if module_hash(vae) != before:
        raise ValueError("decoder reload mismatch")
    metrics, outputs = evaluate(vae, cases, gpt)
    write_json(root / "outputs.json", outputs)
    verify(p)
    gate = metrics["nonpad_reconstruction"] >= .995 and metrics["exact_text_cases"]/len(cases) >= .9
    result = {"status": "passed" if pilot else "reconstruction_gate_passed" if gate else "reconstruction_gate_failed",
        "hypothesis_id": p["hypothesis_id"], "producer_fingerprint": p["fingerprint"], "baseline": baseline,
        **metrics, "cases": len(cases), "model_sha256": file_hash(root / "model.pt"),
        "nondecoder_hash": original_frozen, "nondecoder_unchanged": True, "all_summary_codes_unchanged": True,
        "reload_equal": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter()-start, "task_accuracy": None, "speed_tested": False,
        "final_test_opened": False, "confirmed": False, "underpowered": True}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "train"])
    main(parser.parse_args())
