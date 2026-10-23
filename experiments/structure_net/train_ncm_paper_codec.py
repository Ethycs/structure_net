"""Genuine nested-dropout tokenizer adaptation before conditional NCM training."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import time

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_codec_training import codec_loss, normalize_ema, quantize_cutoff
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_llama import INITIAL, autocast, decode_ids, encode_codes, load_ncm, tokenizers
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_context import unpaired_windows
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import ROOT, TEACHER, verify
from experiments.structure_net.ncm_paper_train import lr_factor
from experiments.structure_net.ncm_training_recipe import cumulative_latent, sample_cutoff
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.train_ncm_memory import save

NOTE = Path("docs/07 - Status Reports/2026-10-22_ncm-paper-recovery-v2-plan.md")


@dataclass(frozen=True)
class Config:
    steps: int = 1024
    batch: int = 32
    lr: float = 5e-5
    weight_decay: float = .01
    warmup: int = 64
    ema_reference_batch: int = 4096
    ema_reference_decay: float = .95
    nested_dropout: float = .5


@torch.inference_mode()
def evaluate(vae, cases, gpt):
    vae.eval()
    correct = total = content_correct = content_total = exact = 0
    outputs = []
    for i in range(0, len(cases), 32):
        chunk = cases[i:i+32]
        targets = torch.tensor([r["native_target_ids"] for r in chunk], device="cuda")
        codes = encode_codes(vae, targets)
        with autocast():
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
    return {"nonpad_correct": correct, "nonpad_total": total, "nonpad_reconstruction": correct/total,
        "content_reconstruction": content_correct/content_total, "exact_text_cases": exact}, outputs


def run(args):
    cfg = Config()
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    pilot = args.command == "pilot"
    cases = q.read(TEACHER / "train.json")
    if pilot:
        cases = cases[:32]
    p = producer(args.output, "systems_codec_training_only_not_quality_evidence" if pilot else "full_tokenizer_adaptation",
        [TEACHER / "train.json", INITIAL / "tokenizer.pt", INITIAL / "generator.pt", INITIAL / "config.yaml"],
        {**asdict(cfg), "actual_steps": 2 if pilot else cfg.steps, "seed": 42,
         "dropout_seed": 52, "sampler_seed": 2042, "augmentation_seed": 3042,
         "pilot_cutoffs": [1, 9] if pilot else None})
    p["hypothesis_id"] = "ncm-paper-regime-recovery-v2"
    for source in (Path(__file__), NOTE, Path("experiments/structure_net/ncm_codec_training.py")):
        p["source_hashes"][str(source)] = file_hash(source)
        (args.output / "producer_source" / ("codec_training_"+source.name)).write_bytes(source.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(args.output / "producer.json", p)
    if not pilot:
        pp, pr = q.read(args.pilot / "producer.json"), q.read(args.pilot / "result.json")
        if pr["status"] != "passed" or pp["source_hashes"] != p["source_hashes"]:
            raise ValueError("matching real codec-training lifecycle pilot required")
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    gpt, _ = tokenizers()
    gpt = CachedVocabulary(gpt)
    vae, net, _ = load_ncm()
    del net
    gc.collect()
    torch.cuda.empty_cache()
    initial = module_hash(vae)
    initial_books = module_hash(vae.vq.embeddings)
    control_ids = torch.tensor([r["native_target_ids"] for r in cases[:2]], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        control_codes = encode_codes(vae, control_ids)
        reference = cumulative_latent(vae, control_codes)
        actual, _, _ = quantize_cutoff(vae, vae._encode_pre_quant(control_ids), len(vae.vq.levels))
        if not torch.allclose(reference, actual, atol=1e-5, rtol=1e-5):
            raise ValueError("full-cutoff quantizer differs from frozen-code reconstruction")
    vae.requires_grad_(True).train()
    for embedding in vae.vq.embeddings:
        embedding.requires_grad_(False)
    normalize_ema(vae.vq, cfg.batch)
    if module_hash(vae.vq.embeddings) != initial_books:
        raise ValueError("EMA normalization moved book vectors")
    summaries = torch.tensor([r["native_target_ids"] for r in cases])
    windows = unpaired_windows(cases, gpt)
    parameters = [p for p in vae.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, betas=(.9, .95), weight_decay=cfg.weight_decay)
    sampler = torch.Generator().manual_seed(2042)
    aug = torch.Generator().manual_seed(3042)
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    decay = cfg.ema_reference_decay**(cfg.batch/cfg.ema_reference_batch)
    trace = []
    started = time.perf_counter()
    max_encoder = max_projection = 0.
    steps = 2 if pilot else cfg.steps
    for step in range(1, steps+1):
        ids = torch.cat([bank[torch.randint(len(bank), (cfg.batch//2,), generator=sampler)]
                         for bank in (summaries, windows)]).cuda()
        cutoff = (1 if step == 1 else len(vae.vq.levels)) if pilot else sample_cutoff(len(vae.vq.levels), cfg.nested_dropout, aug)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr*lr_factor(step, steps, cfg.warmup)
        with autocast():
            loss, details = codec_loss(vae, ids, cutoff, decay=decay)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite tokenizer objective")
        loss.backward()
        max_encoder = max(max_encoder, float(vae.encoder.wte.weight.grad.float().norm()))
        grad = sum(float(p.grad.float().square().sum()) for p in vae.vq.parameters() if p.grad is not None)**.5
        max_projection = max(max_projection, grad)
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        trace.append({"step": step, "loss": float(loss.detach()), **details})
        if step % 32 == 0 or step == steps:
            write_json(args.output / "trace.json", trace)
            print(json.dumps({"stage": "tokenizer", **trace[-1], "total": steps}), flush=True)
    if max_encoder <= 0 or max_projection <= 0 or module_hash(vae.vq.embeddings) == initial_books:
        raise ValueError("encoder, resampling or EMA did not update")
    vae.eval().requires_grad_(False)
    checkpoint = {"model": vae.state_dict(), "optimizer": optimizer.state_dict(),
        "sampler_rng": sampler.get_state(), "augmentation_rng": aug.get_state(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "step": steps, "config": asdict(cfg), "producer_fingerprint": p["fingerprint"],
        "levels": list(vae.vq.levels), "vocabularies": list(vae.vq.vqvae_vocab_sizes), "ema_decay": decay}
    save(args.output / "model.pt", checkpoint)
    before = module_hash(vae)
    loaded = torch.load(args.output / "model.pt", weights_only=True, map_location="cpu")
    vae.load_state_dict(loaded["model"], strict=True)
    if module_hash(vae) != before:
        raise ValueError("tokenizer reload mismatch")
    metrics, outputs = evaluate(vae, cases[:2] if pilot else cases, gpt)
    write_json(args.output / "outputs.json", outputs)
    verify(p)
    result = {"status": "passed", "stage": p["stage"], "producer_fingerprint": p["fingerprint"],
        "steps": steps, "cases": len(cases), **metrics, "reload_equal": True,
        "max_encoder_gradient": max_encoder, "max_projection_gradient": max_projection,
        "ema_books_changed": True, "initial_codec_sha256": initial,
        "codec_module_sha256": before, "model_sha256": file_hash(args.output / "model.pt"),
        "cutoffs_seen": sorted({t["cutoff"] for t in trace}),
        "elapsed_seconds": time.perf_counter()-started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "task_accuracy": None, "semantic_oracle_control_pending": True, "speed_tested": False}
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "train"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    run(parser.parse_args())
