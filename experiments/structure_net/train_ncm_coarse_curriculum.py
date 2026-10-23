"""Fixed 32-example competence test before further NCM held-out adaptation."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import time

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_coarse_curriculum import (
    GlobalResidual, enable_coarse_ar, generate_codes, logits_for_codes,
    loss_for_batch, source_context)
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_llama import (
    autocast, decode_ids, encode_codes, load_ncm, tokenizers)
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import PARENT, PREP, ROOT, verify
from experiments.structure_net.ncm_paper_train_generic import lr_factor
from experiments.structure_net.ncm_training_recipe import Recipe, cumulative_latent
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.train_ncm_memory import save

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-coarse-curriculum-plan.md")
CODEC = ROOT / "20261022_codec_training_v2/model.pt"
GENERATOR = ROOT / "20261022_noisy_native_v1/model.pt"
TEACHER = ROOT / "20261022_teacher_v1/train.json"
RECIPE = Recipe(0., .5, .1, 1.)


@dataclass(frozen=True)
class Config:
    cases: int = 32
    steps: int = 1024
    coarse_steps: int = 256
    batch: int = 4
    lr: float = 3e-4
    dropout: float = .1
    weight_decay: float = .01
    warmup: int = 32
    seed: int = 42
    dropout_seed: int = 52
    sampler_seed: int = 2042
    augmentation_seed: int = 3042


@torch.inference_mode()
def measure(vae, net, adapter, prefix, last, targets, labels, cases, gpt, shifted=False):
    net.eval(); adapter.eval()
    if shifted:
        prefix, last = prefix.roll(1, 0), last.roll(1, 0)
    predictions, tokens, outputs = [], [], []
    for start in range(0, len(cases), 4):
        sl = slice(start, start+4)
        with autocast():
            predicted = generate_codes(vae, net, adapter, prefix[sl].cuda(), last[sl].cuda())
            decoded = vae.decoder(cumulative_latent(vae, predicted)).argmax(-1)
        predictions.append(predicted.cpu()); tokens.append(decoded.cpu())
        for case, codes, ids in zip(cases[sl], predicted.tolist(), decoded.tolist()):
            text, termination = decode_ids(ids, gpt)
            outputs.append({**case, "text": text, "termination": termination,
                            "predicted_codes": codes, "native_output_ids": ids})
    pred, decoded = torch.cat(predictions), torch.cat(tokens)
    nonpad = targets.ne(50257)
    coarse_correct = int(pred[:, :4].eq(labels[:, :4]).sum())
    code_correct = int(pred.eq(labels).sum())
    token_correct = int((decoded.eq(targets) & nonpad).sum())
    metrics = {"cases": len(cases), "code_correct": code_correct, "code_total": labels.numel(),
        "code_accuracy": code_correct/labels.numel(), "coarse_correct": coarse_correct,
        "coarse_total": len(cases)*4, "coarse_accuracy": coarse_correct/(len(cases)*4),
        "nonpad_correct": token_correct, "nonpad_total": int(nonpad.sum()),
        "nonpad_reconstruction": token_correct/int(nonpad.sum()),
        "exact_target_texts": sum(r["text"] == r["target"] for r in outputs),
        "task_accuracy": None, "interpretation": "training memorization diagnostics only"}
    return metrics, outputs


def main(args):
    cfg = Config()
    pilot = args.command == "pilot"
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("select physical GPU0 before torch startup")
    if not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name():
        raise ValueError("planned CUDA device required")
    cases = q.read(TEACHER)[:cfg.cases]
    if pilot:
        cases = cases[:4]
    input_paths = [TEACHER, CODEC, GENERATOR, ROOT / "20261022_noisy_native_v1/result.json",
                   ROOT / "20261022_codec_training_v2/result.json", PREP / "producer.json"]
    manifest = q.read(PREP / "producer.json")["input_hashes"]
    features = []
    for case in cases:
        path = PARENT / "train_features" / (case["id"]+".pt")
        if file_hash(path) != manifest[str(path)]:
            raise ValueError("training feature identity mismatch")
        input_paths.append(path)
        value = torch.load(path, weights_only=True)
        # Original features are left packed: the final position is valid.
        if bool(value["pad_mask"][-1]):
            raise ValueError("last body position unexpectedly padded")
        features.append(value["hidden"][-1].clone())
    root = ROOT / ("20261022_coarse_pilot_v1" if pilot else "20261022_coarse_train_v1")
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "training_memorization_diagnostic",
        input_paths, {**asdict(cfg), "recipe": RECIPE.record(), "pilot": pilot,
                      "coarse_autoregressive": True, "global_residual_zero_init": True})
    for path in [Path(__file__), Path("experiments/structure_net/ncm_coarse_curriculum.py"),
                 Path("experiments/structure_net/ncm_paper_train_generic.py"), PLAN]:
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("coarse_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-coarse-curriculum-v1"
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    if not pilot:
        pp = q.read(ROOT / "20261022_coarse_pilot_v1/producer.json")
        pr = q.read(ROOT / "20261022_coarse_pilot_v1/result.json")
        if pr["status"] != "passed" or pp["source_hashes"] != p["source_hashes"]:
            raise ValueError("matching real lifecycle pilot required")
    torch.manual_seed(cfg.seed)
    torch.cuda.reset_peak_memory_stats()
    gpt, _ = tokenizers(); gpt = CachedVocabulary(gpt)
    vae, net, _ = load_ncm(codec=CODEC, generator=GENERATOR, adaptation=True)
    frozen = module_hash(vae)
    generator_state = torch.load(GENERATOR, map_location="cpu", weights_only=True)
    if generator_state["codec_sha256"] != file_hash(CODEC):
        raise ValueError("continued generator/codec mismatch")
    del generator_state; gc.collect()
    enable_coarse_ar(net)
    adapter = GlobalResidual().cuda()
    last = torch.stack(features)
    ids = [gpt.encode(case["source"], add_special_tokens=False) for case in cases]
    if any(len(x) > 512 for x in ids):
        raise ValueError("source truncation prohibited")
    prefix = torch.tensor([x+[50257]*(512-len(x)) for x in ids])
    targets = torch.tensor([case["native_target_ids"] for case in cases])
    labels = encode_codes(vae, targets.cuda()).cpu()
    # Check context parity and absence of future-label leakage in eval mode.
    with torch.no_grad(), autocast():
        px, h, clean = prefix[:2].cuda(), last[:2].cuda(), labels[:2].cuda()
        native = net.transformer.wte(vae.encoder.wte(px))
        context = source_context(vae, net, adapter, px, h)
        zero_equal = torch.equal(native, context)
        a = logits_for_codes(vae, net, context, px, clean, clean)
        changed = clean.clone(); changed[:, 0] = (changed[:, 0]+1) % vae.vq.vqvae_vocab_sizes[0]
        b = logits_for_codes(vae, net, context, px, changed, changed)
        first_invariant = torch.equal(a[:, :1], b[:, :1])
        past_used = not torch.equal(a[:, 1:2], b[:, 1:2])
    if not zero_equal or not first_invariant or not past_used:
        raise ValueError("zero-residual parity or causal input check failed")
    del a, b, native, context, clean, changed, px, h
    parameters = list(net.parameters())+list(adapter.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, betas=(.9, .95), weight_decay=cfg.weight_decay)
    sampler = torch.Generator().manual_seed(cfg.sampler_seed)
    aug = torch.Generator().manual_seed(cfg.augmentation_seed)
    torch.manual_seed(cfg.dropout_seed); torch.cuda.manual_seed_all(cfg.dropout_seed)
    steps = 2 if pilot else cfg.steps
    trace, samples = [], []
    max_query_grad = max_bridge_grad = 0.
    start = time.perf_counter()
    for step in range(1, steps+1):
        net.train(); adapter.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr*lr_factor(step, steps, cfg.warmup)
        idx = torch.randint(len(cases), (cfg.batch,), generator=sampler)
        samples.extend(idx.tolist())
        coarse_only = step == 1 if pilot else step <= cfg.coarse_steps
        with autocast():
            loss, details = loss_for_batch(vae, net, adapter, prefix[idx].cuda(), last[idx].cuda(),
                labels[idx].cuda(), RECIPE, aug, coarse_only)
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite objective")
        loss.backward()
        max_query_grad = max(max_query_grad, float(net.l0_queries.grad.float().norm()))
        max_bridge_grad = max(max_bridge_grad, float(adapter.proj.weight.grad.float().norm()))
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        trace.append({"step": step, "coarse_only": coarse_only, "loss": float(loss.detach()),
                      "gradient_norm": float(norm), **details})
        if step % 64 == 0 or step == steps:
            write_json(root / "trace.json", trace)
            print(json.dumps({"step": step, "total": steps, "loss": trace[-1]["loss"]}), flush=True)
        if step % 256 == 0 or step == steps:
            checkpoint = {"model": net.state_dict(), "adapter": adapter.state_dict(),
                "optimizer": optimizer.state_dict(), "sampler_rng": sampler.get_state(),
                "augmentation_rng": aug.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "step": step, "config": asdict(cfg),
                "coarse_autoregressive": True, "codec_checkpoint": str(CODEC),
                "codec_sha256": file_hash(CODEC), "producer_fingerprint": p["fingerprint"]}
            save(root / f"step{step}.pt", checkpoint)
            metrics, outputs = measure(vae, net, adapter, prefix, last, targets, labels, cases, gpt)
            write_json(root / f"metrics_step{step}.json", metrics)
            write_json(root / f"outputs_step{step}.json", outputs)
            print(json.dumps(metrics), flush=True)
            del checkpoint
    if max_query_grad <= 0 or max_bridge_grad <= 0:
        raise ValueError("missing real query/bridge gradients")
    if module_hash(vae) != frozen or any(x.grad is not None for x in vae.parameters()):
        raise ValueError("frozen codec changed")
    net.eval(); adapter.eval()
    before = measure(vae, net, adapter, prefix[:2], last[:2], targets[:2], labels[:2], cases[:2], gpt)
    restored = torch.load(root / f"step{steps}.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(restored["model"], strict=True)
    adapter.load_state_dict(restored["adapter"], strict=True)
    enable_coarse_ar(net)
    torch.clear_autocast_cache()
    after = measure(vae, net, adapter, prefix[:2], last[:2], targets[:2], labels[:2], cases[:2], gpt)
    if before != after:
        raise ValueError("reload changed free-running generation")
    shifted, shifted_outputs = measure(vae, net, adapter, prefix, last, targets, labels, cases, gpt, shifted=True)
    write_json(root / "shifted_source_outputs.json", shifted_outputs)
    source_gap = metrics["coarse_accuracy"]-shifted["coarse_accuracy"]
    gate = metrics["code_accuracy"] >= .9 and metrics["nonpad_reconstruction"] >= .95 and source_gap >= .2
    verify(p)
    result = {"status": "passed" if pilot else "mechanism_gate_passed" if gate else "mechanism_gate_failed",
        "hypothesis_id": p["hypothesis_id"], "producer_fingerprint": p["fingerprint"],
        "training_diagnostics": metrics, "shifted_source_diagnostics": shifted,
        "coarse_matched_source_advantage": source_gap, "zero_residual_native_equal": zero_equal,
        "coarse_first_label_invariant": first_invariant, "coarse_past_labels_used": past_used,
        "max_query_gradient": max_query_grad, "max_bridge_gradient": max_bridge_grad,
        "reload_equal": True, "codec_frozen_hash": frozen, "parent_body_frozen": True,
        "parent_body_loaded": False, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter()-start, "sample_schedule_hash": q.digest(samples),
        "task_accuracy": None, "final_test_opened": False, "speed_tested": False,
        "underpowered": True, "confirmed": False}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "train"])
    main(parser.parse_args())
