"""Train a conditional-residual expert objective with a matched static control."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import time

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_coarse_curriculum import GlobalResidual, enable_coarse_ar, generate_codes, logits_for_codes, source_context
from experiments.structure_net.ncm_conditional_residual_targets import Corruption, sample_corruption, training_targets
from experiments.structure_net.ncm_frozen_body import module_hash, supplied_context
from experiments.structure_net.ncm_global_context import last_valid
from experiments.structure_net.ncm_llama import decode_ids, encode_codes, load_ncm, tokenizers
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import PARENT, PREP, ROOT, verify
from experiments.structure_net.ncm_paper_train_generic import lr_factor
from experiments.structure_net.ncm_residual_oracle import nearest_distinct
from experiments.structure_net.ncm_training_recipe import cumulative_latent, hierarchical_loss
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.train_ncm_memory import save

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-conditional-residual-learning-plan.md")
CHECKPOINT = ROOT / "20261022_full_train_v1/step5120.pt"
CODEC = ROOT / "20261022_decoder_train_v1/model.pt"
DATA = ROOT / "20261022_teacher_v1/train.json"
CONDITIONS = ("free", "gold_root", "one_wrong", "all_wrong")


@dataclass(frozen=True)
class Config:
    cases: int = 32
    steps: int = 1024
    batch: int = 4
    lr: float = 1e-4
    warmup: int = 32
    weight_decay: float = .01
    dropout: float = .1
    seed: int = 42
    dropout_seed: int = 52
    sample_seed: int = 2042
    noise_seed: int = 3042
    evaluation_seed: int = 4042
    root_neighbors: int = 8
    first_corruption: float = .5
    final_corruption: float = 0.


@torch.inference_mode()
def evaluate(vae, net, adapter, prefix, last, targets, labels, roots, cases, gpt):
    net.eval(); adapter.eval()
    result, outputs = {}, {}
    for condition in CONDITIONS:
        predicted, tokens, rows = [], [], []
        for start in range(0, len(cases), 4):
            sl = slice(start, start+4)
            px, h = prefix[sl].cuda(), last[sl].cuda()
            if condition == "free":
                codes = generate_codes(vae, net, adapter, px, h)
            else:
                context = source_context(vae, net, adapter, px, h)
                with supplied_context(net, context):
                    codes = net.generate(roots[condition][sl].cuda(), px, temperature=0., top_k=1,
                                         top_p=0., cfg_scale=0., use_kv_cache=False)
                if not torch.equal(codes[:, :4].cpu(), roots[condition][sl]):
                    raise ValueError("forced diagnostic roots changed")
            logits = vae.decoder(cumulative_latent(vae, codes))
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite evaluation logits")
            raw = logits.argmax(-1)
            predicted.append(codes.cpu()); tokens.append(raw.cpu())
            for case, code, ids in zip(cases[sl], codes.tolist(), raw.tolist()):
                text, termination = decode_ids(ids, gpt)
                rows.append({"id": case["id"], "target": case["target"], "text": text,
                    "predicted_codes": code, "native_output_ids": ids, "termination": termination,
                    "target_aware": condition != "free"})
        pred, raw = torch.cat(predicted), torch.cat(tokens)
        valid = targets.ne(50257)
        correct, total = int((raw.eq(targets) & valid).sum()), int(valid.sum())
        result[condition] = {"cases": len(cases), "nonpad_correct": correct, "nonpad_total": total,
            "nonpad_reconstruction": correct/total, "code_correct": int(pred.eq(labels).sum()),
            "code_total": labels.numel(), "exact_texts": sum(r["text"] == r["target"] for r in rows),
            "target_aware": condition != "free", "task_accuracy": None}
        outputs[condition] = rows
    return result, outputs


def main(pilot):
    cfg = Config()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("physicalGPU2 must be selected before torch startup")
    if not torch.cuda.is_available() or "2060 SUPER" not in torch.cuda.get_device_name():
        raise ValueError("planned CUDA device required")
    if torch.cuda.mem_get_info()[0] < 4_800_000_000:
        raise ValueError("insufficient intended GPU2 headroom")
    cases = sorted(q.read(DATA), key=lambda r: text_hash("ncm-residual-learning-v1:"+r["id"]))[:4 if pilot else cfg.cases]
    inputs = [DATA, CODEC, CHECKPOINT, ROOT / "20261022_full_train_v1/result.json"]
    manifest = q.read(PREP / "producer.json")["input_hashes"]
    features = []
    for case in cases:
        path = PARENT / "train_features" / (case["id"]+".pt")
        if file_hash(path) != manifest[str(path)]:
            raise ValueError("training feature identity changed")
        inputs.append(path)
        state = torch.load(path, weights_only=True)
        features.append(last_valid(state["hidden"][None], state["pad_mask"][None])[0].clone())
    base = ROOT / ("20261022_residual_learning_pilot_v1" if pilot else "20261022_residual_learning_train_v1")
    if base.exists():
        raise ValueError("fresh append-only campaign required")
    sources = [Path(__file__), PLAN, *[Path("experiments/structure_net") / name for name in
        ("ncm_conditional_residual_targets.py", "ncm_coarse_curriculum.py", "ncm_global_context.py",
         "ncm_residual_oracle.py", "ncm_paper_train_generic.py")]]
    if not pilot:
        previous = ROOT / "20261022_residual_learning_pilot_v1"
        receipt = q.read(previous / "result.json")
        if receipt["status"] != "passed" or any(file_hash(p) != expected for p, expected in receipt["source_hashes"].items()):
            raise ValueError("matching two-arm lifecycle pilot required")
        inputs.append(previous / "result.json")
    campaign_producer = producer(base, "systems_lifecycle_only_not_quality_evidence" if pilot else "matched_residual_training_campaign",
        inputs, {**asdict(cfg), "pilot": pilot, "arms": ["static", "repair"], "precision": "FP32", "physical_gpu": 2})
    for path in sources:
        campaign_producer["source_hashes"][str(path)] = file_hash(path)
        (base / "producer_source" / ("residual_"+path.name)).write_bytes(path.read_bytes())
    campaign_producer["hypothesis_id"] = "ncm-conditional-residual-learning-v1"
    campaign_producer["fingerprint"] = q.digest({k: v for k, v in campaign_producer.items() if k != "fingerprint"})
    write_json(base / "producer.json", campaign_producer)
    torch.manual_seed(cfg.seed); torch.cuda.reset_peak_memory_stats()
    vae, net, _ = load_ncm(codec=CODEC, generator=CHECKPOINT, adaptation=True)
    enable_coarse_ar(net)
    adapter = GlobalResidual().cuda()
    initial = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if initial["codec_sha256"] != file_hash(CODEC):
        raise ValueError("initial codec/generator pair changed")
    adapter.load_state_dict(initial["adapter"], strict=True)
    initial_net = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    initial_adapter = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    del initial; gc.collect()
    frozen = module_hash(vae)
    gpt, _ = tokenizers(); gpt = CachedVocabulary(gpt)
    source_ids = [gpt.encode(c["source"], add_special_tokens=False) for c in cases]
    if any(len(x) > 512 for x in source_ids):
        raise ValueError("source truncation prohibited")
    prefix = torch.tensor([x+[50257]*(512-len(x)) for x in source_ids])
    last = torch.stack(features)
    targets = torch.tensor([c["native_target_ids"] for c in cases])
    with torch.inference_mode():
        z = vae.encode_z(targets.cuda()).detach().float().cpu()
        labels = encode_codes(vae, targets.cuda()).cpu()
    neighbors, distances = nearest_distinct(vae.vq.embeddings[0].weight, cfg.root_neighbors)
    neighbors = neighbors.cpu()
    evaluation_rng = torch.Generator().manual_seed(cfg.evaluation_seed)
    root = labels[:, :4]
    offsets = torch.randint(cfg.root_neighbors, root.shape, generator=evaluation_rng)
    all_wrong = neighbors[root, offsets]
    one_wrong = root.clone()
    position = torch.randint(4, (len(cases),), generator=evaluation_rng)
    one_wrong[torch.arange(len(cases)), position] = all_wrong[torch.arange(len(cases)), position]
    roots = {"gold_root": root, "one_wrong": one_wrong, "all_wrong": all_wrong}
    if not bool(all_wrong.ne(root).all()):
        raise ValueError("neighbor corruption contains unchanged roots")
    zero_noise = Corruption(torch.zeros_like(labels, dtype=torch.bool).cuda(), labels.cuda())
    for repair in (False, True):
        expert, conditioning, _ = training_targets(vae.vq, z.cuda(), labels.cuda(), zero_noise, repair=repair)
        if not torch.equal(expert.cpu(), labels) or not torch.equal(conditioning.cpu(), labels):
            raise ValueError("no-noise vendor code language parity failed")
    net.eval(); adapter.eval()
    with torch.no_grad():
        px, h, clean = prefix[:2].cuda(), last[:2].cuda(), labels[:2].cuda()
        context = source_context(vae, net, adapter, px, h)
        a = logits_for_codes(vae, net, context, px, clean, clean)
        changed = clean.clone(); changed[:, 0] = (changed[:, 0]+1) % 4096
        b = logits_for_codes(vae, net, context, px, changed, changed)
        if not torch.equal(a[:, :1], b[:, :1]) or torch.equal(a[:, 1:2], b[:, 1:2]):
            raise ValueError("causal coarse input behavior failed")
        # Changing a fine query's own/future labels must not change its logits.
        own = clean.clone(); own[:, 4:] = (own[:, 4:]+1) % 4096
        c = logits_for_codes(vae, net, context, px, clean, own)
        if not torch.equal(a[:, 4:10], c[:, 4:10]):
            raise ValueError("same/future fine-code label leakage")
    del a, b, c, px, h, clean, context, changed, own, expert, conditioning
    results, schedules = {}, {}
    for arm in ("static", "repair"):
        run = base / arm
        p = producer(run, "systems_lifecycle_only_not_quality_evidence" if pilot else "conditional_residual_training_mechanism",
            inputs, {**asdict(cfg), "arm": arm, "pilot": pilot, "precision": "FP32", "physical_gpu": 2})
        for path in sources:
            p["source_hashes"][str(path)] = file_hash(path)
            (run / "producer_source" / ("residual_"+path.name)).write_bytes(path.read_bytes())
        p["hypothesis_id"] = "ncm-conditional-residual-learning-v1"
        p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
        write_json(run / "producer.json", p)
        net.load_state_dict(initial_net, strict=True); enable_coarse_ar(net)
        adapter.load_state_dict(initial_adapter, strict=True)
        initial_hash = {"net": module_hash(net), "adapter": module_hash(adapter)}
        baseline, baseline_outputs = evaluate(vae, net, adapter, prefix, last, targets, labels, roots, cases, gpt)
        write_json(run / "baseline_metrics.json", baseline)
        write_json(run / "baseline_outputs.json", baseline_outputs)
        sampler = torch.Generator().manual_seed(cfg.sample_seed)
        noise_rng = torch.Generator().manual_seed(cfg.noise_seed)
        torch.manual_seed(cfg.dropout_seed); torch.cuda.manual_seed_all(cfg.dropout_seed)
        parameters = list(net.parameters())+list(adapter.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, betas=(.9, .95), weight_decay=cfg.weight_decay)
        steps = 2 if pilot else cfg.steps
        trace, sample_schedule, noise_schedule = [], [], []
        max_query_gradient = max_bridge_gradient = 0.
        start_time = time.perf_counter()
        for step in range(1, steps+1):
            net.train(); adapter.train(); optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr*lr_factor(step, steps, cfg.warmup)
            indices = torch.randint(len(cases), (cfg.batch,), generator=sampler)
            clean = labels[indices].cuda()
            corruption = sample_corruption(clean, vae.vq.levels, vae.vq.vqvae_vocab_sizes,
                neighbors, noise_rng, cfg.first_corruption, cfg.final_corruption)
            expert, conditioning, info = training_targets(vae.vq, z[indices].cuda(), clean, corruption, repair=arm == "repair")
            context = source_context(vae, net, adapter, prefix[indices].cuda(), last[indices].cuda())
            logits = logits_for_codes(vae, net, context, prefix[indices].cuda(), clean, conditioning)
            loss, levels = hierarchical_loss(logits, expert, vae.vq.levels, alpha=1.)
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite training objective")
            loss.backward()
            max_query_gradient = max(max_query_gradient, float(net.l0_queries.grad.float().norm()))
            max_bridge_gradient = max(max_bridge_gradient, float(adapter.proj.weight.grad.float().norm()))
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            trace.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm), "per_level_ce": levels, **info})
            sample_schedule.append(indices.tolist())
            noise_schedule.append({"mask": corruption.mask.cpu(), "replacements": corruption.replacements.cpu()})
            if step % 64 == 0 or step == steps:
                write_json(run / "trace.json", trace)
                print(json.dumps({"arm": arm, "step": step, "total": steps, "loss": float(loss.detach())}), flush=True)
            if step % 512 == 0 or step == steps:
                save(run / f"step{step}.pt", {"model": net.state_dict(), "adapter": adapter.state_dict(),
                    "optimizer": optimizer.state_dict(), "step": step, "sampler_rng": sampler.get_state(),
                    "noise_rng": noise_rng.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(), "codec_sha256": file_hash(CODEC),
                    "producer_fingerprint": p["fingerprint"], "config": asdict(cfg), "arm": arm})
        if max_query_gradient <= 0 or max_bridge_gradient <= 0 or module_hash(vae) != frozen:
            raise ValueError("gradient routing or frozen codec invariant failed")
        if any(p.grad is not None for p in vae.parameters()):
            raise ValueError("frozen codec received gradients")
        final, final_outputs = evaluate(vae, net, adapter, prefix, last, targets, labels, roots, cases, gpt)
        write_json(run / "final_outputs.json", final_outputs)
        write_json(run / "final_metrics.json", final)
        save(run / "schedules.pt", {"sampling": sample_schedule, "noise": noise_schedule, "evaluation_roots": roots})
        # Compare the serialized tensors, rather than storage-dependent .pt bytes.
        schedule_hash = q.digest({"sampling": sample_schedule, "noise": [
            {k: v.tolist() for k, v in row.items()} for row in noise_schedule]})
        schedules[arm] = schedule_hash
        reload = torch.load(run / f"step{steps}.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(reload["model"], strict=True); enable_coarse_ar(net)
        adapter.load_state_dict(reload["adapter"], strict=True)
        del reload; gc.collect(); torch.clear_autocast_cache()
        remeasured, repeated = evaluate(vae, net, adapter, prefix, last, targets, labels, roots, cases, gpt)
        if repeated != final_outputs or remeasured != final:
            raise ValueError("checkpoint reload changed diagnostic predictions")
        verify(p)
        result = {"status": "passed" if pilot else "completed_training_mechanism", "arm": arm,
            "cases": len(cases), "steps": steps, "baseline": baseline, "final": final,
            "initial_hash": initial_hash, "codec_frozen_hash": frozen,
            "max_query_gradient": max_query_gradient, "max_bridge_gradient": max_bridge_gradient,
            "sample_noise_schedule_hash": schedule_hash, "reload_equal": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "elapsed_seconds": time.perf_counter()-start_time,
            "producer_fingerprint": p["fingerprint"], "task_accuracy": None,
            "final_test_opened": False, "speed_tested": False, "underpowered": True, "confirmed": False}
        write_json(run / "result.json", result); results[arm] = result
        del optimizer, logits, loss, context, expert, conditioning; gc.collect(); torch.cuda.empty_cache()
    if schedules["static"] != schedules["repair"] or results["static"]["initial_hash"] != results["repair"]["initial_hash"]:
        raise ValueError("matched initialization or random schedule changed")
    static, repair = results["static"], results["repair"]
    advantage = repair["final"]["all_wrong"]["nonpad_reconstruction"]-static["final"]["all_wrong"]["nonpad_reconstruction"]
    clean_delta = repair["final"]["free"]["nonpad_reconstruction"]-repair["baseline"]["free"]["nonpad_reconstruction"]
    passed = advantage >= .05 and clean_delta >= -.02
    aggregate = {"status": "passed" if pilot else ("mechanism_gate_passed" if passed else "mechanism_gate_failed"),
        "results": results, "all_wrong_repair_advantage": advantage, "free_reconstruction_delta": clean_delta,
        "matched_initialization_and_schedules": True, "no_noise_vendor_parity": True,
        "coarse_causality_and_fine_no_label_leakage": True, "minimum_neighbor_squared_distance": float(distances.min()),
        "source_hashes": campaign_producer["source_hashes"], "producer_fingerprint": campaign_producer["fingerprint"], "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False, "underpowered": True, "confirmed": False}
    verify(campaign_producer)
    write_json(base / "result.json", aggregate)
    print(json.dumps({k: v for k, v in aggregate.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "train"])
    main(parser.parse_args().command == "pilot")
