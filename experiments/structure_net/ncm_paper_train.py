"""Paper-style unpaired -> bridge -> paired NCM adaptation, with frozen codec."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import time

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_frozen_body import bridge, external_codes, module_hash
from experiments.structure_net.ncm_llama import (
    INITIAL, autocast, decode_ids, encode_codes, infer_ncm, load_ncm, tokenizers)
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_context import right_pack, unpaired_windows
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import PARENT, PREP, ROOT, TEACHER, verify
from experiments.structure_net.ncm_training_recipe import Recipe, cumulative_latent, generator_loss
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.train_ncm_memory import save

NOTE = Path("docs/07 - Status Reports/2026-10-22_ncm-paper-training-configuration.md")
RECIPE = Recipe(0., .30, 0., 0.)


@dataclass(frozen=True)
class Config:
    unpaired_steps: int = 512
    bridge_steps: int = 512
    conditional_steps: int = 2048
    microbatch: int = 8
    accumulation: int = 2
    generator_lr: float = 1e-4
    bridge_lr: float = 3e-4
    weight_decay: float = .1
    warmup: int = 128
    seed: int = 42
    dropout_seed: int = 52
    sampler_seed: int = 2042
    augmentation_seed: int = 3042


def lr_factor(step, total, warmup):
    # Short lifecycle runs still execute nonzero real updates.
    warmup = min(warmup, max(1, total//4))
    if step <= warmup:
        return step/warmup
    return .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1, total-warmup)))


def load_context(cases):
    """CPU feature bank, only the selected microbatch is transferred to CUDA."""
    hidden = torch.empty(len(cases), 512, 3072, dtype=torch.bfloat16)
    masks = torch.empty(len(cases), 512, dtype=torch.bool)
    for i, case in enumerate(cases):
        state = torch.load(PARENT / "train_features" / (case["id"]+".pt"), weights_only=True)
        h, m = right_pack(state["hidden"][None], state["pad_mask"][None])
        hidden[i], masks[i] = h[0], m[0]
    return hidden, masks


def make_checkpoint(net, adapter, optimizer, sampler, aug, step, stage, codec, fingerprint):
    result = {"model": net.state_dict(), "adapter": adapter.state_dict() if adapter is not None else None,
        "optimizer": optimizer.state_dict(), "sampler_rng": sampler.get_state(),
        "augmentation_rng": aug.get_state(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(), "stage": stage, "step": step,
        "codec_checkpoint": str(codec), "codec_sha256": file_hash(codec),
        "producer_fingerprint": fingerprint, "config": asdict(Config()), "recipe": RECIPE.record(),
        "levels": list(net.vqvae_local[0].vq.levels),
        "vocabularies": list(net.vqvae_local[0].vq.vqvae_vocab_sizes)}
    return result


@torch.inference_mode()
def evaluate(vae, net, adapter, cases, gpt, features=None):
    net.eval()
    if adapter is not None:
        adapter.eval()
    output = []
    for start in range(0, len(cases), 4):
        chunk = cases[start:start+4]
        if adapter is None:
            ids = [gpt.encode(r["source"], add_special_tokens=False) for r in chunk]
            if any(len(x) > 512 for x in ids):
                raise ValueError("prefix truncation prohibited")
            prefix = torch.tensor([x+[50257]*(512-len(x)) for x in ids], device="cuda")
            generated = infer_ncm(vae, net, prefix, gpt)
        else:
            if features is None:
                states = [torch.load(ROOT / "20261022_features_v1/features" / (r["id"]+".pt"), weights_only=True) for r in chunk]
                hidden = torch.stack([s["hidden"] for s in states]).cuda()
                masks = torch.stack([s["pad_mask"] for s in states]).cuda()
            else:
                hidden = features[0][start:start+len(chunk)].cuda()
                masks = features[1][start:start+len(chunk)].cuda()
            with autocast():
                codes = external_codes(net, adapter(hidden.float()), masks)
                raw = vae.decoder(cumulative_latent(vae, codes)).argmax(-1).tolist()
            generated = [{"text": decode_ids(ids, gpt)[0], "native_output_ids": ids,
                          "termination": decode_ids(ids, gpt)[1]} for ids in raw]
        output.extend({**case, **result} for case, result in zip(chunk, generated))
    return output


def run(args):
    cfg = Config()
    pilot = args.command == "pilot"
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    teacher = ROOT / "20261022_teacher_pilot_v1" if pilot else TEACHER
    cases = q.read(teacher / "train.json")
    if pilot:
        cases = cases[:16]
    elif (q.read(TEACHER / "result.json")["status"] != "passed" or
          q.read(ROOT / "20261022_teacher_audit_v1/result.json")["status"] != "passed"):
        raise ValueError("paired-label gates required")
    codec = args.codec or INITIAL / "tokenizer.pt"
    if not pilot:
        control = q.read(args.codec_control)
        oracle = q.read(args.oracle_control)
        if (control["nonpad_reconstruction"] < .98 or oracle["status"] != "completed" or
                oracle["passed"] < 31 or oracle["cases"] != 32):
            raise ValueError("codec competence gates required")
        pilot_result = q.read(args.pilot / "result.json")
        if pilot_result["status"] != "passed":
            raise ValueError("training lifecycle pilot required")
    steps = {"unpaired": 2 if pilot else cfg.unpaired_steps,
        "bridge": 2 if pilot else cfg.bridge_steps, "conditional": 2 if pilot else cfg.conditional_steps}
    stages = ["unpaired", "bridge", "conditional"] if pilot else (["unpaired"] if args.command == "unpaired" else
        (["bridge", "conditional"] if args.command == "hybrid" else ["conditional"]))
    inputs = [teacher / "train.json", codec, INITIAL / "generator.pt", INITIAL / "config.yaml"]
    if not pilot:
        inputs.extend([args.codec_control, args.oracle_control, ROOT / "20261022_teacher_audit_v1/result.json",
                       args.pilot / "result.json", args.pilot / "producer.json"])
        if args.command in ("native", "hybrid"):
            inputs.extend([args.unpaired / "model.pt", args.unpaired / "result.json"])
            if q.read(args.unpaired / "result.json")["status"] != "passed":
                raise ValueError("unpaired stage must complete first")
    if args.command == "hybrid" or pilot:
        manifest = q.read(PREP / "producer.json")["input_hashes"]
        for case in cases:
            path = PARENT / "train_features" / (case["id"]+".pt")
            if file_hash(path) != manifest[str(path)]:
                raise ValueError("cached feature mismatch")
            inputs.append(path)
    p = producer(args.output, "systems_lifecycle_only_not_quality_evidence" if pilot else args.command,
                  inputs, {**asdict(cfg), "actual_stage_steps": steps, "recipe": RECIPE.record(), "cases": len(cases)})
    sources = [Path(__file__), Path("experiments/structure_net/ncm_paper_context.py"), NOTE]
    for source in sources:
        p["source_hashes"][str(source)] = file_hash(source)
        (args.output / "producer_source" / ("training_"+source.name)).write_bytes(source.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(args.output / "producer.json", p)
    if not pilot:
        pp = q.read(args.pilot / "producer.json")
        if p["source_hashes"] != pp["source_hashes"]:
            raise ValueError("training sources differ from passed pilot")
    torch.manual_seed(cfg.seed)
    torch.cuda.reset_peak_memory_stats()
    gpt, _ = tokenizers()
    gpt = CachedVocabulary(gpt)
    generator = INITIAL / "generator.pt" if pilot or args.command == "unpaired" else args.unpaired / "model.pt"
    vae, net, _ = load_ncm(codec=codec, generator=generator)
    if generator != INITIAL / "generator.pt":
        state = torch.load(generator, map_location="cpu", weights_only=True)
        if state["codec_sha256"] != file_hash(codec):
            raise ValueError("unpaired generator/codec pair mismatch")
        del state
    frozen = module_hash(vae)
    targets = torch.tensor([r["native_target_ids"] for r in cases])
    windows = unpaired_windows(cases, gpt)
    def encode_all(ids):
        return torch.cat([encode_codes(vae, ids[i:i+32].cuda()).cpu() for i in range(0, len(ids), 32)])
    target_codes = encode_all(targets)
    window_codes = encode_all(windows) if "unpaired" in stages else None
    prefix = []
    for row in cases:
        ids = gpt.encode(row["source"], add_special_tokens=False)
        if len(ids) > 512:
            raise ValueError("prefix truncation prohibited")
        prefix.append(ids + [50257]*(512-len(ids)))
    prefix = torch.tensor(prefix)
    features = load_context(cases) if args.command == "hybrid" or pilot else None
    torch.manual_seed(cfg.seed)
    adapter = bridge().cuda() if features is not None else None
    started = time.perf_counter()
    summary, all_picks = {}, {}
    for stage in stages:
        net.zero_grad(set_to_none=True)
        net.requires_grad_(stage != "bridge").train(stage != "bridge")
        vae.eval().requires_grad_(False)
        if adapter is not None:
            adapter.requires_grad_(True).train()
        groups = []
        if stage != "bridge":
            groups.append({"params": [x for x in net.parameters() if x.requires_grad], "lr": cfg.generator_lr, "peak_lr": cfg.generator_lr})
        if stage == "bridge" or (stage == "conditional" and adapter is not None):
            groups.append({"params": list(adapter.parameters()), "lr": cfg.bridge_lr, "peak_lr": cfg.bridge_lr})
        parameters = [x for group in groups for x in group["params"]]
        optimizer = torch.optim.AdamW(groups, betas=(.9, .95), weight_decay=cfg.weight_decay)
        sampler = torch.Generator().manual_seed(cfg.sampler_seed + (100 if stage == "bridge" else 0))
        aug = torch.Generator().manual_seed(cfg.augmentation_seed + (100 if stage == "bridge" else 0))
        torch.manual_seed(cfg.dropout_seed + (100 if stage == "bridge" else 0))
        torch.cuda.manual_seed_all(cfg.dropout_seed + (100 if stage == "bridge" else 0))
        before_net = module_hash(net) if stage == "bridge" else None
        trace, picks = [], []
        max_adapter_gradient = max_query_gradient = 0.
        for step in range(1, steps[stage]+1):
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = group["peak_lr"] * lr_factor(step, steps[stage], 32 if stage == "bridge" else cfg.warmup)
            total_loss = 0.
            for micro in range(cfg.accumulation):
                labels = window_codes if stage == "unpaired" else target_codes
                idx = torch.randint(len(labels), (cfg.microbatch,), generator=sampler)
                picks.extend(idx.tolist())
                clean = labels[idx].cuda()
                with autocast():
                    if stage == "unpaired":
                        native_prefix = torch.full((len(idx), 512), 50257, device="cuda", dtype=torch.long)
                        native_prefix[:, 0] = 50256
                        loss, details = generator_loss(vae, net, clean, RECIPE, aug, prefix=native_prefix)
                    elif adapter is None:
                        loss, details = generator_loss(vae, net, clean, RECIPE, aug, prefix=prefix[idx].cuda())
                    else:
                        context = adapter(features[0][idx].cuda().float())
                        loss, details = generator_loss(vae, net, clean, RECIPE, aug, context=context,
                                                       pad_mask=features[1][idx].cuda())
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite objective")
                (loss/cfg.accumulation).backward()
                total_loss += float(loss.detach())/cfg.accumulation
            if stage != "bridge" and net.l0_queries.grad is not None:
                max_query_gradient = max(max_query_gradient, float(net.l0_queries.grad.float().norm()))
            if stage != "unpaired" and adapter is not None:
                norm = sum(float(x.grad.float().square().sum()) for x in adapter.parameters() if x.grad is not None)**.5
                max_adapter_gradient = max(max_adapter_gradient, norm)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            trace.append({"step": step, "loss": total_loss, "gradient_norm": float(norm), **details})
            if step % 64 == 0 or step == steps[stage]:
                print(json.dumps({"stage": stage, "step": step, "total": steps[stage], "loss": total_loss}), flush=True)
                write_json(args.output / (stage+"_trace.json"), trace)
            if not pilot and stage == "conditional" and step % 512 == 0:
                # These are development-only outputs. Preserve each checkpoint.
                path = args.output / f"step{step}.pt"
                save(path, make_checkpoint(net, adapter, optimizer, sampler, aug, step, stage, codec, p["fingerprint"]))
                outputs = evaluate(vae, net, adapter, q.read(PREP / "development_sources.json"), gpt)
                write_json(args.output / f"development_step{step}.json", outputs)
                net.train()
                if adapter is not None:
                    adapter.train()
        if stage == "bridge" and module_hash(net) != before_net:
            raise ValueError("generator changed during bridge-only warmup")
        if module_hash(vae) != frozen or any(x.grad is not None for x in vae.parameters()):
            raise ValueError("codec changed or received gradients")
        if stage != "bridge" and max_query_gradient <= 0:
            raise ValueError("missing coarse-query gradient")
        if stage in ("bridge", "conditional") and adapter is not None and max_adapter_gradient <= 0:
            raise ValueError("missing adapter gradient")
        path = args.output / ("model.pt" if stage == stages[-1] else stage+".pt")
        save(path, make_checkpoint(net, adapter, optimizer, sampler, aug, steps[stage], stage, codec, p["fingerprint"]))
        summary[stage] = {"updates": steps[stage], "sampled_examples": len(picks),
            "unique_examples": len(set(picks)), "first_loss": trace[0]["loss"], "last_loss": trace[-1]["loss"],
            "max_query_gradient": max_query_gradient, "max_adapter_gradient": max_adapter_gradient}
        all_picks[stage] = text_hash(json.dumps(picks))
        del optimizer, parameters, groups
        gc.collect()
        torch.cuda.empty_cache()
    net.eval()
    if adapter is not None:
        adapter.eval()
    with torch.inference_mode():
        before = evaluate(vae, net, adapter, cases[:2], gpt,
                          (features[0][:2], features[1][:2]) if features is not None else None)
    checkpoint = torch.load(args.output / "model.pt", map_location="cpu", weights_only=True)
    if checkpoint["codec_sha256"] != file_hash(codec):
        raise ValueError("checkpoint codec identity changed")
    net.load_state_dict(checkpoint["model"], strict=True)
    if adapter is not None:
        adapter.load_state_dict(checkpoint["adapter"], strict=True)
    torch.clear_autocast_cache()
    after = evaluate(vae, net, adapter, cases[:2], gpt,
                     (features[0][:2], features[1][:2]) if features is not None else None)
    if before != after:
        raise ValueError("checkpoint reload changed output")
    if not pilot and args.command in ("native", "hybrid"):
        outputs = evaluate(vae, net, adapter, cases[:32], gpt,
                           (features[0][:32], features[1][:32]) if features is not None else None)
        write_json(args.output / "training_diagnostics.json", outputs)
    verify(p)
    result = {"status": "passed", "label": p["stage"], "producer_fingerprint": p["fingerprint"],
        "stage_results": summary, "sample_schedule_hashes": all_picks, "reload_equal": True,
        "codec_sha256": file_hash(codec), "codec_frozen_sha256": frozen,
        "parent_body_loaded": False, "parent_body_frozen": True,
        "elapsed_seconds": time.perf_counter()-started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "test_accuracy": None, "speed_tested": False, "underpowered": True}
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "unpaired", "native", "hybrid"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec", type=Path)
    parser.add_argument("--codec-control", type=Path)
    parser.add_argument("--oracle-control", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--unpaired", type=Path)
    run(parser.parse_args())
