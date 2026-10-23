"""Bounded matched adaptation of the saved frozen-body NCM, with clean targets."""
import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import shutil
import time

import torch

from experiments.structure_net.ncm_frozen_body import (
    CODEC, GENERATOR, bridge, external_codes, external_logits, module_hash)
from experiments.structure_net.ncm_llama import (
    GPT2, INITIAL, VENDOR, autocast, decode_ids, encode_codes, final_latent,
    label_ids, load_ncm, tokenizers)
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_recipe_control import PLAN, ROOT
from experiments.structure_net.ncm_recipe_data import PARENT
from experiments.structure_net.ncm_recipe_teacher import QUALITY
from experiments.structure_net.ncm_training_recipe import (
    LEGACY, UPDATED, cumulative_latent, decoder_loss, generator_loss)
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.train_ncm_memory import frozen_digest, save

CONTROL = ROOT / "20261022_control_v2/result.json"


@dataclass(frozen=True)
class Config:
    cases: int = 32
    batch: int = 8
    codec_steps: int = 128
    generator_steps: int = 512
    lr: float = 1e-4
    seed: int = 42
    dropout_seed: int = 52
    sampler_seed: int = 2042
    augmentation_seed: int = 3042


def verify(producer):
    for p, expected in {**producer["source_hashes"], **producer["input_hashes"]}.items():
        if file_hash(p) != expected:
            raise ValueError("recipe producer changed: " + p)


def freeze(args, cases):
    if not torch.cuda.is_available() or args.output.exists():
        raise ValueError("CUDA and fresh output required")
    if json.loads(CONTROL.read_text())["status"] != "passed":
        raise ValueError("original XSum checkpoint control failed")
    if json.loads((QUALITY / "result.json").read_text())["status"] != "passed":
        raise ValueError("clean training target gate failed")
    if len(cases) != Config().cases or len({r["id"] for r in cases}) != len(cases):
        raise ValueError("fixed 32 clean training cases required")
    campaign = json.loads((PARENT / "campaign_results.json").read_text())
    if file_hash(PARENT / "campaign_results.json") != "b514e093a892cd2dd4cdf46cdd018d900d41371fa5b233f623c6764b98e406b3":
        raise ValueError("parent experiment changed")
    for path in [PARENT / "hybrid.pt", *[PARENT / "train_features" / (r["id"]+".pt") for r in cases]]:
        if file_hash(path) != campaign["artifacts"][str(path)]:
            raise ValueError("saved hybrid or frozen features changed")
    sources = [Path(__file__), PLAN,
        Path("docs/07 - Status Reports/2026-10-22_ncm-recipe-public-target-clarification.md"),
        Path("docs/07 - Status Reports/2026-10-22_ncm-recipe-teacher-target-amendment.md")]
    sources += [Path("experiments/structure_net") / n for n in
        ("ncm_training_recipe.py", "ncm_recipe_control.py", "ncm_recipe_data.py", "ncm_recipe_teacher.py", "ncm_llama.py",
         "ncm_frozen_body.py", "ncm_memory_data.py", "ncm_vocabulary.py", "train_ncm_memory.py")]
    sources += sorted(VENDOR.rglob("*.py"))
    inputs = [CODEC, GENERATOR, INITIAL / "config.yaml", PARENT / "hybrid.pt",
        PARENT / "campaign_results.json", CONTROL, QUALITY / "selected.json", QUALITY / "result.json"]
    inputs += [PARENT / "train_features" / (r["id"]+".pt") for r in cases]
    inputs += [p for p in GPT2.iterdir() if p.is_file()]
    producer = {"schema": "nal.ncm-recipe-adaptation.v1", "hypothesis_id": "ncm-recipe-update-v1",
        "config": asdict(Config()), "recipes": {"legacy": LEGACY.record(), "updated": UPDATED.record()},
        "source_hashes": {str(p): file_hash(p) for p in sources},
        "input_hashes": {str(p): file_hash(p) for p in inputs},
        "physical_gpu": 0, "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "precision": "FP32 parameters, BF16 autocast", "workers": 1, "parent_body_loaded": False,
        "parent_body_sha256": campaign["training"]["body_sha256"],
        "scope": "matched 32-example training competence only; no held-out accuracy or speed"}
    if args.command == "train":
        pilot = json.loads((args.pilot / "result.json").read_text())
        pp = json.loads((args.pilot / "producer.json").read_text())
        if pilot["status"] != "passed" or any(pp[k] != producer[k] for k in
                ("source_hashes", "input_hashes", "config", "recipes")):
            raise ValueError("matching CUDA training lifecycle pilot required")
    args.output.mkdir(parents=True)
    copies = args.output / "producer_source"
    copies.mkdir()
    for i, p in enumerate(sources):
        shutil.copyfile(p, copies / f"{i:03d}_{p.name}")
    producer["fingerprint"] = text_hash(json.dumps(producer, sort_keys=True))
    write_json(args.output / "producer.json", producer)
    write_json(args.output / "selection.json", {"train": cases})
    return producer


@torch.inference_mode()
def evaluate(vae, net, adapter, hidden, masks, targets, codes, cases, tokenizer):
    net.eval()
    vae.eval()
    adapter.eval()
    outputs = []
    for i, case in enumerate(cases):
        with autocast():
            context = adapter(hidden[i:i+1].float())
            logits = external_logits(vae, net, context, masks[i:i+1], codes[i:i+1])
            generated = external_codes(net, context, masks[i:i+1])
            text_logits = vae.decoder(cumulative_latent(vae, generated))
            oracle_logits = vae.decoder(cumulative_latent(vae, codes[i:i+1]))
        ids = text_logits.argmax(-1)[0].tolist()
        decoded, termination = decode_ids(ids, tokenizer)
        match = logits.argmax(-1).eq(codes[i:i+1])
        content = targets[i:i+1].ne(50257) & targets[i:i+1].ne(50256)
        outputs.append({"id": case["id"], "source": case["source"], "target": case["target"],
            "text": decoded, "native_output_ids": ids, "termination": termination,
            "coarse_correct": int(match[:, :vae.vq.levels[0]].sum()),
            "coarse_total": vae.vq.levels[0], "teacher_codes_correct": int(match.sum()),
            "codes_total": match.numel(), "free_codes_correct": int(generated.eq(codes[i:i+1]).sum()),
            "oracle_content_correct": int((oracle_logits.argmax(-1).eq(targets[i:i+1]) & content).sum()),
            "oracle_content_total": int(content.sum())})
    return outputs


def run(args):
    cfg = Config()
    cases = json.loads((QUALITY / "selected.json").read_text())["train"]
    # Teacher rows retain original source metadata. Do not mislabel the saved
    # target's cl100k length as a measurement of the newly generated target.
    for case in cases:
        if "target_tokens" in case:
            case["original_saved_target_cl100k_tokens"] = case.pop("target_tokens")
    producer = freeze(args, cases)
    started = time.perf_counter()
    torch.manual_seed(cfg.seed)
    torch.cuda.reset_peak_memory_stats()
    gpt, _ = tokenizers()
    gpt = CachedVocabulary(gpt)
    vae, net, _ = load_ncm(codec=CODEC, generator=GENERATOR, adaptation=True)
    state = torch.load(PARENT / "hybrid.pt", map_location="cpu", weights_only=True)
    initial_net, initial_adapter = state["generator"], state["adapter"]
    del state
    initial_codec = torch.load(CODEC, map_location="cpu", weights_only=True)["model"]
    features = [torch.load(PARENT / "train_features" / (r["id"]+".pt"), weights_only=True) for r in cases]
    hidden = torch.stack([f["hidden"] for f in features]).cuda()
    masks = torch.stack([f["pad_mask"] for f in features]).cuda()
    del features
    targets = torch.tensor([label_ids(r["target"], gpt) for r in cases], device="cuda")
    codes = encode_codes(vae, targets)
    with torch.no_grad(), autocast():
        if not torch.equal(cumulative_latent(vae, codes[:1]), final_latent(vae, codes[:1])):
            raise RuntimeError("full cumulative latent differs from vendor reconstruction")
        if torch.equal(cumulative_latent(vae, codes[:1], 1), cumulative_latent(vae, codes[:1])):
            raise RuntimeError("nested cutoff did not remove finer levels")
    frozen = frozen_digest(vae)
    results = {}
    matched_picks = {}
    for arm, recipe in (("legacy", LEGACY), ("updated", UPDATED)):
        net.load_state_dict(initial_net, strict=True)
        vae.load_state_dict(initial_codec, strict=True)
        adapter = bridge().cuda()
        adapter.load_state_dict(initial_adapter, strict=True)
        torch.manual_seed(cfg.dropout_seed)
        augmentation = torch.Generator().manual_seed(cfg.augmentation_seed)
        max_query_gradient = max_adapter_gradient = 0.
        picks, traces, cutoffs = [], {}, []
        for stage, configured_steps in (("codec", cfg.codec_steps), ("generator", cfg.generator_steps)):
            vae.eval().requires_grad_(False)
            for parameter in vae.parameters():
                parameter.grad = None
            if stage == "codec":
                vae.decoder.train().requires_grad_(True)
                parameters = list(vae.decoder.parameters())
            else:
                net.train()
                adapter.train()
                codec_before_generator = module_hash(vae)
                parameters = list(net.parameters()) + list(adapter.parameters())
            optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, betas=(.9, .95), weight_decay=.01)
            sampler = torch.Generator().manual_seed(cfg.sampler_seed)
            steps = 2 if args.command == "pilot" else configured_steps
            trace = []
            for step in range(1, steps+1):
                pick_cpu = torch.randint(len(cases), (cfg.batch,), generator=sampler)
                picks.append(text_hash(json.dumps([stage, step, pick_cpu.tolist()])))
                pick = pick_cpu.cuda()
                optimizer.zero_grad(set_to_none=True)
                factor = step/32 if step <= 32 else .1 + .45*(1+math.cos(math.pi*(step-32)/(configured_steps-32)))
                for group in optimizer.param_groups:
                    group["lr"] = cfg.lr * factor
                with autocast():
                    if stage == "codec":
                        loss, cutoff = decoder_loss(vae, codes[pick], targets[pick], recipe, augmentation)
                        details = {"cutoff": cutoff}
                        cutoffs.append(cutoff)
                    else:
                        context = adapter(hidden[pick].float())
                        loss, details = generator_loss(vae, net, codes[pick], recipe, augmentation,
                                                      context=context, pad_mask=masks[pick])
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite recipe objective")
                loss.backward()
                if stage == "generator":
                    max_query_gradient = max(max_query_gradient, float(net.l0_queries.grad.float().norm()))
                    gradient = sum(float(p.grad.float().square().sum()) for p in adapter.parameters() if p.grad is not None)**.5
                    max_adapter_gradient = max(max_adapter_gradient, gradient)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                trace.append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(norm), **details})
                if step % 32 == 0 or step == steps:
                    print(json.dumps({"arm": arm, "stage": stage, "step": step, "loss": trace[-1]["loss"]}), flush=True)
            traces[stage] = trace
            write_json(args.output / (arm+"_"+stage+"_trace.json"), trace)
            if frozen_digest(vae) != frozen:
                raise RuntimeError("encoder/codebooks changed")
            if stage == "generator" and module_hash(vae) != codec_before_generator:
                raise RuntimeError("codec changed during generator training")
            checkpoint = {"model": vae.state_dict()} if stage == "codec" else {
                "generator": net.state_dict(), "adapter": adapter.state_dict(),
                "codec_checkpoint": str(args.output / (arm+"_codec.pt")),
                "codec_sha256": file_hash(args.output / (arm+"_codec.pt"))}
            checkpoint.update(optimizer=optimizer.state_dict(), augmentation_rng=augmentation.get_state(),
                sampler_rng=sampler.get_state(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(), step=steps, producer_fingerprint=producer["fingerprint"],
                recipe=recipe.record(), levels=list(vae.vq.levels),
                vocabularies=list(vae.vq.vqvae_vocab_sizes), frozen_encoder_codebooks_sha256=frozen)
            save(args.output / (arm+"_"+stage+".pt"), checkpoint)
            del optimizer, checkpoint, parameters
            gc.collect()
            torch.cuda.empty_cache()
        if max_query_gradient <= 0 or max_adapter_gradient <= 0:
            raise RuntimeError("missing coarse-query/adapter gradient")
        vae.eval().requires_grad_(False)
        net.eval()
        adapter.eval()
        outputs = evaluate(vae, net, adapter, hidden, masks, targets, codes,
                           cases[:2] if args.command == "pilot" else cases, gpt)
        write_json(args.output / (arm+"_outputs.json"), {"outputs": outputs})
        with torch.inference_mode(), autocast():
            context = adapter(hidden[:1].float())
            before = external_logits(vae, net, context, masks[:1], codes[:1]).clone()
        loaded = torch.load(args.output / (arm+"_generator.pt"), map_location="cpu", weights_only=True)
        if loaded["codec_sha256"] != file_hash(loaded["codec_checkpoint"]):
            raise RuntimeError("generator/codec checkpoint pairing changed")
        net.load_state_dict(loaded["generator"], strict=True)
        adapter.load_state_dict(loaded["adapter"], strict=True)
        codec_state = torch.load(args.output / (arm+"_codec.pt"), map_location="cpu", weights_only=True)
        before_codec = module_hash(vae)
        vae.load_state_dict(codec_state["model"], strict=True)
        if module_hash(vae) != before_codec:
            raise RuntimeError("codec reload changed weights")
        with torch.inference_mode(), autocast():
            after = external_logits(vae, net, adapter(hidden[:1].float()), masks[:1], codes[:1])
        if not torch.equal(before, after):
            raise RuntimeError("generator checkpoint reload changed logits")
        results[arm] = {"cases": len(outputs), "coarse_code_accuracy": sum(o["coarse_correct"] for o in outputs)/sum(o["coarse_total"] for o in outputs),
            "free_code_accuracy": sum(o["free_codes_correct"] for o in outputs)/sum(o["codes_total"] for o in outputs),
            "oracle_content_accuracy": sum(o["oracle_content_correct"] for o in outputs)/sum(o["oracle_content_total"] for o in outputs),
            "max_query_gradient": max_query_gradient, "max_adapter_gradient": max_adapter_gradient,
            "cutoffs_seen": sorted(set(cutoffs)), "reload_equal": True,
            "codec_unchanged_during_generator": True, "frozen_encoder_codebooks": True,
            "first_generator_loss": traces["generator"][0]["loss"],
            "last_generator_loss": traces["generator"][-1]["loss"]}
        matched_picks[arm] = picks
        del loaded, codec_state, adapter
        gc.collect()
        torch.cuda.empty_cache()
    if matched_picks["legacy"] != matched_picks["updated"]:
        raise RuntimeError("matched training batches differed")
    verify(producer)
    result = {"status": "passed" if args.command == "pilot" else "trained_pending_train_competence_judging",
        "label": "systems_lifecycle_only_not_quality_evidence" if args.command == "pilot" else "training_set_memorization_not_heldout_accuracy",
        "arms": results, "producer_fingerprint": producer["fingerprint"], "matched_batches": True,
        "frozen_encoder_codebooks_sha256": frozen, "parent_body_loaded": False,
        "elapsed_seconds": time.perf_counter()-started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "heldout_accuracy": None, "speed_tested": False}
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "train"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    run(parser.parse_args())
