"""Continue the existing NCM on the shared, saved conversational summaries."""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import time

import torch

from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_llama import (
    BASE, GPT2, INITIAL, VENDOR, autocast, code_loss, encode_codes, final_latent,
    generator_input, infer_ncm, label_ids, load_ncm, tokenizers)


PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-memory-training-plan.md")
CODEC = BASE / "20261021_codec_v1/model.pt"
GENERATOR = BASE / "20261021_generator_v1/model.pt"
EXPECTED = {CODEC: "b87f7288edfc0eb1f5b9122d1139cc2e61aa2331be7d6fe53f0582993086ede2",
            GENERATOR: "ff7133b2de50bc1f66ef1e490fec4c723e0fc5a5c1957107209e923b198c3e3c"}


@dataclass(frozen=True)
class Config:
    seed: int = 42
    codec_steps: int = 512
    generator_steps: int = 1024
    codec_batch: int = 16
    generator_batch: int = 8
    codec_lr: float = 2e-4
    generator_lr: float = 1e-4
    eval_limit: int = 256
    prefix: int = 512
    canvas: int = 48


def word_unigram_f1(prediction, reference):
    p = Counter(re.findall(r"\w+", prediction.lower()))
    r = Counter(re.findall(r"\w+", reference.lower()))
    denominator = sum(p.values()) + sum(r.values())
    return 2 * sum((p & r).values()) / denominator if denominator else 0.0


def codec_loss(logits, target):
    return torch.nn.functional.cross_entropy(logits.float().flatten(0, 1),
                                            target.flatten(), ignore_index=50257)


def save(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def frozen_digest(vae):
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(vae.state_dict().items()):
        if not name.startswith("decoder."):
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def provenance(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    for path, expected in EXPECTED.items():
        if file_hash(path) != expected:
            raise ValueError("initial checkpoint hash mismatch")
    manifest = json.loads((args.data / "manifest.json").read_text())
    for name in ("native_train.jsonl", "native_test.jsonl"):
        if file_hash(args.data / name) != manifest["files"][name]["sha256"]:
            raise ValueError("dataset readback mismatch")
    if args.output.exists():
        raise ValueError("fresh append-only output required")
    args.output.mkdir(parents=True)
    paths = [Path(__file__), Path("experiments/structure_net/ncm_memory_data.py"),
             Path("experiments/structure_net/ncm_llama.py"),
             Path("experiments/structure_net/prepare_llama_cotenancy.py"), PLAN,
             Path("docs/07 - Status Reports/2026-10-22_ncm-memory-training-lifecycle-fix.md")]
    paths += sorted(VENDOR.rglob("*.py"))
    copies = args.output / "producer_source"
    copies.mkdir()
    source_hashes = {}
    for i, path in enumerate(paths):
        shutil.copyfile(path, copies / f"{i:03d}_{path.name}")
        source_hashes[str(path)] = file_hash(path)
    inputs = [args.data / "manifest.json", args.data / "native_train.jsonl",
              args.data / "native_test.jsonl", CODEC, GENERATOR, INITIAL / "config.yaml"]
    inputs += [p for p in GPT2.iterdir() if p.is_file()]
    result = {"hypothesis_id": "ncm-memory-summaries-v1", "schema": "nal.ncm-memory-training.v1",
              "stage": args.command, "config": asdict(Config()), "source_hashes": source_hashes,
              "input_hashes": {str(p): file_hash(p) for p in inputs},
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "client_date": "2026-10-22", "torch": torch.__version__,
              "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
              "physical_gpu": 0, "workers": 1, "underpowered": True}
    result["fingerprint"] = text_hash(json.dumps(result, sort_keys=True))
    write_json(args.output / "producer.json", result)
    return result


def verify(producer):
    for path, expected in {**producer["source_hashes"], **producer["input_hashes"]}.items():
        if file_hash(path) != expected:
            raise ValueError("producing source/input changed: " + path)


def rows(path):
    # Unicode paragraph separators can occur inside valid JSON strings. Only
    # physical newlines delimit JSONL records; str.splitlines() also splits U+2028.
    with path.open() as source:
        result = [json.loads(line) for line in source]
    if not result or len({r["id"] for r in result}) != len(result):
        raise ValueError("nonempty unique rows required")
    for r in result:
        if (not r["native_eligible"] or text_hash(r["source"]) != r["source_sha256"]
                or text_hash(r["target"]) != r["target_sha256"]):
            raise ValueError("data pairing mismatch")
    return result


def targets_and_prefixes(records, gpt):
    targets = torch.tensor([label_ids(r["target"], gpt) for r in records])
    prefixes = []
    for r in records:
        ids = gpt.encode(r["source"], add_special_tokens=False)
        if not 0 < len(ids) <= 512:
            raise ValueError("source exceeds native capacity")
        prefixes.append(ids + [50257] * (512-len(ids)))
    return targets, torch.tensor(prefixes)


def pack(vae, records, gpt):
    targets, prefixes = targets_and_prefixes(records, gpt)
    codes = []
    for start in range(0, len(records), 16):
        codes.append(encode_codes(vae, targets[start:start+16].cuda()).cpu())
        if start % 4096 == 0:
            print(json.dumps({"stage": "encode", "rows": start, "total": len(records)}), flush=True)
    return {"targets": targets, "prefixes": prefixes, "codes": torch.cat(codes)}


def optimizer(parameters, lr):
    return torch.optim.AdamW(parameters, lr=lr, betas=(.9, .95), weight_decay=.01)


def one_step(vae, net, packed, indices, stage, opt):
    opt.zero_grad(set_to_none=True)
    codes = packed["codes"][indices].cuda()
    if stage == "codec":
        vae.decoder.train()
        with autocast():
            logits = vae.decoder(final_latent(vae, codes))
            loss = codec_loss(logits, packed["targets"][indices].cuda())
        parameters = list(vae.decoder.parameters())
    else:
        net.train()
        with autocast():
            loss = code_loss(vae, net, packed["prefixes"][indices].cuda(), codes)
        parameters = list(net.parameters())
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite objective")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    opt.step()
    return {"loss": float(loss), "gradient_norm": float(norm)}


def checkpoint(path, model, opt, step, sampler, producer):
    save(path, {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                "sampler_rng": sampler.get_state(), "producer_fingerprint": producer["fingerprint"]})


def pilot(args):
    producer = provenance(args)
    gpt, _ = tokenizers()
    vae, net, _ = load_ncm(codec=CODEC, generator=GENERATOR, adaptation=True)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    chosen = rows(args.data / "native_train.jsonl")[:16]
    packed = pack(vae, chosen, gpt)
    frozen = frozen_digest(vae)
    vae.decoder.requires_grad_(True)
    opt = optimizer(vae.decoder.parameters(), Config().codec_lr)
    codec = one_step(vae, net, packed, torch.arange(16) % len(chosen), "codec", opt)
    vae.eval().requires_grad_(False)
    del opt
    opt = optimizer(net.parameters(), Config().generator_lr)
    generator = one_step(vae, net, packed, torch.arange(8) % len(chosen), "generator", opt)
    if net.l0_queries.grad is None or not net.l0_queries.grad.abs().sum() > 0:
        raise RuntimeError("level-zero query gradient missing")
    if frozen_digest(vae) != frozen:
        raise RuntimeError("frozen encoder/codebooks changed")
    net.eval()
    p = packed["prefixes"][:1].cuda()
    c = packed["codes"][:1].cuda()
    with torch.no_grad(), autocast():
        before = net(generator_input(vae, net, p, c), prefix_pad_mask=p.eq(50257)).clone()
    save(args.output / "reload.pt", {"model": net.state_dict()})
    net.load_state_dict(torch.load(args.output / "reload.pt", weights_only=True)["model"], strict=True)
    with torch.no_grad(), autocast():
        after = net(generator_input(vae, net, p, c), prefix_pad_mask=p.eq(50257))
    if not torch.equal(before, after):
        raise RuntimeError("checkpoint reload changed output")
    verify(producer)
    result = {"status": "passed", "label": "systems_lifecycle_only_not_quality_evidence",
              "codec": codec, "generator": generator, "reload_equal": True,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "producer_fingerprint": producer["fingerprint"]}
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


@torch.inference_mode()
def evaluate(vae, net, records, packed, gpt):
    vae.eval().requires_grad_(False)
    net.eval()
    outputs = []
    code_ce_sum, text_ce_sum, correct, token_count = 0., 0., 0, 0
    for start in range(0, len(records), 8):
        prefix = packed["prefixes"][start:start+8].cuda()
        codes = packed["codes"][start:start+8].cuda()
        targets = packed["targets"][start:start+8].cuda()
        with autocast():
            ce = code_loss(vae, net, prefix, codes)
            logits = vae.decoder(final_latent(vae, codes))
            text_ce = codec_loss(logits, targets)
        valid = targets.ne(50257)
        n = int(valid.sum())
        code_ce_sum += float(ce) * len(prefix)
        text_ce_sum += float(text_ce) * n
        correct += int((logits.argmax(-1).eq(targets) & valid).sum())
        token_count += n
        generated = infer_ncm(vae, net, prefix, gpt)
        for record, out in zip(records[start:start+8], generated):
            outputs.append({"id": record["id"], "source_id": record["source_id"],
                            "source": record["source"], "target": record["target"], **out,
                            "word_unigram_f1": word_unigram_f1(out["text"], record["target"]),
                            "exact_reference_match": out["text"].strip() == record["target"].strip()})
    return {"cases": len(records), "code_cross_entropy": code_ce_sum / len(records),
            "codec_text_cross_entropy": text_ce_sum / token_count,
            "codec_token_accuracy": correct / token_count,
            "mean_word_unigram_f1": sum(r["word_unigram_f1"] for r in outputs) / len(outputs),
            "exact_reference_matches": sum(r["exact_reference_match"] for r in outputs),
            "empty_outputs": sum(not r["text"] for r in outputs), "outputs": outputs,
            "interpretation": "Reference-fit diagnostics, not calibrated task accuracy."}


def train(args):
    if args.pilot is None or json.loads((args.pilot / "result.json").read_text())["status"] != "passed":
        raise ValueError("successful CUDA lifecycle pilot required")
    producer = provenance(args)
    pilot_producer = json.loads((args.pilot / "producer.json").read_text())
    for key in ("input_hashes", "source_hashes", "config"):
        if pilot_producer[key] != producer[key]:
            raise ValueError("pilot and training producer/data differ")
    cfg = Config()
    gpt, _ = tokenizers()
    training = rows(args.data / "native_train.jsonl")
    testing = rows(args.data / "native_test.jsonl")
    if ({r["source_id"] for r in training} & {r["source_id"] for r in testing}
            or {r["source_sha256"] for r in training} & {r["source_sha256"] for r in testing}):
        raise ValueError("train/test overlap")
    testing = sorted(testing, key=lambda r: text_hash("evaluation:"+r["id"]))[:cfg.eval_limit]
    write_json(args.output / "test_selection.json", {"ids": [r["id"] for r in testing],
                                                    "selection": "fixed ID hash, maximum 256"})
    vae, net, _ = load_ncm(codec=CODEC, generator=GENERATOR, adaptation=True)
    torch.manual_seed(cfg.seed)
    torch.cuda.reset_peak_memory_stats()
    frozen = frozen_digest(vae)
    started = time.perf_counter()
    train_pack = pack(vae, training, gpt)
    test_pack = pack(vae, testing, gpt)
    save(args.output / "training_codes.pt", {"ids": [r["id"] for r in training], **train_pack})
    save(args.output / "test_codes.pt", {"ids": [r["id"] for r in testing], **test_pack})
    stage_results = {}
    for stage, steps, batch, lr in [("codec", cfg.codec_steps, cfg.codec_batch, cfg.codec_lr),
                                     ("generator", cfg.generator_steps, cfg.generator_batch, cfg.generator_lr)]:
        vae.eval().requires_grad_(False)
        if stage == "codec":
            model = vae
            vae.decoder.requires_grad_(True)
            parameters = list(vae.decoder.parameters())
        else:
            model = net
            parameters = list(net.parameters())
        opt = optimizer(parameters, lr)
        sampler = torch.Generator().manual_seed(cfg.seed + (1000 if stage == "codec" else 2000))
        trace = []
        seen_ids = set()
        stage_start = time.perf_counter()
        for step in range(1, steps+1):
            pick = torch.randint(len(training), (batch,), generator=sampler)
            seen_ids.update(pick.tolist())
            if stage == "generator":
                factor = step/50 if step <= 50 else .1 + .9*.5*(1+math.cos(math.pi*(step-50)/(steps-50)))
                for group in opt.param_groups:
                    group["lr"] = lr * factor
            measurement = one_step(vae, net, train_pack, pick, stage, opt)
            trace.append({"step": step, **measurement})
            if step % 32 == 0:
                print(json.dumps({"stage": stage, **trace[-1]}), flush=True)
                write_json(args.output / (stage+"_trace.json"), trace)
            if step in (steps//2, steps):
                checkpoint(args.output / f"{stage}_continuation_{step}.pt", model, opt, step, sampler, producer)
        if frozen_digest(vae) != frozen:
            raise RuntimeError("frozen encoder/codebooks changed")
        path = args.output / (stage+".pt")
        save(path, {"model": model.state_dict(), "step": steps,
                    "producer_fingerprint": producer["fingerprint"]})
        stage_results[stage] = {"updates": steps, "unique_training_pairs_sampled": len(seen_ids),
                                "seconds": time.perf_counter()-stage_start,
                                "first_32_mean_loss": sum(x["loss"] for x in trace[:32])/32,
                                "last_32_mean_loss": sum(x["loss"] for x in trace[-32:])/32,
                                "checkpoint": str(path), "sha256": file_hash(path)}
        del opt
        vae.eval().requires_grad_(False)
        torch.cuda.empty_cache()
    # All optimization is finished before the fixed held-out comparison.
    final = evaluate(vae, net, testing, test_pack, gpt)
    write_json(args.output / "test_trained.json", final)
    vae.load_state_dict(torch.load(CODEC, map_location="cpu", weights_only=True)["model"], strict=True)
    net.load_state_dict(torch.load(GENERATOR, map_location="cpu", weights_only=True)["model"], strict=True)
    baseline = evaluate(vae, net, testing, test_pack, gpt)
    write_json(args.output / "test_original.json", baseline)
    verify(producer)
    without_outputs = lambda result: {k: v for k, v in result.items() if k != "outputs"}
    result = {"status": "completed", "hypothesis_id": producer["hypothesis_id"],
              "classification": "shared_summary_training_completed_quality_unqualified",
              "training_pairs": len(training), "test_pairs_available": len(rows(args.data / "native_test.jsonl")),
              "test_pairs_evaluated": len(testing), "stages": stage_results,
              "test_original": without_outputs(baseline), "test_trained": without_outputs(final),
              "elapsed_seconds": time.perf_counter()-started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "frozen_encoder_codebooks_sha256": frozen,
              "producer_fingerprint": producer["fingerprint"], "config": asdict(cfg),
              "production_accuracy": None, "speed_tested": False, "underpowered": True}
    write_json(args.output / "result.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "train"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    if args.command == "pilot":
        pilot(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
