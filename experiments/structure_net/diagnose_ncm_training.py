"""Post-hoc, forward-only localization of the frozen-body NCM quality failure."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import time

import torch
import torch.nn.functional as F

from experiments.structure_net.ncm_frozen_body import (
    CODEC, GENERATOR, bridge, external_codes, external_logits, supplied_context, verify)
from experiments.structure_net.ncm_llama import (
    GPT2, INITIAL, VENDOR, autocast, decode_ids, final_latent, generator_input,
    load_ncm, tokenizers)
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_vocabulary import CachedVocabulary

PARENT = Path("data/experiments/ncm_frozen_body/20261022_train_v1")
PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-training-diagnosis-plan.md")
PILOT_FIX = Path("docs/07 - Status Reports/2026-10-22_ncm-diagnosis-cache-amendment.md")
CAMPAIGN_SHA = "b514e093a892cd2dd4cdf46cdd018d900d41371fa5b233f623c6764b98e406b3"
HYPOTHESIS = "ncm-frozen-body-diagnosis-v1"


def reconstruction_counts(prediction, target):
    """Keep PAD/EOS out of the content metric and expose every denominator."""
    equal = prediction.eq(target)
    content = target.ne(50257) & target.ne(50256)
    nonpad = target.ne(50257)
    return {name: {"correct": int((equal & mask).sum()), "total": int(mask.sum())}
            for name, mask in (("content", content), ("nonpad", nonpad),
                               ("canvas", torch.ones_like(equal)))}


def released_penalty_tokens(logits, allowance):
    """Released eval_shard.py formula, with fixed published XSum settings."""
    batch, length, vocab = logits.shape
    lg = logits.double()
    allow = (allowance[:vocab].to(lg.device).double() * length)[None, :]
    count = torch.zeros(batch, vocab, dtype=torch.float64, device=lg.device)
    out, history = [], []
    for t in range(length):
        step = lg[:, t].clone()
        step -= 2.0 * (count-allow).clamp_min(0)
        step += ((length-t)/length) * (allow-count).clamp_min(0)
        for j, previous in enumerate(reversed(history[-4:])):
            step.scatter_add_(1, previous[:, None], torch.full(
                (batch, 1), -5.0 * .5**j, dtype=torch.float64, device=lg.device))
        token = step.argmax(-1)
        out.append(token)
        history.append(token)
        count.scatter_add_(1, token[:, None], torch.ones(
            batch, 1, dtype=torch.float64, device=lg.device))
    return torch.stack(out, 1)


def code_metrics(logits, labels, levels):
    losses = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="none").view_as(labels)
    equal = logits.argmax(-1).eq(labels)
    offset, result = 0, []
    for level in levels:
        result.append({"positions": level, "ce": float(losses[:, offset:offset+level].mean()),
                       "correct": int(equal[:, offset:offset+level].sum()),
                       "total": int(equal[:, offset:offset+level].numel())})
        offset += level
    return {"ce": float(losses.mean()), "correct": int(equal.sum()),
            "total": equal.numel(), "per_level": result}


def released_text(ids, tokenizer):
    stop = ids.index(50256) if 50256 in ids else len(ids)
    return tokenizer.decode([t for t in ids[:stop] if t not in (50256, 50257)],
                            skip_special_tokens=True).strip()


def decoded(vae, codes, target, tokenizer, allowance):
    logits = vae.decoder(final_latent(vae, codes))
    if not torch.isfinite(logits).all():
        raise RuntimeError("nonfinite decoder logits")
    raw = logits.argmax(-1)
    penalized = released_penalty_tokens(logits, allowance)
    result = {}
    for name, ids in (("argmax", raw), ("released_penalties", penalized)):
        values = ids[0].tolist()
        text, termination = decode_ids(values, tokenizer)
        active = values[:next((i for i, t in enumerate(values) if t in (50256, 50257)), len(values))]
        result[name] = {"ids": values, "text": text, "termination": termination,
            "released_rendering_text": released_text(values, tokenizer),
            "immediate_repetitions": sum(a == b for a, b in zip(active, active[1:])),
            "reconstruction": reconstruction_counts(ids, target),
            "exact_target_text": text == decode_ids(target[0].tolist(), tokenizer)[0]}
    return result


def aggregate(records):
    result = {}
    for arm in ("hybrid", "native"):
        selected = [r for r in records if r.get("arm") == arm]
        result[arm] = {}
        for split in ("train", "test"):
            rs = [r for r in selected if r["split"] == split]
            result[arm][split] = {"n": len(rs), "teacher_ce": statistics.mean(r["teacher"]["ce"] for r in rs),
                "teacher_code_accuracy": sum(r["teacher"]["correct"] for r in rs)/sum(r["teacher"]["total"] for r in rs)}
            if split == "test":
                result[arm][split].update({
                    "free_conditioned_ce": statistics.mean(r["free_conditioned"]["ce"] for r in rs),
                    "free_code_accuracy": statistics.mean(r["free_code_accuracy"] for r in rs),
                    "oracle_first_level_code_accuracy": statistics.mean(r["oracle_first_level_code_accuracy"] for r in rs),
                    "generation_forward_agreement": statistics.mean(r["generation_forward_agreement"] for r in rs),
                    "saved_output_exact_ids": sum(r["saved_output_exact_ids"] for r in rs),
                    "per_level": [{"teacher_accuracy": sum(r["teacher"]["per_level"][i]["correct"] for r in rs)/
                                   sum(r["teacher"]["per_level"][i]["total"] for r in rs),
                                   "free_accuracy": statistics.mean(r["free_per_level_accuracy"][i] for r in rs)}
                                  for i in range(9)],
                    "decoded": {kind: {decoder: {
                        "exact_target_text": sum(r["decoded"][kind][decoder]["exact_target_text"] for r in rs),
                        "immediate_repetitions": sum(r["decoded"][kind][decoder]["immediate_repetitions"] for r in rs)}
                        for decoder in ("argmax", "released_penalties")}
                        for kind in ("teacher", "free", "oracle_first_level")}})
    oracle = [r for r in records if r.get("kind") == "oracle_codec"]
    result["oracle_codec"] = {decoder: {
        "n": len(oracle), "exact_target_text": sum(r["decoded"][decoder]["exact_target_text"] for r in oracle),
        **{metric: {"correct": sum(r["decoded"][decoder]["reconstruction"][metric]["correct"] for r in oracle),
                    "total": sum(r["decoded"][decoder]["reconstruction"][metric]["total"] for r in oracle)}
           for metric in ("content", "nonpad", "canvas")}} for decoder in ("argmax", "released_penalties")}
    return result


def run(args):
    if args.output.exists():
        raise ValueError("new append-only root required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if file_hash(PARENT / "campaign_results.json") != CAMPAIGN_SHA:
        raise ValueError("parent campaign changed")
    campaign = json.loads((PARENT / "campaign_results.json").read_text())
    for path, expected in campaign["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("parent artifact changed: " + path)
    verify(json.loads((PARENT / "producer.json").read_text()))
    selection = json.loads((PARENT / "selection.json").read_text())
    selection = {k: selection[k][:(2 if args.command == "pilot" else 64 if k == "test" else 32)]
                 for k in ("train", "test")}
    sources = [Path(__file__), PLAN, PILOT_FIX, Path("experiments/structure_net/ncm_frozen_body.py"),
        Path("experiments/structure_net/ncm_llama.py"), Path("experiments/structure_net/ncm_vocabulary.py"),
        Path("experiments/structure_net/ncm_memory_data.py")]
    sources += sorted(VENDOR.rglob("*.py"))
    inputs = [CODEC, GENERATOR, INITIAL / "config.yaml", INITIAL / "generator.pt",
        PARENT / "campaign_results.json", PARENT / "selection.json", PARENT / "producer.json",
        PARENT / "hybrid.pt", PARENT / "native.pt", PARENT / "train_codes.pt", PARENT / "test_codes.pt"]
    inputs += [PARENT / ("test_" + arm + ".json") for arm in ("native", "hybrid")]
    inputs += [p for p in GPT2.iterdir() if p.is_file()]
    inputs += [PARENT / (split + "_features") / (r["id"] + ".pt")
               for split, rs in selection.items() for r in rs]
    source_hashes = {str(p): file_hash(p) for p in sources}
    if args.command != "pilot":
        pilot = json.loads((args.pilot / "result.json").read_text())
        pp = json.loads((args.pilot / "producer.json").read_text())
        if pilot["status"] != "passed" or pp["source_hashes"] != source_hashes:
            raise ValueError("matching successful diagnostic pilot required")
    args.output.mkdir(parents=True)
    snapshots = args.output / "producer_source"
    snapshots.mkdir()
    for i, path in enumerate(sources):
        shutil.copyfile(path, snapshots / f"{i:03d}_{path.name}")
    producer = {"schema": "nal.ncm-diagnosis.v1", "hypothesis_id": HYPOTHESIS,
        "source_hashes": source_hashes, "input_hashes": {str(p): file_hash(p) for p in inputs},
        "created_utc": datetime.now(timezone.utc).isoformat(), "physical_gpu": 0,
        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "seed": 42,
        "config": {"batch": 1, "train_cases": len(selection["train"]), "test_cases": len(selection["test"]),
                   "train_updates": 0, "published_penalties": [2.0, 1.0, 5.0, 4, .5],
                   "autocast_cache_cleared_after_loading_each_arm": True},
        "exploratory": True, "parent_task_result_unchanged": True}
    producer["fingerprint"] = text_hash(json.dumps(producer, sort_keys=True))
    write_json(args.output / "producer.json", producer)
    started = time.perf_counter()
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    gpt, _ = tokenizers()
    gpt = CachedVocabulary(gpt)
    vae, net, _ = load_ncm(codec=CODEC, generator=GENERATOR)
    allowance = torch.load(INITIAL / "generator.pt", map_location="cpu", weights_only=True)["inference"]["token_unigram"].cuda()
    packed = {split: torch.load(PARENT / (split+"_codes.pt"), weights_only=True) for split in selection}
    records = []
    with torch.inference_mode(), autocast():
        for i, case in enumerate(selection["test"]):
            codes, target = (packed["test"][name][i:i+1].cuda() for name in ("codes", "targets"))
            records.append({"kind": "oracle_codec", "id": case["id"], "target": case["target"],
                            "decoded": decoded(vae, codes, target, gpt, allowance)})
        for arm in ("hybrid", "native"):
            state = torch.load(PARENT / (arm+".pt"), map_location="cpu", weights_only=True)
            net.load_state_dict(state["generator"], strict=True)
            adapter = bridge().cuda().eval() if arm == "hybrid" else None
            if adapter is not None:
                adapter.load_state_dict(state["adapter"], strict=True)
            del state
            # The surrounding autocast scope spans multiple loaded checkpoints.
            # Discard cached BF16 copies when FP32 parameter storage is reused.
            torch.clear_autocast_cache()
            net.eval()
            saved = json.loads((PARENT / ("test_"+arm+".json")).read_text())["outputs"]
            for split, cases in selection.items():
                for i, case in enumerate(cases):
                    codes, target, prefix = (packed[split][name][i:i+1].cuda() for name in ("codes", "targets", "prefixes"))
                    if arm == "hybrid":
                        feat = torch.load(PARENT / (split+"_features") / (case["id"]+".pt"), weights_only=True)
                        context = adapter(feat["hidden"][None].cuda().float())
                        mask = feat["pad_mask"][None].cuda()
                    def logits_for(conditioning):
                        if adapter is not None:
                            return external_logits(vae, net, context, mask, conditioning)
                        return net(generator_input(vae, net, prefix, conditioning), prefix_pad_mask=prefix.eq(50257))
                    teacher = logits_for(codes)
                    row = {"arm": arm, "split": split, "id": case["id"], "target": case["target"],
                           "teacher": code_metrics(teacher, codes, vae.vq.levels)}
                    if split == "test":
                        generated = external_codes(net, context, mask) if adapter is not None else net.generate(
                            None, prefix, top_k=1, top_p=0., cfg_scale=0., use_kv_cache=False)
                        if adapter is not None:
                            with supplied_context(net, context):
                                seeded = net.generate(codes[:, :vae.vq.levels[0]], torch.where(mask,50257,0),
                                    top_k=1, top_p=0., cfg_scale=0., use_kv_cache=False)
                        else:
                            seeded = net.generate(codes[:, :vae.vq.levels[0]], prefix,
                                top_k=1, top_p=0., cfg_scale=0., use_kv_cache=False)
                        own = logits_for(generated)
                        correct = generated.eq(codes)
                        offset, per_level = 0, []
                        for length in vae.vq.levels:
                            per_level.append(float(correct[:, offset:offset+length].float().mean()))
                            offset += length
                        row.update({"free_conditioned": code_metrics(own, codes, vae.vq.levels),
                            "free_code_accuracy": float(correct.float().mean()),
                            "free_per_level_accuracy": per_level,
                            "oracle_first_level_code_accuracy": float(seeded[:, vae.vq.levels[0]:].eq(
                                codes[:, vae.vq.levels[0]:]).float().mean()),
                            "generation_forward_agreement": float(own.argmax(-1).eq(generated).float().mean()),
                            "decoded": {kind: decoded(vae, value, target, gpt, allowance)
                                for kind, value in (("teacher", teacher.argmax(-1)), ("free", generated),
                                                   ("oracle_first_level", seeded))}})
                        row["saved_output_exact_ids"] = row["decoded"]["free"]["argmax"]["ids"] == saved[i]["native_output_ids"]
                    records.append(row)
                    if (i+1) % 16 == 0:
                        print(json.dumps({"arm": arm, "split": split, "completed": i+1}), flush=True)
    write_json(args.output / "records.json", records)
    tested = [r for r in records if r.get("split") == "test"]
    if args.command == "pilot" and not all(r["saved_output_exact_ids"] for r in tested):
        raise RuntimeError("diagnostic pilot differs from saved inference outputs")
    verify(producer)
    result = {"hypothesis_id": HYPOTHESIS, "status": "passed" if args.command == "pilot" else "completed_exploratory_diagnosis",
        "label": "systems_lifecycle_only_not_quality_evidence" if args.command == "pilot" else "posthoc_no_new_task_accuracy",
        "producer_fingerprint": producer["fingerprint"], "summary": aggregate(records),
        "elapsed_seconds": time.perf_counter()-started, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "records_sha256": file_hash(args.output / "records.json"), "trained": False}
    write_json(args.output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "run"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", type=Path)
    run(parser.parse_args())
