"""Isolate native-source and frozen-parent contributions without retraining."""
import argparse
import gc
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_coarse_curriculum import GlobalResidual, enable_coarse_ar
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_global_context import last_valid
from experiments.structure_net.ncm_llama import encode_codes, load_ncm, tokenizers
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import PARENT, PREP, ROOT, verify
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.train_ncm_full_curriculum import measure

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-body-contribution-plan.md")
TRAIN = ROOT / "20261022_full_train_v1"
CODEC = ROOT / "20261022_decoder_train_v1/model.pt"
DATA = ROOT / "20261022_teacher_v1/train.json"
ARMS = ("matched", "shifted_body", "zero_body", "native_only", "shifted_native", "both_shifted")


class NativeOnly(torch.nn.Module):
    def forward(self, native, body_last, pad_mask):
        return native


def validate_scores(module, args, scores):
    if bool((torch.isnan(scores) | torch.isposinf(scores)).any()) or not bool(torch.isfinite(scores).any(-1).all()):
        raise ValueError("invalid generator or decoder logits")


@torch.inference_mode()
def main(pilot):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("physicalGPU0 must be selected before torch startup")
    if not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name():
        raise ValueError("planned CUDA device required")
    if torch.cuda.mem_get_info()[0] < 5_500_000_000:
        raise ValueError("wait for the original parent worker to exit")
    all_cases = q.read(DATA)
    cases = all_cases[:4] if pilot else sorted(all_cases, key=lambda r: text_hash("ncm-body-ablation-v1:"+r["id"]))[:128]
    # Anchor cases are independent of the hash-ordered measurement sample.
    needed = {r["id"]: r for r in [*cases, *all_cases[:4]]}
    manifest = q.read(PREP / "producer.json")["input_hashes"]
    inputs = [DATA, CODEC, TRAIN / "step5120.pt", TRAIN / "result.json", TRAIN / "outputs_step5120.json"]
    features = {}
    for case in needed.values():
        path = PARENT / "train_features" / (case["id"]+".pt")
        if file_hash(path) != manifest[str(path)]:
            raise ValueError("training feature identity changed")
        inputs.append(path)
        value = torch.load(path, weights_only=True)
        features[case["id"]] = last_valid(value["hidden"][None], value["pad_mask"][None])[0].clone()
    root = ROOT / ("20261022_body_ablation_pilot_v1" if pilot else "20261022_body_ablation_measure_v1")
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "training_source_channel_ablation",
        inputs, {"cases": len(cases), "batch": 4, "seed": 42, "precision": "BF16", "arms": ARMS})
    for path in (Path(__file__), PLAN, Path("experiments/structure_net/train_ncm_full_curriculum.py"),
                 Path("experiments/structure_net/ncm_coarse_curriculum.py"),
                 Path("experiments/structure_net/ncm_global_context.py")):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("ablation_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-body-contribution-v1"
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    if not pilot:
        pr = ROOT / "20261022_body_ablation_pilot_v1"
        if q.read(pr / "result.json")["status"] != "passed" or q.read(pr / "producer.json")["source_hashes"] != p["source_hashes"]:
            raise ValueError("matching real CUDA pilot required")
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
    vae, net, _ = load_ncm(codec=CODEC, generator=TRAIN / "step5120.pt", adaptation=True)
    enable_coarse_ar(net)
    net.eval().requires_grad_(False)
    net.register_forward_hook(validate_scores)
    vae.decoder.register_forward_hook(validate_scores)
    adapter = GlobalResidual().cuda().eval().requires_grad_(False)
    checkpoint = torch.load(TRAIN / "step5120.pt", map_location="cpu", weights_only=True)
    if checkpoint["codec_sha256"] != file_hash(CODEC):
        raise ValueError("generator/codec pair changed")
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    del checkpoint; gc.collect()
    gpt, _ = tokenizers(); gpt = CachedVocabulary(gpt)

    def tensors(rows):
        ids = [gpt.encode(r["source"], add_special_tokens=False) for r in rows]
        if any(len(x) > 512 for x in ids):
            raise ValueError("source truncation prohibited")
        prefix = torch.tensor([x+[50257]*(512-len(x)) for x in ids])
        last = torch.stack([features[r["id"]] for r in rows])
        targets = torch.tensor([r["native_target_ids"] for r in rows])
        labels = torch.cat([encode_codes(vae, targets[i:i+32].cuda()).cpu() for i in range(0, len(rows), 32)])
        return prefix, last, targets, labels

    identity = {"codec": module_hash(vae), "generator": module_hash(net), "adapter": module_hash(adapter)}
    anchor = all_cases[:4]
    _, output = measure(vae, net, adapter, *tensors(anchor), anchor, gpt)
    original = q.read(TRAIN / "outputs_step5120.json")[:4]
    if [(r["predicted_codes"], r["native_output_ids"]) for r in output] != [
            (r["predicted_codes"], r["native_output_ids"]) for r in original]:
        raise ValueError("original BF16 training anchor changed")
    prefix, last, targets, labels = tensors(cases)
    metrics = {}
    for arm in ARMS:
        px = prefix.roll(1, 0) if arm in ("shifted_native", "both_shifted") else prefix
        h = last.roll(1, 0) if arm in ("shifted_body", "both_shifted") else last
        if arm == "zero_body":
            h = torch.zeros_like(last)
        bridge = NativeOnly() if arm == "native_only" else adapter
        metrics[arm], outputs = measure(vae, net, bridge, px, h, targets, labels, cases, gpt)
        if any(r["termination"] == "invalid_token" for r in outputs):
            raise ValueError("invalid decoded vocabulary")
        write_json(root / (arm+".json"), outputs)
        print(json.dumps({"arm": arm, **metrics[arm]}), flush=True)
    if identity != {"codec": module_hash(vae), "generator": module_hash(net), "adapter": module_hash(adapter)}:
        raise ValueError("frozen model changed")
    checkpoint = torch.load(TRAIN / "step5120.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint["model"], strict=True); enable_coarse_ar(net)
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    del checkpoint; gc.collect(); torch.clear_autocast_cache()
    _, repeated = measure(vae, net, adapter, *tensors(anchor), anchor, gpt)
    if [(r["predicted_codes"], r["native_output_ids"]) for r in repeated] != [
            (r["predicted_codes"], r["native_output_ids"]) for r in original]:
        raise ValueError("exact reload changed generation")
    verify(p)
    result = {"status": "passed" if pilot else "completed_training_channel_diagnostic", "cases": len(cases),
        "metrics": metrics, "anchor_exact": True, "reload_exact": True, "frozen_hashes": identity,
        "trained": False, "parent_body_loaded": False, "task_accuracy": None, "final_test_opened": False,
        "speed_tested": False, "underpowered": True, "confirmed": False,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "measure"])
    main(parser.parse_args().command == "pilot")
