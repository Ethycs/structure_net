"""Paired text decoding from one fixed generated NCM hierarchy per passage."""
import argparse
import gc
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_decode_policy import pick_tokens, render_author
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_llama import INITIAL, decode_ids, load_ncm, tokenizers
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import PREP, ROOT, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.ncm_training_recipe import cumulative_latent
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.train_ncm_decoder_polish import frozen_hash
from experiments.structure_net.train_ncm_memory import save

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-decoding-policy-plan.md")
ORIGINAL = ROOT / "20261022_codec_training_v2/model.pt"
CODEC = ROOT / "20261022_decoder_train_v1/model.pt"
GENERATOR = ROOT / "20261022_noisy_native_v1/model.pt"
TEACHER = ROOT / "20261022_teacher_v1/train.json"


def setup(pilot):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("physicalGPU2 must be selected before torch startup")
    if not torch.cuda.is_available() or "2060 SUPER" not in torch.cuda.get_device_name():
        raise ValueError("planned CUDA device required")
    source = TEACHER if pilot else PREP / "development_sources.json"
    cases = q.read(source)[:4] if pilot else q.read(source)
    verify_public(cases)
    root = ROOT / ("20261022_decode_pilot_v1" if pilot else "20261022_decode_outputs_v1")
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "paired_decoder_policy_outputs",
        [source, ORIGINAL, CODEC, GENERATOR, TEACHER, INITIAL / "generator.pt", INITIAL / "config.yaml"],
        {"batch": 4, "seed": 42, "physical_gpu": 2, "weights": "FP32", "forward": "FP16",
         "latent_reconstruction": "FP32", "policies": ["local", "paper", "domain"],
         "penalties": {"frequency": 2., "deficit": 1., "adjacency": [5., 4., .5]},
         "generation": {"temperature": 1., "top_k": 1, "top_p": 0., "cfg_scale": 0.}})
    p["hypothesis_id"] = "ncm-decoding-policy-v1"
    p["execution_environment"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"]}
    for path in (Path(__file__), PLAN, Path("experiments/structure_net/ncm_decode_policy.py"),
                 Path("experiments/structure_net/train_ncm_decoder_polish.py")):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("decode_"+path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    if not pilot:
        pr = ROOT / "20261022_decode_pilot_v1"
        if q.read(pr / "result.json")["status"] != "passed" or q.read(pr / "producer.json")["source_hashes"] != p["source_hashes"]:
            raise ValueError("matching CUDA decoding pilot required")
    return root, cases, p


@torch.inference_mode()
def main(pilot):
    root, cases, p = setup(pilot)
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
    gpt, _ = tokenizers(); gpt = CachedVocabulary(gpt)
    vae, net, _ = load_ncm(codec=ORIGINAL, generator=GENERATOR)
    old_language = frozen_hash(vae)
    state = torch.load(GENERATOR, map_location="cpu", weights_only=True)
    if state["codec_sha256"] != file_hash(ORIGINAL):
        raise ValueError("original generator/codec pair mismatch")
    del state; gc.collect()
    replacement = torch.load(CODEC, map_location="cpu", weights_only=True)
    vae.load_state_dict(replacement["model"], strict=True)
    del replacement; gc.collect(); torch.clear_autocast_cache()
    if frozen_hash(vae) != old_language:
        raise ValueError("new decoder changed non-decoder state")
    vae.eval().requires_grad_(False); net.eval().requires_grad_(False)
    frozen_vae, frozen_net = module_hash(vae), module_hash(net)
    released = torch.load(INITIAL / "generator.pt", map_location="cpu", weights_only=True)
    paper_prior = released["inference"]["token_unigram"].double()
    del released; gc.collect()
    target_ids = torch.tensor([row["native_target_ids"] for row in q.read(TEACHER)])
    domain_prior = torch.bincount(target_ids.flatten(), minlength=50304).double()
    domain_prior /= domain_prior.sum()
    save(root / "priors.pt", {"paper": paper_prior, "domain": domain_prior})
    outputs = {name: [] for name in ("local", "paper", "domain", "raw_author_diagnostic")}
    saved_codes, saved_logits = [], []
    reload_equal = None
    for start in range(0, len(cases), 4):
        chunk = cases[start:start+4]
        native = [gpt.encode(row["source"], add_special_tokens=False) for row in chunk]
        if any(len(ids) > 512 for ids in native):
            raise ValueError("source truncation prohibited")
        prefix = torch.tensor([ids+[50257]*(512-len(ids)) for ids in native], device="cuda")
        before_rng = torch.cuda.get_rng_state()
        with torch.autocast("cuda", dtype=torch.float16):
            codes = net.generate(None, prefix, temperature=1., top_k=1, top_p=0., cfg_scale=0., use_kv_cache=False)
        with torch.autocast("cuda", enabled=False):
            latent = cumulative_latent(vae, codes)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = vae.decoder(latent)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("nonfinite decode logits")
        if pilot:
            save(root / "reload.pt", {"model": net.state_dict(), "codec": vae.state_dict(),
                "codec_sha256": file_hash(CODEC), "producer_fingerprint": p["fingerprint"]})
            after_rng = torch.cuda.get_rng_state()
            loaded = torch.load(root / "reload.pt", map_location="cpu", weights_only=True)
            net.load_state_dict(loaded["model"], strict=True); vae.load_state_dict(loaded["codec"], strict=True)
            del loaded; gc.collect(); torch.clear_autocast_cache()
            torch.cuda.set_rng_state(before_rng)
            with torch.autocast("cuda", dtype=torch.float16):
                restored_codes = net.generate(None, prefix, temperature=1., top_k=1, top_p=0., cfg_scale=0., use_kv_cache=False)
            with torch.autocast("cuda", enabled=False):
                restored_latent = cumulative_latent(vae, restored_codes)
            with torch.autocast("cuda", dtype=torch.float16):
                restored_logits = vae.decoder(restored_latent)
            reload_equal = torch.equal(codes, restored_codes) and torch.equal(logits, restored_logits)
            if not reload_equal:
                raise ValueError("decoding changed after exact reload/RNG restore")
            torch.cuda.set_rng_state(after_rng)
        cpu_logits = logits.cpu()
        saved_codes.append(codes.cpu()); saved_logits.append(cpu_logits)
        for name in ("local", "paper", "domain", "raw_author_diagnostic"):
            prior = paper_prior if name == "paper" else domain_prior
            ids = (cpu_logits.argmax(-1) if name in ("local", "raw_author_diagnostic") else
                pick_tokens(cpu_logits, frequency=2., deficit=1., adjacency=(5., 4., .5), allowance=prior)).tolist()
            for case, row in zip(chunk, ids):
                text = decode_ids(row, gpt)[0] if name == "local" else render_author(row, gpt)
                outputs[name].append({**case, "text": text, "native_output_ids": row,
                    "decoder_policy": name, "forward_precision": "FP16", "source_truncated": False})
        print(json.dumps({"generated": start+len(chunk), "total": len(cases)}), flush=True)
    if module_hash(vae) != frozen_vae or module_hash(net) != frozen_net:
        raise ValueError("inference changed model weights")
    for name, records in outputs.items():
        write_json(root / (name+".json"), records)
    save(root / "fixed_codes_logits.pt", {"codes": torch.cat(saved_codes), "logits": torch.cat(saved_logits)})
    verify(p)
    result = {"status": "passed" if pilot else "generated_pending_task_grading", "cases": len(cases),
        "producer_fingerprint": p["fingerprint"], "reload_equal": reload_equal,
        "nondecoder_unchanged": True, "source_generator_sha256": file_hash(GENERATOR),
        "original_codec_sha256": file_hash(ORIGINAL), "new_codec_sha256": file_hash(CODEC),
        "policies_share_exact_logits": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "task_accuracy": None, "final_test_opened": False, "speed_tested": False,
        "underpowered": True, "confirmed": False}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "generate"])
    main(parser.parse_args().command == "pilot")
