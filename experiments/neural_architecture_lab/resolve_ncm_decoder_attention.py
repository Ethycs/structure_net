"""Resolve the actual codec attention class, preserving the earlier mistaken note."""
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_broad_learned_codec as core
from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT = core.ROOT / "20261023_decoder_attention_resolution_v1"
AUDIT = core.ROOT / "20261023_repository_training_audit_v2"


def main():
    if ROOT.exists():
        raise ValueError("fresh append-only source-resolution root required")
    sources = [Path(__file__), core.VENDOR / "models/vae.py", core.VENDOR / "models/common.py",
        core.VENDOR / "models/vqvae.py", AUDIT / "decoder_attention_supplement.json",
        AUDIT / "sources/HazyResearch--ncm/models/vae.py", core.INITIAL / "tokenizer.pt",
        core.INITIAL / "config.yaml"]
    if file_hash(sources[1]) != file_hash(sources[5]):
        raise ValueError("local codec source differs from pinned official source")
    ROOT.mkdir()
    producer = {"schema": "nal.ncm-decoder-attention-resolution.v1", "CPU_only_source_mechanism_probe": True,
        "seed": 1142, "source_hashes": {str(p): file_hash(p) for p in sources},
        "corrects": str(AUDIT / "decoder_attention_supplement.json")}
    write_json(ROOT / "producer.json", producer)
    codec = core.load_codec(device="cpu").eval().requires_grad_(False)
    before = core.module_hash(codec)
    original = F.scaled_dot_product_attention
    calls = []
    def record(*args, **kwargs):
        calls.append({"is_causal": bool(kwargs.get("is_causal", False)),
                      "attn_mask_is_none": kwargs.get("attn_mask") is None})
        return original(*args, **kwargs)
    generator = torch.Generator().manual_seed(1142)
    latent = torch.randn(1, 48, 384, generator=generator)
    changed = latent.clone()
    changed[:, -1] += 5 * torch.randn(1, 384, generator=generator)
    with torch.no_grad(), patch.object(F, "scaled_dot_product_attention", record):
        base = codec.decoder(latent)
        perturbed = codec.decoder(changed)
    difference = float((base[:, 0] - perturbed[:, 0]).abs().max())
    actual_class = type(codec.decoder.middle_block.attn)
    if (actual_class.__module__ != "models.vae" or difference <= 1e-5 or not calls
            or any(r["is_causal"] or not r["attn_mask_is_none"] for r in calls)
            or core.module_hash(codec) != before):
        raise ValueError("actual decoder mechanism differs from inspected noncausal path")
    result = {"status": "source_and_runtime_resolution_passed", "actual_attention_class": actual_class.__module__ + "." + actual_class.__name__,
        "decoder_attention": "bidirectional_within_original_codec_block", "SDPA_calls": calls,
        "future_latent_changes_first_position_logits_max_abs": difference, "model_state_unchanged": True,
        "pinned_local_vae_source_identical": True, "prior_interpretation_incorrect": True,
        "explanation": "Decoder instantiates Block defined in models/vae.py, whose SelfAttention calls SDPA without a causal flag or mask. models/common.py is not the decoder's attention class. Prior common.py-based causal interpretation is retracted; all completed numerical experiments used the correct actual modules unchanged.",
        "source_prediction": False, "task_accuracy": None, "quality_evidence": False}
    for path, expected in producer["source_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("source changed during resolution")
    write_json(ROOT / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
