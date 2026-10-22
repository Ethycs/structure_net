"""Generate a summary through a frozen Llama body and an attached NCM module."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net.ncm_frozen_body import (
    CODEC, Config, MODEL_PATH, bridge, external_codes, parent_inputs, prompt_ids)
from experiments.structure_net.ncm_llama import autocast, decode_ids, final_latent, load_ncm, tokenizers
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_vocabulary import CachedVocabulary


def main(args):
    if args.output.exists():
        raise ValueError("fresh inference output required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no CPU fallback")
    source = args.source.read_text()
    gpt, tokenizer = tokenizers()
    gpt = CachedVocabulary(gpt)
    ids = prompt_ids(tokenizer, source)
    if not source.strip() or len(ids) > Config().prefix:
        raise ValueError("nonempty source fitting the 512-token rendered parent prompt required; no truncation")
    from transformers import AutoModelForCausalLM
    parent = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    inputs = parent_inputs([{"parent_ids": ids}], tokenizer, "cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        hidden = parent.model(**inputs, use_cache=False).last_hidden_state
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vae, net, _ = load_ncm(codec=CODEC, device="cuda")
    net.load_state_dict(state["generator"], strict=True)
    net.eval().requires_grad_(False)
    adapter = bridge().cuda().eval().requires_grad_(False)
    adapter.load_state_dict(state["adapter"], strict=True)
    with torch.inference_mode(), autocast():
        context = adapter(hidden.float())
        codes = external_codes(net, context, inputs["attention_mask"].eq(0))
        logits = vae.decoder(final_latent(vae, codes))
    if not torch.isfinite(logits).all():
        raise RuntimeError("nonfinite output")
    output_ids = logits.argmax(-1)[0].tolist()
    text, termination = decode_ids(output_ids, gpt)
    result = {"text": text, "termination": termination, "native_output_ids": output_ids,
              "source": str(args.source), "source_sha256": file_hash(args.source),
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_hash(args.checkpoint),
              "codec": str(CODEC), "codec_sha256": file_hash(CODEC),
              "parent_model": str(MODEL_PATH), "parent_prompt_tokens": len(ids),
              "generator_steps": len(vae.vq.levels), "source_truncated": False,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "producer_sha256": file_hash(__file__), "timing_benchmark": False}
    write_json(args.output, result)
    print(json.dumps({"text": text, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
