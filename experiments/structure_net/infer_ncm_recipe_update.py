"""Use an updated NCM with its explicitly paired codec and frozen Llama body."""
import argparse
import gc
import json
from pathlib import Path

import torch

from experiments.structure_net.ncm_frozen_body import (
    Config, MODEL_PATH, bridge, external_codes, parent_inputs, prompt_ids)
from experiments.structure_net.ncm_llama import autocast, decode_ids, load_ncm, tokenizers
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_training_recipe import cumulative_latent
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.train_ncm_memory import frozen_digest


def paired_codec(state):
    if any(key not in state for key in ("codec_checkpoint", "codec_sha256", "levels", "vocabularies",
                                        "frozen_encoder_codebooks_sha256")):
        raise ValueError("checkpoint lacks explicit matching codec identity")
    codec = Path(state["codec_checkpoint"])
    if not codec.is_file() or file_hash(codec) != state["codec_sha256"]:
        raise ValueError("paired codec missing or changed")
    return codec


def main(args):
    if args.output.exists() or not torch.cuda.is_available():
        raise ValueError("fresh output and CUDA required")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    codec = paired_codec(state)
    gpt, tokenizer = tokenizers()
    gpt = CachedVocabulary(gpt)
    source = args.source.read_text()
    ids = prompt_ids(tokenizer, source)
    if not source.strip() or len(ids) > Config().prefix:
        raise ValueError("nonempty source fitting the 512-token rendered prompt required; no truncation")
    from transformers import AutoModelForCausalLM
    torch.cuda.reset_peak_memory_stats()
    parent = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    inputs = parent_inputs([{"parent_ids": ids}], tokenizer, "cuda")
    with torch.inference_mode():
        hidden = parent.model(**inputs, use_cache=False).last_hidden_state.detach().cpu()
    del parent
    gc.collect()
    torch.cuda.empty_cache()
    vae, net, _ = load_ncm(codec=codec)
    if (list(vae.vq.levels) != state["levels"] or list(vae.vq.vqvae_vocab_sizes) != state["vocabularies"]
            or frozen_digest(vae) != state["frozen_encoder_codebooks_sha256"]):
        raise ValueError("codec architecture/codebook identity differs from generator")
    net.load_state_dict(state["generator"], strict=True)
    net.eval().requires_grad_(False)
    adapter = bridge().cuda().eval().requires_grad_(False)
    adapter.load_state_dict(state["adapter"], strict=True)
    with torch.inference_mode(), autocast():
        context = adapter(hidden.cuda().float())
        codes = external_codes(net, context, inputs["attention_mask"].eq(0))
        logits = vae.decoder(cumulative_latent(vae, codes))
    if not torch.isfinite(logits).all():
        raise RuntimeError("nonfinite inference output")
    output_ids = logits.argmax(-1)[0].tolist()
    text, termination = decode_ids(output_ids, gpt)
    result = {"text": text, "termination": termination, "native_output_ids": output_ids,
        "source_sha256": file_hash(args.source), "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_hash(args.checkpoint), "codec": str(codec),
        "codec_sha256": file_hash(codec), "paired_codec_verified": True,
        "parent_model": str(MODEL_PATH), "parent_prompt_tokens": len(ids),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "producer_sha256": file_hash(__file__), "speed_tested": False,
        "scope": "fresh single-example body-to-head integration; not quality or speed evidence"}
    write_json(args.output, result)
    print(json.dumps({"text": text, "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
