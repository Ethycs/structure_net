"""Frozen concept rollouts and an independent trainable parallel word branch."""
from copy import deepcopy

import torch
import torch.nn.functional as F


def independent_word_branch(forecaster):
    """Copy learned head parameters while sharing only the frozen codec facade."""
    codec = forecaster.generator.vqvae_local[0]
    original = getattr(codec, "original", codec)
    if forecaster.training or any(p.requires_grad for p in forecaster.parameters()):
        raise ValueError("concept forecaster must be frozen and in eval mode")
    if original.training or any(p.requires_grad for p in original.parameters()):
        raise ValueError("original codec must remain frozen and in eval mode")
    branch = deepcopy(forecaster, {id(codec): codec, id(original): original})
    branch.requires_grad_(False)
    branch.source_head.source_norm.requires_grad_(True)
    branch.source_head.source_projection.requires_grad_(True)
    branch.generator.transformer.requires_grad_(True)
    branch.word_projection.requires_grad_(True)
    branch.word_decoder.requires_grad_(True)
    if branch.generator.vqvae_local[0] is not codec:
        raise ValueError("the two branches must use the identical frozen representation")
    if {id(p) for p in branch.parameters()} & {id(p) for p in forecaster.parameters()}:
        raise ValueError("word gradients must not change any forecaster parameter")
    return branch


@torch.no_grad()
def own_history(forecaster, features, source_mask, source_ids):
    """Eight coarse passes, using only source inputs and earlier predicted codes."""
    if forecaster.training or any(p.requires_grad for p in forecaster.parameters()):
        raise ValueError("rollouts require a fixed eval-mode forecaster")
    g = forecaster.generator
    codec = g.vqvae_local[0]
    source = forecaster.source_head.source_latents(features, source_mask, source_ids)
    prefix = g.transformer.wte(source)
    reconstruction = torch.zeros(len(source), g.levels[-1], codec.embed_dim,
        device=source.device, dtype=source.dtype)
    conditioning = g.build_l0_query_input(len(source), dtype=source.dtype)
    position, codes = 0, []
    for level, width in enumerate(g.levels[:-1]):
        hidden = forecaster._level_hidden(prefix, source_mask, conditioning, level, position)
        current = g.lm_heads[level](hidden).argmax(-1)
        codes.append(current)
        with torch.autocast(features.device.type, enabled=False):
            reconstruction, conditioning = codec.vq.get_next_autoregressive_input(
                level, reconstruction, codec.vq.embeddings[level](current))
        conditioning = conditioning.to(source.dtype)
        position += width
    return {"coarse_codes": torch.cat(codes, dim=1).detach(),
        "word_conditioning": conditioning.detach()}


@torch.no_grad()
def conditioning_from_codes(codec, coarse_codes):
    """Reconstruct final word queries from coarse codes; no finest labels exist."""
    widths = codec.vq.levels[:-1]
    if coarse_codes.ndim != 2 or coarse_codes.shape[1] != sum(widths):
        raise ValueError("all and only earlier-level codes are required")
    reconstruction = torch.zeros(len(coarse_codes), codec.vq.levels[-1], codec.embed_dim,
        device=coarse_codes.device, dtype=codec.vq.embeddings[0].weight.dtype)
    position = 0
    with torch.autocast(coarse_codes.device.type, enabled=False):
        for level, width in enumerate(widths):
            codes = coarse_codes[:, position:position+width]
            reconstruction, conditioning = codec.vq.get_next_autoregressive_input(
                level, reconstruction, codec.vq.embeddings[level](codes))
            position += width
    return conditioning.detach()


def word_logits(branch, features, source_mask, word_conditioning, source_ids):
    """Only the last level runs; source and fixed latent history are the inputs."""
    g = branch.generator
    codec = g.vqvae_local[0]
    if word_conditioning.shape != (len(features), g.levels[-1], codec.embed_dim):
        raise ValueError("complete word-level conditioning required")
    source = branch.source_head.source_latents(features, source_mask, source_ids)
    conditioning = word_conditioning.detach().to(source.dtype)
    hidden = branch._level_hidden(g.transformer.wte(source), source_mask, conditioning,
        len(g.levels)-1, sum(g.levels[:-1]))
    logits = branch._word_logits(hidden, conditioning)
    return logits if branch.word_vocab_limit is None else logits[..., :branch.word_vocab_limit]


def word_loss(logits, target_words, pad=50257):
    if (target_words.shape != logits.shape[:2]
            or not target_words.ne(pad).any(-1).all()):
        raise ValueError("complete nonempty target word frames required")
    value = F.cross_entropy(logits.float().flatten(0, 1), target_words.flatten(), ignore_index=pad)
    if not torch.isfinite(value):
        raise ValueError("nonfinite generated-history word loss")
    return value
