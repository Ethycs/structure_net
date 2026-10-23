"""Attach a released-style concept generator to every frozen-parent source token.

The new head owns its projection and generator weights. The parent's features
are detached; the learned codec stays frozen. No target token history enters
source encoding or the current-level concept query.
"""
import math

import torch
from torch import nn


@torch.no_grad()
def frozen_parent_features(parent, input_ids, source_mask):
    if parent.training or any(p.requires_grad for p in parent.parameters()):
        raise ValueError("source parent must be explicitly frozen and in eval mode")
    if (input_ids.ndim != 2 or source_mask.shape != input_ids.shape
            or not ((source_mask == 0) | (source_mask == 1)).all()
            or not source_mask.bool().any(-1).all()):
        raise ValueError("complete nonempty source frames required")
    body = parent.model if hasattr(parent, "model") else parent
    positions = (source_mask.long().cumsum(-1) - 1).clamp_min(0)
    output = body(input_ids=input_ids, attention_mask=source_mask, position_ids=positions,
        use_cache=False, return_dict=True)
    return output.last_hidden_state.detach()


def validate_generator(generator):
    if (not generator.query_level0 or generator.l0_ar or generator.predict_level0 or generator.use_ctx_encoder
            or generator.use_flex_attention or generator.use_post_upsample_conv
            or not generator.use_rope or not generator.config.use_level_encoding
            or generator.config.attn_pattern != "block_diagonal"):
        raise ValueError("released-style parallel level queries, RoPE and level masks required")


def migrate_generator_state(generator, released_state, reference_codec):
    """Retain learned generator weights while rebuilding only sequence geometry."""
    validate_generator(generator)
    codec = generator.vqvae_local[0]
    original = getattr(codec, "original", codec)
    reference_state = reference_codec.state_dict()
    target_state = original.state_dict()
    if set(reference_state) != set(target_state) or any(
            not torch.equal(v, target_state[k]) for k, v in reference_state.items() if not k.startswith("decoder.")):
        raise ValueError("released generator code semantics require unchanged reference representation")
    if any(not torch.equal(head.E, codec.vq.embeddings[i].weight)
            for i,head in enumerate(generator.lm_heads)):
        raise ValueError("generator output heads contain stale codebook copies")
    levels = tuple(generator.levels)
    old_levels = tuple(reference_codec.vq.levels)
    if (len(levels) != len(old_levels) or levels[0] % old_levels[0]
            or any(a != levels[0] // old_levels[0] * b for a,b in zip(levels, old_levels))):
        raise ValueError("only independent-block expansion of all levels is supported")
    blocks = levels[0] // old_levels[0]
    current = generator.state_dict()
    if set(current) != set(released_state):
        raise ValueError("released generator architecture keys differ")
    expected_levels = torch.cat([torch.full((width,), i, dtype=torch.long) for i,width in enumerate(old_levels)])[None]
    if not torch.equal(released_state["lvl_1L"].cpu(), expected_levels):
        raise ValueError("released generator level boundaries differ")
    prepared = {}
    rebuilt = ("attn_mask_for_masking", "lvl_1L")
    for name, target in current.items():
        value = released_state[name]
        if name in rebuilt:
            prepared[name] = target
        elif name == "l0_queries":
            expanded = value.repeat(1, blocks, 1)
            if expanded.shape != target.shape:
                raise ValueError("released root query geometry differs")
            prepared[name] = expanded
        else:
            if value.shape != target.shape:
                raise ValueError("unapproved learned-weight shape change: " + name)
            prepared[name] = value
    generator.load_state_dict(prepared, strict=True)
    return {"all_generator_weights_from_release": True, "independent_blocks": blocks,
        "root_queries_repeated_from_release": True, "geometry_buffers_rebuilt": list(rebuilt),
        "reference_representation_exact": True, "new_prefix_len": generator.prefix_len}


def teacher_concept_logits(generator, prefix_latent, source_mask, codes):
    validate_generator(generator)
    codec = generator.vqvae_local[0]
    conditioning = torch.cat((generator.build_l0_query_input(len(codes), dtype=prefix_latent.dtype),
        codec.vq.idx_to_ncm_input(codes).to(prefix_latent.dtype)), dim=1)
    return generator(torch.cat((prefix_latent, conditioning), dim=1), prefix_pad_mask=~source_mask.bool())


@torch.no_grad()
def greedy_concepts(generator, prefix_latent, source_mask):
    """Same eager greedy level computation as the reference, with supplied features."""
    validate_generator(generator)
    if generator.training:
        raise ValueError("deterministic concept generation requires eval mode")
    codec = generator.vqvae_local[0]
    batch, prefix, _ = prefix_latent.shape
    if (prefix != generator.prefix_len or source_mask.shape != (batch, prefix)
            or not source_mask.bool().any(-1).all()):
        raise ValueError("source frames must match the configured complete prefix")
    projected = generator.transformer.wte(prefix_latent)
    pad_bias = generator._prefix_pad_bias(~source_mask.bool(), prefix, projected.dtype, projected.device)
    root = generator.transformer.wte(generator.build_l0_query_input(batch, dtype=prefix_latent.dtype))
    root = root + generator.transformer.wle(generator.lvl_1L[:, :generator.levels[0]])
    hidden = torch.cat((projected, root), dim=1)
    size = hidden.shape[1]
    mask = torch.zeros(1, 1, size, size, dtype=hidden.dtype, device=hidden.device)
    mask[:, :, :prefix, prefix:] = -torch.inf
    mask = mask + torch.nn.functional.pad(pad_bias, (0, generator.levels[0]))
    rope = (generator.rope_cos_gen_0, generator.rope_sin_gen_0)
    for block in generator.transformer.blocks:
        hidden = block(hidden, attn_mask=mask, rope_cos_sin=rope)
    hidden = generator.transformer.ln_f(hidden)
    current = generator.lm_heads[0](hidden[:, prefix:]).argmax(-1)
    codes = [current]
    reconstruction = torch.zeros(batch, generator.levels[-1], codec.embed_dim,
        dtype=prefix_latent.dtype, device=prefix_latent.device)
    position = generator.levels[0]
    for level in range(len(generator.levels) - 1):
        reconstruction, conditioning = codec.vq.get_next_autoregressive_input(
            level, reconstruction, codec.vq.embeddings[level](current))
        query = generator.transformer.wte(conditioning.to(prefix_latent.dtype))
        width = generator.levels[level + 1]
        query = query + generator.transformer.wle(generator.lvl_1L[:, position:position+width])
        current = generator._decode_level_logits(query, projected, level, pad_bias, hidx=level+1).argmax(-1)
        codes.append(current)
        position += width
    return torch.cat(codes, dim=1)


class FrozenSourceConceptHead(nn.Module):
    def __init__(self, generator, parent_hidden_size, residual_prefix=False):
        super().__init__()
        validate_generator(generator)
        codec = generator.vqvae_local[0]
        original = getattr(codec, "original", codec)
        if original.training or any(p.requires_grad for p in original.parameters()):
            raise ValueError("concept targets and text decoder must be explicitly frozen")
        self.generator = generator
        self.residual_prefix = bool(residual_prefix)
        self.source_norm = nn.LayerNorm(parent_hidden_size)
        self.source_projection = nn.Linear(parent_hidden_size, codec.embed_dim)
        if self.residual_prefix:
            nn.init.zeros_(self.source_projection.weight)
        else:
            nn.init.normal_(self.source_projection.weight, std=1 / math.sqrt(parent_hidden_size))
        nn.init.zeros_(self.source_projection.bias)

    def source_latents(self, features, source_mask, source_ids=None):
        if (features.ndim != 3 or features.shape[1] != self.generator.prefix_len
                or features.shape[-1] != self.source_projection.in_features
                or source_mask.shape != features.shape[:2]
                or not ((source_mask == 0) | (source_mask == 1)).all()
                or not source_mask.bool().any(-1).all()):
            raise ValueError("every source token must fit the full configured prefix")
        mask = source_mask.bool()
        detached = features.detach().masked_fill(~mask[..., None], 0)
        if not torch.isfinite(detached).all():
            raise ValueError("nonfinite valid source feature")
        latent = self.source_projection(self.source_norm(detached))
        if self.residual_prefix:
            codec = self.generator.vqvae_local[0]
            if source_ids is None or source_ids.shape != source_mask.shape:
                raise ValueError("residual source fusion requires the complete source token IDs")
            with torch.no_grad():
                raw = codec.encoder.wte(source_ids)
            latent = raw + latent
        elif source_ids is not None:
            raise ValueError("source token fusion must be explicitly enabled")
        return latent.masked_fill(~mask[..., None], 0)

    def forward(self, features, source_mask, codes, source_ids=None):
        latent = self.source_latents(features, source_mask, source_ids)
        return teacher_concept_logits(self.generator, latent, source_mask, codes)

    @torch.no_grad()
    def generate(self, features, source_mask, source_ids=None):
        if self.training:
            raise ValueError("the complete new head must be in eval mode")
        latent = self.source_latents(features, source_mask, source_ids)
        return greedy_concepts(self.generator, latent, source_mask)
