"""Coarse AR and zero-initialized frozen-body residuals; immutable vendor code."""
import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net.ncm_frozen_body import supplied_context
from experiments.structure_net.ncm_training_recipe import corrupt_conditioning, hierarchical_loss


def enable_coarse_ar(net):
    if not net.query_level0 or net.use_flex_attention or not net.use_rope:
        raise ValueError("this adaptation requires the released RoPE/eager query architecture")
    net.l0_ar = True
    net.config.l0_autoregressive = True
    p, width = net.prefix_len, net.levels[0]
    mask = net.attn_mask_for_masking.clone()
    mask[:, :, p:p+width, p:p+width] = torch.triu(
        torch.full((width, width), float('-inf'), device=mask.device), diagonal=1)
    net.attn_mask_for_masking.copy_(mask)


class GlobalResidual(nn.Module):
    def __init__(self, source_width=3072, target_width=768):
        super().__init__()
        self.norm = nn.LayerNorm(source_width)
        self.proj = nn.Linear(source_width, target_width)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, native, body_last, pad_mask):
        delta = self.proj(self.norm(body_last.float()))[:, None, :]
        return native + delta * (~pad_mask).unsqueeze(-1)


def source_context(vae, net, bridge, prefix, body_last):
    with torch.no_grad():
        embedded = vae.encoder.wte(prefix)
    native = net.transformer.wte(embedded)
    return bridge(native, body_last, prefix.eq(50257))


def logits_for_codes(vae, net, context, prefix, clean, conditioning):
    if not net.l0_ar:
        raise ValueError("coarse AR flag required")
    with torch.no_grad():
        previous = vae.vq.idx_to_ncm_input(conditioning)
    coarse = net.build_l0_ar_input(clean[:, :net.levels[0]], previous.dtype)
    target = torch.cat((coarse, previous), 1)
    with supplied_context(net, context):
        return net(target, prefix_ids=prefix, prefix_pad_mask=prefix.eq(50257))


def loss_for_batch(vae, net, bridge, prefix, body_last, clean, recipe, rng, coarse_only):
    context = source_context(vae, net, bridge, prefix, body_last)
    conditioning, corruption = corrupt_conditioning(clean, vae.vq.levels,
        vae.vq.vqvae_vocab_sizes, recipe, rng)
    logits = logits_for_codes(vae, net, context, prefix, clean, conditioning)
    if coarse_only:
        width = net.levels[0]
        loss = F.cross_entropy(logits[:, :width].float().flatten(0, 1), clean[:, :width].flatten())
        per_level = [float(loss.detach())]
    else:
        loss, per_level = hierarchical_loss(logits, clean, vae.vq.levels, alpha=1.)
    return loss, {"per_level_ce": per_level, "corruption": corruption}


@torch.no_grad()
def generate_codes(vae, net, bridge, prefix, body_last):
    context = source_context(vae, net, bridge, prefix, body_last)
    with supplied_context(net, context):
        return net.generate(None, prefix, temperature=0., top_k=1, top_p=0.,
                            cfg_scale=0., use_kv_cache=False)
