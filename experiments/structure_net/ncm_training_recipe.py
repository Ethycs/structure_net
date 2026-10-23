"""Explicit hierarchical NCM adaptation, without changing frozen vendor code."""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Recipe:
    nested_dropout: float = .5
    corruption_first: float = .30
    corruption_last: float = 0.
    level_alpha: float = 1.

    def __post_init__(self):
        if not all(0 <= x <= 1 for x in (self.nested_dropout, self.corruption_first, self.corruption_last)):
            raise ValueError("probabilities must be in [0,1]")
        if not 0 <= self.level_alpha <= 2:
            raise ValueError("invalid level loss alpha")

    def record(self):
        return asdict(self)


LEGACY = Recipe(0., 0., 0., 0.)
UPDATED = Recipe()


def sample_cutoff(level_count, probability, generator):
    if level_count < 1 or not 0 <= probability <= 1:
        raise ValueError("invalid nested dropout configuration")
    if level_count == 1 or probability == 0:
        return level_count
    if float(torch.rand((), generator=generator)) >= probability:
        return level_count
    return int(torch.randint(1, level_count, (), generator=generator))


@torch.no_grad()
def cumulative_latent(vae, codes, keep_levels=None):
    """Full-resolution latent at a real cutoff; avoid aliased vendor history."""
    vq = vae.vq
    keep = len(vq.levels) if keep_levels is None else keep_levels
    if not isinstance(keep, int) or not 1 <= keep <= len(vq.levels):
        raise ValueError("invalid hierarchy cutoff")
    if codes.ndim != 2 or codes.shape[1] != sum(vq.levels):
        raise ValueError("complete clean code hierarchy required")
    latent = torch.zeros(codes.shape[0], vq.levels[-1], vae.embed_dim,
                         dtype=torch.float32, device=codes.device)
    offset = 0
    for level, width in enumerate(vq.levels[:keep]):
        contribution = vq.embeddings[level](codes[:, offset:offset+width])
        if level != len(vq.levels)-1:
            contribution = vq._up_to_full(contribution, level, vq.levels[-1])
        contribution = vq._apply_quant_resi(contribution, level)
        latent.add_(contribution)
        offset += width
    return latent


def corrupt_conditioning(clean, levels, vocabularies, recipe, generator):
    """Replace conditioning only; clean targets and their storage stay intact."""
    if clean.ndim != 2 or clean.shape[1] != sum(levels) or len(levels) != len(vocabularies):
        raise ValueError("code schedule/vocab shape mismatch")
    corrupted = clean.detach().clone()
    mask = torch.zeros_like(clean, dtype=torch.bool)
    rates = torch.linspace(recipe.corruption_first, recipe.corruption_last, len(levels)).tolist()
    offset = 0
    # Independent CPU augmentation RNG works identically with CPU/CUDA inputs.
    for width, vocab, rate in zip(levels, vocabularies, rates):
        if vocab < 1:
            raise ValueError("empty codebook")
        labels = clean[:, offset:offset+width]
        if bool(((labels < 0) | (labels >= vocab)).any()):
            raise ValueError("target code outside its level vocabulary")
        if rate > 0:
            chosen = torch.rand(labels.shape, generator=generator) < rate
            replacements = torch.randint(vocab, labels.shape, generator=generator)
            chosen, replacements = chosen.to(clean.device), replacements.to(clean.device)
            corrupted[:, offset:offset+width] = torch.where(chosen, replacements, labels)
            mask[:, offset:offset+width] = chosen
        offset += width
    return corrupted, {"selected": int(mask.sum()), "changed": int(corrupted.ne(clean).sum()),
                       "positions": clean.numel(), "rates": rates}


def hierarchical_loss(logits, clean_targets, levels, alpha=1.):
    if logits.shape[:2] != clean_targets.shape or clean_targets.shape[1] != sum(levels):
        raise ValueError("logit/target schedule mismatch")
    ce = F.cross_entropy(logits.float().flatten(0, 1), clean_targets.flatten(), reduction="none")
    ce = ce.view_as(clean_targets)
    weights = torch.tensor(levels, device=logits.device, dtype=torch.float32).pow(-alpha)
    normalization = sum(length * weight for length, weight in zip(levels, weights))
    offset, losses = 0, []
    for length in levels:
        losses.append(ce[:, offset:offset+length].mean())
        offset += length
    loss = sum(value * weight * length for value, weight, length in zip(losses, weights, levels))/normalization
    return loss, [float(value.detach()) for value in losses]


def decoder_loss(vae, clean_codes, target_ids, recipe, generator):
    cutoff = sample_cutoff(len(vae.vq.levels), recipe.nested_dropout, generator)
    latent = cumulative_latent(vae, clean_codes, cutoff)
    logits = vae.decoder(latent)
    loss = F.cross_entropy(logits.float().flatten(0, 1), target_ids.flatten(), ignore_index=50257)
    return loss, cutoff


def generator_loss(vae, net, clean_codes, recipe, generator, *, context=None,
                   pad_mask=None, prefix=None):
    from experiments.structure_net.ncm_frozen_body import external_logits
    from experiments.structure_net.ncm_llama import generator_input
    conditioning, corruption = corrupt_conditioning(clean_codes, vae.vq.levels,
        vae.vq.vqvae_vocab_sizes, recipe, generator)
    if context is not None:
        if prefix is not None or pad_mask is None:
            raise ValueError("supply either external context/mask or native prefix")
        logits = external_logits(vae, net, context, pad_mask, conditioning)
    else:
        if prefix is None:
            raise ValueError("missing source conditioning")
        logits = net(generator_input(vae, net, prefix, conditioning), prefix_pad_mask=prefix.eq(50257))
    loss, per_level = hierarchical_loss(logits, clean_codes, vae.vq.levels, recipe.level_alpha)
    return loss, {"per_level_ce": per_level, "corruption": corruption}
