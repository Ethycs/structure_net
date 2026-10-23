"""Training-only experts recompute finer labels against actual corrupted history.

This is an experimental extension to NCM's clean-target conditioning corruption.
It requires the reference text latent during training. Inference is unchanged
and must never call this module or receive reference latents/codes.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Corruption:
    mask: torch.Tensor
    replacements: torch.Tensor


def sample_corruption(clean, levels, vocabularies, neighbors, rng, first=.5, last=0.):
    if clean.ndim != 2 or clean.shape[1] != sum(levels) or len(levels) != len(vocabularies):
        raise ValueError("code hierarchy shape mismatch")
    if not 0 <= first <= 1 or not 0 <= last <= 1:
        raise ValueError("noise rates must be probabilities")
    if neighbors.ndim != 2 or neighbors.shape[0] != vocabularies[0] or neighbors.shape[1] < 1:
        raise ValueError("root neighborhood shape mismatch")
    cpu_clean = clean.detach().cpu()
    rates = torch.linspace(first, last, len(levels)).tolist()
    masks, replacements = [], []
    offset = 0
    for i, (width, vocab, rate) in enumerate(zip(levels, vocabularies, rates)):
        labels = cpu_clean[:, offset:offset+width]
        if bool(((labels < 0) | (labels >= vocab)).any()):
            raise ValueError("clean code outside vocabulary")
        masks.append(torch.rand(labels.shape, generator=rng) < rate)
        if i == 0:
            choices = torch.randint(neighbors.shape[1], labels.shape, generator=rng)
            replacements.append(neighbors.detach().cpu()[labels, choices])
        else:
            replacements.append(torch.randint(vocab, labels.shape, generator=rng))
        offset += width
    return Corruption(torch.cat(masks, -1).to(clean.device), torch.cat(replacements, -1).to(clean.device))


@torch.no_grad()
def training_targets(vq, latent, clean, corruption, *, repair):
    """Return teacher labels and teacher-forced history for one training batch.

    Expert labels are computed before adding each level's own corruption, so a
    query's target cannot depend on its own or any future noisy conditioning.
    In repair mode, each finer expert sees the *actual* preceding contributions.
    The static control uses the original clean hierarchy throughout. Both modes
    execute the same quantizer operations and consume the same saved noise.
    """
    if (latent.ndim != 3 or clean.ndim != 2 or len(latent) != len(clean)
            or clean.shape[1] != sum(vq.levels) or latent.shape[1] != vq.levels[-1]
            or clean.dtype != torch.long or getattr(vq, "_stoch_encode", False)):
        raise ValueError("deterministic complete latent/code hierarchy required")
    if (corruption.mask.shape != clean.shape or corruption.replacements.shape != clean.shape
            or corruption.mask.dtype != torch.bool or corruption.replacements.dtype != torch.long):
        raise ValueError("corruption tensor shape/type mismatch")
    if len({latent.device, clean.device, corruption.mask.device, corruption.replacements.device}) != 1:
        raise ValueError("training tensors must share one device")
    with torch.autocast(latent.device.type, enabled=False):
        residual = latent.detach().float().clone()
        batch, length, width = residual.shape
        targets, conditioning = [], []
        offset = 0
        for i, size in enumerate(vq.levels):
            gold = clean[:, offset:offset+size]
            replacements = corruption.replacements[:, offset:offset+size]
            vocab = vq.embeddings[i].num_embeddings
            if bool(((gold < 0) | (gold >= vocab) | (replacements < 0) | (replacements >= vocab)).any()):
                raise ValueError("clean or replacement code outside vocabulary")
            flat = (vq._down_for_quant(residual, i, size) if i != len(vq.levels)-1 else residual).reshape(-1, width)
            expert = vq._distances(flat, vq.embeddings[i].weight.float()).argmin(1).view(batch, size)
            if i == 0 and not torch.equal(expert, gold):
                raise ValueError("clean root labels do not match the target latent")
            label = expert if repair else gold
            given = torch.where(corruption.mask[:, offset:offset+size], replacements, label)
            targets.append(label.clone()); conditioning.append(given)
            contribution = vq.embeddings[i](given).float()
            if i != len(vq.levels)-1:
                contribution = vq._up_to_full(contribution, i, length)
            residual -= vq._apply_quant_resi(contribution, i)
            offset += size
        labels, history = torch.cat(targets, -1), torch.cat(conditioning, -1)
    return labels, history, {"selected": int(corruption.mask.sum()),
        "changed_from_expert": int(history.ne(labels).sum()),
        "relabelled_from_clean": int(labels.ne(clean).sum()), "positions": clean.numel()}
