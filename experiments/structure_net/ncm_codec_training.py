"""Cutoff-specific VQ-VAE objectives and EMA for the released NCM quantizer."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def normalize_ema(vq, batch):
    """Change the EMA's reference mass while preserving every embedding vector."""
    for i, width in enumerate(vq.levels):
        counts = getattr(vq, f"ema_cluster_size_{i}")
        weighted = getattr(vq, f"ema_w_{i}")
        mass = float(counts.sum())
        if mass <= 0:
            raise ValueError("pretrained EMA has no mass")
        counts.mul_(batch*width/mass)
        n = counts.sum()
        smooth = (counts+vq.epsilon)/(n+len(counts)*vq.epsilon)*n
        weighted.copy_(vq.embeddings[i].weight*smooth[:, None])


@torch.no_grad()
def ema_update(vq, level, values, indices, decay):
    if not 0 < decay < 1:
        raise ValueError("EMA decay must be between zero and one")
    counts = getattr(vq, f"ema_cluster_size_{level}")
    weighted = getattr(vq, f"ema_w_{level}")
    hits = torch.bincount(indices.flatten(), minlength=len(counts)).to(counts.dtype)
    sums = torch.zeros_like(weighted)
    sums.index_add_(0, indices.flatten(), values.detach().reshape(-1, values.shape[-1]))
    counts.mul_(decay).add_(hits, alpha=1-decay)
    weighted.mul_(decay).add_(sums, alpha=1-decay)
    n = counts.sum()
    smooth = (counts+vq.epsilon)/(n+len(counts)*vq.epsilon)*n
    vq.embeddings[level].weight.copy_(weighted/smooth[:, None])


def quantize_cutoff(vae, z, cutoff, *, decay=None):
    """Use actual partial reconstruction for both VQ losses and decoder input."""
    vq = vae.vq
    if not 1 <= cutoff <= len(vq.levels):
        raise ValueError("invalid hierarchy cutoff")
    # Keep nearest assignments, resampling and VQ objectives in FP32.
    with torch.autocast(z.device.type, enabled=False):
        z = z.float()
        rest = z.detach().clone()
        reconstruction = torch.zeros_like(z)
        for i, width in enumerate(vq.levels[:cutoff]):
            flat = vq._down_for_quant(rest, i, width) if i != len(vq.levels)-1 else rest
            with torch.no_grad():
                distances = vq._distances(flat.reshape(-1, z.shape[-1]), vq.embeddings[i].weight)
                indices = distances.argmin(-1).view(z.shape[0], width)
                if decay is not None:
                    ema_update(vq, i, flat, indices, decay)
            selected = vq.embeddings[i](indices)
            if i == 0 and vq.learned_l1_sampling:
                selected = flat+(selected-flat).detach()
            if i != len(vq.levels)-1:
                selected = vq._up_to_full(selected, i, z.shape[1])
            contribution = vq._apply_quant_resi(selected, i)
            reconstruction = reconstruction+contribution
            rest = rest-contribution
        commitment = vq.commitment_cost*F.mse_loss(z, reconstruction.detach())
        quantization = F.mse_loss(reconstruction, z.detach())
        straight = z+(reconstruction-z).detach()
    return straight, commitment+quantization, {"commitment": float(commitment.detach()),
        "quantization": float(quantization.detach()), "cutoff": cutoff}


def codec_loss(vae, ids, cutoff, *, decay=None):
    z = vae._encode_pre_quant(ids)
    latent, vq_loss, details = quantize_cutoff(vae, z, cutoff, decay=decay)
    logits = vae.decoder(latent)
    reconstruction = F.cross_entropy(logits.float().flatten(0, 1), ids.flatten(), ignore_index=50257)
    return reconstruction+vq_loss, {**details, "reconstruction_ce": float(reconstruction.detach())}
