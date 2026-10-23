"""Requantize a residual hierarchy after a fixed, potentially incorrect root."""
import torch


@torch.no_grad()
def requantize(vq, z, forced_root=None):
    if z.ndim != 3 or getattr(vq, "_stoch_encode", False):
        raise ValueError("deterministic quantizer and batch/length/channel latents required")
    if forced_root is not None:
        if forced_root.shape != (len(z), vq.levels[0]) or forced_root.dtype != torch.long:
            raise ValueError("forced root shape/type mismatch")
        if bool(((forced_root < 0) | (forced_root >= vq.embeddings[0].num_embeddings)).any()):
            raise ValueError("forced root outside codebook")
    with torch.autocast(z.device.type, enabled=False):
        z = z.float()
        batch, length, width = z.shape
        residual = z.detach().clone()
        reconstruction = torch.zeros_like(residual)
        indices = []
        for i, level in enumerate(vq.levels):
            book = vq.embeddings[i]
            flat = (vq._down_for_quant(residual, i, level) if i != len(vq.levels)-1 else residual).reshape(-1, width)
            chosen = forced_root if i == 0 and forced_root is not None else vq._distances(flat, book.weight).argmin(1).view(batch, level)
            indices.append(chosen)
            contribution = book(chosen)
            if i != len(vq.levels)-1:
                contribution = vq._up_to_full(contribution, i, length)
            contribution = vq._apply_quant_resi(contribution, i)
            reconstruction = reconstruction+contribution
            residual -= contribution
        return torch.cat(indices, -1), reconstruction


@torch.no_grad()
def nearest_distinct(book, count=8):
    if count < 1 or count >= len(book):
        raise ValueError("invalid neighborhood size")
    with torch.autocast(book.device.type, enabled=False):
        # Higher precision avoids cancellation making duplicate vectors look distinct.
        embedding = book.double()
        norms = embedding.square().sum(-1)
        distances = norms[:, None]+norms[None, :]-2*(embedding@embedding.T)
        distances = distances.masked_fill(distances <= 1e-8, float("inf"))
        distances.fill_diagonal_(float("inf"))
        values, indices = distances.topk(count, dim=-1, largest=False)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("too few distinct vectors in the coarse book")
    return indices, values
