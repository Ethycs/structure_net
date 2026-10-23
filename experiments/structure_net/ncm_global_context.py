"""Read the final valid body token independently of padding direction."""
import torch


def last_valid(hidden, pad_mask):
    if hidden.ndim != 3 or pad_mask.shape != hidden.shape[:2] or pad_mask.dtype != torch.bool:
        raise ValueError("body/mask shape or type mismatch")
    positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(pad_mask)
    last = positions.masked_fill(pad_mask, -1).max(-1).values
    if bool(last.lt(0).any()):
        raise ValueError("empty body context")
    return hidden[torch.arange(len(hidden), device=hidden.device), last]
