"""Explicit padding and frozen-context contracts for an NCM source encoder."""
import torch


def right_pack(hidden, pad_mask):
    """Preserve valid-state order, zero padding and retain the gradient path."""
    if hidden.ndim != 3 or pad_mask.shape != hidden.shape[:2] or pad_mask.dtype != torch.bool:
        raise ValueError("context must be [batch,prefix,width] with a Boolean pad mask")
    if bool(pad_mask.all(-1).any()):
        raise ValueError("empty source context")
    order = torch.argsort(pad_mask.long(), dim=-1, stable=True)
    packed = hidden.gather(1, order[..., None].expand_as(hidden))
    packed_mask = pad_mask.gather(1, order)
    return packed.masked_fill(packed_mask[..., None], 0), packed_mask


def unpaired_windows(cases, tokenizer, canvas=48):
    """Training-only complete text windows; include the final short fragment."""
    windows = []
    for case in cases:
        ids = tokenizer.encode(case["source"], add_special_tokens=False)
        for offset in range(0, len(ids), canvas-1):
            chunk = ids[offset:offset+canvas-1]
            windows.append(chunk+[50256]+[50257]*(canvas-len(chunk)-1))
    return torch.tensor(windows, dtype=torch.long)
