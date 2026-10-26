"""Lossless example collation and batch-size-independent conditioning noise."""
import torch

from experiments.structure_net import ncm_packed_sequential_objective as objective


def collate(features, codes, *, device, width=3072, positions=641, vocabulary=8192):
    """Pad complete detached FP32 memories; never clip source positions."""
    if not features or len(features) != len(codes):
        raise ValueError('nonempty paired source memories and hierarchies required')
    if any(x.ndim != 2 or x.shape[1] != width or not 0 < len(x) <= 8192
           or x.dtype != torch.float32 or not torch.isfinite(x).all() for x in features):
        raise ValueError('complete finite FP32 source memories required')
    gold = torch.as_tensor(codes, dtype=torch.long, device=device)
    if (gold.shape != (len(features), positions) or (gold < 0).any()
            or (gold >= vocabulary).any()):
        raise ValueError('complete in-range native hierarchies required')
    # Reject floating IDs rather than silently truncating them in as_tensor.
    if any(type(v) is not int for row in codes for v in row):
        raise ValueError('integer native code IDs required')
    maximum = max(len(x) for x in features)
    memory = torch.zeros(len(features), maximum, width, dtype=torch.float32, device=device)
    valid = torch.zeros(len(features), maximum, dtype=torch.bool, device=device)
    for i, value in enumerate(features):
        memory[i, :len(value)] = value.detach().to(device)
        valid[i, :len(value)] = True
    return memory, valid, gold


def histories(gold, levels, vocabulary, rng, noisy):
    """Use the original example-then-level RNG order for any batch partition."""
    if gold.ndim != 2 or not len(gold) or len(noisy) != len(gold):
        raise ValueError('one corruption flag per nonempty example required')
    parts, changes = [], []
    for i, enabled in enumerate(noisy):
        if type(enabled) is not bool:
            raise ValueError('explicit boolean corruption flags required')
        part, count = objective.history(gold[i:i+1], levels, vocabulary, rng, enabled)
        parts.append(part)
        changes.append(count)
    return torch.cat(parts), changes


def weighted_loss(logits, gold, levels, *, examples_in_update):
    """Each example and each native level receive equal weight, including tails."""
    if type(examples_in_update) is not int or examples_in_update < len(gold):
        raise ValueError('complete effective update size required')
    terms = objective.losses(logits, gold, levels)
    return terms.mean() * len(gold) / examples_in_update
