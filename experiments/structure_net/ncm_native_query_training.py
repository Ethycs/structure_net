"""Target-free noisy history and separately evaluated conditional experts."""
import torch
from experiments.structure_net.ncm_native_concepts import conditional_target, expand


@torch.no_grad()
def actual_history(net, memory, level, generator, rates):
    if not 0 <= level <= len(net.levels) or len(rates) < level or any(not 0 <= r <= 1 for r in rates):
        raise ValueError("invalid generated-history level/corruption")
    prior = net.books[0].new_zeros(len(memory.source_mask),net.canvas,net.books[0].shape[1])
    codes = []
    for i,book in enumerate(net.books[:level]):
        labels = net.level_logits(memory,prior,i).argmax(-1)
        mask = torch.rand(labels.shape,generator=generator) < rates[i]
        replacement = torch.randint(len(book),labels.shape,generator=generator)
        labels = torch.where(mask.to(labels.device),replacement.to(labels.device),labels)
        codes.append(labels)
        prior = prior+expand(book[labels],net.canvas)
    return codes,prior


@torch.no_grad()
def expert_gains(target, books, levels, actual_codes, clean_codes):
    if len(actual_codes) != len(books) or len(clean_codes) != len(books):
        raise ValueError("complete diagnostic history required")
    prior = torch.zeros_like(target,dtype=torch.float32)
    gains = []
    for book,width,actual,static in zip(books,levels,actual_codes,clean_codes):
        expert = conditional_target(target,prior,book,width)
        unchanged = prior+expand(book[static],target.shape[1])
        repaired = prior+expand(book[expert],target.shape[1])
        gain = (target-unchanged).square().mean((1,2))-(target-repaired).square().mean((1,2))
        if (gain < -1e-6).any(): raise ValueError("expert worse at the same actual history")
        gains.append(gain)
        # Advance with the ACTUAL noisy prediction, never with the expert.
        prior = prior+expand(book[actual],target.shape[1])
    return torch.stack(gains,1)
