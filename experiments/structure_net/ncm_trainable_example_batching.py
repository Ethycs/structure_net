"""Batch generator work while preserving codec numerics and learned-root gradients."""
import torch


def teacher_logits(head, features, source_mask, codes):
    if codes.shape != (len(features), sum(head.generator.levels)):
        raise ValueError('complete native hierarchy and source batch required')
    with torch.no_grad():
        prefix_ids = torch.full((len(features), 1), 101, device=features.device, dtype=torch.long)
        prefix = head.codec.encoder.wte(prefix_ids)
        earlier = torch.cat([head.codec.vq.idx_to_ncm_input(part) for part in codes.split(1)]).to(prefix.dtype)
    # Unlike the earlier batching-only probe, retain the learned root's graph.
    # The native fine-tuning reference places both this query and concatenation
    # outside no_grad; only fixed codec computations belong inside it.
    root = head.generator.build_l0_query_input(len(features), dtype=prefix.dtype)
    inputs = torch.cat((prefix, root, earlier), dim=1)
    with head.source_context(features, source_mask):
        return head.generator(inputs)
