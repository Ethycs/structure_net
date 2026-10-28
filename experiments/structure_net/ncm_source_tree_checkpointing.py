"""Recompute source-attention activations without changing tree parameters."""
import torch
from torch.utils.checkpoint import checkpoint

from experiments.structure_net import ncm_source_tree as tree


class CheckpointedTreeAttention(tree.TreeAttention):
    def forward(self, targets, memory):
        if torch.is_grad_enabled():
            return checkpoint(super().forward, targets, memory, use_reentrant=False,
                              preserve_rng_state=True)
        return super().forward(targets, memory)


def install(model):
    metadata = tree.install(model)
    for layer in model.concepts.layers:
        old = model.concepts.adapters[str(layer)]
        new = CheckpointedTreeAttention(old.leaf, old.config)
        new.level_weights, new.tree_out = old.level_weights, old.tree_out
        new.training = old.training
        model.concepts.adapters[str(layer)] = new
    return {**metadata, 'activation_checkpointing': 'nonreentrant_leaf_and_ancestor_attention',
            'checkpoint_preserves_RNG': True, 'additional_parameters_for_checkpointing': 0}
