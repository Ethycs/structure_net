"""Matched source-attention arms, retaining original state keys and parameters."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from experiments.structure_net import ncm_source_coverage as coverage
from experiments.structure_net import ncm_source_tree_padding as tree
from experiments.structure_net.ncm_lm1b_source_attention import SourceAttention

ARMS = ('tree','dense')
PARAMETERS = {'tree':123063936,'dense':115939968}


class CheckpointedSourceAttention(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.width,self.heads,self.dropout = source.width,source.heads,source.dropout
        self.norm,self.q,self.out = source.norm,source.q,source.out
        self.training = source.training

    def core(self,targets,memory):
        return SourceAttention.forward(self,targets,memory)

    def forward(self,targets,memory):
        if torch.is_grad_enabled():
            return checkpoint(self.core,targets,memory,use_reentrant=False,preserve_rng_state=True)
        return self.core(targets,memory)


def install(model,arm):
    if arm not in ARMS: raise ValueError('fixed architecture arm required')
    if arm == 'tree': return tree.install(model)
    metadata = coverage.expand(model)
    for layer in model.concepts.layers:
        model.concepts.adapters[str(layer)] = CheckpointedSourceAttention(model.concepts.adapters[str(layer)])
    return {**metadata,'arm':arm,'activation_checkpointing':'nonreentrant_token_attention',
        'checkpoint_preserves_RNG':True,'ancestor_path':False}
