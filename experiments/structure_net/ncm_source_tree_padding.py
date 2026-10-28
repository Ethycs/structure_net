"""Mask redundant hierarchy levels created by another example's padding."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from experiments.structure_net import ncm_source_tree as tree


class PerSourceTreeMemory(tree.TreeMemory):
    def __new__(cls, memory, active_levels):
        value = super().__new__(cls, memory, memory.ancestors)
        value.active_levels = active_levels
        return value


def per_source_levels(memory, branching):
    valid = memory[2][:,0,0]
    if (valid[:,1:] & ~valid[:,:-1]).any(): raise ValueError('right-padded source required')
    lengths = valid.sum(-1)
    levels = []
    for _ in memory.ancestors:
        levels.append(lengths > 1)
        lengths = (lengths+branching-1)//branching
    active = torch.stack(levels,dim=-1) if levels else valid.new_zeros((len(valid),0))
    return PerSourceTreeMemory(memory,active)


class PerSourceTreeAttention(tree.TreeAttention):
    def forward(self,targets,memory):
        if not isinstance(memory,PerSourceTreeMemory): raise ValueError('per-source hierarchy mask required')
        baseline = self.leaf(targets,memory)
        if not memory.ancestors: return baseline
        normalized = self.leaf.norm(targets)
        q = self.leaf.q(normalized).reshape(len(targets),targets.shape[1],self.leaf.heads,
            self.leaf.width//self.leaf.heads).transpose(1,2)
        contexts = []
        for k,v,mask in memory.ancestors:
            attended = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,
                dropout_p=self.leaf.dropout if self.training else 0.)
            contexts.append(attended.transpose(1,2).reshape_as(targets))
        scores = F.linear(normalized.float(),self.level_weights[:len(contexts)].float())
        active = memory.active_levels[:,None,:]
        # A one-token source has no ancestors: zero its mixture without NaNs.
        weights = scores.masked_fill(~active,torch.finfo(scores.dtype).min).softmax(-1)
        weights = weights.masked_fill(~active,0).to(targets.dtype)
        mixed = (torch.stack(contexts,dim=-2)*weights[...,None]).sum(-2)
        return baseline+F.linear(mixed,self.tree_out)


class CheckpointedPerSourceTreeAttention(PerSourceTreeAttention):
    def forward(self,targets,memory):
        if torch.is_grad_enabled():
            return checkpoint(super().forward,targets,memory,use_reentrant=False,preserve_rng_state=True)
        return super().forward(targets,memory)


def install(model):
    metadata = tree.install(model)
    for layer in model.concepts.layers:
        old = model.concepts.adapters[str(layer)]
        new = CheckpointedPerSourceTreeAttention(old.leaf,old.config)
        new.level_weights,new.tree_out = old.level_weights,old.tree_out
        new.training = old.training
        model.concepts.adapters[str(layer)] = new
    branch = model.concepts.source_tree_config.branching
    handle = model.concepts.source_memory.register_forward_hook(
        lambda module,args,output:per_source_levels(output,branch))
    model.concepts._handles.append(handle)
    return {**metadata,'activation_checkpointing':'nonreentrant_leaf_and_ancestor_attention',
        'checkpoint_preserves_RNG':True,'per_source_level_mask':True,
        'redundant_padded_root_levels_excluded':True,'additional_parameters_for_checkpointing':0}
