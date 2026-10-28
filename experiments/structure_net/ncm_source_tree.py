"""Exploratory source hierarchy for every NCM block; no trained-quality claim.

Leaves retain the existing dense source attention. A zero-output residual reads
pooled ancestors at every block. This is a quality control before sparse routing,
not a fast attention kernel or a replacement for the frozen source encoder.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net import ncm_source_coverage as coverage

SCHEMA_VERSION = 'nal.ncm-source-tree.v1'
HYPOTHESIS_ID = 'ncm-source-tree-mechanism-v1'


@dataclass(frozen=True)
class Config:
    branching: int = 8
    maximum_levels: int = 5

    def __post_init__(self):
        if type(self.branching) is not int or self.branching < 2:
            raise ValueError('branching must be an integer >= 2')
        if type(self.maximum_levels) is not int or self.maximum_levels < 1:
            raise ValueError('at least one ancestor level required')


class TreeMemory(tuple):
    def __new__(cls, leaves, ancestors):
        value = super().__new__(cls, leaves)
        value.ancestors = tuple(ancestors)
        return value


def build_tree(memory, config=Config()):
    """Pool K/V by valid leaf mass, keeping every original leaf unchanged."""
    k, v, mask = memory
    if k.ndim != 4 or v.shape != k.shape or mask.shape != (k.shape[0], 1, 1, k.shape[2]):
        raise ValueError('source K/V and mask geometry differs')
    if mask.dtype != torch.bool or not mask.any(-1).all():
        raise ValueError('nonempty boolean source masks required')
    if k.shape[2] > config.branching ** config.maximum_levels:
        raise ValueError('configured tree cannot reach a single root')
    mass = mask[:, 0, 0].float()
    ancestors = []
    while k.shape[2] > 1:
        length = k.shape[2]
        padding = (-length) % config.branching
        weights = F.pad(mass, (0, padding)).reshape(len(k), -1, config.branching)
        parent_mass = weights.sum(-1)

        def pool(x):
            # Invalid padding may contain arbitrary values; zero it before sums.
            x = x.masked_fill(~mask.transpose(2, 3), 0).float()
            x = F.pad(x, (0, 0, 0, padding))
            x = x.reshape(len(k), k.shape[1], -1, config.branching, k.shape[-1])
            summed = (x * weights[:, None, :, :, None]).sum(-2)
            return (summed / parent_mass[:, None, :, None].clamp_min(1)).to(k.dtype)

        k, v = pool(k), pool(v)
        mask = (parent_mass > 0)[:, None, None, :]
        mass = parent_mass
        ancestors.append((k, v, mask))
    return TreeMemory(memory, ancestors)


class TreeAttention(nn.Module):
    def __init__(self, leaf_attention, config):
        super().__init__()
        self.leaf = leaf_attention
        self.config = config
        width = self.leaf.width
        # These parameters draw no RNG and preserve the original function.
        self.level_weights = nn.Parameter(self.leaf.q.weight.new_zeros(config.maximum_levels, width))
        self.tree_out = nn.Parameter(self.leaf.out.weight.new_zeros(width, width))

    def forward(self, targets, memory):
        if not isinstance(memory, TreeMemory):
            raise ValueError('tree attention requires source tree memory')
        baseline = self.leaf(targets, memory)
        if not memory.ancestors:
            return baseline
        normalized = self.leaf.norm(targets)
        q = self.leaf.q(normalized).reshape(len(targets), targets.shape[1],
            self.leaf.heads, self.leaf.width // self.leaf.heads).transpose(1, 2)
        contexts = []
        for k, v, mask in memory.ancestors:
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                dropout_p=self.leaf.dropout if self.training else 0.)
            contexts.append(attended.transpose(1, 2).reshape_as(targets))
        scores = F.linear(normalized.float(), self.level_weights[:len(contexts)].float())
        weights = scores.softmax(-1).to(targets.dtype)
        mixed = (torch.stack(contexts, dim=-2) * weights[..., None]).sum(-2)
        return baseline + F.linear(mixed, self.tree_out)


def install(model, config=Config()):
    """Install on a loaded prototype, retaining all original parameter objects."""
    head = model.concepts
    if head._bound is not None or hasattr(head, 'source_tree_config'):
        raise ValueError('fresh unbound source head required')
    if len(head.generator.transformer.blocks) != 12 or head.layers not in ((8, 10, 11), tuple(range(12))):
        raise ValueError('original or all-twelve-block source topology required')
    if set(head.adapters) != {str(i) for i in head.layers} or len(head._handles) != len(head.layers):
        raise ValueError('source adapter bindings differ')
    if head.source_memory.maximum > config.branching ** config.maximum_levels:
        raise ValueError('tree must cover the entire allowed source memory')
    before = sum(p.numel() for p in model.parameters())
    expanded = coverage.expand(model) if head.layers == (8, 10, 11) else None
    for layer in head.layers:
        old = head.adapters[str(layer)]
        replacement = TreeAttention(old, config)
        # Preserve the existing leaf's mode, independently of the wrapper mode.
        replacement.training = old.training
        head.adapters[str(layer)] = replacement
    handle = head.source_memory.register_forward_hook(lambda module, args, output: build_tree(output, config))
    head._handles.append(handle)
    head.source_tree_config = config
    return {'schema': SCHEMA_VERSION, 'hypothesis_id': HYPOTHESIS_ID,
        'source_blocks': list(head.layers), 'branching': config.branching,
        'maximum_levels': config.maximum_levels, 'source_coverage_expansion': expanded,
        'added_parameters': sum(p.numel() for p in model.parameters()) - before,
        'dense_leaf_path_retained': True, 'sparse_routing': False,
        'quality_measured': False, 'speed_measured': False}
