"""Identity-initialized expansion of the pinned NCM generator blocks."""
import copy

import torch


def expand(model, depth):
    head = model.concepts
    generator = head.generator
    original = list(generator.transformer.blocks)
    if depth not in (12, 24) or len(original) != 12 or head._bound is not None:
        raise ValueError('unbound original twelve-block NCM and depth12/24 required')
    if depth == 12:
        return {'blocks': 12, 'inserted': [], 'source_block_positions': list(head.layers)}
    blocks = []
    for block in original:
        if block.normformer or block.use_rezero or block.use_layerscale:
            raise ValueError('plain residual blocks required for identity initialization')
        added = copy.deepcopy(block)
        # deepcopy retains hook functions closing over the original source head.
        # Keep the hooks on the originals, never on the new residual blocks.
        added._forward_hooks.clear()
        if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
               for m in added.modules()):
            raise ValueError('unexpected hook in copied transformer block')
        added.attn.kv_caching(False)
        with torch.no_grad():
            for projection in (added.attn.c_proj, added.mlp.c_proj):
                projection.weight.zero_()
                if projection.bias is not None:
                    projection.bias.zero_()
        blocks.extend((block, added))
    generator.transformer.blocks = torch.nn.ModuleList(blocks)
    generator.config = copy.deepcopy(generator.config)
    generator.config.n_layer = depth
    return {'blocks': depth, 'inserted': list(range(1, 24, 2)),
            'source_block_positions': [2*i for i in head.layers]}
