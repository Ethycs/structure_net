"""Raw native source KV and intact shared decoder blocks for conditional queries."""
from contextlib import contextmanager
from dataclasses import dataclass
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from experiments.structure_net.ncm_native_readout import LoraLinear


class QueryLoraLinear(LoraLinear):
    def __init__(self, base, rank=8):
        super().__init__(base, rank)
        self.enabled = False

    def forward(self, x):
        if not self.enabled or (not torch.is_grad_enabled() and torch.count_nonzero(self.b).item() == 0):
            return self.base(x)
        return super().forward(x)


@dataclass(frozen=True)
class NativePrefix:
    keys_values: tuple
    source_mask: torch.Tensor
    anchor_token: torch.Tensor


class NativeQueryBackbone(nn.Module):
    def __init__(self, language, rank=8):
        super().__init__()
        self.language = language.requires_grad_(False)
        for layer in self.language.model.layers:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                attention = layer.self_attn
                setattr(attention, name, QueryLoraLinear(getattr(attention, name), rank))
        self.adapters = [v for v in language.modules() if isinstance(v, QueryLoraLinear)]

    @contextmanager
    def query_adapters(self, enabled):
        before = [v.enabled for v in self.adapters]
        try:
            for value in self.adapters: value.enabled = enabled
            yield
        finally:
            for value, previous in zip(self.adapters, before): value.enabled = previous

    @torch.no_grad()
    def source_cache(self, input_ids, attention_mask):
        if input_ids.shape != attention_mask.shape or input_ids.shape[1] < 2 or not attention_mask[:, -1].bool().all():
            raise ValueError("native prompt with an unpadded final anchor required")
        mask = attention_mask[:, :-1].clone()
        if not mask.bool().any(-1).all(): raise ValueError("nonempty source prefix required")
        positions = (mask.long().cumsum(-1)-1).clamp_min(0)
        with self.query_adapters(False):
            output = self.language.model(input_ids=input_ids[:, :-1], attention_mask=mask,
                position_ids=positions, use_cache=True, return_dict=True)
        cache = tuple(tuple(v.detach() for v in pair) for pair in output.past_key_values.to_legacy_cache())
        return NativePrefix(cache, mask, input_ids[:, -1:].clone())

    def predecessor_embeddings(self, memory, previous_words):
        if previous_words.ndim != 2 or previous_words.shape[0] != memory.source_mask.shape[0]:
            raise ValueError("previous native words must match source batch")
        # previous_words are explicit predecessors, not labels at the positions
        # currently being predicted. The prompt anchor predicts the first word.
        ids = torch.cat((memory.anchor_token, previous_words), 1)
        return self.language.model.embed_tokens(ids)

    def query_logits(self, memory, query, offsets=None):
        if query.ndim != 3 or query.shape[0] != memory.source_mask.shape[0] or not query.shape[1]:
            raise ValueError("native query dimensions differ")
        batch, width = query.shape[:2]
        if offsets is None: offsets = torch.arange(width, device=query.device)
        if offsets.ndim != 1 or len(offsets) != width: raise ValueError("query offsets differ")
        positions = memory.source_mask.sum(-1, keepdim=True)+offsets[None]
        mask = torch.cat((memory.source_mask, torch.ones(batch, width, device=query.device, dtype=memory.source_mask.dtype)), 1)
        cache = DynamicCache.from_legacy_cache(memory.keys_values)
        with self.query_adapters(True):
            output = self.language.model(inputs_embeds=query, attention_mask=mask, position_ids=positions,
                past_key_values=cache, use_cache=True, return_dict=True)
        return self.language.lm_head(output.last_hidden_state)
