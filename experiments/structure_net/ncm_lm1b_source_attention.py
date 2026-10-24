"""Trainable source attention around a frozen, native-geometry NCM generator.

This supplies mechanisms only. It is not a trained or qualified summarizer.
The original parent is never owned by this head; source features are detached.
"""
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F

from experiments.structure_net import ncm_lm1b_codec as release
from experiments.structure_net.ncm_memory_data import file_hash

CONFIG_SHA = "4db04299a87f7eb1627de0abb2f0b257e50e5ea0138edda771ad287cdabd7edf"


class SourceMemory(nn.Module):
    def __init__(self, source_width, width, heads, maximum=8192):
        super().__init__()
        if width % heads:
            raise ValueError("complete attention heads required")
        self.source_width, self.width, self.heads, self.maximum = source_width, width, heads, maximum
        self.norm = nn.LayerNorm(source_width, elementwise_affine=False)
        self.kv = nn.Linear(source_width, 2 * width, bias=False)

    def forward(self, features, valid):
        if (features.ndim != 3 or features.shape[-1] != self.source_width or valid.shape != features.shape[:2]
                or valid.dtype != torch.bool or not 0 < features.shape[1] <= self.maximum
                or not valid.any(-1).all() or (valid[:, 1:] & ~valid[:, :-1]).any()):
            raise ValueError("complete nonempty right-padded source features and mask required")
        x = features.detach().masked_fill(~valid[..., None], 0)
        if not torch.isfinite(x).all():
            raise ValueError("nonfinite valid source feature")
        k, v = self.kv(self.norm(x)).chunk(2, dim=-1)
        shape = (len(x), x.shape[1], self.heads, self.width // self.heads)
        return k.reshape(shape).transpose(1, 2), v.reshape(shape).transpose(1, 2), valid[:, None, None, :]


class SourceAttention(nn.Module):
    def __init__(self, width, heads, dropout=.1):
        super().__init__()
        self.width, self.heads, self.dropout = width, heads, dropout
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.q = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, targets, memory):
        k, v, mask = memory
        if targets.shape[0] != k.shape[0] or targets.shape[-1] != self.width:
            raise ValueError("query and source batch/width differ")
        q = self.q(self.norm(targets)).reshape(len(targets), targets.shape[1], self.heads, self.width // self.heads).transpose(1, 2)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.)
        return self.out(attended.transpose(1, 2).reshape_as(targets))


class FrozenMemoryConceptHead(nn.Module):
    def __init__(self, generator, codec, source_width=3072, layers=(8, 10, 11), seed=68142, dropout=.1):
        super().__init__()
        if (not generator.query_level0 or not generator.use_post_upsample_conv or not generator.shared_output_head
                or generator.prefix_len != 1 or generator.use_ctx_encoder or generator.use_flex_attention):
            raise ValueError("native LM1B query/post-convolution/shared-head geometry required")
        if not layers or len(set(layers)) != len(layers) or any(not 0 <= i < len(generator.transformer.blocks) for i in layers):
            raise ValueError("distinct native block indices required")
        self.generator = generator.eval().requires_grad_(False)
        self.codec = codec.eval().requires_grad_(False)
        width, heads = generator.config.n_embd, generator.config.n_head
        self.layers, self.initialization_seed = tuple(layers), seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.source_memory = SourceMemory(source_width, width, heads)
            self.adapters = nn.ModuleDict({str(i): SourceAttention(width, heads, dropout) for i in layers})
        self._bound = None
        self._handles = []
        for i in layers:
            def hook(module, args, output, layer=i):
                if self._bound is None:
                    return output
                if output.ndim != 3 or output.shape[1] < 1:
                    raise ValueError("native prefix plus target geometry required")
                # Prefix CLS is unchanged; only native target queries read source memory.
                prefix, targets = output[:, :1], output[:, 1:]
                return torch.cat((prefix, targets + self.adapters[str(layer)](targets, self._bound)), dim=1)
            self._handles.append(generator.transformer.blocks[i].register_forward_hook(hook))
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.generator.eval()
        self.codec.eval()
        return self

    @contextmanager
    def source_context(self, features, source_mask):
        if self._bound is not None:
            raise ValueError("nested or stale source context")
        self._bound = self.source_memory(features, source_mask)
        try:
            yield
        finally:
            self._bound = None

    def teacher_logits(self, features, source_mask, codes):
        if codes.ndim != 2 or codes.shape != (len(features), sum(self.generator.levels)):
            raise ValueError("complete native teacher concept hierarchy required")
        with torch.no_grad():
            prefix_ids = torch.full((len(features), 1), 101, device=features.device, dtype=torch.long)
            prefix = self.codec.encoder.wte(prefix_ids)
            root = self.generator.build_l0_query_input(len(features), dtype=prefix.dtype)
            earlier = self.codec.vq.idx_to_ncm_input(codes).to(prefix.dtype)
            inputs = torch.cat((prefix, root, earlier), dim=1)
        with self.source_context(features, source_mask):
            return self.generator(inputs)

    def forward(self, features, source_mask, codes):
        return self.teacher_logits(features, source_mask, codes)

    @torch.no_grad()
    def generate_codes(self, features, source_mask):
        if self.training:
            raise ValueError("deterministic source-only generation requires eval mode")
        prefix = torch.full((len(features), 1), 101, device=features.device, dtype=torch.long)
        with self.source_context(features, source_mask):
            return self.generator.generate(None, prefix, temperature=0., top_k=0, top_p=1., use_kv_cache=False, cfg_scale=0.)

    def adapter_state(self):
        return {name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            for name, module in (("source_memory", self.source_memory), ("adapters", self.adapters))}

    def load_adapter_state(self, saved):
        if set(saved) != {"source_memory", "adapters"}:
            raise ValueError("exact adapter-only state required")
        self.source_memory.load_state_dict(saved["source_memory"], strict=True)
        self.adapters.load_state_dict(saved["adapters"], strict=True)

    def close(self):
        if self._bound is not None:
            raise ValueError("cannot detach active source context")
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def from_release(device="cpu", **adapter_config):
    path = release.RELEASE / "files/lm1b/small/config.yaml"
    if file_hash(path) != CONFIG_SHA:
        raise ValueError("pinned native configuration changed")
    for name, sha in release.FILES.items():
        if file_hash(release.RELEASE / "files" / name) != sha:
            raise ValueError("pinned author checkpoint changed")
    OmegaConf, VQVAE, Generator = release.vendor.vendor_classes()
    cfg = OmegaConf.load(path)
    codec = VQVAE(**cfg.tokenizer.vqvae)
    codec.load_state_dict(torch.load(release.RELEASE / "files/lm1b/tokenizer.pt", map_location="cpu", weights_only=True, mmap=True)["model"], strict=True)
    generator = Generator(codec, cfg.model)
    saved = torch.load(release.RELEASE / "files/lm1b/small/generator.pt", map_location="cpu", weights_only=True, mmap=True)["model"]
    generator.load_state_dict(saved, strict=True)
    if set(generator.state_dict()) != set(saved) or any(not torch.equal(v, saved[k]) for k, v in generator.state_dict().items()):
        raise ValueError("every released generator state must remain exact")
    return FrozenMemoryConceptHead(generator, codec, **adapter_config).to(device).float()
