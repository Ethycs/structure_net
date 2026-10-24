"""Learned source attention in native NCM concept prediction and word decoding."""
from contextlib import contextmanager

import torch
from torch import nn

from experiments.structure_net import ncm_lm1b_source_attention as concepts
from experiments.structure_net.ncm_memory_data import file_hash

ROOT = concepts.release.ROOT
DOMAIN_CHECKPOINT = ROOT / "20261024_lm1b_domain_decoder_train_v1/model.pt"
DOMAIN_SHA = "0a6b7d4df1697260e5959f5a2f06b4a5c65838c54d8f2f30e1275e0cd3ab0113"
DOMAIN_CODEC_HASH = "569bf1ad07bc4d2cc762098b7f7e9fa0f6c68fccaa954f65a191d5a3b77f9c16"


class SourceGroundedNCM(nn.Module):
    def __init__(self, concept_head, source_width=3072, word_heads=6, word_layers=(0, 1, 2, 3), seed=90142, dropout=.1):
        super().__init__()
        self.concepts = concept_head
        codec = concept_head.codec
        blocks = [codec.decoder.middle_block, *codec.decoder.extra_blocks]
        if not word_layers or len(set(word_layers)) != len(word_layers) or any(not 0 <= i < len(blocks) for i in word_layers):
            raise ValueError("distinct existing word-decoder block indices required")
        self.word_layers, self.initialization_seed = tuple(word_layers), seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.word_memory = concepts.SourceMemory(source_width, codec.embed_dim, word_heads)
            self.word_adapters = nn.ModuleDict({str(i): concepts.SourceAttention(codec.embed_dim, word_heads, dropout) for i in word_layers})
        self._word_bound, self._word_handles = None, []
        for i in word_layers:
            def hook(module, args, output, layer=i):
                if self._word_bound is None: return output
                return output + self.word_adapters[str(layer)](output, self._word_bound)
            self._word_handles.append(blocks[i].register_forward_hook(hook))
        self.train(False)

    @property
    def codec(self):
        return self.concepts.codec

    def train(self, mode=True):
        super().train(mode)
        self.concepts.train(mode)
        return self

    @contextmanager
    def word_context(self, features, mask):
        if self._word_bound is not None: raise ValueError("nested or stale word-source context")
        self._word_bound = self.word_memory(features, mask)
        try:
            yield
        finally:
            self._word_bound = None

    def word_logits(self, features, mask, codes):
        if codes.shape != (len(features), sum(self.concepts.generator.levels)):
            raise ValueError("complete native concept hierarchy required")
        # VQVAE.decode is decorated no_grad. Bypass it so NEW attention can learn;
        # only the frozen code-to-latent operation is under no_grad here.
        with torch.no_grad(): latent = self.codec.vq.idx_to_z_q(codes)
        with self.word_context(features, mask):
            return self.codec.decoder(latent.detach())

    def forward(self, features, mask, codes):
        return {"concept_logits": self.concepts.teacher_logits(features, mask, codes),
            "word_logits": self.word_logits(features, mask, codes)}

    @torch.no_grad()
    def generate(self, features, mask):
        if self.training: raise ValueError("source-only generation requires eval mode")
        codes = self.concepts.generate_codes(features, mask)
        words = self.word_logits(features, mask, codes).argmax(-1)
        return {"codes": codes, "raw_word_ids": words}

    def attention_state(self):
        return {"concept_attention": self.concepts.adapter_state(),
            "word_memory": {k: v.detach().cpu().clone() for k, v in self.word_memory.state_dict().items()},
            "word_adapters": {k: v.detach().cpu().clone() for k, v in self.word_adapters.state_dict().items()}}

    def load_attention_state(self, saved):
        if set(saved) != {"concept_attention", "word_memory", "word_adapters"}:
            raise ValueError("only the exact new attention state may load")
        self.concepts.load_adapter_state(saved["concept_attention"])
        self.word_memory.load_state_dict(saved["word_memory"], strict=True)
        self.word_adapters.load_state_dict(saved["word_adapters"], strict=True)

    def close(self):
        if self._word_bound is not None: raise ValueError("cannot remove active word hooks")
        self.concepts.close()
        for handle in self._word_handles: handle.remove()
        self._word_handles.clear()


def from_domain(device="cpu", **kwargs):
    if file_hash(DOMAIN_CHECKPOINT) != DOMAIN_SHA: raise ValueError("fixed task-domain decoder checkpoint changed")
    body = concepts.from_release(device="cpu")
    saved = torch.load(DOMAIN_CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    if saved["step"] != 1024: raise ValueError("fixed final domain decoder required")
    body.codec.decoder.load_state_dict(saved["decoder"], strict=True)
    if concepts.release.vendor.module_hash(body.codec) != DOMAIN_CODEC_HASH:
        raise ValueError("exact measured task-domain codec required")
    return SourceGroundedNCM(body, **kwargs).to(device).float()
