"""NCM adaptation: retain coarse concept levels and predict final text tokens.

This removes quantized text reconstruction from the final output contract.
It is a new architecture, not a replication or a trained quality result.
"""
from copy import deepcopy

import torch
from torch import nn


class WordFinalConceptHead(nn.Module):
    def __init__(self, source_head):
        super().__init__()
        self.source_head = source_head
        generator = source_head.generator
        codec = generator.vqvae_local[0]
        original = getattr(codec, "original", codec)
        self.blocks = getattr(codec, "blocks", 1)
        self.block_length = original.context_length
        self.width = generator.levels[-1]
        if self.width != self.blocks * self.block_length:
            raise ValueError("full text-token level must cover every independent codec block")
        # Reuse the released finest-level projection and a new decoder copy.
        # The frozen codec itself is never changed or registered in this head.
        self.word_projection = deepcopy(generator.lm_heads[-1].proj).requires_grad_(True)
        self.word_decoder = deepcopy(original.decoder).requires_grad_(True)
        generator.lm_heads[-1].requires_grad_(False)

    @property
    def generator(self):
        return self.source_head.generator

    def _word_logits(self, hidden, prior):
        latent = prior + self.word_projection(hidden)
        batch, width, dim = latent.shape
        if width != self.width:
            raise ValueError("the entire text frame must be decoded without clipping")
        logits = self.word_decoder(latent.reshape(batch*self.blocks, self.block_length, dim))
        return logits.reshape(batch, width, logits.shape[-1])

    def forward(self, features, source_mask, target_concepts):
        """No target word IDs or finest VQ labels enter prediction inputs."""
        g = self.generator
        codec = g.vqvae_local[0]
        source = self.source_head.source_latents(features, source_mask)
        previous = codec.vq.idx_to_ncm_input(target_concepts).to(source.dtype)
        queries = torch.cat((g.build_l0_query_input(len(source), dtype=source.dtype), previous), dim=1)
        target = g.transformer.wte(queries) + g.transformer.wle(g.lvl_1L)
        prefix = g.transformer.wte(source)
        hidden = torch.cat((prefix, target), dim=1)
        mask = g.attn_mask_for_masking + g._prefix_pad_bias(
            ~source_mask.bool(), hidden.shape[1], g.attn_mask_for_masking.dtype, hidden.device)
        for block in g.transformer.blocks:
            hidden = block(hidden, attn_mask=mask, rope_cos_sin=(g.rope_cos, g.rope_sin))
        hidden = g.transformer.ln_f(hidden)[:, g.prefix_len:]
        coarse, position = [], 0
        for level, width in enumerate(g.levels[:-1]):
            coarse.append(g.lm_heads[level](hidden[:, position:position+width]))
            position += width
        words = self._word_logits(hidden[:, position:], previous[:, -self.width:])
        return {"coarse_logits": coarse, "word_logits": words}

    def _level_hidden(self, prefix, source_mask, conditioning, level, position):
        g = self.generator
        query = g.transformer.wte(conditioning)
        width = g.levels[level]
        query = query + g.transformer.wle(g.lvl_1L[:, position:position+width])
        hidden = torch.cat((prefix, query), dim=1)
        size = hidden.shape[1]
        mask = torch.zeros(1,1,size,size,dtype=hidden.dtype,device=hidden.device)
        mask[:, :, :g.prefix_len, g.prefix_len:] = -torch.inf
        mask = mask + g._prefix_pad_bias(~source_mask.bool(),size,hidden.dtype,hidden.device)
        rope = (getattr(g,f"rope_cos_gen_{level}"),getattr(g,f"rope_sin_gen_{level}"))
        for block in g.transformer.blocks:
            hidden = block(hidden,attn_mask=mask,rope_cos_sin=rope)
        return g.transformer.ln_f(hidden)[:,g.prefix_len:]

    @torch.no_grad()
    def generate(self, features, source_mask):
        if self.training:
            raise ValueError("the complete new head must be in eval mode")
        g = self.generator
        codec = g.vqvae_local[0]
        source = self.source_head.source_latents(features,source_mask)
        prefix = g.transformer.wte(source)
        reconstruction = torch.zeros(len(source),self.width,codec.embed_dim,dtype=source.dtype,device=source.device)
        conditioning = g.build_l0_query_input(len(source),dtype=source.dtype)
        coarse, position = [], 0
        for level,width in enumerate(g.levels):
            hidden = self._level_hidden(prefix,source_mask,conditioning,level,position)
            if level == len(g.levels)-1:
                return {"coarse_codes":coarse,"raw_word_ids":self._word_logits(hidden,conditioning).argmax(-1)}
            current = g.lm_heads[level](hidden).argmax(-1)
            coarse.append(current)
            reconstruction,conditioning = codec.vq.get_next_autoregressive_input(
                level,reconstruction,codec.vq.embeddings[level](current))
            conditioning = conditioning.to(source.dtype)
            position += width
        raise RuntimeError("the word level was not reached")
