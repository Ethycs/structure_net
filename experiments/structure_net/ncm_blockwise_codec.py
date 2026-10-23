"""Expose independent learned-codec blocks as one NCM level-major hierarchy.

No weights are added or trained here. Learned resampling, convolutions and
normalization run within each original block, not across block boundaries.
"""
import torch
from torch import nn


def to_level_major(block_codes,levels,blocks):
    if block_codes.ndim!=2 or blocks<1 or len(block_codes)%blocks or block_codes.shape[1]!=sum(levels):
        raise ValueError("complete block-major hierarchy required")
    batch=len(block_codes)//blocks
    return torch.cat([chunk.reshape(batch,blocks*width) for chunk,width in
        zip(block_codes.split(tuple(levels),dim=1),levels)],dim=1)


def to_block_major(level_codes,levels,blocks):
    if level_codes.ndim!=2 or blocks<1 or level_codes.shape[1]!=blocks*sum(levels):
        raise ValueError("complete level-major hierarchy required")
    batch=len(level_codes)
    return torch.cat([chunk.reshape(batch*blocks,width) for chunk,width in
        zip(level_codes.split(tuple(blocks*v for v in levels),dim=1),levels)],dim=1)


class BlockwiseQuantizer:
    def __init__(self,original,blocks):
        self.original=original;self.blocks=blocks
        self.block_levels=tuple(original.levels)
        self.levels=tuple(blocks*v for v in self.block_levels)
        self.embed_dim=original.embed_dim
        self.embeddings=original.embeddings
        self.vqvae_vocab_sizes=original.vqvae_vocab_sizes

    @torch.no_grad()
    def idx_to_z_q(self,codes):
        folded=to_block_major(codes,self.block_levels,self.blocks)
        latent=self.original.idx_to_z_q(folded)
        return latent.reshape(len(codes),self.levels[-1],self.embed_dim)

    @torch.no_grad()
    def idx_to_ncm_input(self,codes):
        folded=to_block_major(codes,self.block_levels,self.blocks)
        values=self.original.idx_to_ncm_input(folded)
        batch=len(codes)
        return torch.cat([x.reshape(batch,self.blocks*width,self.embed_dim) for x,width in
            zip(values.split(self.block_levels[1:],dim=1),self.block_levels[1:])],dim=1)

    @torch.no_grad()
    def get_next_autoregressive_input(self,level_idx,z_hat,z_q_level):
        if not 0<=level_idx<len(self.levels):
            raise ValueError("invalid concept level")
        batch=len(z_hat);width=self.block_levels[level_idx];full=self.block_levels[-1]
        if z_hat.shape!=(batch,self.blocks*full,self.embed_dim) or z_q_level.shape!=(batch,self.blocks*width,self.embed_dim):
            raise ValueError("complete independent latent blocks required")
        full_latent,next_input=self.original.get_next_autoregressive_input(level_idx,
            z_hat.reshape(batch*self.blocks,full,self.embed_dim),
            z_q_level.reshape(batch*self.blocks,width,self.embed_dim))
        next_width=self.block_levels[min(level_idx+1,len(self.block_levels)-1)]
        return (full_latent.reshape(batch,self.blocks*full,self.embed_dim),
                next_input.reshape(batch,self.blocks*next_width,self.embed_dim))


class BlockwiseTextDecoder(nn.Module):
    def __init__(self,original,blocks,block_length,embed_dim):
        super().__init__();self.original=original
        self.blocks=blocks;self.block_length=block_length;self.embed_dim=embed_dim

    def train(self,mode=True):
        # Keep the frozen codec's dropout disabled even inside a trained head.
        return super().train(False)

    def forward(self,latent):
        if latent.ndim!=3 or latent.shape[1:]!=(self.blocks*self.block_length,self.embed_dim):
            raise ValueError("complete blockwise decoder frame required")
        batch=len(latent)
        values=self.original(latent.reshape(batch*self.blocks,self.block_length,self.embed_dim))
        return values.reshape(batch,self.blocks*self.block_length,values.shape[-1])


class BlockwiseCodec:
    """Frozen codec facade compatible with the released concept generator."""
    def __init__(self,original,blocks=4):
        if blocks<1 or original.training or any(p.requires_grad for p in original.parameters()):
            raise ValueError("an explicitly frozen eval codec and positive block count are required")
        self.original=original;self.blocks=blocks
        self.embed_dim=original.embed_dim;self.in_vocab_size=original.in_vocab_size
        self.context_length=blocks*original.context_length
        self.encoder=original.encoder
        self.vq=BlockwiseQuantizer(original.vq,blocks)
        self.decoder=BlockwiseTextDecoder(original.decoder,blocks,original.context_length,original.embed_dim).eval()

    @torch.no_grad()
    def encode(self,words):
        if words.ndim!=2 or words.shape[1]!=self.context_length:
            raise ValueError("full untruncated word frame required")
        codes=self.original.encode(words.reshape(len(words)*self.blocks,self.original.context_length))
        return to_level_major(codes,self.vq.block_levels,self.blocks)

    @torch.no_grad()
    def decode(self,codes):
        return self.decoder(self.vq.idx_to_z_q(codes))
