"""Bounded scoring for attention repair over native NCM code hierarchies.

This module is an unqualified mechanism. Oracle codes are accepted by the loss,
never by prediction. The parent, generator and native codebooks stay frozen.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    source_width: int = 3072
    code_width: int = 384
    width: int = 256
    heads: int = 4
    blocks: int = 2
    expansion: int = 4
    dropout: float = .1
    levels: tuple = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)
    vocabularies: tuple = (8192,) * 12
    seed: int = 129142
    score_scale: float = 4.


def bounded_vectors(value):
    value=value.float()
    return value/(1+value.norm(dim=-1,keepdim=True))


class RepairBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.dropout = c.heads, c.dropout
        self.self_norm = nn.LayerNorm(c.width)
        self.self_qkv = nn.Linear(c.width, 3 * c.width, bias=False)
        self.self_out = nn.Linear(c.width, c.width, bias=False)
        self.cross_norm = nn.LayerNorm(c.width)
        self.cross_q = nn.Linear(c.width, c.width, bias=False)
        self.cross_out = nn.Linear(c.width, c.width, bias=False)
        self.ff_norm = nn.LayerNorm(c.width)
        self.ff = nn.Sequential(nn.Linear(c.width, c.expansion * c.width), nn.GELU(),
            nn.Linear(c.expansion * c.width, c.width))

    def heads_view(self, x):
        return x.reshape(len(x), x.shape[1], self.heads, x.shape[-1] // self.heads).transpose(1, 2)

    def forward(self, x, memory, valid):
        dropout = self.dropout if self.training else 0.
        q, k, v = self.self_qkv(self.self_norm(x)).chunk(3, dim=-1)
        h = F.scaled_dot_product_attention(self.heads_view(q), self.heads_view(k),
            self.heads_view(v), dropout_p=dropout).transpose(1, 2).reshape_as(x)
        x = x + self.self_out(h)
        k, v = memory
        h = F.scaled_dot_product_attention(self.heads_view(self.cross_q(self.cross_norm(x))),
            k, v, attn_mask=valid[:, None, None, :], dropout_p=dropout).transpose(1, 2).reshape_as(x)
        x = x + self.cross_out(h)
        return x + self.ff(self.ff_norm(x))


class AttentionCodeRepair(nn.Module):
    def __init__(self, config=Config()):
        super().__init__()
        c = self.config = config
        if (c.width % c.heads or len(c.levels) != len(c.vocabularies) or c.blocks < 1
                or c.score_scale<=0 or not c.levels or any(v < 1 for v in c.levels)):
            raise ValueError('complete levels and divisible attention width required')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.seed)
            self.input = nn.Linear(c.code_width, c.width)
            self.position = nn.Embedding(sum(c.levels), c.width)
            self.level = nn.Embedding(len(c.levels), c.width)
            self.source_norm = nn.LayerNorm(c.source_width, elementwise_affine=False)
            self.source_kv = nn.Linear(c.source_width, 2 * c.width, bias=False)
            self.blocks = nn.ModuleList(RepairBlock(c) for _ in range(c.blocks))
            self.final_norm = nn.LayerNorm(c.width)
            self.delta = nn.Linear(c.width, c.code_width, bias=False)
            # A zero residual plus positive copy preference retains every input code.
            nn.init.zeros_(self.delta.weight)
            self.copy_preference = nn.Parameter(torch.tensor(2.))
        self.register_buffer('level_ids', torch.tensor([i for i, n in enumerate(c.levels) for _ in range(n)]))

    def forward(self, codes, features, valid, books):
        c = self.config
        if (codes.dtype != torch.long or codes.shape != (len(features), sum(c.levels))
                or features.ndim != 3 or features.shape[-1] != c.source_width
                or valid.shape != features.shape[:2] or valid.dtype != torch.bool
                or not valid.any(-1).all() or (valid[:, 1:] & ~valid[:, :-1]).any()
                or len(books) != len(c.levels)):
            raise ValueError('complete predicted codes and right-padded source features required')
        source = features.detach().masked_fill(~valid[..., None], 0)
        if not torch.isfinite(source).all():
            raise ValueError('finite valid source features required')
        offset, embedded = 0, []
        for n, size, book in zip(c.levels, c.vocabularies, books):
            part = codes[:, offset:offset + n]
            if (book.shape != (size, c.code_width) or book.requires_grad
                    or (part < 0).any() or (part >= size).any()):
                raise ValueError('exact frozen native books and in-range predicted codes required')
            embedded.append(F.embedding(part, bounded_vectors(book)).detach())
            offset += n
        original = torch.cat(embedded, dim=1)
        positions = torch.arange(sum(c.levels), device=codes.device)
        x = self.input(original) + self.position(positions)[None] + self.level(self.level_ids)[None]
        k, v = self.source_kv(self.source_norm(source)).chunk(2, dim=-1)
        shape = (len(features), features.shape[1], c.heads, c.width // c.heads)
        memory = (k.reshape(shape).transpose(1, 2), v.reshape(shape).transpose(1, 2))
        for block in self.blocks:
            x = block(x, memory, valid)
        query = original.float() + self.delta(self.final_norm(x)).float()
        # Classifier preconditioning only: native lookup/decoder books are unchanged.
        query = query/query.norm(dim=-1,keepdim=True).clamp_min(1.)
        offset, logits = 0, []
        for n, book in zip(c.levels, books):
            part = query[:, offset:offset + n]
            book = bounded_vectors(book)
            distance = (part.square().sum(-1, keepdim=True) + book.square().sum(-1)
                - 2 * (part @ book.T)).clamp_min(0.)
            scores = -c.score_scale*distance
            # The head uses actual codebook geometry instead of an unrelated class matrix.
            scores = scores.scatter_add(-1, codes[:, offset:offset + n, None],
                self.copy_preference.clamp(.1, 8.).expand(len(codes), n, 1))
            logits.append(scores)
            offset += n
        return logits

    @torch.no_grad()
    def repair(self, codes, features, valid, books):
        if self.training:
            raise ValueError('deterministic repair requires eval mode')
        return torch.cat([x.argmax(-1) for x in self(codes, features, valid, books)], dim=1)


def level_loss(logits, oracle, levels):
    if (len(logits) != len(levels) or oracle.dtype != torch.long
            or oracle.ndim != 2 or oracle.shape[1] != sum(levels)):
        raise ValueError('all complete clean target levels required in training loss')
    offset, losses = 0, []
    for scores, n in zip(logits, levels):
        if scores.shape[:2] != (len(oracle), n):
            raise ValueError('complete level output required')
        losses.append(F.cross_entropy(scores.flatten(0, 1), oracle[:, offset:offset + n].flatten()))
        offset += n
    return torch.stack(losses).mean()
