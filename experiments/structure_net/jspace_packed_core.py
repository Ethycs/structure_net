"""Dense packing of a polynomial core and nonlinear residual for Qwen MLPs."""
from itertools import combinations_with_replacement

import torch
from torch import nn
from torch.nn import functional as F
import triton
import triton.language as tl


@triton.jit
def _pack_kernel(PRE, OUT, ANCHOR, MEAN, SCALE, COEFF, DMEAN, DSCALE, FI, FJ,
                 KEEP: tl.constexpr, PAD: tl.constexpr, REMOVED: tl.constexpr,
                 ANCHORS: tl.constexpr, NF: tl.constexpr, FB: tl.constexpr,
                 KEEP_BLOCKS: tl.constexpr, BLOCK: tl.constexpr, RB: tl.constexpr):
    token = tl.program_id(0)
    block = tl.program_id(1)
    if block < KEEP_BLOCKS:
        i = block * BLOCK + tl.arange(0, BLOCK)
        gate = tl.load(PRE + token * (2 * PAD) + i, i < KEEP, 0).to(tl.float32)
        up = tl.load(PRE + token * (2 * PAD) + PAD + i, i < KEEP, 0).to(tl.float32)
        # Match the two BF16 rounding points in torch SiLU(gate) * up.
        silu = (gate * tl.sigmoid(gate)).to(tl.bfloat16).to(tl.float32)
        hidden = (silu * up).to(tl.bfloat16)
        tl.store(OUT + token * (KEEP + REMOVED) + i, hidden, i < KEEP)
    else:
        r = (block - KEEP_BLOCKS) * RB + tl.arange(0, RB)
        f = tl.arange(0, FB)
        ai = tl.load(FI + f, f < NF, 0)
        aj = tl.load(FJ + f, f < NF, 0)
        pi = tl.load(ANCHOR + ai)
        pj = tl.load(ANCHOR + aj)
        gi = tl.load(PRE + token * (2 * PAD) + pi).to(tl.float32)
        gj = tl.load(PRE + token * (2 * PAD) + pj).to(tl.float32)
        ui = tl.load(PRE + token * (2 * PAD) + PAD + pi).to(tl.float32)
        uj = tl.load(PRE + token * (2 * PAD) + PAD + pj).to(tl.float32)
        hi = ((gi * tl.sigmoid(gi)).to(tl.bfloat16).to(tl.float32) * ui).to(tl.bfloat16).to(tl.float32)
        hj = ((gj * tl.sigmoid(gj)).to(tl.bfloat16).to(tl.float32) * uj).to(tl.bfloat16).to(tl.float32)
        zi = (hi - tl.load(MEAN + ai)) / tl.load(SCALE + ai)
        zj = (hj - tl.load(MEAN + aj)) / tl.load(SCALE + aj)
        feature = tl.where(f == 0, 1., tl.where(f <= ANCHORS, zi, zi * zj))
        coeff = tl.load(COEFF + r[:, None] * NF + f[None, :],
                        (r[:, None] < REMOVED) & (f[None, :] < NF), 0.)
        prediction = tl.sum(coeff * feature[None, :], 1)
        prediction = prediction * tl.load(DSCALE + r, r < REMOVED, 0.) + tl.load(DMEAN + r, r < REMOVED, 0.)
        tl.store(OUT + token * (KEEP + REMOVED) + KEEP + r, prediction, r < REMOVED)


class StackedCoreMLP(nn.Module):
    """One gate/up GEMM, one fused packing kernel, one dense output GEMM.

    The dense activation is [retained nonlinear channels, polynomial channels].
    Removed gate/up rows are absent; at most 15 zero rows pad the retained GEMM.
    The 153-term polynomial coefficient block stays factored and is not expanded
    into an output-by-quadratic weight matrix. Layers remain sequential.
    """
    def __init__(self, original, fit=None, removed=()):
        super().__init__()
        if any(layer.bias is not None for layer in (original.gate_proj, original.up_proj, original.down_proj)):
            raise ValueError("This Qwen packing implementation requires bias-free projections")
        dtype, device = original.gate_proj.weight.dtype, original.gate_proj.weight.device
        calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        n, width = original.gate_proj.weight.shape
        removed = list(removed)
        if len(set(removed)) != len(removed) or any(i < 0 or i >= n for i in removed):
            raise ValueError("Invalid removed-channel set")
        keep = [i for i in range(n) if i not in set(removed)]
        self.kept, self.removed = len(keep), len(removed)
        self.padded = triton.cdiv(self.kept, 16) * 16
        matrices = [F.pad(layer.weight.detach()[keep], (0, 0, 0, self.padded-self.kept))
                    for layer in (original.gate_proj, original.up_proj)]
        self.gate_up_weight = nn.Parameter(torch.cat(matrices, 0), requires_grad=False)
        self.down_weight = nn.Parameter(original.down_proj.weight.detach()[:, keep+removed].contiguous(), requires_grad=False)
        if removed:
            anchors = list(fit["anchors"])
            if set(anchors) & set(removed):
                raise ValueError("Cannot eliminate an anchor")
            self.register_buffer("anchor_positions", torch.tensor([keep.index(a) for a in anchors], device=device))
            for name, tensor in {
                "anchor_mean": fit["mean"][anchors], "anchor_scale": fit["scale"][anchors],
                "coefficient": fit["coefficients"]["2"][removed],
                "dropped_mean": fit["mean"][removed], "dropped_scale": fit["scale"][removed],
            }.items():
                self.register_buffer(name, tensor.to(device=device, dtype=calc_dtype).contiguous())
            pairs = list(combinations_with_replacement(range(len(anchors)), 2))
            fi = [0, *range(len(anchors)), *[i for i, j in pairs]]
            fj = [0, *range(len(anchors)), *[j for i, j in pairs]]
        else:
            self.register_buffer("anchor_positions", torch.zeros(1, device=device, dtype=torch.long))
            for name in ("anchor_mean", "anchor_scale", "coefficient", "dropped_mean", "dropped_scale"):
                self.register_buffer(name, torch.ones(1, device=device, dtype=calc_dtype))
            fi, fj = [0], [0]
        self.register_buffer("feature_i", torch.tensor(fi, device=device, dtype=torch.long))
        self.register_buffer("feature_j", torch.tensor(fj, device=device, dtype=torch.long))

    def reference(self, x):
        pre = F.linear(x, self.gate_up_weight)
        hidden = F.silu(pre[..., :self.padded]) * pre[..., self.padded:]
        kept = hidden[..., :self.kept]
        if self.removed:
            z = (kept[..., self.anchor_positions].to(self.coefficient.dtype)-self.anchor_mean)/self.anchor_scale
            a = len(self.anchor_positions)
            phi = torch.cat([torch.ones_like(z[..., :1]), z,
                             z[..., self.feature_i[a+1:]]*z[..., self.feature_j[a+1:]]], -1)
            predicted = F.linear(phi, self.coefficient)*self.dropped_scale+self.dropped_mean
            kept = torch.cat([kept, predicted.to(kept.dtype)], -1)
        return F.linear(kept, self.down_weight)

    def forward(self, x):
        if x.device.type != "cuda" or x.dtype != torch.bfloat16:
            raise ValueError("The fused kernel requires CUDA BF16; use reference() for CPU auditing")
        pre = F.linear(x, self.gate_up_weight)
        rows = pre.numel() // (2*self.padded)
        hidden = torch.empty((*x.shape[:-1], self.kept+self.removed), device=x.device, dtype=x.dtype)
        kept_blocks = triton.cdiv(self.kept, 256)
        _pack_kernel[(rows, kept_blocks+triton.cdiv(self.removed, 4))](
            pre, hidden, self.anchor_positions, self.anchor_mean, self.anchor_scale,
            self.coefficient, self.dropped_mean, self.dropped_scale, self.feature_i, self.feature_j,
            self.kept, self.padded, self.removed, len(self.anchor_positions), len(self.feature_i),
            triton.next_power_of_2(len(self.feature_i)), kept_blocks, 256, 4,
            num_warps=4, enable_fp_fusion=False)
        return F.linear(hidden, self.down_weight)
