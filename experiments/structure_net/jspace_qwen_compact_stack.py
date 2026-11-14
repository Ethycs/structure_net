"""Arbitrary contiguous compressed Qwen suffix with fixed coordinate bridges.

Owns its supplied blocks; callers preserving a teacher must deepcopy first.
Training uses differentiable Torch operations, optionally checkpointed per block.
Packed BF16 inference can use the frozen Triton pointwise kernels. A Turing
worker must explicitly choose FP16 activations/AMP and geometry_compute_dtype;
FP16 execution never takes the BF16 Triton path. Worker/master rounding may
differ and requires post-pack BF16 validation on the deployment card.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .jspace_qwen_affine_block import AffineQwenBlock
from .jspace_qwen_affine_fast import _run_reduced, triton


class _Bridge(nn.Module):
    def __init__(self, first, second):
        super().__init__()
        a, b = first.position_count, second.position_count
        if a != b and min(a, b) != 1:
            raise ValueError("Adjacent anchor tables need equal lengths or a global row")
        if min(a, b) > 1 and first.position_policy != second.position_policy:
            raise ValueError("Adjacent position-specific blocks need the same position policy")
        self.position_source = 0 if a > 1 else 1
        count = max(a, b)
        with torch.no_grad(), torch.autocast(device_type=first.basis.device.type, enabled=False):
            dtype = torch.float64 if first.basis.dtype == second.basis.dtype == torch.float64 else torch.float32
            u, v = first.basis.to(dtype), second.basis.to(dtype)
            self.register_buffer("weight", (v.T @ u).detach().contiguous())
            delta = first.anchor.expand(count, -1) - second.anchor.expand(count, -1)
            self.register_buffer("shift", (delta.to(v.dtype) @ v).detach().contiguous())


class CompactQwenStack(nn.Module):
    """Encode once, execute any adjacent block suffix, reconstruct once.

    ``geometry_mode`` fp32 retains the supplied arithmetic-precision geometry;
    bf16/fp16 replaces it with quantized fixed buffers. Explicit FP16 worker
    compute may read the same stored BF16 geometry via temporary FP16 casts.
    Do not call .half() on the whole stack: anchors/energies remain FP32.

    After pack(strip_unused=True), middle bases and nonexit native anchor tables
    are absent from the deployed state. Standalone constituent block.forward()
    is then unsupported; retain external fit artifacts for later recomposition.
    Reload: construct matching fitted blocks/stack, pack with matching stripping,
    then load_state_dict(strict=True). output_attentions returns one tuple of
    constituent tensors rather than ordinary separate HF decoder entries.
    """

    def __init__(self, blocks, *, geometry_mode="fp32", geometry_compute_dtype=None,
                 pad_to_multiple=128, enabled=True, checkpoint_blocks=False):
        super().__init__()
        blocks = list(blocks)
        if not blocks or any(not isinstance(b, AffineQwenBlock) for b in blocks):
            raise ValueError("Expected one or more compiled AffineQwenBlocks")
        if len({id(b) for b in blocks}) != len(blocks):
            raise ValueError("Stack blocks must be distinct")
        if any(b.layer_idx != blocks[0].layer_idx + i for i, b in enumerate(blocks)):
            raise ValueError("Blocks must preserve contiguous original cache indices")
        if any(b.width != blocks[0].width or b.native_head_dim != blocks[0].native_head_dim
               or b.attention_type != blocks[0].attention_type for b in blocks):
            raise ValueError("Blocks must share native width, rotary width and attention-mask type")
        if any(b.basis is None or b.anchor is None for b in blocks):
            raise ValueError("Construct bridges before stripping; recover geometry from external fit artifacts")
        if any(b.basis.device != blocks[0].basis.device or b.basis.dtype != blocks[0].basis.dtype for b in blocks):
            raise ValueError("Source bases must share device and dtype")
        if geometry_mode not in ("fp32", "bf16", "fp16"):
            raise ValueError("geometry_mode must be fp32, bf16 or fp16")
        if geometry_mode == "fp32" and any(b.basis.dtype not in (torch.float32, torch.float64) for b in blocks):
            raise ValueError("FP32 control requires original arithmetic-precision geometry")
        if geometry_compute_dtype not in (None, torch.float16, torch.bfloat16):
            raise ValueError("Worker geometry compute override must be FP16 or BF16")
        if geometry_compute_dtype is not None and geometry_mode == "fp32":
            raise ValueError("Quantized worker compute is not an FP32 geometry control")
        if isinstance(pad_to_multiple, bool) or not isinstance(pad_to_multiple, int) or pad_to_multiple < 0:
            raise ValueError("pad_to_multiple must be a nonnegative integer")
        if geometry_mode != "fp32" and any(b.anchor.dtype != torch.float32 or b.anchor_energy.dtype != torch.float32 for b in blocks):
            raise ValueError("Quantized geometry requires FP32 native anchors and energies")
        self.blocks = nn.ModuleList(blocks)
        self.bridges = nn.ModuleList([_Bridge(a, b) for a, b in zip(blocks, blocks[1:])])
        self.geometry_mode, self.geometry_compute_dtype = geometry_mode, geometry_compute_dtype
        self.pad_to_multiple, self.enabled = pad_to_multiple, enabled
        self.checkpoint_blocks, self.stripped = checkpoint_blocks, False
        self.width, self.layer_idx, self.attention_type = blocks[0].width, blocks[0].layer_idx, blocks[0].attention_type
        self.layer_indices = tuple(b.layer_idx for b in blocks)
        self.ranks = tuple(b.rank for b in blocks)
        if geometry_mode != "fp32":
            dtype = torch.bfloat16 if geometry_mode == "bf16" else torch.float16
            for block in blocks:
                block.basis = block.basis.to(dtype)
            for bridge in self.bridges:
                bridge.weight = bridge.weight.to(dtype)
        for i, block in enumerate(blocks):
            if block.has_polynomial:
                prefix = torch.cat((torch.zeros(1, dtype=torch.long, device=block.basis.device),
                                    torch.arange(len(block.polynomial_anchor_positions), device=block.basis.device)))
                fi, fj = torch.cat((prefix, block.polynomial_pair_i)), torch.cat((prefix, block.polynomial_pair_j))
            else:
                fi = fj = torch.empty(0, dtype=torch.long, device=block.basis.device)
            self.register_buffer(f"feature_i_{i}", fi); self.register_buffer(f"feature_j_{i}", fj)
            self._pad(block)
        self.training = blocks[0].training

    @property
    def packed(self):
        return all(b.packed for b in self.blocks)

    def compact_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _pad(self, block):
        if not block.packed:
            return
        logical = block.intermediate_size + block.polynomial_feature_count
        target = ((logical + self.pad_to_multiple - 1) // self.pad_to_multiple * self.pad_to_multiple
                  if self.pad_to_multiple else logical)
        if block.mlp_output.shape[1] > target:
            raise ValueError("Existing packed width exceeds the requested padding")
        if block.mlp_output.shape[1] < target:
            block.mlp_output = nn.Parameter(F.pad(block.mlp_output.detach(),
                (0, target - block.mlp_output.shape[1])).contiguous(), requires_grad=False)

    def _geometry(self, coordinates, weight, shift=None):
        dtype = self.geometry_compute_dtype or weight.dtype
        with torch.autocast(device_type=coordinates.device.type, enabled=False):
            result = F.linear(coordinates.to(dtype), weight.to(dtype))
            if shift is not None:
                result = result.to(shift.dtype) + shift
            return result.to(coordinates.dtype)

    def _can_fuse(self, hidden):
        return (self.enabled and triton is not None and hidden.is_cuda and hidden.dtype == torch.bfloat16
                and self.geometry_compute_dtype != torch.float16 and not torch.is_grad_enabled() and not self.training
                and all(b.packed and not b.training and not b.capture_mlp
                        and b.qkv_weight.dtype == b.gate_up_weight.dtype == b.mlp_output.dtype == torch.bfloat16
                        and b.qkv_shift.dtype in (torch.float32, torch.bfloat16)
                        and b.gate_up_shift.dtype in (torch.float32, torch.bfloat16)
                        and (not b.has_polynomial or b.polynomial_anchor_mean.dtype == torch.float32) for b in self.blocks))

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.width or position_embeddings is None:
            raise ValueError("Native [batch, sequence, width] input and rotary embeddings are required")
        if self.geometry_compute_dtype is not None and hidden_states.dtype != self.geometry_compute_dtype:
            raise ValueError("Explicit worker compute requires matching activation dtype")
        checkpointing = self.checkpoint_blocks and torch.is_grad_enabled()
        if checkpointing and (past_key_value is not None or use_cache or output_attentions or any(b.capture_mlp for b in self.blocks)):
            raise ValueError("Activation checkpointing requires uncached execution without attention/MLP capture")
        positions = [b._positions(hidden_states, cache_position, position_ids, past_key_value) for b in self.blocks]
        fused = self._can_fuse(hidden_states)
        z = self._geometry(hidden_states, self.blocks[0].basis.T)
        attentions = []
        for i, block in enumerate(self.blocks):
            if i:
                bridge = self.bridges[i - 1]
                p = positions[i - 1 + bridge.position_source]
                shift = block._rows(bridge.shift, p, *z.shape[:2])
                z = self._geometry(z, bridge.weight, shift)
            def run(value, index=i, current=block):
                return _run_reduced(current, value, positions[index], attention_mask=attention_mask,
                    past_key_value=past_key_value, cache_position=cache_position, position_embeddings=position_embeddings,
                    fuse_qkv=fused, fuse_mlp=fused,
                    padded_width=current.mlp_output.shape[1] if current.packed else current.intermediate_size + current.polynomial_feature_count,
                    feature_i=getattr(self, f"feature_i_{index}"), feature_j=getattr(self, f"feature_j_{index}"))
            if checkpointing:
                def coordinates_only(value, execute=run):
                    return execute(value)[0]
                z = checkpoint(coordinates_only, z, use_reentrant=False)
            else:
                z, probabilities = run(z)
                if output_attentions:
                    attentions.append(probabilities)
        last = self.blocks[-1]
        anchor = last._rows(last.anchor, positions[-1], *hidden_states.shape[:2])
        output = self._geometry(z, last.basis, anchor)
        return (output, tuple(attentions)) if output_attentions else (output,)

    def pack(self, dtype=None, buffer_dtype=None, *, strip_unused=True):
        if self.stripped and not strip_unused:
            raise ValueError("Stripped geometry cannot be restored without external fit artifacts")
        for block in self.blocks:
            if not block.packed:
                basis = block.basis
                if basis.dtype in (torch.float16, torch.bfloat16):
                    block.basis = basis.float()
                try:
                    block.pack(dtype=dtype, buffer_dtype=buffer_dtype)
                finally:
                    block.basis = basis
            self._pad(block)
        if strip_unused:
            for block in self.blocks[1:-1]:
                block.basis = None
            for block in self.blocks[:-1]:
                block.anchor = None
            self.stripped = True
        return self

    def get_extra_state(self):
        return {"version": 1, "geometry_mode": self.geometry_mode, "pad_to_multiple": self.pad_to_multiple,
                "layer_indices": self.layer_indices, "ranks": self.ranks, "packed": self.packed, "stripped": self.stripped}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Construct and pack the same compact-stack structure before strict reload")

    def storage_breakdown(self):
        parameters = sum(p.numel() * p.element_size() for p in self.parameters())
        buffers = sum(b.numel() * b.element_size() for b in self.buffers())
        bridges = sum(b.numel() * b.element_size() for bridge in self.bridges for b in bridge.buffers())
        return {"parameters": parameters, "buffers": buffers, "bridges_included_in_buffers": bridges,
                "total": parameters + buffers, "stripped": self.stripped}


def estimate_qwen3b_stack_bytes(rank=1024, blocks=36, *, anchors=24, removed=1024, pad_to_multiple=128,
                                strip_unused=True, geometry_bytes=2):
    """Estimate current Qwen2.5-3B layout, counting every deployed tensor category.

    Assumes BF16 learned weights/position numerators, FP32 biases/anchors/energy,
    rank-wide residuals, 64-wide heads, 416 position rows and the current graph
    feature/index buffers. Native total comes from the measured pinned model.
    This is a storage projection, not a quality or throughput measurement.
    """
    if not 1 <= blocks <= 36 or not 1 <= rank <= 2048 or not 0 < anchors < 11008 - removed:
        raise ValueError("Invalid Qwen3B rank, block count, anchors or removal budget")
    if geometry_bytes not in (2, 4) or pad_to_multiple <= 0:
        raise ValueError("Expected two/four-byte geometry and positive padding alignment")
    d, p, nq, nkv, hd, native_hd, native_mlp = 2048, 416, 16, 2, 64, 128, 11008
    keep = native_mlp - removed
    features = 1 + anchors + anchors * (anchors + 1) // 2
    padded = (keep + features + pad_to_multiple - 1) // pad_to_multiple * pad_to_multiple
    qkv = (nq + 2 * nkv) * hd
    learned = 2 * rank * (qkv + nq * hd + 2 * keep + padded) + 2 * qkv + 8 * rank
    basis, anchor = geometry_bytes * d * rank, 4 * p * d
    positions = 2 * p * (qkv + 2 * keep) + 4 * p
    metadata = (8 * (nq + nkv) * hd + 4 * nkv * native_hd * hd + 16 * anchors
                + 16 * (anchors * (anchors + 1) // 2) + 8 * native_mlp + 16 * features)
    one_block = learned + basis + anchor + positions + metadata
    bridges = max(blocks - 1, 0) * (geometry_bytes * rank * rank + 4 * p * rank)
    stripped = (max(blocks - 2, 0) * basis + max(blocks - 1, 0) * anchor) if strip_unused else 0
    native_total, native_block = 6171877632, 154153984
    total = native_total - blocks * native_block + blocks * one_block + bridges - stripped
    return {"rank": rank, "blocks": blocks, "one_block_bytes": one_block, "bridge_bytes": bridges,
            "removed_unused_geometry_bytes": stripped, "full_model_bytes": total,
            "native_full_model_bytes": native_total, "reduction_fraction": 1 - total / native_total,
            "margin_below_38_percent_limit_bytes": native_total * .62 - total,
            "assumptions": "BF16 parameters/tables; FP32 anchors/energy; fixed native tied embedding/readout; no adapter",
            "measured": False}
