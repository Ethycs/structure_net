"""Read-only compact-coordinate observations; no training, selection or cache I/O.

The stack calls a reduced executor directly, so ordinary block-forward hooks do
not observe its states. A serialized, thread-scoped observer intercepts that
executor without replacing its arithmetic, then restores the binding in finally.
Callers authenticate checkpoints/data and own reservations and output artifacts.
"""
from contextlib import nullcontext
import threading

import torch

from . import jspace_qwen_compact_stack as compact

_TRACE_LOCK = threading.Lock()


def _guard(hidden, guard):
    if hidden.device.type == 'cuda' and guard is None:
        raise ValueError('CUDA feature diagnostics require a reservation guard')
    if guard is not None:
        guard()


def trace_coordinates(stack, hidden, *, position_ids, position_embeddings,
                      attention_mask=None, layers=None, mode='inference', guard=None):
    """Return actual post-block coordinates and native-width output, detached.

    ``inference`` preserves the existing packed/fused inference eligibility.
    ``differentiable`` enables input autograd on the SAME frozen weights; no
    backward is run, no parameter flag or .grad changes, and fusion is disabled
    by the original executor itself. It can transiently retain an autograd graph.
    Returned tensors remain on the input device for bounded caller-side streaming.
    """
    _guard(hidden, guard)
    if not isinstance(stack, compact.CompactQwenStack):
        raise TypeError('Expected the existing CompactQwenStack')
    if any(m.training for m in stack.modules()) or any(p.requires_grad for p in stack.parameters()):
        raise ValueError('Trace a detached frozen eval checkpoint, not a live trainer')
    if mode not in ('inference', 'differentiable'):
        raise ValueError('Unknown arithmetic mode')
    if mode == 'differentiable' and any(t.is_inference() for t in (*stack.parameters(), *stack.buffers())):
        raise ValueError('Differentiable diagnostics require normally constructed frozen tensors')
    chosen = tuple(stack.layer_indices if layers is None else layers)
    if (not chosen or len(set(chosen)) != len(chosen) or
            any(type(i) is not int or i not in stack.layer_indices for i in chosen)):
        raise ValueError('Requested layers must be unique original layer indices')
    if hidden.ndim != 3 or hidden.shape[-1] != stack.width:
        raise ValueError('Expected native-width [batch, tokens, width] input')
    if position_ids.shape != hidden.shape[:2] or position_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Explicit per-case absolute positions required')
    if not _TRACE_LOCK.acquire(blocking=False):
        raise RuntimeError('Another compact-coordinate trace is active')
    original = compact._run_reduced
    owner = threading.get_ident()
    block_ids = {id(b): b.layer_idx for b in stack.blocks}
    values, execution = {}, {}

    def observed(block, z, positions, **kwargs):
        result = original(block, z, positions, **kwargs)
        if threading.get_ident() == owner and id(block) in block_ids and block.layer_idx in chosen:
            if block.layer_idx in values:
                raise RuntimeError('Diagnostic expects exactly one forward visit per layer')
            values[block.layer_idx] = result[0].detach().clone()
            execution[block.layer_idx] = dict(fuse_qkv=kwargs['fuse_qkv'], fuse_mlp=kwargs['fuse_mlp'])
        return result

    compact._run_reduced = observed
    try:
        # Disable any caller's outer inference_mode before creating autograd input.
        outer = torch.inference_mode(False) if mode == 'differentiable' else nullcontext()
        inner = torch.enable_grad() if mode == 'differentiable' else torch.inference_mode()
        with outer, inner:
            x = hidden.detach().clone().requires_grad_(True) if mode == 'differentiable' else hidden
            ids, rope, mask = position_ids, position_embeddings, attention_mask
            if mode == 'differentiable':
                ids = ids.detach().clone()
                rope = tuple(t.detach().clone() for t in rope)
                mask = None if mask is None else mask.detach().clone()
            output = stack(x, attention_mask=mask, position_ids=ids,
                position_embeddings=rope, use_cache=False)[0].detach().clone()
        if set(values) != set(chosen):
            raise RuntimeError('Incomplete coordinate trace')
        _guard(hidden, guard)
        return dict(mode=mode, output=output, coordinates=values, execution=execution,
                    coordinate_location='post-block, before outgoing fixed bridge')
    finally:
        compact._run_reduced = original
        _TRACE_LOCK.release()


@torch.no_grad()
def native_final_coordinates(stack, native_final_hidden, *, guard=None):
    """Source-consistent native h36 -> z35 projection, NOT an exact inverse.

    The compiler's source convention is h @ U (its original FP32 anchor is
    orthogonal to U). Runtime BF16 U is only approximately orthonormal; do not
    silently subtract an anchor, replace U, solve a Gram system or use a pinverse.
    """
    _guard(native_final_hidden, guard)
    last = stack.blocks[-1]
    if last.basis is None or native_final_hidden.shape[-1] != stack.width:
        raise ValueError('Retained exit basis and matching native width required')
    return stack._geometry(native_final_hidden, last.basis.T)


@torch.no_grad()
def reconstruct_final(stack, coordinates, position_ids, *, guard=None):
    """Apply the exact stored exit basis/anchor and runtime rounding convention."""
    _guard(coordinates, guard)
    last = stack.blocks[-1]
    if last.basis is None or last.anchor is None or coordinates.shape[-1] != last.rank:
        raise ValueError('Retained exit geometry and matching compact rank required')
    positions = last._positions(coordinates, None, position_ids, None)
    anchor = last._rows(last.anchor, positions, *coordinates.shape[:2])
    return stack._geometry(coordinates, last.basis, anchor)


@torch.no_grad()
def feature_metrics(actual, reference, lengths, *, guard=None):
    """Descriptive equal-case metrics, excluding padded tokens; never a gate."""
    _guard(actual, guard)
    if actual.shape != reference.shape or actual.ndim != 3 or not actual.shape[-1]:
        raise ValueError('Aligned [cases, tokens, coordinates] tensors required')
    if len(lengths) != len(actual) or any(type(n) is not int or not 0 < n <= actual.shape[1] for n in lengths):
        raise ValueError('One valid length per case required')
    per_case = []
    for a, b, length in zip(actual, reference, lengths):
        a, b = a[:length].float(), b[:length].float()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise FloatingPointError('Nonfinite valid feature rows')
        error = (a-b).square().mean(-1)
        relative = error / b.square().mean(-1).clamp_min(1e-20)
        per_case.append(dict(mse=float(error.mean()), relative_mse=float(relative.mean()),
            answer_mse=float(error[-1]), answer_relative_mse=float(relative[-1]),
            max_abs=float((a-b).abs().max())))
    if not per_case:
        raise ValueError('Empty feature comparison')
    mean = lambda key: sum(row[key] for row in per_case) / len(per_case)
    return dict(count=len(per_case), valid_tokens=sum(lengths), mse=mean('mse'),
        relative_rmse=mean('relative_mse')**.5, answer_mse=mean('answer_mse'),
        answer_relative_rmse=mean('answer_relative_mse')**.5,
        max_abs=max(row['max_abs'] for row in per_case), per_case=per_case,
        scope='diagnostic only; equal case weight and valid-token mean; no acceptance threshold')


def packed_arithmetic_floor(stack, hidden, lengths, **kwargs):
    """Same packed checkpoint: inference versus input-autograd Torch execution."""
    if not stack.packed:
        raise ValueError('The arithmetic floor requires the same packed checkpoint')
    inferred = trace_coordinates(stack, hidden, mode='inference', **kwargs)
    differentiable = trace_coordinates(stack, hidden, mode='differentiable', **kwargs)
    return dict(by_layer={i:feature_metrics(differentiable['coordinates'][i], z, lengths, guard=kwargs.get('guard'))
                         for i,z in inferred['coordinates'].items()},
        output=feature_metrics(differentiable['output'], inferred['output'], lengths, guard=kwargs.get('guard')),
        inference_execution=inferred['execution'], differentiable_execution=differentiable['execution'],
        scope='arithmetic difference on identical packed weights, not recovery error')
