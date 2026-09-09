"""Inference kernel for exact k=2 causal MoBA, without dense score matrices.

One Triton program handles one query. Only its local and best completed past
block enter token attention. This is a candidate kernel, not a speed claim.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pool(K, M, N: tl.constexpr, H: tl.constexpr, KH: tl.constexpr, D: tl.constexpr,
          KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
          C: tl.constexpr, NC: tl.constexpr, DD: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // KH, bh % KH
    t = block * C + tl.arange(0, C)
    d = tl.arange(0, DD)
    k = tl.load(K + b * KS0 + h * KS1 + t[:, None] * KS2 + d[None, :],
                (t[:, None] < N) & (d[None, :] < D), 0).to(tl.float32)
    mean = tl.sum(k, 0) / tl.minimum(C, N - block * C)
    # The paper's implementation pools in the input dtype before FP32 routing.
    mean = mean.to(K.dtype.element_ty).to(tl.float32)
    tl.store(M + (bh * NC + block) * D + d, mean, d < D)


@triton.jit
def _attend(Q, K, V, M, O, ROUTES, N: tl.constexpr, H: tl.constexpr, KH: tl.constexpr,
            D: tl.constexpr, QS0: tl.constexpr, QS1: tl.constexpr, QS2: tl.constexpr,
            KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
            VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
            C: tl.constexpr, NC: tl.constexpr, CC: tl.constexpr, DD: tl.constexpr):
    pos, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    kh = h // (H // KH)
    d = tl.arange(0, DD)
    q = tl.load(Q + b * QS0 + h * QS1 + pos * QS2 + d, d < D, 0).to(tl.float32)
    blocks = tl.arange(0, CC)
    means = tl.load(M + ((b * KH + kh) * NC + blocks[:, None]) * D + d[None, :],
                    (blocks[:, None] < NC) & (d[None, :] < D), 0).to(tl.float32)
    logits = tl.sum(means * q[None, :], 1)
    logits = tl.where(blocks < pos // C, logits, -float("inf"))
    # Explicit minimum index among maxima matches stable reference tie-breaking.
    maximum = tl.max(logits, 0)
    remote = tl.min(tl.where((blocks < pos // C) & (logits == maximum), blocks, NC), 0)
    tl.store(ROUTES + bh * N + pos, tl.where(pos >= C, remote, -1))
    offsets = tl.arange(0, 2 * C)
    keypos = tl.where(offsets < C, (pos // C) * C + offsets, remote * C + offsets - C)
    valid = (keypos <= pos) & (keypos < N) & ((offsets < C) | (pos >= C))
    kval = tl.load(K + b * KS0 + kh * KS1 + keypos[:, None] * KS2 + d[None, :],
                  valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
    scores = tl.sum(kval * q[None, :], 1) * (D ** -0.5)
    scores = tl.where(valid, scores, -float("inf"))
    weights = tl.exp(scores - tl.max(scores, 0))
    weights /= tl.sum(weights, 0)
    vval = tl.load(V + b * VS0 + kh * VS1 + keypos[:, None] * VS2 + d[None, :],
                  valid[:, None] & (d[None, :] < D), 0).to(tl.float32)
    output = tl.sum(weights[:, None] * vval, 0)
    tl.store(O + (bh * N + pos) * D + d, output, d < D)


def sparse_attention(q, k, v, block_size=32, *, return_routes=False):
    """B,H,T,D tensors, including GQA. FP32 accumulation, inference only."""
    if any(x.requires_grad for x in (q, k, v)) and torch.is_grad_enabled():
        raise RuntimeError("inference kernel has no backward; use the verified training reference")
    if q.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("invalid Q/K/V shapes")
    if not all(x.is_cuda and x.dtype == q.dtype and x.device == q.device and x.stride(-1) == 1 for x in (q, k, v)):
        raise ValueError("requires matching CUDA tensors with contiguous head dimension")
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("unsupported dtype")
    b, h, n, d = q.shape
    kh = k.shape[1]
    if h % kh or block_size not in (16, 32, 64, 128) or not 0 < d <= 128:
        raise ValueError("invalid GQA/head/block dimensions")
    nc = triton.cdiv(n, block_size)
    means = torch.empty((b, kh, nc, d), device=q.device, dtype=torch.float32)
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    routes = torch.empty((b, h, n), device=q.device, dtype=torch.int32)
    _pool[(nc, b * kh)](k, means, n, h, kh, d, *k.stride()[:3], block_size, nc, triton.next_power_of_2(d), num_warps=4)
    _attend[(n, b * h)](q, k, v, means, output, routes, n, h, kh, d,
                        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3],
                        block_size, nc, triton.next_power_of_2(nc), triton.next_power_of_2(d),
                        num_warps=4, enable_fp_fusion=False)
    return (output, routes) if return_routes else output
