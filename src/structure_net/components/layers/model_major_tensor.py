"""Independent model-major dense, two-factor and tensor-train matrix execution.

Weights have a real leading model dimension: no expand/stride-zero broadcasting,
weight sharing across models, or block-diagonal matrices. Within-model tied
embedding/head weights are an architectural property and are handled separately.
"""
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Tensorization:
    name: str
    kind: str
    rank: int = 0
    rank_fraction: float = 0.0

    def __post_init__(self):
        if self.kind not in ("dense", "svd2", "tt4"):
            raise ValueError("unsupported tensorization")
        if self.kind == "svd2" and not ((self.rank > 0 and self.rank_fraction == 0) or (self.rank == 0 and 0 < self.rank_fraction <= 1)):
            raise ValueError("specify either positive rank or rank fraction in (0,1]")
        if self.kind == "tt4" and self.rank < 1:
            raise ValueError("TT rank must be positive")


CONFIGURATIONS = (Tensorization("svd2_r215", "svd2", rank=215),
                  Tensorization("svd2_r768", "svd2", rank=768),
                  Tensorization("tt4_r64", "tt4", rank=64),
                  Tensorization("tt4_r256", "tt4", rank=256))


def modes(size, order=4):
    """Deterministic balanced prime factor allocation, preserving exact size."""
    factors, divisor = [], 2
    while divisor * divisor <= size:
        while size % divisor == 0:
            factors.append(divisor)
            size //= divisor
        divisor += 1
    if size > 1:
        factors.append(size)
    result = [1] * order
    for factor in reversed(factors):
        index = min(range(order), key=lambda i: (result[i], i))
        result[index] *= factor
    return tuple(sorted(result))


def truncated_svd(matrix, rank):
    """FP32 CUDA decomposition in campaigns; exact small SVD in unit tests.

Randomized range finding is deterministic under the campaign's per-matrix seed.
Its approximation is part of tensorization, never part of packing.
"""
    rank = min(rank, min(matrix.shape))
    if min(matrix.shape) <= 256 or rank >= min(matrix.shape):
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        return u[:, :rank], s[:rank], vh[:rank]
    u, s, v = torch.svd_lowrank(matrix, q=min(rank + 16, min(matrix.shape)), niter=2)
    return u[:, :rank], s[:rank], v[:, :rank].T


def decompose(weight, setting):
    """Return unbatched cores and metadata; keep dense if factorization expands."""
    out_size, in_size = weight.shape
    metadata = {"kind": setting.kind, "shape": [out_size, in_size]}
    if setting.kind == "dense":
        return {**metadata, "cores": [weight.detach().clone()]}
    value = weight.float()
    if setting.kind == "svd2":
        rank = min(min(weight.shape), setting.rank or max(1, int(min(weight.shape) * setting.rank_fraction)))
        if rank * (out_size + in_size) >= weight.numel():
            return {"kind": "dense", "shape": list(weight.shape), "cores": [weight.detach().clone()], "reason": "factor_storage_not_smaller"}
        u, s, vh = truncated_svd(value, rank)
        cores = [(u * s).to(weight.dtype), vh.to(weight.dtype)]
        return {**metadata, "rank": rank, "cores": cores}
    output_modes, input_modes = modes(out_size), modes(in_size)
    order = len(output_modes)
    permutation = [j for i in range(order) for j in (i, i + order)]
    tensor = value.reshape(*output_modes, *input_modes).permute(permutation).contiguous()
    cores, left = [], 1
    for index in range(order - 1):
        matrix = tensor.reshape(left * output_modes[index] * input_modes[index], -1)
        rank = min(setting.rank, *matrix.shape)
        u, s, vh = truncated_svd(matrix, rank)
        cores.append(u.reshape(left, output_modes[index], input_modes[index], rank).to(weight.dtype))
        tensor = s[:, None] * vh
        left = rank
    cores.append(tensor.reshape(left, output_modes[-1], input_modes[-1], 1).to(weight.dtype))
    if sum(c.numel() for c in cores) >= weight.numel():
        return {"kind": "dense", "shape": list(weight.shape), "cores": [weight.detach().clone()], "reason": "factor_storage_not_smaller"}
    return {**metadata, "output_modes": output_modes, "input_modes": input_modes, "cores": cores}


def reconstruct(spec):
    if spec["kind"] == "dense":
        return spec["cores"][0]
    if spec["kind"] == "svd2":
        return spec["cores"][0] @ spec["cores"][1]
    tensor = spec["cores"][0].squeeze(0)
    for core in spec["cores"][1:]:
        tensor = torch.tensordot(tensor, core, dims=([-1], [0]))
    tensor = tensor.squeeze(-1)
    order = len(spec["output_modes"])
    return tensor.permute(*range(0, 2 * order, 2), *range(1, 2 * order, 2)).reshape(spec["shape"])


def pack(specs, device=None):
    """Materialize independent slices, even when all replicas start identical."""
    reference = {k: v for k, v in specs[0].items() if k != "cores"}
    if any({k: v for k, v in s.items() if k != "cores"} != reference for s in specs):
        raise ValueError("packed replicas must use corresponding compatible cores")
    result = {**reference, "cores": [torch.stack([s["cores"][i] for s in specs], 0).to(device=device)
                                     for i in range(len(specs[0]["cores"]))]}
    if any(c.stride(0) == 0 or not c.is_contiguous() for c in result["cores"]):
        raise RuntimeError("packing must materialize physical replica storage")
    return result


def apply(x, packed):
    """x: [model, token, input]; every projection contracts through dense bmm."""
    if x.ndim != 3 or x.shape[0] != packed["cores"][0].shape[0]:
        raise ValueError("require model-major inputs matching the number of replicas")
    cores = packed["cores"]
    if packed["kind"] == "dense":
        return torch.bmm(x, cores[0].transpose(1, 2))
    if packed["kind"] == "svd2":
        return torch.bmm(torch.bmm(x, cores[1].transpose(1, 2)), cores[0].transpose(1, 2))
    count, batch, _ = x.shape
    ins, outs = packed["input_modes"], packed["output_modes"]
    # Invariant: [M, B, output-prefix, rank-left, remaining-input-modes].
    state = x.reshape(count, batch, 1, 1, *ins)
    prefix = 1
    for index, core in enumerate(cores):
        _, left, out_mode, in_mode, right = core.shape
        remaining = math.prod(ins[index + 1:])
        state = state.reshape(count, batch, prefix, left, in_mode, remaining)
        state = state.permute(0, 1, 2, 5, 3, 4).reshape(count, batch * prefix * remaining, left * in_mode)
        kernel = core.permute(0, 1, 3, 2, 4).reshape(count, left * in_mode, out_mode * right)
        state = torch.bmm(state, kernel).reshape(count, batch, prefix, remaining, out_mode, right)
        state = state.permute(0, 1, 2, 4, 5, 3).reshape(count, batch, prefix * out_mode, right, remaining)
        prefix *= out_mode
    return state.reshape(count, batch, packed["shape"][0])


def embedding(ids, packed, chunk_tokens=32):
    """Select rows of a compressed vocabulary matrix without reconstructing it."""
    count, tokens = ids.shape
    cores = packed["cores"]
    model_indices = torch.arange(count, device=ids.device)[:, None]
    if packed["kind"] == "dense":
        return cores[0][model_indices, ids]
    if packed["kind"] == "svd2":
        return torch.bmm(cores[0][model_indices, ids], cores[1])
    chunks = []
    for part in ids.split(chunk_tokens, dim=1):
        length = part.shape[1]
        digits, quotient = [], part
        for size in reversed(packed["output_modes"]):
            digits.append(quotient % size)
            quotient = quotient // size
        digits.reverse()
        state = torch.ones(count * length, 1, 1, device=cores[0].device, dtype=cores[0].dtype)
        prefix = 1
        for index, core in enumerate(cores):
            _, left, out_mode, in_mode, right = core.shape
            rows = core.permute(0, 2, 1, 3, 4)[model_indices, digits[index]]
            rows = rows.reshape(count * length, left, in_mode * right)
            state = torch.bmm(state, rows).reshape(count * length, prefix * in_mode, right)
            prefix *= in_mode
        chunks.append(state.reshape(count, length, packed["shape"][1]))
    return torch.cat(chunks, dim=1)
