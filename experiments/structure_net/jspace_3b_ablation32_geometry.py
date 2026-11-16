"""Learn discarded residual spaces without importing LayerNorm's DC symmetry.

For Qwen's RMSNorm, deleting the mean direction is an additional approximation,
not a normalization invariance.  The default parameterization therefore keeps
the normalized all-ones vector in the retained space.  At width 2048 and retained
width 1024, A has shape (2047, 1024); reduced QR supplies 1024 discarded directions
in the mean-free subspace.  Complete QR is used only for detached CPU export.

This module supplies geometry, not training or task-quality claims.  Projection
changes the native function.  Folding preserves the *projected* function and
its original-width RMS denominator, including a fixed affine anchor if present.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn


def _dimensions(width: int, retained: int, preserve_mean: bool) -> None:
    if type(width) is not int or type(retained) is not int or not 1 <= retained < width:
        raise ValueError("Require integer dimensions 1 <= retained < width")
    if type(preserve_mean) is not bool:
        raise TypeError("preserve_mean must be an explicit bool")


def _signed_columns(value: Tensor) -> Tensor:
    """Fix orientation by the first largest-magnitude coordinate in each column."""
    pivots = value.abs().argmax(dim=0)
    signs = value.gather(0, pivots.unsqueeze(0)).squeeze(0).sign()
    return value * torch.where(signs == 0, torch.ones_like(signs), signs).unsqueeze(0)


def _embed_mean_free(coordinates: Tensor) -> Tensor:
    """Apply H[0; coordinates], where H e_0 = 1/sqrt(width), without storing H."""
    width = coordinates.shape[0] + 1
    c = coordinates.new_full((width,), 1 / math.sqrt(width))
    v = -c
    v[0] += 1
    v = v / torch.linalg.vector_norm(v)
    padded = torch.cat((coordinates.new_zeros((1, coordinates.shape[1])), coordinates), dim=0)
    return padded - 2 * v[:, None] * (v @ padded)[None, :]


class DiscardedSubspace(nn.Module):
    """A differentiable thin-QR discarded-space parameterization.

    ``raw`` is the only learned parameter.  Its default precision is FP64 and
    initialization uses a private CPU generator, so creating an interface does
    not alter global random state.  ``preserve_mean=False`` is an explicitly
    different full-discard search; it is never inferred from the task.

    For gradient accumulation, compute D once per optimizer window, detach a
    copy as the application leaf, accumulate its gradient over microbatches,
    then call D.backward(leaf.grad) once.  Recompute D after every optimizer
    update.  This is the same chain rule as accumulating through the QR graph.
    """

    def __init__(self, width: int = 2048, retained: int = 1024, *, seed: int,
                 preserve_mean: bool = True, dtype: torch.dtype = torch.float64,
                 device: torch.device | str = "cpu") -> None:
        super().__init__()
        _dimensions(width, retained, preserve_mean)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("QR parameters require FP32 or FP64")
        if type(seed) is not int or seed < 0:
            raise ValueError("A nonnegative integer seed is required")
        self.width, self.retained, self.preserve_mean = width, retained, preserve_mean
        self.discarded, self.seed = width - retained, seed
        generator = torch.Generator(device="cpu").manual_seed(seed)
        coordinates = width - int(preserve_mean)
        value = torch.randn((coordinates, self.discarded), generator=generator,
                            dtype=dtype) / math.sqrt(coordinates)
        self.raw = nn.Parameter(value.to(device))

    def _coordinate_basis(self) -> Tensor:
        q, r = torch.linalg.qr(self.raw, mode="reduced")
        diagonal = r.diagonal()
        tolerance = torch.finfo(r.dtype).eps * max(self.raw.shape) * r.detach().abs().amax()
        if not bool(torch.isfinite(r).all()) or bool((diagonal.detach().abs() <= tolerance).any()):
            raise ValueError("Discarded-space QR parameter is nonfinite or numerically rank deficient")
        # Positive R diagonal removes QR's otherwise arbitrary column signs.
        return q * diagonal.sign().unsqueeze(0)

    def discarded_basis(self) -> Tensor:
        """Return differentiable D[width, width-retained] with D.T @ D = I."""
        coordinates = self._coordinate_basis()
        return _embed_mean_free(coordinates) if self.preserve_mean else coordinates

    def forward(self, value: Tensor, *, anchor: Tensor | None = None) -> Tensor:
        return project_with_discarded(value, self.discarded_basis(), anchor=anchor)

    @torch.no_grad()
    def retained_basis(self) -> Tensor:
        """Export deterministic B[width, retained] on CPU in FP64.

        B[:,0] is the positive DC direction when preserve_mean=True.  The other
        columns use complete QR of the detached coordinate basis, then canonical
        pivot signs.  This is repeatable on the same numerical backend; hashes
        should not be assumed invariant across LAPACK versions or hardware.
        """
        coordinates = self._coordinate_basis().detach().to(device="cpu", dtype=torch.float64)
        complete, _ = torch.linalg.qr(coordinates, mode="complete")
        complement = complete[:, self.discarded:]
        if self.preserve_mean:
            c = coordinates.new_full((self.width, 1), 1 / math.sqrt(self.width))
            return torch.cat((c, _signed_columns(_embed_mean_free(complement))), dim=1)
        return _signed_columns(complement)

    @torch.no_grad()
    def export(self, *, anchor: Tensor | None = None) -> dict[str, Any]:
        """Detached basis tensors plus explicit dimensional/normalization policy."""
        if anchor is not None and (anchor.ndim < 1 or anchor.shape[-1] != self.width
                                   or not anchor.is_floating_point()):
            raise ValueError("Export anchor must end in the original residual width")
        discarded = self.discarded_basis().detach().to(device="cpu", dtype=torch.float64)
        retained = self.retained_basis()
        return dict(raw=self.raw.detach().to(device="cpu", dtype=torch.float64).clone(),
            discarded=discarded, retained=retained,
            anchor=None if anchor is None else anchor.detach().to(device="cpu", dtype=torch.float64).clone(),
            metadata=dict(
            width=self.width, retained=self.retained, discarded=self.discarded, seed=self.seed,
            parameter_shape=list(self.raw.shape), parameter_dtype=str(self.raw.dtype),
            preserve_mean=self.preserve_mean, normalization="RMSNorm", rms_denominator_width=self.width,
            retained_dc_column=0 if self.preserve_mean else None,
            export_dtype="torch.float64", qr_search="reduced", qr_export="detached_cpu_complete",
            anchor_shape=None if anchor is None else list(anchor.shape),
            anchor_provenance="caller_bound" if anchor is not None else "zero",
            orthogonality_max_error=float((retained.T @ retained - torch.eye(self.retained,
                dtype=retained.dtype)).abs().amax()),
            cross_max_error=float((retained.T @ discarded).abs().amax()),
            quality_claim=False))


def _compute_dtype(*values: Tensor) -> torch.dtype:
    dtype = torch.float32
    for value in values:
        if not value.is_floating_point():
            raise TypeError("Residual geometry requires floating point tensors")
        dtype = torch.promote_types(dtype, value.dtype)
    return dtype


def project_with_discarded(value: Tensor, discarded: Tensor, *, anchor: Tensor | None = None) -> Tensor:
    """Return anchor + (value-anchor)(I-DD.T), preserving value's output dtype.

    D may be a detached leaf used for window-level gradient accumulation.  The
    caller owns its orthogonality; this function never silently renormalizes it.
    Keep application D in FP32 when native activations are BF16; raw QR can stay
    FP64 on CPU.  Device transfers remain differentiable and are caller-owned.
    """
    if discarded.ndim != 2 or value.shape[-1] != discarded.shape[0]:
        raise ValueError("Discarded basis and residual width differ")
    if value.device != discarded.device:
        raise ValueError("Move the application basis to the residual device explicitly")
    if anchor is not None:
        if anchor.ndim < 1 or anchor.shape[-1] != value.shape[-1] or anchor.device != value.device:
            raise ValueError("Anchor must end in full residual width on the residual device")
        try:
            broadcast_shape = torch.broadcast_shapes(value.shape, anchor.shape)
        except RuntimeError as error:
            raise ValueError("Anchor does not broadcast to the residual shape") from error
        if broadcast_shape != value.shape:
            raise ValueError("Anchor must not expand the residual shape")
    dtype = _compute_dtype(value, discarded, *(() if anchor is None else (anchor,)))
    x, d = value.to(dtype), discarded.to(dtype)
    centered = x if anchor is None else x - anchor.to(dtype)
    return (x - (centered @ d) @ d.T).to(value.dtype)


def rms_denominator(compact: Tensor, *, original_width: int, eps: float,
                    retained_basis: Tensor | None = None, anchor: Tensor | None = None) -> Tensor:
    """RMS denominator of x_hat = anchor + compact @ B.T for orthonormal B.

    Division uses original_width, not compact.shape[-1].  With an anchor, the
    cross term is retained even when B.T @ anchor is not zero.  Accumulation is
    at least FP32.  No assumption about the learned RMSNorm scale is made here.
    """
    if type(original_width) is not int or original_width < compact.shape[-1] or eps <= 0:
        raise ValueError("Invalid original width or RMSNorm epsilon")
    if anchor is None:
        dtype = _compute_dtype(compact)
        energy = compact.to(dtype).square().sum(dim=-1, keepdim=True)
    else:
        if (retained_basis is None or retained_basis.shape != (original_width, compact.shape[-1])
                or anchor.ndim < 1 or anchor.shape[-1] != original_width):
            raise ValueError("Anchored RMSNorm needs the full-width basis and anchor")
        if compact.device != anchor.device or compact.device != retained_basis.device:
            raise ValueError("Compact residual, anchor and retained basis must share a device")
        dtype = _compute_dtype(compact, retained_basis, anchor)
        z, b, mu = compact.to(dtype), retained_basis.to(dtype), anchor.to(dtype)
        try:
            broadcast_shape = torch.broadcast_shapes(z.shape[:-1], mu.shape[:-1])
        except RuntimeError as error:
            raise ValueError("Anchor positions do not broadcast to compact residual positions") from error
        if broadcast_shape != z.shape[:-1]:
            raise ValueError("Anchor must not expand compact residual positions")
        energy = (z.square().sum(dim=-1, keepdim=True)
                  + 2 * (z * (mu @ b)).sum(dim=-1, keepdim=True)
                  + mu.square().sum(dim=-1, keepdim=True))
        # Roundoff alone can make an exactly cancelling affine lift slightly negative.
        energy = energy.clamp_min(0)
    return torch.sqrt(energy / original_width + eps)


def fold_rms_linear_weights(weight: Tensor, scale: Tensor, retained_basis: Tensor, *,
                            anchor: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Return W_gamma_B and W_gamma_mu for Linear(RMSNorm(mu + z B.T)).

    Evaluate (z @ W_gamma_B.T + W_gamma_mu) / rms_denominator(z, ...) + bias.
    Linear's ordinary bias remains outside the denominator.  No diagonal scale
    is commuted across B, and no approximation beyond the chosen subspace is
    introduced by this algebraic folding step.
    """
    if (weight.ndim != 2 or scale.shape != (weight.shape[1],) or retained_basis.ndim != 2
            or retained_basis.shape[0] != weight.shape[1]):
        raise ValueError("RMSNorm, linear and retained basis dimensions differ")
    if anchor is not None and (anchor.ndim < 1 or anchor.shape[-1] != scale.shape[0]):
        raise ValueError("Anchor must end in original residual width")
    tensors = (weight, scale, retained_basis, *(() if anchor is None else (anchor,)))
    if any(value.device != weight.device for value in tensors):
        raise ValueError("Fold inputs must share a device")
    dtype = _compute_dtype(*tensors)
    scaled = weight.to(dtype) * scale.to(dtype).unsqueeze(0)
    folded = scaled @ retained_basis.to(dtype)
    offset = scaled.new_zeros(weight.shape[0]) if anchor is None else anchor.to(dtype) @ scaled.T
    return folded, offset


def memory_inventory(*, width: int = 2048, retained: int = 1024, interfaces: int = 36,
                     preserve_mean: bool = True, parameter_bytes: int = 8,
                     application_bytes: int = 4) -> dict[str, int]:
    """Explicit tensor storage only; QR workspace/saved graphs/activations are extra.

    This is an inventory, not a GPU fit prediction.  CPU parameters, gradients
    and Adam moments need not be resident on the GPU.  Streaming one application
    basis has a different lifetime from retaining all application bases.
    """
    _dimensions(width, retained, preserve_mean)
    if any(type(v) is not int or v < 1 for v in (interfaces, parameter_bytes, application_bytes)):
        raise ValueError("Positive integer interface and dtype sizes required")
    parameter = interfaces * (width - int(preserve_mean)) * (width - retained) * parameter_bytes
    basis = width * (width - retained)
    return dict(raw_parameters=parameter, raw_gradients=parameter, adam_first_and_second_moments=2 * parameter,
                qr_discarded_outputs=interfaces * basis * parameter_bytes,
                application_all_interfaces=interfaces * basis * application_bytes,
                application_one_interface=basis * application_bytes)
