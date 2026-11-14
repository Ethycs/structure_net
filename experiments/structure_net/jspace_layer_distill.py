"""Training-only native-boundary losses for already compressed Qwen blocks.

Fixed teachers, coordinate maps, anchors and VJP probes never receive gradients.
These helpers validate finite inputs and can synchronize the device; they are
not inference/CUDA-graph kernels. No module or teacher is added to deployment.
"""
from __future__ import annotations

import math

import torch


def _finite(name, value):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _dtype(*values):
    return torch.float64 if any(v.dtype == torch.float64 for v in values) else torch.float32


def _boundary_pair(student, teacher):
    _finite("student", student)
    _finite("teacher", teacher)
    if student.shape != teacher.shape or student.ndim < 2 or not student.numel():
        raise ValueError("Student and teacher must have equal nonempty [..., sequence, width] shapes")
    if student.device != teacher.device:
        raise ValueError("Student and teacher must be on the same device")


def affine_project_native(y, basis, anchor, positions=None):
    """Return a + U U^T(y-a), preserving y's gradient in native coordinates.

    U has prevalidated orthonormal columns [width, rank]. The compiler checks
    this invariant once; recomputing U^T U here would dominate a small batch.
    Anchor is broadcastable to y,
    or is [positions, width] when integer ``positions`` indexes its rows.
    Position indices must broadcast to y.shape[:-1]. Arithmetic uses FP32,
    or FP64 when any floating input is FP64, independently of autocast.
    """
    for name, value in (("y", y), ("basis", basis), ("anchor", anchor)):
        _finite(name, value)
    if y.ndim < 2 or not y.numel() or basis.ndim != 2 or basis.shape[0] != y.shape[-1]:
        raise ValueError("Basis must have shape [native width, rank] matching a nonempty boundary")
    if not 0 < basis.shape[1] <= basis.shape[0] or anchor.ndim < 1 or anchor.shape[-1] != y.shape[-1]:
        raise ValueError("Invalid basis rank or native anchor width")
    if basis.device != y.device or anchor.device != y.device:
        raise ValueError("Boundary, basis and anchor must be on the same device")
    dtype = _dtype(y, basis, anchor)
    with torch.autocast(device_type=y.device.type, enabled=False):
        u = basis.detach().to(dtype)
        a = anchor.detach().to(dtype)
        if positions is not None:
            if anchor.ndim != 2 or not isinstance(positions, torch.Tensor) or positions.dtype not in (torch.int32, torch.int64):
                raise ValueError("Indexed anchors require a table and integer position tensor")
            positions = positions.to(device=y.device, dtype=torch.long)
            if ((positions < 0) | (positions >= len(anchor))).any():
                raise ValueError("Anchor positions lie outside the supplied table")
            a = a.index_select(0, positions.reshape(-1)).reshape(*positions.shape, y.shape[-1])
        try:
            a = torch.broadcast_to(a, y.shape)
        except RuntimeError as error:
            raise ValueError("Anchor rows must broadcast to the native boundary") from error
        result = a + ((y.to(dtype) - a) @ u) @ u.T
        _finite("projected boundary", result)
        return result


def projected_boundary_mse(student, teacher, basis, anchor, positions=None):
    """Return (trainable retained error, detached discarded teacher floor).

    Both terms are mean squared errors per native coordinate. The floor is
    orthogonal projection error, not an extra optimization target. Their sum
    equals full teacher MSE when the student lies in the specified affine set.
    """
    _boundary_pair(student, teacher)
    target = affine_project_native(teacher.detach().to(_dtype(student, teacher, basis, anchor)),
                                   basis, anchor, positions)
    dtype = _dtype(student, target)
    with torch.autocast(device_type=student.device.type, enabled=False):
        loss = (student.to(dtype) - target).square().mean()
        floor = (teacher.detach().to(dtype) - target).square().mean().detach()
        _finite("retained boundary loss", loss)
        _finite("discarded boundary floor", floor)
        return loss, floor


def full_sequence_vjp_loss(student, teacher, probes):
    """Mean squared whole-boundary VJP inner product, including token cross terms.

    Probes have shape [K, *student.shape]; every dimension after K is reduced
    in one inner product BEFORE squaring. Probe normalization is supplied by
    the caller's fixed Jacobian estimator, not changed by this helper.
    """
    _boundary_pair(student, teacher)
    _finite("probes", probes)
    if probes.ndim != student.ndim + 1 or probes.shape[1:] != student.shape or not probes.shape[0]:
        raise ValueError("Probes must have shape [positive K, *student.shape]")
    if probes.device != student.device:
        raise ValueError("Probes and boundaries must be on the same device")
    dtype = _dtype(student, teacher, probes)
    with torch.autocast(device_type=student.device.type, enabled=False):
        error = (student.to(dtype) - teacher.detach().to(dtype)).reshape(-1)
        products = probes.detach().to(dtype).reshape(probes.shape[0], -1) @ error
        loss = products.square().mean()
        _finite("VJP loss", loss)
        return loss


def rms_readout_jacobian_loss(student, teacher, gain, epsilon, readout_gram, vocabulary_size):
    """Exact first-order full-logit metric for a fixed RMSNorm + linear suffix.

    ``readout_gram`` is the precomputed E^T E of the original vocabulary
    readout E. The result is ||J(teacher)(student-teacher)||^2 / (P * V),
    where P counts every position in the supplied boundary and V is the full
    vocabulary size. No softmax weighting or finite-perturbation claim is made.
    The derivative is evaluated analytically in FP32 (FP64 for double inputs),
    so it does not differentiate the original model's BF16 rounding operations.

    Teacher, RMS gain and Gram are fixed, detached training resources. E^T E
    must be constructed and validated by the caller; no eigendecomposition or
    vocabulary-sized Jacobian is recomputed in a training step.
    """
    _boundary_pair(student, teacher)
    _finite("gain", gain)
    _finite("readout_gram", readout_gram)
    width = student.shape[-1]
    if gain.shape != (width,) or readout_gram.shape != (width, width):
        raise ValueError("Gain and readout Gram must match the native boundary width")
    if gain.device != student.device or readout_gram.device != student.device:
        raise ValueError("Gain, readout Gram and boundaries must be on the same device")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("RMS epsilon must be finite and nonnegative")
    if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int) or vocabulary_size <= 0:
        raise ValueError("Vocabulary size must be a positive integer")
    dtype = _dtype(student, teacher, gain, readout_gram)
    with torch.autocast(device_type=student.device.type, enabled=False):
        t = teacher.detach().to(dtype)
        error = student.to(dtype) - t
        radius = (t.square().mean(-1, keepdim=True) + epsilon).sqrt()
        tangent = (error / radius
                   - t * (t * error).sum(-1, keepdim=True) / (width * radius.pow(3)))
        tangent = (tangent * gain.detach().to(dtype)).reshape(-1, width)
        transformed = tangent @ readout_gram.detach().to(dtype)
        loss = (transformed * tangent).sum() / (tangent.shape[0] * vocabulary_size)
        _finite("RMS readout Jacobian loss", loss)
        return loss


def gradient_statistics(task_loss, local_loss, parameters):
    """Unscaled parameter-gradient norms/dot/cosine without modifying .grad.

    Unused/frozen parameters and constant losses contribute zero. Graphs are
    retained so ordinary training backward remains available after diagnostics.
    Returned floats are suitable for a serialized training-calibration record.
    """
    for name, loss in (("task_loss", task_loss), ("local_loss", local_loss)):
        _finite(name, loss)
        if loss.numel() != 1:
            raise ValueError("Diagnostic losses must be scalar")
    parameters = list(parameters)
    if any(not isinstance(p, torch.Tensor) for p in parameters):
        raise ValueError("Parameters must be tensors")
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Parameters must not contain duplicates")
    parameters = [p for p in parameters if p.requires_grad]

    def gradients(loss):
        if not parameters or not loss.requires_grad:
            return (None,) * len(parameters)
        return torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)

    task_gradients, local_gradients = gradients(task_loss), gradients(local_loss)
    task_squared, local_squared, dot = 0., 0., 0.
    for task, local in zip(task_gradients, local_gradients):
        if task is not None:
            _finite("task gradient", task)
            task = task.detach().to(_dtype(task))
            task_squared += float(task.square().sum())
        if local is not None:
            _finite("local gradient", local)
            local = local.detach().to(_dtype(local))
            local_squared += float(local.square().sum())
        if task is not None and local is not None:
            dot += float((task.to(_dtype(task, local)) * local.to(_dtype(task, local))).sum())
    task_norm, local_norm = math.sqrt(task_squared), math.sqrt(local_squared)
    cosine = max(-1., min(1., dot / (task_norm * local_norm))) if task_norm and local_norm else 0.
    result = {"task_grad_norm": task_norm, "local_grad_norm": local_norm,
              "grad_dot": dot, "grad_cosine": cosine}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Gradient statistics must be finite")
    return result


def coefficient_from_stats(statistics, target_ratio=.1):
    """Freeze lambda = ratio * RMS(task norms) / RMS(local norms).

    Supply training-calibration records from the shared initializer only.
    A zero task gradient yields zero weight; a positive target with no local
    gradient is undefined and raises rather than silently picking a weight.
    """
    if not math.isfinite(target_ratio) or target_ratio < 0:
        raise ValueError("Target ratio must be finite and nonnegative")
    rows = list(statistics)
    if not rows:
        raise ValueError("At least one training-calibration statistic is required")
    task, local = [], []
    for row in rows:
        if not all(math.isfinite(float(value)) for value in row.values()):
            raise ValueError("Calibration statistics must be finite")
        t, l = float(row["task_grad_norm"]), float(row["local_grad_norm"])
        if t < 0 or l < 0:
            raise ValueError("Gradient norms must be nonnegative")
        task.append(t)
        local.append(l)
    # hypot avoids avoidable overflow when summing squared finite Python floats.
    task_norm, local_norm = math.hypot(*task), math.hypot(*local)
    if target_ratio == 0 or task_norm == 0:
        return 0.
    if local_norm == 0:
        raise ValueError("Cannot calibrate a positive weight from zero local gradients")
    coefficient = target_ratio * (task_norm / local_norm)
    if not math.isfinite(coefficient):
        raise ValueError("Calibrated coefficient must be finite")
    return coefficient
