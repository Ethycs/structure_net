#!/usr/bin/env python3
"""Interval-certify the retained floating Chebyshev defect surrogate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


SCHEMA_VERSION = "nal.tinyllm-defect-surrogate-interval-certificate.v1"
HYPOTHESIS_ID = "tinyllm-d6-step15-defect-certificate-v1"


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("invalid interval")


def _down(value: float) -> float:
    return float(np.nextafter(value, -np.inf))


def _up(value: float) -> float:
    return float(np.nextafter(value, np.inf))


def add(left: Interval, right: Interval) -> Interval:
    return Interval(_down(left.lo + right.lo), _up(left.hi + right.hi))


def neg(value: Interval) -> Interval:
    return Interval(_down(-value.hi), _up(-value.lo))


def sub(left: Interval, right: Interval) -> Interval:
    return add(left, neg(right))


def mul(left: Interval, right: Interval) -> Interval:
    values = (
        left.lo * right.lo,
        left.lo * right.hi,
        left.hi * right.lo,
        left.hi * right.hi,
    )
    return Interval(_down(min(values)), _up(max(values)))


def point(value: float) -> Interval:
    return Interval(float(value), float(value))


def scale(value: Interval, coefficient: float) -> Interval:
    return mul(value, point(coefficient))


def contains_zero(value: Interval) -> bool:
    return value.lo <= 0.0 <= value.hi


def chebyshev_values(value: Interval, degree: int) -> list[Interval]:
    result = [point(1.0)]
    if degree == 0:
        return result
    result.append(value)
    for _ in range(2, degree + 1):
        result.append(sub(scale(mul(value, result[-1]), 2.0), result[-2]))
    return result


def chebyshev_interval(coefficients: np.ndarray, x: Interval, y: Interval) -> Interval:
    tx = chebyshev_values(x, coefficients.shape[0] - 1)
    ty = chebyshev_values(y, coefficients.shape[1] - 1)
    total = point(0.0)
    for i in range(coefficients.shape[0]):
        for j in range(coefficients.shape[1]):
            total = add(total, scale(mul(tx[i], ty[j]), float(coefficients[i, j])))
    return total


def derivative_coefficients(coefficients: np.ndarray, axis: int) -> np.ndarray:
    return np.polynomial.chebyshev.chebder(coefficients, axis=axis)


def field_interval(
    coefficients: Sequence[np.ndarray], x: Interval, y: Interval
) -> tuple[Interval, Interval]:
    return tuple(chebyshev_interval(value, x, y) for value in coefficients)  # type: ignore[return-value]


def jacobian_interval(
    derivatives: Sequence[Sequence[np.ndarray]], x: Interval, y: Interval
) -> list[list[Interval]]:
    return [
        [chebyshev_interval(derivatives[row][column], x, y) for column in range(2)]
        for row in range(2)
    ]


def determinant_path_phase(jacobian: Sequence[Sequence[Interval]]) -> Interval:
    # Swapping the usual (phase, path) columns matches the charged-cell orientation.
    ordinary = sub(mul(jacobian[0][0], jacobian[1][1]), mul(jacobian[0][1], jacobian[1][0]))
    return neg(ordinary)


def _linear_combination(coefficients: np.ndarray, values: Sequence[Interval]) -> Interval:
    total = point(0.0)
    for coefficient, value in zip(coefficients, values):
        total = add(total, scale(value, float(coefficient)))
    return total


def krawczyk_box(
    coefficients: Sequence[np.ndarray],
    derivatives: Sequence[Sequence[np.ndarray]],
    x: Interval,
    y: Interval,
) -> dict[str, Any]:
    center = np.asarray(((x.lo + x.hi) / 2.0, (y.lo + y.hi) / 2.0))
    center_field = field_interval(coefficients, point(center[0]), point(center[1]))
    center_jacobian = np.asarray(
        [
            [np.polynomial.chebyshev.chebval2d(center[0], center[1], derivatives[row][column]) for column in range(2)]
            for row in range(2)
        ]
    )
    inverse = np.linalg.inv(center_jacobian)
    jacobian = jacobian_interval(derivatives, x, y)
    inverse_field = [_linear_combination(inverse[row], center_field) for row in range(2)]
    center_term = [sub(point(float(center[row])), inverse_field[row]) for row in range(2)]
    matrix: list[list[Interval]] = []
    for row in range(2):
        entries = []
        for column in range(2):
            product = _linear_combination(inverse[row], [jacobian[item][column] for item in range(2)])
            entries.append(sub(point(1.0 if row == column else 0.0), product))
        matrix.append(entries)
    displacement = [Interval(x.lo - center[0], x.hi - center[0]), Interval(y.lo - center[1], y.hi - center[1])]
    image = []
    for row in range(2):
        correction = add(mul(matrix[row][0], displacement[0]), mul(matrix[row][1], displacement[1]))
        image.append(add(center_term[row], correction))
    strictly_inside = (
        x.lo < image[0].lo < image[0].hi < x.hi
        and y.lo < image[1].lo < image[1].hi < y.hi
    )
    determinant = determinant_path_phase(jacobian)
    return {
        "box": {"phase_normalized": [x.lo, x.hi], "path_normalized": [y.lo, y.hi]},
        "image": {
            "phase_normalized": [image[0].lo, image[0].hi],
            "path_normalized": [image[1].lo, image[1].hi],
        },
        "strictly_inside": strictly_inside,
        "oriented_determinant_interval": [determinant.lo, determinant.hi],
        "positive_oriented_determinant": determinant.lo > 0.0,
    }


def isolate_candidates(
    coefficients: Sequence[np.ndarray],
    *,
    initial_grid: int,
    maximum_depth: int,
) -> dict[str, Any]:
    pending = []
    edges = np.linspace(-1.0, 1.0, initial_grid + 1)
    for i in range(initial_grid):
        for j in range(initial_grid):
            pending.append((Interval(float(edges[i]), float(edges[i + 1])), Interval(float(edges[j]), float(edges[j + 1])), 0))
    excluded = 0
    unresolved: list[tuple[Interval, Interval]] = []
    while pending:
        x, y, depth = pending.pop()
        values = field_interval(coefficients, x, y)
        if not contains_zero(values[0]) or not contains_zero(values[1]):
            excluded += 1
            continue
        if depth >= maximum_depth:
            unresolved.append((x, y))
            continue
        if (x.hi - x.lo) >= (y.hi - y.lo):
            middle = (x.lo + x.hi) / 2.0
            pending.extend(((Interval(x.lo, middle), y, depth + 1), (Interval(middle, x.hi), y, depth + 1)))
        else:
            middle = (y.lo + y.hi) / 2.0
            pending.extend(((x, Interval(y.lo, middle), depth + 1), (x, Interval(middle, y.hi), depth + 1)))
    if not unresolved:
        raise RuntimeError("interval search found no surrogate root candidate")
    hull_x = Interval(min(item[0].lo for item in unresolved), max(item[0].hi for item in unresolved))
    hull_y = Interval(min(item[1].lo for item in unresolved), max(item[1].hi for item in unresolved))
    return {
        "excluded_leaf_count": excluded,
        "unresolved_leaf_count": len(unresolved),
        "candidate_hull": (hull_x, hull_y),
    }


def boundary_audit(
    coefficients: Sequence[np.ndarray], segments: int
) -> dict[str, Any]:
    edges = np.linspace(-1.0, 1.0, segments + 1)
    minimum_lower_norm = float("inf")
    all_excluded = True
    for index in range(segments):
        interval = Interval(float(edges[index]), float(edges[index + 1]))
        boxes = (
            (interval, point(-1.0)),
            (interval, point(1.0)),
            (point(-1.0), interval),
            (point(1.0), interval),
        )
        for x, y in boxes:
            values = field_interval(coefficients, x, y)
            all_excluded &= not (contains_zero(values[0]) and contains_zero(values[1]))
            component_lower = []
            for value in values:
                component_lower.append(0.0 if contains_zero(value) else min(abs(value.lo), abs(value.hi)))
            minimum_lower_norm = min(minimum_lower_norm, math.hypot(*component_lower))
    return {
        "segments_per_edge": segments,
        "all_boundary_boxes_exclude_zero": all_excluded,
        "minimum_interval_lower_norm": minimum_lower_norm,
    }


def run_certificate(source: Path, output: Path, initial_grid: int, maximum_depth: int, boundary_segments: int) -> dict[str, Any]:
    original = json.loads(source.read_text())
    surrogate = original["chebyshev_surrogate"]
    coefficients = [np.asarray(value, dtype=np.float64) for value in surrogate["coefficients"]]
    derivatives = [
        [derivative_coefficients(value, 0), derivative_coefficients(value, 1)]
        for value in coefficients
    ]
    isolation = isolate_candidates(coefficients, initial_grid=initial_grid, maximum_depth=maximum_depth)
    hull_x, hull_y = isolation.pop("candidate_hull")
    krawczyk = krawczyk_box(coefficients, derivatives, hull_x, hull_y)
    boundary = boundary_audit(coefficients, boundary_segments)
    certified = bool(
        krawczyk["strictly_inside"]
        and krawczyk["positive_oriented_determinant"]
        and boundary["all_boundary_boxes_exclude_zero"]
    )
    phase_bounds = original["declared_rectangle"]["phase"]
    path_bounds = original["declared_rectangle"]["path"]
    def physical(interval: Interval, bounds: Sequence[float]) -> list[float]:
        midpoint, radius = np.mean(bounds), np.ptp(bounds) / 2.0
        return [float(midpoint + radius * interval.lo), float(midpoint + radius * interval.hi)]
    result = {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": HYPOTHESIS_ID,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_result": str(source),
        "source_surrogate_degree": surrogate["degree"],
        "coefficient_semantics": "stored binary64 coefficients treated as exact real constants",
        "arithmetic": "binary64 directed outward rounding with nextafter after every primitive operation",
        "isolation": {
            **isolation,
            "candidate_hull_normalized": krawczyk["box"],
            "candidate_hull_physical": {
                "phase": physical(hull_x, phase_bounds),
                "path": physical(hull_y, path_bounds),
            },
        },
        "krawczyk": krawczyk,
        "boundary": boundary,
        "surrogate_unique_positive_root_certified": certified,
        "actual_network_transfer_certified": False,
        "verdict": (
            "floating_chebyshev_surrogate_unique_positive_root_certified_actual_network_transfer_open"
            if certified
            else "surrogate_interval_certificate_not_obtained"
        ),
        "method_boundary": (
            "This is a formal certificate for the stored floating Chebyshev polynomial only. "
            "The sampled network-to-surrogate error is not a rigorous remainder enclosure, "
            "so the actual transformer root remains numerically localized but uncertified."
        ),
    }
    _write_json(output, result)
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/experiments/tinyllm_defect_certification/20260806_d6_step15/results.json"))
    parser.add_argument("--output", type=Path, default=Path("data/experiments/tinyllm_defect_certification/20260806_d6_step15_surrogate_interval/results.json"))
    parser.add_argument("--initial-grid", type=int, default=16)
    parser.add_argument("--maximum-depth", type=int, default=18)
    parser.add_argument("--boundary-segments", type=int, default=1024)
    args = parser.parse_args(argv)
    result = run_certificate(args.source, args.output, args.initial_grid, args.maximum_depth, args.boundary_segments)
    print(json.dumps({"verdict": result["verdict"], "isolation": result["isolation"], "krawczyk": result["krawczyk"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
