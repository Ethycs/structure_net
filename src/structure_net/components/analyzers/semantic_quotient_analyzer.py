"""Task-relative persistent-homology utilities for semantic quotient studies.

These helpers deliberately accept representations, posteriors, or precomputed
distances.  They do not infer a semantic metric from arbitrary activation
clouds.  Experiment code remains responsible for defining the task quotient
and for selecting a semantically designated readout position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from ripser import ripser
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import lsqr
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances


@dataclass(frozen=True)
class PersistenceSummary:
    """Compact, JSON-safe summary of a Vietoris--Rips calculation."""

    point_count: int
    finite_h0_count: int
    h1_count: int
    max_h0_lifetime: float
    total_h0_lifetime: float
    max_h1_lifetime: float
    total_h1_lifetime: float
    normalized_h1_lifetime: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "point_count": self.point_count,
            "finite_h0_count": self.finite_h0_count,
            "h1_count": self.h1_count,
            "max_h0_lifetime": self.max_h0_lifetime,
            "total_h0_lifetime": self.total_h0_lifetime,
            "max_h1_lifetime": self.max_h1_lifetime,
            "total_h1_lifetime": self.total_h1_lifetime,
            "normalized_h1_lifetime": self.normalized_h1_lifetime,
        }


def _square_distance_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance matrix must be square")
    if not np.isfinite(matrix).all():
        raise ValueError("distance matrix must contain only finite values")
    if not np.allclose(matrix, matrix.T, atol=1e-7, rtol=1e-7):
        raise ValueError("distance matrix must be symmetric")
    if np.any(matrix < -1e-9):
        raise ValueError("distance matrix cannot contain negative values")
    matrix = np.maximum(matrix, 0.0)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def fisher_rao_distance_matrix(
    posteriors: np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return categorical Fisher--Rao distances with the factor-two convention."""
    probabilities = np.asarray(posteriors, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("posteriors must have shape (samples, classes>=2)")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("posteriors must be finite and non-negative")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("each posterior must have positive mass")
    probabilities = np.maximum(probabilities / row_sums, epsilon)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    affinities = np.sqrt(probabilities) @ np.sqrt(probabilities).T
    distances = 2.0 * np.arccos(np.clip(affinities, -1.0, 1.0))
    np.fill_diagonal(distances, 0.0)
    return distances


def representation_distance_matrix(
    representations: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    """Compute a Euclidean or cosine activation-space baseline."""
    values = np.asarray(representations, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("representations must have shape (samples>=2, features)")
    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")
    return _square_distance_matrix(pairwise_distances(values, metric=metric))


def persistence_diagrams(distance_matrix: np.ndarray) -> list[np.ndarray]:
    """Compute H0 and H1 Vietoris--Rips diagrams from a task-defined metric."""
    distances = _square_distance_matrix(distance_matrix)
    diagrams = ripser(distances, distance_matrix=True, maxdim=1)["dgms"]
    return [np.asarray(diagram, dtype=np.float64) for diagram in diagrams]


def persistent_cohomology_circle_coordinate(
    distance_matrix: np.ndarray,
    *,
    coefficient: int = 47,
    cocycle_scale_fraction: float = 0.9,
) -> Dict[str, Any]:
    """Derive a circular coordinate from the longest persistent H1 cocycle.

    This implements the least-squares smoothing construction for a lifted
    finite-field cocycle. It discovers a coordinate from representation
    distances and does not use semantic labels or a supervised decoder.
    """
    distances = _square_distance_matrix(distance_matrix)
    if coefficient < 3 or coefficient % 2 == 0:
        raise ValueError("coefficient must be an odd integer of at least 3")
    if not 0.0 < cocycle_scale_fraction < 1.0:
        raise ValueError("cocycle_scale_fraction must be in (0, 1)")
    calculation = ripser(
        distances,
        distance_matrix=True,
        maxdim=1,
        do_cocycles=True,
        coeff=coefficient,
    )
    diagram = np.asarray(calculation["dgms"][1], dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(diagram[:, 1])) if diagram.size else np.array([])
    if not len(finite):
        return {
            "found": False,
            "angles": np.zeros(len(distances), dtype=np.float64),
            "birth": 0.0,
            "death": 0.0,
            "lifetime": 0.0,
            "scale": 0.0,
            "coefficient": coefficient,
        }
    lifetimes = diagram[finite, 1] - diagram[finite, 0]
    selected = int(finite[int(np.argmax(lifetimes))])
    birth, death = map(float, diagram[selected])
    scale = birth + cocycle_scale_fraction * (death - birth)
    cocycle = np.asarray(calculation["cocycles"][1][selected], dtype=np.int64)

    directed_values: Dict[tuple[int, int], int] = {}
    for first, second, raw_value in cocycle:
        value = int(raw_value) % coefficient
        directed_values[(int(first), int(second))] = value
        directed_values[(int(second), int(first))] = (-value) % coefficient
    edges = np.argwhere(np.triu(distances <= scale, k=1))
    if not len(edges):
        raise RuntimeError("persistent cocycle scale produced no graph edges")
    lifted = np.zeros(len(edges), dtype=np.float64)
    midpoint = coefficient // 2
    for index, (first, second) in enumerate(edges):
        value = directed_values.get((int(first), int(second)), 0)
        lifted[index] = value - coefficient if value > midpoint else value

    rows = np.repeat(np.arange(len(edges)), 2)
    columns = edges.reshape(-1)
    values = np.tile(np.array([-1.0, 1.0]), len(edges))
    coboundary = coo_matrix(
        (values, (rows, columns)), shape=(len(edges), len(distances))
    ).tocsr()
    potential = lsqr(
        coboundary,
        -lifted,
        atol=1e-12,
        btol=1e-12,
    )[0]
    angles = 2.0 * np.pi * np.mod(potential, 1.0)
    return {
        "found": True,
        "angles": angles,
        "birth": birth,
        "death": death,
        "lifetime": death - birth,
        "scale": scale,
        "coefficient": coefficient,
    }


def summarize_persistence(distance_matrix: np.ndarray) -> PersistenceSummary:
    """Summarize finite H0 and H1 lifetimes without choosing a display scale."""
    distances = _square_distance_matrix(distance_matrix)
    h0, h1 = persistence_diagrams(distances)
    finite_h0 = h0[np.isfinite(h0[:, 1])] if h0.size else np.empty((0, 2))
    finite_h1 = h1[np.isfinite(h1[:, 1])] if h1.size else np.empty((0, 2))
    h0_lifetimes = finite_h0[:, 1] - finite_h0[:, 0] if finite_h0.size else np.array([])
    h1_lifetimes = finite_h1[:, 1] - finite_h1[:, 0] if finite_h1.size else np.array([])
    positive = distances[distances > 0.0]
    scale = float(np.median(positive)) if positive.size else 1.0
    return PersistenceSummary(
        point_count=distances.shape[0],
        finite_h0_count=len(finite_h0),
        h1_count=len(finite_h1),
        max_h0_lifetime=float(h0_lifetimes.max()) if h0_lifetimes.size else 0.0,
        total_h0_lifetime=float(h0_lifetimes.sum()) if h0_lifetimes.size else 0.0,
        max_h1_lifetime=float(h1_lifetimes.max()) if h1_lifetimes.size else 0.0,
        total_h1_lifetime=float(h1_lifetimes.sum()) if h1_lifetimes.size else 0.0,
        normalized_h1_lifetime=(
            float(h1_lifetimes.max() / scale) if h1_lifetimes.size and scale > 0.0 else 0.0
        ),
    )


def bootstrap_max_h1(
    distance_matrix: np.ndarray,
    *,
    repeats: int,
    sample_fraction: float,
    seed: int,
) -> Dict[str, Any]:
    """Bootstrap the longest H1 interval using point subsamples without replacement."""
    distances = _square_distance_matrix(distance_matrix)
    if repeats <= 0 or not 0.0 < sample_fraction <= 1.0:
        raise ValueError("bootstrap repeats/fraction are invalid")
    sample_count = max(4, int(round(len(distances) * sample_fraction)))
    generator = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = np.sort(generator.choice(len(distances), sample_count, replace=False))
        sample = distances[np.ix_(indices, indices)]
        values.append(summarize_persistence(sample).max_h1_lifetime)
    return {
        "repeats": repeats,
        "sample_fraction": sample_fraction,
        "values": [float(value) for value in values],
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "positive_fraction": float(np.mean(np.asarray(values) > 0.0)),
    }


def knn_geodesic_distance_matrix(
    neighborhood_distances: np.ndarray,
    edge_lengths: np.ndarray,
    *,
    neighbors: int,
) -> np.ndarray:
    """Build a symmetric kNN graph and return task-metric shortest paths.

    ``neighborhood_distances`` determines which edges exist. ``edge_lengths``
    supplies their task-relative lengths, such as pullback-Fisher lengths.
    The neighborhood count is increased only when required for connectivity.
    """
    neighborhood = _square_distance_matrix(neighborhood_distances)
    lengths = _square_distance_matrix(edge_lengths)
    if neighborhood.shape != lengths.shape:
        raise ValueError("neighborhood distances and edge lengths must have equal shape")
    point_count = len(neighborhood)
    if not 1 <= neighbors < point_count:
        raise ValueError("neighbors must be between 1 and point_count - 1")

    order = np.argsort(neighborhood, axis=1)
    for current_neighbors in range(neighbors, point_count):
        graph = np.full((point_count, point_count), np.inf, dtype=np.float64)
        np.fill_diagonal(graph, 0.0)
        for row in range(point_count):
            columns = order[row, 1 : current_neighbors + 1]
            graph[row, columns] = lengths[row, columns]
        graph = np.minimum(graph, graph.T)
        finite = np.isfinite(graph)
        sparse = csr_matrix((graph[finite], np.nonzero(finite)), shape=graph.shape)
        geodesic = shortest_path(sparse, directed=False, unweighted=False)
        if np.isfinite(geodesic).all():
            return _square_distance_matrix(geodesic)
    raise RuntimeError("could not construct a connected k-nearest-neighbor graph")


def circular_phase_alignment(
    recovered_angles: np.ndarray,
    target_angles: np.ndarray,
) -> Dict[str, float]:
    """Measure circular alignment after allowing rotation and orientation reversal."""
    recovered = np.asarray(recovered_angles, dtype=np.float64)
    target = np.asarray(target_angles, dtype=np.float64)
    if recovered.ndim != 1 or target.shape != recovered.shape or len(recovered) < 3:
        raise ValueError("recovered and target angles must be equal one-dimensional arrays")
    if not np.isfinite(recovered).all() or not np.isfinite(target).all():
        raise ValueError("angles must be finite")
    positive = np.mean(np.exp(1j * (recovered - target)))
    negative = np.mean(np.exp(1j * (recovered + target)))
    if abs(positive) >= abs(negative):
        selected = positive
        orientation = 1.0
    else:
        selected = negative
        orientation = -1.0
    return {
        "alignment": float(abs(selected)),
        "orientation": orientation,
        "rotation_radians": float(np.angle(selected)),
    }


def circular_winding_degree(angles: np.ndarray) -> float:
    """Estimate winding degree along an ordered closed semantic-phase grid."""
    values = np.asarray(angles, dtype=np.float64)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("angles must be a finite one-dimensional array of length >= 3")
    closed = np.concatenate((values, values[:1]))
    increments = np.angle(np.exp(1j * np.diff(closed)))
    return float(increments.sum() / (2.0 * np.pi))


def nuisance_collapse_ratio(
    distance_matrix: np.ndarray,
    semantic_groups: np.ndarray,
) -> Dict[str, float]:
    """Compare within-semantic nuisance distance with between-semantic distance."""
    distances = _square_distance_matrix(distance_matrix)
    groups = np.asarray(semantic_groups)
    if groups.ndim != 1 or len(groups) != len(distances):
        raise ValueError("semantic_groups must contain one label per point")
    same = groups[:, None] == groups[None, :]
    diagonal = np.eye(len(groups), dtype=bool)
    within = distances[same & ~diagonal]
    between = distances[~same]
    if not within.size or not between.size:
        raise ValueError("at least two semantic groups with repeated samples are required")
    within_mean = float(within.mean())
    between_mean = float(between.mean())
    return {
        "within_semantic_mean_distance": within_mean,
        "between_semantic_mean_distance": between_mean,
        "nuisance_collapse_ratio": (
            within_mean / between_mean if between_mean > 0.0 else float("inf")
        ),
    }


def paired_geometry_alignment(
    reference_distances: np.ndarray,
    representation_distances: np.ndarray,
    *,
    neighbors: int,
) -> Dict[str, float]:
    """Compare two metrics using their known pointwise correspondence.

    Reports global rank agreement, optimal-scale stress, and directed local
    neighborhood recall. This is a geometry-matching diagnostic, not an
    induced chain map or proof that corresponding homology generators agree.
    """
    reference = _square_distance_matrix(reference_distances)
    representation = _square_distance_matrix(representation_distances)
    if reference.shape != representation.shape:
        raise ValueError("paired distance matrices must have equal shape")
    point_count = len(reference)
    if not 1 <= neighbors < point_count:
        raise ValueError("neighbors must be between 1 and point_count - 1")
    upper = np.triu_indices(point_count, k=1)
    reference_values = reference[upper]
    representation_values = representation[upper]
    if np.std(reference_values) <= 1e-12 or np.std(representation_values) <= 1e-12:
        correlation = 0.0
    else:
        correlation = float(
            spearmanr(reference_values, representation_values).statistic
        )
    denominator = float(np.dot(representation_values, representation_values))
    scale = (
        float(np.dot(reference_values, representation_values) / denominator)
        if denominator > 0.0
        else 0.0
    )
    reference_norm = float(np.linalg.norm(reference_values))
    stress = (
        float(
            np.linalg.norm(reference_values - scale * representation_values)
            / reference_norm
        )
        if reference_norm > 0.0
        else 0.0
    )
    reference_neighbors = np.argsort(reference, axis=1)[:, 1 : neighbors + 1]
    representation_neighbors = np.argsort(representation, axis=1)[
        :, 1 : neighbors + 1
    ]
    recalls = [
        len(set(reference_neighbors[index]).intersection(representation_neighbors[index]))
        / neighbors
        for index in range(point_count)
    ]
    return {
        "distance_spearman": correlation,
        "optimal_scale": scale,
        "normalized_stress": stress,
        "neighborhood_recall": float(np.mean(recalls)),
        "neighbors": float(neighbors),
    }


def complex_defect_charge(
    complex_field: np.ndarray,
    phase_values: np.ndarray,
    path_values: np.ndarray,
) -> Dict[str, Any]:
    """Compute indexed zero cells on a periodic phase/path cylinder.

    The field is sampled as ``field[path_index, phase_index]``. Each cell's
    boundary winding detects a zero charge. With the implemented orientation,
    endpoint degree change equals the negative sum of cell windings. This is a
    discrete numerical certificate; interval bounds are required for an exact
    root-isolation theorem.
    """
    field = np.asarray(complex_field, dtype=np.complex128)
    phases = np.asarray(phase_values, dtype=np.float64)
    path = np.asarray(path_values, dtype=np.float64)
    if field.ndim != 2 or field.shape != (len(path), len(phases)):
        raise ValueError("complex_field must have shape (path, phase)")
    if len(path) < 2 or len(phases) < 3:
        raise ValueError("defect grid requires at least two path and three phase points")
    if not np.isfinite(field.real).all() or not np.isfinite(field.imag).all():
        raise ValueError("complex field must be finite")
    if np.any(np.diff(path) <= 0.0) or np.any(np.diff(phases) <= 0.0):
        raise ValueError("phase and path coordinates must be strictly increasing")

    def increment(first: complex, second: complex) -> float:
        return float(np.angle(second * np.conj(first)))

    def boundary_degree(row: np.ndarray) -> float:
        closed = np.concatenate((row, row[:1]))
        increments = np.angle(closed[1:] * np.conj(closed[:-1]))
        return float(increments.sum() / (2.0 * np.pi))

    cells = []
    phase_period = 2.0 * np.pi
    for path_index in range(len(path) - 1):
        for phase_index in range(len(phases)):
            next_phase = (phase_index + 1) % len(phases)
            corners = (
                field[path_index, phase_index],
                field[path_index, next_phase],
                field[path_index + 1, next_phase],
                field[path_index + 1, phase_index],
            )
            winding = sum(
                increment(corners[index], corners[(index + 1) % 4])
                for index in range(4)
            ) / (2.0 * np.pi)
            index = int(np.rint(winding))
            if index:
                phase_start = float(phases[phase_index])
                phase_end = (
                    float(phases[next_phase])
                    if next_phase
                    else float(phases[0] + phase_period)
                )
                cells.append(
                    {
                        "path_index": path_index,
                        "phase_index": phase_index,
                        "path_start": float(path[path_index]),
                        "path_end": float(path[path_index + 1]),
                        "phase_start": phase_start,
                        "phase_end": phase_end,
                        "cell_winding": index,
                        "defect_index": -index,
                        "minimum_corner_magnitude": float(
                            min(abs(value) for value in corners)
                        ),
                    }
                )
    start_degree = boundary_degree(field[0])
    end_degree = boundary_degree(field[-1])
    endpoint_change = int(np.rint(end_degree - start_degree))
    defect_charge = sum(cell["defect_index"] for cell in cells)
    return {
        "start_degree": start_degree,
        "end_degree": end_degree,
        "endpoint_degree_change": endpoint_change,
        "defect_charge": defect_charge,
        "charge_identity_holds": defect_charge == endpoint_change,
        "defect_cell_count": len(cells),
        "defect_cells": cells,
        "minimum_grid_magnitude": float(np.abs(field).min()),
    }


__all__ = [
    "PersistenceSummary",
    "bootstrap_max_h1",
    "circular_phase_alignment",
    "circular_winding_degree",
    "fisher_rao_distance_matrix",
    "knn_geodesic_distance_matrix",
    "nuisance_collapse_ratio",
    "paired_geometry_alignment",
    "complex_defect_charge",
    "persistence_diagrams",
    "persistent_cohomology_circle_coordinate",
    "representation_distance_matrix",
    "summarize_persistence",
]
