import numpy as np
import pytest

from structure_net.components.analyzers import (
    bootstrap_max_h1,
    circular_phase_alignment,
    circular_winding_degree,
    complex_defect_charge,
    fisher_rao_distance_matrix,
    knn_geodesic_distance_matrix,
    nuisance_collapse_ratio,
    paired_geometry_alignment,
    persistent_cohomology_circle_coordinate,
    representation_distance_matrix,
    summarize_persistence,
)


def test_circle_has_a_dominant_h1_interval_but_line_does_not():
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    line = np.column_stack((np.linspace(-1.0, 1.0, 24), np.zeros(24)))

    circle_summary = summarize_persistence(
        representation_distance_matrix(circle, metric="euclidean")
    )
    line_summary = summarize_persistence(
        representation_distance_matrix(line, metric="euclidean")
    )

    assert circle_summary.h1_count >= 1
    assert circle_summary.max_h1_lifetime > 1.0
    assert line_summary.max_h1_lifetime == 0.0


def test_persistent_cohomology_discovers_circle_coordinate_without_labels():
    target = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    points = np.column_stack((np.cos(target), np.sin(target)))
    coordinate = persistent_cohomology_circle_coordinate(
        representation_distance_matrix(points, metric="euclidean")
    )
    alignment = circular_phase_alignment(coordinate["angles"], target)

    assert coordinate["found"] is True
    assert coordinate["lifetime"] > 1.0
    assert alignment["alignment"] == pytest.approx(1.0)


def test_fisher_rao_distance_is_symmetric_and_zero_on_the_diagonal():
    posteriors = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    distances = fisher_rao_distance_matrix(posteriors)

    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0.0)
    assert np.all(distances[np.triu_indices(3, 1)] > 0.0)


def test_knn_geodesic_uses_task_edge_lengths_and_connects_graph():
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    neighborhood = representation_distance_matrix(points, metric="euclidean")
    edge_lengths = 2.0 * neighborhood

    geodesic = knn_geodesic_distance_matrix(
        neighborhood, edge_lengths, neighbors=2
    )

    assert geodesic.shape == (12, 12)
    assert np.isfinite(geodesic).all()
    assert np.allclose(geodesic, geodesic.T)
    assert geodesic[0, 1] == pytest.approx(2.0 * neighborhood[0, 1])


def test_bootstrap_is_seeded_and_reports_h1_distribution():
    angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    distances = representation_distance_matrix(points, metric="euclidean")

    first = bootstrap_max_h1(
        distances, repeats=4, sample_fraction=0.8, seed=9
    )
    second = bootstrap_max_h1(
        distances, repeats=4, sample_fraction=0.8, seed=9
    )

    assert first == second
    assert len(first["values"]) == 4
    assert first["positive_fraction"] == 1.0


def test_map_aware_circle_diagnostics_allow_rotation_and_orientation():
    target = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    recovered = -target + 0.7
    alignment = circular_phase_alignment(recovered, target)

    assert alignment["alignment"] == pytest.approx(1.0)
    assert alignment["orientation"] == -1.0
    assert circular_winding_degree(recovered) == pytest.approx(-1.0)


def test_nuisance_collapse_ratio_separates_within_and_between_groups():
    points = np.array([[0.0], [0.1], [2.0], [2.1]])
    distances = representation_distance_matrix(points, metric="euclidean")
    result = nuisance_collapse_ratio(distances, np.array([0, 0, 1, 1]))

    assert result["within_semantic_mean_distance"] == pytest.approx(0.1)
    assert result["between_semantic_mean_distance"] == pytest.approx(2.0)
    assert result["nuisance_collapse_ratio"] == pytest.approx(0.05)


def test_paired_geometry_alignment_uses_shared_sample_correspondence():
    target = np.linspace(-1.0, 1.0, 16)[:, None]
    reference = representation_distance_matrix(target, metric="euclidean")
    aligned = paired_geometry_alignment(reference, 3.0 * reference, neighbors=2)
    permuted = np.random.default_rng(5).permutation(len(target))
    mismatched = paired_geometry_alignment(
        reference, reference[np.ix_(permuted, permuted)], neighbors=2
    )

    assert aligned["distance_spearman"] > 0.999
    assert aligned["normalized_stress"] == pytest.approx(0.0)
    assert aligned["neighborhood_recall"] == pytest.approx(1.0)
    assert mismatched["distance_spearman"] < 0.5


def test_complex_defect_charge_equals_degree_change_on_cylinder():
    phases = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    path = np.linspace(0.0, 1.0, 66)
    field = (1.0 - path[:, None]) + path[:, None] * np.exp(1j * phases[None, :])

    result = complex_defect_charge(field, phases, path)

    assert result["endpoint_degree_change"] == 1
    assert result["defect_charge"] == 1
    assert result["charge_identity_holds"] is True
    assert result["defect_cell_count"] == 1


def test_complex_defect_charge_is_zero_for_constant_degree_homotopy():
    phases = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    path = np.linspace(0.0, 1.0, 17)
    field = np.exp(1j * phases[None, :]) * np.ones((len(path), 1))

    result = complex_defect_charge(field, phases, path)

    assert result["endpoint_degree_change"] == 0
    assert result["defect_charge"] == 0
    assert result["defect_cell_count"] == 0
