import numpy as np

from experiments.structure_net.tinyllm_defect_surrogate_interval_certificate import (
    Interval,
    boundary_audit,
    derivative_coefficients,
    isolate_candidates,
    krawczyk_box,
)


def test_interval_isolates_unique_positive_oriented_linear_root():
    # Stored coefficient convention: f0=y-.1, f1=x-.1 gives positive determinant
    # after swapping columns to the charged-cell (path, phase) orientation.
    first = np.zeros((2, 2)); first[0, 0] = -0.1; first[0, 1] = 1.0
    second = np.zeros((2, 2)); second[0, 0] = -0.1; second[1, 0] = 1.0
    coefficients = [first, second]
    derivatives = [
        [derivative_coefficients(value, 0), derivative_coefficients(value, 1)]
        for value in coefficients
    ]
    isolated = isolate_candidates(coefficients, initial_grid=3, maximum_depth=8)
    x, y = isolated["candidate_hull"]
    result = krawczyk_box(coefficients, derivatives, x, y)
    assert result["strictly_inside"]
    assert result["positive_oriented_determinant"]
    assert boundary_audit(coefficients, 32)["all_boundary_boxes_exclude_zero"]


def test_interval_rejects_invalid_bounds():
    try:
        Interval(1.0, -1.0)
    except ValueError:
        return
    raise AssertionError("invalid interval was accepted")
