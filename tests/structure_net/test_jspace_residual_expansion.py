import numpy as np
import pytest
from scipy.optimize import nnls

from experiments.structure_net.jspace_residual_expansion import (
    evaluate_expansions, fit_expansions, nonnegative_pursuit, remainder,
)


def test_repeating_same_projector_cannot_extract_more_but_new_atom_can():
    rng = np.random.default_rng(7)
    training = rng.normal(size=(100, 6))*np.array([1., 8., 4., 2., 1., 1.])
    dictionary = np.eye(6)
    fitted = fit_expansions(training, dictionary, [0], max_rank=5)
    assert fitted["selected_indices"][:3] == [0, 1, 2]
    test = rng.normal(size=(40, 6))
    scores = evaluate_expansions(test, fitted, [1, 2, 4, 5])
    assert scores["fixed_projector_second_pass_energy_fraction"] < 1e-25
    for arm in ("jacobian", "pca", "random"):
        residual = [row["raw"]["residual_energy_fraction"] for row in scores[arm]]
        assert np.all(np.diff(residual) <= 1e-12)
        assert residual[-1] < residual[0]
        np.testing.assert_allclose(fitted[arm] @ fitted[arm].T, np.eye(5), atol=1e-12)


def test_duplicate_dictionary_rows_exhaust_instead_of_inventing_rank():
    x = np.random.default_rng(4).normal(size=(50, 5))
    dictionary = np.array([[1., 0, 0, 0, 0], [2., 0, 0, 0, 0], [1., 1, 0, 0, 0]])
    fitted = fit_expansions(x, dictionary, [0], max_rank=3)
    assert fitted["jacobian"].shape == (2, 5)
    evaluated = evaluate_expansions(x, fitted, [1, 2, 3])
    assert evaluated["jacobian"][-1]["achieved_rank"] == 2
    assert evaluated["jacobian"][-1]["raw"] == evaluated["jacobian"][-2]["raw"]


def test_sparse_pursuit_keeps_sign_constraints_and_full_endpoint_matches_nnls():
    dictionary = np.eye(4)
    values = np.array([[1., 3., 2., -4.], [-1., -2., -3., -4.]])
    result = nonnegative_pursuit(values, dictionary, [0], budgets=[1, 2, 3, 4])
    assert result["selection_order"][0] == [0, 1, 2]
    for budget, coefficients in result["coefficients"].items():
        assert np.all(coefficients >= 0)
        assert np.all(np.sum(coefficients > 1e-10, axis=1) <= budget)
    for i, x in enumerate(values):
        expected, _ = nnls(dictionary.T, x)
        np.testing.assert_allclose(result["coefficients"][4][i], expected, atol=1e-12)
    np.testing.assert_allclose(result["final_residual"], [[0, 0, 0, -4], [-1, -2, -3, -4]], atol=1e-12)
    assert result["max_full_cone_positive_correlation"] < 1e-12


def test_nonorthogonal_nnls_refit_and_residual_reconstruct_input():
    rng = np.random.default_rng(19)
    dictionary = rng.normal(size=(8, 12))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    values = rng.normal(size=(6, 12))
    result = nonnegative_pursuit(values, dictionary, [0, 1], budgets=[2, 4, 6, 8])
    np.testing.assert_allclose(result["coefficients"][8] @ dictionary+result["final_residual"], values, atol=1e-12)
    for i, value in enumerate(values):
        expected, _ = nnls(dictionary.T, value)
        np.testing.assert_allclose(result["coefficients"][8][i], expected, atol=1e-10)
    assert np.all(np.diff([row["residual_energy_fraction"] for row in result["metrics"]]) <= 1e-12)


def test_full_signed_span_and_positive_cone_are_different_objects():
    values = np.array([[1., -2., 3.]])
    dictionary = np.eye(3)
    np.testing.assert_allclose(remainder(values, dictionary), 0, atol=1e-12)
    sparse = nonnegative_pursuit(values, dictionary, [0], budgets=[1, 2, 3])
    assert sparse["metrics"][-1]["residual_energy_fraction"] == pytest.approx(4/14)


def test_invalid_data_and_support_rejected():
    with pytest.raises(ValueError, match="finite"):
        fit_expansions(np.full((20, 4), np.nan), np.eye(4), [0], max_rank=3)
    with pytest.raises(ValueError, match="Initial"):
        fit_expansions(np.ones((20, 4)), np.eye(4), [0, 0], max_rank=3)
    with pytest.raises(ValueError, match="Budgets"):
        nonnegative_pursuit(np.ones((2, 4)), np.eye(4), [0], budgets=[1, 3])
