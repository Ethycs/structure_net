"""Focused checks of benchmark-derived polynomial fitting and its accounting."""
import json

import pytest
import torch

from experiments.structure_net.jspace_benchmark_groebner import (
    COEFFICIENT_DENOMINATOR,
    fit_benchmark_groebner,
    polynomial_features,
    select_nested_anchors,
)


def benchmark():
    generator = torch.Generator().manual_seed(315)

    def activations(count):
        z = torch.randn(count, 3, dtype=torch.float64, generator=generator)
        return torch.stack((z[:, 0], z[:, 1], z[:, 2], z[:, 0] * z[:, 1],
                            z[:, 0].square(), z[:, 1].square(),
                            z[:, 0] * z[:, 2], z[:, 2].square()), dim=1)

    train, dev = activations(128), activations(64)
    jacobian = torch.zeros(48, 8, dtype=torch.float64)
    jacobian[:, :3] = torch.randn(48, 3, dtype=torch.float64, generator=generator) * torch.tensor([10., 7., 3.])
    weight = torch.randn(5, 8, dtype=torch.float64, generator=generator)
    return train, ["a"] * 64 + ["b"] * 64, jacobian, weight, dev, ["a"] * 32 + ["b"] * 32


def fit_example(**kwargs):
    return fit_benchmark_groebner(
        *benchmark(), anchor_widths=(2, 3), remove_count=2,
        svd_rank=6, oversample=2, power_iterations=1,
        ridge=1e-8, chunk_rows=17, **kwargs,
    )


def test_feature_order_matches_existing_compiler():
    z = torch.tensor([[2., 3., 5.]])
    expected = torch.tensor([[1., 2., 3., 5., 4., 6., 10., 9., 15., 25.]])
    torch.testing.assert_close(polynomial_features(z), expected)


def test_randomized_svd_uses_all_rows_and_reports_captured_energy():
    generator = torch.Generator().manual_seed(19)
    j = torch.randn(40, 8, dtype=torch.float64, generator=generator)
    j[-1] *= 20  # A late row must contribute to the reported energy.
    scale = torch.linspace(.5, 2., 8, dtype=torch.float64)
    anchors, info = select_nested_anchors(j, scale, 3, seed=11, svd_rank=5,
                                         oversample=3, chunk_rows=7)
    again, same = select_nested_anchors(j, scale, 3, seed=11, svd_rank=5,
                                       oversample=3, chunk_rows=7)
    weighted = j * scale
    singular = torch.linalg.svdvals(weighted)
    assert anchors == again and len(set(anchors)) == 3
    assert info == same
    assert info["all_rows_used"] is True
    assert info["total_standardized_jacobian_energy"] == pytest.approx(float(weighted.square().sum()), rel=1e-13)
    assert info["captured_energy_fraction"] == pytest.approx(float(singular[:5].square().sum() / singular.square().sum()), rel=1e-12)


def test_fits_have_nested_anchors_train_only_units_and_graph_certificate():
    result = fit_example()
    train = benchmark()[0]
    assert result["fits"]["2"]["anchors"] == result["fits"]["3"]["anchors"][:2]
    for width in (2, 3):
        fit = result["fits"][str(width)]
        expected_features = 1 + width + width * (width + 1) // 2
        assert fit["coefficients"]["2"].shape == (8, expected_features)
        assert fit["removed"].dtype == torch.long and fit["removed"].device.type == "cpu"
        assert len(fit["removed"]) == 2
        assert not set(fit["anchors"]) & set(fit["removed"].tolist())
        torch.testing.assert_close(fit["mean"], train.mean(0), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(fit["scale"], train.std(0), rtol=1e-12, atol=1e-12)
        integer = fit["coefficients"]["2"] * COEFFICIENT_DENOMINATOR
        torch.testing.assert_close(integer, integer.round(), rtol=0, atol=0)
        certificate = fit["metrics"]["certificate"]
        assert certificate["buchberger_product_criterion"] is True
        assert certificate["reduced_graph_basis"] is True
        assert "not identities" in certificate["scope"]
        assert "infinite dimensional" in certificate["standard_monomials"]
        assert fit["metrics"]["interfaces"]["native_dev"]["pooled_output"]["relative_rmse"] < 2e-5
    json.dumps(result["metrics"], allow_nan=False)
    json.dumps(result["anchor_selection"], allow_nan=False)


def test_compiled_output_error_matches_explicit_reconstruction_and_selection():
    train, tf, j, weight, dev, df = benchmark()
    compressed = dev * 1.1 + .3
    result = fit_example(compressed_dev_activations=compressed)
    assert result["metrics"]["selection_interface"] == "compressed_dev"
    scores = {}
    for width, fit in result["fits"].items():
        anchors, removed = fit["anchors"], fit["removed"]
        z = (compressed[:, anchors] - fit["mean"][anchors]) / fit["scale"][anchors]
        reconstructed = compressed.clone()
        prediction = polynomial_features(z) @ fit["coefficients"]["2"].T
        reconstructed[:, removed] = prediction[:, removed] * fit["scale"][removed] + fit["mean"][removed]
        expected = ((reconstructed - compressed) @ weight.T).norm() / (compressed @ weight.T).norm()
        metric = fit["metrics"]["interfaces"]["compressed_dev"]["pooled_output"]["relative_rmse"]
        assert metric == pytest.approx(float(expected), rel=1e-11, abs=1e-12)
        scores[int(width)] = metric
        assert fit["per_channel_nrmse"]["compressed_dev"]["common"].shape == (2, 8)
    assert result["selected_anchor_width"] == min(scores, key=lambda width: (scores[width], width))


def test_fixed_budget_is_flagged_when_relations_are_not_covered():
    train, tf, j, _, dev, df = benchmark()
    result = fit_benchmark_groebner(
        train, tf, j, torch.eye(8, dtype=train.dtype), dev, df,
        anchor_widths=(2,), remove_count=4, svd_rank=4, oversample=2,
        ridge=1e-8, chunk_rows=19,
    )
    fit = result["fits"]["2"]
    reuse = fit["metrics"]["interfaces"]["native_dev"]["redundancy"]
    assert fit["metrics"]["eligibility_nrmse"] == .5
    assert reuse["selected_not_eligible_all_tasks"] >= 1
    assert reuse["all_removed_meet_diagnostic_threshold"] is False
    assert reuse["redundant_occurrences"] == reuse["task_uses"] - reuse["unique_elements"]
    assert reuse["overlap_counts"][0][1] == reuse["overlap_counts"][1][0]


def test_invalid_independent_rank_and_alignment_are_rejected():
    with pytest.raises(ValueError, match="Jacobian energy"):
        select_nested_anchors(torch.zeros(10, 8), torch.ones(8), 2, svd_rank=4)
    train, tf, j, weight, dev, df = benchmark()
    with pytest.raises(ValueError, match="aligned"):
        fit_benchmark_groebner(train, tf, j, weight, dev, df,
                              compressed_dev_activations=dev[:-1], anchor_widths=(2,), remove_count=1)
