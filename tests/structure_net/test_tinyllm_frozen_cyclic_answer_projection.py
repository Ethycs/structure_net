from pathlib import Path

import pytest
import torch

from experiments.structure_net.tinyllm_frozen_cyclic_answer_projection import (
    ALL_ORDERS,
    CyclicProjectionConfig,
    _implementation_sources,
    classify_projection,
    harmonic_project_logits,
)


def test_primary_configuration_is_locked() -> None:
    config = CyclicProjectionConfig()
    assert config.orders == ALL_ORDERS
    assert config.strict_orders == (1, 2, 4)
    assert config.radii == (1.0, 8.0)
    assert config.example_limit == 1024


def test_primary_configuration_rejects_scope_drift() -> None:
    with pytest.raises(ValueError, match="primary cyclic projection"):
        CyclicProjectionConfig(orders=(0, 1, 8), strict_orders=(1,))


def test_identity_and_uniform_projection_controls() -> None:
    generator = torch.Generator().manual_seed(17)
    posterior = torch.softmax(torch.randn(32, 16, generator=generator), dim=1)
    _, identity = harmonic_project_logits(posterior, 8)
    _, uniform = harmonic_project_logits(posterior, 0)
    assert torch.allclose(identity, posterior, atol=2e-6, rtol=0)
    assert torch.allclose(uniform, torch.full_like(uniform, 1 / 16), atol=2e-6, rtol=0)


def _record(
    bins_1,
    bins_8,
    *,
    oracle=True,
    natural=True,
    shifted=False,
):
    return {
        regime: {
            "radius_1": {
                "reachable_bins": list(bins_1),
                "minimum_cross_entropy_selection_pass": oracle,
            },
            "radius_8": {"reachable_bins": list(bins_8)},
            "natural_metrics_pass": natural,
            "shifted_natural_metrics_pass": shifted,
        }
        for regime in ("composition", "extrapolation")
    }


def test_locked_classification_priority() -> None:
    all_bins = range(16)
    source_bins = (2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14)
    deployable = {
        1: _record(all_bins, all_bins),
        2: _record(source_bins, source_bins),
        4: _record(source_bins, source_bins),
    }
    assert classify_projection(deployable) == (
        "cyclic_answer_projection_repairs_deployable_chart"
    )
    chart_only = dict(deployable)
    chart_only[1] = _record(all_bins, all_bins, natural=False)
    assert classify_projection(chart_only) == (
        "cyclic_answer_projection_repairs_chart_not_natural_calibration"
    )
    out_of_range = dict(deployable)
    out_of_range[1] = _record(source_bins, all_bins)
    assert classify_projection(out_of_range) == (
        "cyclic_answer_projection_repairs_only_out_of_range"
    )


def test_no_recovered_bin_classification() -> None:
    source_bins = (2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14)
    records = {order: _record(source_bins, source_bins) for order in (1, 2, 4)}
    assert classify_projection(records) == (
        "cyclic_answer_projection_does_not_repair_coverage"
    )


def test_source_and_preregistration_identities() -> None:
    sources = _implementation_sources()
    assert sources["preregistration"] == (
        "40a24744e956664b1c67950067021f45c54738f7d7dc0ae27e945e48f01ea05d"
    )
    assert Path(
        "data/experiments/tinyllm_frozen_scalar_domain_extension/"
        "20260811_d10_learned_seed29_registered/diagnostics.npz"
    ).is_file()
