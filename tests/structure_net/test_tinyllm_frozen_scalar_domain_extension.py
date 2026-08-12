from pathlib import Path

import pytest
import torch

from experiments.structure_net.tinyllm_frozen_scalar_domain_extension import (
    DomainExtensionConfig,
    SOURCE_MISSING_BINS,
    _bin_regions,
    _implementation_sources,
    classify_external_bins,
)


def test_primary_config_is_locked_and_nested() -> None:
    config = DomainExtensionConfig()
    assert config.radii == (1.0, 2.0, 4.0, 8.0)
    assert config.grid_step == 1 / 2048
    assert config.fixed_grid_points == 32769
    assert config.samples_per_regime == 1024


def test_primary_config_rejects_scope_drift() -> None:
    with pytest.raises(ValueError, match="only the registered"):
        DomainExtensionConfig(seed=17)
    with pytest.raises(ValueError, match="primary domain-extension"):
        DomainExtensionConfig(radii=(1.0, 2.0, 4.0))


@pytest.mark.parametrize(
    ("discovered", "expected"),
    [
        (
            {"composition": SOURCE_MISSING_BINS, "extrapolation": SOURCE_MISSING_BINS},
            "bounded_encoder_range_hole",
        ),
        (
            {"composition": (), "extrapolation": ()},
            "continuation_answer_curve_hole_persists_to_radius_8",
        ),
        (
            {"composition": (0, 1), "extrapolation": (0,)},
            "mixed_range_and_answer_curve_hole",
        ),
    ],
)
def test_locked_classification(discovered, expected) -> None:
    assert classify_external_bins(discovered) == expected


def test_bin_regions_report_resolved_and_missing_bins() -> None:
    candidates = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    posterior = torch.tensor(
        [
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
            [0.1, 0.9, 0.0],
            [0.2, 0.8, 0.0],
            [0.9, 0.1, 0.0],
        ]
    )
    reachable, regions = _bin_regions(candidates, posterior, 3)
    assert reachable == [0, 1]
    assert regions["0"]["minimum_scalar"] == -2.0
    assert regions["1"]["nearest_to_zero_scalar"] == 0.0
    assert regions["2"] is None


def test_frozen_sources_match_preregistration() -> None:
    sources = _implementation_sources()
    assert len(sources["runner"]) == 64
    assert sources["preregistration"] == (
        "1eec9eec7832124d5e4fd8d66632e2bede641cba1f4559f2ae677d909670a96e"
    )
    assert Path(
        "data/experiments/tinyllm_frozen_scalar_interface_decomposition/"
        "20260811_d6_d10_preregistered/campaign_results.json"
    ).is_file()
