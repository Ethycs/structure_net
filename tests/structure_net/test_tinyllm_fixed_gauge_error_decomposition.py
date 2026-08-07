from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import tinyllm_fixed_gauge_error_decomposition as diagnostic
from experiments.structure_net.tinyllm_predictive_circle import CircleTaskConfig


def test_primary_configuration_is_fixed() -> None:
    config = diagnostic.FixedGaugeErrorDecompositionConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    with pytest.raises(ValueError, match="fixed"):
        diagnostic.FixedGaugeErrorDecompositionConfig(seeds=(7,), device="cpu")
    pilot = diagnostic.FixedGaugeErrorDecompositionConfig(
        seeds=(7,), device="cpu", allow_underpowered=True
    )
    assert diagnostic._evidence_role(pilot) == (
        "systems_lifecycle_only_not_quality_evidence"
    )


def test_oracle_carrier_is_exact_degree_two_neutral_coordinate() -> None:
    task = CircleTaskConfig()
    phase = torch.tensor([0.2, 0.2 + torch.pi, 1.1, 1.1 + torch.pi])
    carrier, audit = diagnostic.oracle_orbit_carrier(
        SimpleNamespace(phase=phase), task, 2, torch.device("cpu")
    )
    expected = torch.tensor(
        [
            [torch.cos(torch.tensor(0.4)), torch.sin(torch.tensor(0.4)), 1.0],
            [torch.cos(torch.tensor(2.2)), torch.sin(torch.tensor(2.2)), 1.0],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(carrier, expected, atol=1e-7, rtol=1e-7)
    assert audit["mean_shift_bins"] <= 1e-12
    assert audit["p95_shift_bins"] <= 1e-12


@pytest.mark.parametrize(
    ("replay", "oracle", "controls", "observed_oracle", "oracle_oracle", "expected"),
    [
        (True, True, True, True, True, "sensor_limited"),
        (True, True, True, False, True, "sensor_and_fit_mismatch"),
        (True, True, True, False, False, "writer_or_carrier_limited"),
        (False, True, True, True, True, "invalid"),
        (True, False, True, True, True, "invalid"),
        (True, True, False, True, True, "invalid"),
    ],
)
def test_checkpoint_classification_is_exhaustive(
    replay: bool,
    oracle: bool,
    controls: bool,
    observed_oracle: bool,
    oracle_oracle: bool,
    expected: str,
) -> None:
    assert (
        diagnostic.classify_checkpoint(
            replay_pass=replay,
            oracle_contract=oracle,
            controls_pass=controls,
            observed_fit_oracle_pass=observed_oracle,
            oracle_fit_oracle_pass=oracle_oracle,
        )
        == expected
    )


def test_replay_error_compares_declared_continuous_and_coordinate_fields() -> None:
    current = {
        "coordinate_metrics": {"observed_primary": {"variance_explained": 0.9}},
        "states": {
            "observed_primary": {
                "continuous": {"continuous_pass": True, "mean_moment_shift_bins": 0.1}
            }
        },
    }
    primary = {
        "coordinate_metrics": {"fixed_gauge": {"variance_explained": 0.9}},
        "states": {
            "fixed_gauge": {
                "continuous": {"continuous_pass": True, "mean_moment_shift_bins": 0.1}
            }
        },
    }
    assert diagnostic.replay_error(current, primary) == 0.0
    current["states"]["observed_primary"]["continuous"][
        "mean_moment_shift_bins"
    ] = 0.2
    assert diagnostic.replay_error(current, primary) == pytest.approx(0.1)


def test_primary_campaign_is_hash_locked() -> None:
    campaign, path, details = diagnostic._load_primary(
        diagnostic.FixedGaugeErrorDecompositionConfig(device="cpu")
    )
    assert diagnostic._sha256(path) == diagnostic.PRIMARY_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == (
        diagnostic.PRIMARY_IMPLEMENTATION_SHA256
    )
    assert set(details) == {7, 29, 53}
