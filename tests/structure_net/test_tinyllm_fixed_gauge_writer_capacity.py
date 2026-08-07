from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import tinyllm_fixed_gauge_writer_capacity as capacity


def test_primary_configuration_and_feature_widths_are_fixed() -> None:
    config = capacity.FixedGaugeWriterCapacityConfig(device="cpu")
    assert config.seeds == (7, 29, 53)
    assert config.low_orders == (2, 4)
    assert config.high_order == 40
    with pytest.raises(ValueError, match="fixed"):
        capacity.FixedGaugeWriterCapacityConfig(seeds=(7,), device="cpu")
    pilot = capacity.FixedGaugeWriterCapacityConfig(
        seeds=(7,), device="cpu", allow_underpowered=True
    )
    assert capacity._evidence_role(pilot) == (
        "systems_lifecycle_only_not_quality_evidence"
    )


def test_fourier_feature_ladder_is_nested_and_has_constant_last() -> None:
    angles = torch.tensor([0.2, 1.1], dtype=torch.float64)
    carrier = torch.stack((torch.cos(angles), torch.sin(angles), torch.ones(2)), 1)
    q1 = capacity.fourier_features(carrier, 1)
    q4 = capacity.fourier_features(carrier, 4)
    q40 = capacity.fourier_features(carrier, 40)
    assert q1.shape == (2, 3)
    assert q4.shape == (2, 9)
    assert q40.shape == (2, 81)
    assert torch.allclose(q1[:, :2], q4[:, :2])
    assert torch.all(q1[:, -1] == 1.0)
    assert torch.all(q40[:, -1] == 1.0)


def test_calibration_context_is_orbit_mean_and_sheet_checked() -> None:
    packet = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [3.0, 4.5],
        ]
    )
    context, difference = capacity.orbit_calibration_context(
        SimpleNamespace(calibration=packet), 2, torch.device("cpu")
    )
    assert torch.allclose(context, torch.tensor([[1.0, 2.0], [3.0, 4.25]]).double())
    assert difference == pytest.approx(0.5)


def test_conditional_features_have_preregistered_width() -> None:
    q4 = torch.randn(5, 9, dtype=torch.float64)
    context = torch.randn(5, 8, dtype=torch.float64)
    value = capacity.conditional_features(q4, context)
    assert value.shape == (5, 81)
    reshaped = value.reshape(5, 9, 9)
    assert torch.allclose(reshaped[:, :, 0], q4)


def test_general_writer_recovers_linear_target() -> None:
    features = torch.randn(64, 9, dtype=torch.float64)
    expected = torch.randn(9, 3, dtype=torch.float64)
    target = features @ expected
    writer = capacity.fit_writer(features, target, 1e-10)
    predicted = features @ writer["linear"] + writer["intercept"]
    assert torch.allclose(predicted, target, atol=1e-8, rtol=1e-8)


def test_padded_writer_preserves_the_declared_feature_block() -> None:
    writer = {
        "linear": torch.randn(5, 3, dtype=torch.float64),
        "intercept": torch.randn(3, dtype=torch.float64),
    }
    value = capacity._pad_writer(writer, 7, 12, 19)
    assert torch.all(value["linear"][:7] == 0.0)
    assert torch.allclose(value["linear"][7:12], writer["linear"])
    assert torch.all(value["linear"][12:] == 0.0)
    assert torch.allclose(value["intercept"], writer["intercept"])


@pytest.mark.parametrize(
    ("contracts", "passed", "expected"),
    [
        (False, {}, "invalid"),
        (True, {"quotient_order2": True}, "low_order_curvature_limited"),
        (True, {"quotient_order4": True}, "low_order_curvature_limited"),
        (True, {"quotient_order40": True}, "high_order_curvature_limited"),
        (
            True,
            {"quotient_order4_context": True},
            "calibration_context_limited",
        ),
        (True, {}, "unresolved_writer_limited"),
    ],
)
def test_checkpoint_classification_is_preregistered(
    contracts: bool, passed: dict[str, bool], expected: str
) -> None:
    assert (
        capacity.classify_checkpoint(
            contracts_pass=contracts, candidate_specific_pass=passed
        )
        == expected
    )


def test_predecessor_campaign_is_hash_locked() -> None:
    campaign, path, details = capacity._load_predecessor(
        capacity.FixedGaugeWriterCapacityConfig(device="cpu")
    )
    assert capacity._sha256(path) == capacity.PREDECESSOR_CAMPAIGN_SHA256
    assert campaign["implementation_sha256"] == (
        capacity.PREDECESSOR_IMPLEMENTATION_SHA256
    )
    assert set(details) == {7, 29, 53}
