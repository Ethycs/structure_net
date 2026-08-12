from __future__ import annotations

import torch

from experiments.structure_net.tinyllm_c3_temporal_quotient_preflight import (
    REGIMES,
    analytic_carrier,
    apply_deck_action,
    build_preflight,
    generate_dataset,
    instantaneous_insufficiency_witness,
)


def test_token_action_obeys_exact_c3_group_law() -> None:
    dataset = generate_dataset("composition")
    for left in range(3):
        for right in range(3):
            composed = apply_deck_action(
                apply_deck_action(dataset.base_tokens, right), left
            )
            direct = apply_deck_action(dataset.base_tokens, (left + right) % 3)
            assert torch.equal(composed, direct)
    assert torch.equal(
        apply_deck_action(
            apply_deck_action(
                apply_deck_action(dataset.base_tokens, 1), 1
            ),
            1,
        ),
        dataset.base_tokens,
    )


def test_cubic_carrier_is_invariant_but_not_instantaneously_sufficient() -> None:
    dataset = generate_dataset("composition")
    base, _ = analytic_carrier(dataset.base_tokens, dataset.calibration)
    for element in range(3):
        transformed, _ = analytic_carrier(
            apply_deck_action(dataset.base_tokens, element), dataset.calibration
        )
        assert float(torch.max(torch.abs(transformed - base))) <= 1e-6
    witness = instantaneous_insufficiency_witness()
    assert witness["pass"] is True
    assert witness["final_carrier_absolute_difference"] <= 1e-12
    assert witness["future_target_absolute_difference"] >= 0.25


def test_preflight_passes_all_no_training_contracts() -> None:
    record = build_preflight()
    assert record["valid"] is True
    assert record["classification"] == "c3_temporal_quotient_preflight_passed"
    assert record["optimizer_steps_executed"] == 0
    assert record["checkpoints_loaded"] == 0
    assert all(record["gates"].values())
    for regime in REGIMES:
        assert record["group_action"][regime]["pass"] is True
        carrier = record["invariant_carrier"][regime]
        assert carrier["carrier_contract_pass"] is True
        assert carrier["temporal_prediction"]["pass"] is True
        assert carrier["shuffled_specificity"]["pass"] is True
