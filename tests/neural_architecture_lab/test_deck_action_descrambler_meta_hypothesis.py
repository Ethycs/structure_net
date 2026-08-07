from pathlib import Path

from neural_architecture_lab.deck_action_descrambler_meta_hypothesis import (
    build_deck_action_descrambler_meta_hypothesis,
    build_deck_action_experiment_results,
)


RESULTS = Path("data/experiments/tinyllm_deck_action_descrambler/20260806_d6_preregistered/campaign_results.json")


def test_meta_record_keeps_linear_gate_failure_and_causal_front():
    record = build_deck_action_descrambler_meta_hypothesis(RESULTS)
    assert record["hypothesis"]["confirmed"] is False
    assert record["hypothesis"]["confirmation_status"] == "not_confirmed_causal_quotient_front_without_linear_isotypic_split"
    assert record["hypothesis"]["subclaims"]["early_computational_cover"] == "supported_five_of_five_before_attention"
    assert record["result"]["descriptive_metrics"]["degrees"]["2"]["stable_deck_cuts"] == []
    assert record["result"]["descriptive_metrics"]["degrees"]["3"]["stable_deck_cuts"] == []


def test_all_fifteen_frozen_cells_convert_to_ledger_results():
    record = build_deck_action_descrambler_meta_hypothesis(RESULTS)
    experiments = build_deck_action_experiment_results(record, RESULTS)
    assert len(experiments) == 15
    assert {item.model_parameters for item in experiments} == {29_956_224}
    assert all(item.model_checkpoint for item in experiments)
