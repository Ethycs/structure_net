from __future__ import annotations

import json

import pytest
import torch

from experiments.structure_net import jspace_tasks as tasks


def case(index=0, **changes):
    values = dict(case_id=f"item-{index}", category="test", prompt="Fact: The hidden answer is",
                  intermediate=f"source{index}", answer="one", swap_to=f"target{index}", swap_answer="two")
    values.update(changes)
    return tasks.TwoHopCase(**values)


def test_published_data_adapter_keeps_prompt_and_counterfactual_ground_truth(tmp_path):
    source = tmp_path / "probe-swap.json"
    original = case()
    row = original.to_dict()
    row["name"] = row.pop("case_id")
    source.write_text(json.dumps({"items": [row]}))
    assert tasks.load_cases(source) == [original]
    row["swap_answer"] = "1"
    source.write_text(json.dumps({"items": [row]}))
    with pytest.raises(ValueError, match="change the ground-truth"):
        tasks.load_cases(source)


def test_exclusions_are_independent_of_model_outcomes_and_keep_multitoken_answers():
    leaked = case(0, prompt="Fact: It is daytime, and the time of day is", swap_answer="day")
    valid = case(1, answer="New Zealand", swap_answer="South Africa")
    multi = case(2, intermediate="two words")
    seen = []

    def encode(text):
        seen.append(text)
        return [len(text), 42] if text == " two words" else [sum(map(ord, text))]

    accepted, excluded = tasks.eligible_cases([leaked, valid, multi], encode)
    assert accepted == [valid]
    assert excluded[0]["reasons"] == ["literal_swap_answer_in_prompt"]
    assert excluded[1]["reasons"] == ["intermediate_not_single_token"]
    assert all(value.startswith(" ") for value in seen)
    # Whole words: "day" must not be considered copied from "daytime" alone.
    assert tasks.lexical_contamination(case(prompt="Fact: It is daytime", swap_answer="day")) == ()


def test_pair_group_split_is_order_independent_and_never_pseudoreplicates():
    cases = [case(i) for i in range(20)]
    # Reverse-direction wording variants must not cross splits or inflate n.
    cases += [case(99, intermediate="target0", swap_to="source0")]
    kwargs = dict(seed=17, fit_count=4, development_count=4, evaluation_count=10)
    first = tasks.split_cases(cases, **kwargs)
    second = tasks.split_cases(list(reversed(cases)), **kwargs)
    assert first == second
    groups = [c.pair_group for rows in first.values() for c in rows]
    assert len(groups) == len(set(groups)) == 18
    manifest = tasks.split_manifest(first)
    assert manifest["counts"] == {"fit": 4, "development": 4, "evaluation": 10}
    assert manifest["sha256"] == tasks.split_manifest(second)["sha256"]
    with pytest.raises(ValueError, match="only 20"):
        tasks.split_cases(cases, fit_count=4, development_count=4, evaluation_count=13)


def test_scoring_keeps_raw_denominator_and_conditions_on_parent_only():
    cases = [case(i) for i in range(4)]
    baseline = {"item-0": "1", "item-1": "one.", "item-2": "wrong", "item-3": "wrong"}
    swapped = {"item-0": "two", "item-1": "wrong", "item-2": "2", "item-3": "2"}
    result = tasks.score_predictions(cases, baseline, swapped)
    assert result["baseline_accuracy"]["accuracy"] == .5
    assert result["raw_accuracy"]["accuracy"] == .75
    assert result["parent_correct_accuracy"]["accuracy"] == .5
    assert result["parent_correct_accuracy"]["total"] == 2
    assert result["raw_accuracy"]["total"] == 4
    assert tasks.normalize_answer("two because it has legs") != tasks.normalize_answer("two")
    with pytest.raises(ValueError, match="exactly"):
        tasks.score_predictions(cases, baseline, {"item-0": "two"})


def test_retention_is_scored_against_original_and_zero_denominator_is_explicit():
    cases = [case()]
    result = tasks.score_predictions(cases, {"item-0": "wrong"}, {"item-0": "one"}, target="original")
    assert result["raw_accuracy"]["accuracy"] == 1
    assert result["parent_correct_accuracy"] == {"successes": 0, "total": 0, "accuracy": None, "ci95": None}
    zero = tasks.wilson_interval(0, 40)
    full = tasks.wilson_interval(40, 40)
    assert zero["ci95"][0] == pytest.approx(0)
    assert 0 < zero["ci95"][1] < .1
    assert .9 < full["ci95"][0] < 1
    assert full["ci95"][1] == pytest.approx(1)


def test_paired_uncertainty_uses_discordant_cases_and_is_deterministic():
    first = [False] * 8
    second = [True] * 8
    result = tasks.paired_difference(first, second)
    assert result["second_minus_first"] == 1
    assert result["bootstrap_ci95"] == [1, 1]
    assert result["mcnemar_exact_p"] == pytest.approx(2 / 256)
    assert result == tasks.paired_difference(first, second)
    assert tasks.paired_difference(first, first)["mcnemar_exact_p"] == 1


def test_nonorthogonal_coordinate_swap_matches_paper_and_preserves_complement():
    bank = torch.tensor([[1., 0., 0.], [1., 1., 0.]], dtype=torch.float64)
    h = torch.tensor([2., 3., 4.], dtype=torch.float64)
    changed = tasks.coordinate_swap(h, bank)
    assert torch.allclose(changed, torch.tensor([2., -1., 4.], dtype=h.dtype))
    assert torch.allclose(changed @ torch.linalg.pinv(bank), (h @ torch.linalg.pinv(bank)).flip(0))
    assert torch.allclose(changed - tasks.project_to_span(changed, bank),
                          h - tasks.project_to_span(h, bank))
    # The score-clamp operation is mathematically different for this bank.
    clamped = tasks.replace_scores(h, (h @ bank.T).flip(0), bank)
    assert torch.allclose(clamped, torch.tensor([5., -3., 4.], dtype=h.dtype))
    assert not torch.allclose(changed, clamped)


def test_swap_is_involution_and_noop_at_zero_strength_for_batches():
    bank = torch.tensor([[1., 2., 0., 0.], [0., 1., 1., 0.]], dtype=torch.float64)
    h = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(13), dtype=torch.float64)
    assert torch.allclose(tasks.coordinate_swap(tasks.coordinate_swap(h, bank), bank), h, atol=1e-12)
    assert torch.equal(tasks.coordinate_swap(h, bank, alpha=0), h)
    with pytest.raises(ValueError, match="dependent"):
        tasks.coordinate_swap(h, torch.stack([bank[0], bank[0] * 2]))


def test_projection_handles_rank_deficiency_and_complement_swap_preserves_scores():
    bank = torch.tensor([[1., 0., 0.], [2., 0., 0.]], dtype=torch.float64)
    source = torch.tensor([2., 3., 4.], dtype=torch.float64)
    donor = torch.tensor([20., -30., 7.], dtype=torch.float64)
    basis = tasks.orthonormal_basis(bank)
    assert basis.shape == (1, 3)
    result = tasks.complement_swap(source, donor, bank)
    assert torch.allclose(result, torch.tensor([2., -30., 7.], dtype=source.dtype))
    assert torch.allclose(result @ bank.T, source @ bank.T, atol=1e-12)
    desired = torch.tensor([3., 6.], dtype=source.dtype)
    assert torch.allclose(tasks.replace_scores(source, desired, bank) @ bank.T, desired)
    with pytest.raises(ValueError, match="inconsistent"):
        tasks.replace_scores(source, torch.tensor([3., 7.]), bank)


def test_controls_match_actual_rank_and_pca_refuses_invented_axes():
    random = tasks.random_basis(8, 3, seed=7, dtype=torch.float64)
    assert random.shape == (3, 8)
    assert torch.allclose(random @ random.T, torch.eye(3, dtype=random.dtype), atol=1e-12)
    assert torch.equal(random, tasks.random_basis(8, 3, seed=7, dtype=torch.float64))
    fit = torch.randn(32, 3, generator=torch.Generator().manual_seed(9), dtype=torch.float64) @ random
    pca = tasks.pca_basis(fit, 3)
    assert torch.allclose(tasks.project_to_span(fit, pca), fit, atol=1e-12)
    with pytest.raises(ValueError, match="calibration rank"):
        tasks.pca_basis(fit, 4)
    zero = tasks.project_to_span(fit, torch.empty((0, 8), dtype=fit.dtype))
    assert torch.equal(zero, torch.zeros_like(fit))


def test_nonfinite_input_and_mismatched_batch_are_rejected():
    bank = torch.eye(3)
    with pytest.raises(ValueError, match="finite"):
        tasks.project_to_span(torch.tensor([float("nan"), 0., 0.]), bank)
    with pytest.raises(ValueError, match="hidden width"):
        tasks.project_to_span(torch.zeros(2), bank)
    with pytest.raises(ValueError, match="equal shapes"):
        tasks.complement_swap(torch.zeros(3), torch.zeros(2, 3), bank)
