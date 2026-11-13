"""Controls for prospective causal component swaps; no GPU allocation."""
import copy

import pytest
import torch

from experiments.structure_net.jspace_causal_components import (
    CONFIG, build_tasks, classify, competence, edit_parts, make_patch,
    paired_test, query_start, select_configuration, success_metrics, validate_tasks,
)


def toy_fit():
    full = torch.tensor([[2., 0., 0., 0.], [0., 2., 0., 0.], [-2., 0., 0., 0.], [0., -2., 0., 0.]])
    j = .25*full
    return {"mean": torch.tensor([.2, .3, 0., 0.]), "full": full, "J": j, "remainder": full-j,
            "permutation": torch.tensor([2, 3, 0, 1]), "sign": torch.tensor([1, -1, 1, -1])}


def test_fresh_tasks_exclude_parent_and_pair_donor_exactly():
    counts = {name: 16 for name in CONFIG["counts"]}
    first = build_tasks({"splits": {}}, counts)
    second = build_tasks(first, counts)
    old = {row["world_id"] for rows in first["splits"].values() for row in rows}
    fresh = {row["world_id"] for rows in second["splits"].values() for row in rows}
    assert not old & fresh
    assert len(fresh) == 64
    assert second == build_tasks(first, counts)
    assert {row["template_id"] for rows in second["splits"].values() for row in rows} == {"causal_shared_json_v1"}
    validate_tasks(second, old)
    broken = copy.deepcopy(second)
    broken["donors"]["evaluation"][0]["answer"] = "wrong"
    with pytest.raises(ValueError, match="counterfactual"):
        validate_tasks(broken, old)


def test_edit_decomposition_and_full_coordinate_swap():
    fitted = toy_fit()
    h = torch.tensor([[[4.2, 2.3, 1., -1.], [2.2, .3, 2., 3.]]])
    parts = edit_parts(h, fitted, 0, 1)
    torch.testing.assert_close(parts["full"], parts["J"]+parts["remainder"])
    torch.testing.assert_close(h+parts["full"], torch.tensor([[[2.2, 4.3, 1., -1.], [.2, 2.3, 2., 3.]]]))


@pytest.mark.parametrize("scope,start", [("final", 0), ("query_suffix", 1)])
def test_scopes_and_norm_matched_controls(scope, start):
    fitted = toy_fit()
    h = torch.tensor([[[8., 2., 1., 3.], [4., 2., -1., 2.], [3., -1., 0., 1.]]])
    results = {}
    for arm in CONFIG["arms"]:
        patch, diagnostics = make_patch(fitted, 0, 1, arm=arm, scope=scope, start=start)
        changed = patch(h)
        assert changed.dtype == h.dtype and changed.shape == h.shape
        boundary = 2 if scope == "final" else start
        torch.testing.assert_close(changed[:, :boundary], h[:, :boundary], rtol=0, atol=0)
        assert diagnostics["positions"] == 3-boundary
        results[arm] = changed-h
    torch.testing.assert_close(results["full"], results["J"]+results["remainder"])
    for control in ("random_J", "remainder_norm_J"):
        torch.testing.assert_close(results[control].norm(dim=-1), results["J"].norm(dim=-1))
    torch.testing.assert_close(results["random_full"].norm(dim=-1), results["full"].norm(dim=-1))


def test_zero_J_controls_are_no_ops_and_invalid_scope_fails():
    fitted = toy_fit()
    fitted["J"] = torch.zeros_like(fitted["full"])
    fitted["remainder"] = fitted["full"].clone()
    h = torch.randn(1, 3, 4)
    for arm in ("J", "random_J", "remainder_norm_J"):
        patch, _ = make_patch(fitted, 0, 1, arm=arm, scope="final")
        torch.testing.assert_close(patch(h), h, rtol=0, atol=0)
    with pytest.raises(ValueError, match="scope"):
        make_patch(fitted, 0, 1, arm="J", scope="all_facts")


def test_query_boundary_uses_character_offsets():
    class CharacterTokenizer:
        def __call__(self, text, **kwargs):
            return {"offset_mapping": [(i, i+1) for i in range(len(text))]}
    text = 'Facts {"x":1}\nQuestion?\nAssistant:'
    assert query_start(CharacterTokenizer(), text, "Question?") == text.index("Question?")
    with pytest.raises(ValueError, match="not found"):
        query_start(CharacterTokenizer(), text, "Missing")


def test_competence_gates_and_pair_alignment():
    source = [{"case_id": str(i), "original_correct": i < 9} for i in range(10)]
    donor = [{"case_id": str(i)+":donor", "original_correct": i > 0} for i in range(10)]
    result = competence(source, donor)
    assert result["passed"] and result["eligible_count"] == 8
    donor[1]["original_correct"] = False
    assert not competence(source, donor)["passed"]
    with pytest.raises(ValueError, match="alignment"):
        competence(source, list(reversed(donor)))


def test_selection_ties_and_paired_statistics():
    rows = [{"case_id": str(i), "layer": layer, "scope": scope, "arm": arm, "target_correct": arm == "full"}
            for layer, scope in CONFIG["selection_order"] for arm in ("full", "random_full") for i in range(10)]
    selected = select_configuration(rows, {str(i) for i in range(10)})
    assert selected["selected"]["layer"] == 24 and selected["selected"]["scope"] == "final"
    left = [{"case_id": str(i), "target_correct": True, "original_correct": False} for i in range(10)]
    right = [{**row, "target_correct": False} for row in left]
    stats = paired_test(left, right)
    assert stats["difference"] == 1 and stats["two_sided_exact_p"] == 2/1024
    assert success_metrics(left)["target_correct"] == 10
    with pytest.raises(ValueError, match="Duplicate"):
        paired_test(left+[left[0]], right)


def test_classification_requires_full_assay_before_J_claim():
    metrics = {"full": {"target_success": .2}, "J": {"target_success": .2}}
    comparisons = {key: {"difference": .1, "two_sided_exact_p": .05}
                   for key in ("full_vs_random", "J_vs_random", "J_vs_matched_remainder")}
    result = classify(metrics, comparisons)
    assert result["conditional_J_gate"] and result["confirmed"] is False
    comparisons["full_vs_random"]["two_sided_exact_p"] = .06
    assert classify(metrics, comparisons)["classification"] == "learned_CAV_causality_not_established"
    comparisons["full_vs_random"]["two_sided_exact_p"] = .01
    comparisons["J_vs_random"]["difference"] = .099
    assert classify(metrics, comparisons)["classification"] == "full_CAV_causal_J_advantage_not_established"
