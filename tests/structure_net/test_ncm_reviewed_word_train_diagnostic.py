import pytest
import torch

from experiments.structure_net import ncm_reviewed_word_train_diagnostic as diagnostic


def data():
    original = [{"id": f"old{i}", "source_cl100k_tokens": i % 4 * 512 + 100} for i in range(252)]
    additional = [{"id": f"new{band}_{i}", "source_cl100k_tokens": band * 512 + 100}
        for band, size in enumerate((196, 181, 17, 44)) for i in range(size)]
    return original + additional, {row["id"] for row in original}, original[:16]


def test_selection_retains_old_control_and_is_order_independent():
    rows, original, old = data()
    selected = diagnostic.selection(rows, original, old)
    assert [row["id"] for row in selected[:16]] == [row["id"] for row in old]
    assert len({row["id"] for row in selected}) == 32
    assert [row["id"] for row in diagnostic.selection(list(reversed(rows)), original, old)] == [row["id"] for row in selected]
    assert all(row["id"] not in original for row in selected[16:])
    assert [sum(row["source_cl100k_tokens"] // 512 == band for row in selected[16:]) for band in range(4)] == [4] * 4


def test_selection_rejects_missing_band_without_replacing_cases():
    rows, original, old = data()
    for row in rows:
        if row["id"].startswith("new2_"):
            row["source_cl100k_tokens"] = 100
    with pytest.raises(ValueError, match="four additional"):
        diagnostic.selection(rows, original, old)


def test_selection_rejects_duplicate_or_missing_old_identity():
    rows, original, old = data()
    with pytest.raises(ValueError):
        diagnostic.selection(rows, original, old[:-1] + old[:1])


def test_counts_ignore_only_padding_and_include_eos():
    target = torch.tensor([[12, 50256, 50257]])
    actual = torch.tensor([[12, 7, 8]])
    assert diagnostic.counts(actual, target) == {"correct": 1, "total": 2}
    with pytest.raises(ValueError):
        diagnostic.counts(actual, target[:, :2])


def test_aggregation_weights_tokens_instead_of_cases():
    rows = []
    for group, total, ce in (("original_16", 2, 1.), ("additional_16", 6, 3.)):
        rows.append({"diagnostic_group": group,
            "arms": {arm: {"correct": 1, "total": total, "CE": ce, "exact_target_text": False} for arm in diagnostic.ARMS},
            "own_gold_agreement": {"correct": 1, "total": total},
            "concepts": [{"active_correct": 1, "active_total": 2} for _ in range(8)]})
    result = diagnostic.aggregate(rows)
    assert result["all_32"]["arms"]["actual_history"]["token_agreement"] == .25
    assert result["all_32"]["arms"]["actual_history"]["token_mean_CE"] == 2.5
    assert result["original_16"]["cases"] == 1
