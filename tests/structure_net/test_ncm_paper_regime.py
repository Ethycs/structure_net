"""Contracts that protect the paper-regime data and context interface."""
import json

import pytest
import torch

from experiments.structure_net.ncm_paper_data import parse_teacher, unique_sources
from experiments.structure_net.ncm_paper_context import right_pack, unpaired_windows


def test_teacher_pairing_and_no_silent_missing_labels():
    assert parse_teacher('{"summaries":[{"id":"b","summary":" B "},{"id":"a","summary":"A"}]}',
                         ["a", "b"]) == {"a": "A", "b": "B"}
    with pytest.raises(ValueError):
        parse_teacher('{"summaries":[{"id":"a","summary":"A"},{"id":"a","summary":"A"}]}', ["a", "b"])
    with pytest.raises(ValueError):
        parse_teacher('{"summaries":[]}', ["a"])


def test_teacher_json_fences_do_not_change_label_text():
    assert parse_teacher('```json\n{"summaries":[{"id":"a","summary":"$5, not $10."}]}\n```',
                         ["a"])["a"] == "$5, not $10."


def test_heldout_selection_excludes_whole_groups_and_duplicate_sources(monkeypatch):
    monkeypatch.setattr("experiments.structure_net.ncm_frozen_body.prompt_ids", lambda t, s: [1])
    rows = [{"id": str(i), "group_id": str(i//2), "source_id": str(i),
             "source_sha256": str(i), "source": "source"} for i in range(12)]
    forbidden = {"group_id": {"0"}, "source_id": {"2"}, "source_sha256": {"4"}}
    selected = unique_sources(rows, forbidden, 4, None)
    assert len({r["group_id"] for r in selected}) == 4
    for r in selected:
        assert all(r[k] not in v for k, v in forbidden.items())
    assert selected == unique_sources(list(reversed(rows)), forbidden, 4, None)


def test_context_repacking_preserves_order_and_routes_gradients_only_to_valid_states():
    h = torch.arange(12.).reshape(2, 3, 2).requires_grad_()
    mask = torch.tensor([[True, False, False], [False, True, False]])
    packed, pads = right_pack(h, mask)
    assert torch.equal(packed[0, :2], h[0, 1:])
    assert torch.equal(packed[1, :2], h[1, [0, 2]])
    assert torch.equal(pads, torch.tensor([[False, False, True]]*2))
    packed.sum().backward()
    assert torch.equal(h.grad, (~mask)[..., None].expand_as(h).float())
    with pytest.raises(ValueError):
        right_pack(h, torch.ones_like(mask))


def test_unpaired_windows_cover_all_source_tokens_and_never_use_saved_targets():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(int(text)))
    windows = unpaired_windows([{"source": "100", "target": "not read"}], Tokenizer())
    recovered = [v for window in windows.tolist() for v in window if v not in (50256, 50257)]
    assert recovered == list(range(100))
    assert windows.shape == (3, 48)
