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


def test_quality_control_for_a_different_codec_cannot_authorize_training(tmp_path):
    from experiments.structure_net.ncm_paper_campaign import controls
    codec = tmp_path / "codec.pt"
    codec.write_bytes(b"actual codec")
    (tmp_path / "producer.json").write_text(json.dumps({"input_hashes": {str(codec): "wrong sha"}}))
    with pytest.raises(ValueError, match="different codec"):
        controls(codec, tmp_path / "result.json", tmp_path / "oracle.json")


def test_two_update_pilot_is_never_reused_as_full_training(tmp_path, monkeypatch):
    from experiments.structure_net import ncm_paper_campaign as campaign
    from experiments.structure_net.ncm_memory_data import file_hash
    codec = tmp_path / "codec.pt"
    codec.write_bytes(b"codec")
    monkeypatch.setattr(campaign, "verify", lambda p: None)
    (tmp_path / "producer.json").write_text(json.dumps({"stage": "native", "fingerprint": "f"}))
    (tmp_path / "result.json").write_text(json.dumps({"status": "passed", "producer_fingerprint": "f",
        "codec_sha256": file_hash(codec), "stage_results": {"conditional": {"updates": 2}}}))
    with pytest.raises(ValueError, match="incomplete updates"):
        campaign.completed(tmp_path, "native", codec)
