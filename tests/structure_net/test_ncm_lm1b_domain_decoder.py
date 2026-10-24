from copy import deepcopy

import pytest
import torch

from experiments.structure_net import ncm_lm1b_domain_decoder as run


def rows():
    return [{"id": str(i), "group_id": str(i), "source_id": str(i), "source_sha256": "s"+str(i),
        "spans": [{"source_id": str(i), "turn_text_sha256": "t"+str(i)}]} for i in range(690)]


def test_split_excludes_complete_source_sessions_and_turns():
    values = rows()
    train, held, proof = run.split(values, [str(i) for i in range(64)])
    assert len(train) == 626 and len(held) == 64 and all(p["overlap"] == 0 for p in proof.values())
    changed = deepcopy(values); changed[100]["spans"][0]["source_id"] = "0"
    with pytest.raises(ValueError, match="overlap"):
        run.split(changed, [str(i) for i in range(64)])
    changed = deepcopy(values); changed[100]["spans"][0]["turn_text_sha256"] = "t0"
    with pytest.raises(ValueError, match="overlap"):
        run.split(changed, [str(i) for i in range(64)])


def test_complete_frame_is_unchanged_and_rng_is_not_consumed():
    frame = torch.tensor([101, 500, 501, 101] + [0] * 124)
    rng = torch.Generator().manual_seed(7); before = rng.get_state()
    result, meta = run.augment(frame, rng)
    assert torch.equal(result, frame) and torch.equal(before, rng.get_state())
    assert meta["length"] == 2 and result.data_ptr() != frame.data_ptr()
    invalid = frame.clone(); invalid[8] = 999
    with pytest.raises(ValueError): run.augment(invalid, rng)


def test_native_replay_precomputation_still_uses_original_augmentation():
    frame = torch.tensor([101] + list(range(1000, 1126)) + [101])
    rng = torch.Generator().manual_seed(run.Config().augmentation_seed)
    value, meta = run._native_augment(frame, rng)
    assert meta["mode"] in ("native", "short")
    unchanged, _ = run.augment(value, rng)
    assert torch.equal(value, unchanged)


def test_domain_producer_corrects_generic_training_metadata(tmp_path, monkeypatch):
    (tmp_path / "producer_source").mkdir()
    monkeypatch.setattr(run, "_producing", lambda *a, **k: {"config": {
        "application_targets_used_for_gradient": False, "training_source": "only author TRAIN"},
        "source_hashes": {}, "hypothesis_id": run.HYPOTHESIS})
    p = run.producing(tmp_path, "test")
    assert p["config"]["application_targets_used_for_gradient"] is True
    assert p["config"]["heldout64_targets_used_for_gradient"] is False
    assert p["config"]["training_source"].startswith("626 task")
