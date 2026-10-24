"""Test complete real native frames and rendering boundary without CUDA."""
import pytest
import torch

from experiments.structure_net import ncm_sol64_codec_component as run


def test_all64_original_frames_and_splits_are_preserved():
    run.configure()
    rows = run.cases()
    assert len(rows) == 64
    assert all(len(r['native_word_ids']) == 128 for r in rows)
    assert all(r['native_word_ids'][0] == 101 for r in rows)
    assert run.text_control.split_guard(rows)['source_sha256']['overlap'] == 0
    assert len({r['source_component'] for r in rows}) == 64


def test_full_vocabulary_and_native_BOS_EOS_are_not_repaired():
    tokenizer = run.native.tokenizer()
    logits = torch.zeros(1, 128, 30528)
    logits[0, 0, 101] = 1
    logits[0, 1, 7592] = 1
    logits[0, 2, 101] = 1
    output = run.decode(logits, tokenizer)
    assert output['text'] == 'hello'
    assert output['predicted_EOS_position'] == 2
    assert len(output['raw_word_ids']) == 128
    with pytest.raises(ValueError):
        run.decode(logits[:, :127], tokenizer)
