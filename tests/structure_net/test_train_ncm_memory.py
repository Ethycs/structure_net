import pytest
import torch
import json

from experiments.structure_net.train_ncm_memory import codec_loss, rows, word_unigram_f1
from experiments.structure_net.ncm_memory_data import text_hash


def test_text_loss_ignores_padding_but_retains_eos():
    logits = torch.zeros((1, 3, 50258), requires_grad=True)
    targets = torch.tensor([[7, 50256, 50257]])
    loss = codec_loss(logits, targets)
    loss.backward()
    assert logits.grad[0, 0].abs().sum() > 0
    assert logits.grad[0, 1].abs().sum() > 0
    assert logits.grad[0, 2].abs().sum() == 0
    assert torch.isfinite(loss)


def test_word_overlap_penalizes_missing_and_extra_words():
    assert word_unigram_f1("Cats, cats!", "cats cats") == 1
    assert word_unigram_f1("cats cats dogs", "cats cats") == pytest.approx(.8)
    assert word_unigram_f1("budget deadline", "garden plants") == 0
    assert word_unigram_f1("", "hello") == 0


def test_jsonl_keeps_unicode_paragraph_separators_inside_a_record(tmp_path):
    source = "An original turn.\u2028A second paragraph.\u0085More text."
    target = "A saved summary."
    record = {"id": "row", "source": source, "target": target,
              "native_eligible": True, "source_sha256": text_hash(source),
              "target_sha256": text_hash(target)}
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n")
    assert rows(path) == [record]
