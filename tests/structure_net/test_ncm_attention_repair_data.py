import pytest

from experiments.structure_net import ncm_attention_repair_data as run


class Tokenizer:
    all_special_ids = [0, 101]

    def encode(self, text, add_special_tokens=False):
        return [7, 8]


def test_full_target_binding_rejects_edits_and_clipping():
    row = {'target_text': 'Original', 'target_sha256': run.text_hash('Original'),
        'native_word_ids': [101, 7, 8, 101] + [0]*124}
    assert run.frame(row, Tokenizer()) == row['native_word_ids']
    with pytest.raises(ValueError, match='unaltered complete'):
        run.frame({**row, 'target_text': 'Edited'}, Tokenizer())
    with pytest.raises(ValueError, match='unaltered complete'):
        run.frame({**row, 'native_word_ids': row['native_word_ids'][:-1]}, Tokenizer())
