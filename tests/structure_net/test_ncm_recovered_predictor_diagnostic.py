from copy import deepcopy

import pytest
import torch

from experiments.structure_net import ncm_recovered_predictor_diagnostic as diagnostic


def test_source_binding_rejects_wrong_conditioned_source_and_metadata():
    pair = {'split': 'TRAIN_noisy_candidate', 'source_record_id': 'a',
            'source_sha256': 'text', 'source_component': 'component'}
    meta = {'id': 'a', 'source_sha256': 'text', 'source_component': 'component',
            'alignment_sha256': 'align', 'native_prompt_sha256': 'prompt',
            'source_positions': 3, 'record_sha256': 'record'}
    binding = {k: meta[k] for k in ('id', 'source_sha256', 'alignment_sha256', 'native_prompt_sha256')}
    binding['original_body_sha256'] = diagnostic.BODY_SHA
    entry = {'id': 'a', 'header': {'binding': binding, 'shape': [3, 3072]},
             'metadata_record_sha256': 'record'}
    assert diagnostic.source_binding(pair, meta, entry, 3) == binding
    wrong = deepcopy(pair)
    wrong['source_component'] = 'other'
    with pytest.raises(ValueError, match='binding'):
        diagnostic.source_binding(wrong, meta, entry, 3)
    entry['metadata_record_sha256'] = 'other'
    with pytest.raises(ValueError, match='binding'):
        diagnostic.source_binding(pair, meta, entry, 3)


def test_score_weights_levels_equally_and_does_not_claim_task_accuracy():
    gold = torch.tensor([[0, 1, 1]])
    logits = torch.tensor([[[2., 0.], [0., 2.], [2., 0.]]])
    result = diagnostic.score(logits, gold, levels=(1, 2))
    assert result['correct_codes'] == 2 and result['code_positions'] == 3
    assert result['predicted_codes'] == [0, 1, 0]
    assert result['task_accuracy'] is None
    assert result['mean_level_CE'] == sum(r['CE'] for r in result['levels'])/2
    with pytest.raises(ValueError, match='finite'):
        diagnostic.score(logits*float('nan'), gold, levels=(1, 2))


def test_history_leakage_and_argmax_instability_remain_visible():
    original = torch.tensor([[[1., 1.00001]]])
    assert diagnostic.root_comparison(original, original.clone())['argmax_equal']
    # Numerically close is insufficient if the selected root changes.
    changed = diagnostic.root_comparison(original, original.flip(-1))
    assert changed['allclose'] and not changed['argmax_equal']
    assert not diagnostic.root_comparison(original, original+1)['allclose']
