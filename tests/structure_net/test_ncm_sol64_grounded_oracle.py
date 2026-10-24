"""Matched decoder ablation must change only source access, not codes or weights."""
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import ncm_sol64_grounded_oracle as run


def test_oracle_arms_share_codes_and_decoder_and_clean_context():
    events = []
    latent = torch.zeros(1, 128, 384)
    codes = torch.zeros(1, 641, dtype=torch.long)
    source = torch.zeros(1, 20, 3072)
    mask = torch.ones(1, 20, dtype=torch.bool)

    def lookup(value):
        assert value is codes
        events.append('same-code-latent')
        return latent

    def decode(value):
        assert value is latent
        events.append('same-decoder')
        return 'raw-logits'

    def grounded(features, valid, value):
        assert features is source and valid is mask and value is codes
        events.append('source-on')
        return decode(lookup(value))

    model = SimpleNamespace(training=False, _word_bound=None, concepts=SimpleNamespace(_bound=None),
        codec=SimpleNamespace(vq=SimpleNamespace(idx_to_z_q=lookup), decoder=decode), word_logits=grounded)
    assert run.oracle_logits(model, source, mask, codes, 'source_off') == 'raw-logits'
    assert events == ['same-code-latent', 'same-decoder']
    events.clear()
    assert run.oracle_logits(model, source, mask, codes, 'source_on') == 'raw-logits'
    assert events == ['source-on', 'same-code-latent', 'same-decoder']
    with pytest.raises(ValueError):
        run.oracle_logits(model, source, mask, codes, 'source_shuffle')
    model._word_bound = 'leaked'
    with pytest.raises(ValueError):
        run.oracle_logits(model, source, mask, codes, 'source_on')


def test_same64_source_index_covers_original_teacher_population():
    run.configure()
    rows = run.numeric.cases()
    index = {r['id']: r for r in run.q.read(run.features.metadata.DATA / 'index.json')}
    assert len(index) == 10000 and len(rows) == 64
    for row in rows:
        source = run.q.read(index[row['id']]['path'])
        assert all(source[k] == row[k] for k in ('id', 'source_id', 'group_id', 'source_sha256', 'source_component', 'source'))
    assert run.numeric.text_control.split_guard(rows)['group_id']['overlap'] == 0
