from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.structure_net import ncm_recovered_paired_readout as readout


class Codec(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(1, 1)
        self.calls = 0

    def encode(self, *_):
        raise AssertionError('target encoding is forbidden during prediction')

    def decode(self, codes):
        self.calls += 1
        logits = torch.zeros(1, 128, 30528)
        logits[:, :, int(self.decoder.bias.item())] = 1
        return logits


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.codec = Codec()
        self.concepts = SimpleNamespace(_bound=None)
        self._word_bound = None
        self.word_calls = 0
        self.eval().requires_grad_(False)

    def word_logits(self, features, mask, codes):
        self.word_calls += 1
        logits = torch.zeros(1, 128, 30528)
        logits[:, :, 17] = 1
        return logits


def test_one_source_prediction_shared_by_both_readouts():
    model = Model(); calls = []
    def generate(m, features, mask):
        calls.append((features.clone(), mask.clone()))
        return torch.arange(641).reshape(1, -1)
    parent = readout.predict_parent(model, torch.ones(1, 9, 3), torch.ones(1, 9, dtype=torch.bool), generate)
    saved = deepcopy(model.state_dict())
    new = deepcopy(model.codec.decoder.state_dict()); new['bias'].fill_(29)
    with readout.standalone_decoder(model, new):
        new_output = readout.decode_standalone(model, parent['codes'])
    assert len(calls) == model.word_calls == model.codec.calls == 1
    assert parent['raw_word_ids'] == [17]*128 and new_output['raw_word_ids'] == [29]*128
    assert readout.paired_identity([dict(parent, id='a')], [dict(new_output, id='a')]) == 1
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in saved.items())


def test_decoder_state_restored_on_exception_and_invalid_state_rejected_before_mutation():
    model = Model(); saved = deepcopy(model.state_dict())
    state = deepcopy(model.codec.decoder.state_dict()); state['bias'].fill_(29)
    with pytest.raises(RuntimeError, match='readout failed'):
        with readout.standalone_decoder(model, state):
            raise RuntimeError('readout failed')
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in saved.items())
    for broken in (dict(state, bias=torch.zeros(2)), dict(state, bias=torch.tensor([float('nan')]))):
        with pytest.raises(ValueError):
            with readout.standalone_decoder(model, broken): pass
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in saved.items())


def test_stale_context_and_incomplete_source_cannot_be_used():
    model = Model(); model.concepts._bound = object()
    with pytest.raises(ValueError): readout.decode_standalone(model, [0]*641)
    model.concepts._bound = None
    with pytest.raises(ValueError):
        readout.predict_parent(model, torch.ones(1, 9, 3), torch.zeros(1, 9, dtype=torch.bool), None)
    with pytest.raises(ValueError): readout.decode_standalone(model, [0]*640+[True])
    with pytest.raises(ValueError): readout.decode_standalone(model, [0]*640+[8192])


def test_readout_cannot_modify_codes_and_must_emit_full_finite_logits():
    model = Model()
    def mutate(features, mask, codes):
        codes[0, 0] = 4
        return torch.zeros(1, 128, 30528)
    model.word_logits = mutate
    with pytest.raises(ValueError, match='changed the predicted codes'):
        readout.predict_parent(model, torch.ones(1, 9, 3), torch.ones(1, 9, dtype=torch.bool),
                               lambda *_: torch.zeros(1, 641, dtype=torch.long))
    with pytest.raises(ValueError): readout.words_from_logits(torch.zeros(1, 127, 30528))
    logits = torch.zeros(1, 128, 30528); logits[0, 0, 0] = float('nan')
    with pytest.raises(ValueError): readout.words_from_logits(logits)


def test_paired_population_cannot_drop_reorder_or_replace_codes():
    rows = [dict(id=k, codes=[0]*641, raw_word_ids=[0]*128) for k in ('a', 'b')]
    assert readout.paired_identity(rows, deepcopy(rows)) == 2
    for other in (rows[:1], rows[::-1], [rows[0], rows[0]]):
        with pytest.raises(ValueError): readout.paired_identity(rows, other)
    other = deepcopy(rows); other[1]['codes'][0] = 1
    with pytest.raises(ValueError): readout.paired_identity(rows, other)
