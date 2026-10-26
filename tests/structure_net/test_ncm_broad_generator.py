from dataclasses import FrozenInstanceError, replace
import pytest
import torch

from experiments.structure_net import ncm_broad_generator as study
from experiments.structure_net import ncm_sequential_levels as mechanism
from test_ncm_sequential_decoder import Model


def test_schedule_fixed_budget_and_immutable_configuration():
    c = study.CONFIG
    assert study.rate(c.warmup) == c.max_lr
    assert study.rate(c.updates) == c.min_lr
    assert study.rate(1) == c.max_lr/c.warmup
    assert study.rate(c.warmup+1) < c.max_lr
    for step in (0, c.updates+1):
        with pytest.raises(ValueError): study.rate(step)
    with pytest.raises(FrozenInstanceError): c.updates = 1
    with pytest.raises(ValueError): study.rate(1, replace(c, warmup=c.updates))


def test_controls_are_output_independent_and_include_longest_source():
    rows = [{'id': str(i).zfill(3), 'native_lexical_tokens': 50+i%5} for i in range(20)]
    index = [{'id': r['id'], 'source_positions': 192+i} for i,r in enumerate(rows)]
    actual = study.select_controls(rows, index)
    assert len(set(actual['diagnostic_ids'])) == 16
    assert actual['longest_source_id'] == '019'
    assert '019' in actual['diagnostic_ids']
    assert study.select_controls(rows, list(reversed(index))) == actual
    with pytest.raises(ValueError): study.select_controls(rows, index[:-1])


def test_target_binding_retains_original_data_and_rejects_nontrain():
    row = {'id':'a', 'split':'TRAIN', 'target_text':'text', 'target_sha256':study.catalog.text_hash('text'),
           'native_word_ids':[101,7,101]+[0]*125}
    value = study.bind_codes(row, [1]*641)
    assert all(value[k] == v for k,v in row.items()) and 'oracle_codes' not in row
    for codes in ([1]*640, [8192]*641, [1.0]*641):
        with pytest.raises(ValueError): study.bind_codes(row, codes)
    with pytest.raises(ValueError): study.bind_codes({**row,'split':'DEV'}, [1]*641)
    with pytest.raises(ValueError): study.bind_codes({**row,'target_text':'changed'}, [1]*641)


def test_generator_training_preserves_entire_word_path():
    model = Model()
    model.concepts.source_memory = torch.nn.Linear(4,4)
    model.concepts.adapters = torch.nn.Linear(4,4)
    model.word_memory = torch.nn.Linear(4,4)
    mechanism.configure(model, True)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    expected = {id(p) for mod in (model.concepts.generator, model.concepts.source_memory, model.concepts.adapters)
                for p in mod.parameters()}
    assert trainable == expected
    assert not model.codec.training and not model.word_memory.training and not model.word_adapters.training
    before = {k:v.detach().clone() for k,v in model.codec.state_dict().items()}
    optimizer = study.offload.CPUAdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model.concepts.generator(torch.ones(1,4)).sum().backward()
    assert all(p.grad is None for p in model.codec.parameters())
    optimizer.step()
    assert all(torch.equal(v, before[k]) for k,v in model.codec.state_dict().items())
