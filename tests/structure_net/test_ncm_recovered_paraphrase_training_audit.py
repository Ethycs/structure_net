import copy
from dataclasses import asdict, replace

import pytest
import torch

from experiments.neural_architecture_lab import audit_ncm_recovered_paraphrase_training as audit


def fixture():
    c = replace(audit.study.CONFIG, recovered_sources=7, anchor_cases=3, epochs=4,
                effective_examples=4, microbatch=2, warmup=1)
    pairs = [dict(id=f'p{i}', source_sha256=f's{i//2}', source_component=f'c{i//2}') for i in range(14)]
    groups = audit.study.schedule.balancing.groups_for(pairs, c)
    def resolve(e):
        row = pairs[e.index] if e.population == 'recovered' else {'id': f'a{e.index}', 'source_sha256': f'a{e.index}'}
        return {**row, 'oracle_codes': [e.index]*641, 'native_word_ids': [101]*128, 'source_positions': 100+e.index}
    records = []
    for number in range(1, c.epochs+1):
        epoch = audit.study.schedule.balancing.Epoch(number, groups, c)
        for local in range(c.updates_per_epoch):
            refs = epoch.update(local); micros = []
            for i in range(0, len(refs), c.microbatch):
                selected = refs[i:i+c.microbatch]; rows = [resolve(e) for e in selected]
                _, changes = audit.study.engine.histories(torch.tensor([r['oracle_codes'] for r in rows]), selected, audit.LEVELS)
                micros.append(dict(ids=[r['id'] for r in rows], positions=[e.position for e in selected],
                    populations=[e.population for e in selected], noisy=[e.noisy for e in selected],
                    history_changes=changes, weighted_CE=.25))
            step = (number-1)*c.updates_per_epoch+local+1
            records.append(dict(step=step, epoch=number, examples=len(refs),
                lr=audit.study.schedule.balancing.shared.rate(step, c), gradient_norm=1., scale=128.,
                weighted_CE=sum(m['weighted_CE'] for m in micros), microbatches=micros))
    return c, groups, resolve, records


def test_complete_epoch_rotation_and_partial_updates_replayed():
    c, groups, resolve, records = fixture()
    result = audit.replay(records, groups, resolve, 0, c.updates, c)
    assert result['counts'] == {'recovered': 28, 'anchor': 12, 'noisy': 20}
    assert result['source_visit_histogram'] == {4: 7}
    assert result['distinct_recovered_targets'] == 14
    assert result['target_visit_histogram'] == {2: 14}
    assert result['presentations'] == 40
    assert audit.replay(records[2:4], groups, resolve, 2, 4, c)['updates'] == 2


@pytest.mark.parametrize('tamper', ['denominator', 'pair', 'corruption', 'loss', 'missing', 'extra'])
def test_rejects_wrong_trace_and_unaccounted_work(tamper):
    c, groups, resolve, records = fixture(); records = copy.deepcopy(records)
    if tamper == 'denominator': records[2]['examples'] = 4
    elif tamper == 'pair': records[0]['microbatches'][0]['ids'][0] = 'wrong-target'
    elif tamper == 'corruption': records[0]['microbatches'][0]['history_changes'][0] += 1
    elif tamper == 'loss': records[0]['weighted_CE'] += .01
    elif tamper == 'missing': records.pop()
    else: records.append(records[-1])
    with pytest.raises(ValueError): audit.replay(records, groups, resolve, 0, c.updates, c)


def test_state_hash_matches_module_hash_and_prefix_protection():
    module = torch.nn.Linear(2, 3)
    state = module.state_dict()
    bound = {'word_memory.'+k: v for k, v in state.items()}
    assert audit.state_hash(bound, 'word_memory.') == audit.state_hash(state)
    before = audit.state_hash(bound, 'word_memory.')
    changed = {k: v.clone() for k, v in bound.items()}
    changed['word_memory.bias'][0] += 1
    assert audit.state_hash(changed, 'word_memory.') != before
    with pytest.raises(ValueError): audit.state_hash(bound, 'missing.')


def test_training_fingerprint_binds_seed_optimizer_and_input_identity():
    proof = {'config': asdict(audit.study.CONFIG), 'input_hashes': {'source': 'a'}}
    original = audit.study.task.base.fingerprint(proof)
    for field in ('dropout_seed', 'variant_seed', 'warmup', 'max_lr'):
        changed = copy.deepcopy(proof); changed['config'][field] += 1
        assert audit.study.task.base.fingerprint(changed) != original
    changed = copy.deepcopy(proof); changed['input_hashes']['source'] = 'b'
    assert audit.study.task.base.fingerprint(changed) != original
    with pytest.raises(ValueError): audit.study.task.base.fingerprint({'invalid': float('nan')})
