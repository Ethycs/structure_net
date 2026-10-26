from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from experiments.neural_architecture_lab import audit_ncm_public_training as audit


def fixture():
    config = replace(audit.curriculum.CONFIG, public_cases=11, anchor_cases=3, epochs=2,
                     effective_examples=8, microbatch=4, warmup=1)
    def labels(example):
        return example.population+str(example.index), [example.index]*sum(audit.LEVELS)
    rows = []
    for step in range(1, config.updates+1):
        number, local = divmod(step-1, config.updates_per_epoch)
        refs = audit.curriculum.Epoch(number+1, config).update(local)
        microbatches = []
        for offset in range(0, len(refs), config.microbatch):
            selected = refs[offset:offset+config.microbatch]
            gold = torch.tensor([labels(e)[1] for e in selected])
            _, changes = audit.study.engine.histories(gold, selected, audit.LEVELS)
            microbatches.append({'ids': [labels(e)[0] for e in selected],
                'positions': [e.position for e in selected], 'populations': [e.population for e in selected],
                'noisy': [e.noisy for e in selected], 'history_changes': changes,
                'weighted_CE': len(selected)/len(refs)})
        rows.append({'step': step, 'epoch': number+1, 'examples': len(refs),
            'lr': audit.curriculum.rate(step, config), 'scale': 128., 'weighted_CE': 1.,
            'gradient_norm': 1., 'microbatches': microbatches})
    return config, labels, rows


def test_replays_epoch_tail_and_restored_mid_epoch_without_repeating_updates():
    config, labels, rows = fixture()
    value = audit.replay(rows, labels, 0, config.updates, config=config)
    assert value['presentations'] == {'public': 22, 'anchor': 8}
    assert value['noisy_presentations'] == 14
    assert rows[1]['examples'] == 7
    resumed = audit.replay(rows[1:], labels, 1, config.updates, config=config)
    assert sum(resumed['presentations'].values()) == 22
    assert resumed['first_completed_update'] == 1


@pytest.mark.parametrize('field', ['ids', 'positions', 'populations', 'noisy', 'history_changes'])
def test_rejects_sample_or_corruption_drift(field):
    config, labels, rows = fixture()
    values = rows[0]['microbatches'][0][field]
    values[0] = (not values[0]) if field == 'noisy' else 'incorrect' if field in ('ids', 'populations') else values[0]+1
    with pytest.raises(ValueError, match='differs'):
        audit.replay(rows, labels, 0, config.updates, config=config)


@pytest.mark.parametrize('field,value', [('step', 0), ('examples', 128), ('lr', .5),
    ('gradient_norm', float('nan')), ('weighted_CE', float('inf')), ('scale', 0.)])
def test_rejects_wrong_boundary_or_nonfinite_optimizer_evidence(field, value):
    config, labels, rows = fixture()
    rows[0][field] = value
    with pytest.raises(ValueError):
        audit.replay(rows, labels, 0, config.updates, config=config)


def test_prefix_cannot_claim_complete_trace_and_missing_updates_never_pass():
    config, labels, rows = fixture()
    with pytest.raises(ValueError, match='unaccounted'):
        audit.replay(rows, labels, 0, 1, config=config)
    with pytest.raises(ValueError, match='unaccounted'):
        audit.replay([*rows, None], labels, 0, config.updates, config=config)
    prefix = audit.replay(rows, labels, 0, 1, config=config, complete_trace=False)
    assert prefix['updates'] == 1 and not prefix['complete_supplied_trace']
    with pytest.raises(ValueError, match='missing'):
        audit.replay(rows[:1], labels, 0, config.updates, config=config)
    changed = deepcopy(rows)
    changed[1]['scale'] = 64.
    with pytest.raises(ValueError):
        audit.replay(changed, labels, 0, config.updates, config=config)


def test_rejects_missing_microbatch_and_inconsistent_effective_loss():
    config, labels, rows = fixture()
    changed = deepcopy(rows)
    changed[1]['microbatches'].pop()
    with pytest.raises(ValueError):
        audit.replay(changed, labels, 0, config.updates, config=config)
    rows[0]['microbatches'][0]['weighted_CE'] += .1
    with pytest.raises(ValueError, match='contributions'):
        audit.replay(rows, labels, 0, config.updates, config=config)
