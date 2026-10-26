import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_public_training as training


def test_milestone_continuation_requires_complete_valid_failed_semantic_grade():
    result = {'DEV_qualified': False, 'task_accuracy': 0., 'metrics': {
        c: {'secondary': {'cases': 64, 'valid': 64}} for c in ('short_development', 'long_diagnostic')}}
    assert training.valid_failed_grade(result)
    for key, value in (('DEV_qualified', True), ('task_accuracy', None)):
        changed = copy.deepcopy(result); changed[key] = value
        assert not training.valid_failed_grade(changed)
    changed = copy.deepcopy(result); changed['primary_admitted'] = True
    assert not training.valid_failed_grade(changed)
    result['metrics']['short_development']['secondary']['valid'] = 63
    assert not training.valid_failed_grade(result)


def test_resume_requires_exact_checkpoint_and_never_repeats_prior_milestone(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'STORAGE', tmp_path)
    path = tmp_path/'step.pt'
    path.write_bytes(b'committed checkpoint')
    record = {'path': str(path), 'sha256': training.data.sha(path),
              'completed_updates': training.CONFIG.updates_per_epoch, 'training_identity': 'fixed'}
    path.with_suffix('.json').write_text(json.dumps(record))
    assert training.resume_record(path, 'fixed', 2) == record
    with pytest.raises(ValueError): training.resume_record(path, 'wrong', 2)
    with pytest.raises(ValueError): training.resume_record(path, 'fixed', 4)
    path.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError): training.resume_record(path, 'fixed', 2)


def test_trace_replay_rejects_wrong_case_and_missing_update(tmp_path):
    config = replace(training.CONFIG, public_cases=11, anchor_cases=3, epochs=2,
                     effective_examples=8, warmup=1)
    sources = SimpleNamespace(public=SimpleNamespace(pairs=SimpleNamespace(
        entries=[{'id': 'p'+str(i)} for i in range(11)])), anchors=[{'id': 'a'+str(i)} for i in range(3)])
    epoch = training.curriculum.Epoch(1, config)
    records = []
    for step in range(2):
        refs = epoch.update(step)
        records.append({'step': step+1, 'epoch': 1, 'examples': len(refs),
            'lr': training.curriculum.rate(step+1, config), 'weighted_CE': 1., 'gradient_norm': 1.,
            'microbatches': [{'ids': [('p' if r.population == 'public' else 'a')+str(r.index) for r in refs],
                             'positions': [r.position for r in refs], 'noisy': [r.noisy for r in refs],
                             'populations': [r.population for r in refs]}]})
    path = tmp_path/'trace.jsonl'
    def write(): path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    write()
    result = training.audit_trace(path, sources, 0, 2, config)
    assert result['presentations'] == {'public': 11, 'anchor': 4}
    with pytest.raises(ValueError): training.audit_trace(path, sources, 0, 3, config)
    records[0]['microbatches'][0]['ids'][0] = 'wrong'; write()
    with pytest.raises(ValueError): training.audit_trace(path, sources, 0, 2, config)
