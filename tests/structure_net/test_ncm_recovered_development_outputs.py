from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_recovered_development_outputs as dev


def records():
    trained = dict(status='recovered_paraphrase_training_completed', stage='train', updates=6416,
        recovered_pairs=43894, source_groups=38484, reviewed_anchors=2384, config=asdict(dev.study.CONFIG),
        trainable_parameters=105323136, strict_final_reload_passed=True, protected_tensors_unchanged=True,
        fresh_optimizer_masters_match_initial_weights=True, systems_lifecycle_only_not_quality_evidence=False,
        final_test_opened=False, checkpoint_sha256='fixed')
    replay = dict(updates=6416, presentations=205248,
        counts={'recovered':153936, 'anchor':51312, 'noisy':102624},
        distinct_recovered_sources=38484, distinct_recovered_targets=43894, distinct_reviewed_anchors=2384,
        source_visit_histogram={'4':38484}, all_sample_and_corruption_counts_replayed=True)
    audit = dict(status='recovered_full_training_replayed', stage='train', updates=6416,
        actual_saved_protected_tensors_verified=True, source_checkpoint_sha256='fixed', replay=replay)
    return trained, audit


def test_only_complete_audited_four_epoch_training_is_admitted():
    trained, audit = records(); dev.check_training(trained,audit)
    for changed in (dict(trained, updates=2), dict(trained, protected_tensors_unchanged=False),
                    dict(trained, recovered_pairs=43893), dict(trained, final_test_opened=True),
                    dict(trained, systems_lifecycle_only_not_quality_evidence=True)):
        with pytest.raises(ValueError): dev.check_training(changed,audit)
    for changed in (dict(audit, source_checkpoint_sha256='another'), dict(audit, stage='pilot'),
                    dict(audit, actual_saved_protected_tensors_verified=False)):
        with pytest.raises(ValueError): dev.check_training(trained,changed)
    for key, value in (('presentations',205247), ('distinct_recovered_targets',43893),
                       ('source_visit_histogram',{'3':38484}), ('all_sample_and_corruption_counts_replayed',False)):
        changed = deepcopy(audit); changed['replay'][key] = value
        with pytest.raises(ValueError): dev.check_training(trained,changed)
    changed = deepcopy(audit); changed['replay']['counts']['noisy'] -= 1
    with pytest.raises(ValueError): dev.check_training(trained,changed)


def test_dense_first_judgments_cannot_be_skipped(monkeypatch):
    events = []; inherited = {'earlier': 1}; extended = {**inherited,'dense':2}
    ancestor = SimpleNamespace(audit_cache=lambda cache: events.append(('audit',cache)))
    monkeypatch.setattr(dev.hierarchy,'current_history',lambda arm: (
        'training','evaluation','environment',ancestor,dict(inherited),[]))
    def completed(root):
        events.append(('completed',root))
        return {'canonical_first_judgments':2}
    def append(cache,root,evaluation):
        events.append(('append',root)); assert cache==inherited and evaluation=='evaluation'
        return dict(extended)
    monkeypatch.setattr(dev.previous,'completed',completed)
    monkeypatch.setattr(dev.previous.history.parent.original,'append_grade',append)
    result = dev.closed_history()
    assert result[3]==extended
    assert events==[('completed',dev.hierarchy.ROOTS['dense']['audit']),
                    ('append',dev.hierarchy.ROOTS['dense']['grade']),('audit',extended)]
    monkeypatch.setattr(dev.previous,'completed',lambda root:{'canonical_first_judgments':1})
    with pytest.raises(ValueError,match='complete dense'):dev.closed_history()


def test_paired_rendering_preserves_source_identity_and_rejects_code_changes():
    parent = dict(id='source',source_sha256='source-hash',split='short_development',
        feature_sha256='feature-hash',source_positions=193,codes=[4]*641,raw_word_ids=[1]*128,text='old')
    new = dict(codes=list(parent['codes']),raw_word_ids=[2]*128)
    def render(words,tokenizer):
        assert words==[2]*128 and tokenizer=='tokenizer'
        return {'text':'new','decoded_tokens':126}
    output = dev.merge_readout(parent,new,'tokenizer',render)
    assert output['text']=='new' and output['id']==parent['id']
    assert output['feature_sha256']==parent['feature_sha256'] and output['source_positions']==193
    assert output['codes']==parent['codes'] and output['raw_word_ids']==new['raw_word_ids']
    assert parent['text']=='old' and parent['raw_word_ids']==[1]*128
    with pytest.raises(ValueError):dev.merge_readout(parent,dict(new,codes=[5]*641),'tokenizer',render)
    with pytest.raises(ValueError):dev.merge_readout(parent,dict(new,id='different'),'tokenizer',render)
