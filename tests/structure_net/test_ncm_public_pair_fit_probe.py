from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import ncm_public_pair_fit_probe as probe


def test_selection_ignores_input_order_and_metadata_and_rejects_duplicate_identity():
    rows=[{'id':str(i),'irrelevant_score':i} for i in range(64)]
    chosen=[rows[i]['id'] for i in probe.select(rows)]
    other=[{**r,'irrelevant_score':-r['irrelevant_score']} for r in reversed(rows)]
    assert [other[i]['id'] for i in probe.select(other)]==chosen
    with pytest.raises(ValueError):probe.select([*rows,rows[0]])


def test_pilot_contains_longest_selected_source_and_two_distinct_cases():
    rows=[{'id':str(i)} for i in range(32)];lengths={r['id']:i+1 for i,r in enumerate(rows)}
    assert probe.pilot_ids(rows,lengths)==['0','31']
    lengths['0']=100
    assert probe.pilot_ids(rows,lengths)==['0','1']
    with pytest.raises(ValueError):probe.pilot_ids(rows,{})


def test_loader_replays_pair_binding_and_passes_original_features_without_copy():
    values={'id':'a','source_sha256':'s','target_sha256':'t','native_word_ids':[1,2],'oracle_codes':[3]}
    features=torch.ones(2,3);dataset=[{**values,'features':features}]
    row={**values,'catalog_position':0};loader=probe.Loader(dataset)
    assert loader.get(row) is features
    for key in values:
        bad=deepcopy(row);bad[key]='wrong'
        with pytest.raises(ValueError):loader.get(bad)


def test_diagnostic_captures_teacher_codes_but_keeps_source_only_result(monkeypatch):
    logits=torch.zeros(1,641,2);logits[:,:,1]=1
    head=SimpleNamespace(teacher_logits=lambda *a:logits)
    model=SimpleNamespace(concepts=head)
    original=head.teacher_logits
    def underlying(model,cases,loader,tokenizer,shuffle=False):
        model.concepts.teacher_logits()
        return [{'id':'a','codes':[0]*641,'shuffled':shuffle}]
    training=SimpleNamespace(prior=SimpleNamespace(diagnostics=underlying))
    result=probe.diagnostics(training,model,[],None,None,True)
    assert result[0]['codes']==[0]*641 and result[0]['teacher_codes']==[1]*641
    assert head.teacher_logits is original
