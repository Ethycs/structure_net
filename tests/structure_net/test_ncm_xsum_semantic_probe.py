from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from experiments.structure_net import ncm_xsum_semantic_probe as s
from experiments.neural_architecture_lab import ncm_xsum_semantic_probe_results as results


def score(i,passed=True,valid=True):
    return {'id':str(i),'valid':valid,'passed':passed if valid else None,'text':'A supported summary.','tokens':4,
            'evaluations':[{'faithful':passed,'salient':passed,'readable':passed}] if valid else None}


def test_semantic_pass_accepts_supported_paraphrase_without_reference_matching():
    grade={'faithful':True,'salient':True,'readable':True}
    assert s.q.passes(grade,'A completely different wording.',128)
    assert not s.q.passes(grade,'A completely different wording.',129)
    assert not s.q.passes(grade,' ',1)
    for k in grade:assert not s.q.passes({**grade,k:False},'Text',1)


def test_invalid_judgments_keep_accuracy_unresolved():
    scores=[score(0),score(1,False),score(2,False,False)]
    value=s.metric(scores)
    assert value['semantic_pass_rate'] is None and value['invalid']==1
    assert value['all_case_pass_lower_bound']==1/3 and value['all_case_pass_upper_bound']==2/3
    scores[-1]['passed']=False
    with pytest.raises(ValueError):s.metric(scores)


def test_directional_gate_and_conditional_primary_boundary():
    a=[score(i,i<20) for i in range(128)];b=[score(i,i<13) for i in range(128)]
    assert results.paired(a,b)['directional_gate_passed']
    a[19]=score(19,False);assert not results.paired(a,b)['directional_gate_passed']
    a[19]=score(19,True,False);assert results.paired(a,b)['accuracy_difference'] is None
    assert s.qualifies({'cases':128,'valid':128,'passed':122})
    assert not s.qualifies({'cases':128,'valid':127,'passed':122})
    assert not s.qualifies({'cases':128,'valid':128,'passed':121})


def test_anonymous_single_candidate_packet_and_lossless_source_lines():
    cases=[{'id':'public-id','source':'first line\n\nthird line','texts':{a:'Meaningful summary.' for a in s.ARMS}}]
    packets=s.packets_for(cases)
    for arm in s.ARMS:
        p=packets[arm][0]
        assert p['candidates']==[{'candidate_id':'summary','text':'Meaningful summary.'}]
        assert '\n'.join(r['text'] for r in p['source_lines'])==cases[0]['source']
        assert p==packets['native'][0]


def test_inherited_firsts_cannot_be_changed_or_omitted():
    old={'key':{'row':{'status':'valid','passed':False}}}
    s.preserve(old,{**old,'new':{}})
    with pytest.raises(ValueError):s.preserve(old,{})
    changed=deepcopy(old);changed['key']['row']['passed']=True
    with pytest.raises(ValueError):s.preserve(old,changed)


def source():
    text='Public article.\nComplete text.';digest=hashlib.sha256(text.encode()).hexdigest()
    provenance={'dataset':'EdinburghNLP/xsum','revision':'7d4d486c2f8ef850b1a11aead99b894ff3dd7da9','split':'train'}
    row={'id':'x','source':text,'source_sha256':digest,'source_truncated':False,'source_component':'x','public_provenance':provenance}
    selected={'id':'x','source_sha256':digest,'source_component':'x','source_ids':[3,4]}
    entry={'id':'x','source_sha256':digest,'public_provenance':provenance}
    tokenizer=SimpleNamespace(encode=lambda text,**kw:[3,4],decode=lambda ids:text)
    return row,selected,entry,tokenizer


@pytest.mark.parametrize('fault',['text','truncation','identity','token_ids','provenance'])
def test_exact_source_binding_rejects_silent_changes(fault):
    row,selected,entry,tokenizer=source();s.verify_source(row,selected,entry,tokenizer)
    if fault=='text':row['source']+=' Extra.'
    if fault=='truncation':row['source_truncated']=True
    if fault=='identity':selected['id']='y'
    if fault=='token_ids':selected['source_ids']=[3]
    if fault=='provenance':row['public_provenance']={**row['public_provenance'],'split':'test'}
    with pytest.raises(ValueError):s.verify_source(row,selected,entry,tokenizer)
