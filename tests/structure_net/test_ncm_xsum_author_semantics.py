from copy import deepcopy
import io
import json

import pytest

from experiments.structure_net import ncm_xsum_author_semantics as study
from experiments.neural_architecture_lab import ncm_xsum_author_semantics_service as service


class Tokenizer:
    def decode(self,ids,skip_special_tokens=True):
        return ' '.join(map(str,ids))


def corrections():
    old = [{'id':str(i),'source':'complete source '+str(i),
        'raw_words':{a:[10,50257,20,50256,30] for a in study.ARMS},
        'texts':{a:'10' for a in study.ARMS}} for i in range(128)]
    new = deepcopy(old)
    for row in new:
        row['prior_texts'] = dict(row['texts'])
        row['texts'] = {a:'10 20' for a in study.ARMS}
        row['rendering_policy'] = 'stop_at_first_EOS_remove_PAD_strip_outer_whitespace'
    return old,new


def test_exact_author_render_is_only_allowed_change():
    old,new = corrections();before = deepcopy(old)
    assert study.checked_correction(old,new,Tokenizer()) == new
    assert old == before


@pytest.mark.parametrize('change',['source','order','raw_words','text','prior_text','missing_case','extra_field'])
def test_no_target_source_or_selected_case_changes(change):
    old,new = corrections()
    if change == 'source':new[0]['source'] += ' added information'
    elif change == 'order':new[0],new[1] = new[1],new[0]
    elif change == 'raw_words':new[0]['raw_words']['native'][0] = 11
    elif change == 'text':new[0]['texts']['native'] = 'a nicer summary'
    elif change == 'prior_text':new[0]['prior_texts']['native'] = 'different'
    elif change == 'missing_case':new.pop()
    elif change == 'extra_field':new[0]['reference'] = 'target'
    with pytest.raises(ValueError):study.checked_correction(old,new,Tokenizer())


def test_request_only_missing_packets_preserve_negative_and_invalid_firsts(tmp_path,monkeypatch):
    requested = []
    monkeypatch.setattr(study.first,'packet_key',lambda p,role:p['case_id'])
    monkeypatch.setattr(study.first,'audit_row',lambda *args:None)

    def request(packet,role,root):
        requested.append(packet['case_id'])
        assert packet['case_id'] == 'new'
        return {'status':'valid','passed':False}

    def scores(cases,packets,cache,arm,role):
        rows = []
        for case,p in zip(cases,packets):
            row = cache[p['case_id']]['row'];valid = row['status'] == 'valid'
            rows.append({'id':case['id'],'valid':valid,'passed':False if valid else None,
                'text':'summary','tokens':1,'evaluations':[{'faithful':False,'salient':False,'readable':True}]})
        return rows

    monkeypatch.setattr(study.q,'request',request)
    monkeypatch.setattr(study.parent,'score_rows',scores)
    cache = {'negative':{'row':{'status':'valid','passed':False},'origin':'original'},
             'invalid':{'row':{'status':'invalid','passed':None},'origin':'original'}}
    before = deepcopy(cache);cases = [{'id':k} for k in ('negative','invalid','new')]
    packets = [{'case_id':r['id']} for r in cases]
    with (tmp_path/'journal.jsonl').open('x') as journal:
        value = study.grade_arm(tmp_path,cases,packets,cache,'native','secondary',journal)
    assert requested == ['new'] and value['new_requests'] == 1 and value['reused_first_judgments'] == 2
    assert all(cache[k] == v for k,v in before.items())
    assert [json.loads(x)['key'] for x in (tmp_path/'journal.jsonl').read_text().splitlines()] == ['new']
    with (tmp_path/'second-journal.jsonl').open('x') as journal:
        repeated = study.grade_arm(tmp_path,cases,packets,cache,'mse','secondary',journal)
    assert requested == ['new'] and repeated['new_requests'] == 0
    assert (tmp_path/'second-journal.jsonl').read_text() == ''


def test_service_uses_corrected_module_and_fresh_units():
    assert service.command('grade')[-2:] == ['experiments.structure_net.ncm_xsum_author_semantics','grade']
    assert service.locations('grade')[0] == 'structure-net-ncm-xsum-author-grade-v1.service'
    assert service.ENV['CUDA_VISIBLE_DEVICES'] == ''
    with pytest.raises(ValueError):service.locations('prepare')


@pytest.mark.parametrize('stage,status,cases',[
    ('pilot','author_semantics_pilot_passed',4),('grade','author_semantics_grade_completed',128)])
def test_service_requires_full_inherited_count_and_expected_result(stage,status,cases):
    value = {'status':status,'cases_per_arm':cases,'metrics':{a:{} for a in study.ARMS},
             'inherited_first_judgments':5197,'model_training':False,'final_test_opened':False}
    assert service.valid_result(stage,value,5197)
    for field,replacement in [('status','launched'),('cases_per_arm',1),('metrics',{}),
        ('inherited_first_judgments',4691),('model_training',True),('final_test_opened',True)]:
        assert not service.valid_result(stage,{**value,field:replacement},5197)
