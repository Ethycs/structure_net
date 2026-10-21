import copy
import json
import pytest
from experiments.structure_net.ncm_quality import (
    Config,blind_packet,calibration_cases,parse_grade,passes,stress_cases,wilson)


def sample():
    packet={'case_id':'opaque','source':'Lee cancelled the event.','references':['The event is cancelled.'],
        'candidates':[{'candidate_id':'a','text':'The event is cancelled.'}]}
    grade={'candidate_id':'a','faithfulness':4,'coverage':3,'coherence':3,'verdict':'supported',
        'claims':[{'claim':'Event cancelled','verdict':'supported','source_quote':'Lee cancelled the event.'}],
        'critical_omissions':[],'rationale':'The source explicitly states cancellation.'}
    return packet,grade


def test_gate_requires_complete_supported_claims_and_task_quality():
    p,g=sample();assert passes(g,p['candidates'][0]['text'],128)
    assert not passes(g,p['candidates'][0]['text'],129)
    assert not passes(g,'',0)
    for field,value in [('faithfulness',3),('coverage',2),('coherence',2),('claims',[]),('critical_omissions',['outcome']),('verdict','contradicted')]:
        c=copy.deepcopy(g);c[field]=value;assert not passes(c,'summary',12)


def test_parser_binds_every_candidate_and_exact_source_evidence():
    p,g=sample();assert parse_grade('```json\n'+json.dumps({'evaluations':[g]})+'\n```',p)['evaluations']==[g]
    for field,value in [('candidate_id','other'),('coverage',True),('coherence',5)]:
        c=copy.deepcopy(g);c[field]=value
        with pytest.raises(ValueError):parse_grade(json.dumps({'evaluations':[c]}),p)
    for quote in ('The event is cancelled.','', 'Lee cancelled ...'):
        c=copy.deepcopy(g);c['claims'][0]['source_quote']=quote
        with pytest.raises(ValueError):parse_grade(json.dumps({'evaluations':[c]}),p)
    with pytest.raises(ValueError):parse_grade(json.dumps({'evaluations':[g,g]}),p)
    with pytest.raises(ValueError):parse_grade(json.dumps({'evaluations':[]}),p)


def test_blind_packet_hides_models_and_reverses_positions_only():
    case={'id':'test_17','source':'A source.','references':['Reference.']}
    outputs={m:{'test_17':{'text':t}} for m,t in [('ncm','Candidate one.'),('llama','Candidate two.')]}
    a=blind_packet(case,outputs);b=blind_packet(case,outputs,True)
    assert a['candidates']==list(reversed(b['candidates']))
    assert not any(m in json.dumps(a) for m in ('ncm','llama','test_17'))


def test_prospective_case_counts_unique_and_controls_not_forced_choice():
    stress=stress_cases();assert len(stress)==40 and len({x['source'] for x in stress})==40
    cal=calibration_cases();assert len(cal)==24
    assert any(all(c['expected_pass'] for c in x['candidates']) for x in cal)
    assert any(not any(c['expected_pass'] for c in x['candidates']) for x in cal)
    assert len({c['candidate_id'] for x in cal for c in x['candidates']})==48
    with pytest.raises(ValueError):Config(primary_cases=100)


def test_wilson_does_not_equate_perfect_small_sample_with_population_certainty():
    assert wilson(0,200)[1]<.02
    assert wilson(190,200)[0]<.95
    assert wilson(200,200)[0]>.95
    with pytest.raises(ValueError):wilson(1,0)
