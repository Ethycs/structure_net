import json

import pytest

from experiments.word_games.relabel_coding import Relabel, suite, FIELDS, CATEGORIES, PREFIXES
from experiments.word_games import run_relabel_coding as run
from experiments.word_games.bridges.client import InvalidGeneration


def test_all_288_input_roundtrips_and_independent_target_oracles_agree():
    data=suite()
    assert len(data)==8 and sum(len(t['inputs']) for t in data.values())==288
    assert all(len(t['inputs'])==len(t['expected'])==36 for t in data.values())


def test_fields_enums_and_free_text_are_separate_namespaces():
    codec=Relabel()
    source={'medium':'metal','restricted':True,'notes':['metal cpu loan:x'],
            'text':'metal','title':'paper','id':'loan:x','allowed_media':['paper']}
    target=codec.encode(source)
    assert target=={'hardware_class':'cpu','requires_permit':True,'messages':['metal cpu loan:x'],
                    'query':'metal','label':'paper','identifier':'job:x','allowed_hardware':['gpu']}
    assert codec.decode(target)==source
    assert codec.output('c03',{'metal':2,'paper':1})=={'cpu':2,'gpu':1}
    assert codec.output('c03',{'cpu':2,'gpu':1},True)=={'metal':2,'paper':1}


def test_identifier_relabeling_preserves_lexical_order_and_rejects_key_collision():
    codec=Relabel()
    for prefix in PREFIXES:
        values=[prefix+s for s in ('z','a','abc','aa','A','01')]
        assert codec.encode(sorted(values))==sorted(codec.encode(values))
    with pytest.raises(ValueError):codec.encode({'loans':{},'jobs':{}})
    assert all(len(t)==len(set(t.values())) for t in (FIELDS,CATEGORIES,PREFIXES))


@pytest.mark.parametrize('bad',[
    {'kind':'instruction','instruction':'do it','needs':['one_step']},
    {'kind':'needs_simplification','instruction':'','needs':['one_step']},
    {'kind':'needs_simplification','needs':[]},
    {'kind':'needs_simplification','needs':['one_step','one_step']},
    {'kind':'instruction','instruction':'   '},
])
def test_disjoint_verdict_rejects_mixed_or_empty_responses(bad):
    with pytest.raises(InvalidGeneration):run.verdict(bad)


def test_both_protocol_branches_are_valid():
    for response in ({'kind':'instruction','instruction':'Count each job once.'},
                     {'kind':'needs_simplification','needs':['one_step']}):
        assert run.verdict(response)==response


def test_clean_cli_exit_with_length_finish_cannot_count_as_completed(tmp_path, monkeypatch):
    def actor(workspace,prompt,out,key,session):
        out.mkdir()
        (out/'events.jsonl').write_text(json.dumps({'type':'step_finish','part':{'reason':'length'}})+'\n')
        return {'errors':[],'returncode':0,'timed_out':False,'session_ids':['same'], 'controller_request':None}
    monkeypatch.setattr(run,'actor',actor)
    result=run.run_arm('native',tmp_path/'run',tmp_path/'workspace','placeholder',object(),object(),
                       {'goal':'Count items.'},{'loans':{}},'Count items.')
    assert result['status']=='incomplete_model_turn' and not result['delivery_complete']
