import json
from types import SimpleNamespace
from experiments.structure_net.ncm_broad_teacher_catalog import numeric_reasons,protected_reasons,replay_valid


def row():
    return dict(teacher_valid=True,capacity_fit=True,target_text='A complete summary.',
                source_cl100k_tokens=256,source_id='a',group_id='g',source_sha256='h',
                spans=[dict(source_id='a',turn_text_sha256='t')])


def test_unknown_native_tokens_are_not_admitted_as_capacity_fit():
    assert numeric_reasons(row(),[100],{0,100,101},4)==['native_special_or_unknown']


def test_invalid_first_is_preserved_not_replaced_by_an_empty_label():
    value=row();value.update(teacher_valid=False,capacity_fit=False,target_text='')
    reasons=numeric_reasons(value,[],{0,100,101},0)
    assert 'invalid_first_preserved' in reasons and 'stored_target_empty_or_over128' in reasons


def test_unknown_and_overlength_do_not_truncate_into_success():
    assert 'target_not_complete_native128' in numeric_reasons(row(),[7]*127,{0,100,101},100)


def test_shared_informative_turn_blocks_different_source_ids():
    protected=dict(families=set(),groups=set(),hashes=set(),turns={'t'})
    assert protected_reasons(row(),protected,lambda x:x)==['protected_informative_turn']


def test_source_family_protects_all_variants():
    protected=dict(families={'a'},groups=set(),hashes=set(),turns=set())
    value=row();value['source_id']='a_variant'
    assert 'protected_source_family' in protected_reasons(value,protected,lambda x:x.split('_')[0])


def test_replay_uses_producing_parser_and_preserves_fenced_candidate(tmp_path):
    value={'summary':'The year is 2020.','support':[]}
    lines=[{'line_id':'L1','text':'The year is 2020.'}]
    candidate={'target_text':value['summary']}
    content='```json\n'+json.dumps(value)+'\n```'
    for stage in ('draft','review'):
        folder=tmp_path/stage;folder.mkdir();path=folder/'attempt_0.json'
        body={'source_lines':lines}
        if stage=='review':body['untrusted_draft']=value
        raw={'http_status':200,'valid':True,'request':{'model':'teacher','messages':[{}, {'content':json.dumps(body)}]},
             'response':{'model':'teacher','choices':[{'finish_reason':'stop','message':{'content':content}}]}}
        path.write_text(json.dumps(raw))
        candidate[stage]={'status':'valid','value':value,'attempts':[str(path)]}
    calls=[]
    def producing_parser(text,row):
        calls.append(text)
        assert text==content and row is candidate
        return value
    paths=replay_valid(candidate,SimpleNamespace(source_lines=lambda _:lines),producing_parser)
    assert len(paths)==len(calls)==2
    assert candidate['target_text']=='The year is 2020.'
