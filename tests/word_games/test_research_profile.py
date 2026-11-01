import json

import pytest

from experiments.word_games.common import sha
from experiments.word_games.research_profile import authorize, translate, validate_mapping, KEYS, build_catalog


def fixture(tmp_path):
    source=tmp_path/'source.txt';source.write_text('Verified value is 42.\nPrivate marker is hidden.\n')
    fact={'id':'F01','topic':'test','path':str(source),'anchor':'Verified value is 42.',
          'source_sha256':sha(source),'subject':'approved subject','property':'value','value':42}
    return {'F01':fact}, {'citations':[{'fact_id':'F01','source_path':str(source),'quote':'Verified value is 42.'}],
                        'unknown':['PRIVATE_MARKER'],'notes':'PRIVATE_MARKER'}


def test_projection_discards_worker_prose(tmp_path):
    catalog,proposal=fixture(tmp_path)
    public,receipts=authorize(proposal,catalog,{'test'})
    assert public[0]['value']==42 and receipts[0]['accepted']
    assert 'PRIVATE_MARKER' not in json.dumps(public)
    assert 'quote' not in public[0] and 'source_path' not in public[0]


@pytest.mark.parametrize('mutation',['scope','identity','quote','source','extra'])
def test_fail_closed(tmp_path,mutation):
    catalog,proposal=fixture(tmp_path)
    topics={'test'}
    if mutation=='scope':topics={'other'}
    if mutation=='identity':proposal['citations'][0]['fact_id']='F99'
    if mutation=='quote':proposal['citations'][0]['quote']='Verified value is 95.'
    if mutation=='source':proposal['citations'][0]['source_path']='.env'
    if mutation=='extra':
        proposal['citations'][0]['value']='PRIVATE_MARKER'
        with pytest.raises(ValueError):authorize(proposal,catalog,topics)
        return
    public,receipts=authorize(proposal,catalog,topics)
    assert public==[] and receipts[0]['accepted'] is False


def test_mapping_and_real_catalog():
    mapping={'name':'archive','key_map':{k:'archive_'+k for k in KEYS}}
    validate_mapping(mapping)
    value={'facts':[{'subject':'Structure Net','value':42}],'revision':1}
    assert translate(translate(value,mapping),mapping,True)==value
    assert len(build_catalog())==36


def test_resume_uses_numeric_latest_checkpoint_and_does_not_skip_failure(tmp_path):
    from experiments.word_games.claude_research_bridge import parent_checkpoint
    from experiments.word_games.common import write
    write(tmp_path/'checkpoints/Q99.json',{'completed':True,'turns':99})
    write(tmp_path/'checkpoints/Q100.json',{'completed':True,'turns':100})
    path,result=parent_checkpoint(tmp_path)
    assert path.name=='Q100.json' and result['turns']==100
    write(tmp_path/'result.json',{'completed':False,'turns':100})
    with pytest.raises(ValueError,match='ambiguous'):
        parent_checkpoint(tmp_path)


def test_absolute_producer_snapshot_stays_inside_artifact_root(tmp_path):
    from pathlib import Path
    from experiments.word_games.claude_research_bridge import frozen_source_path
    relative=Path('experiments/word_games/claude_research_bridge.py')
    expected=tmp_path/'frozen_source'/relative
    assert frozen_source_path(tmp_path,relative)==expected
    assert frozen_source_path(tmp_path,relative.resolve())==expected
    with pytest.raises(ValueError):frozen_source_path(tmp_path,Path('../outside.py'))
