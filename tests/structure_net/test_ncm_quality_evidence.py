import copy
import json
import pytest
from experiments.structure_net.ncm_quality_evidence import source_turns,packet,parse


def test_turn_citations_preserve_original_source_and_speaker_labels():
    source='#Person1#: Who chairs?\n#Person2#: Bob.\n'
    turns=source_turns(source);assert '\n'.join(x['text'] for x in turns)==source
    p=packet({'id':'x','source':source,'references':[]},[{'candidate_id':'opaque','text':'Bob chairs.'}])
    g={'candidate_id':'opaque','faithfulness':4,'coverage':4,'coherence':4,'verdict':'supported',
        'claims':[{'claim':'Bob chairs','verdict':'supported','evidence_ids':['T0','T1']}],
        'critical_omissions':[],'rationale':'The question and answer establish Bob.'}
    r=parse(json.dumps({'evaluations':[g]}),p)['evaluations'][0]
    assert r['claims'][0]['source_quote']=='#Person1#: Who chairs?\n#Person2#: Bob.'
    assert r['claims'][0]['source_quote'] in source
    for bad in ([],['T9'],[1]):
        b=copy.deepcopy(g);b['claims'][0]['evidence_ids']=bad
        with pytest.raises(ValueError):parse(json.dumps({'evaluations':[b]}),p)
