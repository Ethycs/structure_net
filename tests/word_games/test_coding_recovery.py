import copy
import json

from experiments.word_games.coding_recovery_tasks import SPECS, inputs, reference
from experiments.word_games.coding_recovery import grade, score_recovery
from experiments.word_games.bridges.service import Privacy
from experiments.word_games.structured_proof import Mapping, PHI


def test_hidden_cases_have_expected_edges_and_no_input_mutation():
    assert len(SPECS)==8
    for task,_ in SPECS:
        data=inputs(task); before=copy.deepcopy(data)
        assert len(data)==4
        for d in data: json.dumps(reference(task,d),allow_nan=False)
        assert data==before
    assert reference('c02',inputs('c02')[0]) is None
    assert reference('c02',inputs('c02')[1])=='loan:bronze'
    assert reference('c07',inputs('c07')[2])==['loan:atlas']
    assert len(reference('c05',inputs('c05')[0]))==3


def test_privacy_masks_literals_but_keeps_an_exact_host_inverse():
    original=inputs('c01')[1]; p=Privacy({}); m=Mapping(PHI)
    encoded=m.encode(p.shield(original));wire=json.dumps(encoded)
    assert original['notes'][0] not in wire
    assert all(x['title'] not in wire for x in original['loans'].values())
    assert p.unshield(m.decode(encoded))==original
    assert original['notes'][0] in json.dumps(m.encode(original))


def test_grader_runs_real_code_without_network_and_rejects_wrong_types(tmp_path):
    workspace=tmp_path/'workspace';workspace.mkdir()
    (workspace/'solution.py').write_text('def solve(data):\n    return data["n"] + 1\n')
    good=tmp_path/'good';good.mkdir()
    assert grade(workspace,[{'n':1}],[2],good)['passed']
    interrupted=tmp_path/'interrupted';interrupted.mkdir()
    limited=grade(workspace,[{'n':1}],[2],interrupted,delivery_complete=False)
    assert not limited['passed'] and limited['artifact_tests_passed']==1
    (workspace/'solution.py').write_text('def solve(data):\n    return True\n')
    wrong=tmp_path/'wrong';wrong.mkdir()
    assert not grade(workspace,[{}],[1],wrong)['passed']


def test_recovery_controls_are_independently_scored():
    data=inputs('c01')[1]
    raw={'domain':'museum collection loans','titles':[x['title'] for x in data['loans'].values()],
         'canary':data['notes'][0]}
    assert score_recovery(raw,data)=={'source_domain_recovered':True,'titles_recovered':6,
        'titles_total':6,'canary_recovered':True}
    hidden={'domain':'unknown','titles':[],'canary':''}
    assert score_recovery(hidden,data)['titles_recovered']==0
