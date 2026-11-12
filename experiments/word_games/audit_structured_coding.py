"""Independent artifact read-back, exact grade recomputation and conservative record."""
import argparse
import copy
from pathlib import Path

from .common import read, write, sha, canonical, digest


def text_information_loss_witness():
    from .coding_recovery_tasks import inputs, reference
    from .bridges.service import Privacy
    from .structured_proof import Mapping, PHI
    left=inputs('c05')[0]; right=copy.deepcopy(left)
    a='urgent unique synthetic note'; b='routine unique synthetic note'
    left['loans']['loan:bronze']['notes']=[a]
    right['loans']['loan:bronze']['notes']=[b]
    first=Privacy({}); mapping=Mapping(PHI)
    view_left=mapping.encode(first.shield(left))
    second_aliases=copy.deepcopy(first.aliases)
    second_aliases[b]=second_aliases.pop(a)
    view_right=mapping.encode(Privacy(second_aliases).shield(right))
    expected_left,expected_right=reference('c05',left),reference('c05',right)
    assert view_left==view_right and expected_left!=expected_right
    return {'kind':'secondary_constructive_information_loss_witness',
        'same_receiver_input':True,'receiver_input_sha256':digest(view_left),
        'left_source_note':a,'right_source_note':b,'left_answer':expected_left,
        'right_answer':expected_right,'alias_explanation':'Two valid private alias dictionaries assign the same opaque handle to different unique notes; all visible equality patterns are identical.',
        'conclusion':'A receiver given only this packet cannot always answer the original substring predicate correctly in both worlds.'}


def audit(proof_root, coding_root, output):
    output.mkdir(parents=True,exist_ok=False)
    proof=read(proof_root/'result.json'); bounds=read(proof_root/'plan.json')
    coding=read(coding_root/'result.json'); design=read(coding_root/'plan.json')
    assert sha(proof_root/'plan.json')==proof['plan_sha256']
    assert sha(coding_root/'plan.json')==coding['plan_sha256']
    assert proof['cases']==bounds['expected_cases']==85176
    assert proof['counts']['committed']+proof['counts']['rejected']==proof['cases']
    assert proof['passed'] and not proof['counterexamples'] and proof['fault_control_detected']
    assert coding['status']=='completed_exploratory' and coding['completed']==coding['requested']==8
    for path, expected in bounds['sources'].items(): assert sha(path)==expected, path
    for path, expected in design['source_sha256'].items():
        assert sha(path)==expected, path
        assert sha(coding_root/'producer'/path)==expected, path
    artifacts={str(proof_root/'result.json'):sha(proof_root/'result.json'),
               str(coding_root/'result.json'):sha(coding_root/'result.json')}
    counts={a:{'tasks_passed':0,'tasks':8,'tests_passed':0,'tests':32} for a in ('native','translated')}
    for row in coding['rows']:
        root=coding_root/row['task']
        assert read(root/'result.json')==row
        expected=read(root/'hidden_source_tests.json')['expected']
        for arm in counts:
            grade=read(root/arm/'grade.json'); results=grade['rows']
            assert len(results)==4
            outcomes=[]
            for got,want in zip(results,expected):
                assert got['expected']==want
                response=got.get('response',{})
                correct=('error' not in got and 'error' not in response and
                         response.get('input_preserved') is True and canonical(got['observed'])==canonical(want)
                         and grade['delivery_complete'])
                assert got['passed']==correct
                outcomes.append(correct)
            assert sum(outcomes)==grade['tests_passed']==row['arms'][arm]['tests_passed']
            assert all(outcomes)==grade['passed']==row['arms'][arm]['passed']
            counts[arm]['tasks_passed']+=all(outcomes);counts[arm]['tests_passed']+=sum(outcomes)
        for path in root.rglob('*'):
            if path.is_file(): artifacts[str(path)]=sha(path)
    assert counts==coding['coding']
    blind=coding['recovery']; renamed=coding['renamed_key_recovery']
    aliases_exact=sum(r['literal_leakage']['oracle_inverse_exact'] for r in coding['rows'])
    witness=text_information_loss_witness();write(output/'text_information_loss_witness.json',witness)
    summary={'study':design['study'],'status':'independently_audited',
        'classification':'bounded_structure_passed_empirical_limits_measured',
        'proof':{'cases':proof['cases'],'counterexamples':0,'committed':proof['counts']['committed'],
                 'rejected':proof['counts']['rejected'],'replays':proof['cases'],
                 'table_checks':proof['table_checks'],'unrestricted_json_bijection':False},
        'coding':counts,'recovery':blind,'renamed_key_recovery':renamed,
        'actor_resource_failures':{a:sum(r['arms'][a]['resource_failure'] for r in coding['rows']) for a in counts},
        'artifact_coding':{a:{'tasks_passed':sum(r['arms'][a]['artifact_tests_passed']==4 for r in coding['rows']),
            'tests_passed':sum(r['arms'][a]['artifact_tests_passed'] for r in coding['rows'])} for a in counts},
        'text_information_loss_witness':witness,
        'literal_leakage':{field:sum(r['literal_leakage'][field] for r in coding['rows']) for field in (
            'raw_codec_canary','receiver_canary','raw_codec_original_titles','receiver_original_titles','titles_total')},
        'oracle_inverse_exact':aliases_exact,'raw_recovery_positive_control':
            blind['raw_source']['canaries_recovered']==8 and blind['raw_source']['titles_recovered']==48,
        'proof_result':str(proof_root/'result.json'),'proof_result_sha256':sha(proof_root/'result.json'),
        'coding_result':str(coding_root/'result.json'),'coding_result_sha256':sha(coding_root/'result.json'),
        'producer_hashes_verified':True,'artifacts':artifacts,
        'limitations':['One model, one source domain, eight related synthetic tasks; repeated controls are not independent samples.',
            'Not SWE-bench; not arbitrary-code support in the deployed HTTP API.',
            'Bounded adapter proof uses the existing authoritative Engine, not an independent target Engine.',
            'No proof of natural-language equivalence, general model accuracy or information secrecy.']}
    write(output/'audit.json',summary)
    print(canonical({k:v for k,v in summary.items() if k not in ('artifacts','limitations')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('proof','coding','output'):parser.add_argument('--'+flag,type=Path,required=True)
    a=parser.parse_args();audit(a.proof,a.coding,a.output)
