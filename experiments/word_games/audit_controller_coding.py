"""Recompute the stronger-controller coding scores from raw evaluator responses."""
import argparse
import json
import statistics
from pathlib import Path

from .bridges.service import Privacy
from .bridges.vendor.runtime import Mapping
from .common import canonical, read, sha, write
from .controller_coding import PRIOR, SPECS, PHI, CONTROLLER, CONTROLLER_ID


def audit(root, output):
    output.mkdir(parents=True, exist_ok=False)
    plan=read(root/'plan.json'); result=read(root/'result.json')
    assert result['status']=='completed_exploratory' and result['completed']==result['requested']==8
    assert result['plan_sha256']==sha(root/'plan.json')
    assert [r['task'] for r in result['rows']]==[t for t,_ in SPECS]
    for p,expected in plan['source_sha256'].items():
        assert sha(p)==expected,p
        assert sha(root/'producer'/p)==expected,p
    for p,expected in plan['input_sha256'].items():assert sha(p)==expected,p
    mapping=Mapping(PHI); recomputed={arm:{'tasks_passed':0,'tasks':8,'tests_passed':0,'tests':32,
        'clarification_requests':0,'resource_failures':0} for arm in ('native','translated')}
    outcomes=[]; truncated_clean_exit_cases=[]
    strict={arm:{'tasks_passed':0,'tasks':8,'tests_passed':0,'tests':32} for arm in recomputed}
    for row in result['rows']:
        task=row['task']; taskroot=root/task
        assert read(taskroot/'result.json')==row
        tests=read(taskroot/'hidden_source_tests.json')
        assert tests==read(PRIOR/task/'hidden_source_tests.json')
        assert read(taskroot/'private_aliases.json')==read(PRIOR/task/'grading_private_aliases.json')
        entry={'task':task}
        for arm in recomputed:
            armroot=taskroot/arm; scored=read(armroot/'grade.json'); handoff=read(armroot/'handoff.json')
            assert handoff['delivery_complete']==scored['delivery_complete']==row['arms'][arm]['delivery_complete']
            assert handoff['status']==row['arms'][arm]['status']
            actors=[read(p) for p in sorted(armroot.glob('turn_*/actor/result.json'))]
            finishes=[]
            for path in sorted(armroot.glob('turn_*/actor/events.jsonl')):
                finishes.extend(json.loads(line)['part']['reason'] for line in path.read_text().splitlines()
                                if line.strip() and json.loads(line).get('type')=='step_finish')
            last_finish=finishes[-1] if finishes else None
            if last_finish=='length' and handoff['delivery_complete']:
                truncated_clean_exit_cases.append({'task':task,'arm':arm,
                    'counted_task_pass':row['arms'][arm]['passed']})
            if handoff['delivery_complete']:
                assert actors and not any(actors[-1][k] for k in ('errors','returncode','timed_out','controller_request'))
                assert handoff['status']=='delivered'
            sessions={s for actor in actors for s in actor['session_ids']}
            assert len(sessions)<=1
            privacy=Privacy(read(armroot/'grading_aliases.json'))
            assert read(armroot/'evaluation_inputs.json')==(tests['inputs'] if arm=='native' else
                [mapping.encode(privacy.shield(x)) for x in tests['inputs']])
            outcomes_arm=[]; artifacts_arm=[]
            if all('error' in g for g in scored['rows']):
                assert scored['tests_passed']==0
                outcomes_arm=[False]*4;artifacts_arm=[False]*4
            else:
                raw=read(armroot/'evaluation.stdout');assert len(raw)==len(scored['rows'])==4
                for response, grade, expected in zip(raw,scored['rows'],tests['expected']):
                    observed=response.get('result')
                    if arm=='translated':observed=privacy.unshield(mapping.decode(observed))
                    correct=('error' not in response and response.get('input_preserved') is True
                             and canonical(observed)==canonical(expected))
                    delivered=correct and handoff['delivery_complete']
                    assert grade['response']==response and grade['expected']==expected and grade['observed']==observed
                    assert grade['artifact_passed']==correct and grade['passed']==delivered
                    outcomes_arm.append(delivered);artifacts_arm.append(correct)
            assert sum(outcomes_arm)==scored['tests_passed']==row['arms'][arm]['tests_passed']
            assert sum(artifacts_arm)==scored['artifact_tests_passed']==row['arms'][arm]['artifact_tests_passed']
            assert all(outcomes_arm)==scored['passed']==row['arms'][arm]['passed']
            metrics=recomputed[arm]
            metrics['tasks_passed']+=all(outcomes_arm);metrics['tests_passed']+=sum(outcomes_arm)
            metrics['clarification_requests']+=len(handoff['clarification_requests'])
            metrics['resource_failures']+=handoff['status']=='context_or_timeout_failure'
            strict[arm]['tasks_passed']+=all(outcomes_arm) and last_finish=='stop'
            strict[arm]['tests_passed']+=sum(outcomes_arm) if last_finish=='stop' else 0
            entry[arm]={'passed':all(outcomes_arm),'tests_passed':sum(outcomes_arm),
                'artifact_tests_passed':sum(artifacts_arm),'status':handoff['status'],
                'clarifications':len(handoff['clarification_requests']),
                'last_actor_finish_reason':last_finish,
                'seconds':row['arms'][arm]['total_seconds_including_shared_controller']}
        outcomes.append(entry)
    for arm in recomputed:
        recomputed[arm]['median_task_seconds']=statistics.median(x[arm]['seconds'] for x in outcomes)
    assert recomputed==result['coding']
    calls={}
    for path in root.glob('c*/**/*_calls.json'):
        for record in read(path):
            assert 'raw_response' in record, str(path)
            model=record['request']['model']
            expected=CONTROLLER_ID if model==CONTROLLER else plan['qwen_deployment_id']
            assert model in (CONTROLLER,plan['executor']) and record['deployment_id']==expected
            metrics=calls.setdefault(model,{'calls':0,'seconds':0.0})
            metrics['calls']+=1;metrics['seconds']+=record['seconds']
    summary={'study':plan['study'],'status':'independently_audited','coding':recomputed,'rows':outcomes,
        'model_calls':calls,'raw_grades_recomputed':True,'source_and_input_hashes_verified':True,
        'truncated_clean_exit_cases':truncated_clean_exit_cases,
        'strict_completion_coding':strict,
        'completion_caveat':'The existing actor completion rule checks CLI exit and error events but not a terminal length finish. Primary frozen scores are retained; stricter normal-stop scores are reported separately.',
        'prior_coding':read(PRIOR/'result.json')['coding'],'prior_result_sha256':sha(PRIOR/'result.json'),
        'result':str(root/'result.json'),'result_sha256':sha(root/'result.json'),
        'artifacts':{str(p):sha(p) for p in root.rglob('*') if p.is_file()},
        'limitations':['Known eight-task regression set, not general coding accuracy.',
            'Old-pilot comparison is descriptive; this run adds Terra instructions and pause handling.',
            'No confidentiality claim and no production HTTP deployment.',
            'Model identities are gateway aliases and deployment IDs, not independent weight attestations.']}
    write(output/'audit.json',summary)
    print(canonical({k:v for k,v in summary.items() if k not in ('artifacts','limitations')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.run,a.output)
