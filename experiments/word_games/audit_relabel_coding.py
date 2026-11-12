"""Independently reconstruct relabeled coding scores and exact target outputs."""
import argparse
import json
from pathlib import Path
import statistics

from .common import canonical, read, sha, write
from .coding_recovery import grade as execute_grader
from .relabel_coding import Relabel, suite, SPECS, target_reference


def audit(root, output):
    output.mkdir(parents=True,exist_ok=False)
    plan=read(root/'plan.json');result=read(root/'result.json');tasks=read(root/'suite.json')
    assert result['status']=='completed_exploratory' and result['completed']==result['requested']==8
    assert sha(root/'plan.json')==result['plan_sha256'] and sha(root/'suite.json')==plan['suite_sha256']
    assert tasks==suite()
    for path,expected in plan['source_sha256'].items():
        assert sha(path)==expected,path
        assert sha(root/'producer'/path)==expected,path
    assert [r['task'] for r in result['rows']]==[task for task,_ in SPECS]
    codec=Relabel();rows=[];decisions={};solution_hashes={}
    for record in result['rows']:
        name=record['task'];task=tasks[name];taskroot=root/name
        assert read(taskroot/'result.json')==record
        row={'task':name};decisions[name]={}
        for arm in ('native','relabeled'):
            armroot=taskroot/arm;grade=read(armroot/'grade.json');handoff=read(armroot/'handoff.json')
            encoded=task['inputs'] if arm=='native' else [codec.encode(x) for x in task['inputs']]
            assert read(armroot/'evaluation_inputs.json')==encoded
            last=[];actor_results=[]
            for path in sorted(armroot.glob('turn_*/actor/events.jsonl')):
                for line in path.read_text().splitlines():
                    event=json.loads(line)
                    if event.get('type')=='step_finish':last.append(event['part']['reason'])
                actor_results.append(read(path.parent/'result.json'))
            finish=last[-1] if last else None
            assert finish==handoff['last_finish_reason']
            if handoff['delivery_complete']:
                assert finish=='stop' and actor_results
                assert not any(actor_results[-1][key] for key in ('errors','returncode','timed_out','controller_request'))
            assert len({s for item in actor_results for s in item['session_ids']})<=1
            assert handoff['delivery_complete']==grade['delivery_complete']==record['arms'][arm]['delivery_complete']
            outcomes=[];artifacts=[];target_outcomes=[]
            if all('error' in item for item in grade['rows']):
                outcomes=[False]*36;artifacts=[False]*36;target_outcomes=[False]*36
            else:
                raw=read(armroot/'evaluation.stdout');assert len(raw)==len(grade['rows'])==36
                for i,(response,g,want) in enumerate(zip(raw,grade['rows'],task['expected'])):
                    observed=response.get('result')
                    decoded=observed if arm=='native' else codec.output(name,observed,True)
                    want_target=want if arm=='native' else target_reference(name,encoded[i])
                    valid='error' not in response and response.get('input_preserved') is True
                    correct=valid and canonical(decoded)==canonical(want)
                    exact_target=valid and canonical(observed)==canonical(want_target)
                    assert g['response']==response and g['expected']==want and g['observed']==decoded
                    assert g['artifact_passed']==correct and g['passed']==(correct and handoff['delivery_complete'])
                    artifacts.append(correct);outcomes.append(correct and handoff['delivery_complete'])
                    target_outcomes.append(exact_target and handoff['delivery_complete'])
            assert sum(outcomes)==grade['tests_passed']==record['arms'][arm]['tests_passed']
            assert all(outcomes)==grade['passed']==record['arms'][arm]['passed']
            assert sum(artifacts)==grade['artifact_tests_passed']==record['arms'][arm]['artifact_tests_passed']
            # Exact target-vocabulary correctness is stronger than merely decoding.
            assert target_outcomes==outcomes,'Target-output vocabulary mismatch: '+name+':'+arm
            source=Path('.pixi/word-games-relabel-coding')/root.name/(name+'-'+arm)/'solution.py'
            saved=output/'generated_solutions'/name/arm/'solution.py'
            saved.parent.mkdir(parents=True,exist_ok=True)
            with saved.open('xb') as handle:handle.write(source.read_bytes())
            solution_hashes[str(saved)]=sha(saved)
            # Rerun the captured program in a fresh .pixi workspace. Raw data
            # directories hold evidence only, not environments or execution cwd.
            workspace=Path('.pixi/word-games-relabel-audit')/output.name/(name+'-'+arm)
            workspace.mkdir(parents=True,exist_ok=False)
            (workspace/'solution.py').write_bytes(saved.read_bytes())
            decode=(lambda x:x) if arm=='native' else lambda x:codec.output(name,x,True)
            replay=execute_grader(workspace,encoded,task['expected'],output/'reexecution'/name/arm,
                                  decode,handoff['delivery_complete'])
            assert replay['tests_passed']==grade['tests_passed'] and replay['artifact_tests_passed']==grade['artifact_tests_passed']
            assert replay['rows']==grade['rows'],'Captured-program replay differs: '+name+':'+arm
            decisions[name][arm]=outcomes
            row[arm]={'tasks_passed':int(all(outcomes)),'tests_passed':sum(outcomes),
                'original_tests_passed':sum(outcomes[:4]),'artifact_tests_passed':sum(artifacts),
                'status':handoff['status'],'last_finish_reason':finish,
                'clarification_requests':len(handoff['clarification_requests']),
                'seconds':record['arms'][arm]['total_seconds_including_controller']}
        rows.append(row)
    metrics={arm:{'tasks_passed':sum(r[arm]['tasks_passed'] for r in rows),'tasks':8,
        'tests_passed':sum(r[arm]['tests_passed'] for r in rows),'tests':288,
        'original_tests_passed':sum(r[arm]['original_tests_passed'] for r in rows),'original_tests':32,
        'clarification_requests':sum(r[arm]['clarification_requests'] for r in rows),
        'median_task_seconds':statistics.median(r[arm]['seconds'] for r in rows)} for arm in ('native','relabeled')}
    assert metrics==result['coding']
    calls={}
    for path in root.glob('c*/**/*_calls.json'):
        for record in read(path):
            assert 'raw_response' in record
            model=record['request']['model'];response=json.loads(record['raw_response'])
            assert model in (plan['controller'],plan['qwen']) and response['model']==model
            expected=plan['controller_id'] if model==plan['controller'] else plan['qwen_id']
            assert record['deployment_id']==expected
            counts=calls.setdefault(model,{'calls':0,'seconds':0.0})
            counts['calls']+=1;counts['seconds']+=record['seconds']
    summary={'study':plan['study'],'status':'independently_audited','coding':metrics,'rows':rows,
        'paired_case_disagreements':sum(a!=b for v in decisions.values() for a,b in zip(v['native'],v['relabeled'])),
        'exact_target_outputs_verified':True,'semantic_equivalence_checks':288,'input_roundtrip_checks':288,
        'program_reexecution_verified':True,'solution_sha256':solution_hashes,
        'model_calls':calls,'result':str(root/'result.json'),'result_sha256':sha(root/'result.json'),
        'seconds':result['seconds'],'artifacts':{
            **{str(p):sha(p) for p in root.rglob('*') if p.is_file()},
            **{str(p):sha(p) for p in output.rglob('*') if p.is_file()}},
        'limitations':['Eight known task families, one relabeling and one run; not general coding accuracy.',
            'Free text and identifier suffixes are intentionally visible; no confidentiality claim.',
            'Compact contracts and disjoint protocol also changed; old-run comparison is not a single-factor estimate.',
            'Original four examples per task plus 32 generated cases; generated tests are new inputs, not new task families.']}
    write(output/'audit.json',summary)
    print(canonical({k:v for k,v in summary.items() if k not in ('artifacts','limitations')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.run,a.output)
