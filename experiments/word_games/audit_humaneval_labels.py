"""Reconstruct HumanEval grades and replay every saved program independently."""
import argparse
import json
from pathlib import Path
import statistics

from .common import read, sha, write, canonical
from .humaneval_labels import COUNT, interface, grade, wilson


def audit(root, output):
    output.mkdir(parents=True, exist_ok=False)
    plan = read(root / 'plan.json')
    result = read(root / 'result.json')
    problems = read(root / 'suite.json')
    assert result['status'] == 'completed_exploratory'
    assert result['completed'] == result['requested'] == COUNT
    assert result['plan_sha256'] == sha(root / 'plan.json')
    assert plan['suite_sha256'] == sha(root / 'suite.json')
    assert [p['task_id'] for p in problems] == plan['tasks'] == [r['task_id'] for r in result['rows']]
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == sha(root / 'producer' / path) == expected, path
    workroot = Path('.pixi/word-games-humaneval-audit') / output.name
    workroot.mkdir(parents=True, exist_ok=False)
    rows = []
    for problem, original in zip(problems, result['rows']):
        name = problem['task_id'].replace('/', '_')
        taskroot = root / name
        assert read(taskroot / 'result.json') == original
        table, _, _ = interface(problem)
        arms = {}
        for arm in ('native', 'relabeled'):
            armroot = taskroot / arm
            recorded = original['arms'][arm]
            handoff = read(armroot / 'handoff.json')
            score = read(armroot / 'evaluation/grade.json')
            payload = read(armroot / 'evaluation/evaluation_input.json')
            assert payload == {'test': problem['test'],
                'entry_point': problem['entry_point'] if arm == 'native' else table[problem['entry_point']],
                'parameter_map': {} if arm == 'native' else {k: v for k, v in table.items() if k != problem['entry_point']}}
            assert all(recorded[k] == v for k, v in handoff.items())
            calls = sorted(armroot.glob('turn_*/actor/result.json'))
            reasons = []
            for path in calls:
                events = [json.loads(line) for line in (path.parent / 'events.jsonl').read_text().splitlines()]
                reasons.extend(e['part']['reason'] for e in events if e.get('type') == 'step_finish')
            assert recorded['actor_invocations'] == len(calls)
            assert recorded['last_finish_reason'] == (reasons[-1] if reasons else None)
            if recorded['delivery_complete']:
                final_actor = read(calls[-1])
                assert reasons[-1] == 'stop'
                assert not any(final_actor[k] for k in ('returncode', 'timed_out', 'errors', 'controller_request'))
                assert recorded['status'] == 'delivered'
            assert score['delivery_complete'] == recorded['delivery_complete']
            response = read(armroot / 'evaluation/stdout') if score['error'] is None else None
            assert response == score['response']
            artifact = score['error'] is None and response['passed'] is True
            assert artifact == score['artifact_passed'] == recorded['artifact_passed']
            passed = artifact and recorded['delivery_complete']
            assert passed == score['passed'] == recorded['passed']
            assert sha(armroot / 'solution.py') == recorded['solution_sha256']
            workspace = workroot / (name + '-' + arm)
            workspace.mkdir()
            (workspace / 'solution.py').write_bytes((armroot / 'solution.py').read_bytes())
            replay = grade(workspace, problem, arm, output / 'reexecution' / name / arm, recorded['delivery_complete'])
            assert replay == score, (name, arm, 'Independent code replay differed')
            seconds = handoff['seconds'] + original['controller_seconds']
            assert seconds == recorded['total_seconds_including_controller']
            arms[arm] = {'passed': passed, 'artifact_passed': artifact,
                'delivery_complete': recorded['delivery_complete'],
                'clarifications': len(recorded['clarification_requests']),
                'seconds': seconds, 'status': recorded['status'], 'grade_error': recorded['grade_error']}
        rows.append({'task_id': problem['task_id'], 'arms': arms})
    metrics = {}
    for arm in ('native', 'relabeled'):
        correct = sum(r['arms'][arm]['passed'] for r in rows)
        metrics[arm] = {'tasks_passed': correct, 'tasks': COUNT, 'requested': COUNT, 'unattempted': 0,
            'observed_pass_rate': correct / COUNT, 'wilson_95': wilson(correct, COUNT),
            'artifact_tasks_passed': sum(r['arms'][arm]['artifact_passed'] for r in rows),
            'normal_completions': sum(r['arms'][arm]['delivery_complete'] for r in rows),
            'clarification_requests': sum(r['arms'][arm]['clarifications'] for r in rows),
            'median_task_seconds': statistics.median(r['arms'][arm]['seconds'] for r in rows)}
    metrics['paired'] = {key: sum(r['arms']['native']['passed'] == a and r['arms']['relabeled']['passed'] == b for r in rows)
        for key, a, b in [('both_pass', True, True), ('direct_only', True, False), ('relabeled_only', False, True), ('both_fail', False, False)]}
    assert metrics == result['coding']
    model_calls = {}
    for path in root.glob('HumanEval_*/**/*_calls.json'):
        for call in read(path):
            model = call['request']['model']
            assert model in (plan['controller'], plan['qwen'])
            response = json.loads(call['raw_response'])
            assert response['model'] == model
            assert call['deployment_id'] == (plan['controller_id'] if model == plan['controller'] else plan['qwen_id'])
            model_calls[model] = model_calls.get(model, 0) + 1
    audited = {'study': plan['study'], 'status': 'independently_audited', 'coding': metrics,
        'rows': rows, 'program_reexecution_verified': True, 'official_assertions_unchanged': True,
        'model_calls': model_calls, 'result': str(root / 'result.json'),
        'result_sha256': sha(root / 'result.json'), 'campaign_seconds': result['seconds'],
        'artifacts': {str(p): sha(p) for directory in (root, output) for p in directory.rglob('*') if p.is_file()}}
    write(output / 'audit.json', audited)
    print(canonical({k: v for k, v in audited.items() if k not in ('artifacts', 'rows')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.output)
