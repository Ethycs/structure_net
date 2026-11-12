"""Independent raw-grade checks and saved-program replay for the diagnosis."""
import argparse
import json
from pathlib import Path

from .common import canonical, read, sha, write
from .diagnose_humaneval_failures import PRIOR, EXPECTED_PRIOR, diagnostic_cases, evaluate


def audit(run, output):
    output.mkdir(parents=True, exist_ok=False)
    result = read(run / 'result.json'); plan = read(run / 'plan.json')
    assert result['status'] == 'completed_diagnostic' and result['primary_accuracy_unchanged']
    assert sha(PRIOR / 'result.json') == EXPECTED_PRIOR == plan['prior_result_sha256']
    assert sha(run / 'plan.json') == result['plan_sha256']
    assert sha(run / 'cases.json') == plan['cases_sha256']
    cases = read(run / 'cases.json')
    assert cases == {str(n): diagnostic_cases(n) for n in (32,113,160)}
    for path, expected in plan['source_sha256'].items():
        assert sha(path) == sha(run / 'producer' / path) == expected, path
    assert len(result['offline']) == 14 and len(result['live']) == 8
    problems = {int(p['task_id'].split('/')[1]): p for p in read(PRIOR / 'suite.json')}
    workroot = Path('.pixi/word-games-humaneval-diagnosis-audit') / output.name
    workroot.mkdir(parents=True, exist_ok=False)
    summary = {'offline': [], 'live': []}
    termination_differences = []
    for section in ('offline','live'):
        for row in result[section]:
            label = row['variant'] if section == 'offline' else f"{row['repeat']}_{row['condition']}"
            task = row['task']; root = run / section / str(task) / label
            assert read(root / 'result.json') == row
            assert sha(root / 'solution.py') == row['solution_sha256']
            for suite in ('official','targeted'):
                score = read(root / suite / 'grade.json')
                assert score == row[suite]
                raw = read(root / suite / 'stdout') if score['error'] is None else None
                assert raw == score['response']
                artifact = score['error'] is None and raw['passed'] is True
                assert artifact == score['artifact_passed']
                assert score['passed'] == (artifact and score['delivery_complete'])
            assert json.loads(row['targeted']['response']['captured_output']) == row['targeted_details']
            if section == 'live':
                eventlog = [json.loads(x) for x in (root / 'actor/events.jsonl').read_text().splitlines()]
                finishes = [e['part']['reason'] for e in eventlog if e['type'] == 'step_finish']
                assert finishes[-1] == row['finish']
                actor = read(root / 'actor/result.json')
                complete = not any(actor[k] for k in ('errors','returncode','timed_out','controller_request')) and finishes[-1] == 'stop'
                assert complete == row['delivery_complete']
                instructions = read(run / f'instruction_{task}.json')
                assert read(root / 'packet.json')['goal'] == instructions[row['condition']]
            workspace = workroot / f'{section}_{task}_{label}'
            workspace.mkdir()
            (workspace / 'solution.py').write_bytes((root / 'solution.py').read_bytes())
            replay = evaluate(workspace,problems[task],row.get('arm','relabeled'),
                output / section / str(task) / label,cases[str(task)],row.get('delivery_complete',True))
            for suite in ('official','targeted'):
                if replay[suite] != row[suite]:
                    # CPU SIGKILL and outer wall timeout are competing bounds on
                    # the same nonterminating program. Preserve both raw results;
                    # permit no change in correctness or any other score field.
                    before, after = row[suite], replay[suite]
                    resource_errors = {'timeout','exit:137'}
                    assert before['error'] in resource_errors and after['error'] in resource_errors
                    assert not before['passed'] and not after['passed']
                    assert {k:v for k,v in before.items() if k!='error'} == {k:v for k,v in after.items() if k!='error'}
                    termination_differences.append({'section':section,'task':task,'variant':label,
                        'suite':suite,'original_error':before['error'],'replay_error':after['error']})
            assert replay['targeted_details'] == row['targeted_details'], (section,task,label)
            summary[section].append({'task':task,'variant':label,'official_passed':row['official']['passed'],
                'targeted_passed':row['targeted']['passed'],'targeted_cases':row['targeted_details']['cases'],
                'targeted_cases_passed':row['targeted_details']['passed']})
    artifact_paths = [p for directory in (run,output) for p in directory.rglob('*') if p.is_file()]
    record = {'study':plan['study'],'status':'independently_audited','summary':summary,
        'program_reexecution_verified':True,'programs_reexecuted':22,'targeted_inputs':sum(len(v) for v in cases.values()),
        'resource_termination_differences':termination_differences,
        'primary_accuracy_unchanged':True,'result':str(run / 'result.json'),'result_sha256':sha(run / 'result.json'),
        'artifacts':{str(p):sha(p) for p in artifact_paths}}
    write(output / 'audit.json',record)
    print(canonical({k:v for k,v in record.items() if k!='artifacts'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.run,args.output)
