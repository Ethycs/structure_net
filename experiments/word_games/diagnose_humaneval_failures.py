"""Selected-case interventions; frozen benchmark scores are never rewritten."""
import argparse
import ast
import copy
import difflib
import itertools
import json
import os
from pathlib import Path
import random

from .common import canonical, read, sha, write
from .humaneval_labels import grade, interface
from .opencode_harness import run as actor
from .run_humaneval_labels import ACTOR_PROMPT, CONTRACT, last_finish

PRIOR = Path('data/experiments/word_games/20261112_humaneval_labels_v1')
DESIGN = Path('docs/07 - Status Reports/word-games/2026-11-12_humaneval-failure-diagnosis-plan.md')
EXPECTED_PRIOR = '3860f37fefbb9ccee10413b8aa8d9a0b13592c79f54e351a0fc5df00c613b997'


def replace_once(source, before, after):
    if source.count(before) != 1: raise ValueError('Patch anchor not unique: ' + before)
    return source.replace(before, after, 1)


def repair(task, arm, source):
    if task == 32 and arm == 'native':
        return replace_once(source, 'lo, hi = -hi, hi', 'lo, hi = lo * 2, hi * 2')
    if task == 32 and arm == 'relabeled':
        return source + '\n\ndef bridge_operation(parcel_0):\n    return find_zero(parcel_0)\n'
    if task == 113 and arm == 'relabeled':
        return replace_once(source,
            'result.append("the number of odd elements in the string " + str(count) + " of the input.")',
            'result.append("the number of odd elements in the string i of the input.".replace("i", str(count)))')
    if task == 160 and arm == 'native':
        source = replace_once(source, 'expr[j] = expr[j] ** expr[j + 2]',
                              'expr[j - 1] = expr[j - 1] ** expr[j + 1]')
        assert source.count('expr.pop(j + 1)') == 2
        source = source.replace('expr.pop(j + 1)', 'expr.pop(j)')
        source = replace_once(source, 'i = j + 1', 'i = j + 2')
        for op in ('*', '//', '+', '-'):
            source = replace_once(source, f'expr[i] = expr[i] {op} expr[i + 2]',
                                  f'expr[i - 1] = expr[i - 1] {op} expr[i + 1]')
        assert source.count('expr.pop(i + 1)') == 4
        return source.replace('expr.pop(i + 1)', 'expr.pop(i)')
    if task == 160 and arm == 'relabeled':
        before = '''    i = 0
    while i < len(ops):
        if ops[i] == '**':
            vals[i] = vals[i] ** vals[i + 1]
            vals.pop(i + 1)
            ops.pop(i)
        else:
            i += 1'''
        after = '''    i = len(ops) - 1
    while i >= 0:
        if ops[i] == '**':
            vals[i] = vals[i] ** vals[i + 1]
            vals.pop(i + 1)
            ops.pop(i)
        i -= 1'''
        return replace_once(source, before, after)
    raise ValueError('No declared intervention')


def diagnostic_cases(task):
    if task == 32:
        return [{'args': [[-root, 1] if degree == 1 else [-root, 1, -root, 1]],
                 'expected': root, 'mode': 'root'}
                for degree in (1, 3) for root in (-8, -4, -2, -1, 0, 0.5, 1, 2, 4, 8)]
    if task == 113:
        rng = random.Random(731114)
        lists = [[], [''], ['0'], ['13579'], ['1' * 12], ['3', '11111111'],
                 ['24680', '999'], ['1234567']]
        lists += [[''.join(rng.choice('0123456789') for _ in range(rng.randrange(31)))
                   for _ in range(rng.randrange(5))] for _ in range(32)]
        return [{'args': [items], 'expected': [
            'the number of odd elements in the string i of the input.'.replace('i', str(sum(c in '13579' for c in s)))
            for s in items], 'mode': 'exact'} for items in lists]
    if task == 160:
        cases = []
        for n in (1, 2):
            for ops in itertools.product(('+', '-', '*', '//', '**'), repeat=n):
                for vals in itertools.product(range(4), repeat=n+1):
                    expression = str(vals[0]) + ''.join(op + str(v) for op, v in zip(ops, vals[1:]))
                    try:
                        # Only generated integer constants and the five fixed operators.
                        expected = eval(expression, {'__builtins__': {}})
                    except ZeroDivisionError:
                        continue  # Undefined expressions are outside the diagnostic domain.
                    cases.append({'args': [list(ops), list(vals)], 'expected': expected, 'mode': 'exact'})
        return cases
    raise ValueError(task)


def diagnostic_problem(problem, cases):
    result = copy.deepcopy(problem)
    # A short per-case timer exposes infinite loops without losing all later
    # diagnostics. Official suites retain their original three-second CPU cap.
    result['test'] = '''def check(candidate):
    import copy, json, math, signal
    rows=[]
    def timeout(signum, frame): raise TimeoutError('diagnostic_100ms_case_limit')
    previous=signal.signal(signal.SIGALRM,timeout)
    try:
        for index, case in enumerate(CASES):
            row={'index':index,'args':case['args'],'expected':case['expected']}
            try:
                signal.setitimer(signal.ITIMER_REAL,0.1)
                observed=candidate(*copy.deepcopy(case['args']))
                signal.setitimer(signal.ITIMER_REAL,0)
                row['observed']=observed
                if case['mode']=='root':
                    residual=sum(c*observed**power for power,c in enumerate(case['args'][0]))
                    row['residual']=residual
                    row['passed']=math.isfinite(observed) and abs(residual)<1e-5
                else: row['passed']=observed==case['expected']
            except BaseException as exc:
                row.update(passed=False,error=type(exc).__name__,message=str(exc)[:120])
            finally: signal.setitimer(signal.ITIMER_REAL,0)
            rows.append(row)
    finally: signal.signal(signal.SIGALRM,previous)
    print(json.dumps({'cases':len(rows),'passed':sum(r['passed'] for r in rows),
        'first_failures':[r for r in rows if not r['passed']][:5]},allow_nan=False))
    assert all(r['passed'] for r in rows),'targeted_diagnostic_failures'
'''.replace('CASES', repr(cases))
    return result


def corrected_instruction(task, controller, translated):
    if task == 32:
        return replace_once(translated, 'find_zero(xs: list)', 'bridge_operation(parcel_0: list)')
    if task == 113:
        complete = controller.replace('odd_count(lst)', 'bridge_operation(parcel_0)').replace('`lst`', '`parcel_0`')
        if not complete.startswith(translated) or complete == translated:
            raise ValueError('Correction is not an exact missing suffix')
        return complete
    raise ValueError(task)


def evaluate(workspace, problem, arm, output, cases, delivered=True):
    official = grade(workspace, problem, arm, output / 'official', delivered)
    targeted = grade(workspace, diagnostic_problem(problem, cases), arm, output / 'targeted', delivered)
    raw = (targeted.get('response') or {}).get('captured_output', '')
    details = json.loads(raw) if raw else None
    return {'official': official, 'targeted': targeted, 'targeted_details': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key-env-file', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    assert sha(PRIOR / 'result.json') == EXPECTED_PRIOR
    old_plan = read(PRIOR / 'plan.json')
    for path, expected in old_plan['source_sha256'].items():
        assert sha(path) == sha(PRIOR / 'producer' / path) == expected, path
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    workspace_root = Path('.pixi/word-games-humaneval-diagnosis') / output.name
    workspace_root.mkdir(parents=True, exist_ok=False)
    problems = {int(p['task_id'].split('/')[1]): p for p in read(PRIOR / 'suite.json')
                if p['task_id'] in ('HumanEval/32', 'HumanEval/113', 'HumanEval/160')}
    cases = {task: diagnostic_cases(task) for task in problems}
    write(output / 'cases.json', cases)
    source = Path(__file__).resolve().relative_to(Path.cwd())
    producers = {**old_plan['source_sha256'], str(source): sha(source), str(DESIGN): sha(DESIGN)}
    plan = {'study': 'word-games-humaneval-failure-diagnosis-v1', 'classification': 'posthoc_selected_case_diagnosis',
            'prior_result_sha256': EXPECTED_PRIOR, 'source_sha256': producers,
            'cases_sha256': sha(output / 'cases.json'), 'live_tasks': [32,113], 'repeats': 2,
            'no_hidden_feedback_to_actor': True, 'primary_accuracy_unchanged': True}
    write(output / 'plan.json', plan)
    for path in producers:
        dest = output / 'producer' / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(path).read_bytes())
    offline, live = [], []
    for task, problem in problems.items():
        taskroot = PRIOR / f'HumanEval_{task}'
        versions = [('canonical', 'native', problem['prompt'] + problem['canonical_solution'], None)]
        for arm in ('native','relabeled'):
            original = (taskroot / arm / 'solution.py').read_text()
            versions.append((arm + '_original', arm, original, None))
            if task != 113 or arm == 'relabeled':
                versions.append((arm + '_corrected', arm, repair(task, arm, original), original))
        for name, arm, code, before in versions:
            root = output / 'offline' / str(task) / name
            root.mkdir(parents=True)
            workspace = workspace_root / f'offline_{task}_{name}'
            workspace.mkdir()
            (workspace / 'solution.py').write_text(code)
            (root / 'solution.py').write_text(code)
            if before is not None:
                (root / 'correction.diff').write_text(''.join(difflib.unified_diff(
                    before.splitlines(True),code.splitlines(True),fromfile='original',tofile='diagnostic_copy')))
            scored = evaluate(workspace, problem, arm, root, cases[task])
            row = {'task': task, 'variant': name, 'arm': arm, 'solution_sha256': sha(root / 'solution.py'), **scored}
            write(root / 'result.json', row)
            offline.append(row)
            print(canonical({'stage':'offline','task':task,'variant':name,
                'official':scored['official']['passed'],'targeted':scored['targeted']['passed'],
                'details':scored['targeted_details']}),flush=True)
    for row in offline:
        if row['variant'] == 'canonical':
            if not (row['official']['passed'] and row['targeted']['passed']):
                write(output / 'offline_control_failure.json', row)
                raise RuntimeError('Canonical diagnostic control failed; live work not started')
        if row['variant'].endswith('_original'):
            prior_grade = read(PRIOR / f"HumanEval_{row['task']}" / row['arm'] / 'evaluation/grade.json')
            if prior_grade['artifact_passed'] != row['official']['artifact_passed']:
                write(output / 'offline_reproduction_failure.json', row)
                raise RuntimeError('Frozen original no longer reproduces; live work not started')
    key = next(line.split('=',1)[1].strip().strip('"').strip("'") for line in args.key_env_file.read_text().splitlines() if line.startswith('LITELLM_KEY='))
    failure = None
    try:
        for task in (32,113):
            taskroot = PRIOR / f'HumanEval_{task}'
            problem = problems[task]
            controller = read(taskroot / 'initial_controller/controller.json')['instruction']
            original = read(taskroot / 'relabeled/turn_0/translator.json')['instruction']
            corrected = corrected_instruction(task, controller, original)
            write(output / f'instruction_{task}.json', {'original':original,'corrected':corrected})
            scaffold = interface(problem)[2]
            for repeat in range(2):
                for condition in (('original','corrected') if repeat == 0 else ('corrected','original')):
                    root = output / 'live' / str(task) / f'{repeat}_{condition}'
                    root.mkdir(parents=True)
                    workspace = workspace_root / f'live_{task}_{repeat}_{condition}'
                    workspace.mkdir()
                    packet = {'goal':original if condition == 'original' else corrected,
                              'contract':CONTRACT,'scaffold':scaffold}
                    write(root / 'packet.json',packet)
                    (workspace / 'task.json').write_text(json.dumps(packet,indent=2)+'\n')
                    (workspace / 'solution.py').write_text(scaffold)
                    outcome = actor(workspace, ACTOR_PROMPT, root / 'actor', key)
                    finish = last_finish(root / 'actor/events.jsonl')
                    names = {e.get('error',{}).get('name') for e in outcome['errors']}
                    resource = outcome['timed_out'] or 'ContextOverflowError' in names
                    if (outcome['errors'] or outcome['returncode']) and not resource:
                        raise RuntimeError('actor_transport_or_protocol_failure')
                    delivered = not (outcome['errors'] or outcome['returncode'] or outcome['timed_out'] or outcome['controller_request']) and finish == 'stop'
                    (root / 'solution.py').write_bytes((workspace / 'solution.py').read_bytes())
                    scored = evaluate(workspace,problem,'relabeled',root,cases[task],delivered)
                    row = {'task':task,'repeat':repeat,'condition':condition,'finish':finish,
                           'delivery_complete':delivered,'seconds':outcome['seconds'],
                           'solution_sha256':sha(root / 'solution.py'),**scored}
                    write(root / 'result.json',row)
                    live.append(row)
                    write(output / f'live_progress_{len(live):02d}.json',live)
                    print(canonical({'stage':'live','task':task,'repeat':repeat,'condition':condition,
                        'official':scored['official']['passed'],'targeted':scored['targeted']['passed'],
                        'delivered':delivered}),flush=True)
    except Exception as exc:
        failure = type(exc).__name__ + ':' + str(exc)
    result = {'study':plan['study'],'status':'completed_diagnostic' if failure is None else 'stopped',
              'error':failure,'offline':offline,'live':live,'primary_accuracy_unchanged':True,
              'plan_sha256':sha(output / 'plan.json')}
    write(output / 'result.json',result)
    print(canonical({'status':result['status'],'offline_programs':len(offline),'live_programs':len(live)}),flush=True)
    if failure: raise SystemExit(1)


if __name__ == '__main__':
    main()
