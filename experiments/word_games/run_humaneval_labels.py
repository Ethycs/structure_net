"""Paired public coding: Terra instruction, optional Qwen labels, Qwen actor."""
import argparse
import ast
import json
import os
from pathlib import Path
import statistics
import time

from .bridges.client import Client, InvalidGeneration, ModelUnavailable
from .coding_recovery import model_call
from .common import canonical, read, sha, write
from .controller_coding import CA, CONTROLLER, CONTROLLER_ID, CONTROLLER_SCHEMA, MAX_CLARIFICATIONS
from .humaneval_labels import DATA, COUNT, SEED, Rename, selected, interface, grade, wilson
from .opencode_harness import run as actor, CONFIG
from .run_relabel_coding import SCHEMA, verdict, last_finish

DESIGN = Path('docs/07 - Status Reports/word-games/2026-11-12_humaneval-labels-plan.md')
CONTRACT = ('Implement the specified Python function in solution.py using the standard library. '
    'Preserve the supplied interface and useful helper code. Literal values and return formats '
    'are exact. Hidden tests will use other inputs. Do not read or change anything outside this workspace.')
ACTOR_PROMPT = ('Read task.json and solution.py, then implement the instruction using file tools. '
    'Do not merely describe code. Use the supplied function name and parameter names exactly. '
    'On continuation the current goal replaces the prior instruction. If needed, ask the stronger '
    'controller with the typed needs_simplification response. After inspecting your edit, finish '
    'with a short confirmation rather than restating the program.')


def interpret(client, root, problem, previous=None, request=None):
    prompt = ('You are the stronger controller upstream of a Qwen coding executor. Interpret the '
        'public Python problem and give a concise, complete implementation instruction. Preserve '
        'the interface, all requirements, boundary cases, literals, public examples and return '
        'formats. Provide no implementation or code solution. You may clarify an algorithm in '
        'plain language. Do not use tools. Return JSON with instruction.\nPUBLIC PROBLEM\n' + problem['prompt'])
    if request:
        prompt += '\nPREVIOUS INSTRUCTION\n' + previous + '\nTYPED REQUEST\n' + canonical(request) + '\nRewrite more simply without dropping requirements.'
    result = model_call(client, root, 'controller', prompt, 1024, CONTROLLER_SCHEMA)
    if set(result) != {'instruction'} or not isinstance(result['instruction'], str) or not result['instruction'].strip():
        raise InvalidGeneration('invalid_controller_instruction')
    return result['instruction']


def translate(client, root, instruction, table, scaffold):
    prompt = ('Translate this coding instruction to the target interface labels. Use the exact '
        'identifier table for function and parameter names. Preserve every computation, constraint, '
        'ordering rule, boundary case, numeric value and string literal. Do not rename literal '
        'data simply because it resembles an identifier. Update function calls in public examples. '
        'The label change does not change the task. Return kind instruction and instruction, or '
        'kind needs_simplification and needs, never both branches. Do not output a code solution. '
        '\nIDENTIFIER TABLE\n' + canonical(table) + '\nTARGET SCAFFOLD\n' + scaffold +
        '\nCONTROLLER INSTRUCTION\n' + instruction)
    return verdict(model_call(client, root, 'translator', prompt, 768, SCHEMA))


def run_arm(arm, root, workspace, key, controller, qwen, problem, instruction, initial_error=None):
    root.mkdir(parents=True)
    workspace.mkdir(parents=True)
    table, native, relabeled = interface(problem)
    scaffold = native if arm == 'native' else relabeled
    (workspace / 'solution.py').write_text(scaffold)
    start = time.monotonic()
    session = None
    requests, turns = [], []
    finish = None
    status = 'controller_invalid' if initial_error else 'clarification_exhausted'
    error = initial_error
    for index in range(0 if initial_error else MAX_CLARIFICATIONS + 1):
        step = root / f'turn_{index}'
        step.mkdir()
        request = None
        try:
            active = instruction
            if arm == 'relabeled':
                response = translate(qwen, step, instruction, table, scaffold)
                if response['kind'] == 'needs_simplification':
                    request = {'kind': 'needs_simplification', 'needs': response['needs'], 'stage': 'translator'}
                else:
                    active = response['instruction']
            if request is None:
                packet = {'goal': active, 'contract': CONTRACT, 'scaffold': scaffold}
                write(step / 'packet.json', packet)
                (workspace / 'task.json').write_text(json.dumps(packet, indent=2) + '\n')
                outcome = actor(workspace, ACTOR_PROMPT, step / 'actor', key, session)
                turns.append(outcome)
                finish = last_finish(step / 'actor/events.jsonl')
                names = {e.get('error', {}).get('name') for e in outcome['errors']}
                if outcome['timed_out'] or 'ContextOverflowError' in names:
                    status = 'context_or_timeout_failure'; break
                if names == {'InvalidInstructionFeedback'}:
                    status = 'invalid_actor_feedback'; break
                if outcome['errors'] or outcome['returncode']:
                    raise ModelUnavailable('actor_transport_failure')
                if len(outcome['session_ids']) != 1 or (session and outcome['session_ids'] != [session]):
                    raise RuntimeError('actor_session_identity_mismatch')
                session = outcome['session_ids'][0]
                if finish != 'stop':
                    status = 'incomplete_model_turn'; break
                if outcome['controller_request']:
                    request = {'kind': 'needs_simplification', 'needs': outcome['controller_request']['needs'], 'stage': 'executor'}
                else:
                    status = 'delivered'; break
            requests.append(request)
            write(step / 'controller_request.json', request)
            if index < MAX_CLARIFICATIONS:
                instruction = interpret(controller, step / 'revision', problem, instruction, request)
        except InvalidGeneration as exc:
            if str(exc) == 'unexpected_model_route': raise
            status = 'invalid_model_output'; error = str(exc); break
    result = {'status': status, 'error': error, 'delivery_complete': status == 'delivered',
              'last_finish_reason': finish, 'clarification_requests': requests,
              'actor_invocations': len(turns), 'session_id': session,
              'seconds': time.monotonic() - start}
    write(root / 'handoff.json', result)
    (root / 'solution.py').write_bytes((workspace / 'solution.py').read_bytes())
    result['solution_sha256'] = sha(root / 'solution.py')
    scored = grade(workspace, problem, arm, root / 'evaluation', result['delivery_complete'])
    result.update(passed=scored['passed'], artifact_passed=scored['artifact_passed'],
                  grade_error=scored['error'] or (scored['response'] or {}).get('error'))
    return result


def aggregate(rows):
    metrics = {}
    for arm in ('native', 'relabeled'):
        successes = sum(row['arms'][arm]['passed'] for row in rows)
        metrics[arm] = {'tasks_passed': successes, 'tasks': len(rows), 'requested': COUNT,
            'unattempted': COUNT - len(rows), 'observed_pass_rate': successes / len(rows) if rows else None,
            'wilson_95': wilson(successes, len(rows)),
            'artifact_tasks_passed': sum(row['arms'][arm]['artifact_passed'] for row in rows),
            'normal_completions': sum(row['arms'][arm]['delivery_complete'] for row in rows),
            'clarification_requests': sum(len(row['arms'][arm]['clarification_requests']) for row in rows),
            'median_task_seconds': statistics.median(row['arms'][arm]['total_seconds_including_controller'] for row in rows) if rows else None}
    metrics['paired'] = {key: sum(bool(row['arms']['native']['passed']) == a and bool(row['arms']['relabeled']['passed']) == b for row in rows)
        for key, a, b in [('both_pass', True, True), ('direct_only', True, False), ('relabeled_only', False, True), ('both_fail', False, False)]}
    return metrics


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    workroot = Path('.pixi/word-games-humaneval-preflight') / output.name
    workroot.mkdir(parents=True, exist_ok=False)
    rows = []
    for problem in selected():
        name = problem['task_id'].replace('/', '_')
        table, _, _ = interface(problem)
        source = problem['prompt'] + problem['canonical_solution']
        for arm in ('native', 'relabeled'):
            workspace = workroot / (name + '-' + arm)
            workspace.mkdir()
            code = source if arm == 'native' else ast.unparse(Rename(table).visit(ast.parse(source))) + '\n'
            (workspace / 'solution.py').write_text(code)
            score = grade(workspace, problem, arm, output / name / arm)
            rows.append({'task_id': problem['task_id'], 'arm': arm, 'passed': score['passed']})
    result = {'classification': 'systems_lifecycle_only_not_quality_evidence', 'rows': rows,
              'passed': len(rows) == 2*COUNT and all(row['passed'] for row in rows)}
    write(output / 'result.json', result)
    print(canonical({'passed': result['passed'], 'canonical_checks': len(rows)}))
    if not result['passed']: raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key-env-file', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    os.umask(0o077)
    if args.preflight: return preflight(args.output)
    if not args.key_env_file: parser.error('--key-env-file is required for inference')
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    key = next(x.split('=', 1)[1].strip().strip('"').strip("'") for x in args.key_env_file.read_text().splitlines() if x.startswith('LITELLM_KEY='))
    controller = Client(lambda: key, model=CONTROLLER, deployment_id=CONTROLLER_ID, ca=CA)
    qwen = Client(lambda: key, ca=CA)
    problems = selected()
    write(out / 'suite.json', problems)
    root = Path(__file__).resolve().parents[2]
    paths = sorted(Path('experiments/word_games').rglob('*.py')) + [CONFIG.relative_to(root), DESIGN, Path('pyproject.toml'), Path('pixi.lock')]
    plan = {'study': 'word-games-humaneval-labels-v1', 'classification': 'exploratory',
        'tasks': [p['task_id'] for p in problems], 'sample_seed': SEED,
        'controller': CONTROLLER, 'controller_id': CONTROLLER_ID, 'qwen': qwen.model, 'qwen_id': qwen.deployment_id,
        'data_source': read(DATA / 'source.json'), 'suite_sha256': sha(out / 'suite.json'),
        'source_sha256': {str(path): sha(path) for path in paths},
        'context_tokens': 8192, 'actor_output_tokens': 1024, 'translation_output_tokens': 768,
        'controller_output_tokens': 1024, 'steps': 12, 'max_clarifications': MAX_CLARIFICATIONS,
        'strict_normal_finish': True, 'privacy_claim': False}
    write(out / 'plan.json', plan)
    for path in paths:
        destination = out / 'producer' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    workroot = Path('.pixi/word-games-humaneval') / out.name
    workroot.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows, failure = [], None
    try:
        for index, problem in enumerate(problems):
            name = problem['task_id'].replace('/', '_')
            taskroot = out / name
            taskroot.mkdir()
            instruction, initial_error = None, None
            try:
                instruction = interpret(controller, taskroot / 'initial_controller', problem)
            except InvalidGeneration as exc:
                if str(exc) == 'unexpected_model_route': raise
                initial_error = str(exc)
            shared_seconds = sum(call['seconds'] for call in read(taskroot / 'initial_controller/controller_calls.json'))
            arms = {}
            for arm in (('native', 'relabeled') if index % 2 == 0 else ('relabeled', 'native')):
                result = run_arm(arm, taskroot / arm, workroot / (name+'-'+arm), key,
                    controller, qwen, problem, instruction, initial_error)
                result['total_seconds_including_controller'] = result['seconds'] + shared_seconds
                arms[arm] = result
                print(canonical({'task_id': problem['task_id'], 'arm': arm, **result}), flush=True)
            row = {'task_id': problem['task_id'], 'controller_seconds': shared_seconds, 'arms': arms}
            write(taskroot / 'result.json', row)
            rows.append(row)
            write(out / f'progress_{len(rows):02d}.json', {'completed': len(rows), 'coding': aggregate(rows), 'rows': rows})
    except Exception as exc:
        failure = type(exc).__name__ + ':' + str(exc)
    result = {'study': plan['study'], 'status': 'completed_exploratory' if failure is None else 'stopped',
              'error': failure, 'completed': len(rows), 'requested': COUNT, 'rows': rows,
              'seconds': time.monotonic() - started, 'coding': aggregate(rows), 'plan_sha256': sha(out / 'plan.json')}
    write(out / 'result.json', result)
    print(canonical({k: v for k, v in result.items() if k != 'rows'}), flush=True)
    if failure: raise SystemExit(1)


if __name__ == '__main__':
    main()
