"""Paired coding rerun: Terra controller, optional Qwen translation, Qwen actor.

OpenCode still owns the executor's native tool loop. This runner only coordinates
typed model handoffs and bounded clarification turns; it never repairs answers
using test results. The eight old tasks are a known regression set.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import time

from .bridges.client import Client, InvalidGeneration, ModelUnavailable
from .bridges.contract import manual, translate_wire_goal
from .bridges.service import Privacy
from .bridges.vendor.runtime import Mapping, KEYS, ENUMS
from .coding_recovery import grade, model_call
from .coding_recovery_tasks import SPECS, reference
from .common import canonical, digest, read, sha, write
from .instruction_feedback import NEEDS
from .opencode_harness import run as actor, CONFIG
from .structured_proof import PHI

PRIOR = Path('data/experiments/word_games/20261111_coding_recovery_v3')
CA = '/etc/docker/certs.d/10.158.218.193:5000/ca.crt'
CONTROLLER = 'codex_sdk/gpt-5.6-terra'
CONTROLLER_ID = '5d42c0e9-a52d-43b0-9da9-4f9bbea67599'
MAX_CLARIFICATIONS = 2
SCHEMA = {'type': 'object', 'properties': {
    'kind': {'type': 'string', 'enum': ['instruction', 'needs_simplification']},
    'instruction': {'type': 'string'},
    'needs': {'type': 'array', 'maxItems': 3,
              'items': {'type': 'string', 'enum': list(NEEDS)}}},
    'required': ['kind', 'instruction', 'needs'], 'additionalProperties': False}
CONTROLLER_SCHEMA = {'type': 'object', 'properties': {'instruction': {'type': 'string'}},
                     'required': ['instruction'], 'additionalProperties': False}
ACTOR_PROMPT = ('Read the current task.json and implement its goal in solution.py as a Python '
    'standard-library function solve(data). Return the specified value, not a transaction envelope. '
    'The example_input illustrates the schema; hidden inputs will differ. Do not mutate the input. '
    'Use the exact field names and data vocabulary supplied. The manual describes the data; its '
    'transaction-response instructions do not apply to this Python coding task. Replace the stub '
    'using file tools. Do not merely describe the code. If this is a continuation, the current '
    'task.json goal is the controller revision and replaces the previous instruction. If you '
    'need simpler instructions, use the typed needs_simplification pause contract.')


def shape(value):
    """Give the controller types and schema, not specimen values or answers."""
    if isinstance(value, dict):
        if value and all(':' in key for key in value):
            return {'<identifier>': shape(next(iter(value.values())))}
        return {key: shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return {str: 'string', int: 'integer', float: 'number', bool: 'boolean', type(None): 'null'}[type(value)]


def interpret(client, root, goal, example, previous=None, request=None):
    prompt = ('You are the stronger controller upstream of a smaller Qwen translator/executor. '
        'Interpret the coding task and give a concise, complete implementation instruction in '
        'at most 180 words, preferably a few short steps. Keep exact source schema field names, '
        'all predicates, null cases, ordering and tie breakers. Do not output Python code, '
        'concrete answers, or invented data. Do not call tools. Do not remove a requirement to '
        'make the task easier. Return JSON with instruction.\nTASK\n' + goal +
        '\nINPUT SHAPE (types, not concrete values)\n' + canonical(shape(example)))
    if request:
        prompt += ('\nPREVIOUS INSTRUCTION\n' + previous + '\nQWEN REQUEST\n' +
                   canonical(request) + '\nRewrite the full instruction more simply while preserving every requirement.')
    result = model_call(client, root, 'controller', prompt, 768, CONTROLLER_SCHEMA)
    if set(result) != {'instruction'} or not isinstance(result['instruction'], str) or not result['instruction'].strip():
        raise InvalidGeneration('invalid_controller_instruction')
    return result['instruction']


def translate(client, root, instruction, privacy, mapping):
    wire = translate_wire_goal(privacy.goal(instruction), mapping.data)
    result = model_call(client, root, 'translator',
        'Translate this controller instruction into the TARGET domain for a Qwen coding executor. '
        'Preserve every computation, condition, tie breaker, ordering rule, output shape and literal. '
        'The wire names were already mapped; keep them exactly. Preserve Python solve(data), null '
        'and case-insensitive substring behavior. Do not provide code or concrete answers. '
        'Return {"kind":"instruction","instruction":"translated instruction","needs":[]}. '
        'If you cannot interpret the instruction reliably, return kind needs_simplification, an '
        'empty instruction, and one to three distinct needs codes from '+canonical(list(NEEDS))+'. '
        'Do not invent a missing fact.\nTARGET\n'+canonical(mapping.public_manual())+
        '\nCONTROLLER INSTRUCTION\n'+wire, 768, SCHEMA)
    return validate_translation(result)


def validate_translation(result):
    if set(result) != {'kind', 'instruction', 'needs'}:
        raise InvalidGeneration('invalid_translation_fields')
    if result['kind'] == 'instruction' and isinstance(result['instruction'], str) and result['instruction'].strip() and result['needs'] == []:
        return result
    needs = result['needs']
    if (result['kind'] == 'needs_simplification' and result['instruction'] == ''
            and isinstance(needs, list) and 1 <= len(needs) <= 3
            and all(isinstance(n, str) and n in NEEDS for n in needs) and len(set(needs)) == len(needs)):
        return result
    raise InvalidGeneration('invalid_translation_verdict')


def run_arm(arm, root, workspace, key, controller, qwen, goal, example, instruction,
            privacy, mapping, source_manual):
    root.mkdir(parents=True); workspace.mkdir(parents=True)
    (workspace/'solution.py').write_text('def solve(data):\n    raise NotImplementedError\n')
    started = time.monotonic(); session = None; requests = []; turns = []
    status = 'clarification_exhausted'; error = None
    encoded = mapping.encode(privacy.shield(example))
    for turn in range(MAX_CLARIFICATIONS + 1):
        step = root/f'turn_{turn}'; step.mkdir()
        request = None
        try:
            active = instruction
            if arm == 'translated':
                translated = translate(qwen, step, instruction, privacy, mapping)
                if translated['kind'] == 'needs_simplification':
                    request = {'kind': 'needs_simplification', 'needs': translated['needs'], 'stage': 'translator'}
                else:
                    active = translated['instruction']
            if request is None:
                packet = {'goal': active, 'manual': source_manual if arm == 'native' else mapping.public_manual(),
                          'example_input': example if arm == 'native' else encoded}
                write(step/'packet.json', packet)
                (workspace/'task.json').write_text(json.dumps(packet, indent=2, ensure_ascii=False)+'\n')
                outcome = actor(workspace, ACTOR_PROMPT, step/'actor', key, session)
                turns.append(outcome)
                if outcome['errors'] or outcome['timed_out'] or outcome['returncode']:
                    names = {e.get('error', {}).get('name') for e in outcome['errors']}
                    if outcome['timed_out'] or 'ContextOverflowError' in names:
                        status = 'context_or_timeout_failure'; break
                    if names == {'InvalidInstructionFeedback'}:
                        status = 'invalid_actor_feedback'; break
                    raise ModelUnavailable('actor_transport_failure')
                if len(outcome['session_ids']) != 1 or (session and outcome['session_ids'] != [session]):
                    raise RuntimeError('actor_session_identity_mismatch')
                session = outcome['session_ids'][0]
                if outcome['controller_request']:
                    request = {'kind': 'needs_simplification', 'needs': outcome['controller_request']['needs'], 'stage': 'executor'}
                else:
                    status = 'delivered'; break
            requests.append(request)
            write(step/'controller_request.json', request)
            if turn < MAX_CLARIFICATIONS:
                instruction = interpret(controller, step/'revision', goal, example, instruction, request)
        except InvalidGeneration as exc:
            if str(exc) == 'unexpected_model_route':
                raise
            status = 'invalid_model_output'; error = str(exc); break
    result = {'status': status, 'error': error, 'delivery_complete': status == 'delivered',
              'clarification_requests': requests, 'actor_invocations': len(turns),
              'actor_seconds': sum(x['seconds'] for x in turns), 'seconds': time.monotonic()-started,
              'session_id': session}
    write(root/'handoff.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key-env-file', type=Path, required=True)
    args = parser.parse_args(); os.umask(0o077); out = args.output
    out.mkdir(parents=True, exist_ok=False)
    key = next(x.split('=',1)[1].strip().strip('"').strip("'") for x in args.key_env_file.read_text().splitlines() if x.startswith('LITELLM_KEY='))
    controller = Client(lambda:key, model=CONTROLLER, deployment_id=CONTROLLER_ID, ca=CA)
    qwen = Client(lambda:key, ca=CA); mapping = Mapping(PHI)
    source_manual = {'name':'museum collection loans', 'rules':manual({'key_map':{k:k for k in KEYS}, 'enum_map':{k:k for k in ENUMS}})}
    paths = [Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__),
        Path('experiments/word_games/coding_recovery.py'),Path('experiments/word_games/coding_recovery_tasks.py'),
        Path('experiments/word_games/opencode_harness.py'),Path('experiments/word_games/opencode_sandbox.py'),
        Path('experiments/word_games/instruction_feedback.py'),Path('experiments/word_games/bridges/client.py'),
        Path('experiments/word_games/bridges/contract.py'),Path('experiments/word_games/bridges/service.py'),
        Path('experiments/word_games/bridges/vendor/runtime.py'),Path('experiments/word_games/bridges/vendor/source_runtime.py'),
        Path('experiments/word_games/common.py'),PHI,CONFIG]
    # CONFIG is absolute; joining it directly would write back to its source
    # instead of placing a copy beneath the experiment's producer directory.
    repo = Path(__file__).resolve().parents[2]
    paths = [p.resolve().relative_to(repo) for p in paths]
    input_paths = [PRIOR/task/name for task,_ in SPECS for name in ('hidden_source_tests.json','grading_private_aliases.json')]
    plan = {'study':'word-games-strong-controller-coding-v1','classification':'known_regression_set_exploratory',
        'controller':CONTROLLER,'controller_deployment_id':CONTROLLER_ID,'executor':qwen.model,
        'qwen_deployment_id':qwen.deployment_id,'tasks':list(SPECS),'tests_per_task':4,
        'source_sha256':{str(p):sha(p) for p in paths}, 'input_sha256':{str(p):sha(p) for p in input_paths},
        'max_clarifications_per_arm':MAX_CLARIFICATIONS,'initial_plan_shared_between_arms':True,
        'arm_order':'alternating by task','context_tokens':8192,'output_tokens':1024,'steps_per_actor_invocation':12,
        'prior':str(PRIOR),'scope':'Stronger controller instructions and bounded simplification; same legacy mapping, privacy, manual and tests. No information-loss repair or hidden-test feedback.',
        'design_sha256':sha('docs/07 - Status Reports/word-games/2026-11-12_strong-controller-coding-plan.md')}
    write(out/'plan.json', plan)
    for path in paths:
        dest = out/'producer'/path; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(path.read_bytes())
    workroot = Path('.pixi/word-games-controller-coding')/out.name
    workroot.mkdir(parents=True,exist_ok=False)
    rows = []; started = time.monotonic(); failure = None
    try:
        for index, (task, goal) in enumerate(SPECS):
            root = out/task; root.mkdir()
            tests = read(PRIOR/task/'hidden_source_tests.json')
            assert [reference(task,x) for x in tests['inputs']] == tests['expected']
            example = tests['inputs'][1]
            write(root/'hidden_source_tests.json', tests)
            aliases = read(PRIOR/task/'grading_private_aliases.json')
            write(root/'private_aliases.json', aliases)
            instruction = interpret(controller, root/'initial_controller', goal, example)
            initial_seconds = read(root/'initial_controller/controller_calls.json')[0]['seconds']
            arms = {}
            for arm in (('native','translated') if index%2==0 else ('translated','native')):
                privacy = Privacy(copy.deepcopy(aliases)); workspace = workroot/(task+'-'+arm)
                result = run_arm(arm, root/arm, workspace, key, controller, qwen, goal, example,
                                 instruction, privacy, mapping, source_manual)
                encoded = tests['inputs'] if arm=='native' else [mapping.encode(privacy.shield(x)) for x in tests['inputs']]
                decode = (lambda x:x) if arm=='native' else lambda x: privacy.unshield(mapping.decode(x))
                scored = grade(workspace, encoded, tests['expected'], root/arm, decode, result['delivery_complete'])
                write(root/arm/'grading_aliases.json',privacy.aliases)
                result.update(passed=scored['passed'], tests_passed=scored['tests_passed'],
                    artifact_tests_passed=scored['artifact_tests_passed'], tests=4,
                    total_seconds_including_shared_controller=result['seconds']+initial_seconds)
                arms[arm] = result
                print(canonical({'task':task,'arm':arm,**result}),flush=True)
            row = {'task':task,'controller_initial_seconds':initial_seconds,'arms':arms}
            rows.append(row); write(root/'result.json',row)
            write(out/f'progress_{len(rows):02d}.json',{'completed':len(rows),'rows':rows})
    except Exception as exc:
        failure = type(exc).__name__+':'+str(exc)
    summary = {'study':plan['study'],'status':'completed_exploratory' if failure is None else 'stopped',
        'error':failure,'completed':len(rows),'requested':8,'seconds':time.monotonic()-started,
        'plan_sha256':sha(out/'plan.json'),'rows':rows,'controller':CONTROLLER,'qwen':qwen.model}
    if rows:
        summary['coding'] = {arm:{'tasks_passed':sum(r['arms'][arm]['passed'] for r in rows),'tasks':len(rows),
            'tests_passed':sum(r['arms'][arm]['tests_passed'] for r in rows),'tests':4*len(rows),
            'clarification_requests':sum(len(r['arms'][arm]['clarification_requests']) for r in rows),
            'resource_failures':sum(r['arms'][arm]['status']=='context_or_timeout_failure' for r in rows),
            'median_task_seconds':statistics.median(r['arms'][arm]['total_seconds_including_shared_controller'] for r in rows)}
            for arm in ('native','translated')}
    write(out/'result.json',summary)
    print(canonical({k:v for k,v in summary.items() if k!='rows'}),flush=True)
    if failure:raise SystemExit(1)


if __name__ == '__main__':
    main()
