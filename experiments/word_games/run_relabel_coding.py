"""Terra -> Qwen label translation -> Qwen coding, with matched direct control."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

from .bridges.client import Client, InvalidGeneration, ModelUnavailable
from .coding_recovery import grade, model_call
from .controller_coding import interpret, shape, CA, CONTROLLER, CONTROLLER_ID, MAX_CLARIFICATIONS
from .common import canonical, read, sha, write
from .instruction_feedback import NEEDS
from .opencode_harness import run as actor, CONFIG
from .relabel_coding import Relabel, FIELDS, CATEGORIES, PREFIXES, SEED, GENERATED, SPECS, suite

DESIGN=Path('docs/07 - Status Reports/word-games/2026-11-12_relabel-coding-plan.md')
SCHEMA={'anyOf':[
    {'type':'object','properties':{'kind':{'const':'instruction'},'instruction':{'type':'string','minLength':1}},
     'required':['kind','instruction'],'additionalProperties':False},
    {'type':'object','properties':{'kind':{'const':'needs_simplification'},
     'needs':{'type':'array','items':{'type':'string','enum':list(NEEDS)},'minItems':1,'maxItems':3}},
     'required':['kind','needs'],'additionalProperties':False}]}
CONTRACT=('Implement a Python standard-library function solve(data). Do not mutate its input. '
    'Return the requested value, not an action or transaction. The example illustrates the schema; '
    'tests use other values and empty collections. Dates are ISO dates. Schema field names are exact; '
    'dictionary identifiers and record identifiers agree. Free text is real text, not opaque handles.')
ACTOR_PROMPT=('Read the current task.json and implement its goal in solution.py using file tools. '
    'The contract and example_input define the interface. Do not merely describe code. '
    'On a continuation, use the revised goal in the current task.json. Ask the stronger controller '
    'with the typed needs_simplification response if needed. After making and checking the edit, '
    'finish with a short confirmation; do not restate the whole task or program.')


def verdict(value):
    if isinstance(value,dict):
        if (set(value)=={'kind','instruction'} and value['kind']=='instruction'
                and isinstance(value['instruction'],str) and value['instruction'].strip()):return value
        needs=value.get('needs')
        if (set(value)=={'kind','needs'} and value['kind']=='needs_simplification'
                and isinstance(needs,list) and 1<=len(needs)<=3
                and all(isinstance(n,str) and n in NEEDS for n in needs)
                and len(set(needs))==len(needs)):return value
    raise InvalidGeneration('invalid_relabel_verdict')


def translate(client, root, instruction, example):
    return verdict(model_call(client,root,'translator',
        'You translate a coding instruction by changing labels while preserving the exact computation. '
        'The original domain is loans/galleries; the target labels are jobs/pools. Do not invent '
        'scheduling behavior from the new names. Map schema FIELD names using fields. Map category '
        'VALUES only within medium/allowed_media using category_values. Change identifier prefixes '
        'using identifier_prefixes; suffixes stay unchanged, preserving order within a namespace. '
        'Free text, numbers, dates, booleans, nulls and other values stay unchanged. Never replace '
        'ordinary words in free text merely because they match a field or enum. Preserve every '
        'comparison, null case, sort/tie rule and output shape. Return only kind instruction and '
        'instruction, or kind needs_simplification and needs. Do not include both branches. No code '
        'or answers.\nTYPED LABEL TABLE\n'+canonical({'fields':FIELDS,'category_values':CATEGORIES,'identifier_prefixes':PREFIXES})+
        '\nTARGET INPUT SHAPE\n'+canonical(shape(Relabel().encode(example)))+
        '\nSOURCE CONTROLLER INSTRUCTION\n'+instruction,768,SCHEMA))


def last_finish(path):
    reasons=[e['part']['reason'] for e in (json.loads(line) for line in path.read_text().splitlines())
             if e.get('type')=='step_finish']
    return reasons[-1] if reasons else None


def run_arm(arm, root, workspace, key, controller, qwen, task, example, instruction):
    root.mkdir(parents=True);workspace.mkdir(parents=True)
    (workspace/'solution.py').write_text('def solve(data):\n    raise NotImplementedError\n')
    start=time.monotonic(); session=None;requests=[];turns=[];finish=None
    status='clarification_exhausted';error=None
    for index in range(MAX_CLARIFICATIONS+1):
        step=root/f'turn_{index}';step.mkdir();request=None
        try:
            active=instruction
            if arm=='relabeled':
                response=translate(qwen,step,instruction,example)
                if response['kind']=='needs_simplification':
                    request={'kind':'needs_simplification','needs':response['needs'],'stage':'translator'}
                else:active=response['instruction']
            if request is None:
                packet={'goal':active,'contract':CONTRACT,'example_input':example if arm=='native' else Relabel().encode(example)}
                write(step/'packet.json',packet)
                (workspace/'task.json').write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n')
                outcome=actor(workspace,ACTOR_PROMPT,step/'actor',key,session)
                turns.append(outcome);finish=last_finish(step/'actor/events.jsonl')
                names={e.get('error',{}).get('name') for e in outcome['errors']}
                if outcome['timed_out'] or 'ContextOverflowError' in names:
                    status='context_or_timeout_failure';break
                if names=={'InvalidInstructionFeedback'}:
                    status='invalid_actor_feedback';break
                if outcome['errors'] or outcome['returncode']:raise ModelUnavailable('actor_transport_failure')
                if len(outcome['session_ids'])!=1 or (session and outcome['session_ids']!=[session]):
                    raise RuntimeError('actor_session_identity_mismatch')
                session=outcome['session_ids'][0]
                if finish!='stop':status='incomplete_model_turn';break
                if outcome['controller_request']:
                    request={'kind':'needs_simplification','needs':outcome['controller_request']['needs'],'stage':'executor'}
                else:status='delivered';break
            requests.append(request);write(step/'controller_request.json',request)
            if index<MAX_CLARIFICATIONS:
                instruction=interpret(controller,step/'revision',task['goal'],example,instruction,request)
        except InvalidGeneration as exc:
            if str(exc)=='unexpected_model_route':raise
            status='invalid_model_output';error=str(exc);break
    result={'status':status,'error':error,'delivery_complete':status=='delivered',
        'clarification_requests':requests,'actor_invocations':len(turns),'last_finish_reason':finish,
        'session_id':session,'seconds':time.monotonic()-start}
    write(root/'handoff.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--key-env-file',type=Path,required=True)
    args=p.parse_args();os.umask(0o077);out=args.output;out.mkdir(parents=True,exist_ok=False)
    key=next(x.split('=',1)[1].strip().strip('"').strip("'") for x in args.key_env_file.read_text().splitlines() if x.startswith('LITELLM_KEY='))
    controller=Client(lambda:key,model=CONTROLLER,deployment_id=CONTROLLER_ID,ca=CA)
    qwen=Client(lambda:key,ca=CA);codec=Relabel();tasks=suite()
    write(out/'suite.json',tasks)
    prior=Path('data/experiments/word_games/20261111_coding_recovery_v3')
    for name,_ in SPECS:
        old=read(prior/name/'hidden_source_tests.json')
        assert tasks[name]['inputs'][:4]==old['inputs'] and tasks[name]['expected'][:4]==old['expected']
    repo=Path(__file__).resolve().parents[2]
    paths=[Path(__file__),Path('experiments/word_games/relabel_coding.py'),
        Path('experiments/word_games/controller_coding.py'),Path('experiments/word_games/coding_recovery.py'),
        Path('experiments/word_games/coding_recovery_tasks.py'),Path('experiments/word_games/opencode_harness.py'),
        Path('experiments/word_games/opencode_sandbox.py'),Path('experiments/word_games/instruction_feedback.py'),
        Path('experiments/word_games/bridges/client.py'),Path('experiments/word_games/common.py'),
        Path('experiments/word_games/bridges/vendor/source_runtime.py'),CONFIG,DESIGN]
    paths=[path.resolve().relative_to(repo) for path in paths]
    plan={'study':'word-games-isomorphic-label-coding-v1','controller':CONTROLLER,'controller_id':CONTROLLER_ID,
        'qwen':qwen.model,'qwen_id':qwen.deployment_id,'tasks':8,'cases_per_task':4+GENERATED,'seed':SEED,
        'source_sha256':{str(path):sha(path) for path in paths},'suite_sha256':sha(out/'suite.json'),
        'mapping':{'fields':FIELDS,'category_values':CATEGORIES,'identifier_prefixes':PREFIXES},
        'semantic_checks':8*(4+GENERATED),'input_roundtrips':8*(4+GENERATED),
        'privacy_claim':False,'compact_contract_in_both_arms':True,'strict_model_finish_required':True,
        'max_clarifications':MAX_CLARIFICATIONS,'context_tokens':8192,'output_tokens':1024,'steps_per_invocation':12}
    write(out/'plan.json',plan)
    for path in paths:
        dest=out/'producer'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    workroot=Path('.pixi/word-games-relabel-coding')/out.name;workroot.mkdir(parents=True,exist_ok=False)
    rows=[];failure=None;started=time.monotonic()
    try:
        for index,(name,_) in enumerate(SPECS):
            task=tasks[name];root=out/name;root.mkdir();example=task['inputs'][1]
            instruction=interpret(controller,root/'initial_controller',task['goal'],example)
            shared_seconds=read(root/'initial_controller/controller_calls.json')[0]['seconds'];arms={}
            for arm in (('native','relabeled') if index%2==0 else ('relabeled','native')):
                workspace=workroot/(name+'-'+arm)
                result=run_arm(arm,root/arm,workspace,key,controller,qwen,task,example,instruction)
                data=task['inputs'] if arm=='native' else [codec.encode(x) for x in task['inputs']]
                decode=(lambda x:x) if arm=='native' else lambda x:codec.output(name,x,True)
                scored=grade(workspace,data,task['expected'],root/arm,decode,result['delivery_complete'])
                result.update(passed=scored['passed'],tests_passed=scored['tests_passed'],tests=scored['tests'],
                    artifact_tests_passed=scored['artifact_tests_passed'],
                    original_tests_passed=sum(r['passed'] for r in scored['rows'][:4]),
                    total_seconds_including_controller=result['seconds']+shared_seconds)
                arms[arm]=result;print(canonical({'task':name,'arm':arm,**result}),flush=True)
            row={'task':name,'controller_seconds':shared_seconds,'arms':arms};rows.append(row);write(root/'result.json',row)
            write(out/f'progress_{len(rows):02d}.json',{'completed':len(rows),'rows':rows})
    except Exception as exc:failure=type(exc).__name__+':'+str(exc)
    result={'study':plan['study'],'status':'completed_exploratory' if failure is None else 'stopped',
        'error':failure,'completed':len(rows),'requested':8,'seconds':time.monotonic()-started,'rows':rows,
        'plan_sha256':sha(out/'plan.json')}
    if rows:
        result['coding']={arm:{'tasks_passed':sum(r['arms'][arm]['passed'] for r in rows),'tasks':len(rows),
            'tests_passed':sum(r['arms'][arm]['tests_passed'] for r in rows),'tests':(4+GENERATED)*len(rows),
            'original_tests_passed':sum(r['arms'][arm]['original_tests_passed'] for r in rows),'original_tests':4*len(rows),
            'clarification_requests':sum(len(r['arms'][arm]['clarification_requests']) for r in rows),
            'median_task_seconds':statistics.median(r['arms'][arm]['total_seconds_including_controller'] for r in rows)}
            for arm in ('native','relabeled')}
    write(out/'result.json',result);print(canonical({k:v for k,v in result.items() if k!='rows'}),flush=True)
    if failure:raise SystemExit(1)


if __name__=='__main__':main()
