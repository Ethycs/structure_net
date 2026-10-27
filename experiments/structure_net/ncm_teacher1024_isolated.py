"""Finish untouched teacher sources through the bounded, isolated native worker."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import subprocess
import sys
import time

import requests

from experiments.structure_net import ncm_teacher1024_writer_recovery as recovery

prior,original,data,base = recovery.prior,recovery.original,recovery.data,recovery.base
teacher = original.teacher
OLD_REPLAY = original.replay
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-teacher1024-isolated.md')
TEST = Path('tests/structure_net/test_ncm_teacher1024_isolated.py')
SERVICE = Path('experiments/neural_architecture_lab/ncm_teacher1024_isolated_service.py')
ROOTS = {s:data.ROOT/f'20261027_teacher1024_{s}_v5' for s in ('prepare','pilot','pilot_audit','generate','audit')}
EARLIER = recovery.ROOTS['prepare']
FAILED = recovery.ROOTS['pilot']
TERMINAL = data.ROOT/'20261027_teacher1024_supervision_pilot_v4'/'terminal'
SOURCE = prior.old.ROOTS['prepare']
ENDPOINT = data.ROOT/'20261027_teacher_isolated_endpoint_v1'
URL = 'http://127.0.0.1:8091/v1/chat/completions'
NATIVE_MODELS = {'draft':'gpt-5.6-terra','review':'gpt-5.6-sol'}
BATCH_CASES = 16


def configure():
    training,environment = prior.configure()
    if teacher.MODELS != {k:'codex_sdk/'+v for k,v in NATIVE_MODELS.items()}:
        raise ValueError('unchanged qualified Terra draft and Sol review required')
    teacher.request = request;original.replay = replay_stage
    return training,environment


def require(stage):
    value = base.require(ROOTS[stage]);base.verify_proof(data.read(ROOTS[stage]/'producer.json'));return value


def freeze(root,stage,paths,environment):
    if root.exists():raise ValueError('fresh append-only stage required')
    files = {Path(__file__).resolve(),PLAN.resolve(),TEST.resolve(),SERVICE.resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()):files.add(p)
    paths = list(dict.fromkeys([*paths,Path('pixi.lock'),Path('pyproject.toml')]))
    proof = {'schema':original.SCHEMA,'hypothesis_id':original.HYPOTHESIS,'stage':stage,'environment':environment,
        'config':{'original_cases':1024,'preserved_firsts':446,'preserved_exhausted_failures':16,
            'untouched_sources':578,'workers':8,'cases_per_drained_batch':BATCH_CASES,'models':teacher.MODELS,
            'native_response_models':NATIVE_MODELS,'request_URL':URL,'max_attempts_per_stage':3,
            'client_timeout_seconds':180,'idle_wait_limit_seconds':180,'idle_between_batches':True,
            'global_consecutive_schema_stop':3,'no_semantic_or_capacity_rerolls':True,
            'prompts_and_parser_unchanged':True,'student_training_admitted':False,'task_accuracy':None},
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},'input_hashes':{str(p):data.sha(p) for p in paths}}
    return recovery.write_producer(root,proof,sorted(files))


def inventory():
    _,environment = configure();old_state = base.require(EARLIER)
    base.verify_proof(data.read(EARLIER/'producer.json'))
    receipt = data.read(FAILED/'actual_tool_receipt.json');terminal = data.read(TERMINAL/'result.json')
    data.verify(receipt['artifacts']);base.verify_proof(data.read(FAILED/'producer.json'))
    if (receipt['actual_tool_exit_code'] != 1 or terminal['worker_process_exit_code'] != 1
            or terminal['state']['MainPID'] != '0' or not terminal['worker_terminal']
            or terminal['state']['InvocationID'] != 'ac766853aaa64ff69bb5ea03b62c0944'
            or old_state['observed'] != 438 or old_state['invalid_firsts'] != 8):
        raise ValueError('actual drained second failure and complete earlier inventory required')
    rows = data.read(SOURCE/'cases.json');retained = data.read(EARLIER/'first_candidates.json')
    partial = data.read(FAILED/'partial_firsts.json');progress = data.read(FAILED/'progress.json')
    raw_ids = {p.name for p in (FAILED/'requests').iterdir() if p.is_dir()}
    saved_ids = {p.stem for p in (FAILED/'cases').glob('*.json')}
    observed,untouched = prior.partition(rows,retained,partial,raw_ids,saved_ids)
    if (len(observed) != 446 or len(untouched) != 578 or len(partial) != 8
            or progress['completed_firsts'] != 8 or progress['pending_jobs'] != 0
            or len(progress['failures']['fatal']) != 8 or not progress['submissions_stopped']):
        raise ValueError('exact446 observed and578 untouched partition required')
    inputs = [EARLIER/n for n in ('producer.json','result.json','actual_tool_receipt.json','first_candidates.json','unattempted_ids.json')]
    inputs += [FAILED/n for n in ('producer.json','failure.json','actual_tool_receipt.json','partial_firsts.json','progress.json','failures.json')]
    inputs += [TERMINAL/'result.json',TERMINAL/'actual_tool_receipt.json',SOURCE/'cases.json']
    inputs += sorted((FAILED/'requests').rglob('*.json'))+sorted((FAILED/'cases').glob('*.json'))
    root = ROOTS['prepare'];proof = freeze(root,'replay446_firsts_after_second_timeout',inputs,environment)
    encoding,tokenizer = original.previous.data.storage_encoding(),original.previous.native.tokenizer()
    lookup = {r['id']:r for r in rows};paths = [];errors = Counter()
    for value in partial:
        paths += prior.replay_paths(lookup[value['id']],value,FAILED,encoding,tokenizer)
        if data.read(FAILED/'cases'/(value['id']+'.json')) != value or value['teacher_valid']:
            raise ValueError('unaltered second-batch exhausted first required')
        for stage in ('draft','review'):
            obs = value[stage]
            if obs is not None and obs['status']=='invalid':errors.update(data.read(p).get('error_type') for p in obs['attempts'])
    if set(paths) != set((FAILED/'requests').rglob('*.json')) or errors != Counter({'ReadTimeout':24}):
        raise ValueError('all24 original exhausted transport attempts must be preserved')
    invalid = [r['id'] for r in observed if not r['teacher_valid']]
    if len(invalid) != 16:raise ValueError('all16 negatives must remain in the denominator')
    for name,value in (('first_candidates.json',observed),('unattempted_ids.json',untouched),('failed_ids.json',invalid)):
        data.write(root/name,value)
    original.finish(root,proof,{'status':'teacher446_firsts_replayed_578_unattempted','original_cases':1024,
        'observed':446,'unattempted':578,'invalid_firsts':16,'teacher_valid':430,'capacity_fit':430,
        'all_inflight_drained':True,'new_requests':0},[root/n for n in ('first_candidates.json','unattempted_ids.json','failed_ids.json')])


def credential():
    plan = data.read(ENDPOINT/'launch_plan.json');path = Path(plan['private_environment_file'])
    if path.stat().st_mode & 0o777 != 0o600:raise ValueError('private endpoint credential must remain mode0600')
    return dict(line.split('=',1) for line in path.read_text().splitlines())['LLM_ENDPOINT_API_KEY']


def endpoint_inputs():
    for name in ('launch_actual_tool_receipt.json','configuration_actual_tool_receipt.json'):
        receipt = data.read(ENDPOINT/name)
        if receipt['actual_tool_exit_code'] != 0:raise ValueError('actual endpoint launch and configuration receipts required')
        data.verify(receipt['artifacts'])
    plan = data.read(ENDPOINT/'launch_plan.json');data.verify(plan['input_hashes'])
    return [ENDPOINT/n for n in ('launch.json','launch_plan.json','unit_snapshot.service',
        'launch_actual_tool_receipt.json','configuration_verification_v2.json','configuration_actual_tool_receipt.json')]


def endpoint_state():
    launched = data.read(ENDPOINT/'launch.json');plan = data.read(ENDPOINT/'launch_plan.json')
    raw = subprocess.check_output(['systemctl','--user','show',plan['unit'],
        '--property=MainPID,ActiveState,SubState,InvocationID,NRestarts'],text=True)
    state = dict(s.split('=',1) for s in raw.splitlines())
    if (state['InvocationID'] != launched['state']['InvocationID'] or state['MainPID'] != launched['state']['MainPID']
            or state['ActiveState'] != 'active' or state['SubState'] != 'running' or state['NRestarts'] != '0'):
        raise ValueError('exact live isolated endpoint required')
    response = requests.get('http://127.0.0.1:8091/healthz',timeout=10);response.raise_for_status()
    workers = response.json()['workers'];host = workers['codex_host']
    if workers['max_workers'] != 8 or host['max_rss_mb'] != 4096 or host['max_turns'] != 100:
        raise ValueError('qualified endpoint concurrency and memory guards differ')
    return {'state':state,'workers':workers,'health_elapsed_seconds':response.elapsed.total_seconds()}


def idle_snapshot(path):
    started = time.monotonic()
    while True:
        value = endpoint_state()
        if value['workers']['in_flight'] == 0:
            value['wait_seconds'] = time.monotonic()-started;data.write(path,value);return value
        if time.monotonic()-started >= 180:raise RuntimeError('endpoint did not drain; no next batch may start')
        time.sleep(1)


def valid_native_transport(raw,stage):
    response = raw.get('response',{})
    if not isinstance(response,dict):return False
    choices = response.get('choices')
    if not isinstance(choices,list) or not choices or not isinstance(choices[0],dict):return False
    return (raw.get('transport') == URL and raw.get('http_status') == 200
        and response.get('model') == NATIVE_MODELS[stage] and choices[0].get('finish_reason') == 'stop')


def request(row,stage,root,encoding,draft=None):
    body = teacher.payload(row,stage,draft)
    if body['model'] != 'codex_sdk/'+NATIVE_MODELS[stage]:raise ValueError('exact qualified requested model required')
    tokens = len(encoding.encode(json.dumps(body['messages'],ensure_ascii=False),disallowed_special=()))+16
    if tokens > 6300:raise ValueError('whole prompt exceeds6300; clipping forbidden')
    headers = {'Authorization':'Bearer '+credential()};attempts = []
    directory = root/'requests'/row['id']/stage;directory.mkdir(parents=True,exist_ok=False)
    for i in range(3):
        path = directory/f'attempt_{i}.json';raw = {'request':body,'prompt_tokens':tokens,'valid':False,'transport':URL}
        started = time.monotonic()
        try:
            response = requests.post(URL,headers=headers,json=body,timeout=180)
            raw.update(http_status=response.status_code,response_text=response.text)
            raw['response'] = response.json()
            response.raise_for_status()
            if not valid_native_transport(raw,stage):raise ValueError('exact native model and complete stop required')
            value = teacher.parse(raw['response']['choices'][0]['message']['content'],row);raw['valid'] = True
        except Exception as error:raw.update(error_type=type(error).__name__,error=str(error))
        raw['elapsed_seconds'] = time.monotonic()-started;data.write(path,raw);attempts.append(str(path))
        if raw['valid']:return {'status':'valid','value':value,'attempts':attempts}
    return {'status':'invalid','value':None,'attempts':attempts}


def replay_stage(row,stage,observation,encoding,draft=None):
    attempts = observation['attempts'];raws = [data.read(p) for p in attempts]
    if all('transport' not in r for r in raws):return OLD_REPLAY(row,stage,observation,encoding,draft)
    if not 1 <= len(attempts) <= 3 or any(r.get('transport') != URL for r in raws):
        raise ValueError('one unchanged transport and original attempt allowance required')
    body = teacher.payload(row,stage,draft);tokens = len(encoding.encode(json.dumps(body['messages'],ensure_ascii=False),disallowed_special=()))+16
    found = None
    for i,(name,raw) in enumerate(zip(attempts,raws)):
        if (Path(name).name != f'attempt_{i}.json' or found is not None or raw['request'] != body
                or raw['prompt_tokens'] != tokens or tokens > 6300):raise ValueError('raw prompt or first-attempt order differs')
        if 'response' in raw and json.loads(raw['response_text']) != raw['response']:
            raise ValueError('raw response text and unmodified parsed response differ')
        parsed = None
        if valid_native_transport(raw,stage):
            try:parsed = teacher.parse(raw['response']['choices'][0]['message']['content'],row)
            except (ValueError,KeyError,TypeError,IndexError):pass
        if raw['valid'] != (parsed is not None):raise ValueError('native first-response validity differs')
        if parsed is not None:found = parsed
    expected = {'status':'valid' if found is not None else 'invalid','value':found,'attempts':attempts}
    if observation != expected or found is None and len(attempts) != 3:
        raise ValueError('complete original first or exhausted stage required')
    return found


def fatal(value):
    if value['teacher_valid']:return False
    return any(obs is not None and obs['status']=='invalid' and not valid_native_transport(data.read(obs['attempts'][-1]),stage)
        for stage in ('draft','review') if (obs := value[stage]) is not None)


def collect_batch(rows,root,encoding,tokenizer,streak=0,candidate=None):
    if not 1 <= len(rows) <= BATCH_CASES or len({r['id'] for r in rows}) != len(rows):raise ValueError('one bounded unique batch required')
    candidate = teacher.candidate if candidate is None else candidate
    root.mkdir();(root/'cases').mkdir();iterator = iter(rows);generated = {};failures = [];stopped = False
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = {}
        def submit():
            row = next(iterator,None)
            if row is not None:pending[pool.submit(candidate,row,root,encoding,tokenizer)] = row
        for _ in range(8):submit()
        while pending:
            ready,_ = wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                row = pending.pop(future)
                event = {'id':row['id']}
                try:
                    value = future.result()
                    if value['id'] != row['id']:raise ValueError('candidate identity changed')
                    data.write(root/'cases'/(row['id']+'.json'),value);generated[row['id']] = value
                    event.update(teacher_valid=value['teacher_valid'],fatal=False)
                    if value['teacher_valid']:streak = 0
                    else:
                        is_fatal = fatal(value);failures.append({'id':row['id'],'fatal':is_fatal});streak += 1
                        event['fatal'] = is_fatal
                        stopped = stopped or is_fatal or streak >= 3
                except Exception as error:
                    stopped = True;failures.append({'id':row['id'],'error_type':type(error).__name__,'error':str(error)})
                    event.update(error_type=type(error).__name__,error=str(error))
                event.update(consecutive_schema_invalid=streak,submissions_stopped=stopped)
                with (root/'completion_order.jsonl').open('a') as log:log.write(json.dumps(event,allow_nan=False)+'\n')
            if not stopped:
                for _ in ready:submit()
            original.scheduler.write_json(root/'progress.json',{'completed_firsts':len(generated),'pending_jobs':len(pending),
                'submissions_stopped':stopped,'consecutive_schema_invalid':streak,'failures':failures})
    values = [generated[r['id']] for r in rows if r['id'] in generated]
    data.write(root/'partial_firsts.json',values)
    data.write(root/'terminal.json',{'completed':not stopped,'firsts':len(values),'streak':streak,'failures':failures,'all_inflight_drained':True})
    if stopped:raise RuntimeError('batch stopped; all client firsts drained and preserved; no next batch admitted')
    return values,streak


def replay_batches(root,rows):
    directories = sorted((root/'batches').iterdir())
    expected_names = [f'{i:04d}' for i in range((len(rows)+BATCH_CASES-1)//BATCH_CASES)]
    if [p.name for p in directories] != expected_names:raise ValueError('complete fixed batch partition required')
    streak = 0
    for number,directory in enumerate(directories):
        selected = rows[number*BATCH_CASES:(number+1)*BATCH_CASES]
        values = data.read(directory/'partial_firsts.json');lookup = {r['id']:r for r in values}
        if [r['id'] for r in values] != [r['id'] for r in selected]:raise ValueError('whole ordered batch required')
        events = [json.loads(line) for line in (directory/'completion_order.jsonl').read_text().splitlines()]
        if len(events) != len(values) or {e['id'] for e in events} != set(lookup):raise ValueError('all completion events required')
        for event in events:
            value = lookup[event['id']];valid = value['teacher_valid'];is_fatal = fatal(value)
            streak = 0 if valid else streak+1
            if (event.get('teacher_valid') != valid or event.get('fatal') != is_fatal
                    or event.get('consecutive_schema_invalid') != streak or event.get('submissions_stopped') is not False
                    or is_fatal or streak >= 3):raise ValueError('global stop rule must hold across batch boundaries')
        terminal = data.read(directory/'terminal.json')
        if not terminal['completed'] or not terminal['all_inflight_drained'] or terminal['streak'] != streak:
            raise ValueError('successful drained batch terminal required')
        for side in ('before','after'):
            health = data.read(root/'health'/f'{side}_{number:04d}.json');workers = health['workers']
            if workers['in_flight'] != 0 or workers['max_workers'] != 8 or workers['codex_host']['max_rss_mb'] != 4096 or workers['codex_host']['max_turns'] != 100:
                raise ValueError('each fixed batch must be bracketed by an idle guarded endpoint')
    return {'batches':len(directories),'new_firsts':len(rows),'global_streak_replayed':True,'all_batches_drained':True}


def sources():
    state = require('prepare');rows = data.read(SOURCE/'cases.json');retained = data.read(ROOTS['prepare']/'first_candidates.json')
    ids = data.read(ROOTS['prepare']/'unattempted_ids.json');known = {r['id'] for r in retained}
    untouched = [r for r in rows if r['id'] not in known]
    if state['observed'] != 446 or state['invalid_firsts'] != 16 or [r['id'] for r in untouched] != ids or len(ids) != 578:
        raise ValueError('complete original446 firsts and578 untouched sources required')
    return rows,retained,untouched


def generate(pilot):
    _,environment = configure();rows,retained,untouched = sources();paths = endpoint_inputs()
    ids = set(original.pilot_ids(untouched));inherited = []
    paths += [ROOTS['prepare']/n for n in ('producer.json','result.json','actual_tool_receipt.json','first_candidates.json','unattempted_ids.json')]
    if not pilot:
        state = require('pilot_audit')
        if not state['generation_admitted']:raise ValueError('all eight isolated firsts must qualify before full continuation')
        inherited = data.read(ROOTS['pilot']/'labels.json')
        if {r['id'] for r in inherited} != ids:raise ValueError('unchanged isolated pilot required')
        paths += [ROOTS['pilot_audit']/n for n in ('producer.json','result.json','actual_tool_receipt.json')]+[ROOTS['pilot']/'labels.json']
    selected = [r for r in untouched if (r['id'] in ids) == pilot]
    root = ROOTS['pilot' if pilot else 'generate'];proof = freeze(root,'isolated_pilot' if pilot else 'drained_batches570',paths,environment)
    (root/'batches').mkdir();(root/'health').mkdir();generated = [];streak = 0
    encoding,tokenizer = original.previous.data.storage_encoding(),original.previous.native.tokenizer()
    try:
        for number,offset in enumerate(range(0,len(selected),BATCH_CASES)):
            idle_snapshot(root/'health'/f'before_{number:04d}.json')
            values,streak = collect_batch(selected[offset:offset+BATCH_CASES],root/'batches'/f'{number:04d}',encoding,tokenizer,streak)
            generated += values;idle_snapshot(root/'health'/f'after_{number:04d}.json')
            original.scheduler.write_json(root/'progress.json',{'completed_firsts':len(generated),'population':len(selected),'completed_batches':number+1,'pending_jobs':0})
            print(json.dumps({'completed_firsts':len(generated),'population':len(selected),'completed_batches':number+1}),flush=True)
    finally:
        partial = [v for p in sorted((root/'batches').glob('*/partial_firsts.json')) for v in data.read(p)]
        data.write(root/'partial_firsts.json',partial)
    labels = generated if pilot else prior.combine(rows,retained,inherited,generated)
    data.write(root/'labels.json',labels)
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl') and 'producer_source' not in p.parts and p.name != 'producer.json']
    original.finish(root,proof,{'status':'teacher_pilot_completed' if pilot else 'all1024_first_candidates_completed',
        'cases':len(labels),'teacher_valid':sum(r['teacher_valid'] for r in labels),'capacity_fit':sum(r['capacity_fit'] for r in labels),
        'new_candidates':len(generated),'pilot':pilot,'preserved_exhausted_failures':16,'raw_replay_pending':True},files)


def audit(pilot):
    _,environment = configure();rows,retained,untouched = sources();stage = 'pilot' if pilot else 'generate';state = require(stage)
    values = data.read(ROOTS[stage]/'labels.json');ids = set(original.pilot_ids(untouched))
    selected = [r for r in untouched if r['id'] in ids] if pilot else rows
    if [r['id'] for r in values] != [r['id'] for r in selected]:raise ValueError('entire original ordered source population required')
    paths = [ROOTS[stage]/n for n in ('producer.json','result.json','actual_tool_receipt.json','labels.json')]
    paths += [ROOTS['prepare']/n for n in ('producer.json','result.json','actual_tool_receipt.json','first_candidates.json')]
    root = ROOTS['pilot_audit' if pilot else 'audit'];proof = freeze(root,'raw_isolated_pilot_replay' if pilot else 'raw_complete1024_replay',paths,environment)
    encoding,tokenizer = original.previous.data.storage_encoding(),original.previous.native.tokenizer()
    all_attempts = set()
    for row,value in zip(selected,values):
        original.replay_candidate(row,value,encoding,tokenizer)
        for role in ('draft','review'):
            obs = value[role]
            if obs:all_attempts.update(obs['attempts'])
    new_raw = {str(p) for p in (ROOTS[stage]/'batches').rglob('attempt_*.json')}
    if {p for p in all_attempts if Path(p).is_relative_to(ROOTS[stage])} != new_raw:
        raise ValueError('all and only raw first attempts must be accounted for')
    if not pilot:
        require('pilot_audit');by_id = {r['id']:r for r in values}
        if any(by_id[r['id']] != r for r in retained+data.read(ROOTS['pilot']/'labels.json')):
            raise ValueError('previously observed positive or negative first changed')
    new_rows = [r for r in untouched if (r['id'] in ids) == pilot]
    batch_replay = replay_batches(ROOTS[stage],new_rows)
    data.write(root/'batch_replay.json',batch_replay)
    valid,fit = sum(r['teacher_valid'] for r in values),sum(r['capacity_fit'] for r in values)
    if state['teacher_valid'] != valid or state['capacity_fit'] != fit:raise ValueError('raw candidate counts differ')
    original.finish(root,proof,{'status':'teacher_pilot_firsts_replayed' if pilot else 'all1024_teacher_firsts_replayed',
        'cases':len(values),'teacher_valid':valid,'capacity_fit':fit,'generation_admitted':pilot and valid == fit == 8,
        'preserved_exhausted_failures':0 if pilot else 16,'new_requests':0,'batch_replay':batch_replay},[root/'batch_replay.json'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=tuple(ROOTS));args = parser.parse_args()
    root = ROOTS[args.stage];existed = root.exists()
    try:
        if args.stage == 'prepare':inventory()
        elif args.stage in ('pilot','generate'):generate(args.stage == 'pilot')
        else:audit(args.stage == 'pilot_audit')
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'error_type':type(error).__name__,'error':str(error),
            'preserve_firsts':True,'retry_same_sources_prohibited':True})
        raise
