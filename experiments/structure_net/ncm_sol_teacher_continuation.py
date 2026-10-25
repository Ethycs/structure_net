"""Collect only8631 untouched sources, preserving the schema-exhausted first."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import subprocess
import sys
import os

from experiments.structure_net import ncm_sol_teacher_expansion as previous
from experiments.neural_architecture_lab import audit_ncm_sol_teacher_partial as inventory
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = previous.native, previous.q, previous.ROOT
MODULE = 'experiments.structure_net.ncm_sol_teacher_continuation'
HYPOTHESIS = 'ncm-sol-teacher-untouched8631-firsts-v1'
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-sol-teacher8631-continuation-plan.md')
TEST = Path('tests/structure_net/test_ncm_sol_teacher_continuation.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_sol_teacher_continuation.py')
WORKFLOW, PREP, GENERATE = (ROOT / ('20261024_sol_teacher8631_' + n + '_v1') for n in ('workflow','prepare','candidates'))
WORKERS = 8


def sources():
    previous.require_teacher()
    if q.read(inventory.DEST/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual independent188-first inventory required')
    state = q.read(inventory.DEST/'result.json')
    if state['observed'] != 188 or state['invalid_firsts'] != 1 or state['unattempted'] != 8631:
        raise ValueError('unchanged188 firsts including invalid1 required')
    for path,digest in state['artifacts'].items():
        if file_hash(path) != digest:
            raise ValueError('original first or raw attempt changed')
    ids = q.read(inventory.DEST/'unattempted_ids.json')
    observed = {r['id'] for r in q.read(inventory.DEST/'first_candidates.json')}
    rows = [r for r in previous.sources() if r['id'] not in observed]
    if [r['id'] for r in rows] != ids or len(rows) != 8631 or {r['id'] for r in rows} & observed:
        raise ValueError('only exact8631 previously untouched sources may be submitted')
    return rows


def fatal_exhaustion(value):
    """A complete correct-model reply with bad support syntax is a retained case failure."""
    if value['teacher_valid']:
        return False
    for stage in ('draft','review'):
        observation = value[stage]
        if observation is None or observation['status'] != 'invalid':
            continue
        raw = q.read(observation['attempts'][-1])
        response = raw.get('response',{})
        choices = response.get('choices') or [{}]
        if (raw.get('http_status') != 200 or response.get('model') != previous.teacher.MODELS[stage]
                or choices[0].get('finish_reason') != 'stop'):
            return True
    return False


def producing(root,stage,inputs=()):
    files = [Path(__file__),PLAN,TEST,AUDITOR,Path(previous.__file__),previous.PLAN,Path(inventory.__file__),
        Path(previous.teacher.__file__),Path(previous.original.__file__),Path(previous.teacher.base.recipe.__file__)]
    common = [inventory.DEST/n for n in ('result.json','actual_tool_receipt.json','first_candidates.json','unattempted_ids.json','exhausted_firsts.json')]
    common += [previous.WORKFLOW/'actual_tool_receipt.json',previous.PILOT/'labels.json',previous.PILOT/'result.json',
        previous.PILOT/'worker_terminal_receipt.json',previous.qualification.RESULTS/'campaign_results.json',
        previous.RESERVATION/'reservation.json',ROOT/'20261024_sol_teacher8819_payload_scope_preflight_v2/result.json',
        ROOT/'20261024_sol_teacher8819_payload_scope_preflight_v2/actual_tool_receipt.json']
    p = native.vendor.freeze(root,stage,list(dict.fromkeys([*inputs,*common,*files])),{
        'hypothesis_id':HYPOTHESIS,'new_unattempted_sources':8631,'observed_firsts_preserved':1369,
        'prior_invalid_firsts_preserved':64,'same_qualified_recipe':True,'models':previous.teacher.MODELS,
        'systems':previous.original.SYSTEMS,'workers':8,'maximum_pending':8,'max_attempts_per_stage':3,
        'no_semantic_or_capacity_retries':True,'schema_exhausted_first_is_retained_not_retried':True,
        'stop_on_transport_model_or_termination_exhaustion':True,'stop_after_consecutive_schema_invalid_pipelines':3,
        'drain_inflight_on_global_failure':True,'original_eight_case_transport_pilot_reused':True,
        'new_semantic_judgments':False,'student_training_admitted':False,'student_task_accuracy':None,
        'component64_excluded_from_future_gradients':True,'final_test_opened':False,'speed_tested':False})
    p['hypothesis_id'] = HYPOTHESIS
    for i,path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root/'producer_source'/f'continuation_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k:v for k,v in p.items() if k!='fingerprint'})
    write_json(root/'producer.json',p)
    return p


def prepare():
    previous.configure()
    rows = sources()
    proof = previous.qualification.reference.verify_public(rows)
    native.vendor.terminal(previous.PILOT)
    if q.read(previous.PILOT/'result.json')['status'] != 'eight_first_candidate_systems_passed':
        raise ValueError('same qualified original eight-case pilot required')
    p = producing(PREP,'strictly_unattempted8631_source_bindings',[previous.qualification.reference.PUBLIC_FILE])
    write_json(PREP/'cases.json',rows)
    write_json(PREP/'public_proof.json',{'public_file_sha256':previous.qualification.reference.PUBLIC_SHA256,'exact_sources':proof})
    native.vendor.verify(p)
    write_json(PREP/'result.json',{'status':'all8631_untouched_sources_prepared','cases':8631,'no_firsts_rerolled':True})


def candidate_jobs(rows,root,encoding,tokenizer,candidate=None):
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('unique untouched IDs required')
    candidate = previous.original.candidate if candidate is None else candidate
    iterator,generated,failures,stopped,streak = iter(rows),{},{'schema_exhausted':[],'fatal':[]},False,0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        pending = {}
        def submit():
            row = next(iterator,None)
            if row is not None:
                pending[pool.submit(candidate,row,root,encoding,tokenizer)] = row
        for _ in range(WORKERS):
            submit()
        while pending:
            ready,_ = wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                row = pending.pop(future)
                try:
                    value = future.result()
                    if value['id'] != row['id']:
                        raise ValueError('candidate source identity changed')
                    write_json(root/'cases'/(row['id']+'.json'),value)
                    generated[row['id']] = value
                    if not value['teacher_valid']:
                        fatal = fatal_exhaustion(value)
                        failures['fatal' if fatal else 'schema_exhausted'].append(row['id'])
                        streak += 1
                        stopped = stopped or fatal or streak >= 3
                    else:
                        streak = 0
                except Exception as error:
                    stopped = True
                    failures['fatal'].append({'id':row['id'],'error_type':type(error).__name__,'error':str(error)})
            if not stopped:
                for _ in ready:
                    submit()
            write_json(root/'progress.json',{'completed_firsts':len(generated),'population':len(rows),'pending_jobs':len(pending),
                'submissions_stopped':stopped,'failures':failures,'consecutive_schema_invalid':streak})
            if len(generated)%16 == 0 or not pending:
                print(json.dumps({'teacher_firsts_completed':len(generated),'population':len(rows),'submissions_stopped':stopped}),flush=True)
    write_json(root/'partial_firsts.json',[generated[r['id']] for r in rows if r['id'] in generated])
    write_json(root/'failures.json',failures)
    if stopped:
        raise RuntimeError('global service/schema failure preserved; inflight drained; no first may be rerolled')
    return [generated[r['id']] for r in rows]


def generate():
    previous.configure()
    native.vendor.terminal(PREP)
    rows = sources()
    if rows != q.read(PREP/'cases.json'):
        raise ValueError('exact untouched source order required')
    p = producing(GENERATE,'all8631_unchanged_first_candidates',[PREP/n for n in ('cases.json','public_proof.json','worker_terminal_receipt.json')])
    (GENERATE/'cases').mkdir()
    labels = candidate_jobs(rows,GENERATE,previous.data.storage_encoding(),native.tokenizer())
    write_json(GENERATE/'labels.json',labels)
    native.vendor.verify(p)
    result = {'status':'all8631_first_candidates_completed_raw_replay_pending','cases':len(labels),
        'teacher_valid':sum(r['teacher_valid'] for r in labels),'capacity_fit':sum(r['capacity_fit'] for r in labels),
        'labels_sha256':file_hash(GENERATE/'labels.json'),'prior1369firsts_preserved':True,
        'student_training_admitted':False,'canonical_semantic_cache_changed':False,'student_task_accuracy':None}
    write_json(GENERATE/'result.json',result)
    print(json.dumps(result),flush=True)


def workflow():
    previous.configure()
    sources()
    if any(p.exists() for p in (WORKFLOW,PREP,GENERATE)):
        raise ValueError('fresh untouched-source continuation required')
    p = producing(WORKFLOW,'supervise_only_unattempted8631_firsts')
    receipts = []
    for stage,root in (('prepare',PREP),('generate',GENERATE)):
        native.vendor.verify(p)
        command = [sys.executable,'-u','-m',MODULE,stage]
        child = subprocess.Popen(command)
        write_json(WORKFLOW/'progress.json',{'active_stage':stage,'child_pid':child.pid,'stages':receipts})
        code = child.wait()
        receipt = {'stage':stage,'exit_code':code,'child_pid':child.pid,'observer_pid':os.getpid(),'observer':'subprocess.wait','command':command}
        write_json(WORKFLOW/(stage+'_receipt.json'),receipt)
        if root.exists():
            write_json(root/'worker_terminal_receipt.json',receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('untouched-source continuation failed: '+stage)
    native.vendor.verify(p)
    write_json(WORKFLOW/'result.json',{'status':'untouched8631_workflow_completed_raw_replay_pending','stages':receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('prepare','generate','workflow'))
    args = parser.parse_args()
    {'prepare':prepare,'generate':generate,'workflow':workflow}[args.stage]()
