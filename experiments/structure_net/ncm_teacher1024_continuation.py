"""Preserve timed-out teacher firsts; collect only untouched reserved sources."""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.structure_net import ncm_teacher1024_recovery as old

data,base,original = old.data,old.base,old.original
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-teacher1024-timeout-continuation.md')
TEST = Path('tests/structure_net/test_ncm_teacher1024_continuation.py')
SERVICE = Path('experiments/neural_architecture_lab/ncm_teacher1024_continuation_service.py')
FAILED = old.ROOTS['generate']
TERMINAL = data.ROOT/'20261027_teacher1024_supervision_generate_v2'/'terminal'
ROOTS = {s:data.ROOT/f'20261027_teacher1024_{s}_v3' for s in ('prepare','pilot','pilot_audit','generate','audit')}


def configure():
    old.install()
    return old.configure()


def freeze(root,stage,inputs,environment):
    proof = old.freeze(root,stage,[*inputs,PLAN,TEST,SERVICE,Path(__file__)],environment)
    proof['config'].update(continuation='only586 untouched reserved sources after drained timeout failure',
        original438_firsts_preserved=True,original8_exhausted_failures_preserved=True,
        exhausted_firsts_receive_no_new_attempt_allowance=True)
    proof['fingerprint'] = base.fingerprint(proof);data.write(root/'producer.json',proof)
    return proof


def partition(rows,retained,partial,raw_ids,saved_ids):
    ids = [r['id'] for r in rows];a = [r['id'] for r in retained];b = [r['id'] for r in partial]
    if (len(ids) != len(set(ids)) or len(a) != len(set(a)) or len(b) != len(set(b))
            or set(a)&set(b) or not set(a)|set(b) <= set(ids)
            or set(b) != set(raw_ids) or set(b) != set(saved_ids)):
        raise ValueError('all submitted cases must be exhausted or complete; preserve disjoint firsts')
    by_id = {r['id']:r for r in (*retained,*partial)}
    observed = [by_id[k] for k in ids if k in by_id]
    untouched = [k for k in ids if k not in by_id]
    if len(observed)+len(untouched) != len(rows):raise ValueError('complete original denominator required')
    return observed,untouched


def replay_paths(source,value,root,encoding,tokenizer):
    original.replay_candidate(source,value,encoding,tokenizer)
    paths = []
    for stage in ('draft','review'):
        observation = value[stage]
        actual = sorted((root/'requests'/source['id']/stage).glob('attempt_*.json'))
        if observation is None:
            if actual:raise ValueError('unexpected review attempts after failed draft')
            continue
        expected = [root/'requests'/source['id']/stage/f'attempt_{i}.json' for i in range(len(observation['attempts']))]
        if ([Path(p) for p in observation['attempts']] != expected or actual != expected
                or observation['status'] == 'invalid' and len(expected) != 3):
            raise ValueError('all and only original first attempts required')
        paths += expected
    return paths


def inventory():
    _,environment = configure()
    for stage in ('prepare','pilot','pilot_audit'):original.require(stage)
    terminal = data.read(TERMINAL/'result.json');receipt = data.read(FAILED/'actual_tool_receipt.json')
    data.verify(receipt['artifacts']);data.verify(terminal['artifacts'])
    base.verify_proof(data.read(FAILED/'producer.json'))
    if (receipt['actual_tool_exit_code'] != 1 or receipt['method'] != 'systemd_terminal_state_observed'
            or terminal['status'] != 'terminal_failure' or not terminal['worker_terminal']
            or terminal['worker_process_exit_code'] != 1 or terminal['state']['MainPID'] != '0'
            or terminal['state']['InvocationID'] != '5a1d8477f5e74a3f9927b8add1e1ec34'):
        raise ValueError('actual terminal failed original worker required')
    rows = data.read(old.ROOTS['prepare']/'cases.json');retained = data.read(old.ROOTS['pilot']/'labels.json')
    partial = data.read(FAILED/'partial_firsts.json');progress = data.read(FAILED/'progress.json')
    raw_ids = {p.name for p in (FAILED/'requests').iterdir() if p.is_dir()}
    saved_ids = {p.stem for p in (FAILED/'cases').glob('*.json')}
    observed,untouched = partition(rows,retained,partial,raw_ids,saved_ids)
    if (len(rows) != 1024 or len(retained) != 8 or len(partial) != 430 or len(untouched) != 586
            or progress['completed_firsts'] != 430 or progress['pending_jobs'] != 0
            or not progress['submissions_stopped'] or len(progress['failures']['fatal']) != 8
            or progress['failures']['schema_exhausted']):raise ValueError('complete drained430-first terminal inventory required')
    inputs = [p/n for p in (old.ROOTS['prepare'],old.ROOTS['pilot'],old.ROOTS['pilot_audit'])
        for n in ('producer.json','result.json','actual_tool_receipt.json')]
    inputs += [FAILED/n for n in ('producer.json','failure.json','progress.json','partial_firsts.json','failures.json','actual_tool_receipt.json')]
    inputs += [TERMINAL/'result.json',TERMINAL/'actual_tool_receipt.json',old.ROOTS['prepare']/'cases.json',old.ROOTS['pilot']/'labels.json']
    inputs += sorted((FAILED/'requests').rglob('*.json'))+sorted((FAILED/'cases').glob('*.json'))
    root = ROOTS['prepare'];proof = freeze(root,'independent_raw438_first_inventory',inputs,environment)
    encoding,tokenizer = original.previous.data.storage_encoding(),original.previous.native.tokenizer()
    lookup = {r['id']:r for r in rows};paths = [];errors = Counter()
    for value in observed:
        origin = old.ROOTS['pilot'] if value['id'] in {r['id'] for r in retained} else FAILED
        paths += replay_paths(lookup[value['id']],value,origin,encoding,tokenizer)
        if origin == FAILED and data.read(FAILED/'cases'/(value['id']+'.json')) != value:
            raise ValueError('saved first differs from partial inventory')
        if not value['teacher_valid']:
            for stage in ('draft','review'):
                observation = value[stage]
                if observation is not None and observation['status'] == 'invalid':
                    for path in observation['attempts']:errors[data.read(path).get('error_type','missing_error_type')] += 1
    expected_raw = set(sorted((FAILED/'requests').rglob('*.json'))) | set(sorted((old.ROOTS['pilot']/'requests').rglob('*.json')))
    invalid = [r['id'] for r in observed if not r['teacher_valid']]
    if set(paths) != expected_raw or set(invalid) != set(progress['failures']['fatal']) or errors != Counter({'ReadTimeout':24}):
        raise ValueError('all raw attempts and eight exhausted transport failures must be preserved')
    for name,value in (('first_candidates.json',observed),('unattempted_ids.json',untouched),
            ('failed_ids.json',invalid),('error_counts.json',dict(errors))):data.write(root/name,value)
    original.finish(root,proof,{'status':'teacher438_firsts_replayed_586_unattempted','original_cases':1024,
        'observed':len(observed),'unattempted':len(untouched),'invalid_firsts':len(invalid),
        'teacher_valid':sum(r['teacher_valid'] for r in observed),'capacity_fit':sum(r['capacity_fit'] for r in observed),
        'all_inflight_drained':True,'new_requests':0},[root/n for n in
        ('first_candidates.json','unattempted_ids.json','failed_ids.json','error_counts.json')])


def require(stage):
    value = base.require(ROOTS[stage]);base.verify_proof(data.read(ROOTS[stage]/'producer.json'));return value


def inputs():
    state = require('prepare')
    if (state['observed'] != 438 or state['unattempted'] != 586 or state['invalid_firsts'] != 8
            or not state['all_inflight_drained']):raise ValueError('exact independently replayed timeout inventory required')
    rows = data.read(old.ROOTS['prepare']/'cases.json')
    old_firsts = data.read(ROOTS['prepare']/'first_candidates.json')
    ids = data.read(ROOTS['prepare']/'unattempted_ids.json');known = {r['id'] for r in old_firsts}
    untouched = [r for r in rows if r['id'] not in known]
    if [r['id'] for r in untouched] != ids:raise ValueError('only exact untouched source order may be submitted')
    return rows,old_firsts,untouched


def combine(rows,*groups):
    values = [r for group in groups for r in group];by_id = {r['id']:r for r in values}
    if len(by_id) != len(values) or set(by_id) != {r['id'] for r in rows}:
        raise ValueError('one unchanged first per original source required')
    return [by_id[r['id']] for r in rows]


def generate(pilot):
    _,environment = configure();rows,retained,untouched = inputs()
    original.require('pilot_audit')
    pilot_ids = set(original.pilot_ids(untouched));inherited = []
    paths = [ROOTS['prepare']/n for n in ('producer.json','result.json','actual_tool_receipt.json','first_candidates.json','unattempted_ids.json')]
    paths += [old.ROOTS['pilot_audit']/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    if not pilot:
        passed = require('pilot_audit')
        if not passed['generation_admitted']:raise ValueError('eight new firsts must pass independent raw pilot audit')
        inherited = data.read(ROOTS['pilot']/'labels.json')
        if {r['id'] for r in inherited} != pilot_ids:raise ValueError('exact preserved continuation pilot required')
        paths += [ROOTS['pilot_audit']/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
        paths += [ROOTS['pilot']/'labels.json']
    selected = [r for r in untouched if (r['id'] in pilot_ids) == pilot]
    root = ROOTS['pilot' if pilot else 'generate'];proof = freeze(root,'continuation_pilot' if pilot else 'untouched_remaining578',paths,environment)
    (root/'cases').mkdir()
    values = original.scheduler.candidate_jobs(selected,root,original.previous.data.storage_encoding(),original.previous.native.tokenizer())
    labels = values if pilot else combine(rows,retained,inherited,values)
    data.write(root/'labels.json',labels)
    original.finish(root,proof,{'status':'teacher_pilot_completed' if pilot else 'all1024_first_candidates_completed',
        'cases':len(labels),'teacher_valid':sum(r['teacher_valid'] for r in labels),
        'capacity_fit':sum(r['capacity_fit'] for r in labels),'pilot':pilot,'new_candidates':len(values),
        'preserved_original_firsts':438,'preserved_exhausted_failures':8,'raw_replay_pending':True},
        [root/'labels.json',root/'partial_firsts.json',root/'failures.json',*sorted((root/'requests').rglob('*.json'))])


def audit(pilot):
    _,environment = configure();rows,retained,untouched = inputs()
    stage = 'pilot' if pilot else 'generate';state = require(stage)
    values = data.read(ROOTS[stage]/'labels.json');ids = set(original.pilot_ids(untouched))
    selected = [r for r in untouched if r['id'] in ids] if pilot else rows
    if [r['id'] for r in values] != [r['id'] for r in selected]:raise ValueError('complete fixed source order required')
    root = ROOTS['pilot_audit' if pilot else 'audit']
    paths = [ROOTS[stage]/n for n in ('producer.json','result.json','actual_tool_receipt.json','labels.json')]
    paths += [ROOTS['prepare']/n for n in ('producer.json','result.json','actual_tool_receipt.json','first_candidates.json')]
    proof = freeze(root,'replay_continuation_pilot' if pilot else 'replay_all1024_including_preserved_failures',paths,environment)
    encoding,tokenizer = original.previous.data.storage_encoding(),original.previous.native.tokenizer()
    for row,value in zip(selected,values):original.replay_candidate(row,value,encoding,tokenizer)
    if not pilot:
        require('pilot_audit');by_id = {r['id']:r for r in values}
        if any(by_id[r['id']] != r for r in retained+data.read(ROOTS['pilot']/'labels.json')):
            raise ValueError('a previously observed first was replaced')
    valid,fit = sum(r['teacher_valid'] for r in values),sum(r['capacity_fit'] for r in values)
    if state['teacher_valid'] != valid or state['capacity_fit'] != fit:raise ValueError('raw count replay differs')
    original.finish(root,proof,{'status':'teacher_pilot_firsts_replayed' if pilot else 'all1024_teacher_firsts_replayed',
        'cases':len(values),'teacher_valid':valid,'capacity_fit':fit,
        'generation_admitted':pilot and valid == fit == 8,'new_requests':0,
        'preserved_exhausted_failures':0 if pilot else 8},[])


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
