"""Bind fixed task-supervision training to the complete isolated collection."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from experiments.structure_net import ncm_teacher1024_isolated as recovered
from experiments.structure_net import ncm_task_supervision_training as training

data,base,catalog,targets = training.data,training.base,training.catalog,training.targets
OLD_COLLECTION = catalog.collection
OLD_COLLECTION_ROOTS = dict(OLD_COLLECTION.ROOTS)
OLD_CATALOG_ROOTS = dict(catalog.ROOTS)
OLD_TARGET_ROOTS = dict(targets.ROOTS)
OLD_TRAIN_ROOTS = dict(training.ROOTS)
COLLECTION_ROOTS = {**OLD_COLLECTION_ROOTS,'generate':recovered.ROOTS['generate'],'audit':recovered.ROOTS['audit']}
CATALOG_ROOTS = {**OLD_CATALOG_ROOTS,**{s:data.ROOT/f'20261027_expanded_teacher_catalog_{s}_v3' for s in ('prepare','audit')}}
TARGET_ROOTS = {**OLD_TARGET_ROOTS,**{s:data.ROOT/f'20261027_expanded_teacher_codes_{s}_v3' for s in ('full','full_audit')}}
TRAIN_ROOTS = {k:(v if k[0] == 'existing' and k[1] in ('pilot','pilot_audit') else
    data.ROOT/f'20261027_task_supervision_{k[0]}_{k[1]}_v3') for k,v in OLD_TRAIN_ROOTS.items()}
STORAGE = Path('/data2/structure_net/ncm_task_supervision_20261027_v3')
ROOT = data.ROOT/'20261027_isolated_task_supervision_preflight_v1'
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-isolated-task-supervision.md')
TEST = Path('tests/structure_net/test_ncm_isolated_task_supervision.py')
SERVICE = Path('experiments/neural_architecture_lab/ncm_isolated_task_supervision_service.py')
CAT_FREEZE,TARGET_FREEZE,TRAIN_FREEZE = catalog.freeze,targets.freeze,training.freeze
TRAIN_PREREQUISITES = training.prerequisites
INSTALLED = False


def root_bindings():
    return {'collection':{k:str(v) for k,v in COLLECTION_ROOTS.items()},
        'catalog':{k:str(v) for k,v in CATALOG_ROOTS.items()},'targets':{k:str(v) for k,v in TARGET_ROOTS.items()},
        'training':{a+':'+s:str(v) for (a,s),v in TRAIN_ROOTS.items()},'checkpoint_storage':str(STORAGE)}


def require_collection(stage):
    root = COLLECTION_ROOTS[stage];value = base.require(root);base.verify_proof(data.read(root/'producer.json'))
    if stage in ('generate','audit'):
        expected = 'all1024_first_candidates_completed' if stage == 'generate' else 'all1024_teacher_firsts_replayed'
        if value['status'] != expected or value['cases'] != 1024:
            raise ValueError('complete recovered1024 first population required')
        if stage == 'audit' and (value['preserved_exhausted_failures'] != 16 or value['new_requests'] != 0):
            raise ValueError('all sixteen historical failures and request-free full replay required')
    return value


def extra_inputs():return [Path(__file__),PLAN,TEST,SERVICE]


def require_preflight():
    value = base.require(ROOT);base.verify_proof(data.read(ROOT/'producer.json'))
    if value['status'] != 'isolated_task_supervision_lineage_verified' or data.read(ROOT/'bindings.json') != root_bindings():
        raise ValueError('exact completed isolated collection and immutable existing controls required')
    return extra_inputs()+[ROOT/n for n in ('producer.json','result.json','actual_tool_receipt.json','bindings.json')]


def catalog_freeze(root,stage,environment,paths):
    return CAT_FREEZE(root,stage,environment,[*paths,*require_preflight()])


def target_freeze(root,stage,environment,model_training,paths):
    return TARGET_FREEZE(root,stage,environment,model_training,[*paths,*require_preflight()])


def train_freeze(root,arm,stage,model_training,environment,paths):
    return TRAIN_FREEZE(root,arm,stage,model_training,environment,[*paths,*require_preflight()])


def prerequisites(arm,stage):
    extra = require_preflight();rows,index,manifest,paths = TRAIN_PREREQUISITES(arm,stage)
    return rows,index,manifest,[*paths,*extra]


def install():
    global INSTALLED
    if INSTALLED:return
    # Keep collector modules untouched: only this consumer receives the new completed roots.
    proxy = SimpleNamespace(ROOTS=COLLECTION_ROOTS,previous=OLD_COLLECTION.previous,
        install=lambda:None,configure=recovered.configure,
        original=SimpleNamespace(require=require_collection,replay_candidate=recovered.original.replay_candidate))
    catalog.collection = proxy;catalog.ROOTS = CATALOG_ROOTS;catalog.freeze = catalog_freeze
    targets.ROOTS = TARGET_ROOTS;targets.freeze = target_freeze
    training.ROOTS = TRAIN_ROOTS;training.STORAGE = STORAGE;training.SERVICE = SERVICE
    training.freeze = train_freeze;training.prerequisites = prerequisites
    INSTALLED = True


def validate_firsts(rows,retained,labels,old_pilot,new_pilot):
    ids = [r['id'] for r in rows];by_id = {r['id']:r for r in labels}
    if (len(ids) != 1024 or len(set(ids)) != 1024 or [r['id'] for r in labels] != ids
            or len(retained) != 446 or len({r['id'] for r in retained}) != 446
            or sum(not r['teacher_valid'] for r in retained) != 16
            or len(old_pilot) != 8 or len(new_pilot) != 8
            or len({r['id'] for r in old_pilot+new_pilot}) != 16
            or not {r['id'] for r in old_pilot} <= {r['id'] for r in retained}
            or {r['id'] for r in new_pilot} & {r['id'] for r in retained}
            or any(by_id.get(r['id']) != r for r in retained+old_pilot+new_pilot)):
        raise ValueError('complete original1024 outcomes and all exact retained firsts required')
    return {'outcomes':1024,'preserved_firsts':446,'preserved_historical_invalid':16,
        'total_invalid':sum(not r['teacher_valid'] for r in labels),
        'teacher_valid':sum(r['teacher_valid'] for r in labels),'capacity_fit':sum(r['capacity_fit'] for r in labels)}


def preflight():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or ROOT.exists():raise ValueError('fresh CPU-only lineage preflight required')
    _,environment = recovered.configure()
    paths = catalog.require_old()+training.require_parent()+extra_inputs()
    for root in (OLD_CATALOG_ROOTS['pilot'],OLD_TARGET_ROOTS['pilot'],OLD_TARGET_ROOTS['pilot_audit'],
            OLD_TRAIN_ROOTS['existing','pilot'],OLD_TRAIN_ROOTS['existing','pilot_audit'],
            *[recovered.ROOTS[s] for s in ('prepare','pilot','pilot_audit','generate','audit')]):
        base.require(root);base.verify_proof(data.read(root/'producer.json'))
        paths += [root/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    old_pilot = data.read(OLD_COLLECTION_ROOTS['pilot']/'labels.json')
    retained = data.read(recovered.ROOTS['prepare']/'first_candidates.json')
    rows = data.read(OLD_COLLECTION_ROOTS['prepare']/'cases.json')
    labels = data.read(recovered.ROOTS['generate']/'labels.json')
    new_pilot = data.read(recovered.ROOTS['pilot']/'labels.json')
    observed = validate_firsts(rows,retained,labels,old_pilot,new_pilot)
    generated,audited = require_collection('generate'),require_collection('audit')
    for value in (generated,audited):
        if any(value[k] != observed[k] for k in ('teacher_valid','capacity_fit')):
            raise ValueError('complete replayed positive and negative outcome counts required')
    by_id = {r['id']:r for r in labels}
    code_pilot = data.read(OLD_TARGET_ROOTS['pilot']/'cases.json')
    if len(code_pilot) != 24 or sum(r['id'] in {v['id'] for v in old_pilot} for r in code_pilot) != 8:
        raise ValueError('exact original sixteen controls and eight fresh target-code controls required')
    for r in code_pilot:
        if r['id'] in by_id:
            first = by_id[r['id']]
            if any(r[k] != first[k] for k in ('source_sha256','target_text','native_word_ids')):
                raise ValueError('preserved code pilot target no longer matches first response')
    pilot = base.require(OLD_TRAIN_ROOTS['existing','pilot'])
    audit = base.require(OLD_TRAIN_ROOTS['existing','pilot_audit'])
    if (pilot['cases'] != 1376 or not pilot['strict_resumed_update_passed'] or not pilot['gradient_parity_passed']
            or audit['status'] != 'task_supervision_lifecycle_replayed'
            or audit['source_checkpoint_sha256'] != pilot['checkpoint_sha256']):
        raise ValueError('original complete-population training lifecycle must remain qualified')
    paths += [OLD_COLLECTION_ROOTS['pilot']/'labels.json',OLD_COLLECTION_ROOTS['prepare']/'cases.json',
        recovered.ROOTS['prepare']/'first_candidates.json',recovered.ROOTS['generate']/'labels.json',
        recovered.ROOTS['pilot']/'labels.json',
        OLD_TARGET_ROOTS['pilot']/'cases.json',Path('pixi.lock'),Path('pyproject.toml')]
    files = {p.resolve() for p in extra_inputs()}
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()):files.add(p)
    proof = {'schema':'nal.ncm-isolated-task-supervision.v1','hypothesis_id':training.HYPOTHESIS,
        'stage':'lineage_preflight','environment':environment,'config':{'bindings':root_bindings(),
            'complete_collection_required':1024,'minimum_new_training_pairs':900,'preserved_historical_invalid_firsts':16,
            'native_and_historical_raw_replay':True,'training_budget_changed':False,'quality_gates_changed':False},
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},'input_hashes':{str(p):data.sha(p) for p in paths}}
    recovered.recovery.write_producer(ROOT,proof,sorted(files));data.write(ROOT/'bindings.json',root_bindings())
    base.verify_proof(proof)
    value = {'schema':proof['schema'],'hypothesis_id':training.HYPOTHESIS,
        'status':'isolated_task_supervision_lineage_verified','original_cases':1024,**observed,
        'unattempted':0,'original_code_pilot_preserved':True,
        'existing_training_pilot_preserved':True,'model_training':False,'new_requests':0,'new_judge_requests':0,
        'task_accuracy':None,'final_test_opened':False,'full_training_admitted':False,
        'artifacts':{str(p):data.sha(p) for p in (ROOT/'producer.json',ROOT/'bindings.json')}}
    data.write(ROOT/'result.json',value);print(json.dumps({k:v for k,v in value.items() if k!='artifacts'}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);sub = parser.add_subparsers(dest='action',required=True)
    sub.add_parser('preflight')
    c = sub.add_parser('catalog');c.add_argument('stage',choices=('prepare','audit'))
    c = sub.add_parser('codes');c.add_argument('stage',choices=('encode','audit'))
    for name in ('training','audit'):
        c = sub.add_parser(name);c.add_argument('--arm',choices=training.ARMS,required=True)
        c.add_argument('--stage',choices=('pilot','train'),required=True)
    args = parser.parse_args()
    if args.action == 'preflight':root = ROOT
    elif args.action == 'catalog':root = CATALOG_ROOTS[args.stage]
    elif args.action == 'codes':root = TARGET_ROOTS['full' if args.stage == 'encode' else 'full_audit']
    elif args.action == 'training':root = TRAIN_ROOTS[args.arm,args.stage]
    else:root = TRAIN_ROOTS[args.arm,'pilot_audit' if args.stage == 'pilot' else 'audit']
    existed = root.exists()
    try:
        if args.action == 'preflight':preflight()
        else:
            install();require_preflight()
            if args.action == 'catalog':catalog.audit() if args.stage == 'audit' else catalog.prepare(False)
            elif args.action == 'codes':getattr(targets,args.stage)('full')
            elif args.action == 'training':training.run(args.arm,args.stage)
            else:
                from experiments.neural_architecture_lab import audit_ncm_task_supervision as auditor
                auditor.audit(args.arm,args.stage)
    except Exception as error:
        if root.exists() and not existed:data.write(root/'failure.json',{'error_type':type(error).__name__,'error':str(error)})
        raise
