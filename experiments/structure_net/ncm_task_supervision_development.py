"""Source-only semantic evaluation of both completed task-supervision arms."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from experiments.structure_net import ncm_task_supervision_training as study
from experiments.structure_net import ncm_public_author_lineage as history
from experiments.structure_net import ncm_public_development as public_sources
from experiments.neural_architecture_lab import finalize_ncm_public_curriculum as public_results

data, base = study.data, study.base
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-task-supervision-evaluation.md')
TEST = Path('tests/structure_net/test_ncm_task_supervision_development.py')
AUDITOR = Path('experiments/neural_architecture_lab/audit_ncm_task_supervision_development.py')
PREFLIGHT = data.ROOT/'20261027_task_supervision_DEV_preflight_v1'
OFFLINE = data.ROOT/'20261027_task_supervision_DEV_offline_v1'
STAGES = ('inference_pilot','outputs','judge_pilot','grade','audit')
ROOTS = {a:{s:data.ROOT/f'20261027_task_supervision_{a}_DEV_{s}_v1' for s in STAGES} for a in study.ARMS}
COHORTS = ('short_development','long_diagnostic')
BOOT = None


def completed(root):
    public_results.receipt(root)
    value = data.completed(root)
    base.verify_proof(data.read(root/'producer.json'))
    return value


def check_training_record(value, audited, arm, cases):
    if (value['status'] != 'task_supervision_training_completed' or value['arm'] != arm
            or value['stage'] != 'train' or value['config'] != asdict(study.CONFIG)
            or value['updates'] != 2048 or value['presentations'] != 65536 or value['cases'] != cases
            or not value['strict_final_reload_passed'] or not value['protected_tensors_unchanged']
            or value['systems_lifecycle_only_not_quality_evidence']
            or not value['fresh_optimizer_masters_match_initial_weights']
            or audited['status'] != 'task_supervision_training_replayed' or audited['arm'] != arm
            or audited['stage'] != 'train' or audited['cases'] != cases or audited['updates'] != 2048
            or audited['source_checkpoint_sha256'] != value['checkpoint_sha256']):
        raise ValueError('both complete fixed-budget arms and exact checkpoint audits required')


def bound_training(binding_root):
    binding_root = Path(binding_root)
    bound = completed(binding_root)
    if bound['hypothesis_id'] != study.HYPOTHESIS:
        raise ValueError('task-supervision lineage binding required')
    bindings = data.read(binding_root/'bindings.json')
    paths = [binding_root/n for n in ('producer.json','result.json','actual_tool_receipt.json','bindings.json')]
    catalog_root, catalog_audit = (Path(bindings['catalog'][s]) for s in ('prepare','audit'))
    target_root, target_audit = (Path(bindings['targets'][s]) for s in ('full','full_audit'))
    for root,flag in ((catalog_audit,'training_data_admitted'),(target_audit,'training_targets_admitted')):
        if completed(root).get(flag) is not True:
            raise ValueError('complete expanded data and frozen-codec targets must be admitted')
    for root in (catalog_root,target_root):completed(root)
    arms = data.read(catalog_root/'arm_ids.json');rows = data.read(catalog_root/'cases.json')
    if (len(arms['existing']) != 1376 or len(arms['new_admitted']) < 900
            or arms['expanded'] != arms['existing']+arms['new_admitted']
            or [r['id'] for r in rows] != arms['expanded']):
        raise ValueError('entire existing and expanded source populations required')
    study.catalog.require_old()
    if rows[:1376] != data.read(study.catalog.old.DEST/'cases.json'):
        raise ValueError('historical training rows changed')
    paths += [catalog_root/'cases.json',catalog_root/'arm_ids.json',target_root/'cases.json']
    paths += [r/n for r in (catalog_root,catalog_audit,target_root,target_audit)
              for n in ('producer.json','result.json','actual_tool_receipt.json')]
    trained = {}
    for arm in study.ARMS:
        root, audit = (Path(bindings['training'][arm+':'+s]) for s in ('train','audit'))
        value, checked = completed(root),completed(audit)
        check_training_record(value,checked,arm,len(arms[arm]))
        if (checked['source_result_sha256'] != data.sha(root/'result.json')
                or value['checkpoint_sha256'] != data.sha(root/'model.pt')
                or data.read(root/'producer.json')['parent_checkpoint_sha256'] != study.schedule.PARENT_CHECKPOINT_SHA):
            raise ValueError('exact final checkpoint and fixed epoch2 initializer required')
        trained[arm] = {'root':root,'result':value}
        paths += [root/n for n in ('producer.json','result.json','actual_tool_receipt.json','model.pt')]
        paths += [audit/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    if trained['existing']['result']['frozen_hashes'] != trained['expanded']['result']['frozen_hashes']:
        raise ValueError('the original frozen body and reconstruction path must match')
    return bindings,trained,rows,paths


def current_history():
    global BOOT
    if BOOT is None:
        training,evaluation,environment = history.configure(4,1)
        BOOT = training,evaluation,environment,evaluation.lineage
    training,evaluation,environment,ancestor = BOOT
    cache = ancestor.base_cache();paths = list(ancestor.inputs())
    for epoch in (4,8):
        stages = history.roots(epoch)
        if not any(stages[s].exists() for s in ('judge_pilot','grade')):continue
        roots = public_results.roots(epoch)
        # A partial/new judging branch cannot be skipped or silently reset.
        public_results.receipt(roots['semantic_replay'],('result.json','scores.json'))
        public_results.receipt(roots['results'],('campaign_results.json',))
        campaign = data.read(roots['results']/'campaign_results.json');data.verify(campaign['artifacts'])
        cache = history.parent.original.append_grade(cache,stages['grade'],evaluation)
        if campaign['canonical_first_judgments'] != len(cache):
            raise ValueError('complete intervening public judgment lineage required')
        paths += [stages['grade']/n for n in ('producer.json','result.json','actual_tool_receipt.json','exact_packets.json','judgment_cache.json')]
        paths += [roots['semantic_replay']/n for n in ('result.json','scores.json','actual_tool_receipt.json')]
        paths += [roots['results']/n for n in ('campaign_results.json','actual_tool_receipt.json')]
    ancestor.audit_cache(cache)
    return training,evaluation,environment,ancestor,cache,paths


def sources(evaluation, train_rows):
    cohorts,manifest = evaluation.old.source_inputs()
    identity = evaluation.split_guard(train_rows,cohorts)
    index = study.catalog.guard.TextIndex([r['source'] for c in COHORTS for r in cohorts[c]])
    for row in train_rows:
        if index.matches(row['source']):raise ValueError('reviewed TRAIN/DEV full-text overlap')
    public = public_sources.public_guard(evaluation,cohorts)
    return cohorts,manifest,{'reviewed_union':identity,'reviewed_full_text_overlap':0,'public':public}


def freeze(root,stage,arm,environment,paths,inherited):
    if root.exists():raise ValueError('fresh append-only evaluation root required')
    files = {Path(__file__).resolve(),PLAN.resolve(),TEST.resolve(),AUDITOR.resolve(),study.PLAN.resolve()}
    if BOOT is None:raise ValueError('qualified model and semantic runtime must be configured before freezing')
    files.update(p.resolve() for p in BOOT[0].native.vendor.VENDOR.rglob('*.py'))
    for module in tuple(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()):files.add(p)
    proof = {'schema':'nal.ncm-task-supervision-dev.v1','hypothesis_id':study.HYPOTHESIS,
        'stage':stage,'arm':arm,'environment':environment,'config':{'source_only':True,
            'target_fields_in_prediction':False,'DEV_cases':128,'cohorts':list(COHORTS),
            'rubric':BOOT[0].q.RUBRIC,'judges':BOOT[0].q.JUDGES,
            'inherited_first_judgments':inherited,'metric':'complete_summary_semantic_pass',
            'stored_token_cap':128,'final_test_opened':False,'speed_tested':False},
        'source_hashes':{str(p):data.sha(p) for p in sorted(files)},
        'input_hashes':{str(p):data.sha(p) for p in dict.fromkeys([*paths,Path('pixi.lock'),Path('pyproject.toml')])}}
    return write_proof(root,proof,files)


def write_proof(root,proof,files):
    proof['fingerprint'] = base.fingerprint(proof)
    root.mkdir();(root/'producer_source').mkdir()
    for i,p in enumerate(sorted(files)):(root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
    data.write(root/'producer.json',proof);return proof


def finish(root,proof,value):
    base.verify_proof(proof)
    files = [p for p in root.rglob('*.json') if 'producer_source' not in p.parts]
    result = {'schema':proof['schema'],'hypothesis_id':study.HYPOTHESIS,'arm':proof['arm'],**value,
        'final_test_opened':False,'speed_tested':False,'producer_fingerprint':proof['fingerprint'],
        'artifacts':{str(p):data.sha(p) for p in files}}
    data.write(root/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k != 'artifacts'}),flush=True)


def preflight(binding_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':raise ValueError('CPU-only evaluation preflight required')
    bindings,trained,rows,paths = bound_training(binding_root)
    training,evaluation,environment,ancestor,cache,history_paths = current_history()
    cohorts,_,guard = sources(evaluation,rows)
    paths += history_paths+[evaluation.old.SOURCE/'cases.json',evaluation.old.FEATURES/'manifest.json']
    proof = freeze(PREFLIGHT,'preflight',None,environment,paths,len(cache))
    data.write(PREFLIGHT/'bindings.json',{'origin':str(binding_root),'bindings':bindings})
    data.write(PREFLIGHT/'judgment_cache.json',cache);data.write(PREFLIGHT/'split_proof.json',guard)
    finish(PREFLIGHT,proof,{'status':'task_supervision_semantic_preflight_passed','cases':128,
        'canonical_first_judgments':len(cache),'model_training':False,'new_judge_requests':0,'task_accuracy':None})


def offline():
    """Exercise actual ancestry, existing-source guard and historical decisions."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only offline qualification required')
    training,evaluation,environment,ancestor,cache,paths = current_history()
    reviewed = study.catalog.old.DEST/'cases.json';rows = data.read(reviewed)
    cohorts,_,guard = sources(evaluation,rows)
    historical = history.roots(2)['grade'];public_results.receipt(historical)
    paths += [reviewed,evaluation.old.SOURCE/'cases.json',evaluation.old.FEATURES/'manifest.json']
    paths += [historical/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    historical_scores = {}
    for c in COHORTS:
        path = historical/c/'secondary_student_case_scores.json';paths.append(path)
        historical_scores[c] = {'secondary':data.read(path)}
    replayed = summarize(historical_scores);recorded = data.read(historical/'result.json')
    if (replayed['task_accuracy'] != recorded['task_accuracy'] or replayed['DEV_qualified'] != recorded['DEV_qualified']
            or replayed['valid'] != 128 or replayed['passed'] != 0 or len(rows) != 1376
            or len(cache) < 5263 or sum(map(len,cohorts.values())) != 128):
        raise ValueError('actual complete historical semantics and source populations must replay')
    proof = freeze(OFFLINE,'offline_systems_qualification',None,environment,paths,len(cache))
    data.write(OFFLINE/'split_proof.json',guard);data.write(OFFLINE/'historical_score_replay.json',replayed)
    if cache != current_history()[4]:raise ValueError('repeated configuration changed first judgments')
    finish(OFFLINE,proof,{'status':'task_supervision_semantic_offline_qualification_passed',
        'canonical_first_judgments':len(cache),'DEV_cases':128,'reviewed_TRAIN_cases_checked':1376,
        'historical_epoch2_passes':replayed['passed'],'expanded_sources_checked':False,
        'both_arm_training_admitted':False,'new_judge_requests':0,'model_training':False,
        'task_accuracy':None,'systems_lifecycle_only_not_quality_evidence':True})


def configure(arm):
    if arm not in study.ARMS:raise ValueError('fixed paired arm required')
    checked = completed(PREFLIGHT);bound = data.read(PREFLIGHT/'bindings.json')
    bindings,trained,rows,paths = bound_training(bound['origin'])
    if bindings != bound['bindings']:raise ValueError('training lineage binding changed')
    training,evaluation,environment,ancestor,cache,history_paths = current_history()
    if cache != data.read(PREFLIGHT/'judgment_cache.json'):
        raise ValueError('intervening first judgments require a new preserved evaluation lineage')
    paths += history_paths+[PREFLIGHT/n for n in ('producer.json','result.json','actual_tool_receipt.json','bindings.json','judgment_cache.json','split_proof.json')]
    if arm == 'expanded':
        completed(ROOTS['existing']['audit'])
        cache = history.parent.original.append_grade(cache,ROOTS['existing']['grade'],evaluation)
        paths += [ROOTS['existing'][s]/n for s in ('grade','audit') for n in ('producer.json','result.json','actual_tool_receipt.json')]
        paths += [ROOTS['existing']['grade']/n for n in ('judgment_cache.json','exact_packets.json')]
    ancestor.audit_cache(cache)
    lineage = SimpleNamespace(COUNT=len(cache),base_cache=lambda:deepcopy(cache),inputs=lambda:list(paths),
        configure=ancestor.configure,require_judges=ancestor.require_judges,
        uncertain_allowances=ancestor.uncertain_allowances,audit_cache=ancestor.audit_cache)
    evaluation.lineage = lineage;evaluation.RUNS = ROOTS[arm]
    evaluation.sources = lambda:sources(evaluation,rows)
    evaluation.freeze = lambda root,stage,extra:freeze(root,stage,arm,environment,
        [*paths,evaluation.old.SOURCE/'cases.json',evaluation.old.FEATURES/'manifest.json',*extra],len(cache))
    return training,evaluation,trained[arm],environment


def generate(arm,pilot):
    training,evaluation,trained,_ = configure(arm)
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 3060'):
        raise ValueError('physical GPU0 RTX3060 required; leave public RTX2060 trainer unchanged')
    cohorts,manifest,guard = evaluation.sources()
    chosen = evaluation.old.inference_probes(cohorts,manifest) if pilot else [r for c in COHORTS for r in cohorts[c]]
    stage = 'inference_pilot' if pilot else 'outputs';root = ROOTS[arm][stage]
    extra = [Path(manifest[r['id']]['path']) for r in chosen];before = {}
    if not pilot:
        completed(ROOTS[arm]['inference_pilot'])
        before = {r['id']:r for r in data.read(ROOTS[arm]['inference_pilot']/'outputs.json')}
        extra += [ROOTS[arm]['inference_pilot']/n for n in ('producer.json','result.json','actual_tool_receipt.json','outputs.json')]
    proof = evaluation.freeze(root,stage,extra);data.write(root/'split_proof.json',guard)
    torch.cuda.reset_peak_memory_stats();model = study.load_initial(training)
    saved = torch.load(trained['root']/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(saved['model'],strict=True);del saved
    model.eval().requires_grad_(False)
    expected = trained['result']['final_model_sha256']
    if (training.native.vendor.module_hash(model) != expected
            or training.prior.protected(model) != trained['result']['frozen_hashes']):
        raise ValueError('exact final model and unchanged reconstruction path required')
    tokenizer = training.native.tokenizer();outputs = []
    try:
        for row in chosen:
            output = evaluation.predict(model,row,manifest,tokenizer)
            if pilot and evaluation.predict(model,row,manifest,tokenizer) != output:
                raise ValueError('source-only inference does not repeat exactly')
            if row['id'] in before and before[row['id']] != output:raise ValueError('full output differs from pilot')
            outputs.append(output);study.broad.write_json(root/'outputs.json',outputs)
            if torch.cuda.max_memory_allocated() > study.CONFIG.peak_limit_bytes:raise ValueError('4.9GB inference allocation exceeded')
            if pilot or len(outputs)%16 == 0:print(json.dumps({'generated':len(outputs),'total':len(chosen)}),flush=True)
        if training.native.vendor.module_hash(model) != expected:raise ValueError('inference changed model weights')
        finish(root,proof,{'status':'complete_source_only_'+stage,'cases':len(outputs),
            'outputs_sha256':data.sha(root/'outputs.json'),'model_sha256':expected,
            'source_disjoint':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'task_accuracy':None})
    finally:model.close()


def summarize(scores):
    if set(scores) != set(COHORTS):raise ValueError('both fixed DEV cohorts required')
    ids = []
    for c in COHORTS:
        rows = scores[c]['secondary'];ids += [r['id'] for r in rows]
        if set(scores[c])-{'primary','secondary'}:raise ValueError('only qualified judge roles allowed')
        for role_rows in scores[c].values():
            if len(role_rows) != 64:raise ValueError('complete64 cases per cohort required')
            if any(type(r['valid']) is not bool or
                   (type(r['passed']) is not bool if r['valid'] else r['passed'] is not None)
                   for r in role_rows):raise ValueError('valid semantic decisions required')
    if len(set(ids)) != 128:raise ValueError('unique128 source cases required')
    valid = sum(r['valid'] for c in COHORTS for r in scores[c]['secondary'])
    passed = sum(r['passed'] is True for c in COHORTS for r in scores[c]['secondary'])
    admitted = valid == 128 and all(sum(r['passed'] is True for r in scores[c]['secondary']) >= 61 for c in COHORTS)
    joints = None
    if admitted:
        from experiments.structure_net.ncm_lm1b_native_development import joint_gate
        if any('primary' not in scores[c] for c in COHORTS):raise ValueError('admitted primary judgments required')
        joints = {c:joint_gate(scores[c]['secondary'],scores[c]['primary']) for c in COHORTS}
    elif any('primary' in scores[c] for c in COHORTS):raise ValueError('unadmitted primary judgments')
    qualified = admitted and all(v['qualified'] for v in joints.values())
    return {'valid':valid,'passed':passed,'task_accuracy':passed/128 if valid == 128 else None,
        'primary_admitted':admitted,'joint':joints,'DEV_qualified':qualified,
        'joint_passed':sum(v['joint_passed'] for v in joints.values()) if joints and all(v['all_valid'] for v in joints.values()) else None}


def compare(existing,expanded):
    summaries = {'existing':summarize(existing),'expanded':summarize(expanded)}
    counts = dict(both_pass=0,both_fail=0,existing_only=0,expanded_only=0,unresolved=0)
    for c in COHORTS:
        a,b = existing[c]['secondary'],expanded[c]['secondary']
        if [r['id'] for r in a] != [r['id'] for r in b]:raise ValueError('same ordered source cases required')
        for x,y in zip(a,b):
            key = ('unresolved' if not x['valid'] or not y['valid'] else
                   'both_pass' if x['passed'] and y['passed'] else 'existing_only' if x['passed'] else
                   'expanded_only' if y['passed'] else 'both_fail')
            counts[key] += 1
    delta = summaries['expanded']['passed']-summaries['existing']['passed'] if not counts['unresolved'] else None
    candidates = [a for a in study.ARMS if summaries[a]['DEV_qualified']]
    selected = max(candidates,key=lambda a:(summaries[a]['joint_passed'],a == 'existing')) if candidates else None
    return {'arms':summaries,'paired_Terra':counts,'additional_passes':delta,
        'descriptive_supervision_benefit':None if delta is None else delta >= 7,
        'selected_DEV_candidate':selected,'final_task_goal_achieved':False,'speed_tested':False}


def main():
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('offline','preflight',*STAGES))
    parser.add_argument('--arm',choices=study.ARMS);parser.add_argument('--bindings',type=Path)
    args = parser.parse_args()
    if args.stage == 'offline':
        if args.bindings is not None or args.arm is not None:parser.error('offline qualification takes no binding or arm')
        root = OFFLINE
    elif args.stage == 'preflight':
        if args.bindings is None or args.arm is not None:parser.error('preflight requires --bindings only')
        root = PREFLIGHT
    else:
        if args.arm is None or args.bindings is not None:parser.error('evaluation stage requires --arm only')
        root = ROOTS[args.arm][args.stage]
    if root.exists():raise ValueError('fresh append-only stage required')
    try:
        if args.stage == 'offline':offline()
        elif args.stage == 'preflight':preflight(args.bindings)
        elif args.stage in ('inference_pilot','outputs'):generate(args.arm,args.stage == 'inference_pilot')
        elif args.stage == 'audit':
            from experiments.neural_architecture_lab.audit_ncm_task_supervision_development import audit
            audit(args.arm)
        else:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '':raise ValueError('CPU-only semantic grading required')
            with (data.ROOT/'task_supervision_semantic_grade.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                _,evaluation,_,_ = configure(args.arm)
                evaluation.grade(args.stage == 'judge_pilot')
    except Exception as error:
        if root.exists() and not (root/'failure.json').exists():
            data.write(root/'failure.json',{'error_type':type(error).__name__,'error':str(error),
                'preserve_all_first_responses':True,'automatic_retry':False})
        raise


if __name__ == '__main__':main()
