"""Unchanged whole-summary DEV evaluation for both matched on-policy arms."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

from experiments.structure_net import ncm_onpolicy_residual_training as study
from experiments.structure_net import audit_ncm_onpolicy_training as audit
from experiments.structure_net import ncm_broad_generator_development as previous

STAGES = ('inference_pilot','outputs','judge_pilot','grade')
ROOTS = {a:{s:study.broad.runtime.ROOT/f'20261025_onpolicy_residual_{a}_DEV_{s}_v1'
            for s in STAGES} for a in study.objective.ARMS}
BASE_CACHE = previous.ROOTS['grade']/'judgment_cache.json'
BASE_CACHE_SHA = 'a7f48e32cf446804974bf5f9d4cce1985249f43ebfd80c9e9ea1837389786fb8'
_BOOT = None


def preserve_firsts(inherited, current, *, expected_count=None, allowed_new=None):
    if expected_count is not None and len(current) != expected_count:
        raise ValueError('canonical first-judgment count differs')
    if any(current.get(k) != v for k,v in inherited.items()):
        raise ValueError('canonical first judgment changed or disappeared')
    if allowed_new is not None and set(current)-set(inherited)-set(allowed_new):
        raise ValueError('new judgment outside fixed arm packets')
    return current


def completed_training_audits(training):
    paths = []
    for arm in study.objective.ARMS:
        training.preparation.completed(study.ROOTS[arm,'train'])
        state = training.q.read(study.ROOTS[arm,'train']/'result.json')
        if state['updates'] != 1024 or state['TRAIN_cases'] != 1376 or not state['oracle_unchanged']:
            raise ValueError('both complete matched training arms required before DEV')
        value = audit.validate_audit(arm,training)
        if value['updates'] != 1024 or value['cases'] != 1376 or value['onpolicy_presentations'] != 2048:
            raise ValueError('full training-state audit required')
        paths += [audit.ROOTS[arm,'train']/n for n in ('campaign_results.json','actual_tool_receipt.json')]
    root = audit.ROOTS['residual_targets','label']
    if training.q.read(root/'actual_tool_receipt.json')['actual_tool_exit_code'] != 0:
        raise ValueError('actual completed residual-label replay required')
    value = training.q.read(root/'campaign_results.json')
    if value['status'] != 'all_recorded_onpolicy_residual_labels_replayed' or value['histories'] != 2048:
        raise ValueError('all2048 recorded generated-history labels must be replayed')
    for p,sha in value['artifacts'].items():
        if study.broad.file_hash(p) != sha: raise ValueError('audited residual labels changed')
    paths += [root/n for n in ('campaign_results.json','actual_tool_receipt.json')]
    return paths


def configure(arm, *, preflight=False):
    global _BOOT
    if arm not in study.objective.ARMS: raise ValueError('fixed matched arm required')
    # Snapshot the ancestor once; configuring the sibling in the same finalizer
    # must not recursively wrap a lineage that has already been extended.
    if _BOOT is None:
        training,evaluation,environment = previous.configure()
        _BOOT = training,evaluation,environment,evaluation.lineage,evaluation.sources
    training,evaluation,environment,ancestor,sources = _BOOT
    q = training.q
    base_paths = [BASE_CACHE,study.PARENT_RESULTS/'campaign_results.json',
        study.PARENT_RESULTS/'backend_exact_readback.json',study.PARENT_RESULTS/'backend_actual_tool_receipt.json']

    def original_cache():
        if (study.broad.file_hash(BASE_CACHE) != BASE_CACHE_SHA
                or study.broad.file_hash(study.PARENT_RESULTS/'campaign_results.json') != study.PARENT_RESULTS_SHA):
            raise ValueError('completed3923-first broad-generator lineage changed')
        cache = preserve_firsts(ancestor.base_cache(),q.read(BASE_CACHE),expected_count=3923)
        ancestor.audit_cache(cache)
        return cache

    sibling_paths = []
    sibling = arm == 'residual_targets' and not preflight
    if sibling:
        root = ROOTS['fixed_targets']['grade']
        training.preparation.completed(root)
        sibling_paths = [root/n for n in ('producer.json','result.json','actual_tool_receipt.json',
            'judgment_cache.json','exact_packets.json','cases.json')]

    def base_cache():
        base = original_cache()
        if not sibling: return base
        root = ROOTS['fixed_targets']['grade']
        result = q.read(root/'result.json')
        packets = q.read(root/'exact_packets.json')
        roles = ('secondary','primary') if result['primary_admitted'] else ('secondary',)
        allowed = {evaluation.old.oracle.first.packet_key(packet,role)
                   for c in evaluation.old.COHORTS for packet in packets[c]['student'] for role in roles}
        cache = preserve_firsts(base,q.read(root/'judgment_cache.json'),
                               expected_count=result['canonical_first_judgments'],allowed_new=allowed)
        ancestor.audit_cache(cache)
        return cache

    inherited_count = len(base_cache())
    def lineage_inputs():
        return [*ancestor.inputs(),*base_paths,*sibling_paths]
    evaluation.lineage = SimpleNamespace(COUNT=inherited_count,base_cache=base_cache,inputs=lineage_inputs,
        configure=ancestor.configure,require_judges=ancestor.require_judges,
        uncertain_allowances=ancestor.uncertain_allowances,audit_cache=ancestor.audit_cache)

    def freeze(root,stage,extra):
        audit_paths = completed_training_audits(training)
        files = {Path(__file__),study.PLAN,Path(study.__file__),Path(evaluation.__file__)}
        for module in tuple(sys.modules.values()):
            name = getattr(module,'__file__',None)
            if name:
                p = Path(name).resolve()
                if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                        or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                    files.add(p)
        paths = [evaluation.old.SOURCE/'cases.json',evaluation.old.FEATURES/'manifest.json',
            study.broad.catalog.DEST/'cases.json',*audit_paths,*extra]
        paths += [study.ROOTS[arm,'train']/n for n in ('producer.json','result.json','actual_tool_receipt.json','model.pt')]
        proof = {'schema':'nal.ncm-onpolicy-dev.v1','hypothesis_id':study.HYPOTHESIS,'arm':arm,'stage':stage,
            'config':{'source_only':True,'target_fields_in_prediction':False,'full_source_features':True,
                'source_disjoint_DEV_cases':128,'cohorts':evaluation.old.COHORTS,'rubric':q.RUBRIC,
                'judges':q.JUDGES,'inherited_judgments':inherited_count,
                'task_accuracy_metric':'whole_summary_semantic_pass','stored_token_cap':128,
                'exact_match_task_metric':False,'final_test_opened':False,'speed_tested':False},
            'environment':environment,'source_hashes':{str(p):study.broad.file_hash(p) for p in sorted(files)},
            'input_hashes':{str(p):study.broad.file_hash(p) for p in dict.fromkeys(paths)}}
        proof['fingerprint'] = q.digest(proof)
        root.mkdir();(root/'producer_source').mkdir()
        for i,p in enumerate(sorted(files)):
            (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
        study.broad.write_json(root/'producer.json',proof)
        return proof

    # The old constructor is unchanged. generate() strictly loads the selected
    # complete checkpoint afterward; replacing load_initial would recurse.
    training.RUNS,training.PLAN = {'train':study.ROOTS[arm,'train']},study.PLAN
    evaluation.RUNS,evaluation.sources,evaluation.freeze = ROOTS[arm],sources,freeze
    return training,evaluation,environment


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('arm',choices=study.objective.ARMS)
    parser.add_argument('stage',choices=('preflight',*STAGES))
    args = parser.parse_args()
    training,evaluation,_ = configure(args.arm,preflight=args.stage == 'preflight')
    if args.stage == 'preflight':
        cohorts,_,guard = evaluation.sources()
        print(study.json.dumps({'arm':args.arm,'source_disjoint':guard,'base_prior_firsts':len(evaluation.lineage.base_cache()),
            'DEV_cases':sum(len(v) for v in cohorts.values()),'new_requests':0,'final_test_opened':False,
            'sibling_cache_pending':args.arm == 'residual_targets'}),flush=True)
        return
    if ROOTS[args.arm][args.stage].exists(): raise ValueError('fresh append-only DEV root required')
    completed_training_audits(training)
    if args.stage in ('inference_pilot','outputs'): evaluation.generate(args.stage == 'inference_pilot')
    else: evaluation.grade(args.stage == 'judge_pilot')


if __name__ == '__main__':
    main()
