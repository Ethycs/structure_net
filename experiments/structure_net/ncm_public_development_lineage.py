"""Public milestone evaluation retaining all completed depth-study firsts."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from experiments.structure_net import ncm_public_development as original

study, data, previous = original.study, original.data, original.previous
STAGES, roots = original.STAGES, original.roots
PLAN = Path('docs/07 - Status Reports/2026-10-26_ncm-public-evaluation-lineage.md')
TEST = Path('tests/structure_net/test_ncm_public_development_lineage.py')
PREFLIGHT = data.ROOT/'20261026_public_curriculum_lineage_preflight_v2'
DEPTH_RESULTS = data.ROOT/'20261026_ncm_depth_results_v1'
DEPTH_CAMPAIGN_SHA = '2c6e762dc36e2da0ce3e6128a2e5eb7caa9f801a755465eb4f870dc94f6e1268'
DEPTH_CACHE_SHA = 'fe9aa60771cd22ddeb8478de3b656894a27a51a7824acbbf4eb0809b25ad8ba6'
DEPTH_ROOTS = tuple(data.ROOT/f'20261026_ncm_depth_{d}_DEV_grade_v1' for d in (12, 24))
GRADE_FILES = ('producer.json', 'result.json', 'actual_tool_receipt.json',
               'judgment_cache.json', 'exact_packets.json')


def append_grade(cache, root, evaluation):
    result = data.completed(root)
    receipt = data.read(root/'actual_tool_receipt.json')
    if any(receipt['artifacts'].get(str(root/n)) != data.sha(root/n)
           for n in ('producer.json', 'result.json')):
        raise ValueError('actual receipt must bind the completed grade producer and result')
    if result['inherited_first_judgments'] != len(cache):
        raise ValueError('completed grade must extend the full preceding lineage')
    packets = data.read(root/'exact_packets.json')
    if (set(packets) != set(evaluation.old.COHORTS)
            or any(set(packets[c]) != {'student'} or len(packets[c]['student']) != 64
                   for c in evaluation.old.COHORTS)):
        raise ValueError('both fixed DEV cohorts required')
    roles = ('secondary', 'primary') if result['primary_admitted'] else ('secondary',)
    allowed = {evaluation.old.oracle.first.packet_key(p, role)
               for c in evaluation.old.COHORTS for p in packets[c]['student'] for role in roles}
    return previous.preserve_firsts(cache, data.read(root/'judgment_cache.json'),
        expected_count=result['canonical_first_judgments'], allowed_new=allowed)


def depth_cache(inherited, evaluation):
    campaign = DEPTH_RESULTS/'campaign_results.json'
    if data.sha(campaign) != DEPTH_CAMPAIGN_SHA:
        raise ValueError('completed depth campaign changed')
    receipt = data.read(DEPTH_RESULTS/'actual_tool_receipt.json')
    if (receipt['actual_tool_exit_code'] != 0
            or receipt['artifacts'].get(str(campaign)) != DEPTH_CAMPAIGN_SHA):
        raise ValueError('completed depth aggregate needs its actual successful receipt')
    data.verify(receipt['artifacts'])
    recorded = data.read(campaign)
    data.verify(recorded['artifacts'])
    if data.sha(original.BASE/'judgment_cache.json') != original.BASE_SHA:
        raise ValueError('original 4179-first cache changed')
    cache = previous.preserve_firsts(inherited,
        data.read(original.BASE/'judgment_cache.json'), expected_count=4179)
    for root in DEPTH_ROOTS:
        cache = append_grade(cache, root, evaluation)
    if (len(cache) != 4435 or recorded['canonical_first_judgments'] != len(cache)
            or data.sha(DEPTH_ROOTS[-1]/'judgment_cache.json') != DEPTH_CACHE_SHA):
        raise ValueError('all 4435 completed first judgments required')
    return cache


def lineage_paths(previous_roots=()):
    paths = [original.BASE/n for n in ('result.json', 'actual_tool_receipt.json', 'judgment_cache.json')]
    paths += [DEPTH_RESULTS/n for n in ('campaign_results.json', 'actual_tool_receipt.json')]
    for root in (*DEPTH_ROOTS, *previous_roots):
        paths += [root/n for n in GRADE_FILES]
    return paths


def configure(epoch, attempt):
    study.pipeline.determinism.configure_runtime()
    training, evaluation, environment = previous.configure('residual_targets')
    ancestor = evaluation.lineage
    catalog = study.pipeline.bundle.catalog
    final_root = study.root_for(epoch, attempt)
    previous_roots = [roots(e)['grade'] for e in study.CONFIG.milestones if e < epoch]

    def base_cache():
        cache = depth_cache(ancestor.base_cache(), evaluation)
        for root in previous_roots:
            cache = append_grade(cache, root, evaluation)
        ancestor.audit_cache(cache)
        return cache

    inherited_count = len(base_cache())
    evaluation.lineage = SimpleNamespace(COUNT=inherited_count, base_cache=base_cache,
        inputs=lambda: [*ancestor.inputs(), *lineage_paths(previous_roots)], configure=ancestor.configure,
        require_judges=ancestor.require_judges, uncertain_allowances=ancestor.uncertain_allowances,
        audit_cache=ancestor.audit_cache)

    def sources():
        cohorts, manifest = evaluation.old.source_inputs()
        reviewed = evaluation.split_guard(data.read(study.pipeline.broad.catalog.DEST/'cases.json'), cohorts)
        return cohorts, manifest, {'reviewed': reviewed, 'public': original.public_guard(evaluation, cohorts)}

    def freeze(root, stage, extra):
        state = data.completed(final_root)
        if (state['status'] != 'public_curriculum_milestone_training_completed' or state['epoch'] != epoch
                or state['updates'] != epoch*study.CONFIG.updates_per_epoch or not state['strict_reload_passed']):
            raise ValueError('complete fixed public-training milestone required')
        files = {Path(__file__).resolve(), PLAN.resolve(), TEST.resolve(), study.pipeline.PLAN.resolve(),
                 Path('tests/structure_net/test_ncm_public_development.py').resolve()}
        for module in tuple(sys.modules.values()):
            name = getattr(module, '__file__', None)
            if name:
                p = Path(name).resolve()
                if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                        or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                    files.add(p)
        paths = [final_root/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'model.pt')]
        paths += [PREFLIGHT/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
        paths += [catalog.AUDIT/n for n in ('result.json', 'actual_tool_receipt.json')]
        paths += [catalog.INSPECTION_AUDIT/n for n in ('result.json', 'actual_tool_receipt.json')]
        paths += [catalog.INSPECT/'producer.json', catalog.INSPECT/'result.json', catalog.DEST/'training_index.jsonl',
            evaluation.old.SOURCE/'cases.json', evaluation.old.FEATURES/'manifest.json',
            study.pipeline.broad.catalog.DEST/'cases.json', *evaluation.lineage.inputs(), *extra]
        root.mkdir(); (root/'producer_source').mkdir()
        for i, p in enumerate(sorted(files)):
            (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
        proof = {'schema': 'nal.ncm-public-curriculum-dev.v2', 'stage': stage, 'epoch': epoch,
            'hypothesis_id': 'ncm-public-curriculum-v1', 'config': {'source_only': True,
                'target_fields_in_prediction': False, 'DEV_cases': 128, 'public_TRAIN_cases': 162353,
                'reviewed_TRAIN_cases': 1376, 'rubric': training.q.RUBRIC, 'judges': training.q.JUDGES,
                'inherited_first_judgments': inherited_count, 'completed_depth_firsts': 4435,
                'task_metric': 'complete_summary_semantic_pass', 'stored_token_cap': 128,
                'final_test_opened': False, 'speed_tested': False}, 'environment': environment,
            'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
            'input_hashes': {str(p): data.sha(p) for p in dict.fromkeys(paths)}}
        proof['fingerprint'] = training.q.digest(proof)
        data.write(root/'producer.json', proof)
        return proof

    training.RUNS, training.PLAN = {'train': final_root}, study.pipeline.PLAN
    evaluation.RUNS, evaluation.sources, evaluation.freeze = roots(epoch), sources, freeze
    return training, evaluation, environment


def preflight():
    if PREFLIGHT.exists():
        raise ValueError('fresh lineage preflight required')
    training, evaluation, environment = configure(1, 1)
    inputs = [PLAN, TEST, *evaluation.lineage.inputs(), evaluation.old.SOURCE/'cases.json',
              study.pipeline.broad.catalog.DEST/'cases.json']
    catalog = study.pipeline.bundle.catalog
    for root in (catalog.AUDIT, catalog.INSPECTION_AUDIT, catalog.DEST, catalog.INSPECT):
        inputs += [root/n for n in ('result.json', 'actual_tool_receipt.json')]
    inputs += [catalog.INSPECT/'producer.json', catalog.DEST/'training_index.jsonl']
    hashes = data.start(PREFLIGHT, 'public_evaluation_lineage_preflight', inputs, environment,
        {'systems_only': True, 'expected_firsts': 4435, 'new_judge_requests': 0, 'final_test_opened': False})
    cohorts, _, guard = evaluation.sources()
    cache = evaluation.lineage.base_cache()
    if len(cache) != 4435 or sum(map(len, cohorts.values())) != 128:
        raise ValueError('complete 4435-first lineage and128 DEV cases required')
    data.verify(hashes)
    result = {'status': 'public_evaluation_depth_lineage_preflight_passed', 'cases': 128,
        'inherited_first_judgments': len(cache), 'source_guard': guard,
        'new_judge_requests': 0, 'model_training': False, 'task_accuracy': None,
        'final_test_opened': False, 'artifacts': {str(PREFLIGHT/'producer.json'): data.sha(PREFLIGHT/'producer.json')}}
    data.write(PREFLIGHT/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('preflight', *STAGES))
    parser.add_argument('--epoch', type=int, choices=study.CONFIG.milestones, default=1)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if args.stage == 'preflight':
        if args.epoch != 1 or args.attempt != 1:
            parser.error('prospective lineage preflight uses epoch1 attempt1')
        preflight()
    else:
        state = data.completed(PREFLIGHT)
        if state['status'] != 'public_evaluation_depth_lineage_preflight_passed':
            raise ValueError('successful lineage preflight required')
        training, evaluation, _ = configure(args.epoch, args.attempt)
        if roots(args.epoch)[args.stage].exists():
            raise ValueError('fresh append-only semantic evaluation stage required')
        if args.stage in ('inference_pilot', 'outputs'):
            evaluation.generate(args.stage == 'inference_pilot')
        else:
            evaluation.grade(args.stage == 'judge_pilot')
