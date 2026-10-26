"""Use the existing semantic DEV protocol for fixed public-training milestones."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from experiments.structure_net import ncm_public_training as study
from experiments.structure_net import ncm_onpolicy_development_recovery as previous

data = study.data
STAGES = ('inference_pilot', 'outputs', 'judge_pilot', 'grade')
BASE = data.ROOT/'20261025_onpolicy_residual_residual_targets_DEV_grade_v2'
BASE_SHA = 'cd9779b03401569aa9e975323fa3055e3ba721b694b17a1e96d8cdae6f37419a'


def roots(epoch):
    if epoch not in study.CONFIG.milestones:
        raise ValueError('fixed public-training milestone required')
    return {s: data.ROOT/f'20261026_public_curriculum_epoch{epoch}_DEV_{s}_v1' for s in STAGES}


def public_guard(evaluation, cohorts):
    catalog = study.pipeline.bundle.catalog
    data.completed(catalog.AUDIT)
    data.completed(catalog.INSPECTION_AUDIT)
    data.completed(catalog.DEST)
    data.completed(catalog.INSPECT)
    inspection = data.read(catalog.INSPECT/'producer.json')
    current_sources = evaluation.old.SOURCE/'cases.json'
    if inspection['input_hashes'].get(str(current_sources)) != data.sha(current_sources):
        raise ValueError('public text-overlap screening must bind these exact DEV sources')
    state = data.read(catalog.INSPECT/'result.json')
    if (not state['open_source_and_protected_exact_hash_checks_completed']
            or state['counts']['rows'] != 204045):
        raise ValueError('completed exhaustive public screen required')
    held = [r for c in evaluation.old.COHORTS for r in cohorts[c]]
    keys = ('id', 'source_id', 'group_id', 'source_sha256', 'source_component')
    protected = {key: {r[key] for r in held} for key in keys}
    count = 0
    with (catalog.DEST/'training_index.jsonl').open() as stream:
        for entry in map(json.loads, stream):
            family = 'xsum_'+catalog.REVISION+'_train_'+entry['public_provenance']['article_id']
            values = {'id': entry['id'], 'source_sha256': entry['source_sha256'],
                      'source_id': family, 'group_id': family, 'source_component': family}
            if any(values[k] in protected[k] for k in keys):
                raise ValueError('public TRAIN/DEV source identity overlap')
            count += 1
    if count != study.CONFIG.public_cases:
        raise ValueError('every admitted public source must enter the DEV guard')
    return {'public_cases': count, 'DEV_cases': len(held), 'identity_overlaps': 0,
            'audited_open_text_overlap_screen_reused': True,
            'text_screen_DEV_source_sha256': data.sha(current_sources),
            'text_rule': 'normalized equality or32distinct5grams and80percent shorter containment',
            'sealed_final_full_text_comparison': False}


def configure(epoch, attempt):
    study.pipeline.determinism.configure_runtime()
    training, evaluation, environment = previous.configure('residual_targets')
    ancestor = evaluation.lineage
    catalog = study.pipeline.bundle.catalog
    final_root = study.root_for(epoch, attempt)
    previous_roots = [roots(e)['grade'] for e in study.CONFIG.milestones if e < epoch]

    def base_cache():
        inherited = ancestor.base_cache()
        if data.sha(BASE/'judgment_cache.json') != BASE_SHA:
            raise ValueError('completed4179-first lineage changed')
        cache = previous.preserve_firsts(inherited, data.read(BASE/'judgment_cache.json'), expected_count=4179)
        for root in previous_roots:
            result = data.completed(root)
            packets = data.read(root/'exact_packets.json')
            roles = ('secondary', 'primary') if result['primary_admitted'] else ('secondary',)
            allowed = {evaluation.old.oracle.first.packet_key(p, role) for c in evaluation.old.COHORTS
                       for p in packets[c]['student'] for role in roles}
            cache = previous.preserve_firsts(cache, data.read(root/'judgment_cache.json'),
                expected_count=result['canonical_first_judgments'], allowed_new=allowed)
        ancestor.audit_cache(cache)
        return cache

    inherited_count = len(base_cache())
    lineage_paths = [BASE/name for name in ('result.json', 'actual_tool_receipt.json', 'judgment_cache.json')]
    for root in previous_roots:
        lineage_paths += [root/name for name in ('result.json', 'actual_tool_receipt.json',
                                               'judgment_cache.json', 'exact_packets.json')]
    evaluation.lineage = SimpleNamespace(COUNT=inherited_count, base_cache=base_cache,
        inputs=lambda: [*ancestor.inputs(), *lineage_paths], configure=ancestor.configure,
        require_judges=ancestor.require_judges, uncertain_allowances=ancestor.uncertain_allowances,
        audit_cache=ancestor.audit_cache)

    def sources():
        cohorts, manifest = evaluation.old.source_inputs()
        anchor_guard = evaluation.split_guard(data.read(study.pipeline.broad.catalog.DEST/'cases.json'), cohorts)
        return cohorts, manifest, {'reviewed': anchor_guard, 'public': public_guard(evaluation, cohorts)}

    def freeze(root, stage, extra):
        state = data.completed(final_root)
        if (state['status'] != 'public_curriculum_milestone_training_completed' or state['epoch'] != epoch
                or state['updates'] != epoch * study.CONFIG.updates_per_epoch or not state['strict_reload_passed']):
            raise ValueError('complete fixed public-training milestone required')
        files = {Path(__file__).resolve(), study.pipeline.PLAN.resolve(),
                 Path('tests/structure_net/test_ncm_public_development.py').resolve()}
        for module in tuple(sys.modules.values()):
            name = getattr(module, '__file__', None)
            if name:
                p = Path(name).resolve()
                if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                        or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                    files.add(p)
        paths = [final_root/name for name in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'model.pt')]
        paths += [catalog.AUDIT/name for name in ('result.json', 'actual_tool_receipt.json')]
        paths += [catalog.INSPECTION_AUDIT/name for name in ('result.json', 'actual_tool_receipt.json')]
        paths += [catalog.INSPECT/'producer.json', catalog.INSPECT/'result.json', catalog.DEST/'training_index.jsonl',
            evaluation.old.SOURCE/'cases.json', evaluation.old.FEATURES/'manifest.json',
            study.pipeline.broad.catalog.DEST/'cases.json', *evaluation.lineage.inputs(), *extra]
        root.mkdir(); (root/'producer_source').mkdir()
        for i, p in enumerate(sorted(files)):
            (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
        proof = {'schema': 'nal.ncm-public-curriculum-dev.v1', 'stage': stage, 'epoch': epoch,
            'hypothesis_id': 'ncm-public-curriculum-v1', 'config': {'source_only': True,
                'target_fields_in_prediction': False, 'DEV_cases': 128, 'public_TRAIN_cases': 162353,
                'reviewed_TRAIN_cases': 1376, 'rubric': training.q.RUBRIC, 'judges': training.q.JUDGES,
                'inherited_first_judgments': inherited_count, 'task_metric': 'complete_summary_semantic_pass',
                'stored_token_cap': 128, 'final_test_opened': False, 'speed_tested': False},
            'environment': environment, 'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
            'input_hashes': {str(p): data.sha(p) for p in paths}}
        proof['fingerprint'] = training.q.digest(proof)
        data.write(root/'producer.json', proof)
        return proof

    training.RUNS, training.PLAN = {'train': final_root}, study.pipeline.PLAN
    evaluation.RUNS, evaluation.sources, evaluation.freeze = roots(epoch), sources, freeze
    return training, evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('preflight', *STAGES))
    parser.add_argument('--epoch', type=int, choices=study.CONFIG.milestones, default=1)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    training, evaluation = configure(args.epoch, args.attempt)
    if args.stage == 'preflight':
        cohorts, _, guard = evaluation.sources()
        print(json.dumps({'status': 'public_curriculum_DEV_preflight_passed', 'source_guard': guard,
            'cases': sum(map(len, cohorts.values())), 'inherited_firsts': len(evaluation.lineage.base_cache()),
            'new_judge_requests': 0, 'final_test_opened': False}), flush=True)
    else:
        if roots(args.epoch)[args.stage].exists():
            raise ValueError('fresh append-only semantic evaluation stage required')
        if args.stage in ('inference_pilot', 'outputs'):
            evaluation.generate(args.stage == 'inference_pilot')
        else:
            evaluation.grade(args.stage == 'judge_pilot')
