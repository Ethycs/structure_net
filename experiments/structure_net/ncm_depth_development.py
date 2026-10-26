"""Unchanged semantic DEV scoring for the matched generator-depth experiment."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from experiments.structure_net import ncm_depth_training as study
from experiments.structure_net import ncm_onpolicy_development_recovery as previous

STAGES = ('inference_pilot', 'outputs', 'judge_pilot', 'grade')
BASE = study.broad.runtime.ROOT/'20261025_onpolicy_residual_residual_targets_DEV_grade_v2'
BASE_SHA = 'cd9779b03401569aa9e975323fa3055e3ba721b694b17a1e96d8cdae6f37419a'


def roots(depth):
    if depth not in (12, 24):
        raise ValueError('fixed depth comparison required')
    return {s: study.broad.runtime.ROOT/f'20261026_ncm_depth_{depth}_DEV_{s}_v1' for s in STAGES}


def audit_training(training, depth):
    root = study.ROOTS[depth, 'train']
    training.preparation.completed(root)
    result = training.q.read(root/'result.json')
    proof = training.q.read(root/'producer.json')
    training.native.vendor.verify(proof)
    if (result['depth'] != depth or result['updates'] != 1024 or result['TRAIN_cases'] != 1376
            or not result['strict_reload_passed'] or not result['initial_predictions_identical']
            or study.broad.file_hash(root/'model.pt') != result['checkpoint_sha256']):
        raise ValueError('complete matched depth training required')
    rows = training.q.read(study.broad.RUNS['prepare']/'cases.json')
    sampler = torch.Generator().manual_seed(study.CONFIG.sampler_seed)
    noise = torch.Generator().manual_seed(study.CONFIG.corruption_seed)
    levels = (1, 8, 16, 24, 32, 40, 52, 64, 76, 92, 108, 128)
    permutation, seen = [], set()
    with (root/'trace.jsonl').open() as stream:
        trace = [json.loads(line) for line in stream]
    if len(trace) != 1024:
        raise ValueError('all fixed updates required')
    for step, item in enumerate(trace, 1):
        chosen = []
        for _ in range(4):
            if not permutation:
                permutation.extend(torch.randperm(len(rows), generator=sampler).tolist())
            chosen.append(rows[permutation.pop()])
        if item['step'] != step or item['lr'] != study.broad.rate(step, study.CONFIG):
            raise ValueError('update order or learning rate changed')
        for offset, micro in zip((0, 2), item['microbatches'], strict=True):
            selected = chosen[offset:offset+2]
            gold = torch.tensor([r['oracle_codes'] for r in selected])
            _, changes = study.batching.histories(gold, levels, 8192, noise, [False, True])
            if micro['ids'] != [r['id'] for r in selected] or micro['changes'] != changes:
                raise ValueError('sample/corruption schedule differs')
            seen.update(micro['ids'])
    if len(seen) != 1376:
        raise ValueError('complete reviewed training corpus was not visited')
    return {'depth': depth, 'updates': len(trace), 'distinct_TRAIN_cases': len(seen),
            'presentations': 4096, 'sample_and_corruption_schedule_replayed': True}


def configure(depth, *, preflight=False):
    study.deterministic.configure_runtime()
    training, evaluation, environment = previous.configure('residual_targets')
    ancestor, original_sources = evaluation.lineage, evaluation.sources
    sibling = roots(12)['grade'] if depth == 24 and not preflight else None

    def base_cache():
        if study.broad.file_hash(BASE/'judgment_cache.json') != BASE_SHA:
            raise ValueError('completed 4179-first semantic lineage changed')
        cache = previous.preserve_firsts(ancestor.base_cache(), training.q.read(BASE/'judgment_cache.json'),
                                        expected_count=4179)
        if sibling is not None:
            training.preparation.completed(sibling)
            result = training.q.read(sibling/'result.json')
            packets = training.q.read(sibling/'exact_packets.json')
            roles = ('secondary', 'primary') if result['primary_admitted'] else ('secondary',)
            allowed = {evaluation.old.oracle.first.packet_key(p, role) for c in evaluation.old.COHORTS
                       for p in packets[c]['student'] for role in roles}
            cache = previous.preserve_firsts(cache, training.q.read(sibling/'judgment_cache.json'),
                expected_count=result['canonical_first_judgments'], allowed_new=allowed)
        ancestor.audit_cache(cache)
        return cache

    paths = [BASE/n for n in ('result.json', 'actual_tool_receipt.json', 'judgment_cache.json')]
    if sibling is not None:
        paths += [sibling/n for n in ('result.json', 'actual_tool_receipt.json', 'judgment_cache.json', 'exact_packets.json')]
    inherited = len(base_cache())
    evaluation.lineage = SimpleNamespace(COUNT=inherited, base_cache=base_cache,
        inputs=lambda: [*ancestor.inputs(), *paths], configure=ancestor.configure,
        require_judges=ancestor.require_judges, uncertain_allowances=ancestor.uncertain_allowances,
        audit_cache=ancestor.audit_cache)

    def freeze(root, stage, extra):
        audits = [audit_training(training, d) for d in (12, 24)]
        files = {Path(__file__).resolve(), study.PLAN.resolve()}
        for module in tuple(sys.modules.values()):
            name = getattr(module, '__file__', None)
            if name:
                p = Path(name).resolve()
                if p.is_file() and p.suffix == '.py' and (p.is_relative_to(Path('experiments').resolve())
                        or p.is_relative_to(training.native.vendor.VENDOR.resolve())):
                    files.add(p)
        inputs = [evaluation.old.SOURCE/'cases.json', evaluation.old.FEATURES/'manifest.json',
                  study.broad.catalog.DEST/'cases.json', *evaluation.lineage.inputs(), *extra]
        inputs += [study.ROOTS[d, 'train']/n for d in (12, 24)
                   for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'model.pt', 'trace.jsonl')]
        root.mkdir(); (root/'producer_source').mkdir()
        for i, p in enumerate(sorted(files)):
            (root/'producer_source'/f'{i}_{p.name}').write_bytes(p.read_bytes())
        proof = {'schema': 'nal.ncm-depth-dev.v1', 'hypothesis_id': 'ncm-generator-depth-v1',
            'stage': stage, 'depth': depth, 'environment': environment,
            'config': {'source_only': True, 'DEV_cases': 128, 'rubric': training.q.RUBRIC,
                       'judges': training.q.JUDGES, 'inherited_firsts': inherited,
                       'metric': 'complete_summary_semantic_pass', 'stored_token_cap': 128,
                       'final_test_opened': False},
            'source_hashes': {str(p): study.broad.file_hash(p) for p in sorted(files)},
            'input_hashes': {str(p): study.broad.file_hash(p) for p in dict.fromkeys(inputs)}}
        proof['fingerprint'] = training.q.digest(proof)
        study.broad.write_json(root/'producer.json', proof)
        study.broad.write_json(root/'training_audit.json', audits)
        return proof

    # Only wrap the historical constructor, never the recursive BROAD loader.
    original_factory = training.load_initial
    def factory():
        model = original_factory()
        study.ncm_depth.expand(model, depth)
        return model
    training.load_initial = factory
    training.RUNS, training.PLAN = {'train': study.ROOTS[depth, 'train']}, study.PLAN
    evaluation.RUNS, evaluation.sources, evaluation.freeze = roots(depth), original_sources, freeze
    return training, evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('depth', type=int, choices=(12, 24))
    parser.add_argument('stage', choices=('preflight', *STAGES))
    args = parser.parse_args()
    training, evaluation = configure(args.depth, preflight=args.stage == 'preflight')
    if args.stage == 'preflight':
        cohorts, _, guard = evaluation.sources()
        print(json.dumps({'depth': args.depth, 'source_disjoint': guard,
                          'inherited_firsts': len(evaluation.lineage.base_cache()),
                          'DEV_cases': sum(map(len, cohorts.values())), 'new_judge_requests': 0}), flush=True)
    else:
        root = roots(args.depth)[args.stage]
        if root.exists():
            raise ValueError('fresh append-only depth DEV root required')
        if args.stage in ('inference_pilot', 'outputs'):
            evaluation.generate(args.stage == 'inference_pilot')
        else:
            evaluation.grade(args.stage == 'judge_pilot')
