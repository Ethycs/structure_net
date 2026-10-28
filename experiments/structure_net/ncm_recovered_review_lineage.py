"""Extend future model evaluation with audited recovered-label first decisions."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

from experiments.structure_net import ncm_recovered_label_review as review
from experiments.structure_net import ncm_public_author_lineage as author

data, task = review.data, review.history
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-recovered-review-lineage.md')
TEST = Path('tests/structure_net/test_ncm_recovered_review_lineage.py')
PREFLIGHT = data.ROOT/'20261027_recovered_review_lineage_preflight_v1'
INSTALLED = False


def preserve_union(inherited, current, allowed):
    review.protocol.preserve(inherited, current)
    if set(current) != set(inherited) | set(allowed):
        raise ValueError('all and only fixed recovered-label firsts required')
    return deepcopy(current)


def branch_inputs():
    paths = [Path(__file__), PLAN, TEST]
    for root in review.ROOTS.values():
        paths += [root/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    paths += [review.ROOTS['prepare']/n for n in ('cases.json', 'exact_packets.json', 'judgment_cache.json')]
    paths += [review.ROOTS['grade']/n for n in ('exact_packets.json', 'judgment_cache.json')]
    return paths


def extend(inherited):
    expected = data.read(review.ROOTS['prepare']/'judgment_cache.json')
    if inherited != expected:
        raise ValueError('exact completed author branch must precede label firsts')
    states = {}
    for stage, root in review.ROOTS.items():
        states[stage] = review.receipt(root)
        review.base.verify_proof(data.read(root/'producer.json'))
    if (states['grade']['status'] != 'recovered_saved_labels_graded'
            or states['audit']['status'] != 'recovered_saved_label_firsts_and_metrics_replayed'
            or states['grade']['metrics'] != states['audit']['metrics']
            or any(s['inherited_first_judgments'] != len(inherited) for s in states.values())):
        raise ValueError('complete audited label branch required')
    packets = data.read(review.ROOTS['prepare']/'exact_packets.json')
    if len(packets) != 128 or packets != data.read(review.ROOTS['grade']/'exact_packets.json'):
        raise ValueError('same fixed128 label packets required')
    allowed = {review.first.packet_key(p, role) for p in packets for role in review.ROLES}
    cache = preserve_union(inherited, data.read(review.ROOTS['grade']/'judgment_cache.json'), allowed)
    if len(cache) != states['grade']['canonical_first_judgments'] or len(cache) != states['audit']['canonical_first_judgments']:
        raise ValueError('audited canonical count differs')
    review.qualified.audit_cache(cache)
    return cache


def install():
    global INSTALLED
    if INSTALLED:
        return
    author.install()
    original = author.parent.original
    old_append, old_paths = original.append_grade, original.lineage_paths
    def append_grade(cache, root, evaluation):
        current = old_append(cache, root, evaluation)
        return extend(current) if root == author.roots(2)['grade'] else current
    def lineage_paths(previous_roots=()):
        paths = old_paths(previous_roots)
        if author.roots(2)['grade'] in previous_roots:
            paths = list(dict.fromkeys([*paths, *branch_inputs()]))
        return paths
    original.append_grade, original.lineage_paths = append_grade, lineage_paths
    original.PLAN, original.TEST, original.PREFLIGHT = PLAN, TEST, PREFLIGHT
    INSTALLED = True


def preflight():
    if PREFLIGHT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('fresh CPU-only lineage preflight required')
    review.configure(); author.require_preflight(); install()
    _, evaluation, environment, _, cache, paths = task.current_history()
    cohorts, _ = evaluation.old.source_inputs()
    if sum(map(len, cohorts.values())) != 128:
        raise ValueError('unchanged128 source-only DEV cases required')
    grade = data.read(review.ROOTS['grade']/'result.json')
    if len(cache) != grade['canonical_first_judgments']:
        raise ValueError('unexpected intervening branch; preserve it before proceeding')
    proof = review.freeze(PREFLIGHT, 'post_recovered_label_lineage_preflight',
        [*paths, *branch_inputs(), evaluation.old.SOURCE/'cases.json'], len(cache))
    data.write(PREFLIGHT/'judgment_cache.json', cache)
    review.finish(PREFLIGHT, proof, {'status': 'recovered_label_lineage_preflight_passed',
        'cases': 128, 'canonical_first_judgments': len(cache), 'new_judge_requests': 0,
        'label_firsts_included_in_application_accuracy': False, 'environment': environment})


def require_preflight():
    state = review.receipt(PREFLIGHT)
    review.base.verify_proof(data.read(PREFLIGHT/'producer.json'))
    if state['status'] != 'recovered_label_lineage_preflight_passed':
        raise ValueError('completed recovered-label lineage preflight required')
    return state


def public(stage, epoch, attempt):
    if epoch not in (4, 8):
        raise ValueError('earlier public milestones remain historical')
    require_preflight(); install(); author.require_preflight()
    _, evaluation, _ = author.configure(epoch, attempt)
    if author.roots(epoch)[stage].exists():
        raise ValueError('fresh public evaluation stage required')
    if stage in ('inference_pilot', 'outputs'):
        evaluation.generate(stage == 'inference_pilot')
    else:
        evaluation.grade(stage == 'judge_pilot')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('preflight', 'public', 'task'))
    parser.add_argument('--stage', choices=author.STAGES)
    parser.add_argument('--epoch', type=int, choices=(4, 8), default=4)
    parser.add_argument('--attempt', type=int, default=1)
    args, remaining = parser.parse_known_args()
    if args.action == 'preflight':
        if remaining or args.stage: parser.error('preflight takes no stage')
        preflight()
    elif args.action == 'public':
        if remaining or not args.stage: parser.error('public requires --stage')
        public(args.stage, args.epoch, args.attempt)
    else:
        # Forward the original task CLI unchanged after installing its ancestry.
        require_preflight(); install()
        sys.argv = [task.__file__, *remaining]
        task.main()
