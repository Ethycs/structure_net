"""Retain original and author-rendered probe judgments before epoch4 evaluation."""
import argparse
import json
import os
from pathlib import Path

from experiments.structure_net import ncm_public_probe_lineage as parent
from experiments.structure_net import ncm_xsum_author_semantics as author

data, study, probe = parent.data, parent.study, parent.probe
roots, STAGES = parent.roots, parent.STAGES
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-public-author-lineage.md')
TEST = Path('tests/structure_net/test_ncm_public_author_lineage.py')
PREFLIGHT = data.ROOT/'20261027_public_curriculum_author_lineage_preflight_v1'
INSTALLED = False


def branch_inputs():
    paths = [*parent.branch_inputs(), PLAN, TEST, Path(__file__)]
    for stage in ('prepare', 'pilot', 'grade', 'audit'):
        paths += [author.ROOTS[stage]/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    paths += [author.ROOTS['prepare']/'exact_packets.json', author.ROOTS['grade']/'judgment_cache.json']
    paths += [author.ROOTS['results']/n for n in ('campaign_results.json', 'actual_tool_receipt.json',
        'backend_exact_readback.json', 'backend_actual_tool_receipt.json')]
    return list(dict.fromkeys(paths))


def author_cache(inherited):
    if inherited != author.prior_cache():
        raise ValueError('author branch must follow the exact original probe cache')
    states = {}
    for stage in ('prepare', 'pilot', 'grade', 'audit'):
        states[stage] = author.base.require(author.ROOTS[stage])
        author.base.verify_proof(data.read(author.ROOTS[stage]/'producer.json'))
    graded, audited = states['grade'], states['audit']
    result_root = author.ROOTS['results']; campaign = result_root/'campaign_results.json'
    published = data.read(campaign)
    backend = data.read(result_root/'backend_exact_readback.json')
    for name, required in (('actual_tool_receipt.json', campaign),
                           ('backend_actual_tool_receipt.json', result_root/'backend_exact_readback.json')):
        receipt = data.read(result_root/name)
        if receipt['actual_tool_exit_code'] != 0 or receipt['artifacts'].get(str(required)) != data.sha(required):
            raise ValueError('successful actual author report and readback receipts required')
        data.verify(receipt['artifacts'])
    data.verify(published['artifacts'])
    if (backend['status'] != 'independent_registry_readback_passed'
            or backend['campaign_sha256'] != data.sha(campaign)
            or backend['report_sha256'] != data.sha(published['report'])
            or published['hypothesis_id'] != author.HYPOTHESIS
            or graded['status'] != 'author_semantics_grade_completed'
            or audited['status'] != 'author_semantic_firsts_replayed'
            or any(states[s]['inherited_first_judgments'] != len(inherited) for s in states)
            or any(published[k] != audited[k] for k in ('metrics', 'classification', 'canonical_first_judgments'))
            or graded['canonical_first_judgments'] != audited['canonical_first_judgments']):
        raise ValueError('completed audited and registered author branch required')
    cache = data.read(author.ROOTS['grade']/'judgment_cache.json')
    packets = data.read(author.ROOTS['prepare']/'exact_packets.json')
    value = parent.extend(inherited, cache, packets, graded['metrics'], audited['canonical_first_judgments'])
    author.qualified.audit_cache(value)
    return value


def branch_cache(inherited):
    return author_cache(parent.branch_cache(inherited))


def extension_records(original_count, author_count):
    if not 4691 <= original_count <= author_count:
        raise ValueError('ordered nondecreasing branch counts required')
    return [
        {'after_epoch': 2, 'hypothesis_id': probe.HYPOTHESIS, 'before_firsts': 4691,
         'after_firsts': original_count, 'added_firsts': original_count-4691,
         'included_in_application_accuracy': False},
        {'after_epoch': 2, 'hypothesis_id': author.HYPOTHESIS, 'before_firsts': original_count,
         'after_firsts': author_count, 'added_firsts': author_count-original_count,
         'included_in_application_accuracy': False}]


def install():
    global INSTALLED
    if INSTALLED:
        return
    parent.install()
    old_append, old_paths = parent.original.append_grade, parent.original.lineage_paths

    def append_grade(cache, root, evaluation):
        value = old_append(cache, root, evaluation)
        return author_cache(value) if root == roots(2)['grade'] else value

    def lineage_paths(previous_roots=()):
        paths = old_paths(previous_roots)
        return list(dict.fromkeys(paths + (branch_inputs() if roots(2)['grade'] in previous_roots else [])))

    parent.original.append_grade, parent.original.lineage_paths = append_grade, lineage_paths
    parent.original.PLAN, parent.original.TEST, parent.original.PREFLIGHT = PLAN, TEST, PREFLIGHT
    INSTALLED = True


def configure(epoch, attempt):
    if epoch not in (4, 8):
        raise ValueError('completed epoch1/2 evaluation must remain historical')
    install()
    return parent.original.configure(epoch, attempt)


def preflight():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or PREFLIGHT.exists():
        raise ValueError('fresh CPU-only lineage preflight required')
    # This replaces the unexecuted probe-only preflight, not the completed depth preflight.
    data.completed(parent.OLD_PREFLIGHT)
    data.verify(data.read(parent.OLD_PREFLIGHT/'producer.json')['input_hashes'])
    _, evaluation, environment = configure(4, 1)
    paths = [*evaluation.lineage.inputs(), evaluation.old.SOURCE/'cases.json',
             study.pipeline.broad.catalog.DEST/'cases.json', *branch_inputs()]
    hashes = data.start(PREFLIGHT, 'public_author_lineage_preflight', paths, environment,
        {'systems_only': True, 'new_judge_requests': 0, 'final_test_opened': False})
    cohorts, _, guard = evaluation.sources()
    cache = evaluation.lineage.base_cache()
    old = parent.branch_cache(probe.prior_cache())
    if sum(map(len, cohorts.values())) != 128 or cache != author_cache(old):
        raise ValueError('complete merged cache and fixed128 DEV sources required')
    data.verify(hashes)
    value = {'status': 'public_author_lineage_preflight_passed', 'cases': 128,
        'inherited_first_judgments': len(cache), 'lineage_extensions': extension_records(len(old), len(cache)),
        'source_guard': guard, 'new_judge_requests': 0, 'model_training': False,
        'task_accuracy': None, 'final_test_opened': False,
        'artifacts': {str(PREFLIGHT/'producer.json'): data.sha(PREFLIGHT/'producer.json')}}
    data.write(PREFLIGHT/'result.json', value); print(json.dumps(value), flush=True)


def require_preflight():
    state = author.base.require(PREFLIGHT)
    data.verify(data.read(PREFLIGHT/'producer.json')['input_hashes'])
    if state['status'] != 'public_author_lineage_preflight_passed':
        raise ValueError('completed author-preserving preflight required')
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('preflight', *STAGES))
    parser.add_argument('--epoch', type=int, choices=(4, 8), default=4)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    if args.stage == 'preflight':
        if (args.epoch, args.attempt) != (4, 1):
            parser.error('fixed prospective epoch4 lineage preflight required')
        preflight()
    else:
        require_preflight()
        _, evaluation, _ = configure(args.epoch, args.attempt)
        if roots(args.epoch)[args.stage].exists():
            raise ValueError('fresh append-only evaluation stage required')
        if args.stage in ('inference_pilot', 'outputs'):
            evaluation.generate(args.stage == 'inference_pilot')
        else:
            evaluation.grade(args.stage == 'judge_pilot')
