"""Insert the completed XSum probe after epoch2 without changing past firsts."""
import argparse
import json
import os
from pathlib import Path

from experiments.structure_net import ncm_public_development_lineage as original
from experiments.structure_net import ncm_xsum_semantic_probe as probe

data, study, previous = original.data, original.study, original.previous
roots, STAGES = original.roots, original.STAGES
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-public-probe-lineage.md')
TEST = Path('tests/structure_net/test_ncm_public_probe_lineage.py')
PREFLIGHT = data.ROOT/'20261027_public_curriculum_probe_lineage_preflight_v1'
OLD_PREFLIGHT, OLD_PLAN, OLD_TEST = original.PREFLIGHT, original.PLAN, original.TEST
OLD_APPEND, OLD_PATHS = original.append_grade, original.lineage_paths
INSTALLED = False


def extend(inherited, cache, packets, metrics, count):
    """Reject partial branches, replaced firsts and unadmitted extra calls."""
    if set(packets) != set(probe.ARMS) or set(metrics) != set(probe.ARMS):
        raise ValueError('all four fixed arms required')
    allowed = set()
    for arm in probe.ARMS:
        if len(packets[arm]) != 128:
            raise ValueError('all128 probe packets required')
        secondary = metrics[arm]['secondary']
        if secondary is None or secondary['cases'] != 128:
            raise ValueError('complete secondary population required')
        primary = metrics[arm]['primary']
        if (primary is not None) != probe.qualifies(secondary):
            raise ValueError('exact primary admission required')
        roles = ('secondary', 'primary') if primary is not None else ('secondary',)
        for role in roles:
            allowed.update(probe.first.packet_key(p, role) for p in packets[arm])
    value = previous.preserve_firsts(inherited, cache, expected_count=count, allowed_new=allowed)
    if set(value)-set(inherited) != allowed-set(inherited):
        raise ValueError('every admitted first decision required')
    return value


def branch_inputs():
    paths = [PLAN, TEST, Path(__file__), OLD_PLAN, OLD_TEST]
    paths += [OLD_PREFLIGHT/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    for stage in ('prepare','pilot','grade','audit'):
        paths += [probe.ROOTS[stage]/n for n in ('producer.json','result.json','actual_tool_receipt.json')]
    paths += [probe.ROOTS['prepare']/'exact_packets.json', probe.ROOTS['grade']/'judgment_cache.json']
    paths += [probe.ROOTS['results']/n for n in ('campaign_results.json','actual_tool_receipt.json',
        'backend_exact_readback.json','backend_actual_tool_receipt.json')]
    return paths


def branch_cache(inherited):
    probe.qualified.configure()
    if inherited != probe.prior_cache():
        raise ValueError('probe must be inserted immediately after the exact epoch2 ancestor')
    graded = probe.base.require(probe.ROOTS['grade'])
    audited = probe.base.require(probe.ROOTS['audit'])
    for stage in ('prepare','pilot','grade','audit'):
        probe.base.require(probe.ROOTS[stage])
        probe.base.verify_proof(data.read(probe.ROOTS[stage]/'producer.json'))
    result_root = probe.ROOTS['results']; campaign = result_root/'campaign_results.json'
    published = data.read(campaign)
    receipt = data.read(result_root/'actual_tool_receipt.json')
    backend = data.read(result_root/'backend_exact_readback.json')
    backend_receipt = data.read(result_root/'backend_actual_tool_receipt.json')
    for value, required in ((receipt, campaign), (backend_receipt, result_root/'backend_exact_readback.json')):
        if value['actual_tool_exit_code'] != 0 or value['artifacts'].get(str(required)) != data.sha(required):
            raise ValueError('actual successful branch report and readback receipts required')
        data.verify(value['artifacts'])
    data.verify(published['artifacts'])
    if (backend['status'] != 'independent_registry_readback_passed'
            or backend['campaign_sha256'] != data.sha(campaign)
            or backend['report_sha256'] != data.sha(published['report'])
            or published['hypothesis_id'] != probe.HYPOTHESIS
            or graded['status'] != 'semantic_probe_grade_completed'
            or audited['status'] != 'public_probe_raw_semantic_evidence_replayed'
            or graded['inherited_first_judgments'] != 4691
            or any(published[k] != audited[k] for k in ('metrics','classification','canonical_first_judgments'))
            or graded['canonical_first_judgments'] != audited['canonical_first_judgments']):
        raise ValueError('completed audited and registered probe branch required')
    cache = data.read(probe.ROOTS['grade']/'judgment_cache.json')
    packets = data.read(probe.ROOTS['prepare']/'exact_packets.json')
    value = extend(inherited, cache, packets, graded['metrics'], audited['canonical_first_judgments'])
    probe.qualified.audit_cache(value)
    return value


def install():
    global INSTALLED
    if INSTALLED:
        return

    def append_grade(cache, root, evaluation):
        value = OLD_APPEND(cache, root, evaluation)
        return branch_cache(value) if root == roots(2)['grade'] else value

    def lineage_paths(previous_roots=()):
        paths = OLD_PATHS(previous_roots)
        if roots(2)['grade'] in previous_roots:
            paths += branch_inputs()
        return paths

    original.append_grade, original.lineage_paths = append_grade, lineage_paths
    original.PLAN, original.TEST, original.PREFLIGHT = PLAN, TEST, PREFLIGHT
    INSTALLED = True


def configure(epoch, attempt):
    if epoch not in (4,8):
        raise ValueError('completed epoch1/2 evaluation must remain historical')
    install()
    return original.configure(epoch, attempt)


def preflight():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or PREFLIGHT.exists():
        raise ValueError('fresh CPU-only lineage preflight required')
    data.completed(OLD_PREFLIGHT)
    data.verify(data.read(OLD_PREFLIGHT/'producer.json')['input_hashes'])
    training, evaluation, environment = configure(4,1)
    paths = [*evaluation.lineage.inputs(), evaluation.old.SOURCE/'cases.json',
             study.pipeline.broad.catalog.DEST/'cases.json', *branch_inputs()]
    hashes = data.start(PREFLIGHT,'public_probe_lineage_preflight',paths,environment,
        {'systems_only':True,'new_judge_requests':0,'final_test_opened':False})
    cohorts, _, guard = evaluation.sources()
    cache = evaluation.lineage.base_cache()
    if sum(map(len,cohorts.values())) != 128 or cache != branch_cache(probe.prior_cache()):
        raise ValueError('complete merged cache and fixed128 DEV sources required')
    data.verify(hashes)
    value = {'status':'public_probe_lineage_preflight_passed','cases':128,
        'inherited_first_judgments':len(cache),'epoch2_first_judgments':4691,
        'probe_added_first_judgments':len(cache)-4691,'source_guard':guard,
        'new_judge_requests':0,'model_training':False,'task_accuracy':None,'final_test_opened':False,
        'artifacts':{str(PREFLIGHT/'producer.json'):data.sha(PREFLIGHT/'producer.json')}}
    data.write(PREFLIGHT/'result.json',value);print(json.dumps(value),flush=True)


def require_preflight():
    state = probe.base.require(PREFLIGHT)
    data.verify(data.read(PREFLIGHT/'producer.json')['input_hashes'])
    if state['status'] != 'public_probe_lineage_preflight_passed':
        raise ValueError('completed branch-preserving preflight required')
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('preflight',*STAGES))
    parser.add_argument('--epoch',type=int,choices=(4,8),default=4)
    parser.add_argument('--attempt',type=int,default=1)
    args = parser.parse_args()
    if args.stage == 'preflight':
        if (args.epoch,args.attempt) != (4,1):
            parser.error('fixed prospective epoch4 lineage preflight required')
        preflight()
    else:
        require_preflight()
        training,evaluation,_ = configure(args.epoch,args.attempt)
        if roots(args.epoch)[args.stage].exists():
            raise ValueError('fresh append-only evaluation stage required')
        if args.stage in ('inference_pilot','outputs'):
            evaluation.generate(args.stage == 'inference_pilot')
        else:
            evaluation.grade(args.stage == 'judge_pilot')
