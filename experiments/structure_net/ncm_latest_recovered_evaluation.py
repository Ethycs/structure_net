"""Evaluate the fixed recovered checkpoint next, at the user's request."""
import argparse
from pathlib import Path

from experiments.structure_net import ncm_recovered_development_outputs as generated
from experiments.structure_net import ncm_recovered_development_semantics as semantic
from experiments.structure_net import ncm_repaired_judge_route as repaired

data, previous = generated.data, generated.previous
cache = generated.verified
BASE = generated.study.inputs.BASE
PLAN = Path('docs/07 - Status Reports/2026-10-29_ncm-latest-recovered-evaluation.md')
TEST = Path('tests/structure_net/test_ncm_latest_recovered_evaluation.py')
CPU = BASE/'20261029_latest_recovered_evaluation_CPU_v1.json'
INHERITED = 5903
DEFERRED = tuple(root for stages in generated.hierarchy.ROOTS.values() for root in stages.values())
SUPERSEDED = tuple(generated.ROOTS.values()) + tuple(
    root for stages in semantic.ROOTS.values() for root in stages.values()) + (semantic.COMPARISON,)
GENERATION = {stage: BASE/f'20261029_latest_recovered_DEV_{stage}_v1'
              for stage in generated.ROOTS}
SEMANTIC = {arm: {stage: BASE/f'20261029_latest_recovered_{arm}_DEV_{stage}_v1'
                  for stage in semantic.STAGES} for arm in semantic.ARMS}
COMPARISON = BASE/'20261029_latest_recovered_paired_DEV_comparison_v1'
INSTALLED = False


def require_unstarted(paths):
    started = [str(path) for path in paths if path.exists()]
    if started:
        raise ValueError('cannot omit an already started candidate branch: '+', '.join(started))


def closed_history():
    require_unstarted((*DEFERRED, *SUPERSEDED))
    # The existing tree preflight ancestry includes both completed task arms,
    # but no tree or dense outcomes. Reuse that complete unchanged history.
    training, evaluation, environment, ancestor, judgments, paths = generated.hierarchy.current_history('tree')
    if len(judgments) != INHERITED:
        raise ValueError('exact 5903 completed first judgments required before latest-model evaluation')
    ancestor.audit_cache(judgments)
    return training, evaluation, environment, judgments, paths


def bind_priority(proof):
    value = {**proof, 'config': {**proof['config'],
        'evaluation_order_amendment': 'user_requested_latest_recovered_first_20261029',
        'base_first_judgments': INHERITED,
        'input_validation_policy': {**cache.POLICY, 'scope': 'this latest-model evaluation process'}},
        'input_hashes': dict(proof['input_hashes'])}
    for path in (Path(__file__).resolve(), PLAN, TEST, CPU,
                 Path(cache.__file__).resolve(), cache.PLAN, cache.TEST, cache.CPU):
        name, digest = str(path), data.sha(path)
        if name in value['input_hashes'] and value['input_hashes'][name] != digest:
            raise ValueError('latest-model amendment conflicts with an existing producer input')
        value['input_hashes'][name] = digest
    return value


def install():
    global INSTALLED
    if INSTALLED:
        return
    for path, count in ((CPU, 11), (cache.CPU, 3)):
        checked = data.read(path)
        data.verify(checked['artifacts'])
        if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != count:
            raise ValueError('latest-order, unchanged readout and guarded-hash contracts required')
    proofs = cache.ProcessProofs()
    data.sha = proofs.sha
    generated.closed_history = closed_history
    generated.ROOTS, semantic.ROOTS, semantic.COMPARISON = GENERATION, SEMANTIC, COMPARISON
    original_write = previous.write_proof
    previous.write_proof = lambda root, proof, files: original_write(root, bind_priority(proof), files)
    INSTALLED = True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=(*GENERATION, 'export_pilot', 'export_outputs',
                                        'judge_pilot', 'grade', 'audit', 'comparison'))
    parser.add_argument('--arm', choices=semantic.ARMS)
    args = parser.parse_args()
    needs_arm = args.stage not in (*GENERATION, 'comparison')
    if needs_arm != (args.arm is not None):
        parser.error('an arm is required only for saved-output exports and semantic stages')
    install()
    if args.stage in GENERATION:
        root = GENERATION[args.stage]
    elif args.stage == 'comparison':
        root = COMPARISON
    else:
        target = {'export_pilot': 'inference_pilot', 'export_outputs': 'outputs'}.get(args.stage, args.stage)
        root = SEMANTIC[args.arm][target]
    if root.exists():
        raise ValueError('fresh append-only latest-model stage required')
    try:
        if args.stage in GENERATION:
            generated.run(args.stage)
        elif args.stage in ('export_pilot', 'export_outputs'):
            semantic.export(args.arm, 'inference_pilot' if args.stage == 'export_pilot' else 'outputs')
        elif args.stage == 'comparison':
            semantic.comparison()
        else:
            repaired.install()
            repaired.consume('recovered', args.stage, args.arm)
    except Exception as error:
        if root.exists() and not (root/'failure.json').exists():
            data.write(root/'failure.json', {'type': type(error).__name__, 'message': str(error),
                'preserve_all_first_responses': True, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    main()
