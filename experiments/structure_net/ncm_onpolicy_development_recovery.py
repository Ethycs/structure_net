"""Preserve completed v1 evidence while initializing the sibling rubric early."""
import argparse
from pathlib import Path

from experiments.structure_net import ncm_onpolicy_development as original
from experiments.structure_net import ncm_sol_judgment_lineage as qualified

study, STAGES = original.study, original.STAGES
PLAN = Path('docs/07 - Status Reports/2026-10-25_ncm-onpolicy-residual-evaluation-recovery.md')
TEST = Path('tests/structure_net/test_ncm_onpolicy_development_recovery.py')
INCIDENT = study.broad.runtime.ROOT/'20261025_onpolicy_residual_DEV_launch_failure_v1'
ROOTS = {**original.ROOTS, 'residual_targets': {
    s:study.broad.runtime.ROOT/f'20261025_onpolicy_residual_residual_targets_DEV_{s}_v2'
    for s in STAGES}}
INPUTS = (PLAN, TEST, Path(__file__), *(
    INCIDENT/n for n in ('result.json','actual_tool_receipt.json','failed_launch_actual_tool_receipt.json')))
completed_training_audits = original.completed_training_audits
preserve_firsts = original.preserve_firsts


def configure(arm, *, preflight=False):
    # packet_key includes the rubric and judge. Install them BEFORE the
    # original sibling-cache guard runs; its rejection policy is unchanged.
    qualified.configure()
    original.ROOTS = ROOTS
    training,evaluation,environment = original.configure(arm,preflight=preflight)
    prior_freeze = evaluation.freeze

    def freeze(root,stage,extra):
        return prior_freeze(root,stage,[*extra,*INPUTS])

    evaluation.freeze = freeze
    return training,evaluation,environment


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('preflight',*STAGES))
    args = parser.parse_args()
    # Deliberately use the real sibling path even for the offline preflight.
    training,evaluation,_ = configure('residual_targets')
    if args.stage == 'preflight':
        cohorts,_,guard = evaluation.sources()
        print(study.json.dumps({'status':'recovered_sibling_cache_preflight_passed',
            'source_disjoint':guard,'inherited_firsts':len(evaluation.lineage.base_cache()),
            'DEV_cases':sum(len(v) for v in cohorts.values()),'new_requests':0,
            'final_test_opened':False}),flush=True)
        return
    if ROOTS['residual_targets'][args.stage].exists():
        raise ValueError('fresh append-only recovery stage required')
    completed_training_audits(training)
    if args.stage in ('inference_pilot','outputs'):
        evaluation.generate(args.stage == 'inference_pilot')
    else:
        evaluation.grade(args.stage == 'judge_pilot')


if __name__ == '__main__':
    main()
