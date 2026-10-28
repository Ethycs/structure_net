"""Replay/report future public milestones with all three intervening branches."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_recovered_review_lineage as lineage
from experiments.neural_architecture_lab import ncm_public_author_milestones as prior

final, data, author = prior.final, prior.data, lineage.author


def extension_records(original, rendered, recovered):
    if recovered < rendered:
        raise ValueError('recovered-label branch cannot discard earlier firsts')
    return [*author.extension_records(original, rendered),
        {'after_epoch': 2, 'hypothesis_id': lineage.review.HYPOTHESIS,
         'before_firsts': rendered, 'after_firsts': recovered, 'added_firsts': recovered-rendered,
         'included_in_application_accuracy': False}]


def report(epoch, attempt):
    destination, document = final.roots(epoch)['results'], final.report_path(epoch)
    if destination.exists() or document.exists():
        raise ValueError('fresh public milestone report required')
    initial = {str(p): data.sha(p) for p in final.source_files()}
    original = author.parent.branch_cache(author.probe.prior_cache())
    rendered = author.author_cache(original); recovered = lineage.extend(rendered)
    paths = [*author.branch_inputs(), *lineage.branch_inputs(),
        *[lineage.PREFLIGHT/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]]
    milestones, inherited = [], 4435
    for number in final.study.CONFIG.milestones:
        if number > epoch: break
        inherited = prior.inherited_for_milestone(number, inherited, len(recovered))
        root = final.roots(number)['semantic_replay']; final.receipt(root, ('result.json', 'scores.json'))
        state = data.completed(root)
        if state['epoch'] != number or state['inherited_first_judgments'] != inherited:
            raise ValueError('complete epoch ancestry including label branch required')
        if number == epoch and state['attempt'] != attempt:
            raise ValueError('selected attempt differs')
        if milestones and any(state[k] != milestones[0][k] for k in ('training_identity', 'frozen_hashes', 'split_proof')):
            raise ValueError('training, reconstruction or evaluation sources changed')
        inherited = state['canonical_first_judgments']
        milestones.append({k: v for k, v in state.items() if k != 'artifacts'})
        paths += [root/n for n in ('result.json', 'scores.json', 'actual_tool_receipt.json')]
    candidate = final.selection(milestones)
    value = {'schema': 'nal.ncm-public-milestone-results.v4',
        'hypothesis_id': f'ncm-public-curriculum-epoch{epoch}-v1', 'parent_hypothesis_id': 'ncm-public-curriculum-v1',
        'epoch': epoch, 'attempt': attempt, 'classification': milestones[-1]['classification'],
        'milestones': milestones, 'selected_candidate_epoch': candidate, 'metric': 'complete_summary_semantic_pass',
        'cases_per_milestone': 128, 'conformance': 'EXPLORATORY_UNDERPOWERED',
        'independent_training_trajectories': 1, 'canonical_first_judgments': inherited,
        'lineage_extensions': extension_records(len(original), len(rendered), len(recovered)),
        'final_test_opened': False, 'final_task_goal_achieved': False, 'speed_tested': False,
        'artifacts': final.bindings(paths, initial)}
    destination.mkdir(); data.write(destination/'campaign_results.json', value)
    lines = [f'# Public NCM curriculum: epoch{epoch}', '',
        f"Classification: `{value['classification']}`. Final320 remains sealed.", '',
        '| Epoch | Terra semantic passes | Valid judgments | Joint Terra/Sol accuracy | DEV qualified |',
        '|---|---:|---:|---:|---|']
    for row in milestones:
        rate = 'unresolved' if row['task_accuracy'] is None else f"{row['passed']}/128 ({100*row['task_accuracy']:.2f}%)"
        joint = 'unmeasured or unresolved' if row['joint_task_accuracy'] is None else f"{100*row['joint_task_accuracy']:.2f}%"
        lines.append(f"| {row['epoch']} | {rate} | {row['valid']}/128 | {joint} | {row['DEV_qualified']} |")
    lines += ['',
        'Accuracy measures complete readable, faithful, salient summaries within128 stored tokens; paraphrases are accepted.',
        'Training loss, code agreement, saved-label grades and codec reconstruction are separate diagnostics.',
        'Each epoch visits all162,353 public pairs and54,118 reviewed-anchor presentations. The original Llama body and reconstruction path remain fixed.',
        f'After epoch2, the original probe added {len(original)-4691} first judgments, author rendering added {len(rendered)-len(original)}, and recovered saved-label review added {len(recovered)-len(rendered)}. All are preserved and excluded from application model accuracy. Earlier milestones remain historical.',
        'This is one continuing training trajectory, not independent model replications.',
        f'Earliest candidate for the separate sealed-final protocol: `{candidate}`. DEV qualification does not establish the final95% goal.',
        'Inference speed remains unmeasured.', '', f'Decisions and artifact hashes: `{destination}/campaign_results.json`.']
    document.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'epoch': epoch, 'classification': value['classification'], 'selected_candidate_epoch': candidate}), flush=True)


def run(stage, epoch, attempt):
    if epoch not in (4, 8):
        raise ValueError('earlier milestones must remain historical')
    lineage.require_preflight(); author.require_preflight(); lineage.install(); prior.recovery.configure()
    previous = final.source_files
    final.source_files = lambda: sorted(set(previous()) | {Path(__file__).resolve(), Path(lineage.__file__).resolve(), lineage.PLAN.resolve(), lineage.TEST.resolve()})
    if stage == 'replay': final.replay_semantics(epoch, attempt)
    elif stage == 'report': report(epoch, attempt)
    else: raise ValueError('replay or report required')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('stage', choices=('replay', 'report'))
    parser.add_argument('--epoch', type=int, choices=(4, 8), required=True); parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args(); run(args.stage, args.epoch, args.attempt)
