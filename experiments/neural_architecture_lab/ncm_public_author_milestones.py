"""Replay public milestones while retaining both original and corrected probe firsts."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_public_author_lineage as lineage
from experiments.neural_architecture_lab import ncm_public_interruption_recovery as recovery

final, data = recovery.final, lineage.data


def inherited_for_milestone(epoch, prior_count, branch_count):
    if epoch == 4:
        if prior_count != 4691 or branch_count < prior_count:
            raise ValueError('exact epoch2-to-probe lineage transition required')
        return branch_count
    return prior_count


def report(epoch, attempt):
    destination, document = final.roots(epoch)['results'], final.report_path(epoch)
    if destination.exists() or document.exists():
        raise ValueError('fresh milestone report required')
    initial = {str(p):data.sha(p) for p in final.source_files()}
    original_branch = lineage.parent.branch_cache(lineage.probe.prior_cache())
    branch = lineage.author_cache(original_branch)
    milestones, inherited = [], 4435
    paths = [*lineage.branch_inputs(), *[lineage.PREFLIGHT/n for n in
             ('producer.json','result.json','actual_tool_receipt.json')]]
    for number in final.study.CONFIG.milestones:
        if number > epoch:
            break
        inherited = inherited_for_milestone(number,inherited,len(branch))
        root = final.roots(number)['semantic_replay']
        final.receipt(root,('result.json','scores.json'))
        state = data.completed(root)
        if state['epoch'] != number or state['inherited_first_judgments'] != inherited:
            raise ValueError('complete milestone ancestry including probe branch required')
        if number == epoch and state['attempt'] != attempt:
            raise ValueError('selected attempt differs')
        if milestones and any(state[k] != milestones[0][k] for k in ('training_identity','frozen_hashes','split_proof')):
            raise ValueError('training identity, reconstruction or evaluation sources changed')
        inherited = state['canonical_first_judgments']
        milestones.append({k:v for k,v in state.items() if k != 'artifacts'})
        paths += [root/n for n in ('result.json','scores.json','actual_tool_receipt.json')]
    candidate = final.selection(milestones); hashes = final.bindings(paths,initial)
    value = {'schema':'nal.ncm-public-milestone-results.v3',
        'hypothesis_id':f'ncm-public-curriculum-epoch{epoch}-v1',
        'parent_hypothesis_id':'ncm-public-curriculum-v1','epoch':epoch,'attempt':attempt,
        'classification':milestones[-1]['classification'],'milestones':milestones,
        'selected_candidate_epoch':candidate,'metric':'complete_summary_semantic_pass',
        'cases_per_milestone':128,'conformance':'EXPLORATORY_UNDERPOWERED',
        'independent_training_trajectories':1,'canonical_first_judgments':inherited,
        'lineage_extensions':lineage.extension_records(len(original_branch),len(branch)),
        'final_test_opened':False,'final_task_goal_achieved':False,'speed_tested':False,'artifacts':hashes}
    destination.mkdir();data.write(destination/'campaign_results.json',value)
    lines = [f'# Public NCM curriculum: epoch{epoch}','',
        f"Classification: `{value['classification']}`. Final320 remains sealed.",'',
        '| Epoch | Terra semantic passes | Valid judgments | Joint Terra/Sol accuracy | DEV qualified |',
        '|---|---:|---:|---:|---|']
    for row in milestones:
        accuracy = 'unresolved' if row['task_accuracy'] is None else f"{row['passed']}/128 ({100*row['task_accuracy']:.2f}%)"
        joint = 'unmeasured or unresolved' if row['joint_task_accuracy'] is None else f"{100*row['joint_task_accuracy']:.2f}%"
        lines.append(f"| {row['epoch']} | {accuracy} | {row['valid']}/128 | {joint} | {row['DEV_qualified']} |")
    lines += ['',
        'Accuracy measures complete readable, faithful, salient summaries within128 stored tokens; paraphrases are accepted.',
        'Training loss, code agreement and codec reconstruction are separate diagnostics and are not task accuracy.',
        'Each epoch visits all162,353 public pairs and54,118 reviewed-anchor presentations. The original Llama body and reconstruction path remain fixed.',
        f'The original probe added {len(original_branch)-4691} first judgments after epoch 2; the author-rendering correction added {len(branch)-len(original_branch)}. Both branches are retained and excluded from application accuracy. Earlier milestones are unchanged.',
        'This is one continuing training trajectory evaluated at fixed milestones, not independent model replications.',
        f'Earliest candidate for the separate sealed-final protocol: `{candidate}`. DEV qualification does not establish the final95% goal.',
        'Inference speed remains unmeasured.','',f'Detailed decisions and hashes: `{destination}/campaign_results.json`.']
    document.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'epoch':epoch,'classification':value['classification'],'selected_candidate_epoch':candidate}),flush=True)


def run(stage, epoch, attempt):
    if epoch not in (4,8):
        raise ValueError('completed epoch1/2 evidence must remain unchanged')
    lineage.require_preflight();lineage.install();recovery.configure()
    prior_files = final.source_files
    final.source_files = lambda:sorted(set(prior_files()) | {lineage.PLAN.resolve(),lineage.TEST.resolve(),Path(__file__).resolve()})
    if stage == 'replay':
        final.replay_semantics(epoch,attempt)
    elif stage == 'report':
        report(epoch,attempt)
    else:
        raise ValueError('semantic replay or report required')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('replay','report'))
    parser.add_argument('--epoch',type=int,choices=(4,8),required=True)
    parser.add_argument('--attempt',type=int,default=1)
    args = parser.parse_args();run(args.stage,args.epoch,args.attempt)
