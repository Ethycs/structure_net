"""Replay both matched on-policy DEV arms without changing the quality gate."""
import argparse
import json
from pathlib import Path

from experiments.structure_net import ncm_onpolicy_development as development
from experiments.neural_architecture_lab.finalize_ncm_broad_generator import classify

study = development.study
RESULTS = study.broad.runtime.ROOT/'20261025_onpolicy_residual_results_v1'
REPORT = Path('docs/08 - Analysis/2026-10-25_ncm-onpolicy-residual.md')


def paired_passes(fixed, residual):
    """Keep unavailable judgments separate from valid paired disagreements."""
    ids = [r['id'] for r in fixed]
    if len(set(ids)) != len(ids) or ids != [r['id'] for r in residual] or not ids:
        raise ValueError('identical unique ordered paired cases required')
    counts = {k:0 for k in ('both_pass','fixed_only','residual_only','both_fail','unresolved')}
    for left,right in zip(fixed,residual):
        for row in (left,right):
            if type(row['valid']) is not bool or (row['valid'] and type(row['passed']) is not bool
                    or not row['valid'] and row['passed'] is not None):
                raise ValueError('valid Boolean decisions or unresolved None required')
        if not left['valid'] or not right['valid']:
            counts['unresolved'] += 1
        elif left['passed'] and right['passed']: counts['both_pass'] += 1
        elif left['passed']: counts['fixed_only'] += 1
        elif right['passed']: counts['residual_only'] += 1
        else: counts['both_fail'] += 1
    return {'cases':len(ids),**counts,'valid_pairs':len(ids)-counts['unresolved'],
            'accuracy_difference_residual_minus_fixed':None if counts['unresolved'] else
                (counts['residual_only']-counts['fixed_only'])/len(ids)}


def select_candidate(arms):
    if set(arms) != set(study.objective.ARMS):
        raise ValueError('both matched arm results required')
    qualified = []
    for arm,value in arms.items():
        expected = classify(value['valid'],value['passed'],value['primary_admitted'],value['joint'])
        if value['classification'] != expected or value['DEV_qualified'] != (expected == 'DEV_qualified'):
            raise ValueError('qualification differs from the fixed same-case gate')
        if value['DEV_qualified']:
            joint = sum(r['joint_passed'] for r in value['joint'].values())/128
            if value['joint_task_accuracy'] != joint:
                raise ValueError('joint accuracy differs from same-case counts')
            qualified.append(arm)
    return max(qualified,key=lambda a:(arms[a]['joint_task_accuracy'],a == 'residual_targets')) if qualified else None


def replay_arm(arm):
    from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
    from experiments.neural_architecture_lab.finalize_ncm_lm1b_native_development import replay_outputs

    training,evaluation,environment = development.configure(arm)
    q,old,lineage = training.q,evaluation.old,evaluation.lineage
    roots = development.ROOTS[arm]
    lineage.configure()
    paths = development.completed_training_audits(training)
    train_root = study.ROOTS[arm,'train']
    for root in (train_root,*roots.values()): training.preparation.completed(root)
    trained = q.read(train_root/'result.json')
    if (trained['updates'] != study.CONFIG.updates or trained['TRAIN_cases'] != 1376
            or trained['initial_model_sha256'] != study.MODEL_SHA or not trained['strict_reload_passed']
            or not trained['oracle_unchanged'] or trained['task_accuracy'] is not None
            or trained['final_test_opened'] or trained['speed_tested']):
        raise ValueError('fixed final training checkpoint and protected reconstruction required')
    cohorts,manifest,split = evaluation.sources()
    rows = [r for c in old.COHORTS for r in cohorts[c]]
    outputs,pilot = [q.read(roots[s]/'outputs.json') for s in ('outputs','inference_pilot')]
    tokenizer = training.native.tokenizer()
    replay_outputs(outputs,rows,manifest,tokenizer)
    replay_outputs(pilot,old.inference_probes(cohorts,manifest),manifest,tokenizer)
    by_id = {r['id']:r for r in outputs}
    if any(by_id[r['id']] != r for r in pilot):
        raise ValueError('complete predictions differ from systems pilot')
    for stage in ('inference_pilot','outputs'):
        result,proof = [q.read(roots[stage]/n) for n in ('result.json','producer.json')]
        if (result['model_sha256'] != trained['final_model_sha256']
                or result['outputs_sha256'] != study.broad.file_hash(roots[stage]/'outputs.json')
                or proof['environment'] != environment or proof['arm'] != arm
                or result['producer_fingerprint'] != proof['fingerprint']
                or proof['fingerprint'] != q.digest({k:v for k,v in proof.items() if k != 'fingerprint'})
                or result['cases'] != (2 if stage == 'inference_pilot' else 128)):
            raise ValueError('fixed checkpoint, output, runtime or arm identity differs')
    cases = {c:[{**r,'texts':{'student':by_id[r['id']]['text']}} for r in cohorts[c]] for c in old.COHORTS}
    packets = {c:{'student':[q.packet(r['id'],r['source'],
        [{'candidate_id':'summary','text':r['texts']['student']}]) for r in cases[c]]} for c in old.COHORTS}
    inherited = lineage.base_cache()
    cache,pilot_cache = [q.read(roots[s]/'judgment_cache.json') for s in ('grade','judge_pilot')]
    pilot_allowed = {old.oracle.first.packet_key(p,'secondary') for c in old.COHORTS
                     for p in packets[c]['student'][:2]}
    development.preserve_firsts(inherited,pilot_cache,allowed_new=pilot_allowed)
    development.preserve_firsts(pilot_cache,cache)
    lineage.audit_cache(cache)
    for stage in ('judge_pilot','grade'):
        proof = q.read(roots[stage]/'producer.json')
        if (q.read(roots[stage]/'cases.json') != cases or q.read(roots[stage]/'exact_packets.json') != packets
                or proof['arm'] != arm or proof['environment'] != environment
                or proof['fingerprint'] != q.digest({k:v for k,v in proof.items() if k != 'fingerprint'})
                or proof['config']['inherited_judgments'] != len(inherited)
                or q.read(roots[stage]/'result.json')['producer_fingerprint'] != proof['fingerprint']):
            raise ValueError('exact complete source-only packets or producer binding differs')
    scores,metrics,pilot_metrics = {},{c:{} for c in old.COHORTS},{c:{} for c in old.COHORTS}

    def replay(stage,cohort,role,count,decisions):
        expected = replay_scores(cases[cohort][:count],
            {'student':packets[cohort]['student'][:count]},decisions,'student',role)
        directory = roots[stage]/cohort
        reported = q.read(directory/f'{role}_student_result.json')
        if (q.read(directory/f'{role}_student_case_scores.json') != expected
                or any(reported.get(k) != v for k,v in old.task_metric(expected).items())):
            raise ValueError('raw first judgments and semantic metric replay differ')
        return expected,reported

    for cohort in old.COHORTS:
        pilot_scores,pilot_metrics[cohort]['secondary'] = replay('judge_pilot',cohort,'secondary',2,pilot_cache)
        if not all(r['valid'] for r in pilot_scores): raise ValueError('valid four-case judge pilot required')
        scores[cohort,'secondary'],metrics[cohort]['secondary'] = replay('grade',cohort,'secondary',64,cache)
    pilot_result = q.read(roots['judge_pilot']/'result.json')
    if (not pilot_result['valid_pilot'] or pilot_result['metrics'] != pilot_metrics
            or pilot_result['primary_admitted'] or pilot_result['joint'] is not None
            or pilot_result['DEV_qualified'] or pilot_result['task_accuracy'] is not None
            or pilot_result['canonical_first_judgments'] != len(pilot_cache)
            or pilot_result['inherited_first_judgments'] != len(inherited)):
        raise ValueError('systems pilot denominator, role or first lineage differs')
    admitted = old.secondary_admitted({c:metrics[c]['secondary'] for c in old.COHORTS})
    joints = None
    if admitted:
        joints = {}
        for c in old.COHORTS:
            scores[c,'primary'],metrics[c]['primary'] = replay('grade',c,'primary',64,cache)
            joints[c] = old.joint_gate(scores[c,'secondary'],scores[c,'primary'])
    elif any((roots['grade']/c/'primary_student_case_scores.json').exists() for c in old.COHORTS):
        raise ValueError('primary judge ran without admission')
    roles = ('secondary','primary') if admitted else ('secondary',)
    allowed = {old.oracle.first.packet_key(p,role) for c in old.COHORTS
               for p in packets[c]['student'] for role in roles}
    development.preserve_firsts(inherited,cache,allowed_new=allowed)
    qualified = admitted and all(v['qualified'] for v in joints.values())
    valid = sum(metrics[c]['secondary']['valid'] for c in old.COHORTS)
    passed = sum(metrics[c]['secondary']['passed'] for c in old.COHORTS)
    accuracy = passed/128 if valid == 128 else None
    joint_accuracy = (sum(v['joint_passed'] for v in joints.values())/128
                      if joints and all(v['all_valid'] for v in joints.values()) else None)
    grade = q.read(roots['grade']/'result.json')
    expected = {'metrics':metrics,'primary_admitted':admitted,'joint':joints,'DEV_qualified':qualified,
        'task_accuracy':accuracy,'canonical_first_judgments':len(cache),'inherited_first_judgments':len(inherited),
        'final_test_opened':False,'speed_tested':False,'task_accuracy_judge':q.JUDGES['secondary'],
        'metric':'complete_summary_semantic_pass'}
    if any(grade.get(k) != v for k,v in expected.items()):
        raise ValueError('grade aggregate, first lineage or qualification gate differs')
    paths += [train_root/n for n in ('producer.json','result.json','actual_tool_receipt.json',
        'initial_outputs.json','trace.jsonl')]
    for root in roots.values():
        paths += [p for p in root.rglob('*.json') if 'producer_source' not in p.parts]
    result = {'arm':arm,'classification':classify(valid,passed,admitted,joints),'cases':128,
        'valid':valid,'passed':passed,'task_accuracy':accuracy,'joint_task_accuracy':joint_accuracy,
        'wilson_95':q.wilson(passed,128) if valid == 128 else None,**expected,
        'training_updates':trained['updates'],'training_cases':trained['TRAIN_cases'],
        'training_seconds':trained['training_seconds'],'peak_allocated_bytes':trained['peak_allocated_bytes'],
        'trainable_parameters':trained['trainable_parameters'],'frozen_reconstruction_preserved':True,
        'checkpoint_sha256':trained['checkpoint_sha256'],'model_sha256':trained['final_model_sha256'],
        'TRAIN_diagnostics':trained['final_TRAIN'],'shuffled_source16':trained['shuffled_source16'],
        'split_proof':split}
    return result,scores,cache,paths,training,evaluation


def finalize():
    if RESULTS.exists(): raise ValueError('fresh append-only matched aggregate required')
    arms,all_scores,paths = {},{},[Path(__file__),study.PLAN]
    previous_cache = None
    for arm in study.objective.ARMS:
        value,scores,cache,inputs,training,evaluation = replay_arm(arm)
        if previous_cache is not None:
            if value['inherited_first_judgments'] != len(previous_cache):
                raise ValueError('sibling did not inherit the complete first arm cache')
            development.preserve_firsts(previous_cache,cache)
        arms[arm],all_scores[arm],previous_cache = value,scores,cache
        paths += inputs
    q,old = training.q,evaluation.old
    if arms['fixed_targets']['inherited_first_judgments'] != 3923:
        raise ValueError('all3923 historical canonical firsts must precede the comparison')
    if arms['fixed_targets']['split_proof'] != arms['residual_targets']['split_proof']:
        raise ValueError('matched source-disjoint cohorts differ')
    initial = [q.read(study.ROOTS[a,'train']/'initial_outputs.json') for a in study.objective.ARMS]
    traces = [[json.loads(line) for line in (study.ROOTS[a,'train']/'trace.jsonl').read_text().splitlines()]
              for a in study.objective.ARMS]
    sampling = [[[(m['id'],m['onpolicy']) for m in row['metrics']] for row in trace] for trace in traces]
    if initial[0] != initial[1] or sampling[0] != sampling[1]:
        raise ValueError('matched initialization or source/history-mode sampling differs')
    pairs = {c:paired_passes(all_scores['fixed_targets'][c,'secondary'],
                            all_scores['residual_targets'][c,'secondary']) for c in old.COHORTS}
    pairs['all128'] = paired_passes(
        [r for c in old.COHORTS for r in all_scores['fixed_targets'][c,'secondary']],
        [r for c in old.COHORTS for r in all_scores['residual_targets'][c,'secondary']])
    candidate = select_candidate(arms)
    unresolved = any(v['classification'] == 'DEV_transport_invalid' for v in arms.values())
    classification = ('DEV_candidate_qualified' if candidate else 'DEV_comparison_transport_invalid' if unresolved
        else 'DEV_semantic_partial_progress' if any(v['passed'] for v in arms.values())
        else 'DEV_both_semantic_quality_failed')
    campaign = {'schema':'nal.ncm-onpolicy-residual-results.v1','hypothesis_id':study.HYPOTHESIS,
        'classification':classification,'conformance':'EXPLORATORY_UNDERPOWERED',
        'arms':arms,'paired_secondary_decisions':pairs,'selected_candidate':candidate,
        'selected_checkpoint_sha256':arms[candidate]['checkpoint_sha256'] if candidate else None,
        'comparison_has_unresolved_judgments':unresolved,
        'selection_rule':'qualified arms only; higher joint DEV accuracy; exact tie to residual_targets',
        'cases_per_arm':128,'independent_training_seeds':1,'same_initialization':True,'same_sampling':True,
        'source_only':True,'task_metric':'complete_summary_semantic_pass',
        'canonical_first_judgments':len(previous_cache),'inherited_first_judgments':3923,
        'all_historical_and_pilot_firsts_preserved':True,'final_test_opened':False,
        'final_task_goal_achieved':False,'speed_tested':False,
        'artifacts':{str(p):study.broad.file_hash(p) for p in dict.fromkeys(paths)}}
    RESULTS.mkdir();study.broad.write_json(RESULTS/'campaign_results.json',campaign)
    print(json.dumps({k:v for k,v in campaign.items() if k not in ('artifacts','arms')},allow_nan=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.parse_args()
    finalize()
