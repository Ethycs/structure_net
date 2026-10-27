"""Re-render outputs and replay raw semantic firsts for task-supervision arms."""
import argparse
import json
import os

from experiments.structure_net import ncm_task_supervision_development as dev
from experiments.neural_architecture_lab.finalize_ncm_lm1b_codec_semantics import replay_scores
from experiments.neural_architecture_lab.finalize_ncm_lm1b_native_development import replay_outputs

data = dev.data


def preserve(before,after,allowed):
    if any(after.get(k) != v for k,v in before.items()):raise ValueError('a historical first changed or disappeared')
    if set(after)-set(before) != set(allowed)-set(before):raise ValueError('all and only admitted first decisions required')


def audit(arm):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':raise ValueError('CPU-only raw semantic replay required')
    training,evaluation,trained,environment = dev.configure(arm)
    q,old,lineage = training.q,evaluation.old,evaluation.lineage
    stages = dev.ROOTS[arm];destination = stages['audit'];paths = []
    for stage in ('inference_pilot','outputs','judge_pilot','grade'):
        root = stages[stage];state = dev.completed(root);proof = data.read(root/'producer.json')
        if (proof['schema'] != 'nal.ncm-task-supervision-dev.v1' or proof['arm'] != arm
                or proof['stage'] != stage or proof['config']['rubric'] != q.RUBRIC
                or proof['config']['judges'] != q.JUDGES
                or state['producer_fingerprint'] != proof['fingerprint']):
            raise ValueError('unchanged arm, stage, rubric and producer required')
        paths += [p for p in root.rglob('*.json') if 'producer_source' not in p.parts]
    cohorts,manifest,split = evaluation.sources();rows = [r for c in dev.COHORTS for r in cohorts[c]]
    outputs,probes = [data.read(stages[s]/'outputs.json') for s in ('outputs','inference_pilot')]
    tokenizer = training.native.tokenizer()
    replay_outputs(outputs,rows,manifest,tokenizer)
    replay_outputs(probes,old.inference_probes(cohorts,manifest),manifest,tokenizer)
    by_id = {r['id']:r for r in outputs}
    if any(by_id[r['id']] != r for r in probes):raise ValueError('source-only outputs differ from repeated pilot')
    for stage,count in (('outputs',128),('inference_pilot',len(probes))):
        root = stages[stage];state = data.read(root/'result.json')
        if (state['cases'] != count or state['model_sha256'] != trained['result']['final_model_sha256']
                or state['outputs_sha256'] != data.sha(root/'outputs.json')
                or state['peak_allocated_bytes'] > dev.study.CONFIG.peak_limit_bytes
                or data.read(root/'split_proof.json') != split):
            raise ValueError('checkpoint, source coverage, split proof or memory limit differs')
    cases = {c:[{**r,'texts':{'student':by_id[r['id']]['text']}} for r in cohorts[c]] for c in dev.COHORTS}
    packets = {c:{'student':[q.packet(r['id'],r['source'],[{'candidate_id':'summary','text':r['texts']['student']}])
                            for r in cases[c]]} for c in dev.COHORTS}
    inherited = lineage.base_cache();pilot_cache = data.read(stages['judge_pilot']/'judgment_cache.json')
    cache = data.read(stages['grade']/'judgment_cache.json');lineage.audit_cache(cache)
    pilot_allowed = {old.oracle.first.packet_key(p,'secondary') for c in dev.COHORTS for p in packets[c]['student'][:2]}
    preserve(inherited,pilot_cache,pilot_allowed)
    for stage in ('judge_pilot','grade'):
        root = stages[stage]
        if (data.read(root/'cases.json') != cases or data.read(root/'exact_packets.json') != packets
                or data.read(root/'producer.json')['config']['inherited_first_judgments'] != len(inherited)):
            raise ValueError('exact full source and anonymous summary packets required')
    metrics = {c:{} for c in dev.COHORTS};scores = {c:{} for c in dev.COHORTS};pilot_metrics = {c:{} for c in dev.COHORTS}

    def decisions(stage,cohort,role,count,firsts):
        expected = replay_scores(cases[cohort][:count],{'student':packets[cohort]['student'][:count]},firsts,'student',role)
        root = stages[stage]/cohort;reported = data.read(root/f'{role}_student_result.json')
        if (data.read(root/f'{role}_student_case_scores.json') != expected
                or any(reported.get(k) != v for k,v in old.task_metric(expected).items())):
            raise ValueError('raw semantic firsts differ from reported components or decisions')
        return expected,reported

    for c in dev.COHORTS:
        pilot_scores,pilot_metrics[c]['secondary'] = decisions('judge_pilot',c,'secondary',2,pilot_cache)
        if not all(r['valid'] for r in pilot_scores):raise ValueError('all four semantic pilot decisions must be valid')
        scores[c]['secondary'],metrics[c]['secondary'] = decisions('grade',c,'secondary',64,cache)
    admitted = old.secondary_admitted({c:metrics[c]['secondary'] for c in dev.COHORTS})
    if admitted:
        for c in dev.COHORTS:scores[c]['primary'],metrics[c]['primary'] = decisions('grade',c,'primary',64,cache)
    elif any((stages['grade']/c/'primary_student_case_scores.json').exists() for c in dev.COHORTS):
        raise ValueError('primary judge ran before Terra cohort admission')
    roles = ('secondary','primary') if admitted else ('secondary',)
    allowed = {old.oracle.first.packet_key(p,role) for c in dev.COHORTS for p in packets[c]['student'] for role in roles}
    preserve(inherited,cache,allowed);preserve(pilot_cache,cache,allowed)
    pilot = data.read(stages['judge_pilot']/'result.json')
    if (pilot['valid_pilot'] is not True or pilot['metrics'] != pilot_metrics or pilot['primary_admitted']
            or pilot['DEV_qualified'] or pilot['joint'] is not None or pilot['task_accuracy'] is not None
            or pilot['canonical_first_judgments'] != len(pilot_cache) or pilot['inherited_first_judgments'] != len(inherited)):
        raise ValueError('judge pilot aggregate differs from raw firsts')
    summary = dev.summarize(scores);grade = data.read(stages['grade']/'result.json')
    expected = {k:summary[k] for k in ('task_accuracy','primary_admitted','joint','DEV_qualified')}
    expected.update(metrics=metrics,canonical_first_judgments=len(cache),inherited_first_judgments=len(inherited),
        final_test_opened=False,speed_tested=False,metric='complete_summary_semantic_pass',task_accuracy_judge=q.JUDGES['secondary'])
    if any(grade.get(k) != v for k,v in expected.items()):raise ValueError('semantic aggregate or qualification differs')
    proof = evaluation.freeze(destination,'audit',paths)
    data.write(destination/'scores.json',scores)
    dev.finish(destination,proof,{'status':'task_supervision_semantic_firsts_replayed','cases':128,
        **summary,'metrics':metrics,'canonical_first_judgments':len(cache),'inherited_first_judgments':len(inherited),
        'checkpoint_sha256':trained['result']['checkpoint_sha256'],'model_sha256':trained['result']['final_model_sha256'],
        'split_proof':split,'new_judge_requests':0,'final_task_goal_achieved':False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__);parser.add_argument('--arm',choices=dev.study.ARMS,required=True)
    audit(parser.parse_args().arm)
