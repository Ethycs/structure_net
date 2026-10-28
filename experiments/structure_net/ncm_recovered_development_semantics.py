"""Grade fixed paired recovered-model outputs with the existing semantic protocol.

Readout exports copy already audited source predictions. They do not regenerate
codes or summaries. The exact rubric, conditional Sol gate, first decisions,
invalid-attempt budget and sealed final test retain their original meanings.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from experiments.structure_net import ncm_recovered_development_outputs as generated

study, data, previous = generated.study, generated.data, generated.previous
ARMS = generated.paired.ARMS
COHORTS = previous.COHORTS
TEST = Path('tests/structure_net/test_ncm_recovered_development_semantics.py')
CPU = study.inputs.BASE/'20261028_recovered_DEV_semantics_CPU_v1.json'
STAGES = ('inference_pilot', 'outputs', 'judge_pilot', 'grade', 'audit')
ROOTS = {a: {s: study.inputs.BASE/f'20261028_recovered_{a}_DEV_{s}_v1' for s in STAGES} for a in ARMS}
COMPARISON = study.inputs.BASE/'20261028_recovered_paired_DEV_comparison_v1'
completed, summarize = previous.completed, previous.summarize
PROOFS = None


def selected_rows(outputs, arm, expected_ids):
    if arm not in ARMS or set(outputs) != set(ARMS): raise ValueError('both fixed readouts required')
    generated.paired.paired_identity(outputs['parent_with_source'], outputs['new_codec_only'])
    rows = outputs[arm]
    if [r['id'] for r in rows] != expected_ids: raise ValueError('complete ordered source population required')
    for a,b in zip(outputs['parent_with_source'], outputs['new_codec_only'], strict=True):
        if any(a[k] != b[k] for k in ('id','source_sha256','split','feature_sha256','source_positions')):
            raise ValueError('paired readout source identity differs')
    return deepcopy(rows)


def compare(new, parent):
    scores = dict(zip(ARMS, (new,parent)))
    summaries = {a: summarize(scores[a]) for a in ARMS}
    paired = dict(both_pass=0, both_fail=0, new_only=0, parent_only=0, unresolved=0)
    for cohort in COHORTS:
        left, right = new[cohort]['secondary'], parent[cohort]['secondary']
        if [r['id'] for r in left] != [r['id'] for r in right]:
            raise ValueError('same ordered source cases required for paired semantic comparison')
        for a,b in zip(left,right,strict=True):
            key = ('unresolved' if not a['valid'] or not b['valid'] else
                   'both_pass' if a['passed'] and b['passed'] else 'new_only' if a['passed'] else
                   'parent_only' if b['passed'] else 'both_fail')
            paired[key] += 1
    selected = next((a for a in ARMS if summaries[a]['DEV_qualified']), None)
    return {'arms': summaries, 'paired_Terra': paired,
        'additional_passes_new_decoder': None if paired['unresolved'] else
            summaries['new_codec_only']['passed']-summaries['parent_with_source']['passed'],
        'selected_DEV_candidate': selected, 'selection_order': list(ARMS),
        'selection_uses_fixed_order_not_largest_observed_score': True,
        'final_task_goal_achieved': False, 'final_test_opened': False, 'speed_tested': False}


def freeze(root, stage, arm, training, environment, cache, trained, proofs, extra):
    for path in extra: proofs.bind(path)
    files = {Path(__file__).resolve(), TEST.resolve(), generated.PLAN.resolve()}
    files.update(p.resolve() for p in training.native.vendor.VENDOR.rglob('*.py'))
    for module in tuple(sys.modules.values()):
        filename = getattr(module,'__file__',None)
        if filename:
            p = Path(filename).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()): files.add(p)
    proof = {'schema': 'nal.ncm-task-supervision-dev.v1', 'hypothesis_id': study.schedule.HYPOTHESIS,
        'arm': arm, 'stage': stage, 'environment': environment,
        'config': {'rubric': training.q.RUBRIC, 'judges': training.q.JUDGES,
            'inherited_first_judgments': len(cache), 'cohorts': list(COHORTS), 'DEV_cases': 128,
            'stored_token_cap': 128, 'metric': 'complete_summary_semantic_pass',
            'source_only': True, 'target_fields_in_prediction': False,
            'checkpoint_sha256': trained['checkpoint_sha256'], 'readout': arm,
            'new_decoder_checkpoint_sha256': generated.DECODER_SHA,
            'model_sha256_identifier': 'exact trained source predictor with original readout before decoder swap',
            'readout_states_bound_separately': True, 'same_source_predicted_codes_both_arms': True,
            'candidate_order': list(ARMS), 'source_predictions_regenerated': False,
            'final_test_opened': False, 'speed_tested': False},
        'source_hashes': {str(p): proofs.sha(p) for p in sorted(files)}, 'input_hashes': dict(proofs.inputs)}
    return previous.write_proof(root, proof, files)


def finish(root, proof, value):
    if PROOFS is None: raise ValueError('complete consumer context required')
    PROOFS.verify(proof['source_hashes']); PROOFS.verify(proof['input_hashes'])
    result = {'schema': proof['schema'], 'hypothesis_id': study.schedule.HYPOTHESIS, 'arm': proof['arm'],
        **value, 'producer_fingerprint': proof['fingerprint'], 'final_test_opened': False, 'speed_tested': False,
        'artifacts': {str(p): PROOFS.sha(p) for p in root.rglob('*.json') if 'producer_source' not in p.parts}}
    data.write(root/'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'artifacts'}), flush=True)


def configure(arm):
    global PROOFS
    if arm not in ARMS: raise ValueError('fixed readout candidate required')
    proofs,training,evaluation,environment,cache,trained,decoder,cohorts,manifest,split = generated.context()
    qualified = proofs.read(CPU); proofs.verify(qualified['artifacts'])
    if qualified['actual_tool_exit_code'] != 0 or qualified['tests_passed'] != 4:
        raise ValueError('paired export and semantic decision contracts must pass')
    preflight = proofs.completed(generated.ROOTS['preflight'])
    if (preflight['checkpoint_sha256'] != trained['checkpoint_sha256']
            or proofs.read(generated.ROOTS['preflight']/'judgment_cache.json') != cache):
        raise ValueError('original paired generation and fixed semantic ancestry required')
    for stage in ('inference_pilot','outputs'):
        root = generated.ROOTS[stage]; value = proofs.completed(root)
        if (value['status'] != 'complete_paired_source_only_'+stage
                or not value['all_code_pairs_identical'] or not value['exact_parent_restore']
                or value['model_sha256'] != trained['final_model_sha256']
                or value['outputs_sha256'] != proofs.sha(root/'outputs.json')
                or value['peak_GPU_allocated_bytes'] > study.CONFIG.peak_limit_bytes):
            raise ValueError('complete exact paired source-only generation required')
    if arm == ARMS[1]:
        first = ROOTS[ARMS[0]]
        audited = completed(first['audit'])
        cache = previous.history.parent.original.append_grade(cache, first['grade'], evaluation)
        if audited['canonical_first_judgments'] != len(cache):
            raise ValueError('completed first readout semantic branch required')
        for stage in ('grade','audit'): proofs.completed(first[stage])
        for name in ('judgment_cache.json','exact_packets.json'): proofs.bind(first['grade']/name)
    ancestor = previous.BOOT[3]; ancestor.audit_cache(cache)
    PROOFS = proofs
    evaluation.lineage = SimpleNamespace(COUNT=len(cache), base_cache=lambda:deepcopy(cache),
        inputs=lambda:[Path(p) for p in proofs.inputs], configure=ancestor.configure,
        require_judges=ancestor.require_judges, uncertain_allowances=ancestor.uncertain_allowances,
        audit_cache=ancestor.audit_cache)
    evaluation.RUNS = ROOTS[arm]
    evaluation.sources = lambda:(cohorts,manifest,split)
    evaluation.freeze = lambda destination,stage,extra:freeze(destination,stage,arm,training,
        environment,cache,trained,proofs,extra)
    return training,evaluation,{'root':study.ROOTS['train'],'result':trained},environment


def export(arm, stage):
    """Materialize the original single-readout schema without a model call."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only saved-output export required')
    training,evaluation,trained,_ = configure(arm)
    cohorts,manifest,split = evaluation.sources()
    chosen = evaluation.old.inference_probes(cohorts,manifest) if stage == 'inference_pilot' else [
        r for c in COHORTS for r in cohorts[c]]
    upstream = generated.ROOTS[stage]; value = PROOFS.completed(upstream)
    outputs = selected_rows(PROOFS.read(upstream/'outputs.json'),arm,[r['id'] for r in chosen])
    from experiments.neural_architecture_lab.finalize_ncm_lm1b_native_development import replay_outputs
    replay_outputs(outputs,chosen,manifest,training.native.tokenizer())
    if stage == 'outputs':
        completed(ROOTS[arm]['inference_pilot'])
        PROOFS.completed(ROOTS[arm]['inference_pilot'])
        saved = PROOFS.read(ROOTS[arm]['inference_pilot']/'outputs.json')
        by_id = {r['id']:r for r in outputs}
        if any(by_id[r['id']] != r for r in saved): raise ValueError('saved pilot differs from full paired outputs')
    destination = ROOTS[arm][stage]
    proof = evaluation.freeze(destination,stage,[upstream/'outputs.json',upstream/'result.json',upstream/'actual_tool_receipt.json'])
    data.write(destination/'outputs.json',outputs); data.write(destination/'split_proof.json',split)
    finish(destination,proof,{'status':'paired_readout_exported_without_model_calls', 'cases':len(outputs),
        'outputs_sha256':PROOFS.sha(destination/'outputs.json'),'model_sha256':trained['result']['final_model_sha256'],
        'source_prediction_artifact':str(upstream/'outputs.json'),'source_predictions_regenerated':False,
        'peak_allocated_bytes':value['peak_GPU_allocated_bytes'],'memory_measurement_from_upstream_generation':True,
        'new_model_calls':0,'new_judge_requests':0,'task_accuracy':None})


def audit(arm):
    from experiments.neural_architecture_lab import audit_ncm_task_supervision_development as shared
    shared.dev = sys.modules[__name__]
    shared.audit(arm)


def comparison():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only paired semantic replay required')
    _,evaluation,_,_ = configure(ARMS[1])
    scores = {}; paths = []
    for arm in ARMS:
        root = ROOTS[arm]['audit']; completed(root)
        scores[arm] = PROOFS.read(root/'scores.json')
        paths += [root/n for n in ('producer.json','result.json','actual_tool_receipt.json','scores.json')]
    result = compare(scores[ARMS[0]],scores[ARMS[1]])
    proof = evaluation.freeze(COMPARISON,'paired_comparison',paths)
    finish(COMPARISON,proof,{'status':'fixed_paired_semantic_comparison_replayed', **result,
        'task_accuracy':None,'task_accuracy_is_per_readout':True,'new_judge_requests':0})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=(*STAGES,'comparison')); parser.add_argument('--arm',choices=ARMS)
    args = parser.parse_args()
    if args.stage != 'comparison' and args.arm is None: parser.error('fixed readout arm required')
    if args.stage == 'comparison' and args.arm is not None: parser.error('comparison uses both fixed readouts')
    root = COMPARISON if args.stage == 'comparison' else ROOTS[args.arm][args.stage]
    if root.exists(): raise ValueError('fresh append-only semantic stage required')
    try:
        if args.stage in ('inference_pilot','outputs'): export(args.arm,args.stage)
        elif args.stage == 'audit': audit(args.arm)
        elif args.stage == 'comparison': comparison()
        else:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only grading required')
            with (data.ROOT/'task_supervision_semantic_grade.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                _,evaluation,_,_ = configure(args.arm)
                evaluation.grade(args.stage == 'judge_pilot')
    except Exception as error:
        if root.exists() and not (root/'failure.json').exists():
            data.write(root/'failure.json',{'type':type(error).__name__,'message':str(error),
                'preserve_all_first_responses':True,'automatic_retry':False})
        raise
