"""Source-only semantic DEV for fixed trained tree and dense source attention."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from experiments.structure_net import ncm_source_hierarchy_training as study
from experiments.structure_net import ncm_task_supervision_development as previous
from experiments.structure_net import ncm_recovered_review_lineage as history

data, base = study.data, study.task.base
PLAN = Path('docs/07 - Status Reports/2026-10-27_ncm-source-hierarchy-evaluation.md')
TEST = Path('tests/structure_net/test_ncm_source_hierarchy_development.py')
QUALIFICATION = data.ROOT/'20261027_source_hierarchy_DEV_CPU_v1.json'
COHORTS = previous.COHORTS
STAGES = ('preflight', *previous.STAGES)
ROOTS = {a: {s: data.ROOT/f'20261027_source_hierarchy_{a}_DEV_{s}_v1' for s in STAGES}
        for a in study.ARMS}
completed, summarize = previous.completed, previous.summarize


def check_training(value, audited, arm):
    if (arm not in study.ARMS or value['arm'] != arm or audited['arm'] != arm
            or value['status'] != 'source_hierarchy_training_completed'
            or audited['status'] != 'source_hierarchy_training_replayed'
            or value['stage'] != 'train' or value['config'] != asdict(study.CONFIG)
            or value['updates'] != 2048 or value['cases'] != 1376
            or value['trainable_parameters'] != study.mechanism.PARAMETERS[arm]
            or not value['strict_final_reload_passed'] or not value['protected_tensors_unchanged']
            or value['systems_lifecycle_only_not_quality_evidence']
            or not audited['all_corruption_counts_replayed']
            or value['checkpoint_sha256'] != audited['checkpoint_sha256']):
        raise ValueError('exact completed architecture and audited checkpoint required')
    expected = {'presentations': 65536, 'noisy_presentations': 32768,
                'distinct_sources_visited': 1376}
    if any(value['measured_exposure'].get(k) != v or audited.get(k) != v for k, v in expected.items()):
        raise ValueError('complete fixed same-source exposure required')


def bound_training(arm):
    root, audit = (study.ROOTS[arm, s] for s in ('train', 'audit'))
    value, audited = completed(root), completed(audit)
    check_training(value, audited, arm)
    if (value['checkpoint_sha256'] != data.sha(root/'model.pt')
            or data.read(root/'producer.json')['parent_checkpoint_sha256'] != study.task.schedule.PARENT_CHECKPOINT_SHA):
        raise ValueError('fixed final checkpoint and epoch2 initializer required')
    paths = [r/n for r in (root, audit) for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
    paths += [root/'model.pt', root/'architecture.json']
    return {'root': root, 'result': value}, paths


def current_history(arm):
    history.require_preflight(); history.install()
    training, evaluation, environment, ancestor, cache, paths = previous.current_history()
    # Both earlier task arms must be closed: never omit an in-flight judge branch.
    preceding = [previous.ROOTS[a] for a in ('existing', 'expanded')]
    if arm == 'dense': preceding.append(ROOTS['tree'])
    for stages in preceding:
        audited = completed(stages['audit'])
        cache = previous.history.parent.original.append_grade(cache, stages['grade'], evaluation)
        if audited['canonical_first_judgments'] != len(cache):
            raise ValueError('preceding semantic audit and inherited firsts differ')
        paths += [stages[s]/n for s in ('grade', 'audit')
                  for n in ('producer.json', 'result.json', 'actual_tool_receipt.json')]
        paths += [stages['grade']/n for n in ('judgment_cache.json', 'exact_packets.json')]
    ancestor.audit_cache(cache)
    return training, evaluation, environment, ancestor, cache, paths


def freeze(root, stage, arm, training, environment, paths, inherited):
    if root.exists(): raise ValueError('fresh append-only evaluation required')
    files = {Path(__file__).resolve(), PLAN.resolve(), TEST.resolve(), study.PLAN.resolve()}
    files.update(p.resolve() for p in training.native.vendor.VENDOR.rglob('*.py'))
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            p = Path(filename).resolve()
            if p.is_file() and p.suffix == '.py' and p.is_relative_to(Path('experiments').resolve()): files.add(p)
    proof = {'schema': 'nal.ncm-task-supervision-dev.v1', 'hypothesis_id': study.HYPOTHESIS,
        'stage': stage, 'arm': arm, 'environment': environment,
        'config': {'source_only': True, 'target_fields_in_prediction': False, 'DEV_cases': 128,
            'cohorts': list(COHORTS), 'rubric': training.q.RUBRIC, 'judges': training.q.JUDGES,
            'inherited_first_judgments': inherited, 'metric': 'complete_summary_semantic_pass',
            'stored_token_cap': 128, 'architecture': arm, 'source_attention_blocks': 12,
            'trainable_parameters': study.mechanism.PARAMETERS[arm],
            'memory_excludes_live_Llama_encoder': True, 'final_test_opened': False, 'speed_tested': False},
        'source_hashes': {str(p): data.sha(p) for p in sorted(files)},
        'input_hashes': {str(p): data.sha(p) for p in dict.fromkeys([*paths, Path('pixi.lock'), Path('pyproject.toml')])}}
    return previous.write_proof(root, proof, files)


def finish(root, proof, value):
    base.verify_proof(proof)
    files = [p for p in root.rglob('*.json') if 'producer_source' not in p.parts]
    result = {'schema': proof['schema'], 'hypothesis_id': study.HYPOTHESIS, 'arm': proof['arm'], **value,
        'final_test_opened': False, 'speed_tested': False, 'producer_fingerprint': proof['fingerprint'],
        'artifacts': {str(p): data.sha(p) for p in files}}
    data.write(root/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'artifacts'}), flush=True)


def context(arm):
    trained, paths = bound_training(arm)
    training, evaluation, environment, ancestor, cache, history_paths = current_history(arm)
    rows, _, _, source_paths = study.task.dataset('existing')
    paths += history_paths + source_paths + [evaluation.old.SOURCE/'cases.json', evaluation.old.FEATURES/'manifest.json']
    return training, evaluation, trained, environment, ancestor, cache, paths, rows


def preflight(arm):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only preflight required')
    checked = data.read(QUALIFICATION); data.verify(checked['artifacts'])
    if checked['actual_tool_exit_code'] != 0 or checked['tests_passed'] != 3:
        raise ValueError('successful architecture loading and admission tests required')
    training, evaluation, trained, environment, _, cache, paths, rows = context(arm)
    paths.append(QUALIFICATION)
    cohorts, _, split = previous.sources(evaluation, rows)
    if sum(map(len, cohorts.values())) != 128: raise ValueError('fixed128 source-only DEV required')
    root = ROOTS[arm]['preflight']
    proof = freeze(root, 'preflight', arm, training, environment, paths, len(cache))
    data.write(root/'judgment_cache.json', cache); data.write(root/'split_proof.json', split)
    finish(root, proof, {'status': 'source_hierarchy_semantic_preflight_passed', 'cases': 128,
        'canonical_first_judgments': len(cache), 'checkpoint_sha256': trained['result']['checkpoint_sha256'],
        'new_judge_requests': 0, 'task_accuracy': None})


def configure(arm):
    checked = completed(ROOTS[arm]['preflight'])
    training, evaluation, trained, environment, ancestor, cache, paths, rows = context(arm)
    root = ROOTS[arm]['preflight']
    if (checked['status'] != 'source_hierarchy_semantic_preflight_passed'
            or cache != data.read(root/'judgment_cache.json')
            or checked['checkpoint_sha256'] != trained['result']['checkpoint_sha256']):
        raise ValueError('fixed preflight model and complete judgment history required')
    paths += [root/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'judgment_cache.json', 'split_proof.json')]
    evaluation.lineage = SimpleNamespace(COUNT=len(cache), base_cache=lambda: deepcopy(cache), inputs=lambda: list(paths),
        configure=ancestor.configure, require_judges=ancestor.require_judges,
        uncertain_allowances=ancestor.uncertain_allowances, audit_cache=ancestor.audit_cache)
    evaluation.RUNS = ROOTS[arm]
    evaluation.sources = lambda: previous.sources(evaluation, rows)
    evaluation.freeze = lambda destination, stage, extra: freeze(destination, stage, arm, training,
        environment, [*paths, *extra], len(cache))
    return training, evaluation, trained, environment


def load_selected(training, trained, arm):
    model, _ = study.load_initial(training, arm)
    saved = torch.load(trained['root']/'model.pt', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(saved['model'], strict=True); del saved
    model.eval().requires_grad_(False)
    if (training.native.vendor.module_hash(model) != trained['result']['final_model_sha256']
            or training.prior.protected(model) != trained['result']['frozen_hashes']):
        model.close(); raise ValueError('exact full trained model and unchanged reconstruction path required')
    return model


def generate(arm, pilot):
    training, evaluation, trained, _ = configure(arm)
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID'
            or torch.cuda.get_device_name() != 'NVIDIA GeForce RTX 2060'):
        raise ValueError('same physical GPU1 RTX2060 required for both architecture arms')
    cohorts, manifest, split = evaluation.sources()
    chosen = evaluation.old.inference_probes(cohorts, manifest) if pilot else [r for c in COHORTS for r in cohorts[c]]
    stage = 'inference_pilot' if pilot else 'outputs'; root = ROOTS[arm][stage]
    paths = [Path(manifest[r['id']]['path']) for r in chosen]; before = {}
    if not pilot:
        completed(ROOTS[arm]['inference_pilot'])
        before = {r['id']: r for r in data.read(ROOTS[arm]['inference_pilot']/'outputs.json')}
        paths += [ROOTS[arm]['inference_pilot']/n for n in ('producer.json', 'result.json', 'actual_tool_receipt.json', 'outputs.json')]
    proof = evaluation.freeze(root, stage, paths); data.write(root/'split_proof.json', split)
    torch.cuda.reset_peak_memory_stats(); model = load_selected(training, trained, arm)
    tokenizer = training.native.tokenizer(); outputs = []
    try:
        for row in chosen:
            output = evaluation.predict(model, row, manifest, tokenizer)
            if pilot and evaluation.predict(model, row, manifest, tokenizer) != output: raise ValueError('inference must repeat exactly')
            if row['id'] in before and before[row['id']] != output: raise ValueError('full output differs from pilot')
            outputs.append(output); study.broad.write_json(root/'outputs.json', outputs)
            if torch.cuda.max_memory_allocated() > study.CONFIG.peak_limit_bytes: raise ValueError('4.9GB inference allocation exceeded')
            if pilot or len(outputs) % 16 == 0: print(json.dumps({'generated': len(outputs), 'total': len(chosen)}), flush=True)
        if training.native.vendor.module_hash(model) != trained['result']['final_model_sha256']: raise ValueError('inference changed model')
        finish(root, proof, {'status': 'complete_source_only_'+stage, 'cases': len(outputs),
            'outputs_sha256': data.sha(root/'outputs.json'), 'model_sha256': trained['result']['final_model_sha256'],
            'source_disjoint': True, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'task_accuracy': None})
    finally: model.close()


def audit(arm):
    from experiments.neural_architecture_lab import audit_ncm_task_supervision_development as shared
    # Adapt the consumer interface only; no shared producer or source is modified.
    shared.dev = sys.modules[__name__]
    shared.audit(arm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--arm', choices=study.ARMS, required=True); args = parser.parse_args()
    root = ROOTS[args.arm][args.stage]
    if root.exists(): raise ValueError('fresh append-only stage required')
    try:
        if args.stage == 'preflight': preflight(args.arm)
        elif args.stage in ('inference_pilot', 'outputs'): generate(args.arm, args.stage == 'inference_pilot')
        elif args.stage == 'audit': audit(args.arm)
        else:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only grading required')
            with (data.ROOT/'task_supervision_semantic_grade.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                _, evaluation, _, _ = configure(args.arm)
                evaluation.grade(args.stage == 'judge_pilot')
    except Exception as error:
        if root.exists() and not (root/'failure.json').exists():
            data.write(root/'failure.json', {'type': type(error).__name__, 'message': str(error),
                'preserve_all_first_responses': True, 'automatic_retry': False})
        raise
