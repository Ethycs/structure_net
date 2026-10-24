"""Source-only evaluation of the full new FP16 level-conditioning checkpoint."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

from experiments.structure_net import ncm_lm1b_native_development as base
from experiments.structure_net import ncm_level_conditioning_training as training
from experiments.structure_net import ncm_current_judgment_lineage as lineage
from experiments.structure_net.ncm_memory_data import file_hash, write_json

native, q, ROOT = training.native, training.q, training.ROOT
HYPOTHESIS = 'ncm-level-conditioning-native655-source-only-DEV-v1'
MODULE = 'experiments.structure_net.ncm_level_conditioning_development'
AUDITOR_MODULE = 'experiments.neural_architecture_lab.finalize_ncm_level_conditioning_development'
AUDITOR = Path('experiments/neural_architecture_lab/finalize_ncm_level_conditioning_development.py')
PLAN = Path('docs/07 - Status Reports/2026-10-24_ncm-level-conditioning-development-plan.md')
TEST = Path('tests/structure_net/test_ncm_level_conditioning_development.py')
REPORT = Path('docs/08 - Analysis/2026-10-24_ncm-level-conditioning-development.md')
PREFIX = '20261024_level_conditioning_native655_development_'
WORKFLOW, METADATA, PILOT, OUTPUTS, PACKETS, JUDGE_PILOT, GRADE, RESULTS = (
    ROOT / (PREFIX + name + '_v1') for name in ('workflow', 'metadata', 'inference_pilot',
        'outputs', 'packets', 'judge_pilot', 'grade', 'results'))
TRAIN_AUDIT = ROOT / '20261024_level_conditioning_FP16_native655_training_audit_v1'
SCHEMA = 'nal.ncm-level-conditioning-native655-source-only-dev.v1'
head, semantic, reference, oracle = training.lifecycle.head, base.semantic, base.reference, base.oracle
SOURCE, FEATURES, SOURCE_SHA, FEATURE_SHA, COHORTS = base.SOURCE, base.FEATURES, base.SOURCE_SHA, base.FEATURE_SHA, base.COHORTS
OLD_CACHE, CACHE_SHA = lineage.CACHE, lineage.CACHE_SHA
source_inputs, decode, inference_probes = base.source_inputs, base.decode, base.inference_probes
task_metric, joint_gate, secondary_admitted = base.task_metric, base.joint_gate, base.secondary_admitted
base_cache = lineage.base_cache


def checkpoint_metadata(state, saved):
    if (state['status'] != 'level_conditioning_FP16_native655_training_completed_source_only_DEV_pending'
            or state['updates'] != 1024 or state['train_cases'] != 655
            or state['trainable_parameters'] != 127732736
            or not state['strict_reload_passed'] or not state['root_queries_updated']
            or state['native16_or_pilot_weights_reused']
            or state['initial_trainable_sha256'] != training.capacity.INITIAL
            or state['final_trainable_sha256'] == training.capacity.INITIAL
            or state['protected_codec_hashes'] != {'encoder': training.lifecycle.ENCODER, 'VQ': training.lifecycle.VQ}
            or state['peak_allocated_bytes'] > 4_900_000_000
            or state['feature_cache']['peak_resident_bytes'] > 512 * 1024**2
            or state['precision'] != 'FP32weights/FP16training/FP32hardinference'
            or state['physical_GPU'] != 2 or state['scaler']['scale'] != 128.
            or state['scaler']['_growth_tracker'] != 1024
            or saved['scaler'] != state['scaler']
            or state['source_disjoint_task_accuracy'] is not None or state['speed_tested']
            or not state['all_final_sources_closed']
            or saved['step'] != 1024 or saved['config'] != asdict(training.Config())
            or saved['producer_fingerprint'] != state['producer_fingerprint']):
        raise ValueError('new fixed1024 full-state FP16/scaler/655 checkpoint contract differs')


def checkpoint_contract():
    native.vendor.terminal(training.TRAIN)
    native.vendor.verify(q.read(training.WORKFLOW / 'producer.json'))
    whole = q.read(training.WORKFLOW / 'result.json')
    state = q.read(training.TRAIN / 'result.json')
    if (whole['status'] != 'level_conditioning_FP16_native655_training_workflow_completed_source_only_DEV_pending'
            or q.read(training.WORKFLOW / 'tool_terminal_receipt.json')['exit_code'] != 0
            or [r['stage'] for r in whole['stages']] != ['prepare', 'pilot', 'train']
            or any(r['exit_code'] != 0 for r in whole['stages'])
            or file_hash(training.TRAIN / 'model.pt') != state['checkpoint_sha256']
            or q.read(TRAIN_AUDIT / 'actual_tool_terminal_receipt.json')['exit_code'] != 0):
        raise ValueError('actual new whole training and independent CPU audit exit0 required')
    audit = q.read(TRAIN_AUDIT / 'result.json')
    if (audit['status'] != 'level_conditioning_FP16_native655_actual_trace_state_and_scaler_replayed_DEV_pending'
            or audit['results']['full']['updates'] != 1024
            or audit['results']['full']['checkpoint_sha256'] != state['checkpoint_sha256']):
        raise ValueError('actual full level-conditioning trace/state/scaler audit differs')
    for path, expected in audit['artifacts'].items():
        if file_hash(path) != expected:
            raise ValueError('new audited training evidence changed: ' + path)
    saved = torch.load(training.TRAIN / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    checkpoint_metadata(state, saved)
    return state, saved


def model_contract(model, state):
    if (training.lifecycle.trainable_hash(model) != state['final_trainable_sha256']
            or sum(p.numel() for p in model.parameters() if p.requires_grad) != 127732736
            or training.lifecycle.protected(model) != {'encoder': training.lifecycle.ENCODER, 'VQ': training.lifecycle.VQ}
            or training.attention_hash(model) != state['final_attention_sha256']
            or native.vendor.module_hash(model.concepts.generator) != state['final_native_hashes']['generator']
            or native.vendor.module_hash(model.codec.decoder) != state['final_native_hashes']['word_decoder']):
        raise ValueError('complete new generator/decoder/attention or protected-state hash differs')


def load_selected():
    state, saved = checkpoint_contract()
    model = training.lifecycle.mechanism.enable(training.lifecycle.head.from_domain(device='cpu'))
    model.load_state_dict(saved['model'], strict=True)
    model_contract(model, state)
    return model.cuda().eval().requires_grad_(False), state


def configure():
    lineage.configure()
    for name in ('WORKFLOW', 'METADATA', 'PILOT', 'OUTPUTS', 'PACKETS', 'JUDGE_PILOT', 'GRADE', 'RESULTS', 'REPORT'):
        setattr(base, name, globals()[name])
    base.HYPOTHESIS, base.TRAIN_AUDIT = HYPOTHESIS, TRAIN_AUDIT
    # The imported inference body expects training.training.require_gpu. Bind its
    # namespace explicitly to the new physical2 check, without editing or
    # rebinding any module used by the live producer.
    base.training = SimpleNamespace(**{**vars(training), 'training': SimpleNamespace(require_gpu=training.require_gpu)})
    base.OLD_CACHE, base.CACHE_SHA, base.CACHE_COUNT = lineage.CACHE, lineage.CACHE_SHA, lineage.COUNT
    base.checkpoint_contract, base.model_contract, base.load_selected = checkpoint_contract, model_contract, load_selected
    base.producing, base.base_cache = producing, lineage.base_cache
    base.configure = configure
    base.semantic.require_judges, base.semantic.audit_cache = lineage.require_judges, lineage.audit_cache
    # Process-local binding only; no frozen source or training process is edited.


def producing(root, stage, inputs=(), config=None):
    files = [Path(__file__), PLAN, TEST, AUDITOR, Path(base.__file__), Path(lineage.__file__),
        Path(lineage.historical.__file__), Path(training.__file__), training.PLAN, training.AUDITOR,
        Path(training.lifecycle.__file__), Path(training.lifecycle.mechanism.__file__),
        Path(training.lifecycle.head.__file__), Path(training.lifecycle.head.concepts.__file__),
        Path(base.framed.__file__), Path(base.oracle.__file__), Path(base.semantic.__file__),
        Path(base.reference.first.__file__), Path(base.reference.profile.__file__),
        base.reference.profile.PLAN, *sorted(native.vendor.VENDOR.rglob('*.py'))]
    common = [*lineage.inputs(), base.SOURCE / 'cases.json', base.FEATURES / 'manifest.json',
        TRAIN_AUDIT / 'result.json', TRAIN_AUDIT / 'actual_tool_terminal_receipt.json']
    cfg = {**(config or {}), 'schema': SCHEMA, 'hypothesis_id': HYPOTHESIS,
        'source_only': True, 'source_prediction': True, 'oracle_target_codes': False,
        'target_fields_in_prediction': False, 'all128_cases': True, 'date_string': '23 Oct 2026',
        'training_cases': 655, 'native_trainable_parameters_during_training': 127732736,
        'native_generator_and_word_decoder_trained': True, 'native_encoder_and_VQ_frozen': True,
        'original_Llama_body_and_MLPs_frozen': True, 'all_source_tokens_retained': True,
        'checkpoint_selection': 'fixed final1024 FP16 level-conditioning only',
        'training_precision': 'FP16', 'precision': 'FP32', 'TF32': False, 'physical_GPU': 2,
        'native_levels': list(native.LEVELS), 'raw_words': 128, 'word_vocabulary': 30528,
        'full_concepts': 641, 'output_policy': 'native CLS renderer; full-vocabulary argmax; no repair or masking',
        'canonical_prior_firsts': lineage.COUNT, 'rubric': q.RUBRIC, 'judges': q.JUDGES,
        'source_disjoint_task_accuracy': None, 'task_accuracy': None, 'final_test_opened': False, 'speed_tested': False,
        'comparison_confounded_by_noise_precision_microbatch_and_seed': True}
    p = native.vendor.freeze(root, stage, list(dict.fromkeys([*inputs, *common, *files])), cfg)
    p['hypothesis_id'] = HYPOTHESIS
    for i, path in enumerate(files):
        p['source_hashes'][str(path)] = file_hash(path)
        (root / 'producer_source' / f'level_dev_{i}_{path.name}').write_bytes(path.read_bytes())
    p['fingerprint'] = q.digest({k: v for k, v in p.items() if k != 'fingerprint'})
    write_json(root / 'producer.json', p)
    return p


def workflow():
    configure()
    checkpoint_contract()
    lineage.require_judges()
    lineage.base_cache()
    roots = (WORKFLOW, METADATA, PILOT, OUTPUTS, PACKETS, JUDGE_PILOT, GRADE, RESULTS)
    if any(root.exists() for root in roots) or REPORT.exists():
        raise ValueError('fresh complete source-only level-conditioning DEV roots required')
    p = producing(WORKFLOW, 'supervise_complete128_source_only_DEV')
    receipts = []
    stages = [(MODULE, 'prepare', METADATA, ''), (MODULE, 'inference-pilot', PILOT, '2'),
        (MODULE, 'generate', OUTPUTS, '2'), (MODULE, 'packets', PACKETS, ''),
        (MODULE, 'judge-pilot', JUDGE_PILOT, ''), (MODULE, 'grade', GRADE, ''),
        (AUDITOR_MODULE, 'finalize', RESULTS, ''), (AUDITOR_MODULE, 'register', None, '')]
    for module, stage, root, gpu in stages:
        native.vendor.verify(p)
        command = [sys.executable, '-u', '-m', module, stage]
        child = subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu})
        write_json(WORKFLOW / 'progress.json', {'active_stage': stage, 'child_pid': child.pid, 'stages': receipts})
        code = child.wait()
        receipt = {'stage': stage, 'exit_code': code, 'command': command, 'child_pid': child.pid,
            'observer_pid': os.getpid(), 'observer': 'subprocess.wait'}
        write_json(WORKFLOW / (stage + '_receipt.json'), receipt)
        if root is not None and root.exists():
            write_json(root / 'worker_terminal_receipt.json', receipt)
        receipts.append(receipt)
        if code:
            raise RuntimeError('level-conditioning source-only DEV stage failed: ' + stage)
    native.vendor.verify(p)
    write_json(WORKFLOW / 'result.json', {'status': 'level_conditioning_source_only_DEV_workflow_completed', 'stages': receipts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('prepare', 'inference-pilot', 'generate', 'packets', 'judge-pilot', 'grade', 'workflow'))
    args = parser.parse_args()
    configure()
    {'prepare': base.prepare, 'inference-pilot': lambda: base.generate(True), 'generate': base.generate,
        'packets': base.packets, 'judge-pilot': lambda: base.grade(True), 'grade': base.grade, 'workflow': workflow}[args.stage]()
