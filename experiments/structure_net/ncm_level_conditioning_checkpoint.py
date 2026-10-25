"""Frozen new FP16 checkpoint contract, independent of semantic cache state."""
from dataclasses import asdict
from pathlib import Path
import torch
from experiments.structure_net import ncm_level_conditioning_training as training
from experiments.structure_net.ncm_memory_data import file_hash
native, q, ROOT = training.native, training.q, training.ROOT
TRAIN_AUDIT = ROOT / '20261024_level_conditioning_FP16_native655_training_audit_v1'


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


