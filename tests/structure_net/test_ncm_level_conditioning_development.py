"""Reject stale checkpoints, incorrect precision and target-bearing evaluation."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from experiments.structure_net import ncm_level_conditioning_development as run


def metadata():
    scaler = {'scale': 128., '_growth_tracker': 1024}
    state = {'status': 'level_conditioning_FP16_native655_training_completed_source_only_DEV_pending',
        'updates': 1024, 'train_cases': 655, 'trainable_parameters': 127732736,
        'strict_reload_passed': True, 'root_queries_updated': True, 'native16_or_pilot_weights_reused': False,
        'initial_trainable_sha256': run.training.capacity.INITIAL, 'final_trainable_sha256': 'new-full-state',
        'protected_codec_hashes': {'encoder': run.training.lifecycle.ENCODER, 'VQ': run.training.lifecycle.VQ},
        'peak_allocated_bytes': 3_600_000_000, 'feature_cache': {'peak_resident_bytes': 500_000_000},
        'precision': 'FP32weights/FP16training/FP32hardinference', 'physical_GPU': 2,
        'scaler': scaler, 'source_disjoint_task_accuracy': None, 'speed_tested': False,
        'all_final_sources_closed': True, 'producer_fingerprint': 'new-producer'}
    saved = {'step': 1024, 'config': asdict(run.training.Config()), 'scaler': scaler,
        'producer_fingerprint': 'new-producer'}
    return state, saved


def test_stale_or_incomplete_or_wrong_precision_checkpoint_rejected():
    state, saved = metadata()
    run.checkpoint_metadata(state, saved)
    for patch in ({'status': 'native655_training_completed_source_only_DEV_pending'}, {'updates': 2},
            {'train_cases': 16}, {'precision': 'BF16'}, {'physical_GPU': 0},
            {'native16_or_pilot_weights_reused': True}, {'strict_reload_passed': False},
            {'root_queries_updated': False}, {'scaler': {'scale': 64., '_growth_tracker': 1024}},
            {'scaler': {'scale': 128., '_growth_tracker': 1023}},
            {'feature_cache': {'peak_resident_bytes': 513 * 1024**2}},
            {'protected_codec_hashes': {'encoder': 'changed', 'VQ': run.training.lifecycle.VQ}}):
        with pytest.raises(ValueError):
            run.checkpoint_metadata({**state, **patch}, saved)
    with pytest.raises(ValueError):
        run.checkpoint_metadata(state, {**saved, 'step': 512})


def test_loads_entire_new_model_and_checks_before_cuda_and_freeze(monkeypatch):
    events = []
    full = {'concepts.generator.weight': 'trained', 'concepts.codec.decoder.weight': 'trained',
        'word_adapters.weight': 'trained'}
    state = {'candidate': 'new1024'}

    class Model:
        def load_state_dict(self, value, strict):
            assert value is full and strict
            events.append('full-state')

        def cuda(self):
            events.append('cuda')
            return self

        def eval(self):
            events.append('eval')
            return self

        def requires_grad_(self, value):
            assert value is False
            events.append('freeze')
            return self

    model = Model()
    monkeypatch.setattr(run, 'checkpoint_contract', lambda: (state, {'model': full}))
    monkeypatch.setattr(run.training.lifecycle.mechanism, 'enable', lambda m: m)
    monkeypatch.setattr(run.training.lifecycle.head, 'from_domain', lambda device: model)
    monkeypatch.setattr(run, 'model_contract', lambda m, s: events.append('hashes') if m is model and s is state else pytest.fail())
    assert run.load_selected() == (model, state)
    assert events == ['full-state', 'hashes', 'cuda', 'eval', 'freeze']


def test_actual_complete_source_boundary_and_same_case_gate():
    cases = run.q.read(run.SOURCE / 'cases.json')
    run.base.validate_sources(cases)
    rows = run.q.read(run.training.DATA / 'cases.json')
    assert run.training.split_guard(rows, cases)['group_id']['overlap'] == 0
    bad = deepcopy(cases)
    bad[run.COHORTS[0]][0]['codes'] = [1]
    with pytest.raises(ValueError):
        run.base.validate_sources(bad)
    for field in run.base.FORBIDDEN:
        with pytest.raises(ValueError, match='source metadata only'):
            run.base.predict(None, {field: []}, {}, None)
    a = [{'id': str(i), 'valid': True, 'passed': i >= 3} for i in range(64)]
    b = [{'id': str(i), 'valid': True, 'passed': i not in (3, 4, 5)} for i in range(64)]
    assert run.joint_gate(a, a)['qualified']
    assert run.joint_gate(a, b)['joint_passed'] == 58
    assert not run.joint_gate(a, b)['qualified']


def test_current_lineage_and_gpu_binding_are_prospective():
    assert run.file_hash(run.lineage.CACHE) == run.lineage.CACHE_SHA
    assert run.file_hash(run.lineage.CAMPAIGN) == run.lineage.CAMPAIGN_SHA
    assert len(run.q.read(run.lineage.CACHE)) == run.lineage.COUNT == 3523
    run.configure()
    assert run.base.training.training.require_gpu is run.training.require_gpu
    assert run.base.load_selected is run.load_selected
    assert run.base.base_cache is run.lineage.base_cache
    assert run.q.JUDGES == {'primary': 'codex_sdk/gpt-5.6-sol', 'secondary': 'codex_sdk/gpt-5.6-terra'}


def test_semantic_launch_waits_for_latest_component_lineage(monkeypatch):
    assert run.SEMANTIC_LINEAGE_READY is True
    monkeypatch.setattr(run, 'SEMANTIC_LINEAGE_READY', False)
    with pytest.raises(ValueError, match='latest component-study lineage'):
        run.workflow()
