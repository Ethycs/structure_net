"""No GPU or corpus access: diagnostic selection, pairing and raw-count contracts."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import ncm_public_training_probe as probe


def cases(n=16):
    return [{'id': str(i), 'split': 'TRAIN', 'oracle_codes': [0]*641,
             'native_word_ids': [101, 5, 101]+[0]*125} for i in range(n)]


def output(row):
    return {'id': row['id'], 'source_id_used': row['id'], 'codes': [0]*641,
            'teacher_codes': [0]*641, 'raw_word_ids': list(row['native_word_ids']),
            'active_word_matches': 3, 'active_words': 3, 'task_accuracy': None,
            'text': 'example', 'levels': [{'positions': n, 'teacher_correct': n,
                'rollout_correct': n, 'CE': .1} for n in probe.LEVELS]}


def test_fixed_selection_preserves_existing_order():
    rows = cases()
    order = list(reversed([r['id'] for r in rows]))
    selected = probe.select(rows, {'diagnostic_ids': order, 'longest_source_id': '7'})
    assert [r['id'] for r in selected] == order
    with pytest.raises(FrozenInstanceError):
        probe.CONFIG.controls = 2


@pytest.mark.parametrize('fault', ['duplicate', 'heldout', 'float_code', 'missing_longest'])
def test_invalid_control_population_rejected(fault):
    rows = cases(); ids = [r['id'] for r in rows]; longest = '15'
    if fault == 'duplicate': ids[-1] = ids[0]
    if fault == 'heldout': rows[3]['split'] = 'DEV'
    if fault == 'float_code': rows[2]['oracle_codes'][0] = 0.0
    if fault == 'missing_longest': longest = 'missing'
    with pytest.raises(ValueError):
        probe.select(rows, {'diagnostic_ids': ids, 'longest_source_id': longest})


def test_teacher_capture_leaves_logits_unchanged_and_restores_on_failure():
    logits = torch.zeros(1, 641, 3); logits[:, :, 2] = 1
    original = lambda: logits
    head = SimpleNamespace(teacher_logits=original)
    with pytest.raises(RuntimeError):
        with probe.capture_teacher(head) as captured:
            assert head.teacher_logits() is logits
            assert captured == [[2]*641]
            raise RuntimeError('deliberate diagnostic interruption')
    assert head.teacher_logits is original


def test_raw_counts_replay_teacher_rollout_gap_and_words():
    rows = cases(1); raw = output(rows[0])
    raw['codes'][5] = 1; raw['levels'][1]['rollout_correct'] = 7
    raw['raw_word_ids'][1] = 6; raw['active_word_matches'] = 2
    scores = probe.score_rows([raw], rows)
    assert scores['teacher_correct_codes'] == 641
    assert scores['rollout_correct_codes'] == 640
    assert scores['teacher_rollout_gap'] == pytest.approx(1/641)
    assert scores['native_word_agreement'] == pytest.approx(2/3)
    assert scores['levels'][1]['rollout_correct'] == 7


@pytest.mark.parametrize('fault', ['code_count', 'word_count', 'nan', 'boolean', 'missing_level', 'wrong_source'])
def test_corrupt_saved_measurements_are_rejected(fault):
    rows = cases(1); raw = output(rows[0])
    if fault == 'code_count': raw['teacher_codes'][0] = 1
    if fault == 'word_count': raw['active_word_matches'] = 2
    if fault == 'nan': raw['levels'][0]['CE'] = float('nan')
    if fault == 'boolean': raw['codes'][0] = False
    if fault == 'missing_level': raw['levels'].pop()
    if fault == 'wrong_source': raw['source_id_used'] = 'different'
    with pytest.raises(ValueError):
        probe.score_rows([raw], rows)


def test_shuffled_source_binding_is_independent_of_target_binding():
    rows = cases(2); outputs = [output(r) for r in rows]
    outputs[0]['source_id_used'] = '1'; outputs[1]['source_id_used'] = '0'
    assert probe.score_rows(outputs, rows, shuffled=True)['rollout_correct_codes'] == 1282
    with pytest.raises(ValueError):
        probe.score_rows(outputs, rows)


def test_archived_control_requires_exact_codes_but_bounded_float_loss():
    actual = output(cases(1)[0]); archived = deepcopy(actual); archived.pop('teacher_codes')
    actual['levels'][0]['CE'] += 1e-7
    probe.compare_archived(actual, archived)
    actual['levels'][0]['CE'] += .001
    with pytest.raises(ValueError): probe.compare_archived(actual, archived)
    actual = output(cases(1)[0]); actual['codes'][0] = 1
    with pytest.raises(ValueError): probe.compare_archived(actual, archived)


def legacy_fixture(tmp_path, monkeypatch, audit_exit=0):
    root, audit, campaign = [tmp_path/n for n in ('parent', 'audit', 'campaign')]
    for path in (root, audit, campaign): path.mkdir()
    for name in ('producer.json', 'result.json', 'model.pt', 'outputs.json'):
        (root/name).write_text('{}')
    (root/'actual_tool_receipt.json').write_text(json.dumps({'session_id': 62755,
        'actual_tool_exit_code': 0, 'tool_chunk': '3b6d46'}))
    (audit/'campaign_results.json').write_text(json.dumps({
        'status': 'broad_generator_training_independently_replayed', 'updates': 4096, 'cases': 1376,
        'artifacts': {str(p): probe.data.sha(p) for p in root.iterdir()}}))
    (audit/'actual_tool_receipt.json').write_text(json.dumps({'actual_tool_exit_code': audit_exit}))
    p = campaign/'campaign_results.json'
    p.write_text(json.dumps({'artifacts': {str(v): probe.data.sha(v) for v in audit.iterdir()}}))
    monkeypatch.setattr(probe, 'PARENTS', {'BROAD': root})
    monkeypatch.setattr(probe, 'BROAD_RESULTS', campaign)
    monkeypatch.setattr(probe, 'LEGACY_AUDIT', audit)
    monkeypatch.setattr(probe, 'BROAD_CAMPAIGN_SHA', probe.data.sha(p))
    return root, p


def test_legacy_parent_requires_the_published_audit_chain(tmp_path, monkeypatch):
    root, _ = legacy_fixture(tmp_path, monkeypatch)
    probe.receipt(root)


@pytest.mark.parametrize('fault', ['stale_campaign', 'mutated_parent', 'failed_audit', 'other_root'])
def test_legacy_chain_rejects_broken_or_unrelated_evidence(tmp_path, monkeypatch, fault):
    root, campaign = legacy_fixture(tmp_path, monkeypatch, audit_exit=int(fault == 'failed_audit'))
    if fault == 'stale_campaign': campaign.write_text(campaign.read_text()+' ')
    if fault == 'mutated_parent': (root/'result.json').write_text('{"changed":true}')
    if fault == 'other_root': root = tmp_path/'unrelated'
    with pytest.raises(ValueError): probe.legacy_broad_receipt(root)


def test_new_stages_still_require_artifact_bound_receipts(tmp_path):
    for name in ('producer.json', 'result.json'): (tmp_path/name).write_text('{}')
    (tmp_path/'actual_tool_receipt.json').write_text('{"actual_tool_exit_code":0}')
    with pytest.raises(ValueError): probe.receipt(tmp_path)
