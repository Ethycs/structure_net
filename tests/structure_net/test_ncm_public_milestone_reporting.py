from copy import deepcopy
import json

import pytest

from experiments.neural_architecture_lab import finalize_ncm_public_curriculum as final
from experiments.neural_architecture_lab import register_ncm_public_curriculum as registry


def milestone(epoch, passed, joint=None, valid=128):
    joints = {str(i): {'cases': 64, 'joint_passed': n, 'all_valid': True, 'qualified': n >= 61}
              for i, n in enumerate(joint)} if joint else None
    classification = final.classify(valid, passed, joints is not None, joints)
    return {'epoch': epoch, 'attempt': 1, 'valid': valid, 'passed': passed,
        'primary_admitted': joints is not None, 'joint': joints,
        'classification': classification, 'DEV_qualified': classification == 'DEV_qualified',
        'task_accuracy': passed/128 if valid == 128 else None,
        'joint_task_accuracy': sum(joint)/128 if joint else None}


@pytest.mark.parametrize('rows,expected', [
    ([milestone(1, 0)], None),
    ([milestone(1, 122, [60, 62])], None),
    ([milestone(1, 122, [61, 61])], 1),
    ([milestone(1, 0), milestone(2, 123, [61, 62])], 2),
    ([milestone(1, 0, valid=127)], None),
])
def test_only_first_joint_qualified_milestone_advances(rows, expected):
    assert final.selection(rows) == expected


@pytest.mark.parametrize('rows', [
    [], [milestone(2, 0)], [milestone(1, 0), milestone(4, 0)],
    [milestone(1, 122, [61, 61]), milestone(2, 128, [64, 64])],
    [milestone(1, 0, valid=127), milestone(2, 128, [64, 64])],
])
def test_rejects_skipped_invalid_or_post_qualification_training(rows):
    with pytest.raises(ValueError): final.selection(rows)


def test_invalid_judgments_never_reduce_the_accuracy_denominator():
    row = milestone(1, 120, valid=127)
    row['task_accuracy'] = 120/127
    with pytest.raises(ValueError): final.selection([row])
    row = milestone(1, 122, [61, 61]); row['joint_task_accuracy'] = 1.
    with pytest.raises(ValueError): final.selection([row])


def test_complete_exposure_rejects_missing_or_double_counted_updates():
    config = final.study.CONFIG
    row = {'first_completed_update': 0, 'last_completed_update': config.updates_per_epoch,
        'presentations': {'public': config.public_cases, 'anchor': config.anchors_per_epoch},
        'noisy_presentations': config.presentations_per_epoch//2}
    assert final.check_exposure(1, [row])['public_presentations'] == 162353
    for change in ('public', 'anchor', 'noise', 'start', 'end'):
        bad = deepcopy(row)
        if change in ('public', 'anchor'): bad['presentations'][change] -= 1
        elif change == 'noise': bad['noisy_presentations'] -= 1
        elif change == 'start': bad['first_completed_update'] = 1
        else: bad['last_completed_update'] -= 1
        with pytest.raises(ValueError): final.check_exposure(1, [bad])
    with pytest.raises(ValueError): final.check_exposure(2, [row, row])


def test_checkpoint_owner_rejects_foreign_or_impossible_cursors(tmp_path, monkeypatch):
    monkeypatch.setattr(final.study, 'STORAGE', tmp_path/'checkpoints')
    path = final.study.STORAGE/'epoch1_attempt2/step_000256.pt'
    assert final.checkpoint_owner(path) == (final.study.root_for(1, 2), 256)
    for path in (tmp_path/'foreign.pt', final.study.STORAGE/'epoch1_attempt2/step_001693.pt',
                 final.study.STORAGE/'epoch3_attempt1/step_000256.pt',
                 final.study.STORAGE/'epoch1_attempt0/step_000256.pt'):
        with pytest.raises(ValueError): final.checkpoint_owner(path)


def test_receipt_requires_terminal_state_and_exact_artifact_bytes(tmp_path):
    for name in ('producer.json', 'result.json'): (tmp_path/name).write_text('{}')
    receipt = {'actual_tool_exit_code': 0, 'artifacts': {
        str(tmp_path/n): final.data.sha(tmp_path/n) for n in ('producer.json', 'result.json')}}
    (tmp_path/'actual_tool_receipt.json').write_text(json.dumps(receipt))
    final.receipt(tmp_path)
    (tmp_path/'result.json').write_text('{"changed":true}')
    with pytest.raises(ValueError): final.receipt(tmp_path)
    receipt['actual_tool_exit_code'] = 1
    (tmp_path/'actual_tool_receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError): final.receipt(tmp_path)


def test_snapshot_reads_only_complete_ordered_published_rows(tmp_path):
    path = tmp_path/'trace.jsonl'
    first = b'{"step":1}\n{"step":2}\n'
    path.write_bytes(first+b'{"step":')
    assert final.capture_prefix(path, 2) == first
    with pytest.raises(ValueError): final.capture_prefix(path, 3)
    with pytest.raises(ValueError): final.capture_prefix(path, 0)
    path.write_bytes(b'{"step":2}\n')
    with pytest.raises(ValueError): final.capture_prefix(path, 1)


def test_registry_cannot_turn_dev_qualification_into_final_success():
    row = milestone(1, 122, [61, 61])
    row.update({'cases': 128, 'source_only': True, 'final_test_opened': False,
        'final_task_goal_achieved': False, 'speed_tested': False,
        'training_updates': final.study.CONFIG.updates_per_epoch,
        'public_presentations': 162353, 'anchor_presentations': final.study.CONFIG.anchors_per_epoch,
        'canonical_first_judgments': 4691})
    value = {'hypothesis_id': 'ncm-public-curriculum-epoch1-v1',
        'parent_hypothesis_id': 'ncm-public-curriculum-v1', 'epoch': 1, 'attempt': 1,
        'cases_per_milestone': 128, 'independent_training_trajectories': 1,
        'classification': row['classification'], 'milestones': [row], 'selected_candidate_epoch': 1,
        'canonical_first_judgments': 4691, 'final_test_opened': False,
        'final_task_goal_achieved': False, 'speed_tested': False}
    record = registry.record_for(value)
    assert record['DEV_qualified'] and not record['final_task_accuracy_measured']
    value['final_task_goal_achieved'] = True
    with pytest.raises(ValueError): registry.record_for(value)
