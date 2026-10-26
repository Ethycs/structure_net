from copy import deepcopy
import json

import pytest

from experiments.neural_architecture_lab import ncm_public_interruption_recovery as recovery


def qualified():
    return {'status': 'interrupted_public_prefix256_verified',
        'original_root': str(recovery.ORIGINAL), 'original_tool_exit_code': None,
        'original_tool_session': 68593, 'original_process_pid': 3056752,
        'tool_handle_missing_observed': True, 'matching_processes': [],
        'original_result_present': False, 'checkpoint_sha256': recovery.CHECKPOINT_SHA,
        'training_identity': recovery.IDENTITY, 'recorded_updates': 335,
        'committed_updates': 256, 'uncommitted_tail_updates': 79,
        'task_accuracy': None, 'final_test_opened': False,
        'prefix_replay': {'first_completed_update': 0, 'last_completed_update': 256,
            'complete_supplied_trace': False, 'presentations': {'public': 24576, 'anchor': 8192}}}


def test_unknown_exit_remains_unknown():
    value = qualified(); recovery.require_interruption(value)
    for changed in ({'original_tool_exit_code': 0}, {'original_tool_exit_code': 1},
                    {'matching_processes': [123]}, {'committed_updates': 335},
                    {'original_result_present': True}, {'training_identity': 'changed'},
                    {'checkpoint_sha256': 'changed'}, {'uncommitted_tail_updates': 0}):
        with pytest.raises(ValueError): recovery.require_interruption({**value, **changed})


def test_uncommitted_tail_is_not_counted_in_recovered_prefix():
    value = qualified()
    for field, replacement in [('last_completed_update', 335), ('first_completed_update', 1),
                               ('complete_supplied_trace', True), ('presentations', {'public': 1, 'anchor': 1})]:
        changed = deepcopy(value); changed['prefix_replay'][field] = replacement
        with pytest.raises(ValueError): recovery.require_interruption(changed)


def test_trace_retains_every_complete_ordered_tail_row():
    raw = b''.join((json.dumps({'step': i})+'\n').encode() for i in range(1, 336))
    assert recovery.trace_extent(raw) == {'recorded_updates': 335, 'committed_updates': 256,
                                          'uncommitted_tail_updates': 79}
    for changed in (raw[:-1], raw+b'{"step":336}\n', raw.replace(b'{"step": 1}', b'{"step": 2}', 1)):
        with pytest.raises(ValueError): recovery.trace_extent(changed)


def test_process_guard_only_matches_original_attempt(tmp_path):
    for pid, attempt in [(1, '1'), (2, '2')]:
        directory = tmp_path/str(pid); directory.mkdir()
        args = ['python', '-m', 'experiments.structure_net.ncm_public_training',
                '--end-epoch', '1', '--attempt', attempt]
        (directory/'cmdline').write_bytes(b'\0'.join(v.encode() for v in args)+b'\0')
    assert recovery.matching_processes(tmp_path) == [1]


def test_recovery_does_not_weaken_other_receipts(monkeypatch, tmp_path):
    calls = []
    def original(root, names, *, failed=False):
        calls.append((root, names, failed)); return 'original'
    monkeypatch.setattr(recovery.final, 'receipt', original)
    # configure changes module attributes; restore all of them after this test.
    for field in ('roots', 'source_files', 'bindings'):
        monkeypatch.setattr(recovery.final, field, getattr(recovery.final, field))
    recovery.configure()
    assert recovery.final.receipt(tmp_path, ('x',), failed=True) == 'original'
    assert calls == [(tmp_path, ('x',), True)]
    with pytest.raises(ValueError): recovery.final.receipt(recovery.ORIGINAL, ('wrong',), failed=True)
    assert recovery.final.roots(1)['training_audit'].name.endswith('_v2')
    assert recovery.final.roots(1)['semantic_replay'].name.endswith('_v1')
