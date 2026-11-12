import json
import subprocess

import pytest

from experiments.word_games.instruction_feedback import extract
from experiments.word_games import opencode_sandbox as sandbox


def message(value, mid='final'):
    return {'type': 'text', 'part': {'messageID': mid, 'text':
            value if isinstance(value, str) else json.dumps(value)}}


def test_request_is_fixed_text_and_ignores_private_history_and_tool_output():
    events = [message('PRIVATE SOURCE NAME', 'old'),
              {'type': 'tool_use', 'part': {'text': 'PRIVATE TOOL OUTPUT'}},
              message({'kind': 'needs_simplification', 'needs': ['one_step', 'success_check']})]
    result = extract(events)
    assert result == {'kind': 'needs_simplification', 'needs': ['one_step', 'success_check'],
                      'message': 'Give me one smaller action at a time. Give a concrete check that tells me this step is done.'}
    assert 'PRIVATE' not in json.dumps(result)


def test_explanatory_prefix_is_discarded_before_terminal_verdict():
    events = [message('The source has PRIVATE balance data. I need a policy.\n\n'
                      '{"kind":"needs_simplification","needs":["missing_information"]}')]
    result = extract(events, review=True)
    assert result['needs'] == ['missing_information']
    assert 'PRIVATE' not in json.dumps(result)
    with pytest.raises(ValueError):
        extract([message('{"kind":"ready"} followed by extra text')], review=True)


@pytest.mark.parametrize('bad', [
    {'kind': 'needs_simplification', 'needs': ['one_step'], 'question': 'PRIVATE'},
    {'kind': 'needs_simplification', 'needs': ['PRIVATE']},
    {'kind': 'needs_simplification', 'needs': []},
    {'kind': 'needs_simplification', 'needs': ['one_step', 'one_step']},
    {'kind': 'needs_simplification', 'needs': [{}]},
    '{"kind":"needs_simplification","needs":["one_step"],"needs":["success_check"]}',
])
def test_invalid_feedback_is_never_forwarded(bad):
    with pytest.raises(ValueError):
        extract([message(bad)])


def test_review_requires_explicit_verdict_and_normal_execution_can_finish_in_prose():
    assert extract([message('Finished editing.')]) is None
    assert extract([message({'kind': 'ready'})], review=True) == {'kind': 'ready'}
    for events in ([], [message('Looks fine.')], [message({'kind': 'ready', 'source': 'PRIVATE'})]):
        with pytest.raises(ValueError):
            extract(events, review=True)


def test_real_read_only_review_boundary_can_read_but_cannot_edit(tmp_path, monkeypatch):
    workspace = tmp_path/'task'; workspace.mkdir()
    (workspace/'task.txt').write_text('unchanged')
    config = tmp_path/'config.json'; config.write_text('{}')
    monkeypatch.setattr(sandbox, 'RUNTIME', tmp_path/'runtimes')
    command, env = sandbox.launch(workspace, config, 'placeholder', read_only=True)
    code = '''
from pathlib import Path
import errno
assert Path('task.txt').read_text() == 'unchanged'
for name in ('task.txt', 'new.txt'):
    try:
        Path(name).write_text('not allowed')
    except OSError as exc:
        assert exc.errno in (errno.EROFS, errno.EACCES)
    else:
        raise AssertionError('Review modified the task')
'''
    subprocess.run(command+['/usr/bin/python3', '-c', code], env=env,
                   capture_output=True, check=True, timeout=30)
    assert (workspace/'task.txt').read_text() == 'unchanged'
