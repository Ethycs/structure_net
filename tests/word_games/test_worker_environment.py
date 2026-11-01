import json
import sys

import pytest

from experiments.word_games.worker_environment import JsonLineProcess, ClaudeWorker, worker_command


def child(tmp_path, code):
    return JsonLineProcess([sys.executable, '-u', '-c', code], tmp_path/'stderr.txt')


def test_sequential_requests_keep_one_child_and_complete_logs(tmp_path):
    process = child(tmp_path, 'import sys,json,os\nfor s in sys.stdin:\n print(json.dumps({"result":json.loads(s),"pid":os.getpid()}),flush=True)')
    try:
        first = process.exchange({'q': 1}, lambda e: 'result' in e, timeout=2, log_path=tmp_path/'one.jsonl')
        second = process.exchange({'q': 2}, lambda e: 'result' in e, timeout=2, log_path=tmp_path/'two.jsonl')
        assert first[0]['pid'] == second[0]['pid']
        assert second[0]['result'] == {'q': 2}
        assert json.loads((tmp_path/'two.jsonl').read_text()) == second[0]
    finally:
        process.close()
    assert process.process.poll() == 0
    with pytest.raises(RuntimeError, match='closed'):
        process.exchange({}, lambda e: True, timeout=1)


@pytest.mark.parametrize('code,error', [
    ('import sys\nsys.stdin.readline()', EOFError),
    ('import sys\nsys.stdin.readline()\nprint("broken",flush=True)', ValueError),
    ('import sys,time\nsys.stdin.readline()\ntime.sleep(.3)', TimeoutError),
])
def test_bad_child_poisoned_without_automatic_retry(tmp_path, code, error):
    process = child(tmp_path, code)
    with pytest.raises(error):
        process.exchange({}, lambda e: True, timeout=.15)
    assert process.closed and process.process.poll() is not None


def test_waits_for_result_and_preserves_intermediate_tool_events(tmp_path):
    process = child(tmp_path, 'import sys,json\nsys.stdin.readline()\nprint(json.dumps({"type":"assistant"}),flush=True)\nprint(json.dumps({"type":"result","value":42}),flush=True)')
    try:
        events = process.exchange({}, lambda e: e.get('type') == 'result', timeout=2)
        assert [e['type'] for e in events] == ['assistant', 'result']
    finally:
        process.close()


def test_worker_flags_preserve_boundary_and_session():
    command = worker_command('sonnet', 'private system', {}, 'SESSION', True, True)
    assert command[command.index('--resume')+1] == 'SESSION'
    assert command[command.index('--tools')+1] == 'Read,Grep,Glob'
    assert command[command.index('--input-format')+1] == 'stream-json'
    assert '--restricted' in command and '--safe-mode' in command


def test_wrong_native_session_poisoned_before_release(tmp_path):
    class Stream:
        closed = False
        def exchange(self, *args, **kwargs):
            return [{'type': 'result', 'session_id': 'OTHER', 'structured_output': {}}]
        def close(self):
            self.closed = True
    worker = ClaudeWorker('sonnet', 'system', {}, 'EXPECTED', persistent=True)
    worker.stream = Stream()
    with pytest.raises(ValueError, match='identity'):
        worker.complete('question', tmp_path/'turn')
    assert worker.closed and worker.stream.closed


def test_bridge_preserves_projection_and_checkpoints_with_live_diagnostics(tmp_path, monkeypatch):
    from experiments.word_games import claude_research_bridge as module
    from experiments.word_games.common import read, write, sha
    from experiments.word_games.research_profile import KEYS, translate
    source = tmp_path/'source.txt'
    source.write_text('The measured value is 42.')
    catalog = {'F01': {'id': 'F01', 'topic': 'test', 'question': 'What value?',
        'path': str(source), 'source_sha256': sha(source), 'anchor': source.read_text(),
        'subject': 'measurement', 'property': 'value', 'value': 42}}
    parent = tmp_path/'parent'
    write(parent/'result.json', {'completed': True, 'turns': 20, 'session_id': 'SESSION', 'released_ids': []})
    write(parent/'private/catalog.json', catalog)
    write(parent/'private/control.json', {'canary': 'PRIVATE_CONTROL'})
    mapping = {'name': 'archive', 'key_map': {k: 'archive_'+k for k in KEYS}}
    write(parent/'public/mapping.json', mapping)
    monkeypatch.setattr(module, 'build_catalog', lambda: catalog)
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **k: 'test-cli')
    class Worker:
        stream = None
        def __init__(self, *a, **k):
            self.diagnostic = k['stderr_path']
        def complete(self, prompt, turn):
            with self.diagnostic.open('a') as f:
                f.write('new private diagnostic\n')
            result = {'type': 'result', 'session_id': 'SESSION', 'usage': {'output_tokens': 10},
                'modelUsage': {}, 'structured_output': {'citations': [{'fact_id': 'F01',
                'source_path': str(source), 'quote': source.read_text()}], 'unknown': [], 'notes': 'PRIVATE_CONTROL'}}
            (turn/'worker.jsonl').write_text(json.dumps(result)+'\n')
            return result, [result], .01
        def close(self):pass
    class Tokenizer:
        def __init__(self, *a):pass
        def counts(self, texts):return [20 for _ in texts]
        def close(self):pass
    monkeypatch.setattr(module, 'ClaudeWorker', Worker)
    monkeypatch.setattr(module, 'ReferenceTokenizer', Tokenizer)
    bridge = module.ResearchBridge(tmp_path/'server', resume_from=parent)
    first = bridge.query('Q21', 'test', 'What was measured?')
    assert translate(first, mapping, reverse=True)['facts'][0]['value'] == 42
    assert 'PRIVATE_CONTROL' not in json.dumps(first)
    bridge.query('Q22', 'test', 'Recall the same measurement.')
    checkpoint = read(tmp_path/'server/checkpoints/Q21.json')
    assert 'private/worker_stderr.txt' not in checkpoint['artifacts']
    for name, expected in checkpoint['artifacts'].items():
        assert sha(tmp_path/'server'/name) == expected
    with pytest.raises(ValueError, match='order'):
        bridge.query('Q22', 'test', 'Duplicate request')
    bridge.close()
