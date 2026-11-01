"""Persistent, sequential subprocess transports for the bounded research worker.

No model output is released here. The bridge still validates the complete result
and projects authorized facts before responding to its receiving client.
"""
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


class JsonLineProcess:
    """One in-flight request; EOF, timeout or invalid framing poisons the process."""

    def __init__(self, command, stderr_path, *, env=None):
        self.command = command
        self.stderr_path = Path(stderr_path)
        self.env = env
        self.process = None
        self.stderr = None
        self.closed = False
        self.messages = queue.Queue(maxsize=1024)
        self.lock = threading.Lock()

    def _start(self):
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr = self.stderr_path.open('xb')
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr, env=self.env)
        def consume():
            try:
                while True:
                    line = self.process.stdout.readline(16 * 1024 * 1024 + 1)
                    if not line:
                        self.messages.put(EOFError('child closed its output'))
                        return
                    if len(line) > 16 * 1024 * 1024 or not line.endswith(b'\n'):
                        raise ValueError('invalid or oversized JSON line')
                    self.messages.put(line)
            except Exception as exc:
                self.messages.put(exc)
        threading.Thread(target=consume, daemon=True).start()

    def exchange(self, payload, done, *, timeout, log_path=None):
        with self.lock:
            if self.closed:
                raise RuntimeError('child transport is closed; no automatic retry')
            log = None
            started = time.perf_counter()
            try:
                if self.process is None:
                    self._start()
                if log_path is not None:
                    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                    log = Path(log_path).open('xb')
                self.process.stdin.write((json.dumps(payload, allow_nan=False)+'\n').encode())
                self.process.stdin.flush()
                events = []
                while True:
                    remaining = timeout - (time.perf_counter() - started)
                    if remaining <= 0:
                        raise TimeoutError('child result timeout')
                    try:
                        line = self.messages.get(timeout=remaining)
                    except queue.Empty as exc:
                        raise TimeoutError('child result timeout') from exc
                    if isinstance(line, Exception):
                        raise line
                    if log is not None:
                        log.write(line)
                    event = json.loads(line)
                    if not isinstance(event, dict):
                        raise ValueError('object message required')
                    events.append(event)
                    if done(event):
                        return events
            except Exception:
                self.close()
                raise
            finally:
                if log is not None:
                    log.close()

    def close(self):
        self.closed = True
        if self.process is not None:
            if self.process.stdin:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self.process.stdout:
                self.process.stdout.close()
        if self.stderr is not None:
            self.stderr.close()


def worker_command(model, system, schema, session_id, resume, persistent=False):
    command = ['claude', '--safe-mode', '--restricted', '--setting-sources', '',
        '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}', '--disable-slash-commands',
        '--no-chrome', '--permission-mode', 'dontAsk', '--tools', 'Read,Grep,Glob',
        '--allowedTools', 'Read,Grep,Glob', '--model', model, '--system-prompt', system,
        '--json-schema', json.dumps(schema), '-p', '--output-format', 'stream-json', '--verbose',
        '--resume' if resume else '--session-id', session_id]
    if persistent:
        command += ['--input-format', 'stream-json']
    return command


def worker_env():
    env = dict(os.environ)
    env.pop('CLAUDECODE', None)
    env['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC'] = '1'
    return env


class ClaudeWorker:
    def __init__(self, model, system, schema, session_id, *, persistent=False,
                 resume=False, stderr_path=None):
        self.model, self.system, self.schema = model, system, schema
        self.session_id, self.resume = session_id, resume
        self.persistent = persistent
        self.stream = None
        self.stderr_path = stderr_path
        self.closed = False

    def complete(self, prompt, turn, timeout=300):
        if self.closed:
            raise RuntimeError('worker closed; no automatic retry')
        turn.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            if self.persistent:
                if self.stream is None:
                    self.stream = JsonLineProcess(worker_command(self.model, self.system,
                        self.schema, self.session_id, self.resume, True),
                        self.stderr_path or turn/'stderr.txt', env=worker_env())
                events = self.stream.exchange({'type': 'user', 'session_id': self.session_id,
                    'parent_tool_use_id': None, 'message': {'role': 'user', 'content': prompt}},
                    lambda e: e.get('type') == 'result', timeout=timeout, log_path=turn/'worker.jsonl')
            else:
                command = worker_command(self.model, self.system, self.schema, self.session_id, self.resume)
                with (turn/'worker.jsonl').open('x') as stdout, (turn/'stderr.txt').open('x') as stderr:
                    process = subprocess.run(command, input=prompt, text=True, stdout=stdout,
                        stderr=stderr, env=worker_env(), timeout=timeout, check=False)
                if process.returncode:
                    raise RuntimeError('Claude worker exited unsuccessfully')
                events = [json.loads(line) for line in (turn/'worker.jsonl').read_text().splitlines() if line.strip()]
            final = [e for e in events if e.get('type') == 'result']
            if len(final) != 1 or final[0].get('is_error'):
                raise ValueError('successful Claude result required')
            if final[0].get('session_id') != self.session_id:
                raise ValueError('worker session identity changed')
            if not isinstance(final[0].get('structured_output'), dict):
                raise ValueError('structured worker output required')
            self.resume = True
            return final[0], events, time.perf_counter() - started
        except Exception:
            self.close()
            raise

    def close(self):
        self.closed = True
        if self.stream is not None:
            self.stream.close()


TOKENIZER_CODE = '''import json,sys,urllib.request
for line in sys.stdin:
 request=json.loads(line)
 out=[]
 for text in request['texts']:
  req=urllib.request.Request('http://127.0.0.1:18080/tokenize',data=json.dumps({'content':text,'add_special':False,'parse_special':False}).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=30) as r: out.append(len(json.load(r)['tokens']))
 print(json.dumps({'id':request['id'],'counts':out}),flush=True)
'''


class ReferenceTokenizer:
    def __init__(self, stderr_path):
        self.sequence = 0
        self.lock = threading.Lock()
        self.stream = JsonLineProcess(['docker', 'exec', '-i', 'word-games-triton',
            '/opt/pixi-gemma4/.pixi/envs/default/bin/python', '-u', '-c', TOKENIZER_CODE], stderr_path)

    def counts(self, texts):
        with self.lock:
            if not isinstance(texts, list) or not all(isinstance(s, str) for s in texts):
                raise ValueError('text list required')
            self.sequence += 1
            try:
                events = self.stream.exchange({'id': self.sequence, 'texts': texts},
                    lambda e: True, timeout=60)
                result = events[0]
                if result.get('id') != self.sequence or set(result) != {'id', 'counts'}:
                    raise ValueError('tokenizer response identity changed')
                counts = result['counts']
                if not isinstance(counts, list) or len(counts) != len(texts) or any(type(n) is not int or n < 0 for n in counts):
                    raise ValueError('invalid token counts')
                return counts
            except Exception:
                self.close()
                raise

    def close(self):
        self.stream.close()
