"""Triton Python backend: one native server per explicit model instance."""
import http.client
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

import numpy as np
import triton_python_backend_utils as pb


def decode_scalar(tensor, name):
    values = tensor.as_numpy().reshape(-1)
    if len(values) != 1:
        raise ValueError(name + ' requires one string')
    value = values[0]
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def sampling(text):
    params = json.loads(text)
    if not isinstance(params, dict) or set(params) - {'max_tokens', 'temperature', 'seed', 'response_format'}:
        raise ValueError('unsupported sampling parameter')
    temperature, seed = params.get('temperature', 0), params.get('seed', 73)
    if type(temperature) not in (int, float) or temperature != 0 or type(seed) is not int or seed != 73:
        raise ValueError('temperature zero and seed 73 are fixed')
    if params.get('response_format', {'type': 'json_object'}) != {'type': 'json_object'}:
        raise ValueError('JSON object output is fixed')
    maximum = params.get('max_tokens', 1024)
    if type(maximum) is not int or not 1 <= maximum <= 1024:
        raise ValueError('max_tokens must be an integer from 1 to 1024')
    return maximum


class TritonPythonModel:
    def initialize(self, args):
        self.process = self.log = self.http = None
        self.prefixes = []
        self.prefix_dir = None
        config = json.loads((Path(__file__).parent / 'runtime.json').read_text())
        self.profile = config['profile']
        if config.get('public_prefixes'):
            self.prefix_dir = Path('/runtime') / ('public-prefixes-' + str(time.time_ns()))
            # These files contain public instructions only. Read access lets
            # the host benchmark audit the container-owned snapshot receipts.
            self.prefix_dir.mkdir(mode=0o755)
            config['native_args'] = [*config['native_args'], '--slot-save-path', str(self.prefix_dir)]
        self.http = http.client.HTTPConnection('127.0.0.1', 18080, timeout=240)
        self.log = open('/runtime/native-' + str(time.time_ns()) + '.log', 'w')
        env = dict(os.environ)
        # Only the native child uses the newer libc. Never alter Triton's loader.
        env.pop('LD_PRELOAD', None)
        env.pop('LD_LIBRARY_PATH', None)
        command = ['/native-sysroot/ld-linux-x86-64.so.2', '--library-path',
                   '/native-sysroot:/llama:/cuda:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu',
                   '/llama/llama-server', *config['native_args']]
        try:
            # The explicit loader becomes /proc/self/exe; GGML also searches cwd
            # for its CUDA and CPU plugins, so keep that directory explicit.
            self.process = subprocess.Popen(command, cwd='/llama', env=env, stdout=self.log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('native process exited; inspect /runtime/native-*.log')
                try:
                    self.call('GET', '/health')
                    break
                except (OSError, http.client.HTTPException, RuntimeError):
                    self.http.close()
                    time.sleep(.25)
            else:
                raise TimeoutError('native model load exceeded 240 seconds')
            self.initialize_prefixes(config.get('public_prefixes', []))
        except BaseException:
            self.finalize()
            raise

    def tokenize(self, text):
        return self.call('POST', '/tokenize', {'content': text, 'add_special': True,
                                             'parse_special': True})['tokens']

    def initialize_prefixes(self, prefixes):
        """Snapshot only operator-configured public text, before any user request."""
        started = time.perf_counter()
        for prefix in prefixes:
            if set(prefix) != {'name', 'text'} or not prefix['text']:
                raise ValueError('named, nonempty public prefix required')
            # Leave the final token out to tolerate BPE merges at the boundary.
            # Every hit also verifies the exact token prefix of the full prompt.
            tokens = self.tokenize('<|im_start|>user\n' + prefix['text'])[:-1]
            if not tokens:
                raise ValueError('empty public prefix tokens')
            filename = hashlib.sha256(json.dumps(tokens).encode()).hexdigest() + '.bin'
            evaluated = self.call('POST', '/completion', {
                'prompt': tokens, 'n_predict': 0, 'cache_prompt': False, 'id_slot': 0})
            # This pinned runtime samples one unused token even for n_predict=0,
            # but does not evaluate it into the sequence. Verify stored length,
            # not the response's sampled-token counter, before accepting state.
            if evaluated.get('tokens_cached') != len(tokens):
                raise ValueError('prefix warmup cached unexpected tokens')
            saved = self.call('POST', '/slots/0?action=save', {'filename': filename})
            if saved.get('n_saved') != len(tokens):
                raise ValueError('public snapshot includes unexpected tokens')
            self.prefixes.append({**prefix, 'tokens': tokens, 'filename': filename,
                                  'save': saved, 'prefill': evaluated})
        if prefixes:
            self.call('POST', '/slots/0?action=erase', {})
            receipt = {'seconds': time.perf_counter() - started, 'prefixes': self.prefixes,
                       'public_only': True, 'rebuild_on_every_model_load': True}
            (self.prefix_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')

    def restore_public_prefix(self, prompt, formatted):
        for prefix in self.prefixes:
            if not prompt.startswith(prefix['text']):
                continue
            tokens = self.tokenize(formatted)
            count = len(prefix['tokens'])
            if tokens[:count] != prefix['tokens'] or len(tokens) <= count:
                return formatted, {'hit': False, 'reason': 'token_prefix_mismatch'}
            started = time.perf_counter()
            restored = self.call('POST', '/slots/0?action=restore', {'filename': prefix['filename']})
            if restored.get('n_restored') != count:
                raise ValueError('public prefix restore length mismatch')
            return tokens, {'hit': True, 'name': prefix['name'], 'tokens': count,
                            'restore_seconds': time.perf_counter() - started, 'receipt': restored}
        return formatted, {'hit': False, 'reason': 'unrecognized_public_prefix'}

    def call(self, method, path, body=None):
        encoded = json.dumps(body).encode() if body is not None else None
        # Native HTTP closes idle keep-alive sockets. A request-level connection
        # avoids stale-socket failures without retrying or duplicating generation.
        try:
            self.http.request(method, path, encoded, {'Content-Type': 'application/json', 'Connection': 'close'})
            response = self.http.getresponse()
            data = response.read()
            if response.status != 200:
                raise RuntimeError('native HTTP ' + str(response.status) + ': ' + data.decode()[:500])
            return json.loads(data)
        finally:
            self.http.close()

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                if self.process.poll() is not None:
                    raise RuntimeError('native model process is no longer running; reload model')
                prompt = decode_scalar(pb.get_input_tensor_by_name(request, 'text_input'), 'text_input')
                optional = pb.get_input_tensor_by_name(request, 'sampling_parameters')
                maximum = sampling(decode_scalar(optional, 'sampling_parameters') if optional is not None else '{}')
                formatted = self.call('POST', '/apply-template', {
                    'messages': [{'role': 'user', 'content': prompt}], 'add_generation_prompt': True,
                    'chat_template_kwargs': {'enable_thinking': False}})['prompt']
                expected = '<|im_start|>user\n' + prompt.strip() + '<|im_end|>\n'
                if expected not in formatted or not formatted.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'):
                    raise ValueError('embedded template changed')
                completion_prompt, prefix_info = self.restore_public_prefix(prompt, formatted)
                result = self.call('POST', '/completion', {
                    'prompt': completion_prompt, 'n_predict': maximum, 'temperature': 0, 'seed': 73,
                    'repeat_penalty': 1.0, 'cache_prompt': prefix_info['hit'], 'id_slot': 0,
                    'json_schema': {'type': 'object'},
                    'stream': False, 'return_tokens': True})
                info = {k: v for k, v in result.items() if k != 'content'}
                info['profile'] = self.profile
                info['public_prefix'] = prefix_info
                responses.append(pb.InferenceResponse(output_tensors=[
                    pb.Tensor('text_output', np.array([result['content']], dtype=object)),
                    pb.Tensor('generation_info', np.array([json.dumps(info)], dtype=object))]))
            except Exception as exc:
                if self.http:
                    self.http.close()
                responses.append(pb.InferenceResponse(error=pb.TritonError(str(exc))))
        return responses

    def finalize(self):
        if self.http:
            self.http.close()
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self.log:
            self.log.close()
        if getattr(self, 'prefix_dir', None):
            shutil.rmtree(self.prefix_dir, ignore_errors=True)
