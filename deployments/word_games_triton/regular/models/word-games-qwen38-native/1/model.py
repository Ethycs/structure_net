"""Triton Python backend: one native server per explicit model instance."""
import http.client
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

import numpy as np
import triton_python_backend_utils as pb


_proxy_spec = importlib.util.spec_from_file_location('word_games_chat_proxy', Path(__file__).with_name('chat_proxy.py'))
_proxy_module = importlib.util.module_from_spec(_proxy_spec)
_proxy_spec.loader.exec_module(_proxy_module)
start_chat_proxy = _proxy_module.start_chat_proxy


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
        self.process = self.log = self.http = self.chat_proxy = self.gpu_lock = None
        config = json.loads((Path(__file__).parent / 'runtime.json').read_text())
        self.profile = config['profile']
        assets = Path(__file__).parent / 'runtime-assets'
        native, sysroot, cuda = (assets / name for name in ('native', 'sysroot', 'cuda'))
        self.http = http.client.HTTPConnection('127.0.0.1', config['native_port'], timeout=240)
        log_dir = Path(tempfile.gettempdir()) / args.get('model_name', 'word-games-qwen38-native')
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / ('native-' + str(time.time_ns()) + '.log')
        self.log = self.log_path.open('w')
        env = dict(os.environ)
        # Only the native child uses the newer libc. Never alter Triton's loader.
        env.pop('LD_PRELOAD', None)
        env.pop('LD_LIBRARY_PATH', None)
        # Scope GPU ordering to this child; other Triton models keep their mapping.
        env['CUDA_VISIBLE_DEVICES'] = ','.join(config['gpu_uuids'])
        cache = log_dir / 'cuda-cache'
        cache.mkdir(exist_ok=True)
        env['CUDA_CACHE_PATH'] = str(cache)
        env['CUDA_CACHE_MAXSIZE'] = '2147483648'
        command = [str(sysroot / 'ld-linux-x86-64.so.2'), '--library-path',
                   ':'.join(map(str, (sysroot, native, cuda))) + ':/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu',
                   str(native / 'llama-server'), *config['native_args']]
        try:
            self.gpu_lock = open('/tmp/word-games-model-gpus.lock', 'a')
            fcntl.flock(self.gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # The explicit loader becomes /proc/self/exe; GGML also searches cwd
            # for its CUDA and CPU plugins, so keep that directory explicit.
            self.process = subprocess.Popen(command, cwd=native, env=env, stdout=self.log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
            startup_timeout = config.get('startup_timeout_seconds', 480)
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('native process exited; inspect ' + str(self.log_path))
                try:
                    self.call('GET', '/health')
                    if config.get('public_chat_port'):
                        self.chat_proxy = start_chat_proxy(config['public_chat_port'], config['native_port'])
                    return
                except (OSError, http.client.HTTPException, RuntimeError):
                    self.http.close()
                    time.sleep(.25)
            raise TimeoutError('native model load exceeded ' + str(startup_timeout) + ' seconds')
        except BaseException:
            self.finalize()
            raise

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
                result = self.call('POST', '/completion', {
                    'prompt': formatted, 'n_predict': maximum, 'temperature': 0, 'seed': 73,
                    'repeat_penalty': 1.0, 'cache_prompt': False, 'json_schema': {'type': 'object'},
                    'stream': False, 'return_tokens': True})
                info = {k: v for k, v in result.items() if k != 'content'}
                info['profile'] = self.profile
                responses.append(pb.InferenceResponse(output_tensors=[
                    pb.Tensor('text_output', np.array([result['content']], dtype=object)),
                    pb.Tensor('generation_info', np.array([json.dumps(info)], dtype=object))]))
            except Exception as exc:
                if self.http:
                    self.http.close()
                responses.append(pb.InferenceResponse(error=pb.TritonError(str(exc))))
        return responses

    def finalize(self):
        if self.chat_proxy:
            self.chat_proxy.shutdown()
            self.chat_proxy.server_close()
            self.chat_proxy = None
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
        if self.gpu_lock:
            self.gpu_lock.close()
            self.gpu_lock = None
