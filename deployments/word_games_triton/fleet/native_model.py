"""Generic OpenAI-compatible native backend for the additional GGUF families."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb

spec = importlib.util.spec_from_file_location('native_base', Path(__file__).with_name('native_base.py'))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class TritonPythonModel(base.TritonPythonModel):
    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                if self.process.poll() is not None:
                    raise RuntimeError('Native worker exited; reload this model')
                prompt = base.decode_scalar(pb.get_input_tensor_by_name(request, 'text_input'), 'text_input')
                optional = pb.get_input_tensor_by_name(request, 'sampling_parameters')
                maximum = base.sampling(base.decode_scalar(optional, 'sampling_parameters') if optional is not None else '{}')
                # Let each GGUF's embedded template handle its own roles/tokens.
                result = self.call('POST', '/v1/chat/completions', {
                    'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': maximum,
                    'temperature': 0, 'seed': 73, 'response_format': {'type': 'json_object'}})
                content = result['choices'][0]['message']['content']
                responses.append(pb.InferenceResponse(output_tensors=[
                    pb.Tensor('text_output', np.array([content], dtype=object)),
                    pb.Tensor('generation_info', np.array([json.dumps({
                        'profile': self.profile, 'usage': result.get('usage'),
                        'model': result.get('model')})], dtype=object))]))
            except Exception as exc:
                responses.append(pb.InferenceResponse(error=pb.TritonError(str(exc))))
        return responses
