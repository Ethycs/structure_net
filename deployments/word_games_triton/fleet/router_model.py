"""CPU-only Triton model owns the HTTP selector; weights stay in separate models."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class TritonPythonModel:
    def initialize(self, _args):
        self.manager = module('router').Manager(json.loads(Path(__file__).with_name('catalog.json').read_text()))
        self.server = module('chat_proxy').start_chat_proxy(18080, manager=self.manager)

    def execute(self, requests):
        results = []
        for request in requests:
            try:
                raw = pb.get_input_tensor_by_name(request, 'control').as_numpy().reshape(-1)[0]
                data = json.loads(raw)
                result = self.manager.control(data['action'], data.get('model'))
                results.append(pb.InferenceResponse(output_tensors=[
                    pb.Tensor('result', np.array([json.dumps(result)], dtype=object))]))
            except Exception as exc:
                results.append(pb.InferenceResponse(error=pb.TritonError(str(exc))))
        return results

    def finalize(self):
        if hasattr(self, 'server'):
            self.server.shutdown()
            self.server.server_close()
            # Drain requests that already hold a lease before stopping selection.
            self.manager.stopping = True
            with self.manager.mutex:
                pass
        # Explicit GPU unload is a control action. Keeping the last model loaded
        # avoids recursive Triton unload calls during server shutdown.
