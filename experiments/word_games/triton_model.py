"""Project-owned transport adapter; never changes an existing serving model."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import triton_python_backend_utils as pb


class TritonPythonModel:
    def initialize(self, args):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "GPU-0370fdb0-6490-d871-15e5-c14a1fa671ff",
               "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        self.worker = subprocess.Popen(
            ["/opt/pixi-gemma4/.pixi/envs/default/bin/python", "-u", str(Path(__file__).with_name("phi_worker.py"))],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        if json.loads(self.worker.stdout.readline()) != {"ready": True}:
            raise RuntimeError("Phi-4 worker did not become ready")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                prompt = pb.get_input_tensor_by_name(request, "text_input").as_numpy()[0]
                params = pb.get_input_tensor_by_name(request, "sampling_parameters").as_numpy()[0]
                if isinstance(prompt, bytes): prompt = prompt.decode()
                if isinstance(params, bytes): params = params.decode()
                options = json.loads(params)
                if options.get("temperature") != 0 or not 1 <= options["max_tokens"] <= 1024:
                    raise ValueError("fixed experiment sampling contract required")
                self.worker.stdin.write(json.dumps({"prompt": prompt, "max_tokens": options["max_tokens"]}) + "\n")
                self.worker.stdin.flush()
                value = json.loads(self.worker.stdout.readline())
                if "error" in value: raise RuntimeError(value["error"])
                responses.append(pb.InferenceResponse(output_tensors=[
                    pb.Tensor("text_output", np.array([value["text"]], dtype=object)),
                    pb.Tensor("generation_info", np.array([json.dumps({k: v for k, v in value.items() if k != "text"})], dtype=object))]))
            except Exception as exc:
                responses.append(pb.InferenceResponse(error=pb.TritonError(str(exc))))
        return responses

    def finalize(self):
        if hasattr(self, "worker"):
            self.worker.terminate()
            try: self.worker.wait(timeout=15)
            except subprocess.TimeoutExpired: self.worker.kill(); self.worker.wait()
