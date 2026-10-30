"""Provision an isolated model endpoint using the existing locked Pixi image."""
from pathlib import Path
import subprocess
import requests

from .common import PACKAGE, sha, write
from .transport import BASE_URL, MODEL

WEIGHTS = Path("/home/rabbit/models/Phi-4-GGUF/phi-4-Q4_K_M.gguf")
EXPECTED = "009aba717c09d4a35890c7d35eb59d54e1dba884c7c526e7197d9c13ab5911d9"
REPOSITORY = Path("/home/rabbit/models/_triton")
CONFIG = '''name: "word-games-phi4-14b"
backend: "python"
max_batch_size: 0
input [
  { name: "text_input" data_type: TYPE_STRING dims: [1] },
  { name: "sampling_parameters" data_type: TYPE_STRING dims: [1] }
]
output [
  { name: "text_output" data_type: TYPE_STRING dims: [-1] },
  { name: "generation_info" data_type: TYPE_STRING dims: [-1] }
]
instance_group [{ count: 1 kind: KIND_CPU }]
parameters: { key: "EXECUTION_ENV_PATH" value: { string_value: "/opt/pixi-gemma4/.pixi/envs/default" } }
'''


def prepare(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    before = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free", "--format=csv"], text=True)
    write(out / "before.json", {"nvidia_smi": before})
    actual = sha(WEIGHTS)
    if actual != EXPECTED:
        raise ValueError("cached Phi-4 checkpoint hash differs from its download receipt")
    target = REPOSITORY / MODEL
    if target.exists():
        raise ValueError("preserve existing model endpoint; use status/unload instead of overwriting")
    (target / "1").mkdir(parents=True)
    (target / "config.pbtxt").write_text(CONFIG)
    for source, destination in (("triton_model.py", "model.py"), ("phi_worker.py", "phi_worker.py")):
        (target / "1" / destination).write_bytes((PACKAGE / source).read_bytes())
    provenance = {"model": MODEL, "base_model": "microsoft/phi-4", "parameters": "14B",
        "weights": str(WEIGHTS), "weights_sha256": actual, "quantization": "Q4_K_M",
        "kv_precision": "Q8_0", "context_tokens": 8192, "physical_gpu": 0,
        "gpu_uuid": "GPU-0370fdb0-6490-d871-15e5-c14a1fa671ff", "backend": "llama.cpp via project-owned Triton adapter",
        "source_hashes": {str(p): sha(p) for p in (target / "config.pbtxt", target / "1/model.py", target / "1/phi_worker.py")}}
    write(out / "deployment.json", provenance)
    response = requests.post(f"{BASE_URL}/v2/repository/models/{MODEL}/load", json={}, timeout=240)
    write(out / "load_receipt.json", {"http_status": response.status_code, "body": response.text})
    response.raise_for_status()
    ready = requests.get(f"{BASE_URL}/v2/models/{MODEL}/ready", timeout=10)
    ready.raise_for_status()
    after = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free", "--format=csv"], text=True)
    identity = {**provenance, "ready": True, "nvidia_smi_after": after}
    write(out / "identity.json", identity)
    return identity
