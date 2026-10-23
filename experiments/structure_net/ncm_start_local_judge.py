"""Start only this experiment's isolated Qwen judge after its CUDA75 build."""
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import requests

from experiments.structure_net.ncm_memory_data import file_hash, write_json

BASE = Path("data/experiments/ncm_local_judge")
PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-local-judge-gpu2-amendment.md")


def main():
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or os.environ.get("CUDA_VISIBLE_DEVICES") != "2":
        raise ValueError("physical GPU2 must be selected before process startup")
    built = BASE / "20261022_build_gpu2_v1/result.json"
    while not built.exists():
        time.sleep(5)
    build = json.loads(built.read_text())
    downloaded = json.loads((BASE / "20261022_download_v1/result.json").read_text())
    binary, model = Path(build["server_binary"]), Path(downloaded["model"])
    if build["status"] != "passed" or downloaded["status"] != "passed":
        raise ValueError("local preparation failed")
    if file_hash(binary) != build["server_sha256"] or file_hash(model) != downloaded["sha256"]:
        raise ValueError("server/model identity changed")
    for path, expected in build["library_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("judge library changed")
    root = BASE / "20261022_server_gpu2_v1"
    if root.exists():
        raise ValueError("fresh server record required")
    root.mkdir(parents=True)
    (root / "producer_source.py").write_bytes(Path(__file__).read_bytes())
    (root / "placement.md").write_bytes(PLAN.read_bytes())
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 18081))
    command = [str(binary), "--model", str(model), "--alias", "ncm-qwen3-8b-q6k",
        "--host", "127.0.0.1", "--port", "18081", "--gpu-layers", "28",
        "--ctx-size", "4096", "--parallel", "1", "--threads", "4", "--threads-batch", "4",
        "--batch-size", "512", "--ubatch-size", "128", "--flash-attn", "on",
        "--cache-type-k", "f16", "--cache-type-v", "f16", "--no-op-offload",
        "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}']
    snapshot = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader"], text=True)
    log = (root / "server.log").open("w")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    producer = {"schema": "nal.ncm-local-server.v1", "pid": process.pid, "command": command,
        "source_sha256": file_hash(__file__), "plan_sha256": file_hash(PLAN),
        "model_sha256": downloaded["sha256"], "server_sha256": build["server_sha256"],
        "libraries": build["library_hashes"], "physical_gpu": 2, "gpu_layers": 28,
        "cpu_layers": 8, "slots": 1, "context": 4096, "gpu_snapshot_before": snapshot,
        "resident_services_changed": False}
    write_json(root / "producer.json", producer)
    session = requests.Session()
    session.trust_env = False
    for _ in range(180):
        if process.poll() is not None:
            write_json(root / "startup_result.json", {"status": "failed", "exit_code": process.returncode})
            raise RuntimeError("local judge server stopped during startup; inspect server.log")
        try:
            response = session.get("http://127.0.0.1:18081/health", timeout=2)
            if response.status_code == 200:
                write_json(root / "startup_result.json", {"status": "ready", "pid": process.pid,
                    "model": "ncm-qwen3-8b-q6k", "qualification_pending": True})
                print(json.dumps({"status": "ready", "pid": process.pid, "physical_gpu": 2}), flush=True)
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    else:
        process.terminate()
        raise RuntimeError("local judge startup deadline exceeded")
    code = process.wait()
    log.close()
    write_json(root / "terminal_result.json", {"status": "stopped", "exit_code": code,
        "model_weights_changed": False})


if __name__ == "__main__":
    main()
