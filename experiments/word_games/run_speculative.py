"""Run the benchmark in a disposable container with explicit GPU confinement."""
import argparse
from pathlib import Path
import subprocess

from .common import read, sha, write

IMAGE = "sha256:ea2c80aa8efcf2b1578db2ab869ea1978a243b8327d5e4df9e95cabaaf74cc4f"
GPUS = "GPU-0370fdb0-6490-d871-15e5-c14a1fa671ff,GPU-7b1db3cb-ec9c-beb3-e7db-826c2543edf9"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["pilot", "full"], required=True)
    p.add_argument("--length", type=int, default=4, choices=[2, 4, 8])
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prep = read(args.inputs / "prepared.json")
    print("Verifying target and draft checkpoint hashes", flush=True)
    for path, expected in [(prep["target"]["weights"], prep["target"]["weights_sha256"]), (prep["draft_model"], prep["draft_sha256"])]:
        if sha(path) != expected:
            raise ValueError("checkpoint changed: " + path)
    cuda_cache = Path(".pixi/word-games-cuda-cache").resolve()
    cuda_cache.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm", "--name", "word-games-spec-" + args.stage,
        "--network", "none", "--gpus", '"device=' + GPUS + '"',
        "--user", str(__import__('os').getuid()) + ":" + str(__import__('os').getgid()),
        "-e", "CUDA_VISIBLE_DEVICES=" + GPUS, "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "-e", "PYTHONHOME=/opt/pixi-gemma4/.pixi/envs/default", "-e", "PYTHONPATH=/workspace:/workspace/src",
        "-e", "PYTHONPYCACHEPREFIX=/tmp/word-games-spec-bytecode",
        "-e", "CUDA_CACHE_PATH=/cuda-cache", "-v", str(cuda_cache) + ":/cuda-cache:rw",
        "-v", str(Path.cwd()) + ":/workspace:ro", "-v", str(out) + ":/artifacts:rw",
        "-v", str(args.inputs.resolve()) + ":/inputs:ro", "-v", "/data/models:/model-weights:ro",
        "-w", "/workspace", "--entrypoint", "/opt/pixi-gemma4/.pixi/envs/default/bin/python", IMAGE,
        "-u", "-m", "experiments.word_games.speculative_worker", "--requests", "/inputs/requests.json",
        "--output", "/artifacts/run", "--stage", args.stage, "--length", str(args.length)]
    write(out / "launch.json", {"argv": cmd, "launcher_sha256": sha(__file__),
        "inputs_sha256": sha(args.inputs / "prepared.json"),
        "nvidia_smi_before": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free", "--format=csv"], text=True)})
    with (out / "worker.log").open("x") as log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line); log.flush(); print(line, end="", flush=True)
        code = process.wait()
    write(out / "exit.json", {"exit_code": code,
        "nvidia_smi_after": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free", "--format=csv"], text=True)})
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
