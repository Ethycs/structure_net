"""Run vocabulary-only Phi mapping in the pinned Pixi serving environment."""
import argparse
import os
from pathlib import Path
import subprocess

from .common import sha, write
from .run_speculative import IMAGE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", choices=["build_phi_token_map", "check_qwen_tokenizers"], default="build_phi_token_map")
    args = p.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cmd = ["docker", "run", "--rm", "--network", "none", "--user", f"{os.getuid()}:{os.getgid()}",
           "-e", "CUDA_VISIBLE_DEVICES=", "-e", "PYTHONHOME=/opt/pixi-gemma4/.pixi/envs/default",
           "-e", "PYTHONPATH=/workspace:/workspace/src", "-e", "PYTHONPYCACHEPREFIX=/tmp/word-games-map-bytecode",
           "-v", str(Path.cwd()) + ":/workspace:ro", "-v", str(out) + ":/artifacts:rw",
           "-v", "/data/models:/model-weights:ro", "-w", "/workspace",
           "--entrypoint", "/opt/pixi-gemma4/.pixi/envs/default/bin/python", IMAGE,
           "-u", "-m", "experiments.word_games." + args.worker, "--output", "/artifacts/build"]
    write(out / "launch.json", {"argv": cmd, "launcher_sha256": sha(__file__), "gpu_requested": False})
    with (out / "worker.log").open("x") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    write(out / "exit.json", {"exit_code": result.returncode})
    if result.returncode: raise SystemExit(result.returncode)
    print(str(args.output / "build/campaign_results.json"))


if __name__ == "__main__":
    main()
