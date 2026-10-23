"""Pin local Qwen judge weights and build an isolated server without service changes."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from experiments.structure_net.ncm_memory_data import file_hash, write_json

ROOT = Path("data/experiments/ncm_local_judge")
SOURCE = Path("data/experiments/llama_tensorized_cotenancy/llama.cpp")
REPO = "Qwen/Qwen3-8B-GGUF"
REVISION = "7c41481f57cb95916b40956ab2f0b139b296d974"
FILENAME = "Qwen3-8B-Q6_K.gguf"
SHA = "cb042ccd76795a8830d6be6bd4165245847cc68e41797b13bd61aed4c2cfbce6"
PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-local-judge-recovery-plan.md")


def main(command):
    root = ROOT / ("20261022_"+command+"_v2")
    if root.exists():
        raise ValueError("fresh append-only preparation root required")
    root.mkdir(parents=True)
    (root / "producer_source.py").write_bytes(Path(__file__).read_bytes())
    (root / "plan.md").write_bytes(PLAN.read_bytes())
    producer = {"schema": "nal.ncm-local-judge-preparation.v1", "stage": command,
        "source_sha256": file_hash(__file__), "plan_sha256": file_hash(PLAN),
        "model_repo": REPO, "revision": REVISION, "file": FILENAME, "expected_sha256": SHA,
        "gpu_inference_started": False}
    write_json(root / "producer.json", producer)
    start = time.perf_counter()
    if command == "download":
        import requests
        response = requests.get(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}",
                                params={"blobs": "true"}, timeout=30)
        response.raise_for_status()
        metadata = response.json()
        entry = next(row for row in metadata["siblings"] if row["rfilename"] == FILENAME)
        if metadata["sha"] != REVISION or entry["lfs"]["sha256"] != SHA:
            raise ValueError("published model identity changed")
        write_json(root / "published_metadata.json", {"sha": metadata["sha"], "file": entry})
        temporary = root / (FILENAME+".partial")
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}?download=true"
        with requests.get(url, stream=True, timeout=120) as download:
            download.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in download.iter_content(8*1024*1024):
                    output.write(chunk)
        if temporary.stat().st_size != entry["size"] or file_hash(temporary) != SHA:
            raise ValueError("downloaded judge model failed size/SHA verification")
        target = root / FILENAME
        temporary.rename(target)
        result = {"status": "passed", "model": str(target), "sha256": SHA, "bytes": target.stat().st_size}
    else:
        original = {str(path): file_hash(path) for path in (SOURCE / "build/bin").iterdir() if path.is_file()}
        source_files = [path for path in SOURCE.rglob("*") if path.is_file() and
            "build" not in path.relative_to(SOURCE).parts and ".git" not in path.relative_to(SOURCE).parts and
            (path.suffix in (".cpp", ".c", ".h", ".hpp", ".cu", ".cuh", ".cmake", ".in") or path.name == "CMakeLists.txt")]
        source_hashes = {str(path): file_hash(path) for path in source_files}
        write_json(root / "original_source_hashes.json", source_hashes)
        write_json(root / "original_binary_hashes.json", original)
        copies = root / "vendor_source"
        for path in source_files:
            target = copies / path.relative_to(SOURCE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        build = root / "build"
        shutil.copytree(SOURCE / "build", build, symlinks=True)
        (build / "CMakeCache.txt").unlink()
        cmake = Path("data/experiments/llama_tensorized_cotenancy/tools_env/lib/python3.12/site-packages/cmake/data/bin/cmake").resolve()
        ninja = Path("data/experiments/llama_tensorized_cotenancy/tools_env/bin/ninja").resolve()
        if not cmake.is_file() or not ninja.is_file():
            raise ValueError("pinned build tools missing")
        configure = [str(cmake), "-S", str(SOURCE.resolve()), "-B", str(build.resolve()),
            "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=86",
            "-DCMAKE_MAKE_PROGRAM="+str(ninja), "-DLLAMA_BUILD_SERVER=ON", "-DLLAMA_BUILD_TESTS=OFF"]
        compile_command = [str(cmake), "--build", str(build), "--target", "llama-server", "--parallel", "4"]
        write_json(root / "commands.json", [configure, compile_command])
        with (root / "build.log").open("w") as log:
            subprocess.run(configure, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(compile_command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if any(file_hash(path) != digest for path, digest in {**original, **source_hashes}.items()):
            raise ValueError("original vendor source/binaries changed during isolated build")
        binary = build / "bin/llama-server"
        result = {"status": "passed", "server_binary": str(binary), "server_sha256": file_hash(binary),
            "library_hashes": {str(path): file_hash(path) for path in (build / "bin").iterdir() if path.is_file()},
            "original_files_unchanged": True}
    if file_hash(__file__) != producer["source_sha256"] or file_hash(PLAN) != producer["plan_sha256"]:
        raise ValueError("preparation producer changed")
    result.update(elapsed_seconds=time.perf_counter()-start, gpu_inference_started=False)
    write_json(root / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "library_hashes"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["download", "build"])
    main(parser.parse_args().command)
