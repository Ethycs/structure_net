"""Manage the pinned standalone Triton service without experiment dependencies."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "deployments/word_games_triton"
STATE = ROOT / ".pixi/word-games-triton-service"
MODEL = "word-games-qwen38-native"
CONTAINER = "word-games-triton"
PROJECT = "word-games-model"
URL = "http://127.0.0.1:8020"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(path):
    stat = path.stat()
    return {key: getattr(stat, key) for key in
            ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


def run(*args):
    return subprocess.check_output(list(args), cwd=ROOT, text=True).strip()


def compose(*args):
    return run("docker", "compose", "--project-name", PROJECT,
               "--file", str(RELEASE / "compose.yaml"), *args)


def api(action):
    request = urllib.request.Request(
        URL + f"/v2/repository/models/{MODEL}/{action}", data=b"{}",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return {"status": response.status, "body": response.read().decode()}


def ready(path):
    try:
        with urllib.request.urlopen(URL + path, timeout=5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def inspect():
    result = subprocess.run(["docker", "inspect", CONTAINER], cwd=ROOT,
                            capture_output=True, text=True)
    if result.returncode:
        if "no such object" in result.stderr.lower() or "no such container" in result.stderr.lower():
            return None
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)[0]


def owned(container):
    return (container.get("Config", {}).get("Labels") or {}).get(
        "com.docker.compose.project") == PROJECT


def require_owned(container):
    if container and not owned(container):
        raise RuntimeError("Existing container is unmanaged; use up --migrate-existing once")


def wait_native_exit(timeout=60):
    """Triton's unload response can precede native finalization and GPU release."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        processes = run("docker", "top", CONTAINER, "-eo", "pid,args")
        if not any("/llama/llama-server" in line for line in processes.splitlines()[1:]):
            return
        time.sleep(.25)
    raise TimeoutError("Native model process did not exit after unload")


def verify():
    lock = read(RELEASE / "release.lock.json")
    if lock["schema_version"] != 1 or lock["model"] != MODEL:
        raise ValueError("Unsupported release")
    files = {RELEASE / p: h for p, h in lock["repository_files"].items()}
    for component in ("native_runtime", "native_sysroot"):
        base = ROOT / lock[component]["path"]
        files.update({base / p: h for p, h in lock[component]["files"].items()})
        for name, target in lock[component]["symlinks"].items():
            path = base / name
            if not path.is_symlink() or str(path.readlink()) != target or not path.exists():
                raise ValueError("Changed or missing runtime symlink: " + str(path))
    for path, expected in files.items():
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Changed or missing pinned asset: " + str(path))
    weights = Path(lock["weights"]["path"])
    before = identity(weights)
    expected = lock["weights"]["sha256"]
    receipt = STATE / "weights_verified.json"
    cached = read(receipt) if receipt.exists() else {}
    if before["st_size"] != lock["weights"]["bytes"]:
        raise ValueError("Wrong checkpoint size")
    reused = cached == {"identity": before, "sha256": expected}
    if not reused:
        print("Verifying checkpoint SHA-256...", flush=True)
        if sha(weights) != expected or identity(weights) != before:
            raise ValueError("Checkpoint changed")
        STATE.mkdir(parents=True, exist_ok=True)
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps({"identity": before, "sha256": expected}))
        temporary.replace(receipt)
    image = run("docker", "image", "inspect", lock["images"]["triton"], "--format", "{{.Id}}")
    if image != lock["images"]["triton"]:
        raise ValueError("Wrong Triton image")
    config = json.loads(compose("config", "--format", "json"))["services"]["model"]
    if config["image"] != image or config["container_name"] != CONTAINER:
        raise ValueError("Compose image/container differs from release lock")
    mounts = {item["target"]: item for item in config["volumes"]}
    expected_mounts = {"/model.gguf": weights, "/models": RELEASE / "models",
                       "/llama": ROOT / lock["native_runtime"]["path"],
                       "/native-sysroot": ROOT / lock["native_sysroot"]["path"]}
    for target, source in expected_mounts.items():
        if Path(mounts[target]["source"]).resolve() != source.resolve() or not mounts[target].get("read_only"):
            raise ValueError("Unexpected asset mount: " + target)
    gpu_ids = config["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"]
    if gpu_ids != lock["gpu_uuids"] or config["environment"]["CUDA_VISIBLE_DEVICES"] != ",".join(gpu_ids):
        raise ValueError("GPU mapping differs from release lock")
    return {"verified_files": len(files), "weights_sha256": expected,
            "weight_receipt_reused": reused, "profile": lock["profile"]}


def check_legacy(container):
    """Only migrate the exact earlier research deployment, never an unrelated owner."""
    lock = read(RELEASE / "release.lock.json")
    if container["Image"] != lock["images"]["triton"]:
        raise ValueError("Existing container uses a different image")
    mounts = {m["Destination"]: m for m in container["Mounts"]}
    for target, path in (("/llama", ROOT / lock["native_runtime"]["path"]),
                         ("/native-sysroot", ROOT / lock["native_sysroot"]["path"]),
                         ("/model.gguf", Path(lock["weights"]["path"])),
                         ("/models", ROOT / ".pixi/word-games-triton/models")):
        mount = mounts.get(target, {})
        if Path(mount.get("Source", "")).resolve() != path.resolve() or mount.get("RW") is not False:
            raise ValueError("Existing container is not the known research deployment")
    for relative, expected in lock["repository_files"].items():
        path = ROOT / ".pixi/word-games-triton" / relative
        if sha(path) != expected:
            raise ValueError("Legacy model repository differs from release")


def up(migrate=False):
    result = verify()
    (STATE / "runtime").mkdir(parents=True, exist_ok=True)
    (ROOT / ".pixi/word-games-cuda-cache").mkdir(parents=True, exist_ok=True)
    existing = inspect()
    backup = None
    was_running = was_loaded = False
    if existing and not owned(existing):
        if not migrate:
            require_owned(existing)
        check_legacy(existing)
        was_running = existing["State"]["Running"]
        was_loaded = ready(f"/v2/models/{MODEL}/ready") if was_running else False
        backup = CONTAINER + "-previous-" + str(time.time_ns())
        if was_running:
            # Triton drains in-flight inference and calls the native child's finalizer.
            if was_loaded:
                api("unload")
                wait_native_exit()
            run("docker", "stop", "--time", "75", CONTAINER)
        run("docker", "rename", CONTAINER, backup)
    try:
        compose("up", "-d", "--wait", "--wait-timeout", "360")
        if not ready(f"/v2/models/{MODEL}/ready"):
            api("load")
        if not ready(f"/v2/models/{MODEL}/ready"):
            raise RuntimeError("Model did not become ready")
    except BaseException:
        if backup:
            compose("down", "--timeout", "75")
            run("docker", "rename", backup, CONTAINER)
            if was_running:
                run("docker", "start", CONTAINER)
                for _ in range(120):
                    if ready("/v2/health/live"):
                        break
                    time.sleep(.5)
                if was_loaded:
                    api("load")
        raise
    if backup:
        run("docker", "rm", backup)
    return {**result, "model_ready": True, "url": URL, "model": MODEL}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "up", "down", "load", "unload", "status"))
    parser.add_argument("--migrate-existing", action="store_true")
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "lifecycle.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if args.action == "verify":
            result = verify()
        elif args.action == "up":
            result = up(args.migrate_existing)
        else:
            container = inspect()
            if args.action == "status":
                result = {"container_exists": container is not None,
                          "container_running": bool(container and container["State"]["Running"]),
                          "managed": bool(container and owned(container)),
                          "model_ready": ready(f"/v2/models/{MODEL}/ready"), "url": URL}
            else:
                require_owned(container)
                if args.action == "down":
                    compose("down", "--timeout", "75")
                    result = {"container_exists": inspect() is not None, "model_ready": False}
                elif args.action == "unload" and not (container and container["State"]["Running"]):
                    result = {"model_ready": False, "already_stopped": True}
                else:
                    if args.action == "load":
                        verify()
                    if args.action == "load" and ready(f"/v2/models/{MODEL}/ready"):
                        result = {"already_loaded": True}
                    else:
                        result = api(args.action)
                    if args.action == "unload":
                        wait_native_exit()
                    result["model_ready"] = ready(f"/v2/models/{MODEL}/ready")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
