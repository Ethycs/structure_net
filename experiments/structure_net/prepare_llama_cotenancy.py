"""Prepare pinned public/cache inputs; no model execution or quality claims."""
import hashlib
import json
from pathlib import Path
import requests

MODEL_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots" / MODEL_REVISION
GGUF_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
GGUF_REVISION = "5ab33fa94d1d04e903623ae72c95d1696f09f9e8"
GGUF_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
OUTPUT = Path("data/models/Llama-3.2-3B-cotenancy")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {"hf_model": "meta-llama/Llama-3.2-3B-Instruct", "revision": MODEL_REVISION,
                "model_path": str(MODEL_PATH), "files": {}}
    files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "model.safetensors.index.json",
             "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for name in files:
        path = MODEL_PATH / name
        digest = sha256(path)
        if name.endswith(".safetensors") and digest != path.resolve().name:
            raise RuntimeError("cached checkpoint differs from its SHA-256-addressed blob")
        manifest["files"][name] = {"sha256": digest, "bytes": path.stat().st_size}
        print(json.dumps({"verified": name, **manifest["files"][name]}), flush=True)
    response = requests.get(f"https://huggingface.co/api/models/{GGUF_REPO}/revision/{GGUF_REVISION}", params={"blobs": "true"}, timeout=30)
    response.raise_for_status()
    info = next(f for f in response.json()["siblings"] if f["rfilename"] == GGUF_FILE)
    expected = info["lfs"]["sha256"]
    target = OUTPUT / GGUF_FILE
    if not target.exists():
        temporary = target.with_suffix(".partial")
        url = f"https://huggingface.co/{GGUF_REPO}/resolve/{GGUF_REVISION}/{GGUF_FILE}?download=true"
        with requests.get(url, stream=True, timeout=120) as stream:
            stream.raise_for_status()
            with temporary.open("wb") as f:
                for chunk in stream.iter_content(8 * 1024 * 1024):
                    f.write(chunk)
        if sha256(temporary) != expected:
            raise RuntimeError("downloaded GGUF hash mismatch")
        temporary.replace(target)
    if sha256(target) != expected:
        raise RuntimeError("GGUF hash mismatch")
    manifest["gguf"] = {"repo": GGUF_REPO, "revision": GGUF_REVISION, "file": GGUF_FILE, "path": str(target),
                        "sha256": expected, "bytes": target.stat().st_size,
                        "deployed_artifact_identity_confirmed": False}
    path = OUTPUT / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("preserve existing manifest; new inputs require a new root")
    write_json(path, manifest)
    print(json.dumps({"ready": str(path), "gguf_sha256": expected}), flush=True)


if __name__ == "__main__":
    main()
