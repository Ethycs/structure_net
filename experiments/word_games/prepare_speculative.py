"""Pin and download a Phi-4-mini GGUF; freeze prior WORD_GAMES request inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import requests

from .common import read, sha, write

REPO = "bartowski/microsoft_Phi-4-mini-instruct-GGUF"
REVISION = "7ff82c2aaa4dde30121698a973765f39be5288c0"
FILENAME = "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
EXPECTED = "01999f17c39cc3074afae5e9c539bc82d45f2dd7faa3917c66cbef76fce8c0c2"
MODEL_DIR = Path("/home/rabbit/models/Phi-4-mini-instruct-GGUF")
PARENT = Path("data/experiments/word_games/20261030_phi4_full_v2")


def prepare(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    metadata = requests.get(f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true", timeout=30)
    metadata.raise_for_status()
    info = metadata.json()
    entry = next(s for s in info["siblings"] if s["rfilename"] == FILENAME)
    if entry["lfs"]["sha256"] != EXPECTED:
        raise ValueError("published checkpoint hash changed")
    write(out / "download_source.json", {"repository": REPO, "revision": REVISION, "file": entry})
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = MODEL_DIR / FILENAME
    if target.exists():
        actual = sha(target)
    else:
        partial = target.with_suffix(".gguf.part")
        h = hashlib.sha256()
        response = requests.get(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}", stream=True, timeout=(30, 120))
        response.raise_for_status()
        size, next_report = 0, 256 * 1024**2
        with partial.open("xb") as f:
            for block in response.iter_content(8 * 1024**2):
                f.write(block); h.update(block); size += len(block)
                if size >= next_report:
                    print(json.dumps({"download_bytes": size, "expected_bytes": entry["size"]}), flush=True)
                    next_report += 256 * 1024**2
        actual = h.hexdigest()
        if actual != EXPECTED or size != entry["size"]:
            raise ValueError("download integrity failed; partial retained")
        partial.rename(target)
    if actual != EXPECTED:
        raise ValueError("cached draft checkpoint differs")
    requests_frozen = []
    for path in sorted(PARENT.rglob("request.json")):
        r = read(path)
        prompt = r["body"]["inputs"][0]["data"][0]
        params = json.loads(r["body"]["inputs"][1]["data"][0])
        requests_frozen.append({"id": str(path.parent.relative_to(PARENT)), "prompt": prompt,
            "max_tokens": params["max_tokens"], "source": str(path), "source_sha256": sha(path)})
    write(out / "requests.json", requests_frozen)
    result = {"draft_model": str(target), "draft_sha256": actual, "draft_size": target.stat().st_size,
        "draft_repository": REPO, "draft_revision": REVISION,
        "target": read("data/experiments/word_games/20261030_phi4_runtime_v1/identity.json"),
        "requests": len(requests_frozen), "requests_sha256": sha(out / "requests.json"),
        "native_target_vocab": 100352, "native_draft_vocab": 200064,
        "tokenizers_identical": False, "cross_tokenizer_candidate_conversion_required": True}
    write(out / "prepared.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    prepare(parser.parse_args().output)
