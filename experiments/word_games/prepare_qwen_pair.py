"""Download the official, revision- and SHA256-pinned Qwen GGUF pair."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import requests

from .common import sha, write
from .qwen_pair import PAIRS, directory, HYPOTHESIS


def download(job):
    role, filename, size, expected = job
    config = PAIRS[role]
    target = directory(role) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != size or sha(target) != expected:
            raise ValueError("cached weights differ: " + str(target))
    else:
        partial = target.with_suffix(".gguf.part")
        url = f"https://huggingface.co/{config['repository']}/resolve/{config['revision']}/{filename}"
        response = requests.get(url, stream=True, timeout=(30, 120))
        response.raise_for_status()
        h, count, report_at = hashlib.sha256(), 0, 512 * 1024**2
        with partial.open("xb") as f:
            for block in response.iter_content(8 * 1024**2):
                f.write(block)
                h.update(block)
                count += len(block)
                if count >= report_at:
                    print(json.dumps({"file": filename, "bytes": count, "total": size}), flush=True)
                    report_at += 512 * 1024**2
        if count != size or h.hexdigest() != expected:
            raise ValueError("download integrity failed: " + str(partial))
        partial.rename(target)
    return {"role": role, "path": str(target), "size": size, "sha256": expected,
            "repository": config["repository"], "revision": config["revision"]}


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    write(out / "producer.json", {"hypothesis_id": HYPOTHESIS, "models": PAIRS,
          "source_hashes": {str(p): sha(p) for p in (Path(__file__), Path(__file__).with_name('qwen_pair.py'))}})
    jobs = []
    for role, config in PAIRS.items():
        response = requests.get(f"https://huggingface.co/api/models/{config['repository']}/revision/{config['revision']}?blobs=true", timeout=60)
        response.raise_for_status()
        info = response.json()
        if info['sha'] != config['revision']:
            raise ValueError("repository revision mismatch")
        for filename, size, expected in config['files']:
            entry = next(s for s in info['siblings'] if s['rfilename'] == filename)
            if entry['size'] != size or entry['lfs']['sha256'] != expected:
                raise ValueError("published checkpoint hash differs")
            jobs.append((role, filename, size, expected))
        write(out / f"{role}_source.json", info)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            files = list(pool.map(download, jobs))
        write(out / "models.json", {"hypothesis_id": HYPOTHESIS, "files": files, "verified": True})
        print(json.dumps({"verified": True, "bytes": sum(f['size'] for f in files)}), flush=True)
    except Exception as exc:
        write(out / "interruption.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    prepare(p.parse_args().output)
