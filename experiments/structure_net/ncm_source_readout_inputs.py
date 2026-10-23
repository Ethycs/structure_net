"""Verify pinned public BGE inputs before source-grounded readout experiments."""
import argparse
import json
from pathlib import Path

from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT

REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
CACHE = Path("/home/rabbit/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots") / REVISION
DEST = ROOT / "20261023_source_readout_model_v1"
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-source-grounded-readout-feasibility-plan.md")


def model():
    import requests
    if DEST.exists():
        raise ValueError("fresh public-model verification root required")
    DEST.mkdir()
    (DEST / "source.py").write_bytes(Path(__file__).read_bytes())
    (DEST / "plan.md").write_bytes(PLAN.read_bytes())
    url = f"https://huggingface.co/api/models/BAAI/bge-m3/tree/{REVISION}?recursive=true"
    response = requests.get(url, timeout=60); response.raise_for_status()
    metadata = response.json()
    write_json(DEST / "official_tree.json", metadata)
    receipts = {}
    for name in ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model"):
        entry = next(row for row in metadata if row["path"] == name)
        path = CACHE / name
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError("cached public model file missing or wrong size: "+name)
        digest = file_hash(path)
        if "lfs" in entry:
            if digest != entry["lfs"]["oid"]:
                raise ValueError("cached LFS model identity mismatch: "+name)
        else:
            # Small config JSON is compared with the pinned public bytes as well.
            public = requests.get(f"https://huggingface.co/BAAI/bge-m3/resolve/{REVISION}/{name}", timeout=60)
            public.raise_for_status()
            if path.read_bytes() != public.content:
                raise ValueError("cached public configuration differs: "+name)
        receipts[str(path)] = {"sha256": digest, "bytes": path.stat().st_size,
                               "identity": "published_LFS_SHA256" if "lfs" in entry else "pinned_public_bytes"}
    config = json.loads((CACHE / "config.json").read_text())
    expected = {"model_type": "xlm-roberta", "hidden_size": 1024, "num_hidden_layers": 24,
                "num_attention_heads": 16, "vocab_size": 250002, "max_position_embeddings": 8194}
    if any(config[k] != value for k,value in expected.items()):
        raise ValueError("unexpected pinned BGE architecture")
    result = {"status": "verified_pinned_public_model", "revision": REVISION, "metadata_url": url,
        "architecture": expected, "files": receipts, "source_sha256": file_hash(__file__),
        "plan_sha256": file_hash(PLAN), "model_loaded": False, "trained": False,
        "task_accuracy": None, "final_test_opened": False, "speed_tested": False}
    write_json(DEST / "result.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["model"])
    model()
