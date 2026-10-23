"""Check released-domain inputs against the untouched author's evaluator."""
import argparse
import gc
import io
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, verify

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-released-xsum-control-plan.md")
REVISION = "6580ba41a4b0a976dde876fbe84d25327617b0e6"
DATA = ROOT / "20261022_released_xsum_data_v1"
NAMES = ("passages.pickle", "reference_summaries.pickle")
AMENDMENT = Path("docs/07 - Status Reports/2026-10-22_ncm-released-xsum-runtime-amendment.md")


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("non-primitive pickle type: "+module+"."+name)


def read_pickle(path):
    return PrimitiveUnpickler(io.BytesIO(Path(path).read_bytes())).load()


def download():
    import requests
    if DATA.exists():
        raise ValueError("fresh public download root required")
    DATA.mkdir()
    (DATA / "plan.md").write_bytes(PLAN.read_bytes())
    (DATA / "source.py").write_bytes(Path(__file__).read_bytes())
    url = "https://huggingface.co/api/datasets/hazyresearch/ncm-datasets/tree/"+REVISION+"/xsum"
    response = requests.get(url, timeout=60); response.raise_for_status()
    metadata = response.json()
    write_json(DATA / "metadata.json", metadata)
    receipts = {}
    for name in NAMES:
        entry = next(x for x in metadata if x["path"] == "xsum/"+name)
        expected = entry["lfs"]["oid"]
        if entry["size"] > 128_000_000:
            raise ValueError("unexpected dataset size")
        url = "https://huggingface.co/datasets/hazyresearch/ncm-datasets/resolve/"+REVISION+"/xsum/"+name
        response = requests.get(url, timeout=120); response.raise_for_status()
        path = DATA / name; path.write_bytes(response.content)
        if file_hash(path) != expected or path.stat().st_size != entry["size"]:
            raise ValueError("public dataset LFS identity failed")
        rows = read_pickle(path)
        receipts[name] = {"url": url, "sha256": expected, "bytes": path.stat().st_size,
                          "rows": len(rows), "first_type": type(rows[0]).__name__}
    if len({r["rows"] for r in receipts.values()}) != 1:
        raise ValueError("released source/reference counts differ")
    write_json(DATA / "result.json", {"status": "downloaded_verified_public_inputs", "revision": REVISION,
        "files": receipts, "source_sha256": file_hash(__file__), "plan_sha256": file_hash(PLAN)})
    print(json.dumps({"status": "downloaded_verified_public_inputs", "files": receipts}), flush=True)


def run(pilot):
    import torch
    from experiments.structure_net.ncm_decode_policy import pick_tokens, render_author
    from experiments.structure_net.ncm_frozen_body import module_hash
    from experiments.structure_net.ncm_llama import DEPENDENCIES, INITIAL, VENDOR, load_ncm, tokenizers
    from experiments.structure_net.ncm_paper_controls import producer
    from experiments.structure_net.ncm_training_recipe import cumulative_latent
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("physicalGPU0 must be selected before torch startup")
    if not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name():
        raise ValueError("planned GPU required")
    receipt = q.read(DATA / "result.json")
    for name in NAMES:
        if file_hash(DATA / name) != receipt["files"][name]["sha256"]:
            raise ValueError("released data changed")
    root = ROOT / ("20261022_released_xsum_pilot_v2" if pilot else "20261022_released_xsum_control_v2")
    n = 4 if pilot else 32
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "released_domain_inference_path_control",
        [DATA / name for name in (*NAMES, "result.json")] + [INITIAL / name for name in ("config.yaml", "tokenizer.pt", "generator.pt")],
        {"cases": n, "batch": 4, "seed": 42, "precision": "FP32", "source_prefix": "author first512 tokens"})
    for path in (Path(__file__), PLAN, AMENDMENT, VENDOR / "scripts/eval_shard.py",
                 Path("experiments/structure_net/ncm_decode_policy.py")):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("xsum_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-released-xsum-control-v1"
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    if not pilot:
        pr = ROOT / "20261022_released_xsum_pilot_v2"
        if q.read(pr / "result.json")["status"] != "passed" or q.read(pr / "producer.json")["source_hashes"] != p["source_hashes"]:
            raise ValueError("matching GPU pilot required")
    command = [sys.executable, str(VENDOR / "scripts/eval_shard.py"), "--config", str(INITIAL / "config.yaml"),
        "--checkpoint-dir", str(INITIAL), "--data-dir", str(DATA), "--n", str(n), "--bs", "4", "--out", str(root / "author.pickle")]
    environment = {**os.environ, "NCM_ROOT": str(VENDOR.resolve()),
                   "PYTHONPATH": str(DEPENDENCIES.resolve())+os.pathsep+os.environ.get("PYTHONPATH", "")}
    write_json(root / "author_command.json", command)
    with (root / "author.log").open("w") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    author = read_pickle(root / "author.pickle")
    items, references = (read_pickle(DATA / name)[:n] for name in NAMES)
    gpt, _ = tokenizers()
    vae, net, _ = load_ncm()
    net.eval().requires_grad_(False)
    identity = (module_hash(vae), module_hash(net))
    state = torch.load(INITIAL / "generator.pt", map_location="cpu", weights_only=True)
    allowance = state["inference"]["token_unigram"].double()
    del state; gc.collect()
    rows = []
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, n, 4):
            chunk = items[start:start+4]
            ids = [gpt.encode(x, add_special_tokens=False) if isinstance(x, str) else list(x) for x in chunk]
            prefix = torch.tensor([(x[:512]+[50257]*512)[:512] for x in ids], device="cuda")
            codes = net.generate(None, prefix, temperature=1., top_k=1, top_p=0., cfg_scale=0., use_kv_cache=False)
            logits = vae.decoder(cumulative_latent(vae, codes))
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("nonfinite released text logits")
            tokens = pick_tokens(logits, frequency=2., deficit=1., adjacency=[5., 4., .5], allowance=allowance)
            for offset, raw in enumerate(tokens.tolist()):
                index = start+offset
                reference = references[index]
                reference = reference if isinstance(reference, str) else render_author(reference, gpt)
                text = render_author(raw, gpt)
                rows.append({"index": index, "source": gpt.decode(ids[offset][:512]), "reference": reference,
                             "text": text, "author_text": author["hyps"][index].strip(), "native_output_ids": raw})
    write_json(root / "outputs.json", rows)
    equal = all(r["text"] == r["author_text"] for r in rows)
    unchanged = identity == (module_hash(vae), module_hash(net))
    verify(p)
    result = {"status": ("passed" if pilot else "completed_inference_path_control") if equal and unchanged else "failed",
        "cases": n, "author_local_exact": equal, "models_unchanged": unchanged,
        "nonempty": sum(bool(r["text"]) for r in rows),
        "exact_references": sum(r["text"] == r["reference"] for r in rows),
        "task_accuracy": None, "rouge_measured": False, "trained": False,
        "final_test_opened": False, "speed_tested": False, "confirmed": False, "underpowered": True,
        "peak_local_allocated_bytes": torch.cuda.max_memory_allocated(), "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["download", "pilot", "control"])
    command = parser.parse_args().command
    download() if command == "download" else run(command == "pilot")
