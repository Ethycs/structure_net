"""The handoff's real GGUF query-projection gate; never a full-model verdict."""
import argparse
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from structure_net.components.layers import model_major_tensor as tensor
from experiments.structure_net import llama_cotenancy_common as common
from experiments.structure_net.prepare_llama_cotenancy import write_json, sha256


def run(output, pilot):
    common.setup()
    vendor = common.ROOT / "llama.cpp"
    sys.path.insert(0, str(vendor / "gguf-py"))
    from gguf import GGUFReader
    from gguf.quants import dequantize
    ranks = (64,) if pilot else (64, 128, 215, 256, 384)
    producing = common.producer(output, "projection-pilot" if pilot else "projection-screen", [Path(__file__), vendor / "gguf-py/gguf/quants.py"],
                                {"ranks": ranks, "replicas": 2, "probe_tokens": 128, "samples": 1 if pilot else 5, "seed": 7, "svd": "exact"})
    reader = GGUFReader(producing["model_manifest"]["gguf"]["path"])
    original = next(v for v in reader.tensors if v.name == "blk.0.attn_q.weight")
    weight = torch.from_numpy(dequantize(original.data, original.tensor_type).copy()).to(device="cuda", dtype=torch.float32)
    if list(weight.shape) != [3072, 3072] or original.n_bytes != 5308416:
        raise ValueError("handoff query tensor identity does not match")
    torch.manual_seed(7)
    x = torch.randn(2, 128, 3072, device="cuda", dtype=torch.float16)
    svd_timing, decomposition = common.timed(lambda: torch.linalg.svd(weight, full_matrices=False))
    u, singular, vh = decomposition
    dense = x.float() @ weight.T
    rows, saved = [], {}
    with torch.inference_mode():
        for rank in ranks:
            spec = {"kind": "svd2", "shape": [3072, 3072], "cores": [(u[:, :rank]*singular[:rank]).half(), vh[:rank].half()]}
            packed = tensor.pack([spec, spec])
            torch.cuda.reset_peak_memory_stats()
            actual = tensor.apply(x, packed)
            def separate():
                return torch.cat([tensor.apply(x[i:i+1], {**packed, "cores": [c[i:i+1] for c in packed["cores"]]}) for i in range(2)])
            expected = separate()
            reconstruction = tensor.reconstruct({**spec, "cores": [c.float() for c in spec["cores"]]})
            timings = common.paired({"packed": lambda: tensor.apply(x, packed), "separate": separate}, samples=1 if pilot else 5)
            row = {"rank": rank, "status": "completed", "packing": common.parity(actual, expected),
                   "factor_storage_bytes_two_models": sum(v.numel()*v.element_size() for v in packed["cores"]),
                   "one_q4_projection_bytes": int(original.n_bytes), "reconstruction_rmse": float((reconstruction-weight).square().mean().sqrt()),
                   "reconstruction_relative_l2": float((reconstruction-weight).norm()/weight.norm()),
                   "reconstruction_cosine": float(F.cosine_similarity(reconstruction.flatten(), weight.flatten(), dim=0)),
                   "retained_energy": float(singular[:rank].square().sum()/singular.square().sum()),
                   "output_relative_l2": float((actual.float()-dense).norm()/dense.norm()), "timings": timings,
                   "packing_speedup": timings["separate"]["cuda_median_ms"]/timings["packed"]["cuda_median_ms"],
                   "peak_allocated_bytes_including_svd_control": torch.cuda.max_memory_allocated()}
            rows.append(row)
            saved[str(rank)] = {**spec, "cores": [v.cpu() for v in spec["cores"]]}
            write_json(output / "cells" / f"rank{rank}.json", row)
            print(json.dumps(row), flush=True)
    torch.save(saved, output / "query_factors.pt")
    result = {"hypothesis_id": common.HYPOTHESIS_ID, "schema": common.SCHEMA_VERSION, "status": "completed",
              "classification": "systems_lifecycle_only_not_quality_evidence" if pilot else "projection_only_diagnostic",
              "producer_fingerprint": producing["fingerprint"], "rows": rows, "svd_timing": svd_timing,
              "factor_checkpoint_sha256": sha256(output / "query_factors.pt"), "gpu_after": common.gpu_state(),
              "full_model_quality_tested": False}
    write_json(output / "campaign_results.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    run(args.output, args.pilot)
