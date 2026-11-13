"""Reserved numerical diagnostic; never merges its scores into the pilot."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from structure_net_resources.client import require_gpu_lease
from .jspace_lens import load_small_qwen
from .jspace_pilot import file_hash, write_json


def check(plan_path, output):
    lease = require_gpu_lease([0])
    torch.set_num_threads(4)
    plan = json.loads(Path(plan_path).read_text())
    cases = plan["split_manifest"]["splits"]["development"]
    results = []
    for spec in plan["models"]:
        for dtype_name in ("float16", "bfloat16", "float32"):
            model, tokenizer = load_small_qwen(spec["path"], device="cuda:0", dtype=getattr(torch, dtype_name))
            bad_weights = [name for name, p in model.named_parameters() if not bool(torch.isfinite(p).all())]
            rows = []
            for case in cases:
                first_bad = []
                handles = []
                def make_hook(name):
                    def hook(_module, _inputs, output):
                        h = output[0] if isinstance(output, tuple) else output
                        if not bool(torch.isfinite(h).all()) and not first_bad:
                            first_bad.append(name)
                    return hook
                for index, block in enumerate(model.model.layers):
                    handles.append(block.register_forward_hook(make_hook(f"block_{index}")))
                try:
                    ids = tokenizer(case["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
                    with torch.inference_mode():
                        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
                        logits = model.lm_head(hidden).float()
                        finite = bool(torch.isfinite(hidden).all() and torch.isfinite(logits).all())
                        token_id = int(logits[0].argmax()) if finite else None
                    rows.append({"case_id": case["case_id"], "finite": finite,
                                 "first_bad_block": first_bad[0] if first_bad else None,
                                 "first_token_id": token_id,
                                 "first_token_text": tokenizer.decode([token_id]) if token_id is not None else None})
                finally:
                    for handle in handles:
                        handle.remove()
            result = {"model": spec["path"], "dtype": dtype_name, "bad_weights": bad_weights,
                      "rows": rows, "finite_cases": sum(r["finite"] for r in rows), "total": len(rows)}
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in ("rows", "bad_weights")}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    write_json(output, {"kind": "numerical_diagnostic_not_quality_evidence", "lease_id": lease["id"],
                        "plan_fingerprint": plan["scientific_fingerprint"],
                        "source_sha256": file_hash(__file__), "device": torch.cuda.get_device_name(),
                        "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    check(args.plan, args.output)


if __name__ == "__main__":
    main()
