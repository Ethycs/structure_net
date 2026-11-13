"""Development-only audit of trailing whitespace with a finite FP32 parent."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from structure_net_resources.client import require_gpu_lease
from .jspace_lens import load_small_qwen
from .jspace_pilot import file_hash, write_json
from .jspace_runtime import generate_answer
from .jspace_tasks import normalize_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lease = require_gpu_lease([0])
    torch.set_num_threads(4)
    plan = json.loads(Path(args.plan).read_text())
    cases = plan["split_manifest"]["splits"]["development"]
    results = []
    for spec in plan["models"]:
        model, tokenizer = load_small_qwen(spec["path"], device="cuda:0", dtype=torch.float32)
        for form in ("original", "rstrip"):
            rows = []
            for case in cases:
                prompt = case["prompt"] if form == "original" else case["prompt"].rstrip()
                result = generate_answer(model, tokenizer, prompt, max_new_tokens=8)
                result.pop("observed")
                rows.append({"case_id": case["case_id"], "expected": case["answer"],
                             "correct": normalize_answer(result["answer"] or "") == normalize_answer(case["answer"]),
                             "result": result})
            out = {"model": spec["path"], "form": form, "correct": sum(r["correct"] for r in rows),
                   "total": len(rows), "rows": rows}
            results.append(out)
            print(json.dumps({k:v for k,v in out.items() if k != "rows"}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.output, {"kind": "development_only_parent_diagnostic", "dtype": "float32",
                            "evaluation_unseen": True, "lease_id": lease["id"],
                            "source_sha256": file_hash(__file__), "results": results})


if __name__ == "__main__":
    main()
