"""Generate a true parent reference on the current fixed development sources."""
import argparse
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_frozen_body import MODEL_PATH, module_hash, parent_inputs
from experiments.structure_net.ncm_llama import tokenizers
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import PREP, ROOT, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-matched-parent-control-plan.md")
EXPECTED_BODY = "75cb5e87d431fa688ec1a68dc868636cf3cfbf2b421601ea56dd21848284d38c"


class ValidateScores:
    def __call__(self, input_ids, scores):
        if bool((torch.isnan(scores) | torch.isposinf(scores)).any()) or not bool(torch.isfinite(scores).any(-1).all()):
            raise ValueError("invalid parent generation scores")
        return scores


@torch.inference_mode()
def generate(model, tokenizer, cases):
    results = []
    for start in range(0, len(cases), 4):
        chunk = cases[start:start+4]
        inputs = parent_inputs(chunk, tokenizer, "cuda")
        generated = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id,
            logits_processor=[ValidateScores()])
        for case, raw in zip(chunk, generated[:, 512:].tolist()):
            text = tokenizer.decode(raw, skip_special_tokens=True).strip()
            results.append({**case, "text": text, "native_output_ids": raw,
                "parent_prompt_tokens": len(case["parent_ids"]), "source_truncated": False,
                "generator": "original_full_llama32_3b", "max_new_tokens": 96})
        print(json.dumps({"generated": len(results), "total": len(cases)}), flush=True)
    return results


def main(pilot):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("physicalGPU0 must be selected before torch startup")
    if not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name():
        raise ValueError("planned CUDA device required")
    free, _ = torch.cuda.mem_get_info()
    if free < 7_500_000_000:
        raise ValueError("not enough free device memory; wait for the NCM worker to exit")
    root = ROOT / ("20261022_parent_pilot_v1" if pilot else "20261022_parent_outputs_v1")
    cases = q.read(PREP / "development_sources.json")[:4 if pilot else 64]
    verify_public(cases)
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "original_parent_development_reference",
        [PREP / "development_sources.json", *[path for path in MODEL_PATH.iterdir() if path.is_file()]],
        {"batch": 4, "source_prompt_limit": 512, "max_new_tokens": 96, "do_sample": False,
         "precision": "BF16", "attention": "sdpa", "seed": 42, "physical_gpu": 0,
         "output_budget_differs_from_ncm_canvas": True})
    p["hypothesis_id"] = "ncm-matched-parent-control-v1"
    for path in (Path(__file__), PLAN):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("parent_"+path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    if not pilot:
        pr = ROOT / "20261022_parent_pilot_v1"
        if q.read(pr / "result.json")["status"] != "passed" or q.read(pr / "producer.json")["source_hashes"] != p["source_hashes"]:
            raise ValueError("matching parent generation pilot required")
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
    _, tokenizer = tokenizers()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    if module_hash(model.model) != EXPECTED_BODY:
        raise ValueError("parent body identity changed")
    outputs = generate(model, tokenizer, cases)
    repeat_equal = None
    if pilot:
        repeat_equal = outputs == generate(model, tokenizer, cases)
        if not repeat_equal:
            raise ValueError("parent greedy generation changed on repetition")
    if module_hash(model.model) != EXPECTED_BODY or any(x.grad is not None for x in model.parameters()):
        raise ValueError("original parent changed")
    verify(p)
    write_json(root / "outputs.json", outputs)
    result = {"status": "passed" if pilot else "generated_pending_task_grading", "cases": len(outputs),
        "producer_fingerprint": p["fingerprint"], "body_sha256": EXPECTED_BODY,
        "body_frozen": True, "repeated_greedy_equal": repeat_equal, "trained": False,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False, "output_budget_differs_from_ncm_canvas": True}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "generate"])
    main(parser.parse_args().command == "pilot")
