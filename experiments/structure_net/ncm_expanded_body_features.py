"""Cache unchanged parent features while independent public teacher labels arrive."""
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
from experiments.structure_net.ncm_paper_data import PARENT, PREP, ROOT, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public
from experiments.structure_net.train_ncm_memory import save

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-expanded-body-features-plan.md")
PREPARED = ROOT / "20261022_expanded_data_v2"
EXPECTED_BODY = "75cb5e87d431fa688ec1a68dc868636cf3cfbf2b421601ea56dd21848284d38c"


@torch.inference_mode()
def main(pilot):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("physicalGPU0 must be selected before torch startup")
    if not torch.cuda.is_available() or "3060" not in torch.cuda.get_device_name():
        raise ValueError("planned CUDA device required")
    if torch.cuda.mem_get_info()[0] < 7_500_000_000:
        raise ValueError("not enough free device memory; wait for other own GPU0 work to exit")
    if q.read(PREPARED / "result.json")["status"] != "passed":
        raise ValueError("qualified source preparation required")
    source = PREPARED / "train_sources.json"
    all_cases = q.read(source)
    if len(all_cases) != 8192:
        raise ValueError("all8192 frozen training sources required")
    cases = all_cases[:4] if pilot else all_cases
    verify_public(cases)
    anchors = q.read(PREP / "train_sources.json")[:4]
    anchor_paths = [PARENT / "train_features" / (r["id"]+".pt") for r in anchors]
    root = ROOT / ("20261022_expanded_features_pilot_v1" if pilot else "20261022_expanded_features_train_v1")
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "frozen_expanded_training_features",
        [source, PREPARED / "producer.json", PREPARED / "public_proof.json", PREP / "train_sources.json",
         *anchor_paths, *[p for p in MODEL_PATH.iterdir() if p.is_file()]],
        {"batch": 4, "seed": 42, "precision": "BF16", "attention": "sdpa", "cases": len(cases),
         "save_valid_token_states": True, "save_last_state_bank": True, "parent_layers": 28,
         "offline_training_cache_not_an_inference_speed_claim": True})
    for path in (Path(__file__), PLAN):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("features_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-expanded-body-feature-preparation-v1"
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    if not pilot:
        previous = ROOT / "20261022_expanded_features_pilot_v1"
        if (q.read(previous / "result.json")["status"] != "passed"
                or q.read(previous / "producer.json")["source_hashes"] != p["source_hashes"]):
            raise ValueError("matching real lifecycle pilot required")
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
    _, tokenizer = tokenizers()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    if module_hash(model.model) != EXPECTED_BODY:
        raise ValueError("original parent body changed")
    inputs = parent_inputs(anchors, tokenizer, "cuda")
    reference = model.model(**inputs, use_cache=False).last_hidden_state
    for path, actual in zip(anchor_paths, reference):
        cached = torch.load(path, weights_only=True)
        valid = ~cached["pad_mask"]
        if not torch.equal(actual[valid.to(actual.device)].cpu(), cached["hidden"][valid]):
            raise ValueError("original valid-token feature anchor changed")
    del inputs, reference
    (root / "token_states").mkdir()
    last_bank, output_hashes = [], {}
    for start in range(0, len(cases), 4):
        chunk = cases[start:start+4]
        inputs = parent_inputs(chunk, tokenizer, "cuda")
        hidden = model.model(**inputs, use_cache=False).last_hidden_state
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError("nonfinite parent hidden states")
        for case, value, mask in zip(chunk, hidden, inputs["attention_mask"]):
            valid = value[mask.bool()].cpu().clone()
            if valid.dtype != torch.bfloat16 or len(valid) != len(case["parent_ids"]):
                raise ValueError("feature dtype or source/prompt length changed")
            last_bank.append(valid[-1].clone())
            path = root / "token_states" / (case["id"]+".pt")
            save(path, {"hidden": valid, "id": case["id"], "source_sha256": case["source_sha256"],
                        "parent_ids": case["parent_ids"], "padding_removed": True})
            output_hashes[str(path)] = file_hash(path)
        if (start+len(chunk)) % 128 == 0 or start+len(chunk) == len(cases):
            progress = {"completed": start+len(chunk), "total": len(cases)}
            write_json(root / "progress.json", progress); print(json.dumps(progress), flush=True)
    bank = {"hidden_last": torch.stack(last_bank), "ids": [r["id"] for r in cases],
            "source_sha256": [r["source_sha256"] for r in cases],
            "body_sha256": EXPECTED_BODY, "producer_fingerprint": p["fingerprint"]}
    save(root / "last_states.pt", bank)
    reloaded = torch.load(root / "last_states.pt", weights_only=True)
    if not torch.equal(reloaded["hidden_last"], bank["hidden_last"]) or reloaded["ids"] != bank["ids"]:
        raise ValueError("saved last-state feature bank changed on reload")
    output_hashes[str(root / "last_states.pt")] = file_hash(root / "last_states.pt")
    if module_hash(model.model) != EXPECTED_BODY or any(p.grad is not None for p in model.parameters()):
        raise ValueError("original parent body changed")
    verify(p)
    result = {"status": "passed", "cases": len(cases), "body_sha256": EXPECTED_BODY,
        "body_frozen": True, "anchor_valid_states_exact": True, "bank_reload_exact": True,
        "output_hashes": output_hashes, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "producer_fingerprint": p["fingerprint"], "trained": False, "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False}
    write_json(root / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "output_hashes"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "extract"])
    main(parser.parse_args().command == "pilot")
