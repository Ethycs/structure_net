"""Cache frozen Llama context for the new development cohort only."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_frozen_body import MODEL_PATH, module_hash, parent_inputs
from experiments.structure_net.ncm_llama import tokenizers
from experiments.structure_net.ncm_paper_context import right_pack
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import PREP, ROOT, verify
from experiments.structure_net.train_ncm_memory import save


def run(pilot):
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    root = ROOT / ("20261022_features_pilot_v1" if pilot else "20261022_features_v1")
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "frozen_development_body_features",
        [PREP / "development_sources.json", *[p for p in MODEL_PATH.iterdir() if p.is_file()]],
        {"cases": 4 if pilot else 64, "batch": 4, "seed": 42, "padding": "valid states right packed"})
    if not pilot and q.read(ROOT / "20261022_features_pilot_v1/result.json")["status"] != "passed":
        raise ValueError("body-feature memory pilot required")
    # The extra producing sources are immutable too.
    from experiments.structure_net.ncm_memory_data import file_hash, write_json
    for source in (Path(__file__), Path("experiments/structure_net/ncm_paper_context.py")):
        p["source_hashes"][str(source)] = file_hash(source)
        (root / "producer_source" / ("feature_"+source.name)).write_bytes(source.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    _, tokenizer = tokenizers()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    before = module_hash(model.model)
    if before != "75cb5e87d431fa688ec1a68dc868636cf3cfbf2b421601ea56dd21848284d38c":
        raise ValueError("frozen Llama body identity changed")
    cases = q.read(PREP / "development_sources.json")[:4 if pilot else 64]
    (root / "features").mkdir()
    for start in range(0, len(cases), 4):
        chunk = cases[start:start+4]
        inputs = parent_inputs(chunk, tokenizer, "cuda")
        with torch.inference_mode():
            hidden = model.model(**inputs, use_cache=False).last_hidden_state
            hidden, mask = right_pack(hidden, inputs["attention_mask"].eq(0))
        for case, value, pad in zip(chunk, hidden, mask):
            save(root / "features" / (case["id"]+".pt"),
                {"hidden": value.cpu().clone(), "pad_mask": pad.cpu().clone()})
    if module_hash(model.model) != before or any(p.grad is not None for p in model.parameters()):
        raise ValueError("parent body changed")
    verify(p)
    outputs = list((root / "features").glob("*.pt"))
    result = {"status": "passed", "stage": p["stage"], "cases": len(outputs),
        "body_sha256": before, "body_frozen": True, "output_hashes": {str(p): file_hash(p) for p in outputs},
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "producer_fingerprint": p["fingerprint"],
        "task_accuracy": None}
    write_json(root / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "output_hashes"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "extract"])
    run(parser.parse_args().command == "pilot")
