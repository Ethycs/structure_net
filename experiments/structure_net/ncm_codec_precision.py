"""Isolate cumulative-latent and decoder precision on a fixed failed cohort."""
import gc
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_llama import INITIAL, decode_ids, encode_codes, load_ncm, tokenizers
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import ROOT, TEACHER, verify
from experiments.structure_net.ncm_training_recipe import cumulative_latent
from experiments.structure_net.ncm_vocabulary import CachedVocabulary

NOTE = Path("docs/07 - Status Reports/2026-10-22_ncm-codec-precision-diagnostic.md")


def main():
    if not torch.cuda.is_available():
        raise ValueError("CUDA required")
    root = ROOT / "20261022_codec_precision_v1"
    prior = ROOT / "20261022_codec_control_v1/result.json"
    cases = q.read(TEACHER / "train.json")
    p = producer(root, "exploratory_codec_precision_diagnostic", [TEACHER / "train.json", prior,
        INITIAL / "config.yaml", INITIAL / "tokenizer.pt", INITIAL / "generator.pt"],
        {"batch": 32, "cells": ["fp32_fp32", "fp32_bf16", "bf16_fp32", "bf16_bf16"],
         "encoding_precision": "FP32", "cases": len(cases)})
    for source in (Path(__file__), NOTE):
        p["source_hashes"][str(source)] = file_hash(source)
        (root / "producer_source" / ("precision_"+source.name)).write_bytes(source.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    gpt, _ = tokenizers()
    gpt = CachedVocabulary(gpt)
    vae, net, _ = load_ncm()
    del net
    gc.collect()
    torch.cuda.empty_cache()
    metrics = {name: {"nonpad_correct": 0, "nonpad_total": 0, "content_correct": 0,
                     "content_total": 0, "exact_text_cases": 0, "max_latent_difference": 0.,
                     "max_logit_difference": 0.} for name in p["config"]["cells"]}
    outputs = {name: [] for name in metrics}
    for i in range(0, len(cases), 32):
        chunk = cases[i:i+32]
        targets = torch.tensor([r["native_target_ids"] for r in chunk], device="cuda")
        codes = encode_codes(vae, targets)
        nonpad, content = targets.ne(50257), targets.ne(50257) & targets.ne(50256)
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=False):
                reference_latent = cumulative_latent(vae, codes)
                reference_logits = vae.decoder(reference_latent)
            for name, row in metrics.items():
                latent_precision, decoder_precision = name.split("_")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=latent_precision == "bf16"):
                    latent = cumulative_latent(vae, codes)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=decoder_precision == "bf16"):
                    logits = vae.decoder(latent.float())
                ids = logits.argmax(-1)
                row["nonpad_correct"] += int((ids.eq(targets) & nonpad).sum())
                row["nonpad_total"] += int(nonpad.sum())
                row["content_correct"] += int((ids.eq(targets) & content).sum())
                row["content_total"] += int(content.sum())
                row["max_latent_difference"] = max(row["max_latent_difference"], float((latent-reference_latent).abs().max()))
                row["max_logit_difference"] = max(row["max_logit_difference"], float((logits.float()-reference_logits).abs().max()))
                for case, generated in zip(chunk, ids.tolist()):
                    text, termination = decode_ids(generated, gpt)
                    row["exact_text_cases"] += text == case["target"]
                    outputs[name].append({**case, "text": text, "native_output_ids": generated, "termination": termination})
    for name, row in metrics.items():
        row["nonpad_reconstruction"] = row["nonpad_correct"]/row["nonpad_total"]
        row["content_reconstruction"] = row["content_correct"]/row["content_total"]
        write_json(root / (name+"_outputs.json"), outputs[name])
    old = q.read(prior)
    reproduction = all(metrics["bf16_bf16"][key] == old[key] for key in
                       ("nonpad_correct", "nonpad_total", "exact_text_cases"))
    verify(p)
    result = {"status": "completed" if reproduction else "original_precision_cell_not_reproduced",
        "prior_reproduced": reproduction, "cells": metrics, "cases": len(cases),
        "producer_fingerprint": p["fingerprint"], "trained": False, "task_accuracy": None}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)
    if not reproduction:
        raise ValueError("original precision cell was not reproduced")


if __name__ == "__main__":
    main()
