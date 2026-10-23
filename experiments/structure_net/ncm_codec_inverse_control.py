"""Test token-dictionary inversion of the frozen lookup-plus-GroupNorm codec."""
import argparse
import gc
import json
import os
from pathlib import Path

import torch

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_broad_decoder_transfer import DATA as BROAD_DATA
from experiments.structure_net.ncm_frozen_body import module_hash
from experiments.structure_net.ncm_llama import GPT2, INITIAL, decode_ids, encode_codes, tokenizers, vendor_classes
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_controls import producer
from experiments.structure_net.ncm_paper_data import ROOT, verify
from experiments.structure_net.ncm_training_recipe import cumulative_latent
from experiments.structure_net.ncm_unseen_codec_control import CODEC, MEASURE, NOVEL, SEEN, cohorts, validate_targets
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.train_ncm_memory import save

PLAN = Path("docs/07 - Status Reports/2026-10-22_ncm-codec-inverse-control-plan.md")
PILOT = ROOT / "20261022_codec_inverse_pilot_v1"
MEASUREMENT = ROOT / "20261022_codec_inverse_measure_v1"
MODES = ("neural", "blind_inverse_4", "oracle_statistics", "unquantized_oracle")


def group_statistics(values, groups, epsilon):
    b, t, width = values.shape
    grouped = values.float().reshape(b, t, groups, width//groups)
    return grouped.mean((1, 3), keepdim=True), (grouped.var((1, 3), unbiased=False, keepdim=True)+epsilon).sqrt()


def dictionary_inverse(normalized, mean, scale, positions, vocabulary):
    """Undo known group normalization, then nearest-token lookup; no target IDs."""
    b, t, width = normalized.shape
    groups = mean.shape[2]
    restored = (normalized.float().reshape(b, t, groups, width//groups)*scale+mean).reshape(b, t, width)
    vectors = restored-positions[None, :t].float()
    table = vocabulary.float()
    # The input-vector norm is constant over dictionary candidates.
    distances = table.square().sum(-1)[None, None, :]-2*vectors@table.T
    if not bool(torch.isfinite(distances).all()):
        raise ValueError("nonfinite dictionary distances")
    return distances.argmin(-1)


def blind_inverse(vae, normalized, initial):
    """Infer normalization from predictions only. Target IDs never enter here."""
    current, history = initial.clone(), []
    positions = vae.encoder.wpe.weight[:normalized.shape[1]]
    for _ in range(4):
        mean, scale = group_statistics(vae.encoder(current), vae.pre_quant_norm.num_groups, vae.pre_quant_norm.eps)
        current = dictionary_inverse(normalized, mean, scale, positions, vae.encoder.wte.weight[:50258])
        history.append(current.clone())
    return current, history


@torch.inference_mode()
def main(pilot):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("select physicalGPU1 before startup")
    if not torch.cuda.is_available() or torch.cuda.get_device_name() != "NVIDIA GeForce RTX 2060":
        raise ValueError("planned CUDA device required")
    verify(q.read(BROAD_DATA / "producer.json")); verify(q.read(MEASURE / "producer.json"))
    for path, expected in q.read(BROAD_DATA / "result.json")["output_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("prepared broad-codec data changed")
    selected = cohorts(q.read(SEEN), q.read(NOVEL))
    selected["raw_validation"] = q.read(BROAD_DATA / "raw_validation.json")
    gpt, _ = tokenizers(); gpt = CachedVocabulary(gpt)
    validate_targets([r for rows in selected.values() for r in rows], gpt)
    root = PILOT if pilot else MEASUREMENT
    inputs = [SEEN, NOVEL, BROAD_DATA / "raw_validation.json", BROAD_DATA / "result.json",
              CODEC, INITIAL / "config.yaml", INITIAL / "tokenizer.pt", MEASURE / "result.json"]
    inputs += sorted(p for p in GPT2.iterdir() if p.is_file())
    p = producer(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "codec_dictionary_inverse_information_control",
        inputs, {"batch": 4, "precision": "FP32", "physical_gpu": 1, "seed": 42,
            "modes": MODES, "blind_iterations": 4, "distance": "squared_Euclidean", "vocabulary": 50258,
            "pilot": pilot, "cohort_sizes": {k: 2 if pilot else len(v) for k,v in selected.items()}})
    for path in (Path(__file__), PLAN, Path("experiments/structure_net/ncm_broad_decoder_transfer.py"),
                 Path("experiments/structure_net/ncm_unseen_codec_control.py")):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("inverse_"+path.name)).write_bytes(path.read_bytes())
    p["hypothesis_id"] = "ncm-codec-inverse-control-v1"
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    if not pilot:
        previous, receipt = q.read(PILOT / "producer.json"), q.read(PILOT / "result.json")
        if receipt["status"] != "passed" or any(previous[k] != p[k] for k in ("source_hashes", "input_hashes")):
            raise ValueError("matching lifecycle pilot required")
        verify(previous)
    torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
    OmegaConf, VQVAE, _ = vendor_classes()
    config = OmegaConf.load(INITIAL / "config.yaml")
    metrics, outputs, tensors, identities = {}, {}, {}, {}
    for name, checkpoint in (("released", INITIAL / "tokenizer.pt"), ("adapted", CODEC)):
        vae = VQVAE(**config.tokenizer.vqvae)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        vae.load_state_dict(state["model"], strict=True); del state; gc.collect()
        vae.cuda().float().eval().requires_grad_(False)
        if (len(vae.encoder.blocks) or len(vae.encoder.conv_blocks) or vae.encoder.ln_e is not None or
                vae.encoder.down is not None or not isinstance(vae.pre_quant_norm, torch.nn.GroupNorm) or
                vae.pre_quant_norm.affine or vae.encoder.wpe is None):
            raise ValueError("encoder differs from declared lookup plus group normalization")
        identities[name] = module_hash(vae)
        metrics[name], outputs[name] = {}, {}
        for cohort, records in selected.items():
            cases = records[:2] if pilot else records
            counts = {mode: {"cases": len(cases), "nonpad_correct": 0, "nonpad_total": 0, "all_correct": 0,
                             "all_total": 0, "exact_texts": 0, "target_aware": mode in MODES[2:]} for mode in MODES}
            texts, chunks = {mode: [] for mode in MODES}, {mode: [] for mode in (*MODES, "codes", "iterations")}
            for start in range(0, len(cases), 4):
                chunk = cases[start:start+4]
                target = torch.tensor([r["native_target_ids"] for r in chunk], device="cuda")
                codes = encode_codes(vae, target)
                latent = cumulative_latent(vae, codes)
                neural = vae.decoder(latent).argmax(-1)
                # Keep the target-blind call separate and before target statistics.
                blind, history = blind_inverse(vae, latent, neural)
                mean, scale = group_statistics(vae.encoder(target), vae.pre_quant_norm.num_groups, vae.pre_quant_norm.eps)
                positional, vocab = vae.encoder.wpe.weight, vae.encoder.wte.weight[:50258]
                oracle = dictionary_inverse(latent, mean, scale, positional, vocab)
                unquantized = dictionary_inverse(vae.encode_z(target), mean, scale, positional, vocab)
                if not torch.equal(unquantized, target):
                    raise ValueError("unquantized inverse implementation control failed")
                if pilot:
                    repeated, repeated_history = blind_inverse(vae, latent, neural)
                    if not torch.equal(repeated, blind) or any(not torch.equal(a,b) for a,b in zip(history,repeated_history)):
                        raise ValueError("readout order changes target-blind prediction")
                predicted = dict(zip(MODES, (neural, blind, oracle, unquantized)))
                valid = target.ne(50257)
                chunks["codes"].append(codes.cpu()); chunks["iterations"].append(torch.stack(history, 1).cpu())
                for mode, ids in predicted.items():
                    counts[mode]["nonpad_correct"] += int((ids.eq(target) & valid).sum())
                    counts[mode]["nonpad_total"] += int(valid.sum())
                    counts[mode]["all_correct"] += int(ids.eq(target).sum()); counts[mode]["all_total"] += target.numel()
                    chunks[mode].append(ids.cpu())
                    for case, row in zip(chunk, ids.tolist()):
                        text, termination = decode_ids(row, gpt)
                        counts[mode]["exact_texts"] += text == case["target"]
                        texts[mode].append({"id": case["id"], "target": case["target"], "text": text,
                                           "native_output_ids": row, "termination": termination})
            metrics[name][cohort] = {mode: {**v, "nonpad_reconstruction": v["nonpad_correct"]/v["nonpad_total"]}
                                      for mode,v in counts.items()}
            outputs[name][cohort] = texts
            for mode, value in chunks.items():
                tensors[name+"/"+cohort+"/"+mode] = torch.cat(value)
            print(json.dumps({"codec": name, "cohort": cohort, "cases": len(cases)}), flush=True)
        if module_hash(vae) != identities[name]:
            raise ValueError("codec changed")
        del vae; gc.collect(); torch.cuda.empty_cache()
    write_json(root / "outputs.json", outputs)
    save(root / "codes_and_predictions.pt", tensors)
    reload = torch.load(root / "codes_and_predictions.pt", weights_only=True)
    if any(not torch.equal(value, reload[k]) for k,value in tensors.items()):
        raise ValueError("code/prediction reload differs")
    qualification = {model: all(metrics[model][cohort]["blind_inverse_4"]["nonpad_reconstruction"] >= .995 and
        metrics[model][cohort]["blind_inverse_4"]["exact_texts"]/metrics[model][cohort]["blind_inverse_4"]["cases"] >= .9
        for cohort in ("novel", "raw_validation")) for model in metrics}
    verify(p)
    result = {"status": "passed" if pilot else "completed", "metrics": metrics, "codec_qualified": qualification,
        "unquantized_oracle_all_ids_exact": True, "readout_order_exact": True if pilot else None,
        "frozen_model_hashes": identities, "reload_exact": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "producer_fingerprint": p["fingerprint"], "task_accuracy": None, "trained": False,
        "final_test_opened": False, "speed_tested": False, "confirmed": False, "underpowered": True,
        "output_hashes": {str(root/n): file_hash(root/n) for n in ("outputs.json", "codes_and_predictions.pt")}}
    write_json(root / "result.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("metrics", "output_hashes")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["pilot", "measure"])
    main(parser.parse_args().command == "pilot")
