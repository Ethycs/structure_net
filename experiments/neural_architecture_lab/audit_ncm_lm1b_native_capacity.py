"""CPU replay of fixed16 native-NCM training; no heldout task claim."""
import json
import math
from pathlib import Path

import torch

from experiments.structure_net import ncm_lm1b_native_capacity as run

DEST = run.ROOT / "20261024_lm1b_native16_capacity_audit_v1"


def trace(root, rows, manifest, codes, saved, pilot=False):
    sampler = torch.Generator().manual_seed(run.Config().sampler_seed)
    neighbor = torch.Generator().manual_seed(run.Config().neighbor_seed)
    selected = run.q.read(run.DATA / "selection.json")["pilot_indices"]
    expected_updates, accumulation = (2, 1) if pilot else (256, 4)
    count = 0
    histories = 0
    for count, line in enumerate((root / "trace.jsonl").read_text().splitlines(), 1):
        row = json.loads(line)
        permutation = torch.randperm(16, generator=sampler).tolist()
        rate = run.Config().max_lr if pilot else run.rate(count)
        if (row["step"] != count or row["permutation"] != permutation or row["lr"] != rate
                or not math.isfinite(row["gradient_norm"]) or len(row["microbatches"]) != accumulation):
            raise ValueError("actual fixed-budget permutation/rate/finite trace differs")
        for i, micro in enumerate(row["microbatches"]):
            indices = selected * 2 if pilot else permutation[4 * i:4 * i + 4]
            chosen = [rows[k] for k in indices]
            if (micro["case_ids"] != [r["id"] for r in chosen]
                    or micro["source_positions"] != max(manifest[r["id"]]["source_tokens"] for r in chosen)):
                raise ValueError("complete original TRAIN-case/source-length trace differs")
            clean = torch.tensor([codes[r["id"]] for r in chosen], dtype=torch.long)
            _, proof = run.lifecycle.objective.original.corrupt_history(clean, 5120 + count, neighbor)
            losses = micro["losses"]
            names = {"clean_concept_loss", "clean_word_loss", "noisy_concept_loss", "noisy_word_loss", "corruption"}
            if losses["corruption"] != proof:
                raise ValueError("native neighborhood-noise stream differs")
            if i == 0:
                names |= {"self_concept_loss", "self_word_loss", "source_only_history_codes", "temperature",
                    "auxiliary_concept_weight", "auxiliary_word_weight", "word_gradient_into_concept_attention"}
                history = losses["source_only_history_codes"]
                if (len(history) != 4 or any(len(h) != 641 or any(type(v) is not int or not 0 <= v < 8192 for v in h) for h in history)
                        or losses["temperature"] != 1. or losses["auxiliary_concept_weight"] != .25
                        or losses["auxiliary_word_weight"] != 1. or losses["word_gradient_into_concept_attention"] is not True):
                    raise ValueError("complete source-only TRAIN history and word-gradient weights required")
                histories += 1
            if set(losses) != names or any(not math.isfinite(v) or v < 0 for k, v in losses.items() if k.endswith("_loss")):
                raise ValueError("exact finite native TRAIN objective required")
    if (count != expected_updates or histories != expected_updates
            or not torch.equal(sampler.get_state(), saved["sampler_rng"])
            or not torch.equal(neighbor.get_state(), saved["neighbor_rng"])):
        raise ValueError("complete TRAIN-control sampler/noise/checkpoint RNG replay differs")
    return {"updates": count, "sampled_presentations": count * accumulation * 4, "source_only_history_microbatches": histories}


def main():
    if DEST.exists() or torch.cuda.is_initialized():
        raise ValueError("fresh CPU TRAIN-control audit required")
    run.native.vendor.corpus.quality.dependencies()
    run.require_systems()
    if run.q.read(run.WORKFLOW / "tool_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("actual whole-control tool completion required")
    for root in (run.DATA, run.PILOT, run.TRAIN):
        run.native.vendor.terminal(root)
    rows = run.q.read(run.DATA / "cases.json")
    expected = run.choose(run.q.read(run.training.DATA / "cases.json"), run.source_data.storage_encoding())
    if rows != expected or len(rows) != 16:
        raise ValueError("unchanged predeclared16 TRAIN-only cases required")
    manifest = run.q.read(run.DATA / "manifest.json")
    codes = {r["id"]: r["codes"] for r in run.q.read(run.training.CACHE / "outputs.json")}
    tokenizer = run.native.tokenizer()
    results = {}
    for root, pilot in ((run.PILOT, True), (run.TRAIN, False)):
        result = run.q.read(root / "result.json")
        saved = torch.load(root / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        updates = 2 if pilot else 256
        if (result["updates"] != updates or result["cases"] != 16 or not result["strict_reload_passed"]
                or result["initial_trainable_sha256"] != run.INITIAL or result["pilot_weights_reused"]
                or saved["step"] != updates or saved["config"] != run.asdict(run.Config())
                or result["checkpoint_sha256"] != run.lifecycle.file_hash(root / "model.pt")
                or result["peak_allocated_bytes"] > 8_000_000_000 or result["source_disjoint_task_accuracy"] is not None
                or result["semantic_TRAIN_accuracy"] is not None or result["speed_tested"]
                or any(int(v["step"]) != updates for v in saved["optimizer"]["state"].values())):
            raise ValueError("actual fixed-budget native control state/optimizer/limits differ")
        replay = trace(root, rows, manifest, codes, saved, pilot)
        model = run.lifecycle.mechanism.enable(run.lifecycle.head.from_domain(device="cpu"))
        model.load_state_dict(saved["model"], strict=True)
        if (run.lifecycle.trainable_hash(model) != result["final_trainable_sha256"]
                or run.lifecycle.protected(model) != {"encoder": run.lifecycle.ENCODER, "VQ": run.lifecycle.VQ}
                or run.native.vendor.module_hash(model.concepts.generator) != result["final_native_hashes"]["generator"]
                or run.native.vendor.module_hash(model.codec.decoder) != result["final_native_hashes"]["word_decoder"]):
            raise ValueError("strict actual native checkpoint or frozen target geometry differs")
        model.close()
        del model
        values = run.q.read(root / "outputs.json")
        if [v["id"] for v in values] != [r["id"] for r in rows]:
            raise ValueError("all16 source-only TRAIN outputs required")
        correct, total = 0, 0
        for value, source in zip(values, rows):
            decoded = run.inference.decode(value["raw_word_ids"], tokenizer)
            if (any(value[k] != v for k, v in decoded.items()) or len(value["codes"]) != 641
                    or any(type(v) is not int or not 0 <= v < 8192 for v in value["codes"])
                    or value["source_positions"] != manifest[source["id"]]["source_tokens"]
                    or value["source_sha256"] != source["source_sha256"]):
                raise ValueError("complete native TRAIN generation metadata or rendering differs")
            correct += sum(a == b for a, b in zip(source["native_word_ids"], value["raw_word_ids"]) if a != 0)
            total += sum(v != 0 for v in source["native_word_ids"])
        if correct / total != result["word_ID_fit_diagnostic_only"]:
            raise ValueError("descriptive word-ID fit replay differs")
        results["pilot" if pilot else "full"] = {**replay, "word_ID_fit_diagnostic_only": correct / total,
            "checkpoint_sha256": result["checkpoint_sha256"], "peak_allocated_bytes": result["peak_allocated_bytes"]}
    if run.q.read(run.PILOT / "initial_outputs.json") != run.q.read(run.TRAIN / "initial_outputs.json"):
        raise ValueError("pilot and full run must start with identical warm source-only behavior")
    artifacts = {str(p): run.lifecycle.file_hash(p) for root in (run.DATA, run.PILOT, run.TRAIN, run.WORKFLOW)
        for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts}
    artifacts[str(Path(__file__))] = run.lifecycle.file_hash(__file__)
    result = {"status": "native16_TRAIN_control_trace_and_state_replayed_semantic_review_pending", "results": results,
        "source_disjoint_task_accuracy": None, "semantic_TRAIN_accuracy": None, "all_final_sources_closed": True,
        "speed_tested": False, "artifacts": artifacts}
    DEST.mkdir()
    run.lifecycle.write_json(DEST / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), flush=True)


if __name__ == "__main__":
    main()
