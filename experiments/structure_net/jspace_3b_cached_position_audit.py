"""CPU-only position-level extension of a completed cached teacher loss audit."""
import argparse
from collections import Counter
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from .jspace_3b_cached_teacher_audit import identity, read, sha, write


POSITIONS = ("first", "midpoint", "penultimate", "answer")


def summarize(rows, tokenizer):
    summary = {}
    for split in ("train", "development"):
        split_rows = [row for row in rows if row["split"] == split]
        groups = {"all": split_rows}
        groups.update({family: [r for r in split_rows if r["family"] == family]
                       for family in sorted({r["family"] for r in split_rows})})
        summary[split] = {}
        for family, group in groups.items():
            positions = {}
            for p, name in enumerate(POSITIONS):
                counts = Counter(row["targets"][p] for row in group)
                frequencies = []
                for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                    matched = [r for r in group if r["targets"][p] == token]
                    frequencies.append(dict(token_id=token, decoded=tokenizer.decode([token], skip_special_tokens=False),
                        count=count, fraction=count/len(group),
                        native_ce=sum(r["native"]["ce"][p] for r in matched)/count,
                        accepted_parent_ce=sum(r["accepted_parent"]["ce"][p] for r in matched)/count))
                measurement = dict(unique_targets=len(counts), target_frequency=frequencies)
                for model in ("native", "accepted_parent"):
                    measurement[model] = dict(ce=sum(r[model]["ce"][p] for r in group)/len(group),
                        full_vocab_top1_correct=sum(r[model]["top1_correct"][p] for r in group))
                measurement["ce_reduction_native_to_parent"] = measurement["native"]["ce"] - measurement["accepted_parent"]["ce"]
                positions[name] = measurement
            total_reduction = sum(p["ce_reduction_native_to_parent"] for p in positions.values())
            summary[split][family] = dict(count=len(group), positions=positions,
                structural_first_penultimate_share_of_ce_reduction=(
                    positions["first"]["ce_reduction_native_to_parent"] +
                    positions["penultimate"]["ce_reduction_native_to_parent"])/total_reduction,
                four_position_ce_reduction=total_reduction/4)
    return summary


def run(root, prior_audit, diagnostic_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This audit requires CUDA hidden")
    os.nice(10)
    torch.set_num_threads(1)
    root, prior_audit, diagnostic_root = map(lambda p: Path(p).resolve(), (root, prior_audit, diagnostic_root))
    if root.exists():
        raise FileExistsError(root)
    previous_plan, previous_manifest = read(prior_audit/"plan.json"), read(prior_audit/"manifest.json")
    if (previous_manifest["status"] != "completed"
            or sha(prior_audit/"plan.json") != previous_manifest["plan_sha256"]
            or sha(prior_audit/"result.json") != previous_manifest["result_sha256"]
            or sha(previous_plan["source"]) != previous_plan["source_sha256"]
            or sha(diagnostic_root/"plan.json") != previous_plan["diagnostic_plan_sha256"]):
        raise ValueError("Completed original audit provenance changed")
    diagnostic = read(diagnostic_root/"plan.json")
    campaign_path = Path(diagnostic["campaign"])/"plan.json"
    if sha(campaign_path) != previous_plan["campaign_plan_sha256"]:
        raise ValueError("Campaign identity changed")
    campaign = read(campaign_path)
    inputs = previous_plan["inputs"]
    for entry in inputs.values():
        if identity(entry["path"]) != entry["identity"]:
            raise ValueError("Previously authenticated input changed")
    items = read(inputs["items"]["path"])["items"]
    previous_rows = read(prior_audit/"result.json")["records"]
    if len(items) != 576 or len(previous_rows) != len(items):
        raise ValueError("Exactly the previously audited512TRAIN/64DEV cases required")
    for i, case in enumerate(items):
        n = len(case["input_ids"])
        expected_positions = [0, (n-2)//2, n-2, n-1]
        expected_targets = [case["input_ids"][p+1] for p in expected_positions[:3]] + [campaign["label_ids"][case["answer"]]]
        if (case["loss_positions"] != expected_positions or case["targets"] != expected_targets
                or case["identity"] != previous_rows[i]["identity"]):
            raise ValueError("Original four-position target contract changed")
    model = campaign["model"]
    tokenizer_files = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        path = Path(model["path"])/name
        if sha(path) != model["files"][name]:
            raise ValueError("Pinned tokenizer changed")
        tokenizer_files[str(path)] = model["files"][name]
    tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True, trust_remote_code=False)
    native = np.load(inputs["native"]["path"], mmap_mode="r", allow_pickle=False)
    parent = np.load(inputs["accepted_parent"]["path"], mmap_mode="r", allow_pickle=False)
    root.mkdir(parents=True)
    source = Path(__file__).resolve()
    write(root/"plan.json", dict(schema="jspace.3b-cached-position-audit.v1", source=str(source), source_sha256=sha(source),
        prior_audit=str(prior_audit), prior_manifest_sha256=sha(prior_audit/"manifest.json"),
        prior_plan_sha256=sha(prior_audit/"plan.json"), prior_result_sha256=sha(prior_audit/"result.json"),
        inputs=inputs, original_indices=previous_plan["original_indices"], tokenizer_files=tokenizer_files,
        positions=list(POSITIONS), position_formula="[0,floor((n-2)/2),n-2,n-1]", threads=1,
        nice_increment=10, model_forward=False, heldout_access=False, confirmed=False))
    started = time.monotonic()
    rows = []
    try:
        with torch.inference_mode():
            for start in range(0, len(items), 4):
                indices = list(range(start, min(start+4, len(items))))
                cases = [items[i] for i in indices]
                targets = torch.tensor([c["targets"] for c in cases])
                measurements = {}
                for name, bank, mapping in (("native", native, previous_plan["original_indices"]),
                                            ("accepted_parent", parent, list(range(len(items))))):
                    logits = torch.from_numpy(np.array(bank[[mapping[i] for i in indices]], copy=True)).view(torch.bfloat16).float()
                    if not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite cached logits")
                    ce = -logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
                    top1 = logits.argmax(-1).eq(targets)
                    measurements[name] = []
                    for j, index in enumerate(indices):
                        if not math.isclose(float(ce[j].mean()), previous_rows[index][name]["ce_four_positions"], rel_tol=2e-6, abs_tol=2e-6):
                            raise ValueError("Per-position CE does not reproduce completed original audit")
                        measurements[name].append(dict(ce=ce[j].tolist(), top1_correct=top1[j].tolist()))
                    del logits, ce, top1
                for j, (index, case) in enumerate(zip(indices, cases)):
                    rows.append(dict(source_index=index, identity=case["identity"], split=case["split"], family=case["family"],
                        loss_positions=case["loss_positions"], targets=case["targets"],
                        **{name: values[j] for name, values in measurements.items()}))
        for entry in inputs.values():
            if identity(entry["path"]) != entry["identity"]:
                raise ValueError("Input changed during audit")
        result = dict(status="completed", summary=summarize(rows, tokenizer), records=rows,
            elapsed_seconds=time.monotonic()-started, ce="Full-vocabulary FP32 log-softmax of cached BF16 logits; natural logarithms",
            frequency="Actual supervised next-token targets, except final correct answer label; local pinned tokenizer decode",
            confirmed=False)
        write(root/"result.json", result)
        write(root/"manifest.json", dict(status="completed", plan_sha256=sha(root/"plan.json"),
            result_sha256=sha(root/"result.json"), source_sha256=sha(source)))
        return dict(root=str(root), elapsed_seconds=result["elapsed_seconds"],
            summary={split: result["summary"][split]["all"] for split in result["summary"]})
    finally:
        native._mmap.close()
        parent._mmap.close()


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prior-audit", required=True, type=Path)
    parser.add_argument("--diagnostic-root", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.root, args.prior_audit, args.diagnostic_root)
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}), flush=True)
