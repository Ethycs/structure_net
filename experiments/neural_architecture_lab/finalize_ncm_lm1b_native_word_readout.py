"""Independently replay saved logits with the pinned author word-readout policy."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_lm1b_native_word_readout as control
from experiments.neural_architecture_lab import finalize_ncm_lm1b_grounded_development as shared
from experiments.neural_architecture_lab.ncm_result_registry import register_records
from experiments.structure_net.ncm_memory_data import file_hash, write_json


def finalize():
    control.configure()
    run = control.engine
    control.require_baseline()
    original = {r["id"]: r for r in control.q.read(control.BASE_OUTPUTS / "outputs.json")}
    select = control.native_selector(control.allowance())
    count = 0
    for root, expected in ((control.PILOT, 2), (control.OUTPUTS, 128)):
        control.native.vendor.terminal(root)
        manifest = control.q.read(root / "word_logits_manifest.json")
        outputs = control.q.read(root / "outputs.json")
        result = control.q.read(root / "result.json")
        if (len(outputs) != expected or set(manifest) != {r["id"] for r in outputs}
                or result["word_logits_manifest_sha256"] != file_hash(root / "word_logits_manifest.json")
                or result["word_readout_policy"] != control.POLICY
                or result["model_sha256"] != control.q.read(control.BASE_OUTPUTS / "result.json")["model_sha256"]):
            raise ValueError("matched neural model, complete saved logits and exact word policy required")
        for row in outputs:
            entry = manifest[row["id"]]
            path = root / "word_logits" / (row["id"] + ".pt")
            if str(path) != entry["path"] or file_hash(path) != entry["sha256"]:
                raise ValueError("saved word-logit bytes changed")
            logits = torch.load(path, map_location="cpu", weights_only=True)
            if (logits.shape != (1, 128, 30528) or logits.dtype != torch.float32 or not torch.isfinite(logits).all()
                    or control.tensor_hash(logits) != row["word_logits_tensor_sha256"]
                    or entry["tensor_sha256"] != row["word_logits_tensor_sha256"]):
                raise ValueError("complete finite original FP32 word logits required")
            baseline = original[row["id"]]
            if (logits.argmax(-1).tolist()[0] != baseline["raw_word_ids"] or row["codes"] != baseline["codes"]
                    or row["baseline_plain_word_ids_sha256"] != control.q.digest(baseline["raw_word_ids"])
                    or not row["baseline_concept_codes_unchanged"]):
                raise ValueError("neural concepts or original unpenalized word predictions changed")
            if select(logits).tolist()[0] != row["raw_word_ids"]:
                raise ValueError("independent CPU author word-policy replay differs")
            count += 1
    shared.run = run
    shared.finalize()
    proof = {"status": "all_saved_word_logits_and_native_selector_replayed", "frames": count,
        "task_DEV_cases": 128, "exact_concept_parity": True, "exact_plain_word_argmax_parity": True,
        "policy": control.POLICY, "native_word_selection_precision": "FP64", "neural_precision": "FP32",
        "same_fixed4096_neural_weights": True, "new_model_training": False,
        "full_author_concept_sampling_policy_replicated": False}
    path = control.RESULTS / "native_word_policy_replay.json"
    write_json(path, proof)
    campaign_path = control.RESULTS / "campaign_results.json"
    campaign = control.q.read(campaign_path)
    campaign["word_readout_control"] = proof
    for artifact in (path, Path(__file__), Path(control.__file__), control.AUTHOR_SCRIPT, control.AUTHOR_CONFIG, control.AUTHOR_GENERATOR):
        campaign["artifacts"][str(artifact)] = file_hash(artifact)
    write_json(campaign_path, campaign)
    with control.REPORT.open("a") as report:
        report.write(
            "\nThis follow-up changes only word selection. The same fixed4096 model\n"
            "reproduced all baseline concept IDs and unpenalized word predictions exactly.\n"
            "The unchanged author pick_tokens function was extracted from the pinned\n"
            "evaluation script: frequency11.5, deficit2, adjacency10/window4/decay0.5\n"
            "and the exact released token-unigram allowance, computed in FP64. All\n"
            "128positions of every logit frame were saved; CPU replay matched every\n"
            "selected word ID, including both retained inference pilots.\n\n"
            "Concept generation stays at the baseline deterministic policy and the\n"
            "existing predicted-CLS renderer stays unchanged. This is a word-readout\n"
            "control, not a full replication of author unconditional sampling. No\n"
            "further weights were trained and no prior judgment was rerolled.\n"
            f"Final campaign SHA256: `{file_hash(campaign_path)}`.\n")
    print(json.dumps({"classification": campaign["classification"], "DEV_qualified": campaign["DEV_qualified"],
        "word_policy_replayed_frames": count, "campaign_sha256": file_hash(campaign_path)}), flush=True)


def register():
    path = control.RESULTS / "campaign_results.json"
    data = control.q.read(path)
    register_records(path, control.REPORT, control.RESULTS, {control.HYPOTHESIS: {
        "name": "Author word-readout policy on the fixed source-grounded NCM",
        "description": "Same fixed4096 pretrained NCM model, identical source-generated concepts and unpenalized word logits; native frequency and adjacency word selection scored on all128 source-disjoint cases. Semantic full-summary evaluation, not exact match.",
        "classification": data["classification"], "DEV_qualified": data["DEV_qualified"],
        "canonical_first_judgments": data["canonical_first_judgments"], "source_only": True,
        "all128_cases_retained": True, "neural_weights_changed": False, "final_task_accuracy_measured": False}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    {"finalize": finalize, "register": register}[args.stage]()
