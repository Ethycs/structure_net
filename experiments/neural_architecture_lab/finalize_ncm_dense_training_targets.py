"""Replay full TRAIN dense-label provenance, counts and unchanged decoder state."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_dense_training_targets as run
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.ncm_result_registry import register_records


def replay_reuse(value, saved, row, previous, labels):
    expected = deepcopy(previous[row["id"]])
    expected.update(diagnostic_group=row["diagnostic_group"], label_origin="registered256_exact_reuse")
    if value != expected or not run.prior.same(saved, labels[row["id"]]):
        raise ValueError("reused256 latent, optimizer, complete trace or raw outputs changed")


def finalize():
    if run.os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("independent targets auditor must stay CPU only")
    run.configure()
    rows, previous, labels = run.inputs()
    if run.RESULTS.exists() or run.REPORT.exists():
        raise ValueError("fresh target aggregate and measured report required")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(run.semantic.reference.data.GPT2, local_files_only=True)
    artifacts, records = {}, {}
    for root, selected, updates in ((run.PILOT, run.probes(rows, previous), run.Config().pilot_updates),
            (run.TARGETS, rows, run.Config().updates)):
        run.semantic.reference.data.terminal(root)
        result = run.semantic.reference.q.read(root / "result.json")
        manifest = run.semantic.reference.q.read(root / "manifest.json")
        if list(manifest) != [r["id"] for r in selected] or result["manifest_sha256"] != file_hash(root / "manifest.json"):
            raise ValueError("complete ordered target manifest and actual bytes required")
        values = []
        for row in selected:
            entry = manifest[row["id"]]
            state_path = root / "cases" / (row["id"] + ".pt")
            measurement_path = state_path.with_suffix(".json")
            if (entry["state"] != str(state_path) or entry["measurement"] != str(measurement_path)
                    or entry["state_sha256"] != file_hash(state_path)
                    or entry["measurement_sha256"] != file_hash(measurement_path)
                    or entry["updates"] != updates):
                raise ValueError("bound state/measurement files or fixed budget differ")
            saved = torch.load(state_path, map_location="cpu", weights_only=True)
            value = run.semantic.reference.q.read(measurement_path)
            run.prior.replay_case(value, row, saved, updates, tokenizer)
            target_sha = run.prior.clean.recovery.study.base.tensor_digest({"target": torch.tensor([row["word_ids"]])})
            reused = root == run.TARGETS and row["id"] in previous
            origin = "registered256_exact_reuse" if reused else "fresh_fixed_budget"
            if entry["target_word_sha256"] != target_sha or entry["label_origin"] != origin or value["label_origin"] != origin:
                raise ValueError("actual complete target label or reuse provenance differs")
            if reused:
                replay_reuse(value, saved, row, previous, labels)
            values.append(value)
            artifacts.update({str(p): file_hash(p) for p in (state_path, measurement_path)})
        components = run.aggregate(values)
        gate = None if root == run.PILOT else run.fidelity_gate(components)
        expected_status = "systems_dense_training_targets_passed" if root == run.PILOT else "dense_training_targets_measured"
        if (result["status"] != expected_status or result["cases"] != len(selected) or result["updates"] != updates
                or result["components"] != components or result["latent_fit_gate"] != gate
                or result["reused"] != (0 if root == run.PILOT else 32)
                or not result["codec_unchanged"] or result["codec_parameter_gradients"] or not result["reload_exact"]
                or result["peak_allocated_bytes"] >= run.Config().peak_memory_limit
                or result["task_accuracy"] is not None or result["student_training"] or result["final_test_opened"]
                or result["speed_tested"]):
            raise ValueError("raw target counts, frozen controls, reload or scope differ")
        records[root.name] = {"result": result, "values": values, "manifest": manifest}
        for n in ("producer.json", "result.json", "manifest.json", "worker_terminal_receipt.json"):
            p = root / n
            artifacts[str(p)] = file_hash(p)
    full = records[run.TARGETS.name]
    by_id = {r["id"]: r for r in full["values"]}
    for pilot in records[run.PILOT.name]["values"]:
        value = by_id[pilot["id"]]
        a = torch.load(full["manifest"][pilot["id"]]["state"], map_location="cpu", weights_only=True)
        b = torch.load(records[run.PILOT.name]["manifest"][pilot["id"]]["state"], map_location="cpu", weights_only=True)
        if (value["trace"][:run.Config().pilot_updates] != pilot["trace"]
                or value["arms"][run.prior.ARMS[0]] != pilot["arms"][run.prior.ARMS[0]]
                or not torch.equal(a["initial"], b["initial"])):
            raise ValueError("full optimization failed to retain the actual lifecycle prefix")
    for stage in ("pilot", "targets"):
        p = run.WORKFLOW / "stage_receipts" / (stage + ".json")
        receipt = run.semantic.reference.q.read(p)
        if receipt["stage"] != stage or receipt["exit_code"] != 0 or receipt["observer"] != "subprocess.wait":
            raise ValueError("actual successful numerical child exits required")
        artifacts[str(p)] = file_hash(p)
    producer = run.semantic.reference.q.read(run.WORKFLOW / "producer.json")
    run.semantic.reference.data.verify(producer)
    artifacts.update(producer["input_hashes"])
    artifacts[str(run.WORKFLOW / "producer.json")] = file_hash(run.WORKFLOW / "producer.json")
    gate = full["result"]["latent_fit_gate"]
    campaign = {"hypothesis_id": run.HYPOTHESIS,
        "classification": "dense_full_TRAIN_targets_fit_passed" if gate else "dense_full_TRAIN_targets_fit_failed",
        "latent_fit_gate": gate, "components": full["result"]["components"], "TRAIN_cases": 690,
        "reused": 32, "fresh": 658, "updates_per_case": 256,
        "actual_TRAIN_label_source": str(run.CASES), "model_parameters_trained": False,
        "student_accuracy": None, "confirmed": False, "NCM_success_established": False,
        "final_test_opened": False, "speed_tested": False, "artifacts": artifacts,
        "producer_fingerprint": producer["fingerprint"]}
    run.RESULTS.mkdir()
    path = run.RESULTS / "campaign_results.json"
    write_json(path, campaign)
    table = "\n".join(f"| {name} | {v['correct']}/{v['total']} = {100*v['token_agreement']:.4f}% | {v['exact_target_texts']}/{v['cases']} |"
        for name, v in campaign["components"].items())
    run.REPORT.write_text(f"""# Dense latent labels for the complete NCM training population

Date: 2026-10-23. Hypothesis `{run.HYPOTHESIS}`.
Classification **{campaign['classification']}**. Campaign `{path}`;
SHA256 `{file_hash(path)}`.

| TRAIN population | Target token reconstruction | Exact target texts |
| --- | --- | --- |
{table}

All690 approved training summaries remain in the denominator. The32 registered
256-step latent and optimizer states retain their complete traces and raw
outputs exactly. The other658 labels receive256 fixed FP32 latent-only AdamW
updates through the unchanged original decoder, with math SDPA and TF32 off.
The two-update lifecycle pilot is separate; its exact baseline and update
prefix are retained by full optimization. Every target, EOS/PAD sequence,
latent state, optimizer, raw readout and subgroup count replayed independently.

The199/200 token fidelity gate applies to both training strata and their union.
This is target-aware label preparation. It establishes no source-only student
accuracy or speed gain. Original Llama weights, codec and decoder stay frozen;
no source features or parent model are loaded. Both sealed final cohorts remain
unopened. Student training needs a separate prospective configuration and
subsequent source-disjoint semantic evaluation against the unchanged95% goal.
""")
    print(json.dumps({"classification": campaign["classification"], "components": campaign["components"],
        "campaign_sha256": file_hash(path)}), flush=True)


def register():
    path = run.RESULTS / "campaign_results.json"
    campaign = run.semantic.reference.q.read(path)
    register_records(path, run.REPORT, run.RESULTS, {run.HYPOTHESIS: {
        "name": "Full approved TRAIN population dense latent targets",
        "description": "Fixed256 latent-only target preparation on all690 approved training summaries; no student task accuracy measured.",
        "classification": campaign["classification"]}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("finalize", "register"))
    args = parser.parse_args()
    run.configure()
    finalize() if args.stage == "finalize" else register()
