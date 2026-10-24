"""Construct frozen-decoder latent labels for every approved TRAIN summary.

This prepares labels only. Target words optimize per-case tensors; neither an
NCM student nor the frozen Llama body is trained or evaluated by this study.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

from experiments.structure_net import ncm_decoder_latent_convergence as upstream
from experiments.structure_net import ncm_dense_semantic_development as semantic
from experiments.structure_net.ncm_memory_data import file_hash, write_json

prior = upstream.prior
ROOT = prior.ROOT
HYPOTHESIS = "ncm-dense-full-TRAIN-targets-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-dense-training-targets-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-dense-training-targets.md")
WORKFLOW = ROOT / "20261023_dense_training_targets_workflow_v1"
PILOT = ROOT / "20261023_dense_training_targets_pilot_v1"
TARGETS = ROOT / "20261023_dense_training_targets_measure_v1"
RESULTS = ROOT / "20261023_dense_training_targets_results_v1"
MODULE = "experiments.structure_net.ncm_dense_training_targets"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_dense_training_targets"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_dense_training_targets.py")
CASES = ROOT / "20261023_reviewed_draft_data_v1/cases.json"
CASES_SHA = "ac3c56f7d03cf9160de9e719ef042b44a0bf12dececd778949b46e7288aac983"
UPSTREAM_SHA = "1a5587221af8048f3052296bb17d6c6ab40afdf7e62d689088b7b0061070a834"
STRATA = ("original_252", "reviewed_438", "all_690")


@dataclass(frozen=True)
class Config(prior.Config):
    updates: int = 256
    constructor_seed: int = 50142


def configure():
    prior.clean.dependencies()
    prior.clean.recovery.configure()


def bind_rows(rows):
    if len(rows) != 690 or len({r["id"] for r in rows}) != 690:
        raise ValueError("all690 ordered unique TRAIN identities required")
    result = []
    for i, row in enumerate(rows):
        words = row["word_ids"]
        active = [w for w in words if w != Config().pad]
        if (row["split"] != "train" or len(words) != 192 or len(active) < 2
                or active[-1] != 50256 or 50256 in active[:-1]
                or any(type(w) is not int or not 0 <= w <= 50257 for w in words)
                or words != active + [50257] * (192 - len(active))
                or (i < 252) == ("training_origin" in row)):
            raise ValueError("unchanged TRAIN ordering, complete target and EOS/PAD required")
        result.append({**row, "diagnostic_group": STRATA[int(i >= 252)]})
    return result


def admitted():
    path = semantic.RESULTS / "campaign_results.json"
    if not path.exists():
        return False
    q = semantic.reference.q
    campaign = q.read(path)
    receipt = q.read(semantic.RESULTS / (semantic.HYPOTHESIS + "_registry_readback.json"))
    terminal = q.read(semantic.WORKFLOW / "result.json")
    if (receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != file_hash(path)
            or receipt["exact"]["campaign_sha256"] != file_hash(path)
            or terminal["status"] != "dense_semantic_development_completed"
            or len(terminal["stages"]) != 7 or any(r["exit_code"] for r in terminal["stages"])):
        raise ValueError("complete registered semantic evaluation and seven actual exits required")
    return campaign["classification"] == "dense_semantic_development_quality_failed"


def inputs(require_admission=True):
    configure()
    q = semantic.reference.q
    if file_hash(CASES) != CASES_SHA:
        raise ValueError("immutable full690 TRAIN labels required")
    rows = bind_rows(q.read(CASES)["train"])
    path = upstream.RESULTS / "campaign_results.json"
    campaign = q.read(path)
    receipt = q.read(upstream.RESULTS / (upstream.HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(path) != UPSTREAM_SHA or not campaign["latent_fit_gate"]
            or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != UPSTREAM_SHA
            or receipt["exact"]["campaign_sha256"] != UPSTREAM_SHA):
        raise ValueError("registered complete256-step latent control required")
    for name in ("case_measurements.json", "latents.pt", "result.json", "producer.json", "worker_terminal_receipt.json"):
        p = upstream.MEASURE / name
        if campaign["artifacts"].get(str(p)) != file_hash(p):
            raise ValueError("bound successful32-case numerical evidence changed")
    previous = {r["id"]: r for r in q.read(upstream.MEASURE / "case_measurements.json")}
    labels = torch.load(upstream.MEASURE / "latents.pt", map_location="cpu", weights_only=True)
    if len(previous) != 32 or set(labels) != set(previous) or not set(previous) <= {r["id"] for r in rows}:
        raise ValueError("exact32 reusable TRAIN identities required")
    for row in rows:
        if row["id"] in previous:
            old = previous[row["id"]]
            if (old["source_sha256"] != row["source_sha256"] or old["target_text"] != row["target_text"]
                    or old["target_word_ids"] != row["word_ids"]):
                raise ValueError("reused target must match complete approved TRAIN label")
    source_path = semantic.reference.SOURCES / "cases.json"
    if file_hash(source_path) != semantic.SOURCE_SHA:
        raise ValueError("unchanged development exclusion population required")
    for values in q.read(source_path).values():
        semantic.reference.data.prior.check_disjoint(rows, values)
    if require_admission and not admitted():
        raise ValueError("registered semantic development quality failure required before CUDA work")
    return rows, previous, labels


def probes(rows, previous):
    fresh = [r for r in rows if r["id"] not in previous]
    first = fresh[0]
    longest = sorted([r for r in fresh if r["id"] != first["id"]],
        key=lambda r: (-sum(w != 50257 for w in r["word_ids"]), r["id"]))[0]
    return [first, longest]


def producing(destination, stage, extra=()):
    old = semantic.reference.q.read(upstream.WORKFLOW / "producer.json")
    files = list(dict.fromkeys([Path(__file__), PLAN, AUDITOR_FILE,
        Path(upstream.__file__), Path(prior.__file__), Path(semantic.__file__),
        *[Path(p) for p in old["source_hashes"]]]))
    common = [CASES, semantic.reference.SOURCES / "cases.json"]
    common += [upstream.RESULTS / "campaign_results.json",
        upstream.RESULTS / (upstream.HYPOTHESIS + "_registry_readback.json")]
    common += [upstream.MEASURE / n for n in
        ("case_measurements.json", "latents.pt", "result.json", "producer.json", "worker_terminal_receipt.json")]
    common += [semantic.RESULTS / "campaign_results.json",
        semantic.RESULTS / (semantic.HYPOTHESIS + "_registry_readback.json"), semantic.WORKFLOW / "result.json"]
    common += [prior.continuous.released.INITIAL / n for n in ("config.yaml", "tokenizer.pt")]
    common += sorted(p for p in semantic.reference.data.GPT2.iterdir() if p.suffix in (".json", ".txt"))
    p = semantic.reference.data.freeze(destination, stage, [*common, *files, *extra],
        {**asdict(Config()), "hypothesis_id": HYPOTHESIS, "TRAIN_cases": 690,
         "reused_cases": 32, "fresh_cases": 658, "precision": "FP32_math_SDPA_TF32_off",
         "parent_body_loaded": False, "student_training": False, "oracle_target_labels": True,
         "optimized_variable": "per_case_latent_only", "codec_and_decoder_frozen": True,
         "minimum_token_agreement": "199/200_each_stratum_and_union",
         "selected_target": "fixed256_only", "task_accuracy": None,
         "torch": torch.__version__, "cuda": torch.version.cuda,
         "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (destination / "producer_source" / f"dense_targets_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = semantic.reference.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(destination / "producer.json", p)
    return p


def aggregate(values):
    result = {}
    for group in STRATA:
        selected = values if group == STRATA[-1] else [r for r in values if r["diagnostic_group"] == group]
        correct = sum(r["arms"][prior.ARMS[1]]["correct"] for r in selected)
        total = sum(r["arms"][prior.ARMS[1]]["total"] for r in selected)
        result[group] = {"cases": len(selected), "correct": correct, "total": total,
            "token_agreement": correct / total if total else None,
            "exact_target_texts": sum(r["arms"][prior.ARMS[1]]["exact_target_text"] for r in selected)}
    return result


def fidelity_gate(components):
    if ([components[s]["cases"] for s in STRATA] != [252, 438, 690]
            or any(components[STRATA[0]][k] + components[STRATA[1]][k] != components[STRATA[2]][k]
                   for k in ("correct", "total", "exact_target_texts"))):
        raise ValueError("complete strata must compose to all690")
    return all(v["total"] > 0 and 0 <= v["correct"] <= v["total"]
        and v["correct"] * 200 >= v["total"] * 199 for v in components.values())


def run(pilot=False):
    rows, previous, labels = inputs()
    selected = probes(rows, previous) if pilot else rows
    destination = PILOT if pilot else TARGETS
    extra = []
    if not pilot:
        semantic.reference.data.terminal(PILOT)
        if semantic.reference.q.read(PILOT / "result.json")["status"] != "systems_dense_training_targets_passed":
            raise ValueError("successful fresh latent/decoder/optimizer lifecycle pilot required")
        extra += [PILOT / n for n in ("producer.json", "result.json", "manifest.json", "worker_terminal_receipt.json")]
    p = producing(destination, "systems_lifecycle_only_not_quality_evidence" if pilot else "full690_fixed256_TRAIN_labels", extra)
    prior.gpu()
    torch.manual_seed(Config().constructor_seed)
    torch.cuda.manual_seed_all(Config().constructor_seed)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(semantic.reference.data.GPT2, local_files_only=True)
    codec, facade = prior.load_codec()
    pilot_manifest = {} if pilot else semantic.reference.q.read(PILOT / "manifest.json")
    (destination / "cases").mkdir()
    values, manifest = [], {}
    for row in selected:
        target = torch.tensor([row["word_ids"]], device="cuda")
        active = [w for w in row["word_ids"] if w != Config().pad]
        if tokenizer.decode(active[:-1], skip_special_tokens=False) != row["target_text"]:
            raise ValueError("complete approved target must round-trip exactly")
        reused = row["id"] in previous and not pilot
        updates = Config().pilot_updates if pilot else Config().updates
        if reused:
            saved = labels[row["id"]]
            value = deepcopy(previous[row["id"]])
            value["diagnostic_group"] = row["diagnostic_group"]
            for arm, key in zip(prior.ARMS, ("initial", "refined")):
                if prior.readout(facade, saved[key].cuda(), target, tokenizer) != value["arms"][arm]:
                    raise ValueError("same-GPU reused256 labels must retain exact readouts")
        else:
            initial = prior.encoder_latent(facade, target)
            baseline = prior.readout(facade, initial, target, tokenizer)
            saved, trace = prior.refine(facade, initial, target, updates, Config())
            value = {"id": row["id"], "source_sha256": row["source_sha256"],
                "diagnostic_group": row["diagnostic_group"], "target_word_ids": row["word_ids"],
                "target_text": row["target_text"], "arms": {prior.ARMS[0]: baseline,
                    prior.ARMS[1]: prior.readout(facade, saved["refined"].cuda(), target, tokenizer)},
                "trace": trace, "final_anchor_MSE": float(F.mse_loss(saved["refined"], saved["initial"]))}
        value["label_origin"] = "registered256_exact_reuse" if reused else "fresh_fixed_budget"
        prior.replay_case(value, row, saved, updates, tokenizer)
        if row["id"] in pilot_manifest:
            entry = pilot_manifest[row["id"]]
            earlier = semantic.reference.q.read(entry["measurement"])
            state = torch.load(entry["state"], map_location="cpu", weights_only=True)
            if (value["trace"][:Config().pilot_updates] != earlier["trace"]
                    or value["arms"][prior.ARMS[0]] != earlier["arms"][prior.ARMS[0]]
                    or not torch.equal(saved["initial"], state["initial"])):
                raise ValueError("full fresh optimization changed exact pilot prefix or baseline")
        state_path = destination / "cases" / (row["id"] + ".pt")
        measurement_path = state_path.with_suffix(".json")
        torch.save(saved, state_path)
        write_json(measurement_path, value)
        restored = torch.load(state_path, map_location="cpu", weights_only=True)
        if not prior.same(saved, restored):
            raise ValueError("complete latent and optimizer serialization must be exact")
        manifest[row["id"]] = {"state": str(state_path), "state_sha256": file_hash(state_path),
            "measurement": str(measurement_path), "measurement_sha256": file_hash(measurement_path),
            "target_word_sha256": prior.clean.recovery.study.base.tensor_digest({"target": target}),
            "updates": updates, "label_origin": value["label_origin"]}
        values.append(value)
        write_json(destination / "manifest.json", manifest)
        write_json(destination / "progress.json", {"completed_cases": len(values), "total_cases": len(selected),
            "fresh": sum(v["label_origin"] == "fresh_fixed_budget" for v in values), "updates": updates})
        if pilot or len(values) % 16 == 0:
            print(json.dumps({"completed": len(values), "total": len(selected)}), flush=True)
    if pilot:
        del codec, facade
        gc.collect()
        torch.cuda.empty_cache()
        codec, facade = prior.load_codec()
        for row, value in zip(selected, values):
            saved = torch.load(manifest[row["id"]]["state"], map_location="cpu", weights_only=True)
            opt = prior.optimizer(saved["refined"].cuda().requires_grad_(True), Config())
            opt.load_state_dict(prior.to_cpu(saved["optimizer"]))
            if not prior.same(prior.to_cpu(opt.state_dict()), saved["optimizer"]):
                raise ValueError("fresh actual optimizer reload differs")
            target = torch.tensor([row["word_ids"]], device="cuda")
            for arm, key in zip(prior.ARMS, ("initial", "refined")):
                if prior.readout(facade, saved[key].cuda(), target, tokenizer) != value["arms"][arm]:
                    raise ValueError("fresh decoder reload changed pilot raw output")
    peak = torch.cuda.max_memory_allocated()
    if (peak >= Config().peak_memory_limit or prior.module_hash(codec) != prior.continuous.EXPECTED_CODEC
            or any(v.grad is not None for v in codec.parameters())):
        raise ValueError("frozen original codec or reserved2GB memory bound failed")
    components = aggregate(values)
    semantic.reference.data.verify(p)
    write_json(destination / "result.json", {"status": "systems_dense_training_targets_passed" if pilot else "dense_training_targets_measured",
        "cases": len(values), "updates": updates, "components": components,
        "latent_fit_gate": None if pilot else fidelity_gate(components),
        "reused": sum(v["label_origin"] == "registered256_exact_reuse" for v in values),
        "codec_unchanged": True, "codec_parameter_gradients": False, "reload_exact": True,
        "peak_allocated_bytes": peak, "manifest_sha256": file_hash(destination / "manifest.json"),
        "task_accuracy": None, "student_training": False, "final_test_opened": False,
        "speed_tested": False, "producer_fingerprint": p["fingerprint"]})


def workflow():
    inputs()
    if any(p.exists() for p in (WORKFLOW, PILOT, TARGETS, RESULTS, REPORT,
            Path("data/hypothesis_registry") / (HYPOTHESIS + ".json"))):
        raise ValueError("fresh append-only targets study required")
    p = producing(WORKFLOW, "supervise_full_TRAIN_dense_target_preparation")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}
    try:
        for stage, root in (("pilot", PILOT), ("targets", TARGETS), ("finalize", RESULTS), ("register", RESULTS)):
            command = [sys.executable, "-u", "-m", AUDITOR if stage in ("finalize", "register") else MODULE, stage]
            semantic.reference.data.verify(p)
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "2" if stage in ("pilot", "targets") else ""})
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"status": "observed_worker_terminal", "stage": stage, "exit_code": code,
                "observer_pid": os.getpid(), "child_pid": child.pid, "observer": "subprocess.wait", "command": command}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if root.exists():
                write_json(root / ("registry_terminal_receipt.json" if stage == "register" else "worker_terminal_receipt.json"), receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("dense training-target stage failed: " + stage)
        semantic.reference.data.verify(p)
        journal.update(status="dense_training_targets_completed", active_stage=None,
            classification=semantic.reference.q.read(RESULTS / "campaign_results.json")["classification"])
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "workflow", "pilot", "targets"))
    args = parser.parse_args()
    configure()
    if args.stage == "check":
        rows, previous, _ = inputs(require_admission=False)
        print(json.dumps({"TRAIN_cases": len(rows), "reused": len(previous), "fresh": len(rows) - len(previous),
            "pilot_ids": [r["id"] for r in probes(rows, previous)], "admitted": admitted(), "final_test_opened": False}))
    elif args.stage == "workflow":
        workflow()
    else:
        run(args.stage == "pilot")
