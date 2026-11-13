"""Frozen intact-model relation study: prepare on CPU, reserve/capture, analyze.

Run from the repository root through locked Pixi. This entry point never acquires
GPUs itself: capture requires the controller supervisor's live reservation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

from .jspace_lens import UPSTREAM_COMMIT, fit_token_rows, load_small_qwen
from .jspace_relation_capture import build_control_banks, capture_interventions, capture_natural
from .jspace_relation_tasks import build_dataset, validate_dataset


STUDY_ID = "JSPACE-RELATION-PROPAGATION-01"
SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "docs/07 - Status Reports/jspace/2026-11-13_jspace-relation-neural-protocol.md"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL_PATH = Path.home() / "models/hf/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots" / MODEL_REVISION
CALIBRATION_PATH = ROOT / ".pixi/jspace-calibration.json"
CALIBRATION_SHA256 = "6b5546932ef0062d87e046158b2bf0c54514d533124ba53ec2f0261b847e4dfc"
ARMS = ("jacobian", "pca", "random")


@dataclass(frozen=True)
class Configuration:
    seed: int = 7
    concepts: tuple[str, ...] = ("China", "France", "Germany", "Russia")
    layers: tuple[int, ...] = (5, 6, 11, 12, 17, 18, 23, 24, 29, 30, 34, 35)
    edges: tuple[tuple[int, int], ...] = ((5, 6), (11, 12), (17, 18), (23, 24), (29, 30), (34, 35))
    sites: tuple[int, ...] = (11, 23)
    train_count: int = 512
    development_count: int = 128
    evaluation_count: int = 128
    shakedown_count: int = 16
    intervention_cases_per_country_answer: int = 4
    dtype: str = "bfloat16"
    hidden_width: int = 2048
    model_depth: int = 36
    max_new_tokens: int = 8
    lens_max_seq_len: int = 64
    lens_skip_first: int = 16
    lens_dim_batch: int = 4
    relation_degree: int = 2
    max_candidates_per_degree: int = 8
    max_within_relations: int = 3
    max_joint_relations: int = 4
    max_normalized_rms: float = .10
    max_shuffle_ratio: float = 1 / 3
    shuffle_controls: int = 32
    symbolic_timeout_seconds: int = 5
    max_denominator: int = 1000
    max_rationalization_rms_change: float = .01


CONFIG = Configuration()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    serialized = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(serialized + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run CPU preparation/analysis with CUDA_VISIBLE_DEVICES='' before Python starts")


def source_files():
    names = [PROTOCOL, "pyproject.toml", "pixi.lock"]
    names += [f"experiments/structure_net/jspace_{name}.py" for name in (
        "lens", "tasks", "runtime", "relation_tasks", "relation_capture", "relations",
        "relation_analysis", "relation_study")]
    names += [str(path.relative_to(ROOT)) for path in sorted((ROOT / "src/structure_net_resources").glob("*.py"))]
    return names


def select_intervention_cases(cases, *, per_cell=4):
    """Freeze balanced case IDs without consulting any model output."""
    cells = defaultdict(list)
    for case in cases:
        cells[(case["intermediate"], case["answer"])].append(case)
    if len(cells) != 16 or per_cell < 1 or any(len(rows) < per_cell for rows in cells.values()):
        raise ValueError("Need all 16 country/answer cells with enough cases")
    selected = [case for cell in sorted(cells) for case in sorted(cells[cell], key=lambda c: c["case_id"])[:per_cell]]
    if len({case["case_id"] for case in selected}) != len(selected):
        raise ValueError("Duplicate intervention case ID")
    return selected


def _calibration(path, dataset):
    if file_hash(path) != CALIBRATION_SHA256:
        raise ValueError("Calibration selection changed")
    calibration = json.loads(Path(path).read_text())
    if (len(calibration["rows"]) != 16 or calibration["split"] != "train" or calibration["seed"] != 7
            or file_hash(calibration["source_path"]) != calibration["source_sha256"]):
        raise ValueError("Calibration source or selection mismatch")
    prompts = {" ".join(c["prompt"].split()) for split in dataset["splits"].values() for c in split}
    for row in calibration["rows"]:
        if (hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]
                or len(row["text"].split()) != row["word_count"] or row["word_count"] < 80
                or " ".join(row["text"].split()) in prompts):
            raise ValueError("Calibration row invalid or overlaps task")
    return calibration


def prepare(output, *, model_path=MODEL_PATH, calibration=CALIBRATION_PATH):
    _cpu_only()
    from transformers import AutoTokenizer

    output, model_path = Path(output), Path(model_path)
    if output.exists():
        raise FileExistsError(f"New artifact root required: {output}")
    dataset = build_dataset(seed=CONFIG.seed, train_count=CONFIG.train_count,
                            development_count=CONFIG.development_count,
                            evaluation_count=CONFIG.evaluation_count, shakedown_count=CONFIG.shakedown_count)
    validate_dataset(dataset)
    _calibration(calibration, dataset)
    model_config = json.loads((model_path / "config.json").read_text())
    if (model_path.name != MODEL_REVISION or model_config["hidden_size"] != CONFIG.hidden_width
            or model_config["num_hidden_layers"] != CONFIG.model_depth):
        raise ValueError("This protocol fixes the Qwen2.5-3B-Instruct revision and architecture")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    token_lists = [tokenizer.encode(" " + name, add_special_tokens=False) for name in CONFIG.concepts]
    if any(len(ids) != 1 for ids in token_lists) or len({ids[0] for ids in token_lists}) != 4:
        raise ValueError("Concept names must have distinct single space-prefixed tokens")
    lengths = {}
    for split, cases in dataset["splits"].items():
        sizes = [len(tokenizer.apply_chat_template([{"role": "user", "content": c["prompt"]}],
                                                  tokenize=True, add_generation_prompt=True)) for c in cases]
        lengths[split] = {"minimum": min(sizes), "maximum": max(sizes), "rows": len(sizes)}
    model_files = {p.name: file_hash(p) for p in sorted(model_path.iterdir()) if p.is_file()}
    producers = {name: file_hash(ROOT / name) for name in source_files()}
    selected = select_intervention_cases(dataset["splits"]["evaluation"],
                                        per_cell=CONFIG.intervention_cases_per_country_answer)
    plan = {
        "schema_version": SCHEMA_VERSION, "study_id": STUDY_ID, "conformance": "EXPLORATORY",
        "prepared_at": _utc(), "config": asdict(CONFIG), "upstream_commit": UPSTREAM_COMMIT,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "producer_identity": "content_hashed_worktree; git_head_is_base_revision",
        "producer_sha256": producers, "model": {"path": str(model_path), "revision": MODEL_REVISION,
                                                   "files": model_files},
        "dataset_digest": digest(dataset), "dataset_manifest": dataset["manifest"],
        "calibration_sha256": file_hash(calibration), "prompt_lengths": lengths,
        "concept_token_ids": [ids[0] for ids in token_lists],
        "intervention_case_ids": [c["case_id"] for c in selected],
        "final_layer_dictionary": "identity transport; normalized W_U rows; before final norm",
        "not_claimed": ["exact_neural_ideal", "compression", "full_J_space", "independent_model_replication"],
    }
    plan["scientific_fingerprint"] = digest(plan)
    output.mkdir(parents=True)
    write_json(output / "tasks.json", dataset)
    shutil.copy2(calibration, output / "calibration.json")
    for name in producers:
        target = output / "producer" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    write_json(output / "plan.json", plan)
    return {"status": "prepared", "root": str(output), "study_id": STUDY_ID,
            "fingerprint": plan["scientific_fingerprint"], "prompt_lengths": lengths,
            "intervention_rows": len(selected) * len(CONFIG.sites) * len(ARMS)}


def verify_plan(output, *, verify_model=True):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    if digest({k: v for k, v in plan.items() if k != "scientific_fingerprint"}) != plan["scientific_fingerprint"]:
        raise ValueError("Plan fingerprint mismatch")
    if plan["study_id"] != STUDY_ID or digest(plan["config"]) != digest(asdict(CONFIG)):
        raise ValueError("Study configuration mismatch")
    for name, expected in plan["producer_sha256"].items():
        if file_hash(ROOT / name) != expected or file_hash(output / "producer" / name) != expected:
            raise ValueError(f"Producer changed after freeze: {name}; use a new artifact root")
    dataset = json.loads((output / "tasks.json").read_text())
    if digest(dataset) != plan["dataset_digest"] or file_hash(output / "calibration.json") != plan["calibration_sha256"]:
        raise ValueError("Frozen input changed")
    if verify_model:
        for name, expected in plan["model"]["files"].items():
            if file_hash(Path(plan["model"]["path"]) / name) != expected:
                raise ValueError(f"Model file changed: {name}")
    return plan


def normalized(rows):
    rows = rows.detach().float().cpu()
    norms = rows.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(rows).all()) or bool((norms <= 1e-12).any()):
        raise ValueError("Nonfinite or zero dictionary row")
    return rows / norms


def _save_capture(directory, data):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    save_tensor(directory / "hidden.pt", data["hidden"])
    write_json(directory / "records.json", data["records"])


def capture(output):
    from structure_net_resources.client import require_gpu_lease

    # Mandatory order: a live lease is verified before any CUDA query or load.
    lease = require_gpu_lease([0])
    output = Path(output)
    plan = verify_plan(output)
    directory = output / "capture"
    directory.mkdir(exist_ok=False)
    write_json(directory / "lease.json", {k: lease[k] for k in (
        "id", "holder", "state", "ready", "allocation", "pid", "pid_start", "end")})
    started = time.perf_counter()
    completed = Counter()
    stage = "initialization"
    events = directory / "events.jsonl"
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Allocated CUDA device must support BF16; no silent CPU/precision fallback")
        torch.set_num_threads(4)
        torch.manual_seed(CONFIG.seed)
        torch.cuda.manual_seed_all(CONFIG.seed)
        torch.cuda.reset_peak_memory_stats()
        dataset = json.loads((output / "tasks.json").read_text())
        calibration = json.loads((output / "calibration.json").read_text())
        model, tokenizer = load_small_qwen(plan["model"]["path"], device="cuda:0", dtype=torch.bfloat16)

        def progress(row):
            completed[stage] += 1
            with events.open("a") as stream:
                stream.write(json.dumps({"stage": stage, "elapsed_seconds": time.perf_counter()-started,
                                         "row": row}, sort_keys=True, allow_nan=False) + "\n")
            if completed[stage] == 1 or completed[stage] % 32 == 0:
                require_gpu_lease([0])
                print(json.dumps({"stage": stage, "completed": completed[stage],
                                  "elapsed_seconds": round(time.perf_counter()-started, 2)}), flush=True)

        stage = "shakedown"
        smoke = capture_natural(model, tokenizer, dataset["splits"]["shakedown"],
                                layers=CONFIG.layers, progress=progress)
        _save_capture(output / "shakedown", smoke)
        restored_hidden = torch.load(output / "shakedown/hidden.pt", map_location="cpu", weights_only=True)
        restored_records = json.loads((output / "shakedown/records.json").read_text())
        if not torch.equal(restored_hidden, smoke["hidden"]) or restored_records != smoke["records"]:
            raise ValueError("Shakedown capture save/reload mismatch")
        write_json(output / "shakedown/result.json", {
            "status": "completed", "classification": "systems_lifecycle_only_not_quality_evidence",
            "scientific_fingerprint": plan["scientific_fingerprint"], "rows": len(smoke["records"]),
            "finite": True, "save_reload_equal": True, "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
        del smoke, restored_hidden, restored_records
        stage = "lens_fit"
        fit = fit_token_rows(model, tokenizer, [r["text"] for r in calibration["rows"]],
                             source_layers=CONFIG.layers[:-1], token_ids=plan["concept_token_ids"],
                             dim_batch=CONFIG.lens_dim_batch, max_seq_len=CONFIG.lens_max_seq_len,
                             skip_first=CONFIG.lens_skip_first, progress_callback=progress)
        fit.token_rows[35] = model.get_output_embeddings().weight[plan["concept_token_ids"]].detach().float().cpu()
        fit.metadata["final_layer_added"] = plan["final_layer_dictionary"]
        save_tensor(directory / "lens.pt", {"token_rows": fit.token_rows, "metadata": fit.metadata})
        restored_lens = torch.load(directory / "lens.pt", map_location="cpu", weights_only=True)
        if (restored_lens["metadata"] != fit.metadata
                or any(not torch.equal(restored_lens["token_rows"][layer], fit.token_rows[layer]) for layer in CONFIG.layers)):
            raise ValueError("Lens save/reload mismatch")
        del restored_lens
        jacobian = torch.stack([normalized(fit.token_rows[layer]) for layer in CONFIG.layers])
        requested_mib = lease["allocation"][os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]]
        if torch.cuda.max_memory_reserved() > requested_mib * 1024**2:
            raise RuntimeError("Representative model/lens memory exceeds reservation allocation")
        write_json(directory / "memory_pilot.json", {
            "classification": "systems_lifecycle_only_not_quality_evidence",
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "allocation_mib": requested_mib,
            "intended_concurrency": 1, "torch_version": str(torch.__version__),
            "device_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
        })
        banks = None
        for split in ("train", "development", "evaluation"):
            stage = split
            data = capture_natural(model, tokenizer, dataset["splits"][split], layers=CONFIG.layers, progress=progress)
            _save_capture(directory / split, data)
            if split == "train":
                controls = build_control_banks(data["hidden"], seed=CONFIG.seed)
                banks = {"jacobian": jacobian, "pca": controls["pca"], "random": controls["random"]}
                save_tensor(directory / "banks.pt", {**banks, "pca_training_mean": controls["mean"]})
            del data
        indexed = {case["case_id"]: case for case in dataset["splits"]["evaluation"]}
        stage = "interventions"
        data = capture_interventions(model, tokenizer, [indexed[i] for i in plan["intervention_case_ids"]],
                                     layers=CONFIG.layers, banks=banks, concept_names=CONFIG.concepts,
                                     sites=CONFIG.sites, progress=progress)
        _save_capture(directory / stage, data)
        del data
        require_gpu_lease([0])
        if torch.cuda.max_memory_reserved() > requested_mib * 1024**2:
            raise RuntimeError("Full capture memory exceeds reservation allocation")
        artifacts = {str(p.relative_to(output)): file_hash(p) for base in (directory, output / "shakedown")
                     for p in sorted(base.rglob("*")) if p.is_file()}
        result = {"schema_version": SCHEMA_VERSION, "study_id": STUDY_ID, "status": "completed",
                  "scientific_fingerprint": plan["scientific_fingerprint"], "started_from_plan": "plan.json",
                  "counts": dict(completed), "artifacts": artifacts, "completed_at": _utc(),
                  "elapsed_seconds": time.perf_counter()-started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "lease_id": lease["id"], "release_owner": "inference-ctl supervisor on child exit"}
        write_json(output / "capture_manifest.json", result)
        return result
    except BaseException as exc:
        write_json(directory / "failure.json", {"status": "invalid_capture", "stage": stage,
                   "error_type": type(exc).__name__, "error": str(exc), "completed": dict(completed),
                   "scientific_fingerprint": plan["scientific_fingerprint"],
                   "elapsed_seconds": time.perf_counter()-started})
        raise


def project_scores(hidden, banks):
    if hidden.ndim != 3 or not bool(torch.isfinite(hidden).all()):
        raise ValueError("Expected finite hidden [rows, layers, width]")
    result = {}
    for arm in ARMS:
        bank = banks[arm]
        if bank.shape != (hidden.shape[1], 4, hidden.shape[2]) or not bool(torch.isfinite(bank).all()):
            raise ValueError("Dictionary shape or finiteness mismatch")
        result[arm] = torch.einsum("nld,lcd->nlc", hidden.double(), bank.double()).numpy()
    return result


def verify_capture(output, plan):
    output = Path(output)
    manifest = json.loads((output / "capture_manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["scientific_fingerprint"] != plan["scientific_fingerprint"]:
        raise ValueError("Incomplete capture or fingerprint mismatch")
    expected = {"shakedown": 16, "lens_fit": 16, "train": 512, "development": 128,
                "evaluation": 128, "interventions": 384}
    if manifest["counts"] != expected:
        raise ValueError("Capture counts differ from the frozen schedule")
    required = {f"capture/{split}/{name}" for split in ("train", "development", "evaluation", "interventions")
                for name in ("hidden.pt", "records.json")}
    required |= {"capture/banks.pt", "capture/lens.pt", "capture/memory_pilot.json", "capture/lease.json",
                 "shakedown/result.json", "shakedown/hidden.pt", "shakedown/records.json"}
    if not required.issubset(manifest["artifacts"]):
        raise ValueError("Capture artifact manifest missing required files")
    for name, expected_hash in manifest["artifacts"].items():
        path = output / name
        if not path.resolve().is_relative_to(output.resolve()) or file_hash(path) != expected_hash:
            raise ValueError(f"Capture artifact mismatch: {name}")
    return manifest


def _load_split(output, split, cases=None):
    directory = Path(output) / "capture" / split
    hidden = torch.load(directory / "hidden.pt", map_location="cpu", weights_only=True)
    records = json.loads((directory / "records.json").read_text())
    if hidden.dtype != torch.float32 or hidden.shape != (len(records), len(CONFIG.layers), CONFIG.hidden_width):
        raise ValueError("Captured records and hidden shape differ")
    if any(row["hidden_row_index"] != i or row["observation_layers"] != list(CONFIG.layers)
           for i, row in enumerate(records)):
        raise ValueError("Captured records have changed row/layer order")
    if cases is not None and [row["case"] for row in records] != cases:
        raise ValueError("Captured cases differ from frozen data order")
    if any(row["case_id"] != row["case"]["case_id"] for row in records):
        raise ValueError("Top-level captured case identity differs from its frozen case")
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError("Nonfinite saved activations")
    return hidden, records


def task_metrics(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["arm"], row["source_site"])].append(row)
    return [{"arm": arm, "source_site": site, "rows": len(rows),
             "original_correct": sum(bool(r["original_correct"]) for r in rows),
             "original_accuracy": sum(bool(r["original_correct"]) for r in rows) / len(rows),
             "counterfactual_correct": sum(bool(r["counterfactual_correct"]) for r in rows),
             "counterfactual_accuracy": sum(bool(r["counterfactual_correct"]) for r in rows) / len(rows),
             "answer_status_counts": dict(Counter(r["answer_status"] for r in rows))}
            for (arm, site), rows in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1])))]


def study_summary(report):
    by_bank = {arm: values["summary"] for arm, values in report["banks"].items()}
    primary = by_bank["jacobian"]
    if primary["propagation_passing_dev_and_test"]:
        classification = "validated_approximate_propagation"
    elif primary["layer_relations_passing_test"] or primary["joint_relations_passing_test"]:
        classification = "heldout_relations_without_validated_propagation"
    else:
        classification = "no_heldout_relations_detected_in_bounded_search"
    counts = {arm: values["propagation_passing_dev_and_test"] for arm, values in by_bank.items()}
    return {"primary_classification": classification, "by_bank": by_bank,
            "jacobian_more_validated_consequences_than_both_controls":
                counts["jacobian"] > max(counts["pca"], counts["random"]),
            "comparison_interpretation": "descriptive counts; no statistical J advantage or exact neural proof"}


def analyze(output):
    _cpu_only()
    from .jspace_relation_analysis import evaluate_models, fit_models

    torch.set_num_threads(4)
    output = Path(output)
    plan = verify_plan(output, verify_model=False)
    capture_manifest = verify_capture(output, plan)
    directory = output / "analysis"
    directory.mkdir(exist_ok=False)
    started = time.perf_counter()
    dataset = json.loads((output / "tasks.json").read_text())
    banks = torch.load(output / "capture/banks.pt", map_location="cpu", weights_only=True)
    scores = {arm: {} for arm in ARMS}
    metrics = {}
    for split in ("train", "development"):
        hidden, records = _load_split(output, split, dataset["splits"][split])
        projected = project_scores(hidden, banks)
        for arm in ARMS:
            scores[arm][split] = projected[arm]
        metrics[split] = task_metrics(records)
    frozen = fit_models(scores, layers=CONFIG.layers, edges=CONFIG.edges, seed=CONFIG.seed)
    write_json(directory / "frozen_models.json", frozen)
    freeze_hash = file_hash(directory / "frozen_models.json")
    # Durable freeze is complete before any evaluation score is loaded.
    frozen = json.loads((directory / "frozen_models.json").read_text())
    hidden, records = _load_split(output, "evaluation", dataset["splits"]["evaluation"])
    evaluation = project_scores(hidden, banks)
    metrics["evaluation"] = task_metrics(records)
    baseline = {r["case_id"]: bool(r["original_correct"]) for r in records}
    case_index = {case["case_id"]: case for case in dataset["splits"]["evaluation"]}
    expanded_cases = [case_index[i] for i in plan["intervention_case_ids"] for site in CONFIG.sites for arm in ARMS]
    hidden, records = _load_split(output, "interventions", expanded_cases)
    expected = [(i, site, arm) for i in plan["intervention_case_ids"] for site in CONFIG.sites for arm in ARMS]
    if [(r["case_id"], r["source_site"], r["arm"]) for r in records] != expected:
        raise ValueError("Intervention records differ from the frozen schedule")
    intervention_scores = project_scores(hidden, banks)
    metrics["interventions"] = task_metrics(records)
    metrics["interventions_originally_correct_only"] = task_metrics([r for r in records if baseline[r["case_id"]]])
    report = evaluate_models(frozen, evaluation, intervention_scores, records)
    summary = study_summary(report)
    if file_hash(directory / "frozen_models.json") != freeze_hash:
        raise ValueError("Fitted relation models changed during evaluation")
    result = {"schema_version": SCHEMA_VERSION, "study_id": STUDY_ID, "status": "completed",
              "conformance": "EXPLORATORY", "scientific_fingerprint": plan["scientific_fingerprint"],
              "config": plan["config"], "task_metrics": metrics, "relation_results": report, "summary": summary,
              "frozen_models_sha256": freeze_hash, "capture_manifest_sha256": file_hash(output / "capture_manifest.json"),
              "elapsed_seconds": time.perf_counter()-started, "completed_at": _utc(),
              "exact_neural_claim": False, "compression_tested": False}
    write_json(directory / "result.json", result)
    campaign = {"schema_version": SCHEMA_VERSION, "study_id": STUDY_ID, "status": "completed",
                "conformance": "EXPLORATORY", "scientific_fingerprint": plan["scientific_fingerprint"],
                "config": plan["config"], "capture_counts": capture_manifest["counts"],
                "requested": 1, "completed": 1, "failed": 0, "retried": 0, "reused": 0,
                "task_metrics": metrics, "summary": summary,
                "result_path": "analysis/result.json", "result_sha256": file_hash(directory / "result.json"),
                "frozen_models_sha256": freeze_hash, "exact_neural_claim": False, "compression_tested": False,
                "limitations": ["single_checkpoint", "four_observables", "six_sampled_edges",
                                "one_task_family", "one_data_seed", "bounded_polynomial_search",
                                "exact_algebra_only_for_approximate_surrogate"]}
    write_json(output / "campaign_results.json", campaign)
    return campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "capture", "analyze"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {"prepare": prepare, "capture": capture, "analyze": analyze}[args.stage](args.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
