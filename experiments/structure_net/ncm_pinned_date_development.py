"""Date-matched source-only DEV control; same trained weights and same 128 cases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_clean_semantic_development as previous
from experiments.structure_net.ncm_memory_data import file_hash, write_json

word, development = previous.word, previous.development
old, reference, legacy_eval, legacy_audit = previous.old, previous.reference, previous.legacy_eval, previous.legacy_audit
base, q = reference.data, previous.q
ROOT = word.ROOT
HYPOTHESIS = "ncm-pinned-date-full690-semantic-development-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-pinned-date-development-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-pinned-date-development.md")
MODULE = "experiments.structure_net.ncm_pinned_date_development"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_pinned_date_development"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_pinned_date_development.py")
WORKFLOW = ROOT / "20261024_pinned_date_development_workflow_v1"
METADATA = ROOT / "20261024_pinned_date_development_metadata_v1"
FEATURE_PILOT = ROOT / "20261024_pinned_date_development_feature_pilot_v1"
FEATURES = ROOT / "20261024_pinned_date_development_features_v1"
SOURCES = ROOT / "20261024_pinned_date_development_sources_v1"
PILOT = ROOT / "20261024_pinned_date_development_inference_pilot_v1"
OUTPUTS = ROOT / "20261024_pinned_date_development_outputs_v1"
PACKETS = ROOT / "20261024_pinned_date_development_packets_v1"
JUDGE_PILOT = ROOT / "20261024_pinned_date_development_judge_pilot_v1"
GRADE = ROOT / "20261024_pinned_date_development_grade_v1"
RESULTS = ROOT / "20261024_pinned_date_development_results_v1"
OLD_CACHE = previous.GRADE / "judgment_cache.json"
CACHE_ARCHIVE = previous.RESULTS
CACHE_HYPOTHESIS = previous.HYPOTHESIS
CACHE_CAMPAIGN = "84114f770b11977fd6d64734d7562bc87c41c1f9efb61ab4fb50d10524a6992d"
CACHE_SHA = "5c094eb72ca43bb87c459770b89bcb739d91efed625eeaa679e87b69ec636961"
CHECKPOINT = "bdc462897a81e49a3dd0f16e1d48b63ebbfe078859095f274a97efb3e3f546a6"
DATE = "23 Oct 2026"
CONSTRUCTOR_SEED = previous.CONSTRUCTOR_SEED
FORBIDDEN = previous.FORBIDDEN
inference_probes, judge_probes, parity_check = previous.inference_probes, previous.judge_probes, previous.parity_check
training_parity_inputs, load_selected, predict = previous.training_parity_inputs, previous.load_selected, previous.predict
word_completed = previous.word_completed


class PinnedDateTokenizer:
    """Supply the training date to every chat-template path explicitly."""
    def __init__(self, tokenizer, date=DATE):
        self.tokenizer, self.date = tokenizer, date

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        if "date_string" in kwargs and kwargs["date_string"] != self.date:
            raise ValueError("conflicting prompt date")
        return self.tokenizer.apply_chat_template(*args, **{**kwargs, "date_string": self.date})


def require_date_only_change(old_row, new_row):
    if set(old_row) != set(new_row) or any(old_row[k] != new_row[k]
            for k in old_row if k not in ("context_ids", "native_prompt_sha256")):
        raise ValueError("date control changed a source or alignment field")
    a, b = old_row["context_ids"], new_row["context_ids"]
    changes = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    first_source = min(i for entry in new_row["alignment"] for i, _ in entry)
    if len(a) != len(b) or len(changes) != 1 or changes[0] >= first_source:
        raise ValueError("exactly one date token before the source must change")
    if q.digest(b) != new_row["native_prompt_sha256"]:
        raise ValueError("pinned prompt hash differs")
    return changes


def configure():
    previous.configure()
    # Private worker bindings only; no earlier producer source or evidence is edited.
    for name in ("HYPOTHESIS", "PLAN", "REPORT", "WORKFLOW", "PILOT", "OUTPUTS", "PACKETS",
            "JUDGE_PILOT", "GRADE", "RESULTS", "MODULE", "CONSTRUCTOR_SEED", "CHECKPOINT"):
        setattr(old, name, globals()[name])
    old.configure, old.inputs, old.producing = configure, inputs, producing
    old.load_selected, old.training_parity_inputs, old.predict = load_selected, training_parity_inputs, predict
    old.base_cache, old.prior_cache = base_cache, prior_cache
    for module in (reference, legacy_eval):
        module.PILOT, module.OUTPUTS, module.PACKETS, module.GRADE = PILOT, OUTPUTS, PACKETS, GRADE
    reference.sources, reference.producing, reference.prior_cache = sources, producing, prior_cache
    legacy_eval.prior_cache = prior_cache
    base.DATA, base.PILOT, base.FEATURES, base.producing = METADATA, FEATURE_PILOT, FEATURES, producing


def producing(root, stage, inputs=(), config=None, **unused):
    upstream = q.read(previous.WORKFLOW / "producer.json")
    files = list(dict.fromkeys([Path(__file__), PLAN, AUDITOR_FILE,
        Path("tests/structure_net/test_ncm_pinned_date_development.py"),
        Path(development.__file__), *[Path(p) for p in upstream["source_hashes"]]]))
    common = [OLD_CACHE, CACHE_ARCHIVE / "campaign_results.json",
        CACHE_ARCHIVE / (CACHE_HYPOTHESIS + "_registry_readback.json"),
        word.TRAIN / "model.pt", word.TRAIN / "source_only_reload_probe.json",
        word.RESULTS / "campaign_results.json", word.WORKFLOW / "result.json",
        previous.development.DATA / "cases.json",
        ROOT / "20261024_pinned_date_TRAIN_compatibility_v1.json",
        ROOT / "20261024_calendar_sensitivity_control_v2/head_result.json",
        ROOT / "20261024_calendar_sensitivity_control_v2/outputs.json"]
    for entry in q.read(OLD_CACHE).values():
        common.extend(Path(p) for p in entry["row"]["attempts"])
    p = base.freeze(root, stage, list(dict.fromkeys([*inputs, *common, *files])), {**(config or {}),
        "hypothesis_id": HYPOTHESIS, "schema": "nal.ncm-pinned-date-development.v1",
        "conformance": "EXPLORATORY matched input control after known quality failure",
        "date_string": DATE, "previous_feature_date": "24 Oct 2026", "changed_factor": "one prompt-date token",
        "native_system": base.SYSTEM, "parent_layers": 28, "parent_precision": "BF16",
        "aligned_features": "FP32", "parent_weights_frozen": True, "original_MLPs_frozen": True,
        "source_right_padding": True, "all_source_tokens_retained": True,
        "selected_checkpoint_sha256": CHECKPOINT, "constructor_seed": CONSTRUCTOR_SEED,
        "precision": "FP32weights_decoder_BF16forecaster_and_word_transformer", "matmul_TF32": False,
        "cuDNN_TF32": False, "oracle_target_codes": False, "AR_fallback": False, "output_repair": False,
        "source_truncated": False, "fixed_generator_passes": 9, "output_selection": "single_deterministic_argmax",
        "stored_token_cap": 128, "cohort_cases": 64, "per_judge_minimum_per_cohort": 61,
        "same_case_joint_minimum_per_cohort": 61, "valid_required_per_cohort": 64,
        "judges": q.JUDGES, "rubric": reference.profile.RUBRIC, "canonical_prior_firsts": 1925,
        "upstream_kernel_identity": None, "torch": torch.__version__, "cuda": torch.version.cuda,
        "model_training_executed": False, "same_development_sources_previously_failed": True,
        "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"pinned_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def prepare_metadata():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("metadata is CPU-only")
    parts, paths = development.admitted_sources()
    native, gpt, storage = development.tokenizers()
    native = PinnedDateTokenizer(native)
    prior_rows = q.read(development.DATA / "cases.json")
    prepared = {split: [development.prepare_row(r, split, native, gpt, storage) for r in rows]
        for split, rows in parts.items()}
    differences = {split: [require_date_only_change(a, b) for a, b in zip(prior_rows[split], rows)]
        for split, rows in prepared.items()}
    proof = base.verify_public([r for rows in prepared.values() for r in rows])
    p = producing(METADATA, "explicit_TRAIN_date_complete_DEV_metadata", [*paths, base.PUBLIC_FILE,
        base.MODEL_PATH / "tokenizer_config.json", base.MODEL_PATH / "tokenizer.json",
        base.GPT2 / "tokenizer.json"], {"CPU_only": True, "cases": 128})
    write_json(METADATA / "cases.json", prepared)
    write_json(METADATA / "selection.json", {"feature_pilot_ids": development.select_pilot(prepared)})
    write_json(METADATA / "public_proof.json", {"public_file_sha256": base.PUBLIC_SHA256, "exact_sources": proof})
    write_json(METADATA / "date_only_changes.json", differences)
    base.verify(p)
    write_json(METADATA / "result.json", {"status": "same128_DEV_explicit_TRAIN_date_prepared",
        "counts": {s: len(v) for s, v in prepared.items()}, "cases_sha256": file_hash(METADATA / "cases.json"),
        "all_sources_complete": True, "date_string": DATE, "changed_tokens_per_case": 1,
        "task_accuracy": None, "final_test_opened": False, "producer_fingerprint": p["fingerprint"]})
    print(json.dumps({"metadata_cases": 128, "date_string": DATE, "date_tokens_changed_per_case": 1}), flush=True)


def metadata_inputs():
    development.admitted_sources()
    base.terminal(METADATA)
    if file_hash(METADATA / "cases.json") != q.read(METADATA / "result.json")["cases_sha256"]:
        raise ValueError("pinned metadata changed")
    prepared = q.read(METADATA / "cases.json")
    fields = (*reference.IDENTITY, "split", "source_ids", "source_frame", "alignment_sha256",
        "native_prompt_sha256", "source_component")
    cases = {s: [{k: r[k] for k in fields} for r in rows] for s, rows in prepared.items()}
    if tuple(cases) != reference.COHORTS or {s: len(v) for s, v in cases.items()} != development.EXPECTED:
        raise ValueError("same complete64+64 cases required")
    previous_rows = q.read(development.DATA / "cases.json")
    for split, rows in prepared.items():
        for a, b in zip(previous_rows[split], rows):
            require_date_only_change(a, b)
    inventory = q.read(development.partitions.DEST / "public_inventory.json")
    previous.source_guard(q.read(word.targets.CASES)["train"], [r for rows in cases.values() for r in rows],
        {k: v["tokens"] for k, v in inventory["turns"].items()})
    return cases


def prepare_sources():
    cases = metadata_inputs()
    p = producing(SOURCES, "same128_source_only_exclusion_replay", [METADATA / "cases.json", base.PUBLIC_FILE])
    write_json(SOURCES / "cases.json", cases)
    proof = reference.verify_public([r for rows in cases.values() for r in rows])
    write_json(SOURCES / "public_proof.json", {"exact_sources": proof, "public_file_sha256": reference.PUBLIC_SHA256})
    write_json(SOURCES / "split_audit.json", {"TRAIN_cases": 690, "development_cases": 128,
        "source_components": 128, "TRAIN_development_conflicts": 0, "within_development_conflicts": 0,
        "informative_full_turn_minimum_cl100k_tokens": 192, "full_text_and_family_guards": True,
        "sealed_final_content_opened": False})
    base.verify(p)
    write_json(SOURCES / "result.json", {"status": "same128_source_only_DEV_prepared",
        "cases_sha256": file_hash(SOURCES / "cases.json"), "task_accuracy": None})


def sources():
    base.terminal(SOURCES)
    cases = q.read(SOURCES / "cases.json")
    if file_hash(SOURCES / "cases.json") != q.read(SOURCES / "result.json")["cases_sha256"]:
        raise ValueError("source metadata changed")
    if tuple(cases) != reference.COHORTS or {s: len(v) for s, v in cases.items()} != development.EXPECTED:
        raise ValueError("all128 source-only cases required")
    if any(FORBIDDEN & r.keys() for rows in cases.values() for r in rows):
        raise ValueError("source-only rows required")
    return cases


def inputs():
    configure()
    if not word_completed() or file_hash(word.TRAIN / "model.pt") != CHECKPOINT:
        raise ValueError("unchanged completed final32768 candidate required")
    base.terminal(FEATURES)
    result = q.read(FEATURES / "result.json")
    path = FEATURES / "manifest.json"
    admission = q.read(WORKFLOW / "admission.json")
    if (file_hash(path) != result["manifest_sha256"] or file_hash(path) != admission["feature_manifest_sha256"]
            or result["parent_body_hash_unchanged"] != base.EXPECTED_BODY or result["cases"] != 128
            or not result["all_source_tokens_retained"] or admission["selected_checkpoint_sha256"] != CHECKPOINT):
        raise ValueError("unchanged complete pinned-date original-body features required")
    cases, manifest = sources(), q.read(path)
    if set(manifest) != {r["id"] for rows in cases.values() for r in rows}:
        raise ValueError("feature/source identities differ")
    return cases, manifest


def base_cache():
    campaign_path = CACHE_ARCHIVE / "campaign_results.json"
    campaign = q.read(campaign_path)
    receipt = q.read(CACHE_ARCHIVE / (CACHE_HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(campaign_path) != CACHE_CAMPAIGN or file_hash(OLD_CACHE) != CACHE_SHA
            or campaign["artifacts"].get(str(OLD_CACHE)) != CACHE_SHA or campaign["canonical_first_judgments"] != 1925
            or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != CACHE_CAMPAIGN
            or receipt["exact"]["campaign_sha256"] != CACHE_CAMPAIGN):
        raise ValueError("all1925 canonical firsts required")
    cache = q.read(OLD_CACHE)
    if len(cache) != 1925:
        raise ValueError("complete1925 cache required")
    old.audit_cache(cache)
    return cache


def prior_cache():
    baseline = base_cache()
    if not (JUDGE_PILOT / "worker_terminal_receipt.json").exists():
        return baseline
    base.terminal(JUDGE_PILOT)
    cache = q.read(JUDGE_PILOT / "judgment_cache.json")
    if any(cache.get(k) != v for k, v in baseline.items()):
        raise ValueError("a canonical first judgment changed")
    old.audit_cache(cache)
    return cache


def judge_pilot():
    old.judge_pilot()
    path = JUDGE_PILOT / "result.json"
    result = q.read(path)
    # Preserve the legacy field while adding its correctly generalised equivalent.
    result["all_prior_first_judgments_preserved"] = result["all1669_prior_first_judgments_preserved"]
    write_json(path, result)


def workflow():
    if not word_completed():
        raise ValueError("completed immutable word workflow required")
    base_cache()
    old.require_judges()
    destinations = (WORKFLOW, METADATA, FEATURE_PILOT, FEATURES, SOURCES, PILOT, OUTPUTS,
        PACKETS, JUDGE_PILOT, GRADE, RESULTS, REPORT)
    if any(p.exists() for p in destinations):
        raise ValueError("fresh matched-control destinations required")
    p = producing(WORKFLOW, "matched_date_control_workflow", [], {"GPU_concurrency": 1})
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"]}

    def stage(name, root, gpu=False):
        base.verify(p)
        command = [sys.executable, "-u", "-m", AUDITOR if name in ("finalize", "register") else MODULE, name]
        child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0" if gpu else ""})
        journal.update(status="running", active_stage=name, child_pid=child.pid)
        write_json(WORKFLOW / "progress.json", journal)
        code = child.wait()
        receipt = {"stage": name, "exit_code": code, "child_pid": child.pid,
            "observer_pid": os.getpid(), "observer": "subprocess.wait", "command": command}
        write_json(WORKFLOW / "stage_receipts" / (name + ".json"), receipt)
        if root.exists():
            write_json(root / ("registry_terminal_receipt.json" if name == "register" else "worker_terminal_receipt.json"), receipt)
        journal["stages"].append(receipt)
        if code:
            raise RuntimeError("pinned-date stage failed: " + name)

    try:
        for name, root, gpu in (("metadata", METADATA, False), ("feature-pilot", FEATURE_PILOT, True),
                ("features", FEATURES, True), ("sources", SOURCES, False)):
            stage(name, root, gpu)
        write_json(WORKFLOW / "admission.json", {"selected_checkpoint_sha256": CHECKPOINT,
            "word_campaign_sha256": file_hash(word.RESULTS / "campaign_results.json"),
            "feature_manifest_sha256": file_hash(FEATURES / "manifest.json"),
            "same_word_kernel_terminal": True, "upstream_kernel_identity": None,
            "word_TRAIN_fit_gate": q.read(word.RESULTS / "campaign_results.json")["actual_history_fit_gate"],
            "date_string": DATE, "no_model_updates": True})
        for name, root, gpu in (("pilot", PILOT, True), ("generate", OUTPUTS, True), ("packets", PACKETS, False),
                ("judge-pilot", JUDGE_PILOT, False), ("grade", GRADE, False),
                ("finalize", RESULTS, False), ("register", RESULTS, False)):
            stage(name, root, gpu)
        base.verify(p)
        journal.update(status="pinned_date_development_completed", active_stage=None, child_pid=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "metadata", "feature-pilot", "features", "sources", "pilot",
        "generate", "packets", "judge-pilot", "grade", "workflow"))
    args = parser.parse_args()
    configure()
    if args.stage == "check":
        print(json.dumps({"date_string": DATE, "checkpoint": CHECKPOINT, "first_judgments": len(base_cache())}))
    elif args.stage == "metadata":
        prepare_metadata()
    elif args.stage in ("feature-pilot", "features"):
        base.feature_cache(args.stage == "feature-pilot")
    elif args.stage == "sources":
        prepare_sources()
    elif args.stage in ("pilot", "generate"):
        old.generate(args.stage == "pilot")
    elif args.stage == "packets":
        reference.prepare_packets()
    elif args.stage == "judge-pilot":
        judge_pilot()
    elif args.stage == "grade":
        reference.grade()
    else:
        workflow()
