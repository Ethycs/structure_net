"""Fresh full-summary semantic evaluation after the all690 word continuation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.structure_net import ncm_dense_semantic_development as old
from experiments.structure_net import ncm_full_population_dense_word as word
from experiments.structure_net import ncm_clean_development_features as development
from experiments.structure_net.ncm_expanded_word_data_workflow import kernel_identity
from experiments.structure_net.ncm_memory_data import file_hash, write_json

reference, legacy_eval, legacy_audit = old.reference, old.legacy_eval, old.legacy_audit
ROOT = word.ROOT
HYPOTHESIS = "ncm-clean-full690-semantic-development-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-clean-semantic-development-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-clean-semantic-development.md")
WORKFLOW = ROOT / "20261024_clean_semantic_dev_workflow_v1"
SOURCES = ROOT / "20261024_clean_semantic_dev_sources_v1"
PILOT = ROOT / "20261024_clean_semantic_dev_inference_pilot_v1"
OUTPUTS = ROOT / "20261024_clean_semantic_dev_outputs_v1"
PACKETS = ROOT / "20261024_clean_semantic_dev_packets_v1"
JUDGE_PILOT = ROOT / "20261024_clean_semantic_dev_judge_pilot_v1"
GRADE = ROOT / "20261024_clean_semantic_dev_grade_v1"
RESULTS = ROOT / "20261024_clean_semantic_dev_results_v1"
MODULE = "experiments.structure_net.ncm_clean_semantic_development"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_clean_semantic_development"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_clean_semantic_development.py")
OLD_CACHE = ROOT / "20261023_dense_semantic_dev_grade_v1/judgment_cache.json"
CACHE_ARCHIVE = old.RESULTS
CACHE_CAMPAIGN = "38604600a716a0e94b14d1caaa9bc1e1ad9209b51ab78b18d24f2b33fb53d0b2"
CACHE_SHA = "77f3cda9d75c808780168b4b59d96af7182ea56bb9899fef812b1c577165748b"
CACHE_HYPOTHESIS = old.HYPOTHESIS
CONSTRUCTOR_SEED = 57142
CHECKPOINT = None
FORBIDDEN = development.FORBIDDEN | {"teacher_latent", "text"}
_prediction = old.predict
inference_probes, judge_probes, parity_check = old.inference_probes, old.judge_probes, old.parity_check
q = reference.q


def configure():
    global CHECKPOINT
    word.configure()
    reference.profile.install()
    for name in ("HYPOTHESIS", "PLAN", "REPORT", "WORKFLOW", "PILOT", "OUTPUTS", "PACKETS", "JUDGE_PILOT", "GRADE", "RESULTS", "MODULE", "CONSTRUCTOR_SEED"):
        setattr(old, name, globals()[name])
    if (WORKFLOW / "admission.json").exists():
        CHECKPOINT = q.read(WORKFLOW / "admission.json")["selected_checkpoint_sha256"]
    old.CHECKPOINT = CHECKPOINT
    old.training = word
    old.configure, old.inputs, old.producing = configure, inputs, producing
    old.load_selected, old.training_parity_inputs, old.predict = load_selected, training_parity_inputs, predict
    old.base_cache, old.prior_cache = base_cache, prior_cache
    for module in (reference, legacy_eval):
        module.PILOT, module.OUTPUTS, module.PACKETS, module.GRADE = PILOT, OUTPUTS, PACKETS, GRADE
    reference.sources, reference.producing, reference.prior_cache = sources, producing, prior_cache
    legacy_eval.prior_cache = prior_cache


def source_guard(training, heldout, counts):
    fragments = development.partitions.fragments
    guard = development.partitions.core.guard
    train = [fragments.comparison_view(r, counts) for r in training]
    dev = [fragments.comparison_view(r, counts) for r in heldout]
    guard.require_disjoint(train, dev)
    signatures = [guard.signature(r) for r in dev]
    for i, row in enumerate(dev):
        if any(guard.overlap(row, other, signatures[i], signatures[j]) is not None for j, other in enumerate(dev[:i])):
            raise ValueError("development source family/text/turn overlap")


def metadata_inputs():
    development.admitted_sources()
    reference.data.terminal(development.DATA)
    result = q.read(development.DATA / "result.json")
    if file_hash(development.DATA / "cases.json") != result["cases_sha256"]:
        raise ValueError("complete fresh development metadata changed")
    prepared = q.read(development.DATA / "cases.json")
    if tuple(prepared) != reference.COHORTS or {s: len(v) for s, v in prepared.items()} != development.EXPECTED:
        raise ValueError("fixed ordered64short+64long fresh development cases required")
    fields = (*reference.IDENTITY, "split", "source_ids", "source_frame", "alignment_sha256", "native_prompt_sha256", "source_component")
    cases = {}
    for split, rows in prepared.items():
        if any(r["split"] != split or FORBIDDEN & r.keys() for r in rows):
            raise ValueError("source-only evaluation metadata required")
        cases[split] = [{k: r[k] for k in fields} for r in rows]
    all_rows = [r for rows in cases.values() for r in rows]
    if len({r["source_component"] for r in all_rows}) != 128 or any(r["group_id"] != r["source_component"] for r in all_rows):
        raise ValueError("all128 distinct reserved source components required")
    if file_hash(word.targets.CASES) != word.targets.CASES_SHA:
        raise ValueError("all690 TRAIN exclusion sources must remain unchanged")
    training = q.read(word.targets.CASES)["train"]
    if len(training) != 690:
        raise ValueError("complete TRAIN exclusion population required")
    inventory_path = development.partitions.DEST / "public_inventory.json"
    reservation = q.read(development.partitions.RESULTS / "campaign_results.json")
    if reservation["artifacts"].get(str(inventory_path)) != file_hash(inventory_path):
        raise ValueError("registered public token-count inventory changed")
    inventory = q.read(inventory_path)
    source_guard(training, all_rows, {k: v["tokens"] for k, v in inventory["turns"].items()})
    return cases


def prepare_sources():
    cases = metadata_inputs()
    proof = reference.verify_public([r for rows in cases.values() for r in rows])
    p = producing(SOURCES, "clean_source_only_DEV_metadata_and_actual_TRAIN_exclusion", [reference.PUBLIC_FILE],
        {"CPU_only": True, "cases": 128}, pending=True)
    write_json(SOURCES / "cases.json", cases)
    write_json(SOURCES / "public_proof.json", {"exact_sources": proof, "public_file_sha256": reference.PUBLIC_SHA256})
    write_json(SOURCES / "split_audit.json", {"TRAIN_cases": 690, "development_cases": 128,
        "source_components": 128, "TRAIN_development_conflicts": 0, "within_development_conflicts": 0,
        "informative_full_turn_minimum_cl100k_tokens": 192, "full_text_and_family_guards": True,
        "sealed_final_content_opened": False})
    reference.data.verify(p)
    write_json(SOURCES / "result.json", {"status": "fresh_source_only_semantic_DEV_prepared",
        "counts": {s: len(v) for s, v in cases.items()}, "cases_sha256": file_hash(SOURCES / "cases.json"),
        "source_disjoint": True, "TRAIN_labels_in_evaluation_file": False, "task_accuracy": None,
        "final_test_opened": False, "producer_fingerprint": p["fingerprint"]})
    print(json.dumps({"status": "fresh_source_only_semantic_DEV_prepared", "cases": 128, "source_conflicts": 0}), flush=True)


def sources():
    reference.data.terminal(SOURCES)
    result = q.read(SOURCES / "result.json")
    cases = q.read(SOURCES / "cases.json")
    if (file_hash(SOURCES / "cases.json") != result["cases_sha256"] or tuple(cases) != reference.COHORTS
            or {s: len(v) for s, v in cases.items()} != development.EXPECTED
            or any(r["split"] != s or FORBIDDEN & r.keys() for s, rows in cases.items() for r in rows)):
        raise ValueError("unchanged complete source-only semantic development cohorts required")
    return cases


def word_completed():
    return word.completed(word.WORKFLOW, word.RESULTS, word.HYPOTHESIS,
        "full_population_dense_word_completed", ("pilot", "train", "fit", "finalize", "register"))


def inputs():
    configure()
    if not word_completed():
        raise ValueError("complete registered final word candidate required before inference")
    selected = q.read(WORKFLOW / "admission.json")
    campaign = q.read(word.RESULTS / "campaign_results.json")
    trained = q.read(word.TRAIN / "result.json")
    if (file_hash(word.TRAIN / "model.pt") != CHECKPOINT or trained["checkpoint_sha256"] != CHECKPOINT
            or campaign["training_cases"] != 690 or trained["updates"] != word.Config().updates
            or selected["word_campaign_sha256"] != file_hash(word.RESULTS / "campaign_results.json")):
        raise ValueError("only the fixed final32768 all690 word candidate is selected")
    for name in ("model.pt", "result.json", "source_only_reload_probe.json", "producer.json", "worker_terminal_receipt.json"):
        path = word.TRAIN / name
        if campaign["artifacts"].get(str(path)) != file_hash(path):
            raise ValueError("registered fixed word candidate or reload evidence changed")
    cases = sources()
    reference.data.terminal(development.FEATURES)
    features = q.read(development.FEATURES / "result.json")
    path = development.FEATURES / "manifest.json"
    if (features["manifest_sha256"] != file_hash(path) or selected["feature_manifest_sha256"] != file_hash(path)
            or features["parent_body_hash_unchanged"] != reference.data.EXPECTED_BODY
            or features["cases"] != 128 or not features["all_source_tokens_retained"]):
        raise ValueError("complete original-body fresh DEV features required")
    if not word.completed(development.WORKFLOW, development.RESULTS, development.HYPOTHESIS,
            "clean_development_features_completed", ("prepare", "feature-pilot", "features", "finalize", "register")):
        raise ValueError("complete registered fresh DEV feature workflow required")
    feature_campaign = q.read(development.RESULTS / "campaign_results.json")
    for name in ("manifest.json", "result.json", "producer.json", "worker_terminal_receipt.json"):
        artifact = development.FEATURES / name
        if feature_campaign["artifacts"].get(str(artifact)) != file_hash(artifact):
            raise ValueError("registered original-body DEV feature metadata changed")
    manifest = q.read(path)
    if set(manifest) != {r["id"] for rows in cases.values() for r in rows}:
        raise ValueError("exact128 fresh development feature identities required")
    return cases, manifest


def base_cache():
    path = CACHE_ARCHIVE / "campaign_results.json"
    campaign = q.read(path)
    receipt = q.read(CACHE_ARCHIVE / (CACHE_HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(path) != CACHE_CAMPAIGN or file_hash(OLD_CACHE) != CACHE_SHA
            or campaign["artifacts"].get(str(OLD_CACHE)) != CACHE_SHA or campaign["canonical_first_judgments"] != 1797
            or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != CACHE_CAMPAIGN or receipt["exact"]["campaign_sha256"] != CACHE_CAMPAIGN):
        raise ValueError("all1797 registered canonical first judgments required")
    cache = q.read(OLD_CACHE)
    if len(cache) != 1797:
        raise ValueError("complete1797 baseline cache required")
    old.audit_cache(cache)
    return cache


def prior_cache():
    baseline = base_cache()
    if not (JUDGE_PILOT / "worker_terminal_receipt.json").exists():
        return baseline
    reference.data.terminal(JUDGE_PILOT)
    cache = q.read(JUDGE_PILOT / "judgment_cache.json")
    if any(cache.get(k) != v for k, v in baseline.items()):
        raise ValueError("a prior first judgment changed")
    old.audit_cache(cache)
    return cache


def producing(root, stage, inputs=(), config=None, *, pending=False, identity=None):
    upstream = q.read(word.WORKFLOW / "producer.json")
    legacy_source = q.read(ROOT / "20261023_dense_semantic_dev_workflow_v1/producer.json")
    files = list(dict.fromkeys([Path(__file__), PLAN, AUDITOR_FILE, Path(old.__file__), Path(word.__file__), word.PLAN,
        Path(development.__file__), development.PLAN, Path(reference.__file__), Path(legacy_eval.__file__), Path(legacy_audit.__file__),
        Path(reference.profile.__file__), reference.profile.PLAN, Path(reference.first.__file__), reference.first.AMENDMENT,
        Path(kernel_identity.__code__.co_filename), Path(development.partitions.fragments.__file__),
        *[Path(p) for p in upstream["source_hashes"]], *[Path(p) for p in legacy_source["source_hashes"]]]))
    common = [OLD_CACHE, CACHE_ARCHIVE / "campaign_results.json", CACHE_ARCHIVE / (CACHE_HYPOTHESIS + "_registry_readback.json"),
        word.WORKFLOW / "producer.json", development.WORKFLOW / "producer.json", word.targets.CASES,
        development.partitions.DEST / "public_inventory.json", development.partitions.RESULTS / "campaign_results.json"]
    common += [development.DATA / n for n in ("producer.json", "result.json", "cases.json", "worker_terminal_receipt.json")]
    for entry in q.read(OLD_CACHE).values():
        common += [Path(p) for p in entry["row"]["attempts"]]
    for role in ("secondary", "primary"):
        path = ROOT / ("20261023_budget128_judge_" + role + "_v1")
        common += [path / n for n in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    if not pending:
        common += [WORKFLOW / "admission.json", word.WORKFLOW / "result.json", word.RESULTS / "campaign_results.json",
            word.RESULTS / (word.HYPOTHESIS + "_registry_readback.json")]
        common += [word.TRAIN / n for n in ("model.pt", "result.json", "producer.json", "source_only_reload_probe.json", "worker_terminal_receipt.json")]
        common += [SOURCES / n for n in ("producer.json", "result.json", "cases.json", "split_audit.json", "worker_terminal_receipt.json")]
        common += [development.FEATURES / n for n in ("producer.json", "result.json", "manifest.json", "worker_terminal_receipt.json")]
        common += [development.WORKFLOW / "result.json", development.RESULTS / "campaign_results.json",
            development.RESULTS / (development.HYPOTHESIS + "_registry_readback.json")]
    if stage == "source_disjoint_word_final_NCM_development_quality":
        common += [JUDGE_PILOT / n for n in ("producer.json", "result.json", "judgment_cache.json", "worker_terminal_receipt.json")]
    p = reference.data.freeze(root, stage, [*inputs, *common, *files], {**(config or {}),
        "hypothesis_id": HYPOTHESIS, "schema": "nal.ncm-clean-full690-semantic-development.v1",
        "selected_word_update": word.Config().updates, "selected_checkpoint_sha256": None if pending else CHECKPOINT,
        "constructor_seed": CONSTRUCTOR_SEED, "parent_features_cached": True, "parent_body_loaded": False,
        "precision": "FP32weights_decoder_BF16forecaster_and_word_transformer", "matmul_TF32": False, "cuDNN_TF32": False,
        "source_only": True, "oracle_target_codes": False, "AR_fallback": False, "output_repair": False,
        "source_truncated": False, "fixed_generator_passes": 9, "output_selection": "single_deterministic_argmax",
        "stored_token_cap": 128, "cohort_cases": 64, "per_judge_minimum_per_cohort": 61,
        "same_case_joint_minimum_per_cohort": 61, "valid_required_per_cohort": 64,
        "judges": q.JUDGES, "rubric": reference.profile.RUBRIC, "canonical_prior_firsts": 1797,
        "pending_word_candidate": pending, "upstream_kernel_identity": identity if pending else q.read(WORKFLOW / "producer.json")["config"]["upstream_kernel_identity"],
        "torch": torch.__version__, "cuda": torch.version.cuda, "model_training_executed": False,
        "fresh_source_components": True, "literal_TRAIN_gate_is_not_semantic_admission": True,
        "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"clean_semantic_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def load_selected():
    fixed, branch, codec = word.model(word.TRAIN / "model.pt", word.Config().updates)
    saved = torch.load(word.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    if (word.clean.recovery.study.parameter_hash(branch, frozen_only=True) != word.loader.DENSE_FROZEN_PARAMETERS
            or any(not torch.equal(value.cpu(), saved["model"][name]) for name, value in branch.named_parameters())):
        raise ValueError("all selected word and frozen decoder parameters must match")
    fixed.cuda().eval().requires_grad_(False)
    branch.eval().requires_grad_(False)
    return fixed, branch, codec


def predict(fixed, branch, row, manifest, tokenizer):
    if FORBIDDEN & row.keys():
        raise ValueError("source-only prediction cannot receive labels")
    return _prediction(fixed, branch, row, manifest, tokenizer)


def training_parity_inputs():
    rows, manifest, _ = word.forecaster.data_inputs()
    by_id = {r["id"]: r for r in rows}
    expected = q.read(word.TRAIN / "source_only_reload_probe.json")
    fields = (*reference.IDENTITY, "split", "source_ids", "source_frame", "alignment_sha256", "native_prompt_sha256")
    selected = [{k: by_id[r["id"]][k] for k in fields} for r in expected]
    values = {r["id"]: {"raw_word_ids": r["raw_word_ids"][0], "latent_sha256": r["latent_sha256"],
        "actual_coarse_codes": r["coarse_codes"][0]} for r in expected}
    if len(selected) != 2 or len(values) != 2:
        raise ValueError("both actual full-population TRAIN reload probes required")
    return selected, manifest, values


def judge_pilot():
    inputs()
    old.require_judges()
    reference.data.terminal(PACKETS)
    reference.data.terminal(OUTPUTS)
    rows = judge_probes(q.read(OUTPUTS / "outputs.json"))
    packets = [q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["text"]}]) for r in rows]
    cache = base_cache()
    previous = dict(cache)
    p = producing(JUDGE_PILOT, "systems_semantic_first_decision_path_only", [PACKETS / "exact_packets.json", OUTPUTS / "outputs.json"],
        {"cases": 4, "role": "secondary", "valid_required": 4, "quality_gate": False})
    reference.GRADE = JUDGE_PILOT
    try:
        summary = reference.grade_cohort("pilot", "secondary", rows, packets, cache)
    finally:
        reference.GRADE = GRADE
    if any(cache.get(k) != v for k, v in previous.items()):
        raise ValueError("pilot changed a canonical first judgment")
    write_json(JUDGE_PILOT / "judgment_cache.json", cache)
    reference.data.verify(p)
    if not old.judge_pilot_valid(summary):
        raise ValueError("semantic pilot schema/transport failed; all judgments preserved")
    write_json(JUDGE_PILOT / "result.json", {"status": "systems_semantic_judge_path_passed",
        "cases": 4, "valid": 4, "passed": summary["passed"], "quality_gate": False,
        "all_prior_first_judgments_preserved": True, "unique_first_judgments": len(cache),
        "producer_fingerprint": p["fingerprint"], "task_accuracy": None})


def workflow():
    metadata_inputs()
    base_cache()
    old.require_judges()
    if any(p.exists() for p in (WORKFLOW, SOURCES, PILOT, OUTPUTS, PACKETS, JUDGE_PILOT, GRADE, RESULTS, REPORT,
            Path("data/hypothesis_registry") / (HYPOTHESIS + ".json"))):
        raise ValueError("fresh clean semantic workflow and all destinations required")
    ready = word_completed()
    pid = q.read(word.WORKFLOW / "progress.json")["observer_pid"]
    identity = kernel_identity(pid)
    if (not ready and identity is None) or (identity is not None and word.MODULE.encode() not in Path(f"/proc/{pid}/cmdline").read_bytes()):
        raise ValueError("existing word supervisor must be authoritatively live")
    p = producing(WORKFLOW, "prepare_clean_sources_then_observe_fixed_word_candidate", pending=True, identity=identity)
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [],
        "upstream_kernel_identity": identity, "producer_fingerprint": p["fingerprint"]}

    def stage(name, dest):
        reference.data.verify(p)
        command = [sys.executable, "-u", "-m", AUDITOR if name in ("finalize", "register") else MODULE, name]
        child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0" if name in ("pilot", "generate") else ""})
        journal.update(status="running", active_stage=name, child_pid=child.pid)
        write_json(WORKFLOW / "progress.json", journal)
        code = child.wait()
        receipt = {"stage": name, "exit_code": code, "child_pid": child.pid,
            "observer_pid": os.getpid(), "observer": "subprocess.wait", "command": command}
        write_json(WORKFLOW / "stage_receipts" / (name + ".json"), receipt)
        if dest.exists():
            write_json(dest / ("registry_terminal_receipt.json" if name == "register" else "worker_terminal_receipt.json"), receipt)
        journal["stages"].append(receipt)
        if code:
            raise RuntimeError("clean semantic stage failed: " + name)

    try:
        stage("sources", SOURCES)
        while True:
            ready = word_completed()
            alive = identity is not None and kernel_identity(identity["pid"]) == identity
            if ready and not alive:
                break
            if not ready and not alive:
                raise RuntimeError("word supervisor terminal without complete registered lifecycle")
            journal.update(status="waiting_for_registered_full690_word_candidate", active_stage=None, child_pid=None)
            write_json(WORKFLOW / "progress.json", journal)
            time.sleep(5)
        trained = q.read(word.TRAIN / "result.json")
        write_json(WORKFLOW / "admission.json", {"selected_checkpoint_sha256": trained["checkpoint_sha256"],
            "word_campaign_sha256": file_hash(word.RESULTS / "campaign_results.json"),
            "feature_manifest_sha256": file_hash(development.FEATURES / "manifest.json"),
            "same_word_kernel_terminal": True, "upstream_kernel_identity": identity,
            "word_TRAIN_fit_gate": q.read(word.RESULTS / "campaign_results.json")["actual_history_fit_gate"]})
        configure()
        inputs()
        for name, dest in (("pilot", PILOT), ("generate", OUTPUTS), ("packets", PACKETS),
                ("judge-pilot", JUDGE_PILOT), ("grade", GRADE), ("finalize", RESULTS), ("register", RESULTS)):
            stage(name, dest)
        reference.data.verify(p)
        journal.update(status="clean_semantic_development_completed", active_stage=None, child_pid=None,
            classification=q.read(RESULTS / "campaign_results.json")["classification"])
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "sources", "pilot", "generate", "packets", "judge-pilot", "grade", "workflow"))
    args = parser.parse_args()
    configure()
    if args.stage == "check":
        cases = metadata_inputs()
        old.require_judges()
        print(json.dumps({"counts": {s: len(v) for s, v in cases.items()}, "canonical_first_judgments": len(base_cache()),
            "future_word_update": word.Config().updates, "task_accuracy": None}))
    elif args.stage == "sources":
        prepare_sources()
    elif args.stage == "workflow":
        workflow()
    elif args.stage in ("pilot", "generate"):
        old.generate(args.stage == "pilot")
    elif args.stage == "packets":
        reference.prepare_packets()
    elif args.stage == "judge-pilot":
        judge_pilot()
    else:
        reference.grade()
