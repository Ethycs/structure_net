"""Source-disjoint semantic development evaluation of the fixed dense student."""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from experiments.structure_net import ncm_dense_word_fit as training
from experiments.structure_net import ncm_dense_checkpoint_loader as loader
from experiments.structure_net import ncm_word_final_eval as reference
from experiments.structure_net import ncm_generated_history_eval as legacy_eval
from experiments.neural_architecture_lab import finalize_ncm_generated_history_word as legacy_audit
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_device_geometry import set_source_frame

ROOT = training.ROOT
HYPOTHESIS = "ncm-dense-source-semantic-development-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-dense-semantic-development-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-23_ncm-dense-semantic-development.md")
WORKFLOW = ROOT / "20261023_dense_semantic_dev_workflow_v1"
PILOT = ROOT / "20261023_dense_semantic_dev_inference_pilot_v1"
OUTPUTS = ROOT / "20261023_dense_semantic_dev_outputs_v1"
PACKETS = ROOT / "20261023_dense_semantic_dev_packets_v1"
JUDGE_PILOT = ROOT / "20261023_dense_semantic_dev_judge_pilot_v1"
GRADE = ROOT / "20261023_dense_semantic_dev_grade_v1"
RESULTS = ROOT / "20261023_dense_semantic_dev_results_v1"
MODULE = "experiments.structure_net.ncm_dense_semantic_development"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_dense_semantic_development"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_dense_semantic_development.py")
CHECKPOINT = "0b8afde259c1165ddc4ebb14110568b0695228850dd09eaeadf645095550e266"
PARENT_CAMPAIGN = "1561dd9359a17438993d3b6700f69b3619771525ec465815a896bd1b0eb77d67"
SOURCE_SHA = "07786d65a4d29201ba7be302d8c4fcc9af56a5f1bb48beec55a9cddcd756d3e1"
MANIFEST_SHA = "4d1f62db7d7e5ed12f288059fc3b74c3f4ca44067a094d7cc208301e0284b6af"
OLD_CACHE = ROOT / "20261023_clean_word_train_semantic_grade_v1/judgment_cache.json"
CACHE_ARCHIVE = ROOT / "20261023_clean_word_train_semantic_results_v1"
CACHE_CAMPAIGN = "c88711fd3af4d65c887c8aac4aeb092c82adb117f6dcbf374b2cd42a47cfbcec"
CONSTRUCTOR_SEED = 48142


def configure():
    training.clean.dependencies()
    training.clean.recovery.configure()
    reference.profile.install()
    for module in (reference, legacy_eval):
        module.PILOT, module.OUTPUTS, module.PACKETS, module.GRADE = PILOT, OUTPUTS, PACKETS, GRADE
    reference.producing = producing
    reference.prior_cache = prior_cache
    legacy_eval.prior_cache = prior_cache


def inputs():
    configure()
    q = reference.q
    archive = training.RESULTS / "campaign_results.json"
    receipt = q.read(training.RESULTS / (training.HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(archive) != PARENT_CAMPAIGN or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != PARENT_CAMPAIGN
            or receipt["exact"]["campaign_sha256"] != PARENT_CAMPAIGN
            or file_hash(training.TRAIN / "model.pt") != CHECKPOINT):
        raise ValueError("fixed audited final2048 student required")
    cases = reference.sources()
    if file_hash(reference.SOURCES / "cases.json") != SOURCE_SHA:
        raise ValueError("unchanged ordered64short+64long sources required")
    manifest_path = reference.data.FEATURES / "manifest.json"
    features = q.read(reference.data.FEATURES / "result.json")
    if (file_hash(manifest_path) != MANIFEST_SHA or features["manifest_sha256"] != MANIFEST_SHA
            or features["parent_body_hash_unchanged"] != reference.data.EXPECTED_BODY
            or not features["all_source_tokens_retained"]):
        raise ValueError("unchanged original-body complete source features required")
    study = training.clean.recovery.study
    train_path = study.expanded.DEST / "cases.json"
    upstream = q.read(training.teacher.prior.continuous.WORKFLOW / "producer.json")
    if upstream["input_hashes"].get(str(train_path)) != file_hash(train_path):
        raise ValueError("complete690 TRAIN exclusion population changed")
    rows = q.read(train_path)["train"]
    if len(rows) != 690:
        raise ValueError("all690 previous TRAIN sources must remain excluded")
    for cohort in cases.values():
        reference.data.prior.check_disjoint(rows, cohort)
    reference.data.prior.check_disjoint(*cases.values())
    return cases, q.read(manifest_path)


def base_cache():
    q = reference.q
    archive = CACHE_ARCHIVE / "campaign_results.json"
    campaign = q.read(archive)
    receipt = q.read(CACHE_ARCHIVE / "ncm-clean-word-TRAIN-semantic-v1_registry_readback.json")
    if (file_hash(archive) != CACHE_CAMPAIGN or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != CACHE_CAMPAIGN
            or campaign["artifacts"].get(str(OLD_CACHE)) != file_hash(OLD_CACHE)):
        raise ValueError("all1669 registered canonical first judgments required")
    cache = q.read(OLD_CACHE)
    if len(cache) != 1669:
        raise ValueError("full latest canonical first cache required")
    audit_cache(cache)
    return cache


def audit_cache(cache):
    for key, entry in cache.items():
        if key != reference.first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("canonical packet/rubric/role/model identity differs")
        reference.first.audit_row(entry["packet"], entry["role"], entry["row"])


def prior_cache():
    baseline = base_cache()
    if not (JUDGE_PILOT / "worker_terminal_receipt.json").exists():
        return baseline
    reference.data.terminal(JUDGE_PILOT)
    cache = reference.q.read(JUDGE_PILOT / "judgment_cache.json")
    if any(cache.get(k) != v for k, v in baseline.items()):
        raise ValueError("judge pilot changed a prior first decision")
    audit_cache(cache)
    return cache


def producing(destination, stage, inputs=(), config=None):
    study = training.clean.recovery.study
    files = [Path(__file__), PLAN, AUDITOR_FILE, Path(loader.__file__), Path(training.core.__file__),
        Path(training.__file__), training.PLAN, Path(reference.__file__), Path(legacy_eval.__file__), Path(legacy_audit.__file__),
        Path(reference.profile.__file__), reference.profile.PLAN, Path(reference.first.__file__),
        reference.first.AMENDMENT, Path(reference.data.prior.__file__),
        Path(reference.q.__file__), Path(reference.q.original.__file__), Path(reference.data.__file__),
        Path(study.__file__), Path(study.core.__file__), Path(study.base.__file__),
        Path(study.base.tensor_digest.__code__.co_filename),
        Path(reference.previous.study.teacher.old.summarize_scores.__code__.co_filename),
        Path(reference.previous.study.qualifies.__code__.co_filename),
        Path(training.clean.recovery.__file__), Path(training.clean.__file__),
        Path(training.clean.dependencies.__code__.co_filename),
        Path(loader.factory.__file__), Path(loader.factory.released.__file__),
        Path("experiments/structure_net/ncm_blockwise_codec.py"),
        Path("experiments/structure_net/ncm_frozen_source_head.py"),
        Path("experiments/structure_net/ncm_device_geometry.py"),
        Path("experiments/structure_net/ncm_recipe_teacher.py")]
    files += sorted(reference.data.VENDOR.rglob("*.py"))
    common = [training.RESULTS / "campaign_results.json",
        training.RESULTS / (training.HYPOTHESIS + "_registry_readback.json")]
    common += [training.TRAIN / n for n in ("model.pt", "result.json", "producer.json", "worker_terminal_receipt.json")]
    common += [training.FIT / "case_measurements.json", training.FIT / "result.json",
        training.clean.TRAIN / "model.pt", study.bootstrap.TRAIN / "model.pt"]
    common += [loader.factory.released.INITIAL / n for n in ("config.yaml", "generator.pt", "tokenizer.pt")]
    common += sorted(p for p in reference.data.GPT2.iterdir() if p.is_file() and p.suffix in (".json", ".txt"))
    common += [reference.SOURCES / n for n in ("producer.json", "result.json", "cases.json", "worker_terminal_receipt.json")]
    common += [reference.data.FEATURES / n for n in ("producer.json", "result.json", "manifest.json", "worker_terminal_receipt.json")]
    common += [study.expanded.DEST / "cases.json", OLD_CACHE, CACHE_ARCHIVE / "campaign_results.json",
        CACHE_ARCHIVE / "ncm-clean-word-TRAIN-semantic-v1_registry_readback.json"]
    if stage == "source_disjoint_word_final_NCM_development_quality":
        common += [JUDGE_PILOT / n for n in ("producer.json", "result.json", "judgment_cache.json", "worker_terminal_receipt.json")]
    p = study.data.freeze(destination, stage, [*inputs, *common, *files],
        {**(config or {}), "hypothesis_id": HYPOTHESIS, "selected_checkpoint_sha256": CHECKPOINT,
         "constructor_seed": CONSTRUCTOR_SEED, "parent_features_cached": True, "parent_body_loaded": False,
         "precision": "FP32weights_decoder_BF16forecaster_and_word_transformer",
         "matmul_TF32": False, "cuDNN_TF32": False, "source_only": True,
         "oracle_target_codes": False, "AR_fallback": False, "output_repair": False,
         "source_truncated": False, "fixed_generator_passes": 9,
         "output_selection": "single_deterministic_argmax", "stored_token_cap": 128,
         "per_judge_minimum_per_cohort": 61, "same_case_joint_minimum_per_cohort": 61,
         "valid_required_per_cohort": 64, "judges": reference.q.JUDGES, "rubric": reference.profile.RUBRIC,
         "torch": torch.__version__, "cuda": torch.version.cuda,
         "model_training_executed": False, "reused_source_disjoint_development": True,
         "parent_literal_fit_classification_unchanged": "dense_source_TRAIN_fit_failed",
         "final_test_opened": False, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(dict.fromkeys(files)):
        p["source_hashes"][str(path)] = file_hash(path)
        (destination / "producer_source" / f"dense_semantic_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = reference.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(destination / "producer.json", p)
    return p


def inference_probes(cases):
    short, long = (cases[name] for name in reference.COHORTS)
    return [short[0], sorted(long, key=lambda r: (-len(r["source_ids"]), r["id"]))[0]]


def judge_probes(outputs):
    result = []
    for split in reference.COHORTS:
        rows = [r for r in outputs if r["split"] == split]
        first = rows[0]
        other = sorted([r for r in rows if r["id"] != first["id"]],
            key=lambda r: (-r["source_tokens"], r["id"]))[0]
        result.extend((first, other))
    return result


def load_selected():
    saved = torch.load(training.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    if saved["step"] != training.Config().updates or saved["config"] != asdict(training.Config()):
        raise ValueError("only the fixed final2048 checkpoint is selected")
    fixed, branch, codec, _ = loader.load_models(device="cuda")
    set_source_frame(branch.generator, saved["source_frame"], maximum=8192)
    branch.load_state_dict(saved["model"], strict=True)
    if (training.clean.recovery.study.parameter_hash(branch, frozen_only=True) != loader.DENSE_FROZEN_PARAMETERS
            or any(not torch.equal(p.cpu(), saved["model"][n]) for n, p in branch.named_parameters())):
        raise ValueError("all selected dense parameters and original frozen state must match")
    fixed.cuda().eval().requires_grad_(False)
    branch.eval().requires_grad_(False)
    return fixed, branch, codec


@torch.no_grad()
def predict(fixed, branch, row, manifest, tokenizer):
    if any(k in row for k in ("word_ids", "concept_ids", "target_text", "teacher_latent")):
        raise ValueError("source-only prediction cannot receive labels")
    study = training.clean.recovery.study
    batch = study.source_batch(study.source_view(row), manifest)
    for net in (fixed, branch):
        if net.training:
            raise ValueError("both NCM branches must remain eval")
        set_source_frame(net.generator, row["source_frame"], maximum=8192)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        history = study.core.own_history(fixed, batch["features"], batch["mask"], batch["source_ids"])
    conditioning = study.core.conditioning_from_codes(branch.generator.vqvae_local[0], history["coarse_codes"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        latent, logits = training.core.prediction(branch, batch["features"], batch["mask"], conditioning, batch["source_ids"])
    raw = logits.argmax(-1)[0].cpu().tolist()
    text, end = reference.decode_words(raw, tokenizer)
    coarse = [v[0].cpu().tolist() for v in history["coarse_codes"].split(tuple(fixed.generator.levels[:-1]), dim=1)]
    return {**{k: row[k] for k in reference.IDENTITY}, "split": row["split"], "text": text,
        "raw_word_ids": raw, "coarse_codes": coarse, "global_EOS_position": end if end < 192 else None,
        "latent_sha256": study.base.tensor_digest({"latent": latent}), "source_tokens": len(row["source_ids"]),
        "source_frame": row["source_frame"], "source_truncated": False, "fixed_generator_passes": 9,
        "AR_fallback": False, "oracle_target_codes": False, "output_repair": False}


def training_parity_inputs():
    study = training.clean.recovery.study
    rows = reference.q.read(study.expanded.DEST / "cases.json")["train"]
    expected = reference.q.read(training.FIT / "case_measurements.json")
    by_id = {r["id"]: r for r in rows}
    probes = reference.q.read(training.TRAIN / "source_only_reload_probe.json")
    fields = (*reference.IDENTITY, "split", "source_ids", "source_frame", "alignment_sha256", "native_prompt_sha256")
    selected = [{k: by_id[p["id"]][k] for k in fields} for p in probes]
    manifest = reference.q.read(study.expanded.DEST / "manifest.json")
    return selected, manifest, {r["id"]: r for r in expected}


def parity_check(outputs, expected):
    for row in outputs:
        old = expected[row["id"]]
        if (row["raw_word_ids"] != old["raw_word_ids"] or row["latent_sha256"] != old["latent_sha256"]
                or [v for level in row["coarse_codes"] for v in level] != old["actual_coarse_codes"]):
            raise ValueError("fast constructor changed registered TRAIN source-only predictions")


def generate(pilot=False):
    cases, manifest = inputs()
    reference.data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(CONSTRUCTOR_SEED)
    torch.cuda.manual_seed_all(CONSTRUCTOR_SEED)
    all_rows = [r for cohort in cases.values() for r in cohort]
    selected = inference_probes(cases) if pilot else all_rows
    extra = []
    train_rows, train_manifest, expected_train = training_parity_inputs()
    for row, entries in [(r, manifest) for r in selected] + ([(r, train_manifest) for r in train_rows] if pilot else []):
        entry = entries[row["id"]]
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("full original source feature bytes changed")
        extra.append(Path(entry["path"]))
    if not pilot:
        reference.data.terminal(PILOT)
        extra += [PILOT / n for n in ("producer.json", "result.json", "outputs.json", "train_parity_outputs.json", "worker_terminal_receipt.json")]
    destination = PILOT if pilot else OUTPUTS
    p = producing(destination, "dense_semantic_inference_reload_pilot" if pilot else "dense_source_only_development_outputs",
        extra, {"pilot": pilot, "cases": len(selected), "physical_gpu": 0, "output_selection": "single_deterministic_argmax"})
    torch.cuda.reset_peak_memory_stats()
    fixed, branch, codec = load_selected()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(reference.data.GPT2, local_files_only=True)
    study = training.clean.recovery.study
    before = study.parameter_hash(fixed), study.parameter_hash(branch), loader.module_hash(codec)
    train_outputs = [predict(fixed, branch, row, train_manifest, tokenizer) for row in train_rows] if pilot else None
    if pilot:
        parity_check(train_outputs, expected_train)
    retained = {} if pilot else {r["id"]: r for r in reference.q.read(PILOT / "outputs.json")}
    outputs = []
    for row in selected:
        result = predict(fixed, branch, row, manifest, tokenizer)
        if row["id"] in retained and result != retained[row["id"]]:
            raise ValueError("fresh full inference changed a retained pilot result")
        outputs.append(retained.get(row["id"], result))
        write_json(destination / "outputs.json", outputs)
        if pilot or len(outputs) % 8 == 0:
            print(json.dumps({"generated": len(outputs), "total": len(selected)}), flush=True)
    reference.require_outputs(outputs, selected)
    if pilot:
        del fixed, branch, codec
        gc.collect()
        torch.cuda.empty_cache()
        fixed, branch, codec = load_selected()
        if [predict(fixed, branch, row, manifest, tokenizer) for row in selected] != outputs:
            raise ValueError("fresh dense model reload changed development pilot outputs")
        if [predict(fixed, branch, row, train_manifest, tokenizer) for row in train_rows] != train_outputs:
            raise ValueError("fresh reload changed TRAIN constructor parity outputs")
        write_json(destination / "train_parity_outputs.json", train_outputs)
    if (before != (study.parameter_hash(fixed), study.parameter_hash(branch), loader.module_hash(codec))
            or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("read-only semantic inference changed model state or exceeded memory")
    reference.data.verify(p)
    result = {"status": "systems_dense_semantic_inference_passed" if pilot else "dense_source_outputs_pending_semantic_grade",
        "cases": len(outputs), "counts": {s: sum(r["split"] == s for r in outputs) for s in reference.COHORTS},
        "all_parameters_and_codec_unchanged": True, "selected_checkpoint_sha256": CHECKPOINT,
        "output_sha256": file_hash(destination / "outputs.json"), "retained_pilot_exact": not pilot,
        "fresh_reload_or_TRAIN_parity": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "task_accuracy": None, "final_test_opened": False, "speed_tested": False, "producer_fingerprint": p["fingerprint"]}
    write_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)


def require_judges():
    for role in ("secondary", "primary"):
        root = ROOT / ("20261023_budget128_judge_" + role + "_v1")
        reference.data.terminal(root)
        result = reference.q.read(root / "result.json")
        if result["status"] != "passed" or result["controls"] != 50 or result["correct"] != 50:
            raise ValueError("unchanged50/50qualified semantic judges required")


def judge_pilot_valid(summary):
    return summary["cases"] == 4 and summary["valid"] == 4


def judge_pilot():
    inputs()
    require_judges()
    reference.data.terminal(PACKETS)
    reference.data.terminal(OUTPUTS)
    rows = judge_probes(reference.q.read(OUTPUTS / "outputs.json"))
    packets = [reference.q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["text"]}]) for r in rows]
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
        raise ValueError("pilot rerolled a prior first decision")
    write_json(JUDGE_PILOT / "judgment_cache.json", cache)
    reference.data.verify(p)
    if not judge_pilot_valid(summary):
        raise ValueError("semantic pilot transport/schema failed; preserve all decisions")
    write_json(JUDGE_PILOT / "result.json", {"status": "systems_semantic_judge_path_passed",
        "cases": 4, "valid": 4, "passed": summary["passed"], "quality_gate": False,
        "all1669_prior_first_judgments_preserved": True, "unique_first_judgments": len(cache),
        "producer_fingerprint": p["fingerprint"], "task_accuracy": None})


def workflow():
    inputs()
    base_cache()
    paths = (WORKFLOW, PILOT, OUTPUTS, PACKETS, JUDGE_PILOT, GRADE, RESULTS, REPORT,
        Path("data/hypothesis_registry") / (HYPOTHESIS + ".json"))
    if any(p.exists() for p in paths):
        raise ValueError("fresh semantic-development workflow and all destinations required")
    p = producing(WORKFLOW, "supervise_frozen_dense_student_semantic_development")
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [], "producer_fingerprint": p["fingerprint"], "final_test_opened": False}
    try:
        for stage, dest in (("pilot", PILOT), ("generate", OUTPUTS), ("packets", PACKETS),
                ("judge-pilot", JUDGE_PILOT), ("grade", GRADE), ("finalize", RESULTS), ("register", RESULTS)):
            if dest.exists() != (stage == "register"):
                raise ValueError("fresh semantic stage destination required")
            reference.data.verify(p)
            command = [sys.executable, "-u", "-m", AUDITOR if stage in ("finalize", "register") else MODULE, stage]
            child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0" if stage in ("pilot", "generate") else ""})
            journal.update(active_stage=stage, child_pid=child.pid)
            write_json(WORKFLOW / "progress.json", journal)
            code = child.wait()
            receipt = {"status": "observed_worker_terminal", "stage": stage, "exit_code": code,
                "observer_pid": os.getpid(), "child_pid": child.pid, "observer": "subprocess.wait", "command": command, "destination": str(dest)}
            write_json(WORKFLOW / "stage_receipts" / (stage + ".json"), receipt)
            if dest.exists():
                terminal = dest / ("registry_terminal_receipt.json" if stage == "register" else "worker_terminal_receipt.json")
                if terminal.exists():
                    raise ValueError("terminal receipt overwrite prohibited")
                write_json(terminal, receipt)
            journal["stages"].append(receipt)
            if code:
                raise RuntimeError("actual semantic-development stage failure: " + stage)
            reference.data.verify(p)
        path = RESULTS / "campaign_results.json"
        receipt = reference.q.read(RESULTS / (HYPOTHESIS + "_registry_readback.json"))
        if receipt["status"] != "registered_and_read_back" or receipt["exact"]["campaign_sha256"] != file_hash(path):
            raise ValueError("exact semantic-development registry required")
        journal.update(status="dense_semantic_development_completed", active_stage=None,
            classification=reference.q.read(path)["classification"], final_test_opened=False)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "pilot", "generate", "packets", "judge-pilot", "grade", "workflow"))
    args = parser.parse_args()
    configure()
    if args.stage == "check":
        cases, _ = inputs()
        print(json.dumps({"cohorts": {s: len(r) for s, r in cases.items()}, "prior_first_judgments": len(base_cache()),
            "selected_checkpoint_sha256": CHECKPOINT, "TRAIN_literal_gate_amended": False, "final_test_opened": False}))
    elif args.stage == "workflow":
        workflow()
    elif args.stage in ("pilot", "generate"):
        generate(args.stage == "pilot")
    elif args.stage == "packets":
        reference.prepare_packets()
    elif args.stage == "judge-pilot":
        judge_pilot()
    else:
        reference.grade()
