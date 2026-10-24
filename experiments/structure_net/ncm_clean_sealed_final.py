"""Conditional sealed320 evaluation with fresh original-body integration."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.structure_net import ncm_clean_semantic_development as development
from experiments.structure_net import ncm_sealed_final_quality as old
from experiments.structure_net.ncm_expanded_word_data_workflow import kernel_identity
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.neural_architecture_lab.finalize_ncm_clean_source_partitions import seal_stat

word, data = development.word, development.development
partitions, base, reference = data.partitions, development.reference.data, development.reference
q = development.q
ROOT = word.ROOT
HYPOTHESIS = "ncm-clean-full690-sealed-final-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-clean-sealed-final-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-clean-sealed-final.md")
WORKFLOW = ROOT / "20261024_clean_sealed_final_workflow_v1"
DATA = ROOT / "20261024_clean_sealed_final_data_v1"
FEATURE_PILOT = ROOT / "20261024_clean_sealed_final_feature_pilot_v1"
FEATURES = ROOT / "20261024_clean_sealed_final_features_v1"
PILOT = ROOT / "20261024_clean_sealed_final_inference_pilot_v1"
BODY_REPLAY = ROOT / "20261024_clean_sealed_final_body_replay_v1"
HEAD_REPLAY = ROOT / "20261024_clean_sealed_final_head_replay_v1"
OUTPUTS = ROOT / "20261024_clean_sealed_final_outputs_v1"
PACKETS = ROOT / "20261024_clean_sealed_final_packets_v1"
GRADE = ROOT / "20261024_clean_sealed_final_grade_v1"
RESULTS = ROOT / "20261024_clean_sealed_final_results_v1"
MODULE = "experiments.structure_net.ncm_clean_sealed_final"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_clean_sealed_final"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_clean_sealed_final.py")
SOURCE_CAMPAIGN = "96f602f2f687a589f573a6e5966f07d0acee9a3b43d52e9184ec186359e17538"
COUNTS = {"short_final": 256, "long_final": 64}
SOURCE_SPLITS = {"final_short": "short_final", "final_long": "long_final"}
minimum, qualifies, consensus = old.minimum, old.qualifies, old.consensus
candidate = development


def configure():
    development.configure()
    for name in ("HYPOTHESIS", "PLAN", "DATA", "FEATURE_PILOT", "FEATURES", "PILOT", "OUTPUTS", "PACKETS", "GRADE", "COUNTS"):
        setattr(old, name, globals()[name])
    old.candidate, old.require_admission, old.producing, old.sources = development, require_admission, producing, sources


def sealed_metadata():
    """Metadata/stat only: neither sealed source file is opened or rehashed."""
    campaign_path = partitions.RESULTS / "campaign_results.json"
    campaign = q.read(campaign_path)
    receipt = q.read(partitions.RESULTS / (partitions.HYPOTHESIS + "_registry_readback.json"))
    if (file_hash(campaign_path) != SOURCE_CAMPAIGN or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != SOURCE_CAMPAIGN or receipt["exact"]["campaign_sha256"] != SOURCE_CAMPAIGN):
        raise ValueError("registered full source reservation required")
    for name in ("selection_metadata.json", "seal_receipts.json", "public_inventory.json"):
        path = partitions.DEST / name
        if campaign["artifacts"].get(str(path)) != file_hash(path):
            raise ValueError("reserved source metadata changed")
    selected, seals = q.read(partitions.DEST / "selection_metadata.json"), q.read(partitions.DEST / "seal_receipts.json")
    if set(seals) != set(SOURCE_SPLITS):
        raise ValueError("both final source creation seals required")
    final = {}
    for source_split, split in SOURCE_SPLITS.items():
        seal_stat(seals[source_split], partitions.DEST / (source_split + "_sources.json"))
        rows = selected[source_split]
        if len(rows) != COUNTS[split] or any("source" in r or r["split"] != source_split for r in rows):
            raise ValueError("complete source-free final metadata required")
        final[split] = rows
    if len({r["source_component"] for rows in final.values() for r in rows}) != 320:
        raise ValueError("all320 distinct reserved final components required")
    return final, seals


def development_gate(campaign):
    if not campaign["semantic_development_gate"]:
        return False
    if (campaign["classification"] != "clean_full690_semantic_development_passed_final_unproven"
            or not campaign["same_case_joint_gate"] or not campaign["per_judge_development_gate"]
            or set(campaign["semantic_results"]) != {"short_development", "long_diagnostic"}
            or set(campaign["same_case_joint"]) != {"short_development", "long_diagnostic"}):
        raise ValueError("complete matching development gates required")
    for split, endpoints in campaign["semantic_results"].items():
        joint = campaign["same_case_joint"][split]
        if (not qualifies(endpoints["secondary"], 64) or not qualifies(endpoints["primary"], 64)
                or joint is None or joint["cases"] != 64 or joint["valid"] != 64 or joint["passed"] < 61):
            raise ValueError("same61of64 cases must pass both judges in each development cohort")
    return True


def admission_state():
    if not word.completed(development.WORKFLOW, development.RESULTS, development.HYPOTHESIS,
            "clean_semantic_development_completed", ("sources", "pilot", "generate", "packets", "judge-pilot", "grade", "finalize", "register")):
        return None
    return development_gate(q.read(development.RESULTS / "campaign_results.json"))


def require_admission():
    """This gate precedes every sealed source read/hash, including preparation."""
    if admission_state() is not True:
        raise ValueError("qualified registered same-case development required before final source access")
    campaign = q.read(development.RESULTS / "campaign_results.json")
    for path in [word.TRAIN / "model.pt", word.RESULTS / "campaign_results.json",
            development.GRADE / "judgment_cache.json", development.GRADE / "result.json",
            development.PILOT / "outputs.json", development.SOURCES / "cases.json"]:
        if campaign["artifacts"].get(str(path)) != file_hash(path):
            raise ValueError("registered development candidate or first judgments changed")
    if file_hash(word.TRAIN / "model.pt") != campaign["selected_checkpoint_sha256"]:
        raise ValueError("same selected model must continue into the final test")
    return campaign


def runtime_inputs():
    """Explicitly retain original runtime checkpoint/tokenizer byte identities."""
    expected = {}
    expected.update(q.read(word.forecaster.WORKFLOW / "producer.json")["input_hashes"])
    expected.update(q.read(data.DATA / "producer.json")["input_hashes"])
    expected.update(q.read(data.FEATURES / "producer.json")["input_hashes"])
    paths = [word.loader.factory.released.INITIAL / n for n in ("config.yaml", "generator.pt", "tokenizer.pt")]
    paths += [base.GPT2 / n for n in ("tokenizer.json", "tokenizer_config.json") if (base.GPT2 / n).exists()]
    paths += [base.MODEL_PATH / n for n in ("config.json", "tokenizer.json", "tokenizer_config.json",
        "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")]
    for path in paths:
        if expected.get(str(path)) != file_hash(path):
            raise ValueError("original development runtime checkpoint/tokenizer changed: " + str(path))
    return paths


def expected_models():
    parent = q.read(word.RESULTS / "campaign_results.json")
    admitted = q.read(word.WORKFLOW / "admission.json")
    if admitted != parent["upstream_admission"]:
        raise ValueError("registered matched forecaster selection changed")
    if file_hash(word.forecaster.TRAIN / "model.pt") != admitted["forecaster_checkpoint_sha256"]:
        raise ValueError("same fixed matched forecaster required")
    saved = torch.load(word.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    digest = word.clean.recovery.study.base.tensor_digest({n: saved["model"][n] for n in saved["parameter_names"]})
    return {"forecaster": admitted["forecaster_parameter_sha256"], "word": digest, "codec": word.loader.ORIGINAL_CODEC}


def producing(root, stage, inputs=(), config=None, *, pending=False, identity=None):
    sealed_metadata()
    prior_source = q.read(development.WORKFLOW / "producer.json")
    files = list(dict.fromkeys([Path(__file__), PLAN, AUDITOR_FILE, Path(development.__file__), development.PLAN,
        Path(old.__file__), Path(seal_stat.__code__.co_filename), Path(kernel_identity.__code__.co_filename),
        Path("experiments/neural_architecture_lab/finalize_ncm_sealed_final_quality.py"),
        Path("experiments/neural_architecture_lab/finalize_ncm_clean_development_features.py"),
        *[Path(p) for p in prior_source["source_hashes"]]]))
    common = [development.WORKFLOW / "producer.json", partitions.RESULTS / "campaign_results.json",
        partitions.RESULTS / (partitions.HYPOTHESIS + "_registry_readback.json")]
    common += [partitions.DEST / n for n in ("selection_metadata.json", "seal_receipts.json", "public_inventory.json")]
    if not pending:
        require_admission()
        common += runtime_inputs()
        common += [WORKFLOW / "admission.json", development.WORKFLOW / "result.json",
            development.RESULTS / "campaign_results.json", development.RESULTS / (development.HYPOTHESIS + "_registry_readback.json"),
            development.GRADE / "judgment_cache.json", word.TRAIN / "model.pt", word.forecaster.TRAIN / "model.pt"]
    p = base.freeze(root, stage, [*inputs, *common, *files], {**(config or {}), "hypothesis_id": HYPOTHESIS,
        "schema": "nal.ncm-clean-sealed-final.v1", "counts": COUNTS,
        "joint_minimum": {s: minimum(n) for s, n in COUNTS.items()}, "stored_token_cap": 128,
        "upstream_kernel_identity": identity if pending else q.read(WORKFLOW / "producer.json")["config"]["upstream_kernel_identity"],
        "selected_checkpoint_sha256": None if pending else q.read(WORKFLOW / "admission.json")["selected_checkpoint_sha256"],
        "source_only": True, "source_truncated": False, "oracle_target_codes": False, "AR_fallback": False,
        "output_repair": False, "single_deterministic_argmax": True, "model_training_executed": False,
        "torch": torch.__version__, "cuda": torch.version.cuda, "pending_development_gate": pending,
        "final_test_opened": not pending, "speed_tested": False})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"clean_final_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def final_rows():
    require_admission()
    metadata, seals = sealed_metadata()
    if not (WORKFLOW / "final_access_intent.json").exists():
        raise ValueError("durable final access intent must precede opening sealed source files")
    result = {}
    for source_split, split in SOURCE_SPLITS.items():
        path = partitions.DEST / (source_split + "_sources.json")
        if file_hash(path) != seals[source_split]["creation_sha256"]:
            raise ValueError("sealed source bytes differ from their original creation hash")
        rows = q.read(path)
        if [partitions.metadata(r) for r in rows] != metadata[split]:
            raise ValueError("all original final identities, order and spans must remain")
        result[split] = rows
    if any(r["bin"] != 0 for r in result["short_final"]) or [sum(r["bin"] == b for r in result["long_final"]) for b in range(1, 5)] != [16] * 4:
        raise ValueError("full256short and16long per reserved band required")
    return result


def known_sources():
    known, _, paths = partitions.inputs()
    campaign = q.read(partitions.RESULTS / "campaign_results.json")
    if len(known) != 2828:
        raise ValueError("complete original2828 known source exclusions required")
    for path in paths:
        if campaign["artifacts"].get(str(path)) != file_hash(path):
            raise ValueError("registered known source exclusion input changed")
    return known, paths


def final_exclusion(rows):
    known, paths = known_sources()
    dev = [r for values in q.read(development.SOURCES / "cases.json").values() for r in values]
    inventory = q.read(partitions.DEST / "public_inventory.json")
    development.source_guard([*known, *dev], rows, {k: v["tokens"] for k, v in inventory["turns"].items()})
    return paths


def prepare():
    require_admission()
    if (WORKFLOW / "final_access_intent.json").exists():
        raise ValueError("final source access is single-use; preserve any prior attempt")
    write_json(WORKFLOW / "final_access_intent.json", {"development_gate_passed": True,
        "selected_checkpoint_sha256": q.read(WORKFLOW / "admission.json")["selected_checkpoint_sha256"],
        "source_contents_may_be_opened_after_this_record": True})
    parts = final_rows()
    rows = [r for values in parts.values() for r in values]
    known_paths = final_exclusion(rows)
    proof = base.verify_public(rows)
    native, gpt, storage = data.tokenizers()
    source_paths = [partitions.DEST / (s + "_sources.json") for s in SOURCE_SPLITS]
    p = producing(DATA, "eligible_complete320_final_source_metadata", [*source_paths, *known_paths, base.PUBLIC_FILE,
        development.SOURCES / "cases.json", word.targets.CASES], {"CPU_only": True, "cases": 320})
    cases = {split: [data.prepare_row(r, split, native, gpt, storage) for r in values] for split, values in parts.items()}
    all_rows = [r for values in cases.values() for r in values]
    first = cases["short_final"][0]
    other = [r for r in all_rows if r["id"] != first["id"]]
    native_max = min(other, key=lambda r: (-len(r["context_ids"]), r["id"]))
    gpt_max = min(other, key=lambda r: (-len(r["source_ids"]), r["id"]))
    write_json(DATA / "cases.json", cases)
    write_json(DATA / "selection.json", {"feature_pilot_ids": [first["id"], native_max["id"]],
        "inference_pilot_ids": [first["id"], gpt_max["id"]]})
    write_json(DATA / "public_proof.json", {"public_file_sha256": base.PUBLIC_SHA256, "exact_sources": proof})
    base.verify(p)
    write_json(DATA / "result.json", {"status": "clean_sealed320_metadata_prepared", "counts": {s: len(v) for s, v in cases.items()},
        "cases_sha256": file_hash(DATA / "cases.json"), "source_disjoint": True, "all_sources_complete": True,
        "max_source_frame": max(r["source_frame"] for r in all_rows), "final_test_opened": True,
        "task_accuracy": None, "producer_fingerprint": p["fingerprint"]})


def sources():
    require_admission()
    base.terminal(DATA)
    if file_hash(DATA / "cases.json") != q.read(DATA / "result.json")["cases_sha256"]:
        raise ValueError("fixed final source metadata changed")
    cases = q.read(DATA / "cases.json")
    if ({s: len(v) for s, v in cases.items()} != COUNTS or any(development.FORBIDDEN & r.keys()
            for rows in cases.values() for r in rows)):
        raise ValueError("complete source-only256+64 final cases required")
    return cases


def inputs():
    cases = sources()
    base.terminal(FEATURES)
    manifest = q.read(FEATURES / "manifest.json")
    if (file_hash(FEATURES / "manifest.json") != q.read(FEATURES / "result.json")["manifest_sha256"]
            or set(manifest) != {r["id"] for rows in cases.values() for r in rows}):
        raise ValueError("complete320 frozen source features required")
    return cases, manifest


def features(pilot=False):
    require_admission()
    torch.backends.cudnn.allow_tf32 = False
    base.DATA, base.PILOT, base.FEATURES, base.producing = DATA, FEATURE_PILOT, FEATURES, producing
    base.feature_cache(pilot)


def selected_rows(cases):
    by_id = {r["id"]: r for values in cases.values() for r in values}
    return [by_id[i] for i in q.read(DATA / "selection.json")["inference_pilot_ids"]]


def identity(fixed, branch, codec):
    study = word.clean.recovery.study
    actual = {"forecaster": study.parameter_hash(fixed), "word": study.parameter_hash(branch), "codec": word.loader.module_hash(codec)}
    if actual != q.read(WORKFLOW / "admission.json")["models"]:
        raise ValueError("all actual NCM parameter bytes must match development selection")
    return actual


def require_body_match(row, features, cached):
    if (any(cached.get(k) != row[k] for k in ("id", "source_sha256", "alignment_sha256", "native_prompt_sha256"))
            or features.shape != (len(row["source_ids"]), 3072) or features.dtype != torch.float32
            or features.requires_grad or not torch.isfinite(features).all()
            or not torch.equal(features, cached["features"])):
        raise ValueError("fresh original-body features must retain exact source identity and tensors")


def generate(pilot=False):
    cases, manifest = inputs()
    base.require_gpu(); torch.backends.cudnn.allow_tf32 = False
    rows = selected_rows(cases) if pilot else [r for values in cases.values() for r in values]
    if not pilot:
        for root in (PILOT, BODY_REPLAY, HEAD_REPLAY):
            base.terminal(root)
    extra = []
    for row in rows:
        entry = manifest[row["id"]]
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("actual final source features changed")
        extra.append(Path(entry["path"]))
    root = PILOT if pilot else OUTPUTS
    p = producing(root, "systems_final_inference_reload" if pilot else "all320_fixed_final_outputs", extra,
        {"cases": len(rows), "physical_gpu": 0})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base.GPT2, local_files_only=True)
    torch.manual_seed(development.CONSTRUCTOR_SEED)
    torch.cuda.manual_seed_all(development.CONSTRUCTOR_SEED)
    torch.cuda.reset_peak_memory_stats()
    fixed, branch, codec = development.load_selected()
    before = identity(fixed, branch, codec)
    if pilot:
        train_rows, train_manifest, expected = development.training_parity_inputs()
        outputs = [development.predict(fixed, branch, r, train_manifest, tokenizer) for r in train_rows]
        development.parity_check(outputs, expected)
        write_json(root / "TRAIN_parity_outputs.json", outputs)
        dev_rows = q.read(development.SOURCES / "cases.json")
        by_id = {r["id"]: r for values in dev_rows.values() for r in values}
        dev_manifest = q.read(data.FEATURES / "manifest.json")
        dev_pilot = q.read(development.PILOT / "outputs.json")
        if [development.predict(fixed, branch, by_id[r["id"]], dev_manifest, tokenizer) for r in dev_pilot] != dev_pilot:
            raise ValueError("fixed model changed registered DEV pilot predictions")
        write_json(root / "DEV_parity_outputs.json", dev_pilot)
    retained = {} if pilot else {r["id"]: r for r in q.read(PILOT / "outputs.json")}
    outputs = []
    for row in rows:
        result = development.predict(fixed, branch, row, manifest, tokenizer)
        if row["id"] in retained and result != retained[row["id"]]:
            raise ValueError("fresh full inference changed a retained pilot output")
        outputs.append(retained.get(row["id"], result))
        write_json(root / "outputs.json", outputs)
        if pilot or len(outputs) % 16 == 0:
            print(json.dumps({"generated": len(outputs), "total": len(rows)}), flush=True)
    reference.require_outputs(outputs, rows)
    if pilot:
        del fixed, branch, codec
        gc.collect(); torch.cuda.empty_cache()
        fixed, branch, codec = development.load_selected()
        if [development.predict(fixed, branch, r, manifest, tokenizer) for r in rows] != outputs:
            raise ValueError("fresh final model reload changed inference pilot predictions")
    if identity(fixed, branch, codec) != before or torch.cuda.max_memory_allocated() > 8_000_000_000:
        raise ValueError("unchanged model or8GB inference allocation gate failed")
    base.verify(p)
    write_json(root / "result.json", {"status": "systems_clean_final_inference_passed" if pilot else "clean_final320_outputs_pending_grade",
        "cases": len(outputs), "models": before, "outputs_sha256": file_hash(root / "outputs.json"),
        "fresh_reload_or_retained_pilot_exact": True, "all_parameters_and_codec_unchanged": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "final_test_opened": True,
        "task_accuracy": None, "producer_fingerprint": p["fingerprint"]})


def body_replay():
    cases, manifest = inputs()
    base.require_gpu(); torch.backends.cudnn.allow_tf32 = False
    rows = selected_rows(cases)
    p = producing(BODY_REPLAY, "fresh_original_body_replay_for_actual_final_integration", [FEATURES / "manifest.json"], {"cases": 2})
    from transformers import AutoModelForCausalLM
    parent = AutoModelForCausalLM.from_pretrained(base.MODEL_PATH, torch_dtype=torch.bfloat16,
        local_files_only=True, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    if parent.config.num_hidden_layers != 28 or base.module_hash(parent.model) != base.EXPECTED_BODY:
        raise ValueError("original28-layer frozen parent required")
    torch.cuda.reset_peak_memory_stats()
    (BODY_REPLAY / "features").mkdir()
    replay = {}
    for row in rows:
        ids = torch.tensor([row["context_ids"]], device="cuda")
        hidden = base.source.frozen_parent_features(parent, ids, torch.ones_like(ids))[0]
        features = base.align(hidden, row["alignment"])
        entry = manifest[row["id"]]
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("full-cache source features changed before integration")
        cached = torch.load(entry["path"], map_location="cpu", weights_only=True)
        require_body_match(row, features, cached)
        path = BODY_REPLAY / "features" / (row["id"] + ".pt")
        torch.save({k: features if k == "features" else cached[k] for k in cached}, path)
        replay[row["id"]] = {"path": str(path), "sha256": file_hash(path), "source_tokens": len(features)}
    if (base.module_hash(parent.model) != base.EXPECTED_BODY or any(v.grad is not None for v in parent.parameters())
            or torch.cuda.max_memory_allocated() > 8_600_000_000):
        raise ValueError("original parent/MLPs, gradients or integration memory contract failed")
    write_json(BODY_REPLAY / "manifest.json", replay)
    base.verify(p)
    write_json(BODY_REPLAY / "result.json", {"status": "original_body_replay_exact", "cases": 2,
        "original_body_and_MLPs_sha256": base.EXPECTED_BODY, "cached_features_bit_exact": True,
        "manifest_sha256": file_hash(BODY_REPLAY / "manifest.json"), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "final_test_opened": True, "producer_fingerprint": p["fingerprint"]})


def head_replay():
    cases, _ = inputs()
    base.terminal(BODY_REPLAY); base.terminal(PILOT)
    base.require_gpu(); torch.backends.cudnn.allow_tf32 = False
    manifest = q.read(BODY_REPLAY / "manifest.json")
    if file_hash(BODY_REPLAY / "manifest.json") != q.read(BODY_REPLAY / "result.json")["manifest_sha256"]:
        raise ValueError("fresh body integration features changed")
    p = producing(HEAD_REPLAY, "full_body_to_NCM_output_integration", [BODY_REPLAY / "manifest.json", PILOT / "outputs.json"], {"cases": 2})
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base.GPT2, local_files_only=True)
    fixed, branch, codec = development.load_selected()
    before = identity(fixed, branch, codec)
    outputs = [development.predict(fixed, branch, r, manifest, tokenizer) for r in selected_rows(cases)]
    if outputs != q.read(PILOT / "outputs.json") or identity(fixed, branch, codec) != before:
        raise ValueError("fresh original-body features must reproduce exact final NCM outputs")
    if torch.cuda.max_memory_allocated() > 8_000_000_000:
        raise ValueError("integration head allocation exceeds8GB")
    write_json(HEAD_REPLAY / "outputs.json", outputs)
    base.verify(p)
    write_json(HEAD_REPLAY / "result.json", {"status": "original_body_to_NCM_integration_passed", "cases": 2,
        "models": before, "original_body_and_MLPs_sha256": base.EXPECTED_BODY, "outputs_bit_exact": True,
        "outputs_sha256": file_hash(HEAD_REPLAY / "outputs.json"), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "final_test_opened": True, "producer_fingerprint": p["fingerprint"]})


def workflow():
    sealed_metadata()
    roots = (WORKFLOW, DATA, FEATURE_PILOT, FEATURES, PILOT, BODY_REPLAY, HEAD_REPLAY, OUTPUTS, PACKETS, GRADE, RESULTS, REPORT)
    if any(p.exists() for p in roots):
        raise ValueError("fresh conditional final workflow and all destinations required")
    state = admission_state()
    pid = q.read(development.WORKFLOW / "progress.json")["observer_pid"]
    captured = kernel_identity(pid)
    if (state is None and captured is None) or (captured is not None and development.MODULE.encode() not in Path(f"/proc/{pid}/cmdline").read_bytes()):
        raise ValueError("existing semantic supervisor must be authoritatively live")
    p = producing(WORKFLOW, "observe_DEV_gate_before_any_final_content_access", pending=True, identity=captured)
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "waiting_for_qualified_semantic_development", "observer_pid": os.getpid(),
        "upstream_kernel_identity": captured, "stages": [], "final_test_opened": False, "producer_fingerprint": p["fingerprint"]}
    write_json(WORKFLOW / "progress.json", journal)

    def stage(name, root):
        base.verify(p)
        command = [sys.executable, "-u", "-m", AUDITOR if name in ("decline", "finalize", "register") else MODULE, name]
        gpu = name in ("feature-pilot", "features", "pilot", "body-replay", "head-replay", "generate")
        child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0" if gpu else ""})
        journal.update(status="running", active_stage=name, child_pid=child.pid)
        write_json(WORKFLOW / "progress.json", journal)
        code = child.wait()
        receipt = {"stage": name, "exit_code": code, "child_pid": child.pid, "observer_pid": os.getpid(),
            "observer": "subprocess.wait", "command": command}
        write_json(WORKFLOW / "stage_receipts" / (name + ".json"), receipt)
        if root.exists():
            write_json(root / ("registry_terminal_receipt.json" if name == "register" else "worker_terminal_receipt.json"), receipt)
        journal["stages"].append(receipt)
        if code:
            raise RuntimeError("conditional final stage failed: " + name)

    try:
        while True:
            state = admission_state()
            alive = captured is not None and kernel_identity(captured["pid"]) == captured
            if state is not None and not alive:
                break
            if state is None and not alive:
                raise RuntimeError("semantic supervisor terminal without complete registered outcome")
            time.sleep(5)
        if state is False:
            stage("decline", RESULTS)
            stage("register", RESULTS)
            journal.update(status="development_not_qualified_final_sources_unopened", active_stage=None, child_pid=None, goal_target_met=False)
            write_json(WORKFLOW / "result.json", journal)
            return
        campaign = require_admission()
        paths = runtime_inputs()
        write_json(WORKFLOW / "admission.json", {"semantic_campaign_sha256": file_hash(development.RESULTS / "campaign_results.json"),
            "selected_checkpoint_sha256": campaign["selected_checkpoint_sha256"], "models": expected_models(),
            "baseline_first_cache_sha256": file_hash(development.GRADE / "judgment_cache.json"),
            "runtime_input_hashes": {str(path): file_hash(path) for path in paths},
            "semantic_kernel_terminal": True, "upstream_kernel_identity": captured})
        journal.update(status="admitted_final_source_access", final_source_access_attempted=True, final_test_opened=None)
        write_json(WORKFLOW / "progress.json", journal)
        for name, root in (("prepare", DATA), ("feature-pilot", FEATURE_PILOT), ("features", FEATURES), ("pilot", PILOT),
                ("body-replay", BODY_REPLAY), ("head-replay", HEAD_REPLAY), ("generate", OUTPUTS),
                ("packets", PACKETS), ("grade", GRADE), ("finalize", RESULTS), ("register", RESULTS)):
            stage(name, root)
            if name == "prepare":
                journal["final_test_opened"] = True
        base.verify(p)
        campaign = q.read(RESULTS / "campaign_results.json")
        journal.update(status="clean_sealed_final_completed", active_stage=None, child_pid=None,
            final_test_opened=True, goal_target_met=campaign["goal_target_met"], classification=campaign["classification"])
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "prepare", "feature-pilot", "features", "pilot", "body-replay", "head-replay", "generate", "packets", "grade", "workflow"))
    args = parser.parse_args()
    configure()
    if args.stage == "check":
        metadata, _ = sealed_metadata()
        known, _ = known_sources()
        print(json.dumps({"counts": {s: len(v) for s, v in metadata.items()}, "known_sources": len(known),
            "development_admission": admission_state(), "final_contents_read": False}))
    elif args.stage == "workflow": workflow()
    elif args.stage == "prepare": prepare()
    elif args.stage in ("feature-pilot", "features"): features(args.stage == "feature-pilot")
    elif args.stage in ("pilot", "generate"): generate(args.stage == "pilot")
    elif args.stage == "body-replay": body_replay()
    elif args.stage == "head-replay": head_replay()
    elif args.stage == "packets": old.packets()
    else: old.grade()
