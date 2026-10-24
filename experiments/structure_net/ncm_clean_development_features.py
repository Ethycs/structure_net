"""Complete source-only DEV features after exchange components are reserved."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.structure_net import ncm_clean_exchange_partitions as partitions
from experiments.structure_net import ncm_dense_matched_forecaster as forecaster
from experiments.structure_net import ncm_word_final_data as base
from experiments.structure_net.ncm_expanded_word_data_workflow import kernel_identity
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

ROOT = base.ROOT
HYPOTHESIS = "ncm-clean-development-frozen-features-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-24_ncm-clean-development-features-plan.md")
REPORT = Path("docs/08 - Analysis/2026-10-24_ncm-clean-development-features.md")
WORKFLOW = ROOT / "20261024_clean_development_features_workflow_v1"
METADATA_PILOT = ROOT / "20261024_clean_development_tokenizer_pilot_v1"
DATA = ROOT / "20261024_clean_development_metadata_v1"
PILOT = ROOT / "20261024_clean_development_feature_pilot_v1"
FEATURES = ROOT / "20261024_clean_development_features_v1"
RESULTS = ROOT / "20261024_clean_development_feature_results_v1"
MODULE = "experiments.structure_net.ncm_clean_development_features"
AUDITOR = "experiments.neural_architecture_lab.finalize_ncm_clean_development_features"
AUDITOR_FILE = Path("experiments/neural_architecture_lab/finalize_ncm_clean_development_features.py")
SPLITS = {"development_short": "short_development", "development_long": "long_diagnostic"}
FORBIDDEN = {"target", "target_text", "word_ids", "concept_ids", "summary", "candidate"}
EXPECTED = {"short_development": 64, "long_diagnostic": 64}
MAXIMUM = 8192
q = base.prior.run.q


def admitted_sources():
    """Read only open DEV files and metadata; never hash sealed source files."""
    path = partitions.RESULTS / "campaign_results.json"
    campaign = q.read(path)
    receipt = q.read(partitions.RESULTS / (partitions.HYPOTHESIS + "_registry_readback.json"))
    done = q.read(partitions.WORKFLOW / "result.json")
    digest = file_hash(path)
    if (campaign["classification"] != "clean_exchange_sources_reserved_quality_unmeasured"
            or campaign["counts"] != {"final_short": 256, "final_long": 64,
                "development_short": 64, "development_long": 64}
            or campaign["final_test_opened"] or campaign["model_outputs_or_labels_used"]
            or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != digest
            or receipt["exact"]["campaign_sha256"] != digest
            or done["status"] != "clean_exchange_partitions_completed"
            or [(s["stage"], s["exit_code"]) for s in done["stages"]]
            != [(s, 0) for s in ("pilot", "prepare", "finalize", "register")]):
        raise ValueError("registered complete128DEV/320final source reservation required")
    paths = [path, partitions.RESULTS / (partitions.HYPOTHESIS + "_registry_readback.json"),
        partitions.WORKFLOW / "result.json"]
    parts = {}
    for source_split, split in SPLITS.items():
        source_path = partitions.DEST / (source_split + "_sources.json")
        if campaign["artifacts"].get(str(source_path)) != file_hash(source_path):
            raise ValueError("registered open development bytes changed")
        rows = q.read(source_path)
        if len(rows) != 64 or any(r["split"] != source_split or FORBIDDEN & r.keys() for r in rows):
            raise ValueError("all64 source-only cases per development cohort required")
        parts[split] = rows
        paths.append(source_path)
    rows = [r for values in parts.values() for r in values]
    if (len({r["id"] for r in rows}) != 128 or len({r["source_component"] for r in rows}) != 128
            or any(text_hash(r["source"]) != r["source_sha256"] or r["source_truncated"] for r in rows)):
        raise ValueError("complete unique development source components required")
    return parts, paths


def prepare_row(row, split, native, gpt, storage):
    if FORBIDDEN & row.keys() or text_hash(row["source"]) != row["source_sha256"]:
        raise ValueError("unaltered source-only metadata required")
    messages = [{"role": "system", "content": base.SYSTEM},
        {"role": "user", "content": "<passage>\n" + row["source"] + "\n</passage>"}]
    context = native.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = native(context, add_special_tokens=False, return_offsets_mapping=True)
    direct = native.apply_chat_template(messages, add_generation_prompt=True)
    if encoded["input_ids"] != direct:
        raise ValueError("canonical native prompt tokenization differs")
    start = context.index("<passage>\n") + len("<passage>\n")
    if context[start:start + len(row["source"])] != row["source"]:
        raise ValueError("literal complete passage boundary differs")
    raw = gpt(row["source"], add_special_tokens=False, return_offsets_mapping=True)
    if gpt.decode(raw["input_ids"], skip_special_tokens=False) != row["source"]:
        raise ValueError("source GPT2 round trip failed")
    mapping = base.overlaps(raw["offset_mapping"], encoded["offset_mapping"], start)
    tokens = len(storage.encode(row["source"], disallowed_special=()))
    if not 192 <= tokens <= 2048 or tokens != row["source_cl100k_tokens"]:
        raise ValueError("complete source token range differs")
    return {**row, "partition_split": row["split"], "split": split,
        "context_ids": direct, "source_ids": raw["input_ids"], "alignment": mapping,
        "alignment_sha256": q.digest(mapping), "native_prompt_sha256": q.digest(direct),
        "source_frame": base.source_frame(len(raw["input_ids"]), maximum=MAXIMUM)}


def select_pilot(parts):
    rows = [r for values in parts.values() for r in values]
    first = parts["short_development"][0]
    longest = min((r for r in rows if r["id"] != first["id"]),
        key=lambda r: (-len(r["context_ids"]), r["id"]))
    return [first["id"], longest["id"]]


def producing(root, stage, inputs, config):
    paths = []
    if stage != "systems_actual_public_source_tokenizer_pilot":
        _, paths = admitted_sources()
    files = [Path(__file__), PLAN, AUDITOR_FILE, Path(base.__file__), Path(partitions.__file__),
        Path(base.source.__file__), Path("experiments/structure_net/ncm_native_parent_control.py"),
        Path("experiments/structure_net/ncm_matched_parent_control.py"),
        Path("experiments/structure_net/ncm_prefix_geometry.py"),
        Path("experiments/structure_net/ncm_expanded_word_data_workflow.py"),
        Path("experiments/neural_architecture_lab/ncm_result_registry.py")]
    p = base.freeze(root, stage, [*paths, *inputs, *files], {**config,
        "hypothesis_id": HYPOTHESIS, "schema": "nal.ncm-clean-development-features.v1",
        "counts": config.get("counts", EXPECTED), "max_source_frame": MAXIMUM,
        "native_system": base.SYSTEM, "parent_layers": 28, "parent_precision": "BF16",
        "aligned_features": "FP32", "parent_weights_frozen": True, "original_MLPs_frozen": True,
        "source_right_padding": True, "all_source_tokens_retained": True,
        "target_labels_used": False, "final_test_opened": False, "task_accuracy": None,
        "speed_tested": False, "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers")})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"clean_dev_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def tokenizers():
    base.prior.dependencies()
    from transformers import AutoTokenizer
    import tiktoken
    native = AutoTokenizer.from_pretrained(base.MODEL_PATH, local_files_only=True)
    gpt = AutoTokenizer.from_pretrained(base.GPT2, local_files_only=True)
    gpt.add_special_tokens({"pad_token": "<|pad|>"})
    gpt.model_max_length = 1_000_000
    if gpt.pad_token_id != 50257 or gpt.eos_token_id != 50256:
        raise ValueError("released GPT2 vocabulary identity differs")
    return native, gpt, tiktoken.get_encoding("cl100k_base")


def prepare():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("metadata preparation must stay CPU only")
    base.terminal(METADATA_PILOT)
    if q.read(METADATA_PILOT / "result.json")["status"] != "systems_public_tokenizer_alignment_passed":
        raise ValueError("actual public tokenizer lifecycle pilot required")
    parts, inputs = admitted_sources()
    inputs += [METADATA_PILOT / n for n in ("producer.json", "result.json", "cases.json", "public_proof.json", "worker_terminal_receipt.json")]
    native, gpt, storage = tokenizers()
    rows = [r for values in parts.values() for r in values]
    proof = base.verify_public(rows)
    inputs += [base.PUBLIC_FILE]
    inputs += [base.MODEL_PATH / n for n in ("config.json", "tokenizer.json", "tokenizer_config.json")]
    inputs += [base.GPT2 / n for n in ("tokenizer.json", "tokenizer_config.json") if (base.GPT2 / n).exists()]
    p = producing(DATA, "complete128_clean_DEV_prompt_and_character_alignment", inputs, {"CPU_only": True})
    prepared = {split: [prepare_row(r, split, native, gpt, storage) for r in values] for split, values in parts.items()}
    write_json(DATA / "cases.json", prepared)
    write_json(DATA / "selection.json", {"feature_pilot_ids": select_pilot(prepared)})
    write_json(DATA / "public_proof.json", {"public_file_sha256": base.PUBLIC_SHA256,
        "exact_sources": proof, "matched": len(proof)})
    if q.read(DATA / "cases.json") != prepared or len(proof) != 128:
        raise ValueError("complete metadata strict JSON readback required")
    base.verify(p)
    all_rows = [r for values in prepared.values() for r in values]
    result = {"status": "clean128_DEV_metadata_prepared", "counts": {s: len(v) for s, v in prepared.items()},
        "cases_sha256": file_hash(DATA / "cases.json"), "max_native_prompt": max(len(r["context_ids"]) for r in all_rows),
        "max_GPT2_source": max(len(r["source_ids"]) for r in all_rows),
        "max_source_frame": max(r["source_frame"] for r in all_rows), "all_sources_complete": True,
        "source_only": True, "task_accuracy": None, "final_test_opened": False,
        "producer_fingerprint": p["fingerprint"]}
    write_json(DATA / "result.json", result)
    print(json.dumps(result), flush=True)


def metadata_pilot():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
        raise ValueError("public tokenizer pilot must stay CPU only")
    partitions.terminal(partitions.PILOT)
    paths = [partitions.PILOT / n for n in ("producer.json", "result.json", "pilot_sources.json", "worker_terminal_receipt.json")]
    rows = q.read(partitions.PILOT / "pilot_sources.json")
    if len(rows) != 5 or {r["bin"] for r in rows} != set(range(5)):
        raise ValueError("actual five public source bands required")
    native, gpt, storage = tokenizers()
    proof = base.verify_public(rows)
    paths += [base.PUBLIC_FILE]
    paths += [base.MODEL_PATH / n for n in ("config.json", "tokenizer.json", "tokenizer_config.json")]
    paths += [base.GPT2 / n for n in ("tokenizer.json", "tokenizer_config.json") if (base.GPT2 / n).exists()]
    p = producing(METADATA_PILOT, "systems_actual_public_source_tokenizer_pilot", paths,
        {"CPU_only": True, "systems_lifecycle_only_not_quality_evidence": True, "counts": {"systems_pilot": 5}})
    prepared = [prepare_row(r, "systems_pilot", native, gpt, storage) for r in rows]
    write_json(METADATA_PILOT / "cases.json", prepared)
    write_json(METADATA_PILOT / "public_proof.json", proof)
    if q.read(METADATA_PILOT / "cases.json") != prepared or len(proof) != 5 or torch.cuda.is_initialized():
        raise ValueError("complete CPU metadata/provenance strict JSON lifecycle required")
    base.verify(p)
    result = {"status": "systems_public_tokenizer_alignment_passed", "cases": 5,
        "all_five_source_bands": True, "CPU_only": True, "task_accuracy": None,
        "all_sources_complete": True, "target_labels_used": False,
        "max_native_prompt": max(len(r["context_ids"]) for r in prepared),
        "max_GPT2_source": max(len(r["source_ids"]) for r in prepared),
        "max_source_frame": max(r["source_frame"] for r in prepared),
        "cases_sha256": file_hash(METADATA_PILOT / "cases.json"), "producer_fingerprint": p["fingerprint"]}
    write_json(METADATA_PILOT / "result.json", result)
    print(json.dumps(result), flush=True)


def observe_metadata_pilot():
    if METADATA_PILOT.exists():
        raise ValueError("fresh actual public metadata pilot required")
    command = [sys.executable, "-u", "-m", MODULE, "metadata-pilot-worker"]
    child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    code = child.wait()
    if METADATA_PILOT.exists():
        write_json(METADATA_PILOT / "worker_terminal_receipt.json", {"stage": "metadata-pilot",
            "exit_code": code, "child_pid": child.pid, "observer_pid": os.getpid(),
            "observer": "subprocess.wait", "command": command})
    if code:
        raise RuntimeError("actual public metadata pilot failure preserved")


def configure():
    # Only this fresh worker's globals change; all earlier source files stay intact.
    base.DATA, base.PILOT, base.FEATURES = DATA, PILOT, FEATURES
    base.producing = producing


def forecaster_ready():
    if (forecaster.WORKFLOW / "failure.json").exists():
        raise RuntimeError("upstream forecaster failure preserved")
    path = forecaster.WORKFLOW / "result.json"
    if not path.exists():
        return False
    done = q.read(path)
    campaign_path = forecaster.RESULTS / "campaign_results.json"
    receipt = q.read(forecaster.RESULTS / (forecaster.HYPOTHESIS + "_registry_readback.json"))
    if (done["status"] != "dense_matched_forecaster_completed"
            or [(s["stage"], s["exit_code"]) for s in done["stages"]]
            != [(s, 0) for s in ("train_pilot", "fit_pilot", "train", "fit", "cache", "finalize", "register")]
            or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != file_hash(campaign_path)
            or receipt["exact"]["campaign_sha256"] != file_hash(campaign_path)):
        raise ValueError("observed successful forecaster lifecycle and exact registry required")
    return True


def workflow():
    admitted_sources()
    if any(p.exists() for p in (WORKFLOW, DATA, PILOT, FEATURES, RESULTS, REPORT,
            Path("data/hypothesis_registry") / (HYPOTHESIS + ".json"))):
        raise ValueError("fresh clean-development feature workflow required")
    pending = not forecaster_ready()
    upstream_pid = q.read(forecaster.WORKFLOW / "progress.json")["observer_pid"]
    identity = kernel_identity(upstream_pid)
    if pending and identity is None:
        raise ValueError("pending forecaster supervisor must be authoritatively live")
    if identity is not None and forecaster.MODULE.encode() not in Path(f"/proc/{upstream_pid}/cmdline").read_bytes():
        raise ValueError("upstream kernel identity must belong to the existing forecaster")
    p = producing(WORKFLOW, "prepare_DEV_then_serialize_frozen_body_after_forecaster", [forecaster.WORKFLOW / "producer.json"],
        {"training": False, "upstream_kernel_identity": identity, "upstream_pid": upstream_pid, "poll_seconds": 5,
         "upstream_worker_restarted": False, "feature_GPU_concurrency": 1})
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "observer_pid": os.getpid(), "stages": [],
        "upstream_kernel_identity": identity, "producer_fingerprint": p["fingerprint"]}

    def stage(name, root, cpu=True):
        base.verify(p)
        command = [sys.executable, "-u", "-m", AUDITOR if name in ("finalize", "register") else MODULE, name]
        child = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "" if cpu else "0"})
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
            raise RuntimeError("clean DEV feature stage failed: " + name)

    try:
        stage("prepare", DATA)
        while True:
            ready = forecaster_ready()
            alive = identity is not None and kernel_identity(identity["pid"]) == identity
            if ready and not alive:
                write_json(WORKFLOW / "upstream_terminal_observation.json", {
                    "upstream_pid": upstream_pid, "captured_kernel_identity": identity,
                    "same_kernel_still_live": False, "complete_registered_lifecycle": True,
                    "observer_pid": os.getpid(), "observer": "proc_stat_PID_and_start_ticks",
                    "campaign_sha256": file_hash(forecaster.RESULTS / "campaign_results.json")})
                break
            if not ready and not alive:
                raise RuntimeError("forecaster terminal without required successful lifecycle")
            journal.update(status="waiting_for_existing_forecaster", active_stage=None, child_pid=None)
            write_json(WORKFLOW / "progress.json", journal)
            time.sleep(5)
        for name, root in (("feature-pilot", PILOT), ("features", FEATURES)):
            stage(name, root, cpu=False)
        stage("finalize", RESULTS)
        stage("register", RESULTS)
        base.verify(p)
        journal.update(status="clean_development_features_completed", active_stage=None, child_pid=None)
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("check", "metadata-pilot", "metadata-pilot-worker", "prepare", "feature-pilot", "features", "workflow"))
    args = parser.parse_args()
    configure()
    if args.stage == "metadata-pilot":
        observe_metadata_pilot()
    elif args.stage == "metadata-pilot-worker":
        metadata_pilot()
    elif args.stage == "check":
        parts, _ = admitted_sources()
        print(json.dumps({"counts": {s: len(v) for s, v in parts.items()}, "task_accuracy": None}))
    elif args.stage == "prepare":
        prepare()
    elif args.stage == "workflow":
        workflow()
    else:
        base.feature_cache(args.stage == "feature-pilot")
