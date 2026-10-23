"""Vetted mixed-length training metadata and immutable original-body source KV."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import torch

from experiments.structure_net import train_ncm_native_query_neighborhood as run
from experiments.structure_net import ncm_native_query_dataset as old_data
from experiments.structure_net import ncm_separated_word_bank as bank
from experiments.structure_net import ncm_application_range_sources as pool
from experiments.structure_net.ncm_native_label_quality import dependencies
from experiments.structure_net.ncm_native_parent_control import SYSTEM
from experiments.structure_net.ncm_native_query_distillation import cpu_prefix, pack_prefix
from experiments.structure_net.ncm_native_readout import frozen_hash, restore_trainable
from experiments.structure_net.ncm_native_word_mechanism import tensor_digest
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_data import ROOT, PREP, DATA as ORIGINAL_DATA, freeze, verify
from experiments.structure_net.ncm_recipe_teacher import verify_public

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-quality-teacher-plan.md")
DEST = ROOT / "20261023_isolated_quality_data_v1"
CACHE_PILOT = ROOT / "20261023_isolated_quality_cache_pilot_v1"
CACHE = ROOT / "20261023_isolated_quality_cache_v1"
WIDE = ROOT / "20261023_wide_labels_main_audit_v1"
WIDE_CAMPAIGN = ROOT / "20261023_wide_vetted_labels_results_v1/campaign_results.json"
WIDE_REGISTRY = Path("data/hypothesis_registry/ncm-wide-vetted-labels-v1.json")
POOL_META = ROOT / "20261023_application_source_pool_results_v1/selection_metadata_replay.json"
IDENTITY = ("id", "source_id", "group_id", "source_sha256")


def frame_for_prompt(length):
    if not 2 <= length <= 2304:
        raise ValueError("complete native prompt must fit2304 without clipping")
    return ((length + 255) // 256) * 256


def check_disjoint(training, heldout):
    counts = {}
    for key in IDENTITY:
        values = {r[key] for r in training}
        if len(values) != len(training):
            raise ValueError("training identity duplicate: " + key)
        counts[key] = len(values & {r[key] for r in heldout})
    if any(counts.values()):
        raise ValueError("training/heldout metadata overlap: " + str(counts))
    return counts


def original_final_metadata(proof, training, development):
    known = {r["id"] for r in [*training, *development]}
    final = {key: value for key, value in proof["exact_sources"].items() if key not in known}
    if len(final) != 256:
        raise ValueError("original256final identity/hash derivation differs")
    return final


def terminal(root):
    if run.q.read(root / "worker_terminal_receipt.json")["exit_code"] != 0:
        raise ValueError("observed terminal success required: " + str(root))
    verify(run.q.read(root / "producer.json"))


def producing(root, stage, inputs, config):
    files = [Path(__file__), PLAN, Path(old_data.__file__), old_data.PLAN, Path(bank.__file__), bank.PLAN,
        Path(pool.__file__), pool.PLAN, Path(run.__file__),
        Path("experiments/structure_net/ncm_native_query_cache.py"),
        Path("experiments/structure_net/ncm_native_query_distillation.py"),
        Path("experiments/structure_net/ncm_native_parent_control.py"),
        Path("experiments/structure_net/ncm_native_readout.py"),
        Path("experiments/structure_net/ncm_native_readout_control.py"),
        Path("experiments/structure_net/ncm_native_label_quality.py"),
        Path("experiments/structure_net/ncm_source_readout_grade.py"),
        Path("experiments/structure_net/ncm_budget128_profile.py")]
    producer = freeze(root, stage, inputs + files, {**config,
        "hypothesis_id": "ncm-isolated-native-quality-teacher-v1", "native_system": SYSTEM,
        "word_frame": 128, "max_native_prompt": 2304, "allow_tf32": False,
        "precision": "BF16native/FP32word", "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"), "task_accuracy": None,
        "final_test_opened": False, "speed_tested": False})
    for path in files:
        producer["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("quality_" + path.name)).write_bytes(path.read_bytes())
    producer["fingerprint"] = run.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(root / "producer.json", producer)
    return producer


def prepare():
    dependencies()
    from transformers import AutoTokenizer
    import tiktoken
    terminal(old_data.DEST); terminal(run.cache_preparation.CACHE); terminal(bank.PILOT)
    if run.q.read(bank.PILOT / "result.json")["status"] != "systems_identity_and_word_training_passed_not_quality_evidence":
        raise ValueError("separate-bank native identity control required")
    registered = run.q.read(WIDE_REGISTRY)
    if registered["status"] != "registered_and_read_back" or file_hash(WIDE_CAMPAIGN) != registered["record"]["campaign_sha256"]:
        raise ValueError("registered wide-label evidence changed")
    for path, expected in run.q.read(WIDE_CAMPAIGN)["artifacts"].items():
        if file_hash(path) != expected:
            raise ValueError("wide-label evidence changed: " + path)
    summaries, audits = old_data.selected_summaries()
    wide = run.q.read(WIDE / "accepted_labels.json")
    if len(summaries) != 143 or len(wide) != 109:
        raise ValueError("fixed143+109 accepted training labels required")
    verify_public([*summaries, *wide])
    old_rows = run.q.read(old_data.DEST / "cases.json")
    existing = {r["id"]: r for r in old_rows["summary_train"]}
    manifest = run.q.read(run.cache_preparation.CACHE / "manifest.json")
    if file_hash(run.cache_preparation.CACHE / "manifest.json") != run.q.read(run.cache_preparation.CACHE / "result.json")["manifest_sha256"]:
        raise ValueError("original source-cache manifest changed")
    metadata = run.q.read(POOL_META)
    development = run.q.read(PREP / "development_sources.json")
    heldout = [*development, *old_rows["language_diagnostic"], *metadata["diagnostic"], *metadata["sealed_final"]]
    overlap = check_disjoint([*summaries, *wide], heldout)
    final = original_final_metadata(run.q.read(PREP / "public_proof.json"),
        run.q.read(PREP / "train_sources.json"), development)
    if set(final) & {r["id"] for r in [*summaries, *wide]} or set(final.values()) & {r["source_sha256"] for r in [*summaries, *wide]}:
        raise ValueError("original final identity/hash overlap")
    original_manifest = run.q.read(ORIGINAL_DATA / "manifest.json")
    if (file_hash(ORIGINAL_DATA / "train.jsonl") != original_manifest["files"]["train.jsonl"]["sha256"]
            or original_manifest["source_session_overlap"] != 0 or original_manifest["eligible_input_overlap"] != 0):
        raise ValueError("original source partition proof changed")
    # Membership in the original train partition inherits its session/group
    # exclusion from native_test, where the original256final was selected.
    needed = {r["id"]: r for r in [*summaries, *wide]}
    recovered = set()
    with (ORIGINAL_DATA / "train.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in needed:
                if any(row[k] != needed[row["id"]][k] for k in IDENTITY) or row["split"] != "train":
                    raise ValueError("original train membership differs")
                recovered.add(row["id"])
    if recovered != set(needed):
        raise ValueError("training label absent from original train partition")
    tokenizer = AutoTokenizer.from_pretrained(run.MODEL_PATH, local_files_only=True)
    stored = tiktoken.get_encoding("cl100k_base")
    rows = []
    long_sources = {r["id"]: r for r in run.q.read(pool.POOL / "train_sources.json")}
    for source, is_old in [(r, True) for r in summaries] + [(r, False) for r in wide]:
        text = source["target"] if is_old else source["text"]
        target = tokenizer.encode(text, add_special_tokens=False)
        if (not text.strip() or len(target) + 1 > 128 or len(stored.encode(text, disallowed_special=())) > 128
                or tokenizer.decode(target, skip_special_tokens=False) != text
                or any(v in tokenizer.all_special_ids for v in target)):
            raise ValueError("complete accepted target format or capacity changed")
        context = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "<passage>\n" + source["source"] + "\n</passage>"}], add_generation_prompt=True)
        old_entry = None
        if is_old:
            previous = existing[source["id"]]
            if previous["context_ids"] != context or previous["target_ids"] != target + [tokenizer.eos_token_id]:
                raise ValueError("old accepted source/target changed")
            old_entry = manifest["cases"][source["id"]]
            if file_hash(old_entry["path"]) != old_entry["sha256"]:
                raise ValueError("original summary cache changed")
            frame = 768
        else:
            parent = long_sources[source["id"]]
            if any(source[k] != parent[k] for k in (*IDENTITY, "source", "spans")) or len(context) != source["native_prompt_tokens"]:
                raise ValueError("long source/canonical prompt differs")
            frame = frame_for_prompt(len(context))
        if len(context) > frame:
            raise ValueError("source clipping prohibited")
        rows.append({**{k: source[k] for k in (*IDENTITY, "source", "spans")}, "kind": "summary",
            "context_ids": context, "source_frame": frame, "native_prompt_tokens": len(context),
            "source_cl100k_tokens": len(stored.encode(source["source"], disallowed_special=())),
            "target_ids": target + [tokenizer.eos_token_id], "target_text": text, "window_eos_injected": False,
            "old_cache_entry": old_entry, "source_band": "old_short" if is_old else source["bin"],
            "label_judge": source["label_judge"] if is_old else run.q.JUDGES["secondary"],
            "label_profile": source["label_profile"] if is_old else "budget128-v1",
            "label_evidence": source["label_evidence"] if is_old else str(WIDE / "judgment_cache.json")})
    longest = sorted(rows[143:], key=lambda r: (-r["native_prompt_tokens"], r["id"]))[0]
    inputs = [old_data.DEST / n for n in ("producer.json", "result.json", "cases.json", "worker_terminal_receipt.json")]
    inputs += [run.cache_preparation.CACHE / n for n in ("manifest.json", "result.json", "worker_terminal_receipt.json")]
    inputs += [bank.PILOT / n for n in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    inputs += [old_data.CRITIQUE / n for n in ("producer.json", "result.json", "judgment_cache.json")]
    inputs += [old_data.DRAFT / "labels.json", *audits, WIDE / "accepted_labels.json", WIDE_CAMPAIGN, WIDE_REGISTRY,
        POOL_META, pool.POOL / "train_sources.json", PREP / "public_proof.json", PREP / "train_sources.json",
        PREP / "development_sources.json", ORIGINAL_DATA / "manifest.json", ORIGINAL_DATA / "train.jsonl"]
    inputs += [run.MODEL_PATH / n for n in ("tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json")]
    producer = producing(DEST, "mixed_length_vetted_quality_training_metadata", inputs,
        {"cases": 252, "old_short": 143, "new_long": 109, "canonical_public_verified": True,
         "original_final_metadata_only": True, "new_final_metadata_only": True})
    write_json(DEST / "cases.json", rows)
    write_json(DEST / "selection.json", {"old_ids": [r["id"] for r in rows[:143]],
        "new_ids": [r["id"] for r in rows[143:]], "pilot_ids": [rows[0]["id"], longest["id"]]})
    write_json(DEST / "split_audit.json", {"direct_metadata_overlap_counts": overlap,
        "original_final_id_hash_count": len(final), "original_train_membership_verified": len(recovered),
        "original_partition_session_and_input_overlap": 0, "original_final_texts_read": False,
        "new_final_texts_read": False, "heldout_metadata_cases": len(heldout)})
    verify(producer)
    result = {"status": "vetted_mixed_length_metadata_prepared", "cases": 252, "old_short": 143, "new_long": 109,
        "source_disjointness_verified": True, "max_native_prompt": longest["native_prompt_tokens"],
        "max_source_frame": max(r["source_frame"] for r in rows), "long_by_bin": [sum(r["source_band"] == i for r in rows) for i in range(4)],
        "pilot_ids": [rows[0]["id"], longest["id"]], "cases_sha256": file_hash(DEST / "cases.json"),
        "task_accuracy": None, "NCM_success_established": False, "full_training_admitted": False,
        "producer_fingerprint": producer["fingerprint"]}
    write_json(DEST / "result.json", result); print(json.dumps(result), flush=True)


@torch.no_grad()
def cache_prepare(pilot=False):
    run.require_gpu(); terminal(DEST)
    rows = run.q.read(DEST / "cases.json")
    if file_hash(DEST / "cases.json") != run.q.read(DEST / "result.json")["cases_sha256"]:
        raise ValueError("training metadata changed")
    if pilot:
        ids = run.q.read(DEST / "selection.json")["pilot_ids"]
        selected = [next(r for r in rows if r["id"] == identifier) for identifier in ids]
    else:
        terminal(CACHE_PILOT)
        if run.q.read(CACHE_PILOT / "result.json")["status"] != "systems_cache_identity_passed_not_quality_evidence":
            raise ValueError("longest-frame source-cache pilot required")
        selected = rows
    root = CACHE_PILOT if pilot else CACHE
    inputs = [DEST / n for n in ("producer.json", "result.json", "cases.json", "selection.json", "worker_terminal_receipt.json")]
    inputs += [run.MODEL_PATH / n for n in ("config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")]
    if not pilot:
        inputs += [CACHE_PILOT / n for n in ("producer.json", "result.json", "manifest.json", "worker_terminal_receipt.json")]
    producer = producing(root, "longest_source_cache_systems_pilot" if pilot else "mixed_length_source_KV_cache",
        inputs, {"pilot": pilot, "selected": [r["id"] for r in selected], "batch": 1, "physical_gpu": 0})
    (root / "cases").mkdir()
    net, tokenizer = run.model_and_tokenizer()
    before = frozen_hash(net.backbone)
    manifest = {"schema": "nal.isolated-quality-source-cache.v1", "cases": {}, "ids": [r["id"] for r in selected],
        "producer_fingerprint": producer["fingerprint"], "native_eos": tokenizer.eos_token_id}
    for i, row in enumerate(selected):
        if row["old_cache_entry"] is not None:
            entry = row["old_cache_entry"]
            if file_hash(entry["path"]) != entry["sha256"]:
                raise ValueError("old source cache changed")
            manifest["cases"][row["id"]] = {**entry, "reused_old_cache": True}
        else:
            memory = net.backbone.source_cache(**old_data.source_inputs([row], tokenizer.eos_token_id, "cuda"))
            targets = old_data.target_tensor([row], "cpu")[0]
            example = {"id": row["id"], "prefix": cpu_prefix(memory, 0), "targets": targets,
                "kind": "summary", "window_eos_injected": False, "source_frame": row["source_frame"],
                "producer_fingerprint": producer["fingerprint"]}
            path = root / "cases" / (row["id"] + ".pt")
            torch.save(example, path)
            restored = torch.load(path, map_location="cpu", weights_only=True)
            actual = net.backbone.query_logits(memory, targets[:-1][None].cuda().masked_fill(targets[:-1][None].cuda().eq(-100), tokenizer.eos_token_id))
            reload = net.backbone.query_logits(pack_prefix([restored["prefix"]], "cuda"),
                restored["targets"][:-1][None].cuda().masked_fill(restored["targets"][:-1][None].cuda().eq(-100), tokenizer.eos_token_id))
            if not torch.equal(actual, reload) or not torch.isfinite(actual).all():
                raise ValueError("source cache save/reload logits differ")
            manifest["cases"][row["id"]] = {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size,
                "source_frame": row["source_frame"], "source_sha256": row["source_sha256"], "reused_old_cache": False}
            del memory, actual, reload
        write_json(root / "manifest.json", manifest)
        if pilot or (i + 1) % 16 == 0 or i + 1 == len(selected):
            print(json.dumps({"cached": i + 1, "total": len(selected)}), flush=True)
    if frozen_hash(net.backbone) != before:
        raise ValueError("original source backbone mutated")
    verify(producer)
    result = {"status": "systems_cache_identity_passed_not_quality_evidence" if pilot else "mixed_length_source_caches_prepared",
        "cases": len(selected), "reused_old": sum(e["reused_old_cache"] for e in manifest["cases"].values()),
        "source_only": True, "cache_reload_logits_exact": True, "original_backbone_unchanged": True,
        "max_source_frame": max(r["source_frame"] for r in selected), "manifest_sha256": file_hash(root / "manifest.json"),
        "task_accuracy": None, "NCM_success_established": False, "full_training_admitted": False,
        "producer_fingerprint": producer["fingerprint"]}
    write_json(root / "result.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "cache-pilot", "cache"])
    command = parser.parse_args().command
    prepare() if command == "prepare" else cache_prepare(command == "cache-pilot")
