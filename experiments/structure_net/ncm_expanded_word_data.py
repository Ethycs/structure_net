"""Replay fixed teacher decisions and prepare complete expanded NCM TRAIN data."""
import argparse
import json
from pathlib import Path

import torch

from experiments.neural_architecture_lab import finalize_ncm_wide_vetted_labels as replay
from experiments.structure_net import ncm_additional_whole_labels as labels
from experiments.structure_net import ncm_additional_parent_features as additional
from experiments.structure_net import ncm_word_final_data as base
from experiments.structure_net import ncm_broad_learned_codec as released
from experiments.structure_net.ncm_blockwise_codec import BlockwiseCodec
from experiments.structure_net.ncm_memory_data import file_hash, write_json

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-expanded-word-data-plan.md")
DEST = base.ROOT / "20261023_expanded_word_training_data_v1"
PILOT = base.ROOT / "20261023_expanded_word_target_encoding_pilot_v1"


def pilot():
    labels.configure()
    selected = labels.setup(True)
    files = [Path(__file__), PLAN, Path(replay.__file__), Path(released.__file__),
        Path("experiments/structure_net/ncm_blockwise_codec.py")]
    files += sorted(released.VENDOR.rglob("*.py"))
    inputs = files + [released.INITIAL / "tokenizer.pt", released.INITIAL / "config.yaml"]
    for stage in ("draft", "critique", "revision", "audit"):
        root = labels.root(True, stage)
        base.terminal(root)
        inputs.extend(p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts)
    producer = base.producing(PILOT, "systems_expanded_target_encoding_pilot_not_quality", inputs,
        {"cases": 4, "training": False, "model_execution_device": "CPU_FP32_frozen_codec",
            "word_frame": 192, "stored_limit": 128, "task_accuracy": None})
    for index, path in enumerate(files):
        producer["source_hashes"][str(path)] = file_hash(path)
        (PILOT / "producer_source" / f"target_pilot_{index}_{path.name}").write_bytes(path.read_bytes())
    producer["fingerprint"] = labels.prior.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(PILOT / "producer.json", producer)
    for stage, grade in (("draft", "critique"), ("revision", "audit")):
        rows = replay.replay_generation(labels.root(True, stage), selected)
        replay.replay_grade(labels.root(True, grade), rows, grade)
    accepted = labels.prior.q.read(labels.root(True, "audit") / "accepted_labels.json")
    if len(accepted) != 4 or not all(labels.label_fits(r) for r in accepted):
        raise ValueError("all four fixed pilot labels must pass the existing admission checks")
    codec = released.load_codec(device="cpu").eval().requires_grad_(False)
    before = released.module_hash(codec)
    if before != "50fc5680ce0fb3c82764d865243a2e509d8f16dd8c051113b16080b5d763d680":
        raise ValueError("original frozen codec required")
    facade = BlockwiseCodec(codec, 4)
    output = []
    for row in accepted:
        ids = labels._gpt.encode(row["text"], add_special_tokens=False) + [50256]
        if len(ids) > 192:
            raise ValueError("complete target exceeds the fixed canvas")
        words = ids + [50257] * (192-len(ids))
        packed = torch.tensor([words], dtype=torch.long)
        codes = facade.encode(packed)
        if codes.shape != (1, sum(facade.vq.levels)) or not torch.equal(codes, facade.encode(packed)):
            raise ValueError("complete target concept encoding must repeat exactly")
        output.append({"id": row["id"], "text": row["text"], "word_ids": words, "concept_ids": codes[0].tolist()})
    if before != released.module_hash(codec) or any(p.grad is not None for p in codec.parameters()):
        raise ValueError("target encoding changed the original representation")
    base.verify(producer)
    write_json(PILOT / "targets.json", output)
    write_json(PILOT / "result.json", {"status": "systems_expanded_target_encoding_passed_not_quality",
        "cases": 4, "codec_hash_unchanged": before, "targets_sha256": file_hash(PILOT / "targets.json"),
        "all_first_pilot_decisions_replayed": True, "task_accuracy": None, "student_trained": False,
        "producer_fingerprint": producer["fingerprint"]})


def combine_rows(original, sources, accepted):
    """Use accepted whole summaries only; never repurpose atom-level targets."""
    fields = ("id", "source_id", "group_id", "source_sha256", "source", "spans")
    by_id = {r["id"]: r for r in sources}
    if len(by_id) != len(sources):
        raise ValueError("duplicate additional source ID")
    result = [dict(row) for row in original]
    seen = {key: {row[key] for row in original} for key in base.prior.IDENTITY}
    for label in accepted:
        source = by_id[label["id"]]
        if any(label[key] != source[key] for key in fields):
            raise ValueError("accepted label does not match the complete source")
        if label["text"] != label["target"] or not label["teacher_valid"]:
            raise ValueError("exact accepted whole-summary target required")
        if any(source[key] in seen[key] for key in seen):
            raise ValueError("training cases must be disjoint by identity, session, group and source hash")
        result.append({**source, "split": "train", "target_text": label["text"],
            "training_origin": "additional_first_valid_teacher_revision"})
        for key in seen:
            seen[key].add(source[key])
    return result


def prepare():
    labels.configure()
    selected = labels.setup(False)
    base.terminal(PILOT)
    pilot_result = labels.prior.q.read(PILOT / "result.json")
    if (pilot_result["status"] != "systems_expanded_target_encoding_passed_not_quality"
            or file_hash(PILOT / "targets.json") != pilot_result["targets_sha256"]):
        raise ValueError("qualified complete target encoding pilot required")
    retained_targets = {r["id"]: r for r in labels.prior.q.read(PILOT / "targets.json")}
    base.terminal(base.DATA)
    base.terminal(base.FEATURES)
    base.terminal(additional.DATA)
    base.terminal(additional.FEATURES)
    if DEST.exists():
        raise ValueError("expanded training data requires a fresh append-only root")
    import tiktoken
    from transformers import AutoTokenizer
    storage = tiktoken.get_encoding("cl100k_base")
    gpt = AutoTokenizer.from_pretrained(base.GPT2, local_files_only=True)
    native = AutoTokenizer.from_pretrained(base.MODEL_PATH, local_files_only=True)
    stages, outputs, sources_to_hash = [], {}, []
    for pilot in (True, False):
        cases = [next(c for c in selected if c["bin"] == b) for b in range(4)] if pilot else selected
        for stage in ("draft", "critique", "revision", "audit"):
            root = labels.root(pilot, stage)
            base.terminal(root)
            stages.append(root)
            if stage in ("draft", "revision"):
                rows = replay.replay_generation(root, cases)
                for row in rows:
                    text = row["text"]
                    tokens = native.encode(text, add_special_tokens=False)
                    if (row["stored_tokens"] != len(storage.encode(text, disallowed_special=()))
                            or row["native_tokens_including_eos"] != len(tokens)+1
                            or row["native_roundtrip_exact"] != (native.decode(tokens, skip_special_tokens=False) == text)
                            or row["native_special_tokens_in_text"] != any(t in native.all_special_ids for t in tokens)
                            or any(c["source_quotes"] != [row["source"].split("\n")[int(k[1:])]
                                for k in c["evidence_ids"]] for c in row["claims"])):
                        raise ValueError("teacher capacity/evidence metadata changed")
                outputs[pilot, stage] = rows
                if not pilot:
                    retained = {r["id"]: r for r in outputs[True, stage]}
                    if any(row != retained[row["id"]] for row in rows if row["id"] in retained):
                        raise ValueError("pilot candidates were not retained exactly")
            else:
                rows = outputs[pilot, "draft" if stage == "critique" else "revision"]
                scores = replay.replay_grade(root, rows, stage)
                if stage == "audit":
                    expected = [r for r, s in zip(rows, scores) if s["passed"] is True and labels.label_fits(r)]
                    if labels.prior.q.read(root / "accepted_labels.json") != expected:
                        raise ValueError("accepted teacher export differs from first decisions")
            sources_to_hash.extend(p for p in root.rglob("*") if p.is_file() and "producer_source" not in p.parts)
    audit_root = labels.root(False, "audit")
    endpoint = labels.prior.q.read(audit_root / "result.json")
    accepted = labels.prior.q.read(audit_root / "accepted_labels.json")
    if endpoint["endpoint"]["cases"] != 512 or endpoint["endpoint"]["valid"] != 512:
        raise ValueError("all512first final judgments must be valid")
    if not accepted or len(accepted) != endpoint["accepted_labels"]:
        raise ValueError("nonempty exact accepted training set required")
    cache = labels.prior.q.read(audit_root / "judgment_cache.json")
    canonical = labels.prior.q.read(labels.CANONICAL / "judgment_cache.json")
    if len(canonical) != 930 or any(cache.get(key) != value for key, value in canonical.items()):
        raise ValueError("all prior first judgments, including failures, must remain exact")
    for entry in cache.values():
        sources_to_hash.extend(Path(path) for path in entry["row"]["attempts"])
    old_parts = labels.prior.q.read(base.DATA / "cases.json")
    new_sources = labels.prior.q.read(additional.DATA / "cases.json")["additional_train_sources"]
    for root in (base.DATA, additional.DATA):
        if file_hash(root / "cases.json") != labels.prior.q.read(root / "result.json")["cases_sha256"]:
            raise ValueError("complete source metadata changed")
    rows = combine_rows(old_parts["train"], new_sources, accepted)
    for cohort in (old_parts["short_development"], old_parts["long_diagnostic"]):
        base.prior.check_disjoint(rows, cohort)
    sealed = labels.prior.q.read(base.prior.POOL_META)["sealed_final"]
    for key in base.prior.IDENTITY:
        if {r[key] for r in rows} & {r[key] for r in sealed}:
            raise ValueError("sealed final metadata overlap")
    base.verify_public(rows)
    old_manifest = labels.prior.q.read(base.FEATURES / "manifest.json")
    new_manifest = labels.prior.q.read(additional.FEATURES / "manifest.json")
    for root in (base.FEATURES, additional.FEATURES):
        if file_hash(root / "manifest.json") != labels.prior.q.read(root / "result.json")["manifest_sha256"]:
            raise ValueError("completed parent feature manifest changed")
    manifest = {r["id"]: (old_manifest if r["id"] in old_manifest else new_manifest)[r["id"]] for r in rows}
    if set(old_manifest) & set(new_manifest):
        raise ValueError("old and additional source caches must not overlap")
    files = [Path(__file__), PLAN, Path(replay.__file__), Path(released.__file__),
        Path("experiments/structure_net/ncm_blockwise_codec.py")]
    files += sorted(released.VENDOR.rglob("*.py"))
    inputs = list(dict.fromkeys(sources_to_hash)) + files
    for root, names in ((base.DATA, ("producer.json", "result.json", "cases.json", "worker_terminal_receipt.json")),
            (additional.DATA, ("producer.json", "result.json", "cases.json", "worker_terminal_receipt.json")),
            (base.FEATURES, ("producer.json", "result.json", "manifest.json", "worker_terminal_receipt.json")),
            (additional.FEATURES, ("producer.json", "result.json", "manifest.json", "worker_terminal_receipt.json"))):
        inputs.extend(root / name for name in names)
    inputs.extend([released.INITIAL / "tokenizer.pt", released.INITIAL / "config.yaml", base.prior.POOL_META])
    inputs.extend(PILOT / name for name in ("producer.json", "result.json", "targets.json", "worker_terminal_receipt.json"))
    producer = base.producing(DEST, "expanded_whole_summary_TRAIN_metadata_and_original_codec_labels", inputs,
        {"old_training_cases": 252, "additional_source_population": 512, "additional_accepted": len(accepted),
            "model_execution_device": "CPU_FP32_frozen_codec_target_encoding_only", "training": False,
            "old_target_codes_recomputed": False, "word_frame": 192, "stored_limit": 128,
            "torch_version": str(torch.__version__)})
    for index, path in enumerate(files):
        producer["source_hashes"][str(path)] = file_hash(path)
        (DEST / "producer_source" / f"expanded_data_{index}_{path.name}").write_bytes(path.read_bytes())
    producer["fingerprint"] = labels.prior.q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    write_json(DEST / "producer.json", producer)
    codec = released.load_codec(device="cpu").eval().requires_grad_(False)
    codec_hash = released.module_hash(codec)
    if codec_hash != "50fc5680ce0fb3c82764d865243a2e509d8f16dd8c051113b16080b5d763d680":
        raise ValueError("original frozen codec identity differs")
    facade = BlockwiseCodec(codec, 4)
    first_id = accepted[0]["id"]
    longest_id = sorted(accepted, key=lambda r: (-len(gpt.encode(r["text"], add_special_tokens=False)), r["id"]))[0]["id"]
    repeat_ids = {first_id, longest_id}
    for index, row in enumerate(rows):
        if index < 252:
            continue  # Preserve original exact words/codes, not a numerical re-encoding.
        ids = gpt.encode(row["target_text"], add_special_tokens=False) + [50256]
        if (len(ids) > 192 or len(storage.encode(row["target_text"], disallowed_special=())) > 128
                or gpt.decode(ids[:-1], skip_special_tokens=False) != row["target_text"]
                or set(ids[:-1]) & set(gpt.all_special_ids)):
            raise ValueError("accepted whole target no longer fits without clipping")
        words = ids + [50257] * (192-len(ids))
        packed = torch.tensor([words], dtype=torch.long)
        if row["id"] in retained_targets:
            retained = retained_targets[row["id"]]
            if retained["text"] != row["target_text"] or retained["word_ids"] != words:
                raise ValueError("retained pilot target changed")
            codes = torch.tensor([retained["concept_ids"]], dtype=torch.long)
        else:
            codes = facade.encode(packed)
        if row["id"] in repeat_ids and not torch.equal(codes, facade.encode(packed)):
            raise ValueError("target concept encoding must be exactly repeatable")
        row.update(word_ids=words, concept_ids=codes[0].tolist())
        if (index+1) % 32 == 0 or index == len(rows)-1:
            print(json.dumps({"prepared": index+1, "total": len(rows)}), flush=True)
    if released.module_hash(codec) != codec_hash or any(p.grad is not None for p in codec.parameters()):
        raise ValueError("frozen codec changed during target preparation")
    base.verify(producer)
    write_json(DEST / "cases.json", {"train": rows})
    write_json(DEST / "manifest.json", manifest)
    write_json(DEST / "judgment_cache.json", cache)
    accepted_ids = {r["id"] for r in accepted}
    scores = labels.prior.q.read(audit_root / "secondary_audit/case_scores.json")
    write_json(DEST / "teacher_failures.json", [{**score, "format_capacity_fit": labels.label_fits(row)}
        for row, score in zip(outputs[False, "revision"], scores) if row["id"] not in accepted_ids])
    write_json(DEST / "result.json", {"status": "expanded_whole_summary_TRAIN_data_prepared", "cases": len(rows),
        "old_training_cases": 252, "additional_accepted": len(accepted), "additional_population": 512,
        "additional_accepted_by_bin": endpoint["accepted_by_bin"], "all_first_decisions_replayed": True,
        "canonical_judgments": len(cache), "prior930_exact": True, "codec_hash_unchanged": codec_hash,
        "max_source_frame": max(r["source_frame"] for r in rows), "source_disjoint": True,
        "task_accuracy": None, "student_trained": False, "final_test_opened": False, "speed_tested": False,
        "files": {name: file_hash(DEST/name) for name in ("cases.json", "manifest.json", "judgment_cache.json", "teacher_failures.json")},
        "producer_fingerprint": producer["fingerprint"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["pilot", "prepare"])
    stage = parser.parse_args().stage
    pilot() if stage == "pilot" else prepare()
