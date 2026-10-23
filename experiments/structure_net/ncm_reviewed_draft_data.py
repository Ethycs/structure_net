"""Prepare TRAIN targets from existing first teacher drafts and first critiques."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_expanded_word_data as previous

labels = previous.labels
base = previous.base
q = labels.prior.q
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-reviewed-draft-recovery-plan.md")
PILOT = base.ROOT / "20261023_reviewed_draft_pilot_v1"
DEST = base.ROOT / "20261023_reviewed_draft_data_v1"
ABORT = base.ROOT / "20261023_additional_revision_transport_abort_v1"
CODEC_HASH = "50fc5680ce0fb3c82764d865243a2e509d8f16dd8c051113b16080b5d763d680"


def select_accepted(rows, scores, fits):
    if len(rows) != len(scores) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("complete unique teacher denominator required")
    accepted, rejected = [], []
    for row, score in zip(rows, scores):
        if (row["id"] != score["id"] or row["text"] != score["text"]
                or score["valid"] is not True or type(score["passed"]) is not bool):
            raise ValueError("valid first judgment with identical teacher text required")
        fit = bool(fits(row))
        if score["passed"] and fit:
            accepted.append(row)
        else:
            rejected.append({"label": row, "score": score, "format_capacity_fit": fit})
    if not accepted:
        raise ValueError("nonempty independently reviewed training set required")
    return accepted, rejected


def capture(root, producer):
    files = [Path(__file__), PLAN, Path("experiments/structure_net/ncm_reviewed_draft_recovery.py"),
        Path("experiments/structure_net/ncm_reviewed_draft_workflow.py")]
    for i, path in enumerate(files):
        producer["source_hashes"][str(path)] = base.file_hash(path)
        (root / "producer_source" / f"reviewed_recovery_{i}_{path.name}").write_bytes(path.read_bytes())
    producer["fingerprint"] = q.digest({k: v for k, v in producer.items() if k != "fingerprint"})
    base.write_json(root / "producer.json", producer)
    return producer


def reviewed():
    labels.configure()
    selected = labels.setup(False)
    abort = q.read(ABORT / "observation.json")["actual_parent_receipt"]
    if abort is None or abort["exit_code"] != -15:
        raise ValueError("preserved actual interrupted revision receipt required")
    outputs, scores, roots = {}, {}, []
    for pilot in (True, False):
        cases = [next(c for c in selected if c["bin"] == b) for b in range(4)] if pilot else selected
        draft, critique = labels.root(pilot, "draft"), labels.root(pilot, "critique")
        for root in (draft, critique):
            base.terminal(root)
            roots.append(root)
        rows = previous.replay.replay_generation(draft, cases)
        decisions = previous.replay.replay_grade(critique, rows, "critique")
        if len(rows) != (4 if pilot else 512):
            raise ValueError("fixed complete source population required")
        outputs[pilot], scores[pilot] = rows, decisions
    retained = {row["id"]: row for row in outputs[True]}
    if any(row != retained[row["id"]] for row in outputs[False] if row["id"] in retained):
        raise ValueError("first pilot drafts were regenerated")
    import tiktoken
    from transformers import AutoTokenizer
    storage = tiktoken.get_encoding("cl100k_base")
    native = AutoTokenizer.from_pretrained(base.MODEL_PATH, local_files_only=True)
    for row in outputs[False]:
        tokens = native.encode(row["text"], add_special_tokens=False)
        if (row["stored_tokens"] != len(storage.encode(row["text"], disallowed_special=()))
                or row["native_tokens_including_eos"] != len(tokens)+1
                or row["native_roundtrip_exact"] != (native.decode(tokens, skip_special_tokens=False) == row["text"])
                or row["native_special_tokens_in_text"] != any(v in native.all_special_ids for v in tokens)
                or any(c["source_quotes"] != [row["source"].split("\n")[int(k[1:])] for k in c["evidence_ids"]]
                    for c in row["claims"])):
            raise ValueError("complete capacity or source-evidence metadata changed")
    accepted, rejected = select_accepted(outputs[False], scores[False], labels.label_fits)
    cache = q.read(labels.root(False, "critique") / "judgment_cache.json")
    canonical = q.read(labels.CANONICAL / "judgment_cache.json")
    if len(canonical) != 930 or any(cache.get(k) != v for k, v in canonical.items()):
        raise ValueError("all930 preceding first judgments must remain exact")
    return accepted, rejected, cache, roots


def word_ids(text):
    ids = labels._gpt.encode(text, add_special_tokens=False)
    if (len(ids)+1 > 192 or labels._gpt.decode(ids, skip_special_tokens=False) != text
            or set(ids) & set(labels._gpt.all_special_ids)):
        raise ValueError("complete target cannot be clipped or contain controls")
    return ids+[50256]+[50257]*(191-len(ids))


def make_producer(root, stage, roots, extra, config):
    files = [Path(__file__), PLAN, Path(previous.__file__), previous.PLAN,
        Path(previous.replay.__file__), Path(previous.released.__file__),
        Path("experiments/structure_net/ncm_blockwise_codec.py")]
    files += sorted(previous.released.VENDOR.rglob("*.py"))
    inputs = files + list(extra) + [ABORT / "decision_before_signal.json", ABORT / "observation.json"]
    for directory in roots:
        inputs.extend(p for p in directory.rglob("*") if p.is_file() and "producer_source" not in p.parts)
    inputs += [previous.released.INITIAL / "tokenizer.pt", previous.released.INITIAL / "config.yaml"]
    producer = base.producing(root, stage, list(dict.fromkeys(inputs)), {**config,
        "teacher_target_stage": "first_draft", "teacher_review_stage": "first_critique",
        "semantic_rerolls": False, "student_evaluation_used_for_selection": False,
        "model_execution_device": "CPU_FP32_frozen_codec", "task_accuracy": None,
        "stored_limit": 128, "word_frame": 192})
    return capture(root, producer)


@torch.no_grad()
def encode(pilot=False):
    accepted, rejected, cache, roots = reviewed()
    chosen = [next(row for row in accepted if row["bin"] == b) for b in range(4)]
    destination = PILOT if pilot else DEST
    extra = []
    retained = {}
    if not pilot:
        base.terminal(PILOT)
        result = q.read(PILOT / "result.json")
        if (result["status"] != "systems_reviewed_draft_target_encoding_passed_not_quality"
                or base.file_hash(PILOT / "targets.json") != result["targets_sha256"]):
            raise ValueError("qualified reviewed-draft encoding pilot required")
        retained = {r["id"]: r for r in q.read(PILOT / "targets.json")}
        if set(retained) != {r["id"] for r in chosen}:
            raise ValueError("fixed pilot admission changed")
        extra += [PILOT / n for n in ("producer.json", "result.json", "targets.json", "worker_terminal_receipt.json")]
    old_parts = q.read(base.DATA / "cases.json")
    for root in (base.DATA, base.FEATURES, previous.additional.DATA, previous.additional.FEATURES):
        base.terminal(root)
        extra += [root / n for n in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    for root in (base.DATA, previous.additional.DATA):
        if base.file_hash(root / "cases.json") != q.read(root / "result.json")["cases_sha256"]:
            raise ValueError("complete source metadata changed")
        extra.append(root / "cases.json")
    sources = q.read(previous.additional.DATA / "cases.json")["additional_train_sources"]
    rows = previous.combine_rows(old_parts["train"], sources, accepted)
    for row in rows[252:]:
        row["training_origin"] = "additional_first_valid_teacher_draft_after_first_critique"
    for cohort in (old_parts["short_development"], old_parts["long_diagnostic"]):
        base.prior.check_disjoint(rows, cohort)
    sealed = q.read(base.prior.POOL_META)["sealed_final"]
    for key in base.prior.IDENTITY:
        if {r[key] for r in rows} & {r[key] for r in sealed}:
            raise ValueError("sealed metadata overlap")
    public_proof, train_meta, dev_meta = (q.read(base.PREP / n) for n in
        ("public_proof.json", "train_sources.json", "development_sources.json"))
    original_final = base.prior.original_final_metadata(public_proof, train_meta, dev_meta)
    if set(original_final) & {r["id"] for r in rows} or set(original_final.values()) & {r["source_sha256"] for r in rows}:
        raise ValueError("original sealed identity/hash overlap")
    extra += [base.prior.POOL_META] + [base.PREP / n for n in
        ("public_proof.json", "train_sources.json", "development_sources.json")]
    base.verify_public(rows)
    manifests = []
    for root in (base.FEATURES, previous.additional.FEATURES):
        if base.file_hash(root / "manifest.json") != q.read(root / "result.json")["manifest_sha256"]:
            raise ValueError("completed frozen-parent feature manifest changed")
        manifests.append(q.read(root / "manifest.json")); extra.append(root / "manifest.json")
    if set(manifests[0]) & set(manifests[1]):
        raise ValueError("old and new source caches overlap")
    manifest = {r["id"]: (manifests[0] if r["id"] in manifests[0] else manifests[1])[r["id"]] for r in rows}
    producer = make_producer(destination, "reviewed_teacher_draft_target_pilot" if pilot else
        "expanded_whole_summary_TRAIN_from_reviewed_drafts", roots, extra,
        {"pilot": pilot, "old_training_cases": 252, "additional_population": 512,
            "additional_accepted": len(accepted), "additional_rejected": len(rejected)})
    codec = previous.released.load_codec(device="cpu").eval().requires_grad_(False)
    if previous.released.module_hash(codec) != CODEC_HASH:
        raise ValueError("original frozen codec required")
    facade = previous.BlockwiseCodec(codec, 4)
    targets = []
    selected = chosen if pilot else accepted
    repeats = {selected[0]["id"], max(selected, key=lambda r: len(labels._gpt.encode(r["text"], add_special_tokens=False)))["id"]}
    for index, label in enumerate(selected):
        words = word_ids(label["text"])
        packed = torch.tensor([words], dtype=torch.long)
        if label["id"] in retained:
            old = retained[label["id"]]
            if old["text"] != label["text"] or old["word_ids"] != words:
                raise ValueError("pilot target changed")
            codes = torch.tensor([old["concept_ids"]], dtype=torch.long)
        else:
            codes = facade.encode(packed)
        if codes.shape != (1, sum(facade.vq.levels)):
            raise ValueError("complete concept canvas required")
        if (pilot or label["id"] in repeats) and not torch.equal(codes, facade.encode(packed)):
            raise ValueError("target encoding did not repeat exactly")
        targets.append({"id": label["id"], "text": label["text"], "word_ids": words, "concept_ids": codes[0].tolist()})
        if (index+1) % 32 == 0 or index+1 == len(selected):
            print(json.dumps({"encoded_new_targets": index+1, "total": len(selected)}), flush=True)
    if previous.released.module_hash(codec) != CODEC_HASH or any(p.grad is not None for p in codec.parameters()):
        raise ValueError("codec changed during target preparation")
    base.verify(producer)
    if pilot:
        base.write_json(PILOT / "targets.json", targets)
        result = {"status": "systems_reviewed_draft_target_encoding_passed_not_quality", "cases": 4,
            "targets_sha256": base.file_hash(PILOT / "targets.json"), "codec_hash_unchanged": CODEC_HASH,
            "task_accuracy": None, "producer_fingerprint": producer["fingerprint"]}
    else:
        by_id = {r["id"]: r for r in targets}
        for row in rows[252:]:
            row.update({k: by_id[row["id"]][k] for k in ("word_ids", "concept_ids")})
        if rows[:252] != old_parts["train"]:
            raise ValueError("original252 word/concept targets changed")
        for name, value in (("cases.json", {"train": rows}), ("manifest.json", manifest),
                ("judgment_cache.json", cache), ("teacher_failures.json", rejected),
                ("accepted_labels.json", accepted)):
            base.write_json(DEST / name, value)
        result = {"status": "expanded_whole_summary_TRAIN_data_prepared", "cases": len(rows),
            "old_training_cases": 252, "additional_population": 512, "additional_accepted": len(accepted),
            "additional_rejected": len(rejected), "additional_accepted_by_bin": [sum(r["bin"] == b for r in accepted) for b in range(4)],
            "all_first_decisions_replayed": True, "canonical_judgments": len(cache), "prior930_exact": True,
            "codec_hash_unchanged": CODEC_HASH, "max_source_frame": max(r["source_frame"] for r in rows),
            "source_disjoint": True, "task_accuracy": None, "student_trained": False,
            "final_test_opened": False, "speed_tested": False,
            "files": {name: base.file_hash(DEST / name) for name in ("cases.json", "manifest.json", "judgment_cache.json", "teacher_failures.json", "accepted_labels.json")},
            "producer_fingerprint": producer["fingerprint"]}
    base.write_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["pilot", "prepare"])
    encode(parser.parse_args().stage == "pilot")
