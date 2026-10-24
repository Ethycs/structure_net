"""TRAIN-only mechanism diagnosis of the registered failed word checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import torch

from experiments.structure_net import ncm_reviewed_draft_recovery as recovery
from experiments.structure_net.ncm_device_geometry import set_source_frame
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_native_label_quality import dependencies

ROOT = recovery.ROOT
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-reviewed-word-train-diagnostic-plan.md")
HYPOTHESIS = "ncm-reviewed-word-train-mechanism-v1"
PILOT = ROOT / "20261023_reviewed_word_train_diagnostic_pilot_v1"
DEST = ROOT / "20261023_reviewed_word_train_diagnostic_v1"
WORKFLOW = ROOT / "20261023_reviewed_word_train_diagnostic_workflow_v1"
ARCHIVE = ROOT / "20261023_reviewed_draft_history_results_terminal_v2"
OLD_DIAGNOSTIC = ROOT / "20261023_word_final_generation_gap_v1"
MODULE = "experiments.structure_net.ncm_reviewed_word_train_diagnostic"
CHECKPOINT = "e608f416e2915204a56d5f892d27087dc5cc2ff0e7b0b4cc24dbc44fe272acb3"
ARMS = ("actual_history", "correct_history")


def selection(rows, original_ids, old_selection):
    by_id = {row["id"]: row for row in rows}
    old_ids = [row["id"] for row in old_selection]
    if (len(rows) != 690 or len(by_id) != 690 or len(original_ids) != 252
            or len(old_ids) != 16 or len(set(old_ids)) != 16
            or not set(old_ids) <= set(original_ids) <= set(by_id)):
        raise ValueError("complete690 TRAIN rows and the exact old16 original identities required")
    selected = [{**by_id[key], "diagnostic_group": "original_16"} for key in old_ids]
    additional = [row for row in rows if row["id"] not in original_ids]
    for band in range(4):
        part = [row for row in additional if 0 <= row["source_cl100k_tokens"] <= 2048
                and min(row["source_cl100k_tokens"] // 512, 3) == band]
        if len(part) < 4:
            raise ValueError("four additional TRAIN examples per fixed length band required")
        chosen = sorted(part, key=lambda row: text_hash(HYPOTHESIS + ":" + row["id"]))[:4]
        selected.extend({**row, "diagnostic_group": "additional_16"} for row in chosen)
    if len({row["id"] for row in selected}) != 32:
        raise ValueError("exactly32 unique TRAIN examples required")
    return selected


def counts(actual, target):
    if actual.shape != target.shape or target.ndim != 2:
        raise ValueError("matching full word frames required")
    mask = target.ne(50257)
    if not mask.any():
        raise ValueError("nonempty TRAIN target required")
    return {"correct": int(((actual == target) & mask).sum()), "total": int(mask.sum())}


def aggregate(rows):
    result = {}
    for group in ("all_32", "original_16", "additional_16"):
        chosen = rows if group == "all_32" else [row for row in rows if row["diagnostic_group"] == group]
        arms = {}
        for arm in ARMS:
            total = sum(row["arms"][arm]["total"] for row in chosen)
            correct = sum(row["arms"][arm]["correct"] for row in chosen)
            arms[arm] = {"correct": correct, "total": total,
                "token_agreement": correct / total if total else None,
                "token_mean_CE": sum(row["arms"][arm]["CE"] * row["arms"][arm]["total"] for row in chosen) / total if total else None,
                "exact_target_texts": sum(row["arms"][arm]["exact_target_text"] for row in chosen)}
        agreement = {key: sum(row["own_gold_agreement"][key] for row in chosen) for key in ("correct", "total")}
        agreement["fraction"] = agreement["correct"] / agreement["total"] if agreement["total"] else None
        levels = [{"level": level,
            "active_correct": sum(row["concepts"][level]["active_correct"] for row in chosen),
            "active_total": sum(row["concepts"][level]["active_total"] for row in chosen)} for level in range(8)]
        result[group] = {"cases": len(chosen), "arms": arms, "own_gold_agreement": agreement, "concepts": levels}
    return result


def inputs_and_selection():
    dependencies(); recovery.configure(); recovery.evaluation.configure()
    study = recovery.study
    campaign = study.q.read(ARCHIVE / "campaign_results.json")
    receipt = study.q.read(ARCHIVE / (study.HYPOTHESIS + "_registry_readback.json"))
    if (campaign["classification"] != "generated_history_development_quality_failed"
            or campaign["training"]["checkpoint_sha256"] != CHECKPOINT
            or receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != file_hash(ARCHIVE / "campaign_results.json")
            or receipt["record"]["confirmation_status"] != campaign["classification"]
            or file_hash(study.TRAIN / "model.pt") != CHECKPOINT
            or campaign["final_test_opened"]):
        raise ValueError("registered unchanged failed development model required")
    rows, manifest = study.require_data()
    original = study.q.read(study.data.DATA / "cases.json")["train"]
    old_selection = study.q.read(OLD_DIAGNOSTIC / "selection.json")
    chosen = selection(rows, {row["id"] for row in original}, old_selection)
    old = {row["id"]: row for row in study.q.read(OLD_DIAGNOSTIC / "case_measurements.json")}
    for row in chosen[:16]:
        if old[row["id"]]["target_word_ids"] != row["word_ids"]:
            raise ValueError("old16 comparison must retain identical complete TRAIN targets")
    return chosen, manifest, study.histories_for(rows)


@torch.no_grad()
def measure(forecaster, branch, row, manifest, histories, tokenizer):
    study = recovery.study
    batch = study.source_batch(study.source_view(row), manifest)
    for net in (forecaster, branch):
        set_source_frame(net.generator, row["source_frame"], maximum=8192)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        history = study.core.own_history(forecaster, batch["features"], batch["mask"], batch["source_ids"])
    actual_codes = history["coarse_codes"].cpu()
    if actual_codes[0].tolist() != histories[row["id"]]["coarse_codes"]:
        raise ValueError("fresh forecaster rollout differs from the actual TRAIN cache")
    target = torch.tensor([row["word_ids"]], dtype=torch.long, device="cuda")
    active_words = [word for word in row["word_ids"] if word != 50257]
    if not active_words or active_words[-1] != 50256 or tokenizer.decode(active_words[:-1], skip_special_tokens=False) != row["target_text"]:
        raise ValueError("complete original GPT2 target does not decode exactly")
    gold_codes = torch.tensor([row["concept_ids"][:608]], dtype=torch.long, device="cuda")
    gold = study.core.conditioning_from_codes(branch.generator.vqvae_local[0], gold_codes)
    arms, word_ids = {}, {}
    for arm, conditioning in (("actual_history", history["word_conditioning"]), ("correct_history", gold)):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = study.core.word_logits(branch, batch["features"], batch["mask"], conditioning, batch["source_ids"])
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite diagnostic logits")
        loss = float(study.core.word_loss(logits, target))
        raw = logits.argmax(-1).cpu()
        text, end = recovery.evaluation.reference.decode_words(raw[0].tolist(), tokenizer)
        arms[arm] = {**counts(raw, target.cpu()), "CE": loss, "raw_word_ids": raw[0].tolist(),
            "text": text, "global_EOS_position": end if end < 192 else None,
            "exact_target_text": text == row["target_text"]}
        word_ids[arm] = raw
        del logits
    active_blocks = target.cpu().reshape(1, 4, 48).ne(50257).any(-1)
    concepts, position = [], 0
    for level, width in enumerate(forecaster.generator.levels[:-1]):
        mask = active_blocks.repeat_interleave(width // 4, dim=1)
        expected = gold_codes.cpu()[:, position:position + width]
        actual = actual_codes[:, position:position + width]
        concepts.append({"level": level, "active_correct": int(((actual == expected) & mask).sum()),
            "active_total": int(mask.sum())})
        position += width
    return {"id": row["id"], "source_id": row["source_id"], "source_sha256": row["source_sha256"],
        "diagnostic_group": row["diagnostic_group"], "source_frame": row["source_frame"],
        "source_cl100k_tokens": row["source_cl100k_tokens"], "target_text": row["target_text"],
        "target_word_ids": row["word_ids"], "fresh_history_matches_training_cache": True,
        "actual_coarse_codes": actual_codes[0].tolist(), "concepts": concepts, "arms": arms,
        "own_gold_agreement": counts(word_ids["actual_history"],
            word_ids["correct_history"].masked_fill(target.cpu().eq(50257), 50257))}


def run(pilot=False):
    chosen, manifest, histories = inputs_and_selection()
    study = recovery.study
    study.data.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    first, longest = chosen[0], sorted(chosen, key=lambda row: (-len(row["source_ids"]), row["id"]))[0]
    selected = [first, longest] if pilot else chosen
    root = PILOT if pilot else DEST
    if not pilot:
        study.data.terminal(PILOT)
    inputs = [ARCHIVE / "campaign_results.json", ARCHIVE / (study.HYPOTHESIS + "_registry_readback.json"),
        OLD_DIAGNOSTIC / "selection.json", OLD_DIAGNOSTIC / "case_measurements.json"]
    inputs += [study.TRAIN / name for name in ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json")]
    inputs += [study.CACHE / name for name in ("producer.json", "result.json", "histories.json", "worker_terminal_receipt.json")]
    if not pilot:
        inputs += [PILOT / name for name in ("producer.json", "result.json", "case_measurements.json", "worker_terminal_receipt.json")]
    for row in selected:
        entry = manifest[row["id"]]
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("frozen selected source features changed")
        inputs.append(Path(entry["path"]))
    p = study.producing(root, "TRAIN_only_word_mechanism_pilot" if pilot else "TRAIN_only_word_mechanism_diagnostic",
        inputs + [Path(__file__), PLAN], {"updates": 0, "training_updates": 0,
            "model_checkpoint_updates": 32768, "diagnostic_cases": len(selected), "TRAIN_only": True,
            "arms": ARMS, "semantic_quality_measured": False, "new_judge_requests": 0})
    p["hypothesis_id"] = HYPOTHESIS
    p["config"].update(hypothesis_id=HYPOTHESIS, source_only=False, oracle_target_codes=True,
        correct_history_diagnostic_only=True, teacher_forced_training_auxiliary=False)
    for index, path in enumerate((Path(__file__), PLAN)):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"train_diagnostic_{index}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = study.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    write_json(root / "selection.json", [{key: row[key] for key in
        ("id", "source_id", "source_sha256", "diagnostic_group", "source_cl100k_tokens")} for row in selected])
    forecaster, branch, codec, _ = recovery.evaluation.checkpoint()
    forecaster.eval().requires_grad_(False); branch.eval().requires_grad_(False)
    before = study.parameter_hash(forecaster), study.parameter_hash(branch)
    codec_before = recovery.evaluation.reference.core.released.module_hash(codec)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(study.data.GPT2, local_files_only=True)
    torch.cuda.reset_peak_memory_stats()
    retained = {} if pilot else {row["id"]: row for row in study.q.read(PILOT / "case_measurements.json")}
    measurements = []
    for row in selected:
        item = measure(forecaster, branch, row, manifest, histories, tokenizer)
        if pilot and measure(forecaster, branch, row, manifest, histories, tokenizer) != item:
            raise ValueError("diagnostic pilot must repeat exactly")
        if row["id"] in retained and item != retained[row["id"]]:
            raise ValueError("full diagnostic must reproduce retained pilot after fresh model load")
        measurements.append(retained.get(row["id"], item))
        write_json(root / "case_measurements.json", measurements)
        print(json.dumps({"measured": len(measurements), "total": len(selected)}), flush=True)
    if (before != (study.parameter_hash(forecaster), study.parameter_hash(branch))
            or recovery.evaluation.reference.core.released.module_hash(codec) != codec_before
            or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("read-only parameter/codec identity or memory gate failed")
    study.data.verify(p)
    result = {"status": "systems_TRAIN_diagnostic_passed" if pilot else "TRAIN_word_mechanism_measured",
        "cases": len(measurements), "components": aggregate(measurements),
        "model_checkpoint_sha256": CHECKPOINT, "parameters_and_codec_unchanged": True,
        "fresh_histories_match_training_cache": True, "pilot_retained_exactly": not pilot,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "task_accuracy": None,
        "complete_summary_quality_measured": False, "final_test_opened": False,
        "speed_tested": False, "producer_fingerprint": p["fingerprint"],
        "measurements_sha256": file_hash(root / "case_measurements.json")}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def workflow():
    recovery.configure()
    driver, study = recovery.word_workflow, recovery.study
    if any(path.exists() for path in (WORKFLOW, PILOT, DEST)):
        raise ValueError("fresh diagnostic workflow and both stage roots required")
    driver.DEST = WORKFLOW
    p = study.data.freeze(WORKFLOW, "supervise_fixed_TRAIN_mechanism_diagnostic", [Path(__file__), PLAN],
        {"hypothesis_id": HYPOTHESIS, "model_updates": 0, "new_judge_requests": 0,
         "task_accuracy": None, "final_test_opened": False})
    p["hypothesis_id"] = HYPOTHESIS
    for index, path in enumerate((Path(__file__), PLAN)):
        p["source_hashes"][str(path)] = file_hash(path)
        (WORKFLOW / "producer_source" / f"diagnostic_workflow_{index}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = study.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(WORKFLOW / "producer.json", p)
    (WORKFLOW / "stage_receipts").mkdir()
    journal = {"status": "running", "stages": [], "final_test_opened": False}
    try:
        for stage, root in (("pilot", PILOT), ("measure", DEST)):
            driver.run_stage(stage, [sys.executable, "-u", "-m", MODULE, stage], root, journal, p)
        journal.update(status="TRAIN_diagnostic_completed_not_task_quality", active_stage=None,
            result_sha256=file_hash(DEST / "result.json"))
        write_json(WORKFLOW / "result.json", journal)
    except Exception as error:
        journal.update(status="stopped_failure_preserved", error_type=type(error).__name__, error=str(error))
        write_json(WORKFLOW / "failure.json", journal)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["pilot", "measure", "workflow", "selection"])
    stage = parser.parse_args().stage
    if stage == "selection":
        chosen, _, _ = inputs_and_selection()
        print(json.dumps({"cases": len(chosen), "old_cases": sum(row["diagnostic_group"] == "original_16" for row in chosen)}))
    elif stage == "workflow":
        workflow()
    else:
        run(stage == "pilot")
