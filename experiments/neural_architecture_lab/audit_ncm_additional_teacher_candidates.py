"""Replay fixed TRAIN-label first responses without making semantic judgments."""
import json
import os
from pathlib import Path

from experiments.structure_net import ncm_additional_teacher_pilot as run

DEST = run.ROOT / "20261024_additional64_teacher_raw_audit_v1"
CACHE = run.ROOT / "20261024_lm1b_native_word_readout_grade_v1/judgment_cache.json"
CACHE_SHA = "d27baaf96d563d2528263d002861754ddd98f65bc458d64fc80bc3659a171b28"


def replay_batch(root, index, rows):
    read = run.data.reservation.read
    saved = read(root / "batches" / f"batch_{index:04d}.json")
    expected = run.payload(rows)
    if saved["case_ids"] != [r["id"] for r in rows] or saved["index"] != index or not 1 <= len(saved["attempts"]) <= 3:
        raise ValueError("exact fixed complete batch and bounded attempts required")
    texts, valid_indices = [], []
    for i, path in enumerate(saved["attempts"]):
        if path != str(root / "attempts" / f"batch_{index:04d}_{i}.json"):
            raise ValueError("contiguous original raw attempts required")
        raw = read(path)
        if raw["request"] != expected:
            raise ValueError("exact full-source teacher request differs")
        valid = False
        try:
            response = raw["response"]
            if raw["http_status"] == 200 and response["model"] == run.MODEL and response["choices"][0]["finish_reason"] == "stop":
                texts = run.parse(response["choices"][0]["message"]["content"], len(rows))
                valid = True
        except (KeyError, TypeError, ValueError):
            pass
        if raw["valid"] is not valid:
            raise ValueError("raw response validity replay differs")
        if valid:
            valid_indices.append(i)
    if (valid_indices != [len(saved["attempts"]) - 1] or saved["status"] != "valid" or saved["texts"] != texts):
        raise ValueError("exact first valid response required; later rerolls are forbidden")
    return dict(zip(saved["case_ids"], texts))


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or DEST.exists():
        raise ValueError("fresh CPU raw-response audit required")
    read = run.data.reservation.read
    run.native.vendor.corpus.quality.dependencies()
    if (read(run.WORKFLOW / "tool_terminal_receipt.json")["exit_code"] != 0
            or [s["stage"] for s in read(run.WORKFLOW / "result.json")["stages"]] != ["prepare", "pilot", "generate"]
            or any(s["exit_code"] != 0 for s in read(run.WORKFLOW / "result.json")["stages"])):
        raise ValueError("all actual generation child and whole-tool exits required")
    for root in (run.PREP, run.PILOT, run.GENERATE):
        run.data.reservation.verify(read(root / "producer.json"))
        if read(root / "worker_terminal_receipt.json")["exit_code"] != 0:
            raise ValueError("actual original worker completion required")
    rows = read(run.PREP / "cases.json")
    if rows != run.select(read(run.data.DEST / "cases.json")):
        raise ValueError("fixed64 training-only population differs")
    encoding, tokenizer = run.data.storage_encoding(), run.native.tokenizer()
    ids = read(run.PREP / "pilot_ids.json")
    selected = [r for r in rows if r["id"] in ids]
    prior = {}
    for i, batch in enumerate(run.batches(selected, encoding)):
        prior.update(replay_batch(run.PILOT, i, batch))
    texts = dict(prior)
    remaining = [r for r in rows if r["id"] not in prior]
    for i, batch in enumerate(run.batches(remaining, encoding)):
        texts.update(replay_batch(run.GENERATE, i, batch))
    labels = read(run.GENERATE / "labels.json")
    pilot = {r["id"]: r for r in read(run.PILOT / "labels.json")}
    if [r["id"] for r in labels] != [r["id"] for r in rows] or len(texts) != 64:
        raise ValueError("complete ordered64 candidate denominator required")
    for label, source in zip(labels, rows):
        lexical = tokenizer.encode(label["target_text"], add_special_tokens=False)
        stored = len(encoding.encode(label["target_text"], disallowed_special=()))
        fits = bool(label["target_text"].strip()) and 0 < stored <= 128 and len(lexical) <= 126 and not set(lexical) & {0, 101}
        word_ids = [101, *lexical, 101] + [0] * (126 - len(lexical)) if fits else None
        if (any(label[k] != v for k, v in source.items()) or label["target_text"] != texts[label["id"]]
                or not label["teacher_valid"] or label["semantic_qualified"]
                or label["stored_tokens"] != stored or label["native_lexical_tokens"] != len(lexical)
                or label["capacity_fit"] is not fits or label["native_word_ids"] != word_ids
                or label["id"] in pilot and label != pilot[label["id"]]):
            raise ValueError("first candidate, full native capacity or retained pilot parity differs")
    state = read(run.GENERATE / "result.json")
    if (state["cases"] != 64 or state["valid"] != 64 or state["capacity_fit"] != sum(r["capacity_fit"] for r in labels)
            or state["labels_sha256"] != run.file_hash(run.GENERATE / "labels.json")
            or state["student_training_admitted"] or not state["semantic_qualification_pending"]
            or state["canonical_semantic_cache_changed"] or run.file_hash(CACHE) != CACHE_SHA):
        raise ValueError("candidate-only result or unchanged current canonical cache differs")
    artifacts = {str(p): run.file_hash(p) for root in (run.PREP, run.PILOT, run.GENERATE, run.WORKFLOW)
        for p in root.rglob("*.json") if "producer_source" not in p.parts}
    artifacts[str(Path(__file__))] = run.file_hash(__file__)
    artifacts[str(CACHE)] = CACHE_SHA
    result = {"status": "all64_training_label_first_responses_replayed_semantic_quality_unmeasured", "cases": 64,
        "valid_firsts": 64, "capacity_fit": sum(r["capacity_fit"] for r in labels), "retained_pilot_exact": True,
        "first_response_replay_exact": True, "stored_tokens_min": min(r["stored_tokens"] for r in labels),
        "stored_tokens_max": max(r["stored_tokens"] for r in labels),
        "native_lexical_tokens_max": max(r["native_lexical_tokens"] for r in labels),
        "task_accuracy": None, "semantic_qualification_pending": True, "student_training_admitted": False,
        "canonical_semantic_cache_changed": False, "selected_final_sources_opened": False, "artifacts": artifacts}
    DEST.mkdir()
    run.write_json(DEST / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}), flush=True)


if __name__ == "__main__":
    main()
