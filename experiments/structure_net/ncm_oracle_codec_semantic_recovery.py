"""Recover complete prior caches and stopped raw judgments without semantic rerolls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from experiments.structure_net import ncm_oracle_codec_semantic as study
from experiments.structure_net.ncm_memory_data import file_hash, write_json
from experiments.structure_net.ncm_paper_data import freeze, verify

PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-oracle-codec-semantic-cache-recovery.md")
PREP = study.ROOT / "20261023_oracle_codec_semantic_recovery_prepare_v1"
PILOT = study.ROOT / "20261023_oracle_codec_semantic_recovery_pilot_v1"
GRADE = study.ROOT / "20261023_oracle_codec_semantic_recovery_grade_v1"
OLD_PILOT, OLD_GRADE = study.PILOT, study.GRADE
CANONICAL = [study.ROOT / n / "judgment_cache.json" for n in
    ("20261023_revised_labels_main_critique_v2", "20261023_wide_labels_main_audit_v1", "20261023_isolated_quality_grade_v1")]
HYPOTHESIS = "ncm-oracle-codec-semantic-quality-recovery-v1"
_check_prepared, _producing = study.check_prepared, study.producing


def merge_canonical(caches):
    merged = {}
    for cache in caches:
        for key, entry in cache.items():
            if key in merged and merged[key] != entry:
                raise ValueError("conflicting canonical first judgments")
            merged.setdefault(key, entry)
    return merged


def recover_request(packet, role, request_root):
    """Rebuild only responses actually saved by the terminated process."""
    attempts = sorted((request_root / "attempts").glob("*.json"))
    if not attempts:
        return None
    paths = []
    for path in attempts:
        raw = study.q.read(path)
        paths.append(str(path))
        if raw["valid"]:
            row = {"case_id": packet["case_id"], "packet_sha256": study.q.digest(packet),
                "status": "valid", "judge": study.q.JUDGES[role],
                "evaluations": study.q.parse(raw["response"]["choices"][0]["message"]["content"], packet),
                "attempts": paths}
            study.first.audit_row(packet, role, row)
            return row
    if len(paths) == 3:
        row = {"case_id": packet["case_id"], "packet_sha256": study.q.digest(packet),
            "status": "invalid", "judge": study.q.JUDGES[role], "attempts": paths}
        study.first.audit_row(packet, role, row)
        return row
    raise ValueError("partially invalid saved transport needs explicit continuation")


def prepare():
    _, packets = _check_prepared()
    if study.q.read(OLD_GRADE / "worker_terminal_receipt.json")["exit_code"] != 143:
        raise ValueError("stopped parent terminal receipt required")
    if study.q.read(OLD_GRADE / "scientific_invalidation.json")["classification"] != "invalid_incomplete_prior_first_judgment_cache":
        raise ValueError("preserved parent invalidation required")
    caches = [study.q.read(p) for p in CANONICAL]
    canonical = merge_canonical(caches)
    if len(canonical) != 606:
        raise ValueError("all606 disjoint earlier first judgments required")
    for key, entry in canonical.items():
        if key != study.first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("canonical profile/role mismatch")
        study.first.audit_row(entry["packet"], entry["role"], entry["row"])
    merged = dict(canonical)
    unique_packets = {study.first.packet_key(p, role): (role, p)
        for role in ("secondary", "primary") for rows in packets.values() for p in rows}
    duplicates, recovered, uncertain = [], [], set()
    for old_root in (OLD_PILOT, OLD_GRADE):
        for key, (role, packet) in unique_packets.items():
            request_root = old_root / "requests" / role / key
            if not request_root.exists():
                continue
            row = recover_request(packet, role, request_root)
            if row is None:
                uncertain.add(key)
                continue
            entry = {"packet": packet, "role": role, "row": row, "origin": str(request_root)}
            if key in merged:
                if merged[key] != entry:
                    if key not in canonical:
                        raise ValueError("unexpected noncanonical duplicate valid reply")
                    duplicates.append({"key": key, "canonical_origin": merged[key]["origin"],
                        "ignored_later_origin": str(request_root), "ignored_attempts": row["attempts"],
                        "reason": "earlier first judgment wins regardless of outcome"})
            else:
                merged[key] = entry
                recovered.append(key)
    uncertain -= merged.keys()
    source_files = [Path(__file__), PLAN, Path(study.__file__), study.PLAN]
    inputs = list(CANONICAL) + [OLD_GRADE / "scientific_invalidation.json", OLD_GRADE / "worker_terminal_receipt.json"]
    for root in (OLD_PILOT, OLD_GRADE):
        inputs += [p for p in root.rglob("*.json") if "producer_source" not in p.parts]
    inputs += [study.PREP / n for n in ("producer.json", "result.json", "cases.json", "exact_packets.json", "public_payload_proof.json")]
    p = freeze(PREP, "recover_all_canonical_and_stopped_first_judgments", inputs + source_files,
        {"hypothesis_id": HYPOTHESIS, "canonical_first_judgments": 606,
         "recovered_first_judgments": len(recovered), "ignored_later_duplicates": len(duplicates),
         "possibly_consumed_transport_attempts": len(uncertain), "rubric": study.q.RUBRIC,
         "payloads_unchanged": True, "task_accuracy": None})
    p["hypothesis_id"] = HYPOTHESIS
    for i, path in enumerate(source_files):
        p["source_hashes"][str(path)] = file_hash(path)
        (PREP / "producer_source" / f"recovery_{i}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = study.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(PREP / "producer.json", p)
    write_json(PREP / "judgment_cache.json", merged)
    write_json(PREP / "ignored_later_duplicates.json", duplicates)
    write_json(PREP / "possibly_consumed_transport_keys.json", sorted(uncertain))
    for name in ("exact_packets.json", "public_payload_proof.json"):
        (PREP / name).write_bytes((study.PREP / name).read_bytes())
    verify(p)
    result = {"status": "recovered_canonical_first_judgments_ready", "canonical_first_judgments": 606,
        "recovered_unique_judgments": len(recovered), "ignored_later_duplicates": len(duplicates),
        "possibly_consumed_transport_attempts": len(uncertain), "task_accuracy": None,
        "producer_fingerprint": p["fingerprint"], "hashes": {name: file_hash(PREP / name) for name in
        ("judgment_cache.json", "ignored_later_duplicates.json", "possibly_consumed_transport_keys.json", "exact_packets.json", "public_payload_proof.json")}}
    write_json(PREP / "result.json", result)
    print(json.dumps(result), flush=True)


def limited_request(packet, role, root):
    import requests
    key = next(x.split("=", 1)[1].strip().strip('"').strip("'")
        for x in Path(".env").read_text().splitlines() if x.startswith("LITELLM_KEY="))
    payload = {"model": study.q.JUDGES[role], "temperature": 0, "max_tokens": 4096,
        "messages": [{"role": "system", "content": study.q.RUBRIC},
                     {"role": "user", "content": json.dumps(packet, ensure_ascii=False)}]}
    uncertain = set(study.q.read(PREP / "possibly_consumed_transport_keys.json"))
    limit = 2 if study.first.packet_key(packet, role) in uncertain else 3
    attempts = []
    for n in range(limit):
        path = root / "attempts" / f"{packet['case_id']}_{n}.json"
        if path.exists():
            raise ValueError("attempt overwrite forbidden")
        start = time.monotonic()
        raw = {"attempt": n, "request": payload, "request_sha256": study.q.digest(payload),
            "http_status": None, "response": None, "possibly_consumed_prior_transport_attempt": limit == 2}
        try:
            response = requests.post("https://central-dev.zt:4000/v1/chat/completions",
                headers={"Authorization": "Bearer " + key}, json=payload,
                verify="/home/rabbit/.config/lakefs/caddy-root.crt", timeout=180)
            raw.update(http_status=response.status_code, response=response.json())
            response.raise_for_status()
            if raw["response"].get("model") != study.q.JUDGES[role]:
                raise ValueError("returned judge identity differs")
            evaluations = study.q.parse(raw["response"]["choices"][0]["message"]["content"], packet)
            raw["valid"] = True
        except Exception as error:
            raw.update(valid=False, error_type=type(error).__name__)
        raw["seconds"] = time.monotonic() - start
        write_json(path, raw)
        attempts.append(str(path))
        if raw["valid"]:
            return {"case_id": packet["case_id"], "packet_sha256": study.q.digest(packet),
                "status": "valid", "judge": study.q.JUDGES[role], "evaluations": evaluations, "attempts": attempts}
    # The existing audit requires three exhausted actual attempts. A two-attempt
    # failure with uncertain history is a systems stop, never a fabricated row.
    if limit == 2:
        raise ValueError("two recovery attempts exhausted after possibly consumed prior call")
    return {"case_id": packet["case_id"], "packet_sha256": study.q.digest(packet),
        "status": "invalid", "judge": study.q.JUDGES[role], "attempts": attempts}


def check_prepared():
    result = study.q.read(PREP / "result.json")
    study.teacher.training.data.terminal(PREP)
    for name, expected in result["hashes"].items():
        if file_hash(PREP / name) != expected:
            raise ValueError("recovery preparation changed")
    return _check_prepared()


def producing(root, stage, inputs, config):
    extra = [PREP / n for n in ("producer.json", "result.json", "judgment_cache.json",
        "ignored_later_duplicates.json", "possibly_consumed_transport_keys.json", "exact_packets.json",
        "public_payload_proof.json", "worker_terminal_receipt.json")]
    extra += [Path(__file__), PLAN]
    p = _producing(root, stage, list(inputs) + extra, {**config, "canonical_first_judgments": 606,
        "recovered_cache": str(PREP), "unknown_transport_keys_max_future_attempts": 2})
    for path in (Path(__file__), PLAN):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / ("recovery_" + path.name)).write_bytes(path.read_bytes())
    p["fingerprint"] = study.q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def configure():
    study.PILOT, study.GRADE, study.PLAN, study.HYPOTHESIS = PILOT, GRADE, PLAN, HYPOTHESIS
    study.check_prepared, study.producing = check_prepared, producing
    study.prior_cache = lambda: study.q.read(PREP / "judgment_cache.json")
    study.q.request = limited_request


def main():
    study.teacher.old.dependencies()
    study.teacher.old.profile.install()
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["prepare", "pilot", "grade"])
    stage = parser.parse_args().stage
    if stage == "prepare":
        prepare()
    else:
        configure()
        study.grade(stage == "pilot")


if __name__ == "__main__":
    main()
