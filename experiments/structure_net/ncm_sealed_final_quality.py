"""Conditional sealed-source preparation and complete two-judge NCM evaluation."""
import argparse
import json
from pathlib import Path

import torch

from experiments.structure_net import ncm_generated_history_study as study
from experiments.structure_net import ncm_generated_history_eval as candidate
from experiments.neural_architecture_lab import finalize_ncm_generated_history_word as development
from experiments.structure_net import ncm_word_final_data as base
from experiments.structure_net.ncm_device_geometry import set_source_frame
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json

q = study.q
ROOT = study.ROOT
HYPOTHESIS = "ncm-generated-history-final-quality-v1"
PLAN = Path("docs/07 - Status Reports/2026-10-23_ncm-sealed-final-quality-plan.md")
DATA = ROOT / "20261023_sealed_final_data_v1"
FEATURE_PILOT = ROOT / "20261023_sealed_final_feature_pilot_v1"
FEATURES = ROOT / "20261023_sealed_final_features_v1"
PILOT = ROOT / "20261023_sealed_final_inference_pilot_v1"
OUTPUTS = ROOT / "20261023_sealed_final_outputs_v1"
PACKETS = ROOT / "20261023_sealed_final_packets_v1"
GRADE = ROOT / "20261023_sealed_final_grade_v1"
SHORT_FILE = base.PREP / "sealed_test_sources.json"
LONG_FILE = base.prior.pool.POOL / "sealed_final_sources.json"
COUNTS = {"short_final": 256, "long_final": 64}
_base_producing = base.producing


def minimum(count):
    if type(count) is not int or count < 1:
        raise ValueError("positive complete cohort denominator required")
    return (95*count+99)//100


def qualifies(endpoint, count):
    return (endpoint is not None and endpoint["cases"] == count and endpoint["valid"] == count
        and endpoint["passed"] >= minimum(count))


def consensus(secondary, primary):
    count = len(secondary)
    if count < 1 or len({r["id"] for r in secondary}) != count:
        raise ValueError("complete unique secondary denominator required")
    for row in secondary + ([] if primary is None else primary):
        if (type(row["valid"]) is not bool or (row["valid"] and type(row["passed"]) is not bool)
                or (not row["valid"] and row["passed"] is not None)):
            raise ValueError("validity and complete-summary decision must agree")
    if primary is None:
        return {"cases": count, "valid_both": 0, "joint_passed": 0, "task_accuracy": None,
            "primary_admitted": False, "joint_accuracy_upper_bound":
                sum(r["passed"] is not False for r in secondary)/count,
            "qualified": False, "minimum_joint_passes": minimum(count), "wilson_95": None}
    if [r["id"] for r in primary] != [r["id"] for r in secondary]:
        raise ValueError("both judges must evaluate the same complete source order")
    valid = sum(a["valid"] is True and b["valid"] is True for a, b in zip(secondary, primary))
    passed = sum(a["passed"] is True and b["passed"] is True for a, b in zip(secondary, primary))
    upper = sum(a["passed"] is not False and b["passed"] is not False for a, b in zip(secondary, primary))
    return {"cases": count, "valid_both": valid, "joint_passed": passed,
        "task_accuracy": passed/count if valid == count else None, "primary_admitted": True,
        "joint_accuracy_lower_bound": passed/count, "joint_accuracy_upper_bound": upper/count,
        "qualified": valid == count and passed >= minimum(count), "minimum_joint_passes": minimum(count),
        "wilson_95": q.wilson(passed, count) if valid == count else None}


def require_admission():
    """Must run before ANY read/hash of the two sealed source files."""
    receipt = q.read(development.DEST / (study.HYPOTHESIS+"_registry_readback.json"))
    aggregate = q.read(development.DEST / "campaign_results.json")
    if (receipt["status"] != "registered_and_read_back"
            or receipt["record"]["campaign_sha256"] != file_hash(development.DEST / "campaign_results.json")
            or aggregate["classification"] != "generated_history_development_passed_final_quality_unproven"
            or not aggregate["development"]["development_gate"]):
        raise ValueError("both registered development gates must pass before sealed source access")
    for value in aggregate["development"]["results"].values():
        if not qualifies(value["secondary"], 64) or not qualifies(value["primary"], 64):
            raise ValueError("both judges must qualify both64-case development cohorts")
    paths = [study.TRAIN / "model.pt", study.TRAIN / "result.json", study.TRAIN / "producer.json",
        candidate.GRADE / "result.json", candidate.GRADE / "judgment_cache.json"]
    for path in paths:
        if aggregate["artifacts"].get(str(path)) != file_hash(path):
            raise ValueError("registered candidate or development evidence changed")
    study.require_forecaster()
    return aggregate


def producing(root, stage, inputs, config):
    require_admission()
    files = [Path(__file__), PLAN, Path(study.__file__), study.PLAN, Path(candidate.__file__),
        Path(study.core.__file__), Path("experiments/structure_net/ncm_device_geometry.py")]
    files += sorted(base.VENDOR.rglob("*.py"))
    inputs = list(inputs)+files+[development.DEST / "campaign_results.json",
        development.DEST / (study.HYPOTHESIS+"_registry_readback.json")]
    inputs += [study.TRAIN / name for name in ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json")]
    inputs += [study.bootstrap.TRAIN / name for name in ("producer.json", "result.json", "model.pt", "worker_terminal_receipt.json")]
    inputs += [candidate.GRADE / "judgment_cache.json"]
    p = _base_producing(root, stage, inputs, config)
    p["hypothesis_id"] = HYPOTHESIS
    p["config"].update(hypothesis_id=HYPOTHESIS, final_test_opened=True, source_only=True,
        oracle_target_codes=False, AR_fallback=False, output_repair=False, speed_tested=False,
        source_truncated=False, cohorts=COUNTS, joint_minimum={k: minimum(n) for k, n in COUNTS.items()},
        source_prediction=stage in ("sealed_composed_inference_pilot", "sealed_composed_final_outputs"))
    for index, path in enumerate(files):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"sealed_final_{index}_{path.name}").write_bytes(path.read_bytes())
    p["fingerprint"] = q.digest({k: v for k, v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def prepare():
    require_admission()
    base.prior.dependencies()
    short = q.read(SHORT_FILE)
    long = q.read(LONG_FILE)
    identity = base.prior.original_final_metadata(q.read(base.PREP / "public_proof.json"),
        q.read(base.PREP / "train_sources.json"), q.read(base.PREP / "development_sources.json"))
    sealed = q.read(base.prior.POOL_META)["sealed_final"]
    if len(short) != 256 or {r["id"]: r["source_sha256"] for r in short} != identity:
        raise ValueError("all original256sealed identities and source hashes required")
    metadata = {r["id"]: r for r in sealed}
    if len(long) != 64 or {r["id"] for r in long} != set(metadata):
        raise ValueError("all original64longer sealed identities required")
    for row in long:
        if any(row[key] != metadata[row["id"]][key] for key in base.prior.IDENTITY):
            raise ValueError("long sealed metadata identity differs")
    if [sum(r["bin"] == b for r in long) for b in range(4)] != [16]*4:
        raise ValueError("fixed16sources per long band required")
    pool_result = q.read(base.prior.pool.POOL / "result.json")
    if pool_result["files"].get(str(LONG_FILE)) != file_hash(LONG_FILE):
        raise ValueError("original long sealed source export changed")
    train, _ = study.require_data()
    dev = [r for rows in candidate.reference.sources().values() for r in rows]
    base.prior.check_disjoint(short, long)
    base.prior.check_disjoint([*short, *long], [*train, *dev])
    proof = base.verify_public([*short, *long])
    manifest = q.read(base.prior.ORIGINAL_DATA / "manifest.json")
    if manifest["source_session_overlap"] != 0 or manifest["eligible_input_overlap"] != 0:
        raise ValueError("original train/test partition exclusion proof differs")
    from transformers import AutoTokenizer
    import tiktoken
    native = AutoTokenizer.from_pretrained(base.MODEL_PATH, local_files_only=True)
    gpt = AutoTokenizer.from_pretrained(base.GPT2, local_files_only=True)
    gpt.model_max_length = 1_000_000
    storage = tiktoken.get_encoding("cl100k_base")
    inputs = [SHORT_FILE, LONG_FILE, base.prior.POOL_META, base.prior.pool.POOL / "result.json",
        base.PREP / "public_proof.json", base.PREP / "train_sources.json", base.PREP / "development_sources.json",
        base.prior.ORIGINAL_DATA / "manifest.json", base.PUBLIC_FILE,
        study.expanded.DEST / "cases.json", candidate.reference.SOURCES / "cases.json"]
    inputs += [base.MODEL_PATH / name for name in ("tokenizer.json", "tokenizer_config.json", "config.json")]
    p = producing(DATA, "sealed_complete_source_metadata_after_development_admission", inputs,
        {"cases": 320, "source_targets_read": False, "training": False, "system": base.SYSTEM})
    cases = {}
    for split, rows in (("short_final", short), ("long_final", long)):
        output = []
        for row in rows:
            context = native.apply_chat_template([{"role": "system", "content": base.SYSTEM},
                {"role": "user", "content": "<passage>\n"+row["source"]+"\n</passage>"}],
                add_generation_prompt=True, tokenize=False)
            encoded = native(context, add_special_tokens=False, return_offsets_mapping=True)
            direct = native.apply_chat_template([{"role": "system", "content": base.SYSTEM},
                {"role": "user", "content": "<passage>\n"+row["source"]+"\n</passage>"}], add_generation_prompt=True)
            if direct != encoded["input_ids"]:
                raise ValueError("canonical complete prompt differs")
            start = context.index("<passage>\n")+len("<passage>\n")
            if context[start:start+len(row["source"])] != row["source"]:
                raise ValueError("literal complete source boundary differs")
            raw = gpt(row["source"], add_special_tokens=False, return_offsets_mapping=True)
            if gpt.decode(raw["input_ids"], skip_special_tokens=False) != row["source"]:
                raise ValueError("complete source GPT2 round trip failed")
            alignment = base.overlaps(raw["offset_mapping"], encoded["offset_mapping"], start)
            output.append({**{key: row[key] for key in (*base.prior.IDENTITY, "source", "spans")},
                "split": split, "context_ids": direct, "source_ids": raw["input_ids"], "alignment": alignment,
                "alignment_sha256": q.digest(alignment), "native_prompt_sha256": q.digest(direct),
                "source_frame": base.source_frame(len(raw["input_ids"]), maximum=8192),
                "source_cl100k_tokens": len(storage.encode(row["source"], disallowed_special=()))})
        cases[split] = output
    all_rows = [r for rows in cases.values() for r in rows]
    first = all_rows[0]
    native_longest = sorted(all_rows, key=lambda r: (-len(r["context_ids"]), r["id"]))[0]
    gpt_longest = sorted(all_rows, key=lambda r: (-len(r["source_ids"]), r["id"]))[0]
    if first["id"] in (native_longest["id"], gpt_longest["id"]):
        raise ValueError("distinct first/longest final systems cases required")
    write_json(DATA / "cases.json", cases)
    write_json(DATA / "selection.json", {"feature_pilot_ids": [first["id"], native_longest["id"]],
        "inference_pilot_ids": [first["id"], gpt_longest["id"]]})
    write_json(DATA / "public_proof.json", {"public_file_sha256": base.PUBLIC_SHA256, "exact_sources": proof})
    base.verify(p)
    result = {"status": "sealed_final_metadata_prepared", "counts": {k: len(v) for k, v in cases.items()},
        "source_disjoint": True, "source_targets_read": False, "all_sources_complete": True,
        "max_native_prompt": max(len(r["context_ids"]) for r in all_rows),
        "max_GPT2_source": max(len(r["source_ids"]) for r in all_rows),
        "max_source_frame": max(r["source_frame"] for r in all_rows),
        "cases_sha256": file_hash(DATA / "cases.json"), "final_test_opened": True,
        "task_accuracy": None, "producer_fingerprint": p["fingerprint"]}
    write_json(DATA / "result.json", result)
    print(json.dumps(result), flush=True)


def sources():
    require_admission()
    base.terminal(DATA)
    result = q.read(DATA / "result.json")
    if file_hash(DATA / "cases.json") != result["cases_sha256"]:
        raise ValueError("fixed sealed final metadata changed")
    cases = q.read(DATA / "cases.json")
    if set(cases) != set(COUNTS) or any(len(cases[k]) != n for k, n in COUNTS.items()):
        raise ValueError("complete fixed256+64final cohorts required")
    if any(any(key in r for key in ("target", "target_text", "word_ids", "concept_ids", "text"))
            for rows in cases.values() for r in rows):
        raise ValueError("no target or candidate may enter final source metadata")
    return cases


def features(pilot=False):
    require_admission()
    base.DATA, base.PILOT, base.FEATURES = DATA, FEATURE_PILOT, FEATURES
    base.producing = producing
    base.feature_cache(pilot=pilot)


def generate(pilot=False):
    require_admission()
    base.require_gpu()
    torch.backends.cudnn.allow_tf32 = False
    base.terminal(FEATURES)
    cases = sources()
    all_rows = [r for split in COUNTS for r in cases[split]]
    pilot_ids = q.read(DATA / "selection.json")["inference_pilot_ids"]
    selected = [next(r for r in all_rows if r["id"] == i) for i in pilot_ids] if pilot else all_rows
    manifest = q.read(FEATURES / "manifest.json")
    if file_hash(FEATURES / "manifest.json") != q.read(FEATURES / "result.json")["manifest_sha256"]:
        raise ValueError("sealed frozen feature manifest changed")
    inputs = [DATA / name for name in ("producer.json", "result.json", "cases.json", "selection.json", "worker_terminal_receipt.json")]
    inputs += [FEATURES / name for name in ("producer.json", "result.json", "manifest.json", "worker_terminal_receipt.json")]
    for row in selected:
        entry = manifest[row["id"]]
        if file_hash(entry["path"]) != entry["sha256"]:
            raise ValueError("final source feature changed")
        inputs.append(Path(entry["path"]))
    retained = {}
    if not pilot:
        base.terminal(PILOT)
        if q.read(PILOT / "result.json")["status"] != "systems_sealed_composed_inference_passed":
            raise ValueError("actual maximum-frame inference/reload pilot required")
        inputs += [PILOT / name for name in ("producer.json", "result.json", "outputs.json", "worker_terminal_receipt.json")]
        retained = {r["id"]: r for r in q.read(PILOT / "outputs.json")}
    root = PILOT if pilot else OUTPUTS
    p = producing(root, "sealed_composed_inference_pilot" if pilot else "sealed_composed_final_outputs",
        inputs, {"cases": len(selected), "physical_gpu": 0, "single_deterministic_argmax": True})
    forecaster, branch, codec, state = candidate.checkpoint()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base.GPT2, local_files_only=True)
    before = study.parameter_hash(forecaster), study.parameter_hash(branch)
    frozen = candidate.reference.core.released.module_hash(codec)
    torch.cuda.reset_peak_memory_stats()
    outputs = []
    for row in selected:
        outputs.append(retained[row["id"]] if row["id"] in retained else candidate.predict(forecaster, branch, row, manifest, tokenizer))
        write_json(root / "outputs.json", outputs)
        if pilot or len(outputs) % 8 == 0 or len(outputs) == len(selected):
            print(json.dumps({"generated": len(outputs), "total": len(selected)}), flush=True)
    candidate.reference.require_outputs(outputs, selected)
    if pilot:
        forecaster_state = torch.load(study.bootstrap.TRAIN / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        with torch.no_grad():
            for net in (forecaster, branch):
                for value in net.state_dict().values(): value.zero_()
                net.generator._ncm_device_source_geometry = {}
        for net, saved in ((forecaster, forecaster_state), (branch, state)):
            set_source_frame(net.generator, saved["source_frame"], maximum=8192)
            net.load_state_dict(saved["model"], strict=True)
        if [candidate.predict(forecaster, branch, row, manifest, tokenizer) for row in selected] != outputs:
            raise ValueError("sealed source-only composed inference reload differs")
    if (before != (study.parameter_hash(forecaster), study.parameter_hash(branch))
            or candidate.reference.core.released.module_hash(codec) != frozen
            or torch.cuda.max_memory_allocated() > 8_000_000_000):
        raise ValueError("frozen model/codec or final inference memory gate failed")
    base.verify(p)
    result = {"status": "systems_sealed_composed_inference_passed" if pilot else "sealed_final_outputs_pending_grade",
        "cases": len(outputs), "pilot_outputs_retained_exactly": not pilot,
        "all_parameters_and_codec_unchanged": True, "outputs_sha256": file_hash(root / "outputs.json"),
        "final_test_opened": True, "task_accuracy": None, "speed_tested": False,
        "producer_fingerprint": p["fingerprint"]}
    write_json(root / "result.json", result)
    print(json.dumps(result), flush=True)


def packets():
    require_admission()
    base.terminal(OUTPUTS)
    outputs = q.read(OUTPUTS / "outputs.json")
    expected = [r for split, rows in sources().items() for r in rows]
    candidate.reference.require_outputs(outputs, expected)
    if file_hash(OUTPUTS / "outputs.json") != q.read(OUTPUTS / "result.json")["outputs_sha256"]:
        raise ValueError("fixed final outputs changed")
    proof = base.verify_public(outputs)
    values = {split: [q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["text"]}])
        for r in outputs if r["split"] == split] for split in COUNTS}
    p = producing(PACKETS, "exact_public_final_summary_packets", [OUTPUTS / name for name in
        ("producer.json", "result.json", "outputs.json", "worker_terminal_receipt.json")]+[base.PUBLIC_FILE],
        {"cases": 320, "gateway": "https://central-dev.zt:4000/v1/chat/completions",
            "judges": q.JUDGES, "rubric": candidate.reference.profile.RUBRIC})
    write_json(PACKETS / "packets.json", values)
    write_json(PACKETS / "public_payload_proof.json", {"public_file_sha256": base.PUBLIC_SHA256,
        "exact_sources": proof, "source_fields_public_exact": True, "private_data_in_packets": False,
        "packets_sha256": file_hash(PACKETS / "packets.json")})
    base.verify(p)
    write_json(PACKETS / "result.json", {"status": "sealed_public_packets_prepared", "cases": 320,
        "files": {name: file_hash(PACKETS/name) for name in ("packets.json", "public_payload_proof.json")},
        "producer_fingerprint": p["fingerprint"], "final_test_opened": True})


def grade():
    require_admission()
    base.prior.dependencies()
    reference = candidate.reference
    reference.profile.install()
    base.terminal(PACKETS)
    base.terminal(OUTPUTS)
    for name, expected in q.read(PACKETS / "result.json")["files"].items():
        if file_hash(PACKETS/name) != expected:
            raise ValueError("exact public final payload proof changed")
    outputs = q.read(OUTPUTS / "outputs.json")
    cases = sources()
    reference.require_outputs(outputs, [r for split in COUNTS for r in cases[split]])
    exact_packets = q.read(PACKETS / "packets.json")
    if exact_packets != {split: [q.packet(r["id"], r["source"], [{"candidate_id": "summary", "text": r["text"]}])
            for r in outputs if r["split"] == split] for split in COUNTS}:
        raise ValueError("actual judge packets differ from the public source proof")
    cache = q.read(candidate.GRADE / "judgment_cache.json")
    prior = dict(cache)
    for key, entry in cache.items():
        if key != reference.first.packet_key(entry["packet"], entry["role"]):
            raise ValueError("prior first-judgment identity differs")
        reference.first.audit_row(entry["packet"], entry["role"], entry["row"])
    inputs = [PACKETS / name for name in ("producer.json", "result.json", "packets.json", "public_payload_proof.json", "worker_terminal_receipt.json")]
    inputs += [OUTPUTS / name for name in ("producer.json", "result.json", "outputs.json", "worker_terminal_receipt.json")]
    for role in ("secondary", "primary"):
        qualification = ROOT / ("20261023_budget128_judge_"+role+"_v1")
        base.terminal(qualification)
        if q.read(qualification / "result.json")["status"] != "passed":
            raise ValueError("both qualified128-token judge profiles required")
        inputs += [qualification / name for name in ("producer.json", "result.json", "worker_terminal_receipt.json")]
    p = producing(GRADE, "sealed_complete_summary_consensus_quality", inputs,
        {"cases": 320, "prior_first_judgments": len(prior), "workers": 4, "max_attempts": 3,
            "rubric": reference.profile.RUBRIC, "judges": q.JUDGES, "semantic_rerolls": False})
    reference.GRADE = GRADE
    write_json(GRADE / "judgment_cache.json", cache)
    results = {}
    for split, count in COUNTS.items():
        rows = [r for r in outputs if r["split"] == split]
        secondary = reference.grade_cohort(split, "secondary", rows, exact_packets[split], cache)
        primary = reference.grade_cohort(split, "primary", rows, exact_packets[split], cache) if qualifies(secondary, count) else None
        secondary_scores = q.read(GRADE / f"{split}_secondary_case_scores.json")
        primary_scores = q.read(GRADE / f"{split}_primary_case_scores.json") if primary is not None else None
        results[split] = {"secondary": secondary, "primary": primary,
            "consensus": consensus(secondary_scores, primary_scores)}
    if any(cache.get(key) != value for key, value in prior.items()):
        raise ValueError("an earlier first judgment changed")
    base.verify(p)
    met = all(value["consensus"]["qualified"] for value in results.values())
    result = {"status": "completed", "results": results, "quality_gate": met,
        "classification": "sealed_consensus_target_passed_pending_audit" if met else "sealed_consensus_target_not_met",
        "final_test_opened": True, "NCM_success_established": False, "confirmed": False,
        "speed_tested": False, "prior_first_judgments_preserved": len(prior), "producer_fingerprint": p["fingerprint"]}
    write_json(GRADE / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["prepare", "feature-pilot", "features", "pilot", "generate", "packets", "grade"])
    stage = parser.parse_args().stage
    if stage == "prepare": prepare()
    elif stage in ("feature-pilot", "features"): features(stage == "feature-pilot")
    elif stage in ("pilot", "generate"): generate(stage == "pilot")
    elif stage == "packets": packets()
    else: grade()
