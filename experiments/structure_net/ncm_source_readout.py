"""Evaluate source-copying readouts before training a replacement NCM head."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

from experiments.structure_net import ncm_memory_accuracy_v2 as q
from experiments.structure_net.ncm_memory_data import file_hash, text_hash, write_json
from experiments.structure_net.ncm_paper_data import PREP, ROOT, verify
from experiments.structure_net.ncm_source_readout_inputs import CACHE, DEST as MODEL, PLAN, REVISION

DATA = ROOT / "20261023_source_readout_prepare_v1"
PILOT = ROOT / "20261023_source_readout_pilot_v1"
MEASURE = ROOT / "20261023_source_readout_measure_v1"
GRADING = ROOT / "20261023_source_readout_grading_v1"
DEPS = ROOT / "source_readout_dependencies_v1"
TOKEN_DEPS = Path("data/experiments/ncm_memory/dependencies")
NCM = ROOT / "20261022_full_train_v1/development_step5120.json"
PARENT = ROOT / "20261022_parent_outputs_v1/outputs.json"
REFERENCE = ROOT / "20261022_expanded_teacher_pilot_v2/train.json"
AUDIT = ROOT / "20261022_expanded_recovery_audit_v1"
CONDITIONS = ("lead", "source_centroid", "ncm_query", "parent_query")
HYPOTHESIS = "ncm-source-grounded-readout-feasibility-v1"


def dependencies():
    # Append isolated extras so they cannot replace the CUDA runtime's packages.
    for path in (DEPS, TOKEN_DEPS):
        if str(path.resolve()) not in sys.path:
            sys.path.append(str(path.resolve()))


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def candidates(source, sentences, token_count):
    """Whole contiguous segmentation units; never cut a unit to fit the limit."""
    result, seen = [], set()
    for i, sentence in enumerate(sentences):
        for j in range(i, min(i+3, len(sentences))):
            start, end = sentence["start"], sentences[j]["end"]
            raw = source[start:end]
            text = normalized(raw)
            n = token_count(text)
            if not text or n > 128 or text in seen:
                continue
            seen.add(text)
            result.append({"id": text_hash(str(start)+":"+str(end)+":"+raw), "start": start, "end": end,
                "raw": raw, "text": text, "tokens": n, "sentence_ids": list(range(i, j+1)), "sentence_count": j-i+1})
    return result


def select_index(options, scores):
    if len(options) != len(scores):
        raise ValueError("candidate/score shape mismatch")
    if not options:
        return None
    import math
    if not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("nonfinite selection score")
    return min(range(len(options)), key=lambda i: (-float(scores[i]), options[i]["tokens"], options[i]["start"], options[i]["end"]))


def freeze(root, stage, inputs, config):
    from experiments.structure_net.ncm_paper_controls import producer
    runtime = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers")}
    p = producer(root, stage, inputs, {**config, "runtime_versions": runtime})
    paths = [Path(__file__), PLAN, Path("experiments/structure_net/ncm_source_readout_inputs.py")]
    paths += sorted(DEPS.rglob("*.py"))
    paths += list(DEPS.glob("*.dist-info/METADATA"))
    for i, path in enumerate(paths):
        p["source_hashes"][str(path)] = file_hash(path)
        (root / "producer_source" / f"readout_{i:04d}_{path.name}").write_bytes(path.read_bytes())
    p["hypothesis_id"] = HYPOTHESIS
    p["fingerprint"] = q.digest({k:v for k,v in p.items() if k != "fingerprint"})
    write_json(root / "producer.json", p)
    return p


def prepare():
    dependencies()
    import pysbd
    import tiktoken
    from experiments.structure_net.ncm_recipe_teacher import verify_public
    development = q.read(PREP / "development_sources.json")
    reference = q.read(REFERENCE)
    drafts = [{r["id"]: r for r in q.read(path)} for path in (NCM, PARENT)]
    seals = [ROOT / ("20261022_current_"+arm+"_seal_v1.json") for arm in ("ncm", "parent")]
    for path, seal_path in zip((NCM, PARENT), seals):
        seal = q.read(seal_path)
        if seal["status"] != "sealed_complete_generation" or file_hash(path) != seal["artifacts"][str(path)]:
            raise ValueError("preserved draft generation changed")
    reference_proof = ROOT / "20261022_unseen_codec_measure_v1/producer.json"
    if file_hash(REFERENCE) != q.read(reference_proof)["input_hashes"][str(REFERENCE)]:
        raise ValueError("fixed reference cohort changed")
    if len(development) != 64 or len(reference) != 20 or any(set(d) != {r["id"] for r in development} for d in drafts):
        raise ValueError("fixed64 development and20 separate reference cases required")
    for key in ("id", "source_id", "group_id", "source_sha256"):
        if any(len({r[key] for r in rows}) != len(rows) for rows in (development, reference)):
            raise ValueError("repeated source unit within a fixed cohort")
        if {r[key] for r in development} & {r[key] for r in reference}:
            raise ValueError("development/reference source overlap")
    verify_public(development+reference)
    p = freeze(DATA, "source_candidate_preparation_before_selection", [PREP / "development_sources.json", NCM, PARENT,
        REFERENCE, reference_proof, *seals, MODEL / "result.json", MODEL / "source.py", MODEL / "plan.md"],
        {"development": 64, "reference_oracle": 20, "max_tokens": 128, "max_sentences": 3,
         "pysbd": "0.3.4", "tokenizer": "cl100k_base", "conditions": CONDITIONS})
    enc = tiktoken.get_encoding("cl100k_base")
    segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    records = []
    for cohort, cases in (("development", development), ("reference", reference)):
        for case in cases:
            if text_hash(case["source"]) != case["source_sha256"]:
                raise ValueError("source SHA changed")
            sentences = []
            for span in segmenter.segment(case["source"]):
                if case["source"][span.start:span.end] != span.sent:
                    raise ValueError("sentence span is not an exact source substring")
                text = normalized(span.sent)
                if text:
                    sentences.append({"start": span.start, "end": span.end, "raw": span.sent, "text": text})
            options = candidates(case["source"], sentences, lambda text: len(enc.encode(text, disallowed_special=())))
            base = {k:case[k] for k in ("id", "source_id", "source", "source_sha256", "group_id", "spans")}
            queries = {}
            if cohort == "development":
                for name, mapping in zip(("ncm_query", "parent_query"), drafts):
                    row = mapping[case["id"]]
                    if any(row[k] != case[k] for k in ("source", "source_sha256", "group_id", "source_id")):
                        raise ValueError("draft/source identity mismatch")
                    queries[name] = row["text"]
            else:
                queries["reference_query"] = case["target"]
            records.append({**base, "cohort": cohort, "queries": queries, "sentences": sentences, "candidates": options})
    features = {}
    for case in records:
        for text in [r["text"] for r in case["sentences"]+case["candidates"]]+list(case["queries"].values()):
            key = text_hash(text)
            if key in features and features[key] != text:
                raise ValueError("feature text hash collision")
            features[key] = text
    write_json(DATA / "cases.json", records)
    write_json(DATA / "features.json", [{"sha256": key, "text": features[key]} for key in sorted(features)])
    verify(p)
    result = {"status": "prepared", "development": 64, "reference_oracle": 20,
        "no_candidate_cases": sum(not r["candidates"] for r in records), "features": len(features),
        "total_candidates": sum(len(r["candidates"]) for r in records), "producer_fingerprint": p["fingerprint"],
        "output_hashes": {str(DATA/n): file_hash(DATA/n) for n in ("cases.json", "features.json")},
        "task_accuracy": None, "trained": False, "final_test_opened": False, "speed_tested": False}
    write_json(DATA / "result.json", result); print(json.dumps(result), flush=True)


def selection(cases, vectors):
    import torch
    import torch.nn.functional as F
    outputs = {name: [] for name in (*CONDITIONS, "reference_query")}
    for case in cases:
        options = case["candidates"]
        for condition in CONDITIONS if case["cohort"] == "development" else ("reference_query",):
            scores = []
            index = None
            if condition == "lead":
                singles = [i for i,r in enumerate(options) if r["sentence_count"] == 1]
                index = min(singles, key=lambda i: (options[i]["start"], options[i]["end"])) if singles else None
            elif options:
                if condition == "source_centroid":
                    sentence_vectors = torch.stack([vectors[text_hash(s["text"])] for s in case["sentences"]])
                    query = F.normalize(sentence_vectors.mean(0), dim=0)
                else:
                    query = vectors[text_hash(case["queries"][condition])]
                matrix = torch.stack([vectors[text_hash(r["text"])] for r in options])
                scores = (matrix@query).tolist()
                index = select_index(options, scores)
            chosen = options[index] if index is not None else None
            base = {k:case[k] for k in ("id", "source_id", "source", "source_sha256", "group_id", "spans")}
            outputs[condition].append({**base, "text": chosen["text"] if chosen else "", "tokens": chosen["tokens"] if chosen else 0,
                "selected": chosen, "condition": condition, "target_aware": condition == "reference_query",
                "selection_score": float(scores[index]) if chosen and scores else None})
    return outputs


def run(pilot):
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from experiments.structure_net.ncm_frozen_body import module_hash
    from experiments.structure_net.train_ncm_memory import save
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "2" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise ValueError("select physicalGPU2 before startup")
    if not torch.cuda.is_available() or "2060 SUPER" not in torch.cuda.get_device_name():
        raise ValueError("planned CUDA device required")
    if torch.cuda.mem_get_info()[0] < 3_000_000_000:
        raise ValueError("insufficient GPU2 headroom")
    verify(q.read(DATA / "producer.json"))
    for path, expected in q.read(DATA / "result.json")["output_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("candidate preparation changed")
    model_proof = q.read(MODEL / "result.json")
    if model_proof["revision"] != REVISION or file_hash(MODEL / "source.py") != model_proof["source_sha256"]:
        raise ValueError("public model preparation proof changed")
    for path, receipt in model_proof["files"].items():
        if file_hash(path) != receipt["sha256"]:
            raise ValueError("verified public model input changed")
    cases = q.read(DATA / "cases.json")
    selected = ([r for r in cases if r["cohort"] == "development"][:2]+[r for r in cases if r["cohort"] == "reference"][:2]) if pilot else cases
    all_features = q.read(DATA / "features.json")
    needed = {text_hash(text) for case in selected for text in
        [r["text"] for r in case["sentences"]+case["candidates"]]+list(case["queries"].values())}
    features = [r for r in all_features if r["sha256"] in needed]
    root = PILOT if pilot else MEASURE
    inputs = [DATA/n for n in ("cases.json", "features.json", "result.json", "producer.json")]
    inputs += [MODEL / "result.json", MODEL / "source.py", MODEL / "cpu_state_structure.json"]
    inputs += [Path(path) for path in model_proof["files"]]
    p = freeze(root, "systems_lifecycle_only_not_quality_evidence" if pilot else "fixed_source_readout_selection",
        inputs, {"batch": 8, "model_revision": REVISION, "model_precision": "FP16", "pooling": "L2 normalized CLS",
            "attention": "sdpa", "seed": 42, "physical_gpu": 2, "cases": len(selected), "pilot": pilot})
    try:
        if not pilot:
            prior, receipt = q.read(PILOT / "producer.json"), q.read(PILOT / "result.json")
            if receipt["status"] != "passed" or any(prior[k] != p[k] for k in ("source_hashes", "input_hashes")):
                raise ValueError("matching real GPU lifecycle pilot required")
            verify(prior)
        torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(CACHE, local_files_only=True, use_fast=True)
        lengths = tokenizer([r["text"] for r in all_features], padding=False, truncation=False)["input_ids"]
        longest = max(range(len(lengths)), key=lambda i: len(lengths[i]))
        maximum = len(lengths[longest])
        if maximum > 8192:
            raise ValueError("BGE feature exceeds its context; truncation prohibited")
        config = AutoConfig.from_pretrained(CACHE, local_files_only=True)
        model = AutoModel.from_config(config, attn_implementation="sdpa")
        state = torch.load(CACHE / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True); del state; gc.collect()
        model.half().cuda().eval().requires_grad_(False)
        initial = module_hash(model)
        def embed(texts):
            encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=False, return_token_type_ids=False)
            if encoded["input_ids"].shape[1] > 8192:
                raise ValueError("unplanned BGE context overrun")
            encoded = {k:v.cuda() for k,v in encoded.items()}
            with torch.inference_mode():
                last = model(**encoded).last_hidden_state[:, 0].float()
                dense = F.normalize(last, p=2, dim=-1)
            if not bool(torch.isfinite(dense).all()) or not bool(dense.norm(dim=-1).sub(1).abs().lt(1e-5).all()):
                raise ValueError("nonfinite or non-normalized dense embeddings")
            return dense.cpu()
        if pilot:
            embed([all_features[longest]["text"]]*8)
        vectors = []
        for start in range(0, len(features), 8):
            vectors.append(embed([r["text"] for r in features[start:start+8]]))
            if start % 128 == 0:
                print(json.dumps({"embedded": min(start+8,len(features)), "total": len(features)}), flush=True)
        bank = {"sha256": [r["sha256"] for r in features], "vectors": torch.cat(vectors)}
        mapping = dict(zip(bank["sha256"], bank["vectors"]))
        output = selection(selected, mapping)
        save(root / "embeddings.pt", bank)
        write_json(root / "outputs.json", output)
        reload = torch.load(root / "embeddings.pt", weights_only=True)
        if not torch.equal(reload["vectors"], bank["vectors"]) or reload["sha256"] != bank["sha256"]:
            raise ValueError("embedding serialization changed")
        if selection(selected, dict(zip(reload["sha256"], reload["vectors"]))) != output:
            raise ValueError("selection changed on embedding reload")
        if module_hash(model) != initial or any(v.grad is not None for v in model.parameters()):
            raise ValueError("frozen BGE model changed")
        verify(p)
        result = {"status": "passed" if pilot else "selection_completed_pending_quality", "cases": len(selected),
            "condition_cases": {k:len(v) for k,v in output.items()}, "empty_outputs": {k:sum(not r["text"] for r in v) for k,v in output.items()},
            "feature_count": len(features), "maximum_feature_tokens": maximum, "untruncated": True,
            "frozen_model_hash": initial, "strict_weights_load": True, "reload_exact": True,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "producer_fingerprint": p["fingerprint"],
            "output_hashes": {str(root/n): file_hash(root/n) for n in ("embeddings.pt", "outputs.json")},
            "task_accuracy": None, "trained": False, "NCM_success_established": False,
            "final_test_opened": False, "speed_tested": False, "confirmed": False, "underpowered": True}
        write_json(root / "result.json", result); print(json.dumps(result), flush=True)
    except Exception as exc:
        write_json(root / "failure.json", {"status": "failed", "error": repr(exc), "producer_fingerprint": p["fingerprint"]})
        raise


def grade(pilot):
    dependencies()
    from experiments.structure_net.ncm_paper_grade import grade as judge
    receipt = q.read(AUDIT / "worker_terminal_receipt.json")
    if receipt["status"] != "observed_worker_terminal" or receipt["exit_code"] != 0:
        raise ValueError("authoritatively observed audit termination required before new API grading")
    if q.read(MEASURE / "result.json")["status"] != "selection_completed_pending_quality":
        raise ValueError("completed selection required")
    verify(q.read(MEASURE / "producer.json"))
    for path, expected in q.read(MEASURE / "result.json")["output_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("selection outputs changed")
    outputs = q.read(MEASURE / "outputs.json")
    inputs = [MEASURE/n for n in ("outputs.json", "producer.json", "result.json")]
    inputs += [PLAN, Path(__file__), AUDIT / "result.json", AUDIT / "worker_terminal_receipt.json"]
    if pilot:
        cases = [outputs[condition][i] for i,condition in enumerate(CONDITIONS)]
        judge(cases, ROOT / "20261023_source_readout_judge_pilot_v1", inputs, role="secondary", systems_only=True)
        return
    prior = ROOT / "20261023_source_readout_judge_pilot_v1/result.json"
    r = q.read(prior)
    if r["status"] != "completed" or r["valid"] != 4 or not r["systems_only"]:
        raise ValueError("fresh four-case judge lifecycle pilot required")
    p = freeze(GRADING, "fixed_whole_summary_readout_comparison", inputs+[prior],
        {"conditions": CONDITIONS, "reference_oracle": 20, "development_cases": 64,
         "secondary_pass_minimum": 61, "primary_pass_minimum": 61, "oracle_pass_minimum": 19,
         "judges": q.JUDGES, "first_valid_preserved": True, "one_candidate_per_request": True})
    result = {}
    for condition, rows in outputs.items():
        child = ROOT / ("20261023_source_readout_grade_secondary_"+condition+"_v1")
        secondary = judge(rows, child, inputs+[prior], role="secondary")
        result[condition] = {"secondary": secondary, "primary": None}
        if condition != "reference_query" and secondary["status"] == "completed" and secondary["passed"] >= 61:
            primary_child = ROOT / ("20261023_source_readout_grade_primary_"+condition+"_v1")
            result[condition]["primary"] = judge(rows, primary_child, inputs+[child / "result.json"], role="primary")
        write_json(GRADING / "progress.json", result)
    verify(p)
    aggregate = {"status": "completed", "results": result, "producer_fingerprint": p["fingerprint"],
        "NCM_trained": False, "NCM_success_established": False, "final_test_opened": False,
        "speed_tested": False, "confirmed": False, "underpowered": True}
    write_json(GRADING / "result.json", aggregate); print(json.dumps(aggregate), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "pilot", "select", "grade-pilot", "grade"])
    command = parser.parse_args().command
    if command == "prepare":
        prepare()
    elif command in ("pilot", "select"):
        run(command == "pilot")
    else:
        grade(command == "grade-pilot")
