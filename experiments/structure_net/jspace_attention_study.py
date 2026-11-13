"""JSPACE-ATTENTION-TRANSPORT-01: compact attention concepts and their induced action."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import torch

from .jspace_attention_tasks import make_tasks, calibration, validate_tasks
from .jspace_attention_runtime import fit_dictionary, forward
from .jspace_lens import load_small_qwen
from .jspace_relation_tasks import COUNTRIES
from .jspace_relation_study import ROOT, digest, file_hash, save_tensor, write_json
from .jspace_causal_components import cpu_only


STUDY_ID = "JSPACE-ATTENTION-TRANSPORT-01"
DEFAULT_ROOT = ROOT/"data/experiments/jspace_attention_transport/20261113_v1"
PARENT = ROOT/"data/experiments/jspace_active_cav_heads/20261113_v1/plan.json"
PROTOCOL = ROOT/"docs/07 - Status Reports/jspace/2026-11-13_jspace-attention-transport-plan.md"
CONFIG = {"start": 11, "stop": 35, "candidate_layers": [11, 17, 23, 24, 25, 26, 27, 28, 29],
          "methods": ["projected_donor", "concept_swap"], "calibration_count": 64, "calibration_tokens": 128,
          "vjp_batch": 2, "maximum_prompt_tokens": 192, "dtype": "bfloat16", "semantic_bank_rank_cap": 8,
          "minimum_accuracy": .9, "minimum_joint": .8, "minimum_dense": .5,
          "minimum_J": .2, "minimum_advantage": .1, "maximum_p": .05,
          "primary_transport_hops": 7, "maximum_polynomial_delta_nrmse": .75, "minimum_relative_gain": .1,
          "divergence_absolute_standardized_value": 1e4, "random_seed": 557}


def sources():
    names = ["attention_tasks", "attention_runtime", "attention_transport", "attention_study", "lens", "runtime",
             "tasks", "causal_components", "relation_tasks", "relation_capture", "relation_study", "country_probe",
             "residual_expansion", "donor_localization", "relations", "relation_analysis", "relation_propagation"]
    paths = [ROOT/f"experiments/structure_net/jspace_{name}.py" for name in names]
    paths += [PROTOCOL, ROOT/"tests/structure_net/test_jspace_attention_transport.py", ROOT/"pixi.lock", ROOT/"pyproject.toml"]
    paths += sorted((ROOT/"src/structure_net_resources").glob("*.py"))
    return paths


def prepare(root):
    cpu_only()
    from transformers import AutoTokenizer
    if root.exists():
        raise FileExistsError(root)
    model = json.loads(PARENT.read_text())["model"]
    tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True, trust_remote_code=False)
    tasks, corpus = make_tasks(tokenizer), calibration()
    token_ids, labels = [], []
    for country in COUNTRIES:
        for text in (country, " "+country):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError("Every declared country lexical variant must be one token")
            token_ids.append(ids[0])
            labels.append(text)
    if len(set(token_ids)) != 8:
        raise ValueError("Eight distinct lexical directions required")
    value = {"study_id": STUDY_ID, "schema_version": 1, "config": CONFIG, "model": model,
             "token_ids": token_ids, "lexical_labels": labels, "dataset_digest": digest(tasks), "calibration_digest": digest(corpus),
             "producer_sha256": {str(p): file_hash(p) for p in sources()},
             "input_sha256": {str(PARENT): file_hash(PARENT), corpus["source_path"]: corpus["source_sha256"]},
             "conformance": "PROSPECTIVE_SINGLE_CHECKPOINT_EXPLORATORY_PILOT",
             "created_at": datetime.now(timezone.utc).isoformat()}
    value["scientific_fingerprint"] = digest(value)
    root.mkdir(parents=True)
    for path in sources():
        target = root/"producer"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write_json(root/"plan.json", value)
    write_json(root/"tasks.json", tasks)
    write_json(root/"calibration.json", corpus)
    return {"scientific_fingerprint": value["scientific_fingerprint"], "counts": tasks["counts"], "lexical_labels": labels}


def verify(root, model=False):
    plan = json.loads((root/"plan.json").read_text())
    if plan["config"] != CONFIG or plan["study_id"] != STUDY_ID or digest({k:v for k,v in plan.items() if k != "scientific_fingerprint"}) != plan["scientific_fingerprint"]:
        raise ValueError("Plan/configuration fingerprint changed")
    for section in ("producer_sha256", "input_sha256"):
        for path, expected in plan[section].items():
            if file_hash(path) != expected:
                raise ValueError(f"Frozen input changed: {path}")
            if section == "producer_sha256" and file_hash(root/"producer"/Path(path).relative_to(ROOT)) != expected:
                raise ValueError("Producer archive changed")
    for name, key in (("tasks", "dataset_digest"), ("calibration", "calibration_digest")):
        if digest(json.loads((root/f"{name}.json").read_text())) != plan[key]:
            raise ValueError("Frozen task/corpus changed")
    if model:
        for name, expected in plan["model"]["files"].items():
            if file_hash(Path(plan["model"]["path"])/name) != expected:
                raise ValueError("Checkpoint identity changed")
    return plan


def checked(root, stage):
    plan = verify(root)
    record = json.loads((root/f"{stage}_manifest.json").read_text())
    if record["status"] != "completed" or record["scientific_fingerprint"] != plan["scientific_fingerprint"]:
        raise ValueError("Incomplete/mismatched stage")
    for name, expected in record["artifacts"].items():
        if file_hash(root/name) != expected:
            raise ValueError("Stage artifact changed")
    return record


class Stage:
    def __init__(self, root, name):
        from structure_net_resources.client import require_gpu_lease
        self.lease = require_gpu_lease([0])
        self.check_lease = require_gpu_lease
        self.root, self.name = root, name
        self.plan = verify(root, model=True)
        self.path = root/name
        self.path.mkdir(exist_ok=False)
        self.started, self.counts = time.perf_counter(), Counter()
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        self.model, self.tokenizer = load_small_qwen(self.plan["model"]["path"], device="cuda:0", dtype=torch.bfloat16)
        self.tasks = json.loads((root/"tasks.json").read_text())
        self.nodes = [f"{l}.{k}" for l in range(CONFIG["start"], CONFIG["stop"]+1) for k in ("attention", "residual", "block")]
        write_json(self.path/"lease.json", {k:self.lease[k] for k in ("id", "holder", "state", "ready", "allocation", "end")})

    def event(self, name, value):
        self.counts[name] += 1
        with (self.path/"events.jsonl").open("a") as f:
            f.write(json.dumps({"kind": name, "value": value, "elapsed_seconds": time.perf_counter()-self.started}, allow_nan=False)+"\n")
        if self.counts[name] == 1 or self.counts[name] % 16 == 0:
            self.check_lease([0])
            print(json.dumps({"stage": name, "completed": self.counts[name], "elapsed_seconds": round(time.perf_counter()-self.started, 2)}), flush=True)

    def finish(self, extra=(), **measurements):
        self.check_lease([0])
        if torch.cuda.max_memory_reserved() > sum(self.lease["allocation"].values())*1024**2:
            raise RuntimeError("GPU allocation exceeded")
        verify(self.root)
        files = [p for p in self.path.rglob("*") if p.is_file()]+[self.root/n for n in extra]
        value = {"study_id": STUDY_ID, "status": "completed", "scientific_fingerprint": self.plan["scientific_fingerprint"],
                 "stage": self.name, "counts": dict(self.counts), "elapsed_seconds": time.perf_counter()-self.started,
                 "model_parameters": sum(p.numel() for p in self.model.parameters()), "lease_id": self.lease["id"],
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                 "runtime": {"torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16"},
                 "artifacts": {str(p.relative_to(self.root)):file_hash(p) for p in files}, **measurements}
        write_json(self.root/f"{self.name}_manifest.json", value)
        return {k:value[k] for k in ("stage", "status", "counts", "elapsed_seconds", "peak_reserved_bytes")}


def pack(outputs, nodes):
    return torch.stack([torch.stack([o["scores"][node] for node in nodes]) for o in outputs])


def choose_origin(rows):
    eligible = [r for r in rows if r["source"]["original_correct"] and r["donor"]["original_correct"]]
    candidates = []
    for layer in CONFIG["candidate_layers"]:
        dense = sum(r["edits"][str(layer)]["dense"]["target_correct"] for r in eligible)
        for method in CONFIG["methods"]:
            success = sum(r["edits"][str(layer)][method]["target_correct"] for r in eligible)
            random = sum(r["edits"][str(layer)]["random_"+method]["target_correct"] for r in eligible)
            candidates.append({"layer": layer, "method": method, "eligible": len(eligible), "target_correct": success,
                               "random_target_correct": random, "dense_target_correct": dense,
                               "dense_control_qualified": bool(eligible and dense/len(eligible) >= .5)})
    choices = [c for c in candidates if c["dense_control_qualified"]] or candidates
    selected = min(choices, key=lambda c:(-c["target_correct"], -(c["target_correct"]-c["random_target_correct"]), c["layer"], CONFIG["methods"].index(c["method"])))
    return {**selected, "candidates": candidates, "selection_scope": "development only; failed positive control retains descriptive tracing, never a causal claim"}


def discovery(root):
    stage = Stage(root, "discovery")
    corpus = json.loads((root/"calibration.json").read_text())["rows"]
    smoke = fit_dictionary(stage.model, stage.tokenizer, [corpus[0]["text"]], stage.plan["token_ids"], start=11)
    controls = []
    for pair in stage.tasks["pairs"]["shakedown"]:
        original = forward(stage.model, stage.tokenizer, pair["source"], pair["question_start"], smoke["rows"], keep_attention=(11,))
        repeat = forward(stage.model, stage.tokenizer, pair["source"], pair["question_start"], smoke["rows"],
                         patch_layer=11, donor=original["attention"][11], method="dense")
        if original["row"]["token_ids"] != repeat["row"]["token_ids"] or any(not torch.equal(original["scores"][k],repeat["scores"][k]) for k in original["scores"]):
            raise ValueError("Actual attention identity patch changed the model")
        controls.append({"case_id": pair["source"]["case_id"], "identity_exact": True})
    if any(not torch.equal(smoke["raw"][f"{l}.attention"],smoke["raw"][f"{l}.residual"]) for l in range(11,36)):
        raise ValueError("Attention-write gradient and post-addition gradient disagree")
    save_tensor(stage.path/"shakedown_dictionary.pt", smoke)
    replay = torch.load(stage.path/"shakedown_dictionary.pt",map_location="cpu",weights_only=True)
    if any(not torch.equal(v,replay["raw"][k]) for k,v in smoke["raw"].items()):
        raise ValueError("Dictionary persistence changed")
    write_json(stage.path/"shakedown.json", {"scope":"systems_lifecycle_only_not_quality_evidence", "cases":controls,
                "attention_residual_gradient_identity_exact":True, "dictionary_roundtrip_exact":True})
    dictionary = fit_dictionary(stage.model,stage.tokenizer,[r["text"] for r in corpus],stage.plan["token_ids"],start=11,progress=stage.event)
    save_tensor(root/"dictionary.pt", dictionary)
    results, features = [], {"source":[], "donor":[]}
    for pair in stage.tasks["pairs"]["development"]:
        outputs = {side:forward(stage.model,stage.tokenizer,pair[side],pair["question_start"],dictionary["rows"],keep_attention=CONFIG["candidate_layers"]) for side in ("source","donor")}
        row = {side:outputs[side]["row"] for side in outputs}
        row["edits"] = {}
        for side in outputs:
            features[side].append(pack([outputs[side]],stage.nodes)[0])
        for layer in CONFIG["candidate_layers"]:
            row["edits"][str(layer)] = {}
            for method in ("dense","projected_donor","random_projected_donor","concept_swap","random_concept_swap"):
                out = forward(stage.model,stage.tokenizer,pair["source"],pair["question_start"],dictionary["rows"],
                              patch_layer=layer,donor=outputs["donor"]["attention"][layer],method=method)
                row["edits"][str(layer)][method] = out["row"]
                stage.event("development_edit", {"case_id":pair["source"]["case_id"],"layer":layer,"method":method,"target_correct":out["row"]["target_correct"]})
        results.append(row)
        stage.event("development_world", {"case_id":pair["source"]["case_id"],"source_correct":row["source"]["original_correct"],"donor_correct":row["donor"]["original_correct"]})
    save_tensor(stage.path/"features.pt", {"nodes":stage.nodes, **{k:torch.stack(v) for k,v in features.items()}})
    write_json(stage.path/"records.json",results)
    selection = choose_origin(results)
    write_json(root/"selection.json",selection)
    return stage.finish(("dictionary.pt","selection.json"), selection=selection)


def training(root):
    checked(root,"discovery")
    stage = Stage(root,"training")
    dictionary = torch.load(root/"dictionary.pt",map_location="cpu",weights_only=True)["rows"]
    features, records = {"source":[],"donor":[]}, []
    for pair in stage.tasks["pairs"]["train"]:
        row = {}
        for side in features:
            out = forward(stage.model,stage.tokenizer,pair[side],pair["question_start"],dictionary)
            features[side].append(pack([out],stage.nodes)[0])
            row[side] = out["row"]
            stage.event("training_natural", {"case_id":pair[side]["case_id"],"correct":out["row"]["original_correct"]})
        records.append(row)
    save_tensor(stage.path/"features.pt",{"nodes":stage.nodes,**{k:torch.stack(v) for k,v in features.items()}})
    write_json(stage.path/"records.json",records)
    return stage.finish()


def evaluation(root):
    checked(root,"discovery")
    fit = json.loads((root/"fit.json").read_text())
    if fit["transport_sha256"] != file_hash(root/"transport.pt") or not fit["frozen_before_evaluation"]:
        raise ValueError("Fit changed or not frozen before evaluation")
    stage = Stage(root,"evaluation")
    dictionary = torch.load(root/"dictionary.pt",map_location="cpu",weights_only=True)["rows"]
    selected = json.loads((root/"selection.json").read_text())
    layer, method = selected["layer"], selected["method"]
    other = next(m for m in CONFIG["methods"] if m != method)
    features, edited, records = {"source":[],"donor":[]}, [], []
    for pair in stage.tasks["pairs"]["evaluation"]:
        outputs = {side:forward(stage.model,stage.tokenizer,pair[side],pair["question_start"],dictionary,keep_attention=(layer,)) for side in features}
        row = {side:outputs[side]["row"] for side in features}
        for side in features:
            features[side].append(pack([outputs[side]],stage.nodes)[0])
        row["edits"] = {}
        for arm in ("dense",method,"random_"+method,"remainder",other):
            out = forward(stage.model,stage.tokenizer,pair["source"],pair["question_start"],dictionary,
                          patch_layer=layer,donor=outputs["donor"]["attention"][layer],method=arm)
            row["edits"][arm] = out["row"]
            if arm == method:
                edited.append(pack([out],stage.nodes)[0])
            stage.event("evaluation_edit", {"case_id":pair["source"]["case_id"],"arm":arm,"target_correct":out["row"]["target_correct"]})
        records.append(row)
        stage.event("evaluation_world", {"case_id":pair["source"]["case_id"],"source_correct":row["source"]["original_correct"],"donor_correct":row["donor"]["original_correct"]})
    save_tensor(stage.path/"features.pt",{"nodes":stage.nodes,**{k:torch.stack(v) for k,v in features.items()}})
    save_tensor(stage.path/"edited_features.pt",{"nodes":stage.nodes,"scores":torch.stack(edited)})
    write_json(stage.path/"records.json",records)
    return stage.finish(("fit.json","transport.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","discovery","training","fit","evaluation","analyze"))
    parser.add_argument("--output",type=Path,default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.stage in ("fit","analyze"):
        from . import jspace_attention_transport
        operation = getattr(jspace_attention_transport,args.stage)
    else:
        operation = globals()[args.stage]
    try:
        result = operation(args.output)
    except BaseException as exc:
        path = args.output/f"failure_{args.stage}.json"
        if args.output.exists() and not path.exists() and not isinstance(exc,FileExistsError):
            write_json(path,{"stage":args.stage,"status":"invalid_stage","exception":type(exc).__name__,"error":str(exc)})
        raise
    print(json.dumps(result,allow_nan=False))
