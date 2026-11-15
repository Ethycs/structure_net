"""Small, CPU-only JSON-evidence summary of task acceptance versus selection.

This does not load/re-hash tensors or read task corpora/confirmation cases. It
authenticates small JSON artifacts against completed producer manifests, checks
the selection and accepted-parent chain, and retains recorded checkpoint hashes.
It is a reporting adapter, not a replacement for the producer's full audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

SCHEMA = "jspace.feature-resident-summary.v1"
PRODUCER_SCHEMA = "jspace.3b-feature-resident-campaign.v1"
PHASES = ("capture", "fit", "pilot", "prealign", "feature-pilot", "train", "verify")
REPO = Path(__file__).resolve().parents[2]


def sha(path):
    if Path(path).stat().st_size > 16 * 1024**2:
        raise ValueError(f"Summary only hashes small evidence files: {path}")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps({k:v for k,v in value.items() if k != "fingerprint"},
        sort_keys=True, allow_nan=False).encode()).hexdigest()


class Evidence:
    def __init__(self):
        self.hashes = {}
        self.recorded_binary_hashes = {}

    def read(self, path, expected=None):
        path = Path(path).resolve()
        if path.suffix != ".json":
            raise ValueError("Only JSON evidence may be opened")
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"JSON evidence changed: {path}")
        self.hashes[str(path)] = digest
        return json.loads(path.read_text())

    def plan(self, path, expected=None):
        value = self.read(path, expected)
        if value.get("fingerprint") != fingerprint(value):
            raise ValueError("Plan fingerprint changed")
        return value

    def phase(self, directory, phase, expected_fingerprint, manifest_sha=None):
        directory = Path(directory) / phase
        manifest = self.read(directory / "manifest.json", manifest_sha)
        if manifest.get("status") != "completed" or manifest.get("fingerprint") != expected_fingerprint:
            raise ValueError("Completed matching phase manifest required")
        artifacts = manifest.get("artifacts", {})
        if "result.json" not in artifacts:
            raise ValueError("Phase result is not authenticated")
        values = {}
        for name, digest in artifacts.items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()) or len(digest) != 64:
                raise ValueError("Invalid phase artifact reference")
            if path.suffix == ".json":
                values[name] = self.read(path, digest)
            else:
                # Explicitly record provenance, without reading weights/banks.
                self.recorded_binary_hashes[str(path.resolve())] = digest
        if values["result.json"].get("status") != "completed":
            raise ValueError("Incomplete phase result")
        return values["result.json"], values, artifacts


def validate_report(report):
    metrics, rows = report["metrics"], report["records"]
    if metrics.get("count") != 64 or len(rows) != 64 or len({r["identity"] for r in rows}) != 64:
        raise ValueError("Exactly 64 distinct development records required")
    for row in rows:
        values = row["logits"]
        if (len(values) not in (2, 4) or not all(math.isfinite(v) for v in values)
                or type(row["answer"]) is not int or not 0 <= row["answer"] < len(values)
                or row["prediction"] != max(range(len(values)), key=values.__getitem__)):
            raise ValueError("Invalid development prediction")
    correct = sum(r["prediction"] == r["answer"] for r in rows)
    if metrics.get("correct") != correct or metrics.get("accuracy") != correct / 64:
        raise ValueError("Accuracy does not match the development records")
    if not all(math.isfinite(metrics[k]) for k in ("ce", "original_kl")):
        raise ValueError("Finite selection metrics required")
    family = Counter(r["family"] for r in rows)
    if metrics.get("by_family") is not None and metrics["by_family"] != {
            f:dict(count=n, correct=sum(r["prediction"] == r["answer"] for r in rows if r["family"] == f))
            for f,n in family.items()}:
        raise ValueError("Family accuracy disagrees with records")


def score(row):
    metrics = row["metrics"]
    return metrics["correct"], -metrics["ce"]


def case_keys(report):
    return [(r["identity"],r["family"],r["answer"]) for r in report["records"]]


def selection(trained, values, artifacts):
    history = trained["history"]
    if (trained.get("completed_epochs") != 12 or trained.get("updates") != 384
            or [r["epoch"] for r in history] != list(range(-1, 13))
            or not trained.get("strict_reload_exact") or not trained.get("fixed_state_unchanged")):
        raise ValueError("Complete 12-epoch recovery and strict reload required")
    raw = history[0]["metrics"]
    for row in history:
        name = f"epoch_{row['epoch']:02d}_development.json"
        if (artifacts.get(name) != row["development_sha256"] or values[name]["metrics"] != row["metrics"]):
            raise ValueError("Epoch reference mismatch")
        validate_report(values[name])
        if case_keys(values[name]) != case_keys(values["epoch_-1_development.json"]):
            raise ValueError("Epochs do not use the same ordered development cases")
        finite = all(math.isfinite(row["metrics"][k]) for k in ("ce", "original_kl"))
        ce = finite and row["metrics"]["ce"] <= raw["ce"] + 1e-8
        kl = finite and row["metrics"]["original_kl"] <= raw["original_kl"] + 1e-8
        if row["eligibility"] != dict(eligible=ce and kl,finite=finite,ce_pass=ce,original_kl_pass=kl,tolerance=1e-8):
            raise ValueError("Scientific eligibility changed")
    scientific = max((r for r in history if r["epoch"] == -1 or r["eligibility"]["eligible"]), key=score)
    best = max(history, key=score)
    saved = values["diagnostic_best_task.json"]
    if (scientific["epoch"] != trained["selected_epoch"] or scientific["metrics"] != trained["selected"]
            or trained["initial"] != raw or trained["warm_initial"] != history[1]["metrics"]
            or artifacts.get("selected.pt") != trained["selected_checkpoint_sha256"]
            or saved["epoch"] != best["epoch"] or saved["metrics"] != best["metrics"]
            or saved.get("diagnostic_only") is not True or saved.get("automatically_adopted") is not False
            or artifacts.get("diagnostic_best_task.pt") != saved["checkpoint_sha256"]):
        raise ValueError("Scientific selection or saved best-task candidate mismatch")
    return scientific, best, saved


def accepted_summary(trained, values, artifacts, fresh, accepted, *, seed=False):
    scientific, best, saved = selection(trained, values, artifacts)
    winner = max([scientific, best], key=score)  # Scientific candidate wins exact ties.
    kind = ("feature_" if seed else "") + ("selected" if winner is scientific else "best_task")
    if not seed and winner is scientific:
        kind = "scientific_selected"
    expected_sha = trained["selected_checkpoint_sha256"] if winner is scientific else saved["checkpoint_sha256"]
    candidate = accepted["candidate"]
    quality = dict(passed=winner["metrics"]["correct"] >= 45,native_correct=48,minimum_correct=45,
        count=64,maximum_drop=.05,ce_kl_increase_is_veto=False)
    reference = values[f"epoch_{winner['epoch']:02d}_development.json"]
    if (candidate["kind"] != kind or candidate["epoch"] != winner["epoch"] or candidate["metrics"] != winner["metrics"]
            or candidate["checkpoint_sha256"] != expected_sha or candidate["report_sha256"] != artifacts[f"epoch_{winner['epoch']:02d}_development.json"]
            or accepted.get("quality") != quality or accepted.get("accepted") != quality["passed"]):
        raise ValueError("Task acceptance disagrees with the separately ranked candidate")
    scientific_reference = values[f"epoch_{scientific['epoch']:02d}_development.json"]
    if (not fresh.get("passed") or not fresh.get("fixed_state_unchanged")
            or fresh["metrics"] != scientific_reference["metrics"] or fresh["records"] != scientific_reference["records"]
            or fresh["selected_checkpoint_sha256"] != trained["selected_checkpoint_sha256"]):
        raise ValueError("Scientific fresh replay mismatch")
    return dict(raw_correct=trained["initial"]["correct"],warm_correct=trained["warm_initial"]["correct"],
        scientific_correct=scientific["metrics"]["correct"],scientific_epoch=scientific["epoch"],
        scientific_checkpoint_sha256=trained["selected_checkpoint_sha256"],
        best_task_correct=best["metrics"]["correct"],best_task_epoch=best["epoch"],
        best_task_checkpoint_sha256=saved["checkpoint_sha256"],accepted_correct=winner["metrics"]["correct"],
        accepted_epoch=winner["epoch"],accepted_kind=kind,accepted_checkpoint_sha256=expected_sha,
        accepted=quality["passed"],minimum_correct=45,count=64), reference


def summarize(root):
    root = Path(root).resolve(); evidence = Evidence(); plan = evidence.plan(root / "plan.json")
    if plan.get("schema") != PRODUCER_SCHEMA or plan.get("seed_cut") != 23 or plan.get("phases") != list(PHASES):
        raise ValueError("Expected the thirteen-block feature-resident continuation")
    for path,digest in plan["source_sha256"].items():
        path = Path(path); archived = root / "producer" / path.relative_to(REPO)
        if sha(path) != digest or sha(archived) != digest:
            raise ValueError("Frozen producer source/archive changed")
        evidence.hashes[str(path)] = digest; evidence.hashes[str(archived)] = digest
    pins = {str(Path(p).resolve()):digest for p,digest in plan["input_sha256"].items()}
    feature = Path(plan["feature_root"])
    feature_plan = evidence.plan(feature / "plan.json", pins[str((feature / "plan.json").resolve())])
    if feature_plan["fingerprint"] != plan["seed_fingerprint"]:
        raise ValueError("Feature seed fingerprint mismatch")
    trained, values, artifacts = evidence.phase(feature,"train",feature_plan["fingerprint"],pins[str((feature / "train/manifest.json").resolve())])
    fresh, _, _ = evidence.phase(feature,"verify",feature_plan["fingerprint"],pins[str((feature / "verify/manifest.json").resolve())])
    seed, reference = accepted_summary(trained,values,artifacts,fresh,plan["seed_selection"],seed=True)
    if seed["accepted_kind"] == "feature_best_task":
        proof_root = Path(plan["seed_verification"]["root"])
        proof_plan = evidence.plan(proof_root / "plan.json",pins[str((proof_root / "plan.json").resolve())])
        proof, _, _ = evidence.phase(proof_root,"run",proof_plan["fingerprint"],plan["seed_verification"]["manifest_sha256"])
        if (not proof.get("passed") or not proof.get("strict_state_load") or not proof.get("fixed_state_unchanged")
                or proof["checkpoint_sha256"] != seed["accepted_checkpoint_sha256"]
                or proof["metrics"] != reference["metrics"] or proof["records"] != reference["records"]):
            raise ValueError("Best-task seed needs exact independent fresh verification")
    if not seed["accepted"] or plan["seed_checkpoint_sha256"] != seed["accepted_checkpoint_sha256"] or plan["seed_correct"] != seed["accepted_correct"]:
        raise ValueError("Accepted seed changed")
    seed.update(cut=23,compressed_blocks=13,role="seed_feature_trial")
    rows = [seed]; pending = None; last = seed; population = case_keys(reference)
    for cut in range(22,-1,-1):
        directory = root / "stages" / f"block_{cut:02d}"
        if not directory.exists():
            if any((root / "stages" / f"block_{later:02d}").exists() for later in range(cut)):
                raise ValueError("Non-contiguous stage evidence")
            break
        stage = evidence.plan(directory / "plan.json")
        if (stage.get("root_fingerprint") != plan["fingerprint"] or stage.get("cut") != cut or stage.get("parent_cut") != cut+1
                or stage.get("parent_correct") != last["accepted_correct"] or stage.get("parent_checkpoint_sha256") != last["accepted_checkpoint_sha256"]):
            raise ValueError("Stage must follow the actual accepted parent")
        phases = {}; missing = []
        for phase in PHASES:
            if not (directory / phase / "manifest.json").exists():
                missing.append(phase)
            else:
                if missing:
                    raise ValueError("Later completed phase with missing prerequisites")
                phases[phase] = evidence.phase(directory,phase,stage["fingerprint"])
        if missing:
            if any((root / "stages" / f"block_{later:02d}").exists() for later in range(cut)):
                raise ValueError("Incomplete stage followed by another stage")
            pending = dict(cut=cut,prospective_compressed_blocks=36-cut,status="incomplete_not_accepted",
                completed_phases=list(phases),next_incomplete_phase=missing[0],
                process_liveness="not checked; missing manifest may mean running, paused, or failed")
            break
        trained, values, artifacts = phases["train"]; fresh, verified, verified_artifacts = phases["verify"]
        if case_keys(values["epoch_-1_development.json"]) != population:
            raise ValueError("Stage development cases differ from the accepted seed")
        accepted = fresh["acceptance"]
        row, reference = accepted_summary(trained,values,artifacts,fresh,accepted)
        if (verified.get("acceptance.json") != accepted or verified.get("accepted_development.json") != reference
                or accepted.get("metrics") != reference["metrics"] or accepted.get("records") != reference["records"]
                or not all(accepted.get(k) is True for k in ("strict_state_load","fixed_state_unchanged","exact_packed_replay"))
                or (row["accepted"] and verified_artifacts.get("accepted.pt") != row["accepted_checkpoint_sha256"])
                or (not row["accepted"] and "accepted.pt" in verified_artifacts)):
            raise ValueError("Separate accepted checkpoint/fresh proof mismatch")
        row.update(cut=cut,compressed_blocks=36-cut,role="completed_stage",stage_fingerprint=stage["fingerprint"])
        rows.append(row)
        if not row["accepted"]:
            break
        last = row
    # Refuse mixed snapshots if any already-consumed evidence changed while reading.
    if any(sha(path) != digest for path,digest in evidence.hashes.items()):
        raise ValueError("Evidence changed during summary")
    return dict(schema=SCHEMA,campaign_root=str(root),campaign_fingerprint=plan["fingerprint"],
        status="task_quality_stop" if not rows[-1]["accepted"] else ("incomplete" if last["cut"] else "all_stages_verified_development_only"),
        rows=rows,pending_stage=pending,latest_accepted_cut=last["cut"],accepted_blocks=last["compressed_blocks"],
        remaining_blocks=last["cut"],all_stages_verified=last["cut"]==0,
        latest_accepted_correct=last["accepted_correct"],native_correct=48,minimum_correct=45,count=64,
        confirmed=False,reused_development=True,heldout_measured=False,speed_measured=False,
        validation_scope="Small JSON and producer/archive hashes, selection, replay records and accepted SHA chain. Binary hashes are recorded producer receipts; tensor bytes are not re-read.",
        input_sha256=evidence.hashes,recorded_binary_sha256=evidence.recorded_binary_hashes,
        created_at=datetime.now(timezone.utc).isoformat())


def generate(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output == root or output.is_relative_to(root / "stages") or output.is_relative_to(root / "producer"):
        raise ValueError("Use a new analysis directory outside stage/producer inputs")
    value = summarize(root); output.mkdir(parents=True)
    source = Path(__file__).resolve(); shutil.copy2(source,output / source.name)
    if sha(source)!=sha(output / source.name):raise ValueError("Reporting source changed while archiving")
    for name,content in (("summary.json",value),("manifest.json",dict(schema=SCHEMA,status="completed",
            campaign_fingerprint=value["campaign_fingerprint"],summary_sha256=None,source_sha256=sha(source)))):
        if name == "manifest.json":content["summary_sha256"] = sha(output / "summary.json")
        with (output / name).open("x") as stream:json.dump(content,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    return dict(output=str(output),accepted_blocks=value["accepted_blocks"],status=value["status"],summary_sha256=sha(output / "summary.json"))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path);args=parser.parse_args()
    print(json.dumps(generate(args.root,args.output),allow_nan=False))
