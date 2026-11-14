"""Reserved three-GPU compatibility search over fixed recovered Qwen blocks."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_layer_combination_policy as policy
from . import jspace_layer_combination_runtime as runtime
from . import jspace_recovered_assembly as assembly

recipe = assembly.recipe
SCHEMA = "jspace.layer-combinations.v1"
STUDY = "JSPACE-QWEN-LAYER-COMBINATIONS-01"
PROTOCOL = recipe.ROOT/"docs/07 - Status Reports/jspace/2026-11-14_qwen-layer-combination-repair.md"
PARENT_FINGERPRINT = "987f470548540a645a2b0f1d3bc8fedc1fe043892591f7630e7b10a5fc0928a6"


def prepare(root, assembly_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("CPU preparation requires hidden CUDA")
    root, assembly_root = Path(root).resolve(), Path(assembly_root).resolve()
    if root.exists(): raise FileExistsError(root)
    parent = assembly.verify(assembly_root)
    if parent["fingerprint"] != PARENT_FINGERPRINT: raise ValueError("Wrong independent recovered assembly")
    inputs, sources = dict(parent["input_sha256"]), dict(parent["source_sha256"])
    references = {}
    for arm in assembly.ARMS:
        value = assembly.completed_arm(assembly_root, parent, arm)
        for name in ("result.json", "manifest.json"):
            assembly._pin(inputs, assembly_root/arm/name)
        references[arm] = str(assembly_root/arm/"result.json")
    assembly._pin(inputs, assembly_root/"plan.json")
    for path in [Path(__file__), Path(runtime.__file__), Path(policy.__file__), PROTOCOL,
                 recipe.ROOT/"tests/structure_net/test_jspace_layer_combination_runtime.py",
                 recipe.ROOT/"tests/structure_net/test_jspace_layer_combination_policy.py",
                 recipe.ROOT/"tests/structure_net/test_jspace_layer_combination_search.py"]:
        sources[str(path.resolve())] = recipe.sha(path)
    plan = dict(schema=SCHEMA, study_id=STUDY, assembly_root=str(assembly_root),
        assembly_fingerprint=parent["fingerprint"], cases=parent["cases"], label_ids=recipe.read(Path(parent["source"])/"plan.json")["label_ids"],
        references=references, microbatch=4, cards=runtime.CARDS, partition=runtime.PARTITION,
        candidate_count=6, repair_group_size=3, minimum_correct=47, original_all_compact_correct=27,
        native_correct=50, input_sha256=inputs, input_identity={p:recipe.file_identity(p) for p in inputs},
        source_sha256=sources, protocol=str(PROTOCOL), confirmed=False,
        scope="Exploratory conditional combination search on original 64 development cases; no global optimum or confirmation claim")
    plan["fingerprint"] = recipe.digest(plan)
    root.mkdir(parents=True)
    for path in sources:
        target = root/"producer"/Path(path).relative_to(recipe.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    recipe.write(root/"plan.json", plan)
    return dict(status="prepared", fingerprint=plan["fingerprint"], screen_masks=len(policy.screen_masks()))


def verify(root):
    root = Path(root).resolve(); plan = recipe.read(root/"plan.json")
    if plan.get("schema") != SCHEMA or plan["fingerprint"] != recipe.digest({k:v for k,v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed search plan")
    if assembly.verify(plan["assembly_root"])["fingerprint"] != plan["assembly_fingerprint"]:
        raise ValueError("Changed recovered assembly")
    for path, digest in plan["source_sha256"].items():
        if recipe.sha(path) != digest or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT)) != digest:
            raise ValueError("Changed search producer: "+path)
    for path, identity in plan["input_identity"].items():
        if recipe.file_identity(path) != identity: raise ValueError("Changed pinned search input: "+path)
    return plan


def normalized(value):
    metrics = {k:v for k,v in value.items() if k != "records"}
    metrics["teacher_agreement_count"] = sum(r["prediction"] == r["teacher_prediction"] for r in value["records"])
    return dict(records=value["records"], metrics=metrics)


def control_masks():
    return {arm:policy.compressed(assembly.compact_layers(arm)) for arm in assembly.ARMS}


def checked_controls(results, plan):
    expected = dict(native=50, accepted_two=49, all_compact=27, hybrid=24)
    controls = {}
    for arm, mask in control_masks().items():
        controls[arm] = assembly.compare_control(results[policy.mask_id(mask)]["records"],
            recipe.read(plan["references"][arm]), plan["cases"], expected[arm])
    if not all(v["passed"] for v in controls.values()): raise ValueError("Native/compact control mismatch")
    return controls


def validate_results(results, plan):
    """Recompute selection metrics from paired decisions, not trusted summaries."""
    native = recipe.read(plan["references"]["native"])["records"]
    for key, result in results.items():
        if len(key) != 37 or key[0] != "m" or set(key[1:]) - {"0", "1"}:
            raise ValueError("Malformed result mask identity")
        records, metrics = result["records"], result["metrics"]
        assembly.validate_records(records, plan["cases"])
        correct = agreement = 0; ce = 0.
        for row, teacher in zip(records, native):
            if row["teacher_prediction"] != teacher["prediction"]:
                raise ValueError("Changed native decision reference")
            scores = row["logits"]; peak = max(scores)
            if row["prediction"] != max(range(len(scores)), key=lambda i:scores[i]):
                raise ValueError("Prediction differs from saved logits")
            loss = math.log(sum(math.exp(x-peak) for x in scores)) + peak - scores[row["answer"]]
            if not math.isclose(row["choice_ce"], loss, abs_tol=1e-4, rel_tol=1e-5):
                raise ValueError("Choice loss differs from saved logits")
            ce += row["choice_ce"]
            correct += row["prediction"] == row["answer"]
            agreement += row["prediction"] == teacher["prediction"]
        if (len(records) != 64 or metrics["count"] != 64 or metrics["correct"] != correct
                or metrics["teacher_agreement_count"] != agreement
                or metrics["accuracy"] != correct/64 or metrics["teacher_agreement"] != agreement/64
                or not math.isclose(metrics["choice_ce"], ce/64, abs_tol=1e-10, rel_tol=1e-10)):
            raise ValueError("Aggregate mask scores differ from saved decisions")


def verify_pilot(root, plan):
    root = Path(root); result = recipe.read(root/"pilot/result.json"); manifest = recipe.read(root/"pilot/manifest.json")
    if (result.get("status") != "completed" or result.get("passed") is not True or result.get("systems_only") is not True
            or result.get("fixed_state_unchanged") is not True or manifest.get("passed") is not True
            or manifest.get("status") != "completed" or manifest.get("fingerprint") != plan["fingerprint"]
            or manifest.get("result_sha256") != recipe.sha(root/"pilot/result.json")):
        raise ValueError("Successful matching systems pilot required")
    return result


def run(root, stage):
    from .jspace_full_lease import wait_ready
    if stage not in ("pilot", "search"): raise ValueError("Explicit pilot/search required")
    root = Path(root).resolve(); plan = verify(root); directory = root/stage
    if directory.exists(): raise FileExistsError(directory)
    if stage == "search": verify_pilot(root, plan)
    directory.mkdir()
    started = time.monotonic(); lease = settings = None
    telemetry = dict(calls=0, controller_reads=0, controller_seconds=0.)
    try:
        lease = wait_ready([0,1,2]); settings = runtime.allocator_settings(lease)
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
        for row in settings["settings"]: torch.cuda.reset_peak_memory_stats(row["logical_device"])
        next_check = 0.
        def guard(force=False):
            nonlocal next_check
            telemetry["calls"] += 1
            now = time.monotonic()
            if not force and now < next_check: return
            t = time.monotonic(); current = wait_ready([0,1,2])
            telemetry["controller_reads"] += 1; telemetry["controller_seconds"] += time.monotonic()-t
            if current["id"] != lease["id"] or current["allocation"] != lease["allocation"]:
                raise RuntimeError("Search reservation changed")
            runtime.device_peaks(settings); next_check = time.monotonic()+1.
        parent = assembly.verify(plan["assembly_root"])
        bank = runtime.load_bank(parent, settings["devices"], guard=guard)
        before = runtime.state_digest(bank, guard=guard)
        native_records = recipe.read(plan["references"]["native"])["records"]
        controls_requested = {policy.mask_id(m):m for m in control_masks().values()}
        def progress(**value):
            if value["kind"] == "batch": recipe.event(root, "progress", stage=stage, detail=value)
        def score(masks, cases=None, teacher=None, shared=True):
            guard(force=True); t = time.monotonic()
            value = runtime.score_masks(bank, plan["cases"] if cases is None else cases, masks,
                plan["label_ids"], teacher_records=native_records if teacher is None else teacher,
                microbatch=plan["microbatch"], share_prefixes=shared, guard=guard, progress=progress)
            value["elapsed_seconds"] = time.monotonic()-t
            return value
        if stage == "pilot":
            indices = sorted(range(len(plan["cases"])), key=lambda i:len(plan["cases"][i]["input_ids"]), reverse=True)[:4]
            cases = [plan["cases"][i] for i in indices]; teachers = [native_records[i] for i in indices]
            masks = dict(controls_requested)
            # All singleton branches exercise the full trie depth and retained
            # ancestor-state stack on the longest-context batch.
            for layer in range(36):
                m = policy.restored((layer,)); masks[policy.mask_id(m)] = m
            for group in (tuple(range(6)), tuple(range(0,36,6)), (0,), (0,1), (17,18), (28,29), (34,35)):
                m = policy.restored(group); masks[policy.mask_id(m)] = m
            shared = score(masks, cases, teachers); independent = score(masks, cases, teachers, shared=False)
            errors = []
            for key in masks:
                a, b = shared["results"][key]["records"], independent["results"][key]["records"]
                for x,y in zip(a,b):
                    if x["prediction"] != y["prediction"] or any(abs(u-v)>5e-4+1e-5*abs(v) for u,v in zip(x["logits"],y["logits"])):
                        errors.append(dict(mask=key, identity=x["identity"]))
            result = dict(status="completed", passed=not errors, systems_only=True,
                parity_errors=errors, longest_case_identities=[c["identity"] for c in cases],
                shared_execution=shared["execution"], independent_execution=independent["execution"],
                shared_seconds=shared["elapsed_seconds"], independent_seconds=independent["elapsed_seconds"])
        else:
            control = score(controls_requested)
            results = {k:normalized(v) for k,v in control["results"].items()}
            controls = checked_controls(results, plan)
            recipe.write(directory/"controls.json", controls)
            requested = policy.screen_masks()
            remaining = {k:v for k,v in requested.items() if k not in results}
            screen = score(remaining)
            results.update({k:normalized(v) for k,v in screen["results"].items()})
            recipe.write(directory/"screen.json", dict(results=results, execution=screen["execution"], elapsed_seconds=screen["elapsed_seconds"]))
            candidates = policy.select_candidates(results)
            recipe.write(directory/"candidates.json", candidates)
            recipe.event(root, "candidates", **candidates)
            remaining = {k:v for k,v in policy.refinement_masks(candidates["layers"]).items() if k not in results}
            refinement = score(remaining)
            results.update({k:normalized(v) for k,v in refinement["results"].items()})
            validate_results(results, plan)
            summary = policy.summarize(results, candidates)
            recipe.write(directory/"selection.json", summary["selected"])
            result = dict(status="completed", passed=True, systems_only=False, fingerprint=plan["fingerprint"],
                summary=summary, results=results, controls=controls, confirmed=False,
                telemetry=dict(controls=control["execution"], screen=screen["execution"], refinement=refinement["execution"],
                    control_seconds=control["elapsed_seconds"], screen_seconds=screen["elapsed_seconds"], refinement_seconds=refinement["elapsed_seconds"]))
        after = runtime.state_digest(bank, guard=guard)
        result.update(fixed_state_unchanged=before == after, fixed_state=before, storage=bank.storage_inventory())
        result["passed"] = result["passed"] and before == after
        if not result["passed"]: result["status"] = "failed"
        recipe.write(directory/"result.json", result)
        if not result["passed"]: raise RuntimeError("Search state changed or execution parity failed")
        guard(force=True)
        manifest = dict(status="completed", passed=True, fingerprint=plan["fingerprint"], result_sha256=recipe.sha(directory/"result.json"),
            lease_id=lease["id"], allocation=lease["allocation"], settings=settings, device_peaks=runtime.device_peaks(settings),
            guard=telemetry, elapsed_seconds=time.monotonic()-started)
        if stage == "search":
            manifest["selection_sha256"] = recipe.sha(directory/"selection.json")
            manifest["screen_sha256"] = recipe.sha(directory/"screen.json")
            manifest["candidates_sha256"] = recipe.sha(directory/"candidates.json")
        recipe.write(directory/"manifest.json", manifest)
        return dict(status="completed", stage=stage, fingerprint=plan["fingerprint"],
            selected_layers=result.get("summary",{}).get("selected",{}).get("layers"))
    except BaseException as error:
        recipe.write(directory/"failure.json", dict(status="failed", error=repr(error), fingerprint=plan["fingerprint"],
            lease_id=None if lease is None else lease["id"], allocation=None if lease is None else lease["allocation"], guard=telemetry))
        raise


def verified_selection(root):
    root = Path(root).resolve(); plan = verify(root); verify_pilot(root, plan)
    m = recipe.read(root/"search/manifest.json"); result = recipe.read(root/"search/result.json")
    selected = recipe.read(root/"search/selection.json")
    if (m.get("status") != "completed" or m.get("passed") is not True or m.get("fingerprint") != plan["fingerprint"]
            or m["result_sha256"] != recipe.sha(root/"search/result.json") or m["selection_sha256"] != recipe.sha(root/"search/selection.json")
            or result.get("status") != "completed" or result.get("fixed_state_unchanged") is not True
            or result.get("fingerprint") != plan["fingerprint"] or result.get("passed") is not True
            or result.get("systems_only") is not False):
        raise ValueError("Completed authenticated search required")
    validate_results(result["results"], plan)
    candidates = policy.select_candidates(result["results"])
    if set(result["results"]) != set(policy.screen_masks()) | set(policy.refinement_masks(candidates["layers"])):
        raise ValueError("Unexpected or missing searched masks")
    recomputed = policy.selection(result["results"], candidates["layers"])
    if (recomputed != selected or selected != result["summary"]["selected"]
            or policy.summarize(result["results"], candidates) != result["summary"]):
        raise ValueError("Repair group differs from fixed combination selection")
    checked_controls(result["results"], plan)
    return dict(selected_layers=selected["layers"], assembly_fingerprint=plan["assembly_fingerprint"],
        search_fingerprint=plan["fingerprint"], selection_sha256=m["selection_sha256"], summary_sha256=m["result_sha256"],
        assembly_root=plan["assembly_root"], original_all_compact_correct=27, native_correct=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "search")); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assembly-root", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare" and args.assembly_root is None: parser.error("prepare needs --assembly-root")
    print(json.dumps(prepare(args.root, args.assembly_root) if args.stage == "prepare" else run(args.root, args.stage), allow_nan=False), flush=True)
