"""Isolated optimizer replay of one completed, frozen staged recovery phase.

This is not a campaign continuation. Each arm starts from identical authenticated
raw/prealigned weights and the original fixed objective, order and selection.
Only the resident arm moves a bounded subset of Adam state to a spare GPU.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import shutil
import time

import torch

from . import jspace_3b_staged_campaign as source
from . import jspace_resident_adamw as resident
from .jspace_recovered_teacher import allocator_settings

base, old, training = source.base, source.old, source.training
SCHEMA = "jspace.resident-adam-replay.v1"
ARMS = ("cpu", "resident")
CONFIG = training.CONFIG
RESIDENT_BUDGET_MIB = 6144
TRANSFER_ACCOUNTING = dict(
    covers_all_process_transfers=False,
    generic_master_fingerprint_downloads_in_counters=False,
    generic_master_fingerprint_downloads_in_update_timers=False,
    scope="Optimizer counters cover only transfers instrumented by the optimizer. Inherited pilot master fingerprints use generic bounded CPU downloads before and after train_window; these are outside both update timers and optimizer counters. Evaluation, fixed-state fingerprints and other process transfers are also excluded. Full phase elapsed time includes these operations.")


def prepare(root, source_root, cut, *, resident_budget_mib=RESIDENT_BUDGET_MIB):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Preparation must hide CUDA")
    root, source_root = Path(root).resolve(), Path(source_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    if type(resident_budget_mib) is not int or not 1 <= resident_budget_mib <= RESIDENT_BUDGET_MIB:
        raise ValueError("Resident state plus work budget must be within 6144 MiB")
    parent, stage = source.verify_stage(source_root, cut)
    directory = source.stage_dir(source_root, cut)
    evidence = {phase: source.checked(directory, phase, stage) for phase in source.PHASES}
    if not evidence["verify"]["passed"]:
        raise ValueError("Replay requires a completed fresh-verified source phase")
    inputs = {}
    for path in (source_root/"plan.json", directory/"plan.json"):
        old.pin(inputs, path)
    for phase in source.PHASES:
        source._pin_phase(inputs, directory, phase)
    # Frozen source plans already authenticate the native/fitted file inventories.
    # Pin that closure here so replay does not depend on a later campaign stage.
    inputs.update(parent["input_sha256"])
    inputs.update(stage["input_sha256"])
    paths = source.local_import_closure([
        Path(__file__), Path(resident.__file__),
        base.ROOT/"tests/structure_net/test_jspace_resident_adamw.py",
        base.ROOT/"tests/structure_net/test_jspace_resident_adam_replay.py"])
    sources = {str(path): base.sha(path) for path in paths}
    plan = dict(schema=SCHEMA, source_root=str(source_root), source_fingerprint=parent["fingerprint"],
        source_stage_fingerprint=stage["fingerprint"], cut=cut, config=asdict(CONFIG), arms=list(ARMS),
        resident_budget_mib=resident_budget_mib, resident_policy="deterministic_parameter_order_first_fit",
        allocations=dict(front_mib=11264, tail_mib=7168, allocator_headroom_mib=256),
        historical_selected=evidence["train"]["selected"],
        historical_warm=evidence["train"]["warm_initial"],
        historical_selected_epoch=evidence["train"]["selected_epoch"],
        input_sha256=inputs, input_identity={p: old.file_identity(p) for p in inputs}, source_sha256=sources,
        diagnostic_checkpoint_policy="highest_correct_then_lowest_ce_earliest; never_auto_adopt",
        confirmed=False, scope="Optimizer-only replay; same reused512train/64dev; no heldout or inference-speed claim")
    plan["fingerprint"] = base.digest(plan)
    root.mkdir(parents=True)
    for path, digest in sources.items():
        destination = root/"producer"/Path(path).relative_to(base.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if base.sha(destination) != digest:
            raise ValueError("Producer archival changed bytes")
    base.write(root/"plan.json", plan)
    return plan


def verify_plan(root):
    root = Path(root)
    plan = source.read(root/"plan.json")
    source.check_plan(plan)
    if (plan.get("schema") != SCHEMA or plan.get("config") != asdict(CONFIG) or
            plan.get("arms") != list(ARMS) or type(plan.get("resident_budget_mib")) is not int or
            not 1 <= plan["resident_budget_mib"] <= RESIDENT_BUDGET_MIB or
            plan.get("allocations") != dict(front_mib=11264, tail_mib=7168, allocator_headroom_mib=256)):
        raise ValueError("Optimizer replay protocol changed")
    for path, digest in plan["source_sha256"].items():
        archive = root/"producer"/Path(path).relative_to(base.ROOT)
        if base.sha(path) != digest or base.sha(archive) != digest:
            raise ValueError("Replay source or archive changed")
    for path, identity in plan["input_identity"].items():
        if old.file_identity(path) != identity:
            raise ValueError("Authenticated replay input changed: "+path)
    parent, stage = source.verify_stage(plan["source_root"], plan["cut"])
    if (parent["fingerprint"] != plan["source_fingerprint"] or
            stage["fingerprint"] != plan["source_stage_fingerprint"]):
        raise ValueError("Source campaign or stage fingerprint changed")
    return plan, parent, stage


def checked(root, arm, phase, plan):
    path = Path(root)/arm/phase
    receipt = source.read(path/"manifest.json")
    if (receipt.get("status") != "completed" or receipt.get("fingerprint") != plan["fingerprint"] or
            receipt.get("arm") != arm or receipt.get("phase") != phase):
        raise ValueError("Completed matching replay receipt required")
    artifacts = receipt.get("artifacts", {})
    if "result.json" not in artifacts:
        raise ValueError("Missing authenticated replay result")
    for name, digest in artifacts.items():
        target = path/name
        if not target.resolve().is_relative_to(path.resolve()) or base.sha(target) != digest:
            raise ValueError("Replay artifact changed: "+name)
    result = source.read(path/"result.json")
    if result.get("status") != "completed":
        raise ValueError("Incomplete replay execution")
    if phase == "pilot":
        probe = result.get("probe", {})
        if (result.get("passed") is not True or result.get("systems_only") is not True or
                result.get("discarded") is not True or result.get("warm_replay_exact") is not True or
                probe.get("passed") is not True or probe.get("fixed_state_unchanged") is not True or
                probe.get("fp32_master_changed_by_block") != [True]*(36-plan["cut"]) or
                probe.get("norm_fp32_master_changed") is not True or
                result.get("warm") != plan["historical_warm"]):
            raise ValueError("Successful discarded optimizer pilot required")
    if phase == "train":
        history = result.get("history", [])
        if ([row.get("epoch") for row in history] != list(range(-1, 13)) or
                result.get("completed_epochs") != 12 or any(
                len(row.get("training", [])) != (0 if row["epoch"] <= 0 else 32) for row in history)):
            raise ValueError("Complete protected/warm/twelve-epoch history required")
        updates = [u for row in history for u in row["training"]]
        if ([u.get("update") for u in updates] != list(range(1,385)) or
                any(u.get("gradients_present_by_block") != [True]*(36-plan["cut"]) for u in updates)):
            raise ValueError("Complete 384 all-block optimizer updates required")
        source._epoch_checks(path, history, artifacts, quality=True)
        winner = training.best_eligible(result["history"], result["initial"])
        if (result.get("config") != asdict(CONFIG) or result.get("updates") != 384 or
                result.get("strict_reload_exact") is not True or result.get("fixed_state_unchanged") is not True or
                result["selected"] != winner["metrics"] or result["selected_epoch"] != winner["epoch"] or
                artifacts.get("selected.pt") != result.get("selected_checkpoint_sha256")):
            raise ValueError("Replay fixed recipe or eligible selection changed")
    if phase == "verify":
        trained = checked(root, arm, "train", plan)
        reference = source.read(Path(root)/arm/"train"/f"epoch_{trained['selected_epoch']:02d}_development.json")
        if (result.get("passed") is not True or result.get("exact_packed_replay") is not True or
                result.get("fixed_state_unchanged") is not True or
                result.get("selected_checkpoint_sha256") != trained["selected_checkpoint_sha256"] or
                result.get("metrics") != reference["metrics"] or result.get("records") != reference["records"]):
            raise ValueError("Fresh selected-checkpoint replay failed")
    return result


class TimedOptimizer:
    """Timing-only adapter; the original optimizer math/state stays untouched."""
    def __init__(self, optimizer, *, devices=(), guard=None):
        self.optimizer, self.devices, self.guard = optimizer, tuple(devices), guard

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def synchronize(self):
        for device in self.devices:
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)

    def step(self, *args, **kwargs):
        # ResidentSpill already checks before/after; CPU arm gets equal checks.
        cpu_arm = type(self.optimizer) is resident.StreamingCPUAdamW
        self.synchronize()
        started = time.perf_counter()
        if cpu_arm and self.guard is not None:
            self.guard()
        result = self.optimizer.step(*args, **kwargs)
        self.synchronize()
        if cpu_arm and self.guard is not None:
            self.guard()
        result = dict(result, synchronized_optimizer_seconds=time.perf_counter()-started,
            synchronized_optimizer_timing_scope="Update plus two reservation checks and completion synchronization, equally in both arms")
        return result


class ReplayTrainer(training.CautiousTrainer):
    def train_window(self, *args, **kwargs):
        self.optimizer.synchronize()
        started = time.perf_counter()
        result = super().train_window(*args, **kwargs)
        self.optimizer.synchronize()
        return dict(result, synchronized_train_window_seconds=time.perf_counter()-started)


class DiagnosticCheckpoint:
    """Record the best task score separately; never affect the protected winner."""
    def __init__(self, directory, trainer, raw, protected):
        self.path, self.trainer, self.raw, self.protected = Path(directory), trainer, raw, protected
        self.best = None

    def observe(self, epoch, metrics):
        if self.best is not None and training.selection_key(metrics) <= training.selection_key(self.best["metrics"]):
            return
        state = self.raw if epoch == -1 else self.trainer.snapshot()
        pending = self.path/"diagnostic_best_task.pt.pending"
        old.save(pending, state)
        os.replace(pending, self.path/"diagnostic_best_task.pt")
        self.best = dict(epoch=epoch, metrics=metrics,
            eligibility=training.candidate_eligibility(metrics, self.protected),
            checkpoint_sha256=base.sha(self.path/"diagnostic_best_task.pt"),
            diagnostic_only=True, automatically_adopted=False)
        base.write(self.path/f"diagnostic_improvement_{epoch:02d}.json", self.best)
        pending_receipt = self.path/"diagnostic_best_task.json.pending"
        base.write(pending_receipt, self.best)
        os.replace(pending_receipt, self.path/"diagnostic_best_task.json")


def make_trainer(stack, norm, E, rotary, data, parent, stage, plan, arm, settings, guard):
    trainer = ReplayTrainer(stack, norm, E, rotary, data, parent["label_ids"], config=CONFIG,
        native_correct=48, parent_correct=stage["parent_correct"], guard=guard)
    if arm == "resident":
        trainer.optimizer = resident.ResidentSpillAdamW.from_cpu_optimizer(trainer.optimizer,
            resident_device=f"cuda:{settings['tail']['device']}",
            resident_budget_bytes=plan["resident_budget_mib"]*1024**2, guard=guard)
    trainer.optimizer = TimedOptimizer(trainer.optimizer,
        devices=[f"cuda:{settings[r]['device']}" for r in ("front", "tail")], guard=guard)
    return trainer


def run(root, arm, phase):
    from .jspace_full_lease import wait_ready
    root = Path(root).resolve()
    plan, parent, stage = verify_plan(root)
    if arm not in ARMS or phase not in ("pilot", "train", "verify"):
        raise ValueError("Unknown replay arm or phase")
    output = root/arm/phase
    if output.exists():
        raise FileExistsError(output)
    if phase == "train":
        checked(root, arm, "pilot", plan)
    if phase == "verify":
        trained = checked(root, arm, "train", plan)
    directory = source.stage_dir(plan["source_root"], plan["cut"])
    pilot = source.checked(directory, "pilot", stage)
    source.checked(directory, "prealign", stage)
    lease = None
    data = None
    settings = None
    started = time.perf_counter()
    try:
        lease = wait_ready([0, 1])  # Admission before any CUDA query/allocation.
        settings = allocator_settings(lease)
        for role in settings.values():
            torch.cuda.reset_peak_memory_stats(role["device"])
        torch.set_num_threads(4)
        torch.manual_seed(CONFIG.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(settings["front"]["device"])
        device = f"cuda:{settings['front']['device']}"

        def guard():
            current = wait_ready([0, 1])
            if current["id"] != lease["id"] or current["allocation"] != lease["allocation"]:
                raise RuntimeError("Optimizer replay reservation changed")
            for role in settings.values():
                if torch.cuda.max_memory_reserved(role["device"]) > role["allocator_limit_bytes"]:
                    raise MemoryError("Optimizer replay exceeded reserved allocator limit")

        stack, norm, E, rotary, _ = source.make_stack(parent, stage, device)
        data = source.data_for(parent, plan["cut"], directory/"capture/packed_teacher_logits.npy")
        raw = source.native._load(directory/"pilot/initializer.pt")
        delta = source.native._load(directory/"prealign/selected_entrance.pt")
        source.validate_delta(raw, delta, pilot, plan["cut"])
        stack.load_state_dict(raw["stack"], strict=True)
        norm.load_state_dict(raw["norm"], strict=True)
        stack.blocks[0].load_state_dict(delta["block"], strict=True)
        del delta
        trainer = make_trainer(stack, norm, E, rotary, data, parent, stage, plan, arm, settings, guard)
        if phase == "pilot":
            output.mkdir(parents=True)
            warm, rows = trainer.packed_development()
            reference = source.read(directory/"train/epoch_00_development.json")
            if warm != reference["metrics"] or rows != reference["records"]:
                raise RuntimeError("Optimizer replay warm initializer differs from frozen source")
            order = sorted(data.training, key=lambda i: data.lengths[i], reverse=True)[:16]
            probe = trainer.master_update_probe(order)
            if not probe["passed"]:
                raise RuntimeError("Resident optimizer gradient/update/fixed-state pilot failed")
            result = dict(status="completed", passed=True, systems_only=True, discarded=True,
                warm_replay_exact=True, warm=warm, records=rows, probe=probe,
                optimizer_memory=trainer.optimizer.memory_stats(),
                optimizer_transfers=dict(trainer.optimizer.transferred_bytes),
                transfer_accounting=TRANSFER_ACCOUNTING)
        elif phase == "train":
            diagnostic = DiagnosticCheckpoint(output, trainer, raw, pilot["initial"])

            def event(kind, **values):
                if kind == "development":
                    keys = pilot["initial"].keys()
                    diagnostic.observe(values["epoch"], {k: values[k] for k in keys})
                base.event(root/"progress.jsonl", kind, arm=arm, **values)

            result = trainer.fit(output, protected_state=raw, protected_metrics=pilot["initial"],
                protected_records=pilot["initial_records"], event=event)
            base.write(output/"optimizer_replay.json", dict(arm=arm,
                state=trainer.optimizer.memory_stats(), transfers=trainer.optimizer.transferred_bytes,
                transfer_accounting=TRANSFER_ACCOUNTING,
                diagnostic_best_task=diagnostic.best, selected_unchanged=True))
        else:
            output.mkdir(parents=True)
            before = training.prior.fixed_state(stack, E, rotary)
            selected = source.native._load(root/arm/"train/selected.pt")
            stack.load_state_dict(selected["stack"], strict=True)
            norm.load_state_dict(selected["norm"], strict=True)
            metrics, rows = trainer.packed_development()
            reference = source.read(root/arm/"train"/f"epoch_{trained['selected_epoch']:02d}_development.json")
            fixed = training.prior.fixed_state(stack, E, rotary) == before
            exact = metrics == reference["metrics"] and rows == reference["records"] and fixed
            if not exact:
                raise RuntimeError("Fresh selected packed replay changed logits or metrics")
            result = dict(status="completed", passed=True, metrics=metrics, records=rows,
                selected_checkpoint_sha256=base.sha(root/arm/"train/selected.pt"),
                exact_packed_replay=True, fixed_state_unchanged=fixed, confirmed=False)
        if not (output/"result.json").exists():
            base.write(output/"result.json", result)
        guard()
        receipt = dict(status="completed", fingerprint=plan["fingerprint"], arm=arm, phase=phase,
            lease_id=lease["id"], allocation=lease["allocation"], allocator=settings,
            elapsed_seconds=time.perf_counter()-started,
            peak_reserved_bytes={k: torch.cuda.max_memory_reserved(v["device"]) for k, v in settings.items()},
            artifacts={str(p.relative_to(output)): base.sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
        base.write(output/"manifest.json", receipt)
        return dict(status="completed", arm=arm, phase=phase, selected=result.get("selected", result.get("metrics")))
    except BaseException as error:
        output.mkdir(parents=True, exist_ok=True)
        base.write(output/"failure.json", dict(status="failed", error=repr(error),
            lease_id=None if lease is None else lease["id"], allocator=settings))
        raise
    finally:
        if data is not None:
            data.close()
        gc.collect()


def compare(root):
    root = Path(root)
    plan, _, _ = verify_plan(root)
    rows = {}
    for arm in ARMS:
        trained = checked(root, arm, "train", plan)
        checked(root, arm, "verify", plan)
        updates = [u for row in trained["history"] for u in row["training"]]
        rows[arm] = dict(selected=trained["selected"], selected_epoch=trained["selected_epoch"],
            quality_pass=trained["quality_pass"], updates=len(updates),
            train_window_seconds=sum(u["synchronized_train_window_seconds"] for u in updates),
            optimizer_seconds=sum(u["optimizer"]["synchronized_optimizer_seconds"] for u in updates),
            total_fit_seconds=trained["elapsed_seconds"],
            diagnostic=source.read(root/arm/"train/diagnostic_best_task.json"))
    value = dict(fingerprint=plan["fingerprint"], arms=rows,
        measured_train_window_ratio=rows["cpu"]["train_window_seconds"]/rows["resident"]["train_window_seconds"],
        measured_optimizer_ratio=rows["cpu"]["optimizer_seconds"]/rows["resident"]["optimizer_seconds"],
        confirmed=False, scope="One matched replay, reused development only; diagnostic weights are not adopted; no inference-speed claim")
    base.write(root/"comparison.json", value)
    return value


def historical_summary(root):
    """Compare quality to preserved science, with only measured pilot timings.

    Never synthesize a CPU-arm training receipt. Historical phase wall time is
    descriptive, not a matched timing baseline for this new implementation.
    """
    root = Path(root)
    plan, _, stage = verify_plan(root)
    pilots = {arm: checked(root, arm, "pilot", plan) for arm in ARMS}
    trained = checked(root, "resident", "train", plan)
    verified = checked(root, "resident", "verify", plan)
    original_dir = source.stage_dir(plan["source_root"], plan["cut"])
    original = source.checked(original_dir, "train", stage)
    reference = source.read(original_dir/"train"/f"epoch_{original['selected_epoch']:02d}_development.json")
    rows = verified["records"]
    if [r["identity"] for r in rows] != [r["identity"] for r in reference["records"]]:
        raise ValueError("Historical and replay development populations differ")
    paired = dict(wins=sum(a["prediction"] == a["answer"] and b["prediction"] != b["answer"]
                           for a,b in zip(rows, reference["records"])),
                  losses=sum(a["prediction"] != a["answer"] and b["prediction"] == b["answer"]
                             for a,b in zip(rows, reference["records"])))
    pilot_times = {arm: dict(train_window_seconds=pilot["probe"]["update"]["synchronized_train_window_seconds"],
        optimizer_seconds=pilot["probe"]["update"]["optimizer"]["synchronized_optimizer_seconds"])
        for arm, pilot in pilots.items()}
    value = dict(fingerprint=plan["fingerprint"], historical_selected=original["selected"],
        resident_selected=trained["selected"], resident_quality_pass=trained["quality_pass"],
        resident_selected_epoch=trained["selected_epoch"], paired=paired, pilot_times=pilot_times,
        pilot_train_window_ratio=pilot_times["cpu"]["train_window_seconds"]/pilot_times["resident"]["train_window_seconds"],
        cpu_full_replay_included=False, matched_full_training_speedup=None,
        diagnostic=source.read(root/"resident/train/diagnostic_best_task.json"), confirmed=False,
        scope="Historical completed quality control plus one discarded longest-window pilot per arm; no full-run timing comparison or automatic adoption")
    base.write(root/"historical_summary.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "train", "verify", "compare", "historical-summary"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--cut", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resident-budget-mib", type=int, default=RESIDENT_BUDGET_MIB)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.source_root is None or args.cut is None:
            parser.error("prepare requires --source-root and --cut")
        result = prepare(args.root, args.source_root, args.cut, resident_budget_mib=args.resident_budget_mib)
    elif args.stage == "compare":
        result = compare(args.root)
    elif args.stage == "historical-summary":
        result = historical_summary(args.root)
    else:
        if args.arm is None:
            parser.error("GPU phase requires --arm")
        result = run(args.root, args.arm, args.stage)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
