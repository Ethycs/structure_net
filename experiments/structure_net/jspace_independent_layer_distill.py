"""Independent single-block distillation; native context, no assembly claim.

Training uses immutable original BF16 teacher boundary pairs. Task scoring
instead uses the matched native FP32 development prefix, this one block, and
the original suffix/readout. Workers share no parameters or student prefixes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from . import jspace_next_layer_distill as sequential

single, recipe, joint = sequential.single, sequential.recipe, sequential.joint
SCHEMA = "jspace.independent-layer-distillation.v1"
PROTOCOL = recipe.ROOT/"docs/07 - Status Reports/jspace/2026-11-14_qwen-parallel-single-layer-recovery.md"
SCOPE = "Development-only isolated block replacement in native FP32 context; no assembled quality, confirmation, or speed claim"


class Trainer(single.Trainer):
    """Original feature objective; retain one best checkpoint and all metrics."""
    def fit(self, directory, *, event=lambda *_a, **_k:None, native_correct=50, minimum_correct=47, layer):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
        history, best, best_epoch, best_records = [], None, None, None
        selected = directory/"selected.pt"
        effective = self.config.microbatch*self.config.accumulation
        for epoch in range(self.config.epochs+1):
            if epoch:
                order = list(self.data.training); random.Random(self.config.seed+epoch).shuffle(order)
                for start in range(0, len(order), effective):
                    event("update", epoch=epoch, **self.train_window(order[start:start+effective]))
            metrics, records = self.development()
            development_path = directory/f"epoch_{epoch:02d}_development.json"
            recipe.write(development_path, dict(metrics=metrics, records=records))
            improved = best is None or (metrics["correct"], -metrics["relative_mse"]) > (best["correct"], -best["relative_mse"])
            row = dict(epoch=epoch, metrics=metrics, checkpoint_updated=improved, development_sha256=recipe.sha(development_path))
            if improved:
                sequential._replace_checkpoint(selected, dict(block=joint._cpu(self.block.state_dict()),
                    epoch=epoch, layer=layer, normalization=self.data.normalization))
                best, best_epoch, best_records = metrics, epoch, records
                row["checkpoint_sha256"] = recipe.sha(selected)
            history.append(row); event("development", epoch=epoch, **metrics)
        self.block.load_state_dict(recipe.load(selected)["block"], strict=True)
        metrics, records = self.development()
        if metrics != best or records != best_records:
            raise RuntimeError("Selected independent block reload changed feature/task results")
        result = dict(status="completed", systems_only=False, selected_epoch=best_epoch,
            selected_checkpoint=str(selected), selected_checkpoint_sha256=recipe.sha(selected), selected=best,
            history=history, updates=self.updates, strict_reload_exact=True, trained_layer=layer,
            native_correct=native_correct, minimum_correct=minimum_correct, quality_pass=best["correct"] >= minimum_correct,
            normalization=self.data.normalization, initializer_selected=best_epoch == 0,
            checkpoint_retention="one atomically replaced best checkpoint; all epoch metrics retained", scope=SCOPE,
            teacher="original native input/output pairs; no other worker states", confirmed=False)
        recipe.write(directory/"result.json", result)
        return result


def validate_quality_cases(items, quality_items):
    development = [c for c in items if c["split"] == "development"]
    if len(development) != len(quality_items) or not development:
        raise ValueError("Matched native development population required")
    cursor = 0
    for expected, actual in zip(development, quality_items):
        if any(actual[k] != expected[k] for k in ("identity", "family", "answer", "choices", "input_ids", "split")):
            raise ValueError("Native quality cache identities, order, tokens, or labels changed")
        if actual["offset"] != cursor or actual["stop"] != cursor+len(actual["input_ids"]):
            raise ValueError("Native quality offsets must partition development tokens")
        cursor = actual["stop"]
    return development


def quality_table(items, quality_items, boundary_path):
    """Virtual row IDs preserve token lengths and rotary positions, without casts."""
    validate_quality_cases(items, quality_items)
    bank = np.load(boundary_path, mmap_mode="r", allow_pickle=False)
    try:
        if bank.dtype != np.float32 or bank.ndim != 2 or len(bank) != quality_items[-1]["stop"]:
            raise ValueError("Complete native FP32 development boundary required")
        table = torch.from_numpy(np.array(bank, copy=True))
        if not torch.isfinite(table).all():
            raise FloatingPointError("Nonfinite native quality prefix")
        cases = [dict(c, input_ids=list(range(c["offset"], c["stop"]))) for c in quality_items]
        return table, cases
    finally:
        bank._mmap.close()


def make_quality(plan, items, directory, guard, *, device="cuda:0"):
    from .jspace_qwen_parallel_model import load_native_block, load_readout_norm
    original = recipe.read(Path(plan["source"])/"plan.json")
    quality_items = recipe.read(Path(plan["quality_prefix_root"])/"items.json")["items"]
    table, cases = quality_table(items, quality_items, plan["quality_boundary_path"])
    embedding, norm, _ = load_readout_norm(original["model"]["path"], device="cpu", dtype=torch.bfloat16)
    embedding, norm = embedding.float(), norm.float()
    rotary, calls = recipe.rotary_module(original, device), 0
    def evaluate(block):
        nonlocal calls
        snapshots = [] if block is None else [deepcopy(block).pack(dtype=torch.float32, buffer_dtype=torch.float32)]
        def loader(local):
            guard()
            if local == 0 and snapshots:
                return snapshots.pop()
            return load_native_block(original["model"]["path"], plan["layer"]+local,
                device=device, dtype=torch.bfloat16).float()
        calls += 1
        result = single.stream_score(cases, input_embedding=table, block_loader=loader,
            final_readout=(norm, embedding), label_ids=original["label_ids"], rotary=rotary,
            scratch_dir=Path(directory)/f"scratch_{calls:02d}", layer_count=36-plan["layer"],
            device=device, guard=guard, microbatch=4)
        result["metrics"].update(max_live_decoder_blocks=2, resident_training_optimizer_included_in_peak=True,
            input_scope="FP32 original native development boundary; virtual IDs are row addresses",
            original_prompt_lengths=[len(c["input_ids"]) for c in quality_items])
        return result
    return evaluate


def native_control(observed, reference, cases, reference_sha256):
    from .jspace_single_block_verify import Tolerances, validate_records
    validate_records(reference["records"], cases); validate_records(observed["records"], cases)
    before, after = reference["records"], observed["records"]
    changed = [a["identity"] for a,b in zip(before, after) if a["prediction"] != b["prediction"]]
    tolerance = Tolerances()
    logits_pass = all(abs(x-y) <= tolerance.logit_atol+tolerance.logit_rtol*abs(x)
        for a,b in zip(before, after) for x,y in zip(a["logits"], b["logits"]))
    return dict(passed=not changed and logits_pass, predictions_exact=not changed,
        logits_pass=logits_pass, logit_atol=tolerance.logit_atol, logit_rtol=tolerance.logit_rtol,
        changed_prediction_identities=changed,
        correct=sum(r["prediction"] == r["answer"] for r in after), count=len(cases),
        reference_sha256=reference_sha256, records=after,
        logit_max_abs_delta=max(abs(x-y) for a,b in zip(before, after) for x,y in zip(a["logits"], b["logits"])),
        scope="Actual native current block and suffix from matched FP32 native prefix; exact predictions and existing verifier logit tolerances required")


def quality_cache(root, original, items, layer):
    """Authenticate only consumed boundary, not37 banks for each independent job."""
    from . import jspace_native_dev_cache as cache
    root = Path(root).resolve(); plan = cache.verify(root, completed=False)
    manifest = recipe.read(root/"cache/manifest.json")
    if (plan.get("schema") != "jspace.native-fp32-development-cache.v1"
            or plan["source_fingerprint"] != original["fingerprint"]
            or manifest.get("status") != "completed" or manifest.get("fingerprint") != plan["fingerprint"]):
        raise ValueError("Completed matching native FP32 quality cache required")
    inputs = {}
    for relative in ("items.json", "cache/result.json", f"cache/boundary_{layer:02d}.npy"):
        path = root/relative; expected = manifest["artifact_sha256"][relative]
        if recipe.sha(path) != expected:
            raise ValueError("Changed consumed native quality artifact")
        inputs[str(path)] = expected
    quality_items = recipe.read(root/"items.json")["items"]
    cases = validate_quality_cases(items, quality_items)
    reference = recipe.read(root/"cache/result.json")
    from .jspace_single_block_verify import validate_records
    validate_records(reference["records"], cases)
    if (reference.get("status") != "completed" or reference.get("fingerprint") != plan["fingerprint"]
            or reference.get("correct") != 50 or reference.get("count") != 64
            or reference.get("native_baseline_predictions_exact") is not True
            or reference.get("native_baseline_prediction_agreement") != 1.0
            or sum(r["prediction"] == r["answer"] for r in reference["records"]) != 50):
        raise ValueError("Matched pure native50/64 predictions required")
    for path in (root/"plan.json", root/"cache/manifest.json"):
        inputs[str(path)] = recipe.sha(path)
    return plan, inputs


def prepare(root, source, basis_root, quality_prefix_root, layer):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA for CPU preparation")
    if type(layer) is not int or not 2 <= layer < 36:
        raise ValueError("Independent layers2..35 only; preserve existing block0/block1 studies")
    root, source, basis_root, quality_prefix_root = map(lambda p:Path(p).resolve(), (root, source, basis_root, quality_prefix_root))
    if root.exists():
        raise FileExistsError(root)
    original = recipe.verify(source); items = recipe.read(source/"items.json")
    if (sum(c["split"] == "train" for c in items["items"]) != 512
            or sum(c["split"] == "development" for c in items["items"]) != 64
            or any(len(c["input_ids"]) > 384 for c in items["items"])):
        raise ValueError("Original512 training and64 development cases required")
    fitplan = recipe.read(basis_root/"plan.json"); fits = recipe.read(basis_root/"basis/manifest.json")
    if (fitplan.get("schema") != "jspace.all-layer-enriched-basis.v1"
            or fitplan["fingerprint"] != recipe.digest({k:v for k,v in fitplan.items() if k != "fingerprint"})
            or fitplan["source_fingerprint"] != original["fingerprint"]
            or fits.get("status") != "completed" or fits["fingerprint"] != fitplan["fingerprint"]):
        raise ValueError("Completed matching all-layer enriched geometry required")
    training = {c["identity"] for c in items["items"] if c["split"] == "train"}
    fit_ids, check_ids = set(fitplan["fit_training_identities"]), set(fitplan["internal_training_check_identities"])
    if not fit_ids or not check_ids or fit_ids & check_ids or not (fit_ids | check_ids) <= training:
        raise ValueError("Geometry must be fitted and checked only on disjoint training cases")
    for path, expected in fitplan["source_sha256"].items():
        if recipe.sha(path) != expected or recipe.sha(basis_root/"producer"/Path(path).relative_to(recipe.ROOT)) != expected:
            raise ValueError("Changed enriched basis producer")
    quality_plan, inputs = quality_cache(quality_prefix_root, original, items["items"], layer)
    fit_path = basis_root/"basis"/f"block_{layer:02d}/fit.pt"
    capture = Path(original["teacher_root"])/"capture"
    native_input, target = capture/f"boundary_{layer:02d}.npy", capture/f"boundary_{layer+1:02d}.npy"
    inputs.update({str(fit_path):fits["artifacts"][str(fit_path.relative_to(basis_root))],
        str(native_input):original["input_sha256"][str(native_input)], str(target):original["input_sha256"][str(target)]})
    for path in (source/"plan.json", source/"items.json", basis_root/"plan.json", basis_root/"basis/manifest.json"):
        inputs[str(path)] = recipe.sha(path)
    for path, expected in inputs.items():
        if recipe.sha(path) != expected:
            raise ValueError("Changed teacher, native quality prefix, or layer geometry")
    sources = dict(original["source_sha256"]); sources.update(fitplan["source_sha256"]); sources.update(quality_plan["source_sha256"])
    for path in [Path(__file__).resolve(), Path(sequential.__file__).resolve(), Path(single.__file__).resolve(),
                 Path(__file__).with_name("jspace_single_block_verify.py"), Path(__file__).with_name("jspace_single_block_reserved.py"),
                 Path(__file__).with_name("jspace_semantic_recovery.py"), Path(__file__).with_name("jspace_semantic_loss.py"),
                 Path(__file__).with_name("jspace_joint_recovery.py"), Path(__file__).with_name("jspace_residual_carry.py"),
                 Path(__file__).with_name("jspace_quadratic_evaluate.py"), PROTOCOL,
                 recipe.ROOT/"tests/structure_net/test_jspace_independent_layer_distill.py"]:
        sources[str(path)] = recipe.sha(path)
    plan = dict(schema=SCHEMA, source=str(source), source_fingerprint=original["fingerprint"], layer=layer,
        trained_layers=[layer], native_prefix=list(range(layer)), native_suffix=list(range(layer+1, 36)),
        basis_root=str(basis_root), basis_fingerprint=fitplan["fingerprint"], fit_path=str(fit_path),
        native_input_path=str(native_input), target_path=str(target), config=asdict(single.Config()),
        quality_prefix_root=str(quality_prefix_root), quality_prefix_fingerprint=quality_plan["fingerprint"],
        quality_boundary_path=str(quality_prefix_root/"cache"/f"boundary_{layer:02d}.npy"),
        quality_reference_path=str(quality_prefix_root/"cache/result.json"),
        quality_reference_sha256=inputs[str(quality_prefix_root/"cache/result.json")],
        items_digest=recipe.digest(items), native_correct=50, minimum_correct=47,
        rank=1024, head_dim=128, intermediate_size=11008, confirmation_used=False,
        teacher="Original native BF16 input/output boundary pairs; no student-prefix inputs or other worker weights",
        task_input="Separate actual native FP32 development prefix; native substitution predictions required before training",
        protocol=str(PROTOCOL), scope=SCOPE, input_sha256=inputs,
        input_identity={p:recipe.file_identity(p) for p in inputs}, source_sha256=sources)
    plan["fingerprint"] = recipe.digest(plan); root.mkdir(parents=True)
    for path, expected in sources.items():
        if recipe.sha(path) != expected:
            raise ValueError("Changed inherited producer")
        destination = root/"producer"/Path(path).relative_to(recipe.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
    recipe.write(root/"plan.json", plan); recipe.write(root/"items.json", items)
    return dict(status="prepared", layer=layer, fingerprint=plan["fingerprint"])


def verify(root):
    root = Path(root); plan = recipe.read(root/"plan.json")
    if plan.get("schema") != SCHEMA or plan["fingerprint"] != recipe.digest({k:v for k,v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed independent-layer plan")
    original = recipe.verify(plan["source"])
    if original["fingerprint"] != plan["source_fingerprint"] or recipe.digest(recipe.read(root/"items.json")) != plan["items_digest"]:
        raise ValueError("Changed original teacher or case alignment")
    for path, expected in plan["source_sha256"].items():
        if recipe.sha(path) != expected or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT)) != expected:
            raise ValueError("Changed independent-layer producer")
    for path, identity in plan["input_identity"].items():
        if recipe.file_identity(path) != identity:
            raise ValueError("Changed frozen teacher, geometry, or native quality cache")
    return plan


def verify_pilot(root, plan):
    joint.verify_pilot(root, plan["fingerprint"])
    control = recipe.read(Path(root)/"pilot/result.json")["native_control"]
    if (control.get("passed") is not True or control.get("predictions_exact") is not True or control.get("logits_pass") is not True
            or control.get("correct") != 50 or control.get("count") != 64
            or control.get("reference_sha256") != plan["quality_reference_sha256"]):
        raise ValueError("Successful matching native substitution control required before training")


def run(root, execution_stage):
    if execution_stage not in ("pilot", "train", "verify"):
        raise ValueError("Explicit pilot/train/verify stages only")
    root = Path(root).resolve(); plan = verify(root)
    directory = root/("verification" if execution_stage == "verify" else execution_stage)
    if directory.exists():
        raise FileExistsError(directory)
    if execution_stage == "train":
        verify_pilot(root, plan)
    selected = sequential.completed_training(root, plan) if execution_stage == "verify" else None
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_native_block
    lease = data = None; cuda_started = False; started = time.monotonic()
    guard_telemetry = dict(calls=0, wall_seconds=0.)
    try:
        lease = wait_ready([0]); allocation = dict(lease["allocation"])
        def guard():
            began = time.monotonic()
            try:
                current = wait_ready([0])
                if current["id"] != lease["id"] or current["allocation"] != allocation:
                    raise RuntimeError("Independent layer reservation changed")
                joint.check_reserved_peak(allocation, torch.cuda.max_memory_reserved())
            finally:
                guard_telemetry["calls"] += 1
                guard_telemetry["wall_seconds"] += time.monotonic()-began
        allocator = sequential.cap_allocator(lease)
        torch.set_num_threads(4); torch.manual_seed(plan["config"]["seed"])
        torch.cuda.init(); cuda_started = True; torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        original = recipe.read(Path(plan["source"])/"plan.json"); items = recipe.read(root/"items.json")["items"]
        native = load_native_block(original["model"]["path"], plan["layer"], device="cuda:0", dtype=torch.bfloat16)
        block = single.build_enriched_block(native, recipe.load(plan["fit_path"]), parameter_dtype=torch.float32)
        del native
        if ((block.width, block.rank, block.head_dim, block.intermediate_size) != (2048, 1024, 128, 11008)
                or len(list(block.parameters())) != 7):
            raise ValueError("Fixed half-rank/full-head/seven-tensor architecture required")
        data = single.NativePairs(items, plan["native_input_path"], plan["target_path"])
        config = single.Config(**plan["config"]); quality = make_quality(plan, items, directory, guard)
        trainer = Trainer(block, recipe.rotary_module(original, "cuda:0"), data, quality, config=config, guard=guard)
        if execution_stage == "pilot":
            directory.mkdir(exist_ok=False)
            reference = recipe.read(plan["quality_reference_path"])
            cases = [c for c in items if c["split"] == "development"]
            control = native_control(quality(None), reference, cases, plan["quality_reference_sha256"])
            recipe.write(directory/"native_control.json", control)
            if not control["passed"] or control["correct"] != 50:
                raise RuntimeError("Native substitution control failed; no scientific training authorized")
            initial, _ = trainer.development()
            order = sorted(data.training, key=lambda i:len(items[i]["input_ids"]), reverse=True)
            update = trainer.train_window(order[:config.microbatch*config.accumulation])
            metrics, _ = trainer.development()
            result = dict(status="completed", systems_only=True, updates=1, update=update, native_control=control,
                initial=initial, development=metrics, normalization=data.normalization, scope=SCOPE)
            recipe.write(directory/"result.json", result)
        elif execution_stage == "train":
            result = trainer.fit(directory, layer=plan["layer"],
                event=lambda kind, **v:recipe.event(root, kind, stage=execution_stage, **v))
        else:
            from .jspace_single_block_verify import compare
            directory.mkdir(exist_ok=False)
            checkpoint = recipe.load(root/"train/selected.pt")
            if (checkpoint["epoch"] != selected["selected_epoch"] or checkpoint["layer"] != plan["layer"]
                    or checkpoint["normalization"] != data.normalization):
                raise ValueError("Selected epoch/layer/original-teacher normalization changed")
            block.load_state_dict(checkpoint["block"], strict=True)
            metrics, records = trainer.development()
            reference = recipe.read(root/"train"/f"epoch_{checkpoint['epoch']:02d}_development.json")
            with torch.no_grad():
                keys = trainer.losses(data.batch(data.development[:1], "cuda:0"), diagnostics=True).keys()
            comparison = compare(reference, {k:metrics[k] for k in keys}, records,
                [c for c in items if c["split"] == "development"])
            result = dict(status="completed", fingerprint=plan["fingerprint"], campaign_fingerprint=plan["fingerprint"],
                checkpoint_sha256=recipe.sha(root/"train/selected.pt"), selected_epoch=checkpoint["epoch"],
                passed=comparison["passed"], strict_state_load=True, comparison=comparison, records=records,
                training_quality_pass=selected["quality_pass"], confirmed=False, process_id=os.getpid(), scope=SCOPE)
            recipe.write(directory/"result.json", result)
            if not result["passed"]:
                raise RuntimeError("Independent layer fresh-process selected parity failed")
        guard()
        manifest = dict(status="completed", systems_only=execution_stage == "pilot", fingerprint=plan["fingerprint"],
            result_sha256=recipe.sha(directory/"result.json"), lease_id=lease["id"], allocation=allocation,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), allocator=allocator,
            reservation_guard=guard_telemetry, elapsed_seconds=time.monotonic()-started)
        if execution_stage == "verify":
            manifest.update(passed=result["passed"], checkpoint_sha256=result["checkpoint_sha256"])
        recipe.write(directory/"manifest.json", manifest)
        return dict(status="completed", stage=execution_stage, layer=plan["layer"],
            quality_pass=result.get("quality_pass", result.get("training_quality_pass")), fingerprint=plan["fingerprint"])
    except Exception as error:
        directory.mkdir(parents=True, exist_ok=True)
        recipe.write(directory/"failure.json", dict(status="failed", fingerprint=plan["fingerprint"],
            error_type=type(error).__name__, error=str(error), lease_id=None if lease is None else lease["id"],
            allocation=None if lease is None else lease["allocation"], reservation_guard=guard_telemetry,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None))
        raise
    finally:
        if data is not None:
            data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "train", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source", type=Path); parser.add_argument("--basis-root", type=Path)
    parser.add_argument("--quality-prefix-root", type=Path); parser.add_argument("--layer", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        if any(v is None for v in (args.source, args.basis_root, args.quality_prefix_root, args.layer)):
            parser.error("prepare requires --source --basis-root --quality-prefix-root --layer")
        value = prepare(args.root, args.source, args.basis_root, args.quality_prefix_root, args.layer)
    else:
        value = run(args.root, args.stage)
    print(json.dumps(value, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
