"""Two-GPU FP32 joint recovery from the actual selected36-block assembly.

No basis or projection is refitted. The student supplies its own intermediate
states; matched original FP32 teacher logits and answer labels supervise the
midpoint and answer. Only block parameters and final RMS gain may change.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from . import jspace_recovered_assembly as assembly
from . import jspace_semantic_recovery as semantic
from .jspace_semantic_loss import SemanticLossConfig
from .jspace_streaming_adamw import StreamingCPUAdamW

joint, recipe = assembly.joint, assembly.recipe
SCHEMA = "jspace.recovered-joint.v1"
PROTOCOL = recipe.ROOT/"docs/07 - Status Reports/jspace/2026-11-14_qwen-recovered-joint-distillation.md"
CARDS = {"3060":("GPU-0370fdb0", 11264), "2060_SUPER":("GPU-7b1db3cb", 7168)}
SCOPE = "Development-only joint recovery of the selected assembly; no confirmation or deployment-speed claim"


@dataclass(frozen=True)
class Config:
    epochs: int = 3
    microbatch: int = 1
    accumulation: int = 16
    learning_rate: float = 1e-5
    gradient_clip: float = 1.
    seed: int = 27865
    optimizer_chunk_bytes: int = 32*1024**2
    max_cpu_state_bytes: int = 24*1024**3
    objective: SemanticLossConfig = field(default_factory=lambda:SemanticLossConfig(
        temperature=2., full_kl_weight=.2, choice_kl_weight=.5, choice_ce_weight=.3))

    def __post_init__(self):
        joint.JointConfig(epochs=self.epochs, microbatch=self.microbatch, accumulation=self.accumulation,
            learning_rate=self.learning_rate, gradient_clip=self.gradient_clip, seed=self.seed,
            optimizer_chunk_bytes=self.optimizer_chunk_bytes, max_cpu_state_bytes=self.max_cpu_state_bytes)
        if not isinstance(self.objective, SemanticLossConfig): raise ValueError("Validated semantic objective required")


def config_from_dict(value):
    value = dict(value); value["objective"] = SemanticLossConfig(**value["objective"])
    return Config(**value)


class PartitionedCarryModel(joint.JointCarryModel):
    """Transfer residuals only at partition boundaries; selected rows return to E."""
    def __init__(self, embedding, blocks, norm_gain, epsilon, rotary, *, split=24, checkpoint_blocks=True):
        super().__init__(embedding, blocks, norm_gain, epsilon, rotary, checkpoint_blocks=checkpoint_blocks)
        if type(split) is not int or not 0 < split < len(blocks): raise ValueError("Interior fixed partition required")
        self.split = split
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in [*self.parameters(), *self.buffers()]):
            raise ValueError("Recovered continuation retains all weights and geometry inFP32")
        devices = [next(b.parameters()).device for b in blocks]
        if (any(d != devices[0] for d in devices[:split]) or any(d != devices[split] for d in devices[split:])
                or devices[0] != embedding.device or self.norm_gain.device != embedding.device):
            raise ValueError("Two contiguous partitions and readout on the first device required")

    def forward(self, input_ids, position_ids, attention_mask, loss_positions):
        hidden = F.embedding(input_ids, self.embedding)
        base_rope = self.rotary(hidden, position_ids)
        contexts = {}
        for block in self.blocks:
            device = next(block.parameters()).device
            if device not in contexts:
                contexts[device] = (position_ids.to(device), attention_mask.to(device),
                    tuple(value.to(device) for value in base_rope))
            positions, mask, rope = contexts[device]
            # Device copies stay OUTSIDE checkpoint closures so each closure
            # executes and replays on the device of its input and parameters.
            hidden = hidden.to(device)
            def apply(value, current=block, ids=positions, allowed=mask, embedding=rope):
                return current(value, attention_mask=allowed, position_ids=ids, position_embeddings=embedding)
            hidden = checkpoint(apply, hidden, use_reentrant=False) if (
                self.training and self.checkpoint_blocks and torch.is_grad_enabled()) else apply(hidden)
        rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        selected = hidden[rows, loss_positions.to(hidden.device)].to(self.embedding.device)
        normalized = selected*torch.rsqrt(selected.square().mean(-1, keepdim=True)+self.epsilon)
        return F.linear(normalized*self.norm_gain, self.embedding)


def allocator_settings(lease, *, cuda=None, visible=None):
    """Resolve physical UUIDs before CUDA initialization; never assume slot order."""
    cuda = torch.cuda if cuda is None else cuda
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "") if visible is None else visible
    ids, allocation = visible.split(","), lease["allocation"]
    if len(ids) != 2 or len(allocation) != 2 or cuda.is_initialized():
        raise ValueError("Fresh process with exactly two visible leased GPU UUIDs required")
    resolved = []
    for token in ids:
        matches = [uuid for uuid in allocation if token.startswith("GPU-") and uuid.startswith(token)]
        if len(matches) != 1: raise ValueError("Visible GPU must resolve to exactly one leased physical UUID")
        resolved.append(matches[0])
    if len(set(resolved)) != 2: raise ValueError("Two distinct physical GPUs required")
    settings, devices = [], {}
    for logical, physical in enumerate(resolved):
        roles = [role for role,(prefix, _) in CARDS.items() if physical.startswith(prefix)]
        if len(roles) != 1: raise ValueError("Only the frozen3060/SUPER partition is supported")
        role = roles[0]; mib = allocation[physical]
        if type(mib) is not int or mib != CARDS[role][1]: raise ValueError("Frozen per-card reservation budget required")
        total = int(cuda.get_device_properties(logical).total_memory)
        allowed = (mib-256)*1024**2
        if not 0 < allowed <= total: raise ValueError("Allocator cap exceeds physical device capacity")
        if cuda.memory.get_allocator_backend() != "native": raise ValueError("Native CUDA allocator required")
        cuda.set_per_process_memory_fraction(allowed/total, device=logical)
        devices[role] = f"cuda:{logical}"
        settings.append(dict(logical_device=logical, physical_uuid=physical, role=role, lease_bytes=mib*1024**2,
            allocator_limit_bytes=allowed, device_total_memory=total, memory_fraction=allowed/total))
    return dict(devices=devices, physical_by_logical=resolved, settings=settings,
        lease_id=lease["id"], allocation=dict(allocation))


def device_peaks(settings, *, cuda=None):
    cuda = torch.cuda if cuda is None else cuda
    result = {}
    for row in settings["settings"]:
        reserved = cuda.max_memory_reserved(row["logical_device"])
        if reserved > row["allocator_limit_bytes"]: raise MemoryError("Reserved peak exceeded that GPU's allocator cap")
        result[row["physical_uuid"]] = dict(peak_reserved_bytes=reserved,
            peak_allocated_bytes=cuda.max_memory_allocated(row["logical_device"]),
            allocator_limit_bytes=row["allocator_limit_bytes"], role=row["role"])
    return result


def load_initial_block(native, selection, *, layer):
    if recipe.sha(selection["checkpoint"]) != selection["checkpoint_sha256"]:
        raise ValueError("Recovered initial checkpoint changed")
    block = semantic.build_enriched_block(native, recipe.load(selection["fit_path"]), parameter_dtype=torch.float32)
    state = recipe.load(selection["checkpoint"])
    if state["epoch"] != selection["selected_epoch"] or (layer and state.get("layer") != layer):
        raise ValueError("Recovered initial checkpoint epoch/layer changed")
    block.load_state_dict(state["block"], strict=True)
    if any(p.dtype != torch.float32 for p in block.parameters()): raise ValueError("No recovered-weight rounding permitted")
    return block.train().requires_grad_(True)


def build_model(assembly_plan, devices, *, guard=lambda:None):
    from .jspace_qwen_parallel_model import load_native_block, load_readout_norm
    first, second = devices["3060"], devices["2060_SUPER"]
    guard(); embedding, norm, rotary = load_readout_norm(assembly_plan["model"]["path"], device=first, dtype=torch.bfloat16)
    embedding, norm, rotary = embedding.float(), norm.float(), rotary.float()
    blocks = []
    for layer in range(36):
        guard(); device = first if layer < 24 else second
        native = load_native_block(assembly_plan["model"]["path"], layer, device=device, dtype=torch.bfloat16)
        block = load_initial_block(native, assembly_plan["blocks"][str(layer)], layer=layer)
        del native
        if (block.width, block.rank, block.head_dim, block.intermediate_size) != (2048, 1024, 128, 11008):
            raise ValueError("Recovered half-rank/full-head architecture changed")
        blocks.append(block)
    return PartitionedCarryModel(embedding, blocks, norm.weight, norm.variance_epsilon, rotary)


def fixed_fingerprint(model, chunk_bytes=32*1024**2):
    digest = hashlib.sha256(); count = size = 0
    for name, value in model.named_buffers():
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)], sort_keys=True).encode())
        flat = value.detach().contiguous().view(-1)
        step = max(1, chunk_bytes//value.element_size())
        for start in range(0, flat.numel(), step):
            digest.update(flat[start:start+step].cpu().numpy().tobytes())
        count += 1; size += value.numel()*value.element_size()
    return dict(sha256=digest.hexdigest(), tensor_count=count, bytes=size)


def changed_parameter_blocks(model, initial_states):
    changed, unchanged, first = [], [], {}
    count = 0
    for index, (block, original) in enumerate(zip(model.blocks, initial_states)):
        count += 1
        for name, parameter in block.named_parameters():
            if not torch.equal(parameter.detach().cpu(), original[name]):
                changed.append(index); first[str(index)] = name; break
        else: unchanged.append(index)
    if count != len(model.blocks): raise ValueError("Initial parameter references must cover every block")
    return dict(changed_blocks=changed, unchanged_blocks=unchanged, first_changed_parameter=first)


def selection_key(metrics): return metrics["correct"], -metrics["choice_ce"]


def baseline_control(records, reference, cases):
    return assembly.compare_control(records, reference, cases, 27)


def replace_selected(model, path, metadata):
    path = Path(path); pending, retired = path.with_name(path.name+".pending"), path.with_name(path.name+".retired")
    if pending.exists() or retired.exists(): raise FileExistsError("Incomplete best-checkpoint replacement preserved")
    joint.save_selected(model, pending, metadata=metadata)
    if path.exists(): os.replace(path, retired)
    os.replace(pending, path)
    if retired.exists(): shutil.rmtree(retired)  # Only this run's superseded mutable best state.
    return recipe.sha(path/"manifest.json")


class Trainer(semantic.SemanticTrainer):
    def __init__(self, model, dataset, *, config=Config(), guard=None):
        self.model, self.dataset, self.config, self.guard = model, dataset, config, guard
        self.device = model.embedding.device
        if any(p.device.type == "cuda" for p in model.parameters()) and guard is None:
            raise ValueError("Two-GPU training requires a reservation guard")
        if model.embedding.requires_grad or model.embedding.shape[0] != dataset.vocabulary:
            raise ValueError("Frozen original full-vocabulary tiedE required")
        self.named_parameters = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not self.named_parameters or any(p.dtype != torch.float32 for _,p in self.named_parameters):
            raise ValueError("Only FP32 block parameters and final norm may train")
        self._guard()
        self.optimizer = StreamingCPUAdamW([p for _,p in self.named_parameters], lr=config.learning_rate,
            weight_decay=0., chunk_bytes=config.optimizer_chunk_bytes, max_cpu_state_bytes=config.max_cpu_state_bytes)
        self.updates = 0

    def development(self):
        metrics, records = super().development()
        metrics["selection"] = "highest_development_accuracy_then_lowest_conditional_choice_CE"
        return metrics, records

    def fit(self, directory, *, reference=None, event=lambda *_a, **_k:None):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
        history, best, best_epoch, best_records = [], None, None, None
        selected = directory/"selected"; effective = self.config.microbatch*self.config.accumulation
        for epoch in range(self.config.epochs+1):
            if epoch:
                order = list(self.dataset.training); random.Random(self.config.seed+epoch).shuffle(order)
                for start in range(0, len(order), effective):
                    event("update", epoch=epoch, **self.train_window(order[start:start+effective]))
            metrics, records = self.development()
            if epoch == 0 and reference is not None:
                control = baseline_control(records, reference, [self.dataset.items[i] for i in self.dataset.development])
                recipe.write(directory/"initial_control.json", control)
                if not control["passed"]: raise RuntimeError("Fresh scientific initialization did not replay27/64 assembly")
            report_path = directory/f"epoch_{epoch:02d}_development.json"
            recipe.write(report_path, dict(metrics=metrics, records=records))
            improved = best is None or selection_key(metrics) > selection_key(best)
            row = dict(epoch=epoch, metrics=metrics, development_sha256=recipe.sha(report_path), checkpoint_updated=improved)
            if improved:
                row["checkpoint_manifest_sha256"] = replace_selected(self.model, selected, dict(epoch=epoch, development=metrics))
                best, best_epoch, best_records = metrics, epoch, records
            history.append(row); event("development", epoch=epoch, **metrics)
        joint.load_selected(self.model, selected)
        metrics, records = self.development()
        if metrics != best or records != best_records: raise RuntimeError("Strict joint reload changed selected development results")
        result = dict(status="completed", systems_only=False, selected_epoch=best_epoch, selected_directory=str(selected),
            selected_manifest_sha256=recipe.sha(selected/"manifest.json"), selected=best, initial=history[0]["metrics"],
            history=history, updates=self.updates, strict_reload_exact=True, initializer_selected=best_epoch == 0,
            quality_pass=best["correct"] >= 47, minimum_correct=47, native_correct=50,
            intermediate_improvement=best["correct"] > history[0]["metrics"]["correct"],
            optimizer=self.optimizer.memory_stats(), optimizer_resume="New experiment requires explicit optimizer reset",
            confirmed=False, scope=SCOPE)
        recipe.write(directory/"result.json", result)
        return result


def prepare(root, assembly_root, teacher_root):
    from . import jspace_recovered_teacher as teacher
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("Hide CUDA for CPU preparation")
    root, assembly_root, teacher_root = map(lambda p:Path(p).resolve(), (root, assembly_root, teacher_root))
    if root.exists(): raise FileExistsError(root)
    parent = assembly.verify(assembly_root)
    baseline = assembly.completed_arm(assembly_root, parent, "all_compact")
    if baseline["accuracy"]["correct"] != 27: raise ValueError("Observed all36 compact27/64 initializer required")
    teacher_plan = teacher.verify(teacher_root, completed=True)
    if teacher_plan["source_fingerprint"] != parent["source_fingerprint"]:
        raise ValueError("Matched original teacher source required")
    original = recipe.read(Path(parent["source"])/"plan.json")
    items = recipe.read(Path(parent["source"])/"items.json")
    if (teacher_plan["items_digest"] != recipe.digest(items)
            or sum(c["split"] == "train" for c in items["items"]) != 512
            or sum(c["split"] == "development" for c in items["items"]) != 64):
        raise ValueError("Exactly original512train/64development identities required")
    inputs = dict(parent["input_sha256"])
    inputs.update({str(assembly_root/name):recipe.sha(assembly_root/name) for name in
        ("plan.json", "all_compact/result.json", "all_compact/manifest.json")})
    teacher_manifest = recipe.read(teacher_root/"cache/manifest.json")
    for relative, expected in teacher_manifest["artifact_sha256"].items():
        inputs[str(teacher_root/relative)] = expected
    for name in ("plan.json", "cache/manifest.json"):
        inputs[str(teacher_root/name)] = recipe.sha(teacher_root/name)
    sources = dict(parent["source_sha256"]); assembly._merge_hashes(sources, teacher_plan["source_sha256"])
    for path in (Path(__file__).resolve(), Path(semantic.__file__).resolve(), Path(joint.__file__).resolve(),
                 Path(__file__).with_name("jspace_streaming_adamw.py"), Path(__file__).with_name("jspace_semantic_loss.py"),
                 recipe.ROOT/"tests/structure_net/test_jspace_recovered_joint.py", PROTOCOL):
        assembly._merge_hashes(sources, {str(path):recipe.sha(path)})
    plan = dict(schema=SCHEMA, assembly_root=str(assembly_root), assembly_fingerprint=parent["fingerprint"],
        source=parent["source"], source_fingerprint=parent["source_fingerprint"], model=parent["model"],
        teacher_root=str(teacher_root), teacher_fingerprint=teacher_plan["fingerprint"],
        teacher_precision="Original native BF16-rounded parameters promoted toFP32, full forward and logitsFP32",
        blocks=parent["blocks"], config=asdict(Config()), partition={"3060":list(range(24)), "2060_SUPER":list(range(24,36))},
        cards=CARDS, input_sha256=inputs, input_identity={p:recipe.file_identity(p) for p in inputs}, source_sha256=sources,
        items_digest=recipe.digest(items), label_ids=original["label_ids"], protocol=str(PROTOCOL),
        initial_correct=27, native_correct=50, minimum_correct=47, confirmed=False, scope=SCOPE,
        precision="All selected weights, fixed geometry, residual activations and tied readout remainFP32",
        freezing="All bases/anchors/projection buffers and tiedE fixed; all36 block parameter tensors and final RMS gain train",
        optimizer_resume="Fresh CPU AdamW state; no pilot weights or previous optimizer state reused")
    plan["fingerprint"] = recipe.digest(plan); root.mkdir(parents=True)
    for path, expected in sources.items():
        if recipe.sha(path) != expected: raise ValueError("Changed continuation producer")
        destination = root/"producer"/Path(path).relative_to(recipe.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
    recipe.write(root/"plan.json", plan); recipe.write(root/"items.json", items)
    return dict(status="prepared", fingerprint=plan["fingerprint"])


def verify(root):
    root = Path(root); plan = recipe.read(root/"plan.json")
    if plan.get("schema") != SCHEMA or plan["fingerprint"] != recipe.digest({k:v for k,v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed recovered joint plan")
    if assembly.verify(plan["assembly_root"])["fingerprint"] != plan["assembly_fingerprint"]:
        raise ValueError("Changed selected assembly")
    if recipe.digest(recipe.read(root/"items.json")) != plan["items_digest"]: raise ValueError("Changed training/development rows")
    for path, expected in plan["source_sha256"].items():
        if recipe.sha(path) != expected or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT)) != expected:
            raise ValueError("Changed recovered joint producer")
    for path, value in plan["input_identity"].items():
        if recipe.file_identity(path) != value: raise ValueError("Changed pinned teacher/selected input")
    return plan


def verify_pilot(root, plan):
    root = Path(root); manifest = recipe.read(root/"pilot/manifest.json"); result = recipe.read(root/"pilot/result.json")
    if (manifest.get("status") != "completed" or manifest.get("fingerprint") != plan["fingerprint"]
            or manifest.get("result_sha256") != recipe.sha(root/"pilot/result.json")
            or result.get("systems_only") is not True or result.get("updates") != 1
            or result["initial_control"].get("passed") is not True or result["initial_control"].get("correct") != 27
            or result.get("fixed_state_unchanged") is not True
            or result["parameter_changes"]["changed_blocks"] != list(range(36))
            or result["parameter_changes"]["unchanged_blocks"]):
        raise ValueError("Completed matching27/64 replay and all-block gradient/unchanged-geometry pilot required")


def completed_training(root, plan):
    root = Path(root); manifest = recipe.read(root/"train/manifest.json"); result = recipe.read(root/"train/result.json")
    if (manifest.get("status") != "completed" or manifest.get("fingerprint") != plan["fingerprint"]
            or manifest.get("result_sha256") != recipe.sha(root/"train/result.json")
            or result.get("status") != "completed" or result.get("systems_only") is not False
            or result.get("strict_reload_exact") is not True or result.get("updates") != 96
            or [r["epoch"] for r in result["history"]] != list(range(4))):
        raise ValueError("Completed prospective three-epoch continuation required")
    winner = max(result["history"], key=lambda r:selection_key(r["metrics"]))
    path = root/"train/selected"
    if (result["selected_epoch"] != winner["epoch"] or result["selected"] != winner["metrics"]
            or Path(result["selected_directory"]).resolve() != path
            or result["selected_manifest_sha256"] != winner["checkpoint_manifest_sha256"]
            or recipe.sha(path/"manifest.json") != result["selected_manifest_sha256"]):
        raise ValueError("Selected state differs from frozen accuracy/choiceCE selection")
    for row in result["history"]:
        if recipe.sha(root/"train"/f"epoch_{row['epoch']:02d}_development.json") != row["development_sha256"]:
            raise ValueError("Changed development record")
    return result


def run(root, stage):
    from .jspace_full_lease import wait_ready
    from .jspace_recovered_teacher import FP32JointDataset
    from .jspace_single_block_verify import Tolerances, validate_records
    if stage not in ("pilot", "train", "verify"): raise ValueError("Explicit pilot/train/verify stage required")
    root = Path(root).resolve(); plan = verify(root); directory = root/("verification" if stage == "verify" else stage)
    if directory.exists(): raise FileExistsError(directory)
    if stage == "train": verify_pilot(root, plan)
    selected = completed_training(root, plan) if stage == "verify" else None
    lease = settings = dataset = None; started = time.monotonic(); telemetry = dict(calls=0, wall_seconds=0.)
    try:
        lease = wait_ready([0, 1]); settings = allocator_settings(lease)
        for row in settings["settings"]: torch.cuda.reset_peak_memory_stats(row["logical_device"])
        torch.set_num_threads(4); torch.manual_seed(plan["config"]["seed"]); torch.backends.cuda.matmul.allow_tf32 = False
        def guard():
            began = time.monotonic()
            try:
                current = wait_ready([0, 1])
                if current["id"] != lease["id"] or current["allocation"] != lease["allocation"]:
                    raise RuntimeError("Joint continuation reservation changed")
                device_peaks(settings)
            finally:
                telemetry["calls"] += 1; telemetry["wall_seconds"] += time.monotonic()-began
        parent = recipe.read(Path(plan["assembly_root"])/"plan.json")
        model = build_model(parent, settings["devices"], guard=guard)
        dataset = FP32JointDataset(plan["teacher_root"])
        trainer = Trainer(model, dataset, config=config_from_dict(plan["config"]), guard=guard)
        reference = recipe.read(Path(plan["assembly_root"])/"all_compact/result.json")
        cases = [dataset.items[i] for i in dataset.development]
        before = fixed_fingerprint(model); guard()
        if stage == "pilot":
            directory.mkdir(exist_ok=False)
            initial, records = trainer.development(); control = baseline_control(records, reference, cases)
            recipe.write(directory/"initial_control.json", control)
            if not control["passed"]: raise RuntimeError("FP32 warm start did not reproduce27/64 assembly")
            order = sorted(dataset.training, key=lambda i:len(dataset.items[i]["input_ids"]), reverse=True)
            update = trainer.train_window(order[:trainer.config.microbatch*trainer.config.accumulation])
            changes = changed_parameter_blocks(model, (recipe.load(parent["blocks"][str(i)]["checkpoint"])["block"] for i in range(36)))
            after = fixed_fingerprint(model)
            if before != after or changes["unchanged_blocks"]: raise RuntimeError("Pilot changed fixed geometry or failed to update every block")
            development, _ = trainer.development()
            result = dict(status="completed", systems_only=True, updates=1, initial=initial, initial_control=control,
                update=update, development=development, fixed_state_unchanged=True, fixed_state=before,
                parameter_changes=changes, optimizer=trainer.optimizer.memory_stats(), confirmed=False)
            recipe.write(directory/"result.json", result)
        elif stage == "train":
            result = trainer.fit(directory, reference=reference, event=lambda kind, **kw:recipe.event(root, kind, stage=stage, **kw))
            if fixed_fingerprint(model) != before: raise RuntimeError("Scientific training changed fixed geometry or tiedE")
        else:
            directory.mkdir(exist_ok=False)
            checkpoint_manifest = joint.load_selected(model, root/"train/selected")
            metrics, records = trainer.development()
            ref = recipe.read(root/"train"/f"epoch_{selected['selected_epoch']:02d}_development.json")
            validate_records(records, cases); validate_records(ref["records"], cases)
            tolerance = Tolerances()
            logits_pass = all(abs(a-b) <= tolerance.logit_atol+tolerance.logit_rtol*abs(a)
                for x,y in zip(ref["records"], records) for a,b in zip(x["logits"], y["logits"]))
            predictions_exact = all(a["prediction"] == b["prediction"] for a,b in zip(ref["records"], records))
            metric_keys = [*semantic.METRICS, "kl", "ce"]
            metrics_pass = all(math.isclose(metrics[k], ref["metrics"][k], abs_tol=tolerance.feature_atol,
                rel_tol=tolerance.feature_rtol) for k in metric_keys)
            unchanged = fixed_fingerprint(model) == before
            result = dict(status="completed", passed=logits_pass and predictions_exact and metrics_pass and unchanged,
                fingerprint=plan["fingerprint"], selected_epoch=selected["selected_epoch"],
                selected_manifest_sha256=recipe.sha(root/"train/selected/manifest.json"), strict_state_load=True,
                predictions_exact=predictions_exact, logits_pass=logits_pass, metrics_pass=metrics_pass,
                fixed_state_unchanged=unchanged, training_quality_pass=selected["quality_pass"], metrics=metrics, records=records,
                tolerances=asdict(tolerance), checkpoint_metadata=checkpoint_manifest["metadata"], confirmed=False, scope=SCOPE)
            recipe.write(directory/"result.json", result)
            if not result["passed"]: raise RuntimeError("Fresh joint checkpoint verification failed")
        guard(); peaks = device_peaks(settings)
        recipe.write(directory/"manifest.json", dict(status="completed", systems_only=stage == "pilot", fingerprint=plan["fingerprint"],
            result_sha256=recipe.sha(directory/"result.json"), lease_id=lease["id"], allocation=lease["allocation"],
            allocator=settings, device_peaks=peaks, reservation_guard=telemetry, elapsed_seconds=time.monotonic()-started,
            passed=result.get("passed"), quality_pass=result.get("quality_pass", result.get("training_quality_pass"))))
        return dict(status="completed", stage=stage, fingerprint=plan["fingerprint"],
            quality_pass=result.get("quality_pass", result.get("training_quality_pass")))
    except BaseException as error:
        directory.mkdir(parents=True, exist_ok=True)
        recipe.write(directory/"failure.json", dict(status="failed", fingerprint=plan["fingerprint"], error=repr(error),
            lease_id=None if lease is None else lease["id"], allocation=None if lease is None else lease["allocation"],
            allocator=settings, reservation_guard=telemetry, elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        if dataset is not None: dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "pilot", "train", "verify")); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assembly-root", type=Path); parser.add_argument("--teacher-root", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.assembly_root is None or args.teacher_root is None: parser.error("prepare requires --assembly-root and --teacher-root")
        value = prepare(args.root, args.assembly_root, args.teacher_root)
    else: value = run(args.root, args.stage)
    print(json.dumps(value, allow_nan=False), flush=True)
