"""One explicitly requested next compressed layer, after its predecessor passes.

The accepted prefix supplies actual FP32 native-width inputs. Targets and
normalization always come from two immutable ORIGINAL teacher boundaries.
There is deliberately no automatic multi-layer coordinator. Prefix banks are
regenerable artifacts; this module does not silently delete prior evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from . import jspace_single_block_distill as single

recipe, joint = single.recipe, single.joint
SCHEMA = "jspace.next-layer-distillation.v1"


def cap_allocator(lease, *, cuda=None, visible=None):
    """Bound CUDA caching before model allocation, with room outside PyTorch."""
    from .jspace_single_block_reserved import allocator_settings
    return allocator_settings(lease, dict(headroom_mib=256, allocator_backend="native"),
        cuda=cuda, visible=visible)


class PrefixPairs(single.NativePairs):
    """Student inputs and original-teacher inputs are distinct bank roles."""
    def __init__(self, items, prefix_path, native_input_path, target_path):
        self.prefix = None
        super().__init__(items, native_input_path, target_path)
        try:
            self.prefix = np.load(prefix_path, mmap_mode="r", allow_pickle=False)
            if self.prefix.dtype != np.float32 or self.prefix.shape != self.inputs.shape:
                raise ValueError("Aligned native-width FP32 student-prefix bank required")
        except Exception:
            self.close()
            raise

    def close(self):
        super().close()
        if self.prefix is not None and not self.prefix._mmap.closed:
            self.prefix._mmap.close()

    def batch(self, indices, device):
        value = super().batch(indices, device)
        value["native_hidden"] = value["hidden"]
        hidden = np.zeros(tuple(value["hidden"].shape), dtype=np.float32)
        for row, i in enumerate(indices):
            case = self.items[i]
            hidden[row, :len(case["input_ids"])] = self.prefix[case["offset"]:case["stop"]]
        value["hidden"] = torch.from_numpy(hidden).to(device)
        if not torch.isfinite(value["hidden"]).all():
            raise FloatingPointError("Nonfinite accepted-prefix states")
        return value

    @torch.no_grad()
    def fit_floor(self, device, config, guard):
        energies = []
        for start in range(0, len(self.training), config.microbatch):
            guard()
            # Bypass student-prefix substitution: neither its error nor its
            # development rows may change the original update normalizer.
            batch = single.NativePairs.batch(self, self.training[start:start+config.microbatch], device)
            energy = (batch["target"]-batch["hidden"]).square().mean(-1)
            energies.append(energy[batch["valid"]].cpu())
        values = torch.cat(energies)
        median = float(values.median())
        if median <= 0 or not torch.isfinite(values).all():
            raise ValueError("Degenerate original-teacher update energy")
        self.normalization = dict(train_token_count=len(values), train_case_count=len(self.training),
            median_teacher_update_energy=median, floor=config.floor_fraction*median,
            floor_fraction=config.floor_fraction,
            denominator="detached mean((ORIGINAL native_h[l+1]-ORIGINAL native_h[l])^2); never student input")
        return self.normalization


def feature_losses(student, batch, *, floor, basis=None):
    values = single.feature_losses(student, batch["native_hidden"], batch["target"],
        batch["valid"], floor=floor)
    if basis is not None:
        with torch.no_grad():
            correction = batch["target"]-batch["hidden"]
            outside = correction-(correction@basis)@basis.T
            energy = (batch["target"]-batch["native_hidden"]).square().mean(-1).clamp_min(floor)
            relative = outside.square().mean(-1)/energy
            valid = batch["valid"]
            values["irreducible_outside_relative_mse"] = (
                (relative*valid).sum(-1)/valid.sum(-1).clamp_min(1)).mean()
            values["retained_relative_mse"] = values["loss"].detach()-values["irreducible_outside_relative_mse"]
    return values


def _replace_checkpoint(path, value):
    """Only this new stage's mutable best file is replaced, atomically."""
    path = Path(path)
    temporary = path.with_suffix(".pending")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
    os.replace(temporary, path)


class Trainer(single.Trainer):
    def losses(self, batch, *, diagnostics=False):
        return feature_losses(self.forward(batch), batch, floor=self.data.normalization["floor"],
            basis=self.block.basis if diagnostics else None)

    def fit(self, directory, *, event=lambda *_a, **_k: None, native_correct=50,
            minimum_correct=47, layer=1):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        effective = self.config.microbatch*self.config.accumulation
        history, best, best_epoch, best_records = [], None, None, None
        selected = directory/"selected.pt"
        for epoch in range(self.config.epochs+1):
            if epoch:
                order = list(self.data.training)
                random.Random(self.config.seed+epoch).shuffle(order)
                for start in range(0, len(order), effective):
                    event("update", epoch=epoch, **self.train_window(order[start:start+effective]))
            metrics, records = self.development()
            path = directory/f"epoch_{epoch:02d}_development.json"
            recipe.write(path, dict(metrics=metrics, records=records))
            improved = best is None or (metrics["correct"], -metrics["relative_mse"]) > (
                best["correct"], -best["relative_mse"])
            row = dict(epoch=epoch, metrics=metrics, checkpoint_updated=improved,
                development_sha256=recipe.sha(path))
            if improved:
                _replace_checkpoint(selected, dict(block=joint._cpu(self.block.state_dict()),
                    epoch=epoch, layer=layer, normalization=self.data.normalization))
                best, best_epoch, best_records = metrics, epoch, records
                row["checkpoint_sha256"] = recipe.sha(selected)
            history.append(row)
            event("development", epoch=epoch, **metrics)
        self.block.load_state_dict(recipe.load(selected)["block"], strict=True)
        metrics, records = self.development()
        if metrics != best or records != best_records:
            raise RuntimeError("Selected next-layer reload changed feature/task results")
        result = dict(status="completed", systems_only=False, selected_epoch=best_epoch,
            selected_checkpoint=str(selected), selected_checkpoint_sha256=recipe.sha(selected),
            selected=best, history=history, updates=self.updates, strict_reload_exact=True,
            native_correct=native_correct, minimum_correct=minimum_correct,
            quality_pass=best["correct"] >= minimum_correct, normalization=self.data.normalization,
            trained_layer=layer, checkpoint_retention="one atomically replaced best checkpoint; all epoch metrics retained",
            scope="Development-only cumulative accepted prefix + current layer + original suffix; no confirmation or speed claim")
        recipe.write(directory/"result.json", result)
        return result


def accepted_predecessor(root, verification_root):
    """Validate a completed task-qualified predecessor before any GPU work."""
    root = Path(root).resolve()
    plan = recipe.read(root/"plan.json")
    if plan.get("schema") == single.SCHEMA:
        single.verify(root)
        layer = 0
        input_path = Path(plan["input_path"])
    elif plan.get("schema") == SCHEMA:
        verify(root, prefix=True)
        layer = plan["layer"]
        input_path = root/"prefix/input.npy"
    else:
        raise ValueError("Expected a single-layer distillation predecessor")
    runtime_policy = None
    if (root/"runtime_policy.json").exists():
        from .jspace_single_block_reserved import verify as verify_runtime, verify_runtime_receipt
        _, runtime_policy = verify_runtime(root)
        verify_runtime_receipt(root, "train", plan, runtime_policy)
    result = recipe.read(root/"train/result.json")
    manifest = recipe.read(root/"train/manifest.json")
    if (manifest.get("status") != "completed" or manifest.get("systems_only") is not False
            or manifest.get("fingerprint") != plan["fingerprint"]
            or manifest.get("result_sha256") != recipe.sha(root/"train/result.json")
            or result.get("status") != "completed" or result.get("systems_only") is not False
            or result.get("strict_reload_exact") is not True or result.get("quality_pass") is not True
            or result.get("native_correct") != 50 or result.get("minimum_correct") != 47
            or result["selected"]["count"] != 64 or result["selected"]["correct"] < 47):
        raise ValueError("A completed, reloaded predecessor with cumulative >=47/64 is required")
    epoch = result["selected_epoch"]
    selected = Path(result["selected_checkpoint"]).resolve()
    if selected.parent != root/"train":
        raise ValueError("Selected checkpoint must belong to predecessor training")
    rows = recipe.read(root/"train"/f"epoch_{epoch:02d}_development.json")
    history = next(r for r in result["history"] if r["epoch"] == epoch)
    expected = result.get("selected_checkpoint_sha256", history.get("checkpoint_sha256"))
    if (recipe.sha(selected) != expected or recipe.sha(root/"train"/f"epoch_{epoch:02d}_development.json") != history["development_sha256"]
            or rows["metrics"] != result["selected"] or single.ranking_summary(rows["records"])["correct"] != result["selected"]["correct"]):
        raise ValueError("Predecessor selected checkpoint/qualification changed")
    if [row["epoch"] for row in result["history"]] != list(range(plan["config"]["epochs"]+1)):
        raise ValueError("Complete predecessor epoch history is required")
    winner = max(result["history"], key=lambda r:(r["metrics"]["correct"], -r["metrics"]["relative_mse"]))
    if winner["epoch"] != epoch or winner["metrics"] != result["selected"]:
        raise ValueError("Predecessor violates the fixed checkpoint selection rule")
    cases = [c for c in recipe.read(root/"items.json")["items"] if c["split"] == "development"]
    from .jspace_single_block_verify import validate_records
    validate_records(rows["records"], cases)
    verification_root = Path(verification_root).resolve()
    vplan = recipe.read(verification_root/"plan.json")
    vm = recipe.read(verification_root/"verification/manifest.json")
    vr = recipe.read(verification_root/"verification/result.json")
    if (vplan.get("fingerprint") != recipe.digest({k:v for k,v in vplan.items() if k != "fingerprint"})
            or vplan.get("schema") not in ("jspace.single-block-independent-verification.v1", SCHEMA)
            or (vplan.get("schema") == SCHEMA and verification_root != root)
            or (vplan.get("schema") != SCHEMA and (vplan.get("campaign_fingerprint") != plan["fingerprint"]
                or Path(vplan["campaign"]).resolve() != root))
            or vm.get("status") != "completed" or vm.get("passed") is not True
            or vm.get("fingerprint") != vplan["fingerprint"]
            or vm.get("result_sha256") != recipe.sha(verification_root/"verification/result.json")
            or vr.get("status") != "completed" or vr.get("passed") is not True
            or vr.get("fingerprint") != vplan["fingerprint"]
            or vr.get("campaign_fingerprint") != plan["fingerprint"]
            or vr.get("checkpoint_sha256") != expected or vr.get("selected_epoch") != epoch
            or vr.get("strict_state_load") is not True or vr.get("training_quality_pass") is not True):
        raise ValueError("Successful independent verification of this exact selected checkpoint is required")
    for path, expected_source in vplan["source_sha256"].items():
        if recipe.sha(path) != expected_source or recipe.sha(verification_root/"producer"/Path(path).relative_to(recipe.ROOT)) != expected_source:
            raise ValueError("Changed independent verification producer")
    validate_records(vr["records"], cases)
    if any(a["prediction"] != b["prediction"] for a,b in zip(rows["records"], vr["records"])):
        raise ValueError("Independent verification predictions differ from selected checkpoint")
    return dict(root=str(root), layer=layer, fingerprint=plan["fingerprint"], plan=plan,
        checkpoint=str(selected), checkpoint_sha256=expected, input_path=str(input_path),
        verification_root=str(verification_root), runtime_policy=runtime_policy)


def prepare(root, predecessor, basis_root, verification_root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA for CPU preparation")
    root, basis_root = Path(root).resolve(), Path(basis_root).resolve()
    if root.exists():
        raise FileExistsError(root)
    previous = accepted_predecessor(predecessor, verification_root)
    layer = previous["layer"]+1
    if not 1 <= layer < 36:
        raise ValueError("Only the next single layer1..35 may be prepared")
    prior = previous["plan"]
    source = Path(prior["source"])
    original = recipe.verify(source)
    items = recipe.read(source/"items.json")
    if recipe.digest(items) != prior["items_digest"]:
        raise ValueError("Predecessor task population differs from original teacher")
    fitplan = recipe.read(basis_root/"plan.json")
    fits = recipe.read(basis_root/"basis/manifest.json")
    if (fitplan.get("schema") != "jspace.all-layer-enriched-basis.v1"
            or fitplan["fingerprint"] != recipe.digest({k:v for k,v in fitplan.items() if k != "fingerprint"})
            or fitplan["source_fingerprint"] != original["fingerprint"]
            or fits.get("status") != "completed" or fits["fingerprint"] != fitplan["fingerprint"]):
        raise ValueError("Completed matching all-layer enriched geometry required")
    train_ids = {c["identity"] for c in items["items"] if c["split"] == "train"}
    fit_ids, check_ids = set(fitplan["fit_training_identities"]), set(fitplan["internal_training_check_identities"])
    if not fit_ids or not check_ids or fit_ids & check_ids or not (fit_ids | check_ids) <= train_ids:
        raise ValueError("Geometry must be fitted and checked only on disjoint training cases")
    for path, expected in fitplan["source_sha256"].items():
        if recipe.sha(path) != expected or recipe.sha(basis_root/"producer"/Path(path).relative_to(recipe.ROOT)) != expected:
            raise ValueError("Changed enrichment producer")
    fit_path = basis_root/"basis"/f"block_{layer:02d}/fit.pt"
    cache = Path(original["teacher_root"])/"capture"
    native_input, target = cache/f"boundary_{layer:02d}.npy", cache/f"boundary_{layer+1:02d}.npy"
    inputs = {str(fit_path):fits["artifacts"][str(fit_path.relative_to(basis_root))],
        str(native_input):original["input_sha256"][str(native_input)],
        str(target):original["input_sha256"][str(target)],
        previous["checkpoint"]:previous["checkpoint_sha256"]}
    for name in ("plan.json", "train/result.json", "train/manifest.json",
            f"train/epoch_{recipe.read(Path(previous['root'])/'train/result.json')['selected_epoch']:02d}_development.json"):
        path = Path(previous["root"])/name
        inputs[str(path)] = recipe.sha(path)
    for path in (source/"plan.json", source/"items.json", basis_root/"plan.json", basis_root/"basis/manifest.json",
            Path(previous["input_path"]), Path(prior["fit_path"])):
        inputs[str(path)] = recipe.sha(path)
    for name in ("plan.json", "verification/manifest.json", "verification/result.json"):
        path = Path(previous["verification_root"])/name
        inputs[str(path)] = recipe.sha(path)
    if previous["runtime_policy"] is not None:
        for name in ("runtime_policy.json", "runtime_train.json"):
            path = Path(previous["root"])/name
            inputs[str(path)] = recipe.sha(path)
    for path, expected in inputs.items():
        if recipe.sha(path) != expected:
            raise ValueError("Changed predecessor, teacher, or current geometry")
    sources = {str(Path(__file__)):recipe.sha(__file__),
        str(recipe.ROOT/"tests/structure_net/test_jspace_next_layer_distill.py"):
            recipe.sha(recipe.ROOT/"tests/structure_net/test_jspace_next_layer_distill.py")}
    sources.update(prior["source_sha256"])
    sources.update(original["source_sha256"])
    verifier = Path(__file__).with_name("jspace_single_block_verify.py")
    sources[str(verifier)] = recipe.sha(verifier)
    allocator_source = Path(__file__).with_name("jspace_single_block_reserved.py")
    allocator_test = recipe.ROOT/"tests/structure_net/test_jspace_single_block_reserved.py"
    sources.update({str(p):recipe.sha(p) for p in (allocator_source, allocator_test)})
    config = single.Config(**prior["config"])
    if config != single.Config():
        raise ValueError("Next stage preserves the original12epoch single-block training configuration")
    plan = dict(schema=SCHEMA, source=str(source), source_fingerprint=original["fingerprint"],
        predecessor_root=previous["root"], predecessor_fingerprint=previous["fingerprint"],
        predecessor_verification_root=previous["verification_root"],
        predecessor_checkpoint=previous["checkpoint"], predecessor_input_path=previous["input_path"],
        predecessor_fit_path=prior["fit_path"], layer=layer, trained_layers=[layer],
        frozen_prefix_layers=list(range(layer)), native_suffix=list(range(layer+1, 36)),
        basis_root=str(basis_root), basis_fingerprint=fitplan["fingerprint"], fit_path=str(fit_path),
        native_input_path=str(native_input), target_path=str(target), config=prior["config"],
        items_digest=recipe.digest(items), native_baseline=prior["native_baseline"], native_correct=50, minimum_correct=47,
        rank=1024, head_dim=128, intermediate_size=11008, confirmation_used=False,
        objective="equal-case/token output MSE divided by ORIGINAL teacher update energy; student input is never normalization target",
        prefix="actual frozen accepted prefix, full native width FP32; no coordinate reconstruction or compressed teacher targets",
        checkpoint_retention="one current best per stage plus all epoch metrics",
        prefix_retention="one new regenerable bank for this explicit stage; no automatic progression or deletion",
        input_sha256=inputs, input_identity={p:recipe.file_identity(p) for p in inputs}, source_sha256=sources)
    plan["fingerprint"] = recipe.digest(plan)
    root.mkdir(parents=True)
    for path, expected in sources.items():
        if recipe.sha(path) != expected:
            raise ValueError("Changed inherited producer")
        destination = root/"producer"/Path(path).relative_to(recipe.ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    recipe.write(root/"plan.json", plan)
    recipe.write(root/"items.json", items)
    return dict(status="prepared", layer=layer, fingerprint=plan["fingerprint"])


def verify(root, *, prefix=False):
    root = Path(root)
    plan = recipe.read(root/"plan.json")
    if plan.get("schema") != SCHEMA or plan["fingerprint"] != recipe.digest({k:v for k,v in plan.items() if k != "fingerprint"}):
        raise ValueError("Changed next-layer plan")
    original = recipe.verify(plan["source"])
    if original["fingerprint"] != plan["source_fingerprint"] or recipe.digest(recipe.read(root/"items.json")) != plan["items_digest"]:
        raise ValueError("Changed original teacher or case alignment")
    for path, expected in plan["source_sha256"].items():
        if recipe.sha(path) != expected or recipe.sha(root/"producer"/Path(path).relative_to(recipe.ROOT)) != expected:
            raise ValueError("Changed next-layer producer")
    for path, identity in plan["input_identity"].items():
        if recipe.file_identity(path) != identity:
            raise ValueError("Changed frozen predecessor or teacher input")
    if prefix:
        manifest = recipe.read(root/"prefix/manifest.json")
        if (manifest.get("status") != "completed" or manifest.get("fingerprint") != plan["fingerprint"]
                or manifest.get("artifact_sha256") != recipe.sha(root/"prefix/input.npy")):
            raise ValueError("A completed matching accepted-prefix cache is required")
    return plan


def hidden_batch(items, bank, indices, device):
    lengths = [len(items[i]["input_ids"]) for i in indices]
    values = np.zeros((len(indices), max(lengths), bank.shape[1]), dtype=bank.dtype)
    for row, i in enumerate(indices):
        c = items[i]
        values[row, :lengths[row]] = bank[c["offset"]:c["stop"]]
    hidden = torch.from_numpy(values)
    if bank.dtype == np.int16:
        hidden = hidden.view(torch.bfloat16)
    elif bank.dtype != np.float32:
        raise ValueError("Only native BF16 bits or accepted FP32 prefix inputs are supported")
    hidden = hidden.to(device=device, dtype=torch.float32)
    positions = torch.arange(max(lengths), device=device).expand(len(indices), -1)
    valid = positions < torch.tensor(lengths, device=device)[:, None]
    mask = (positions[:, :, None] >= positions[:, None, :]) & valid[:, :, None] & valid[:, None, :]
    if not torch.isfinite(hidden).all():
        raise FloatingPointError("Nonfinite predecessor inputs")
    return hidden, positions, mask[:, None]


@torch.inference_mode()
def write_prefix(block, rotary, items, input_path, output_path, *, device, guard=lambda:None, microbatch=4):
    source = np.load(input_path, mmap_mode="r", allow_pickle=False)
    destination = None
    try:
        if source.shape[0] != items[-1]["stop"] or source.ndim != 2:
            raise ValueError("Predecessor input-bank alignment changed")
        if Path(output_path).exists():
            raise FileExistsError(output_path)
        destination = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=source.shape)
        block.eval().requires_grad_(False)
        for start in range(0, len(items), microbatch):
            guard()
            indices = list(range(start, min(start+microbatch, len(items))))
            hidden, positions, mask = hidden_batch(items, source, indices, device)
            value = block(hidden, attention_mask=mask, position_ids=positions,
                position_embeddings=rotary(hidden, positions))
            if value.shape != hidden.shape or not torch.isfinite(value).all():
                raise FloatingPointError("Invalid selected-prefix output")
            for row, i in enumerate(indices):
                c = items[i]
                destination[c["offset"]:c["stop"]] = value[row, :len(c["input_ids"])].cpu().numpy()
        destination.flush()
    finally:
        source._mmap.close()
        if destination is not None:
            destination._mmap.close()


def make_quality(plan, items, directory, guard, *, device="cuda:0"):
    from .jspace_qwen_parallel_model import load_native_block, load_readout_norm
    original = recipe.read(Path(plan["source"])/"plan.json")
    prefix = np.load(Path(directory).parent/"prefix/input.npy", mmap_mode="r", allow_pickle=False)
    cases, tables, offset = [], [], 0
    try:
        for c in items:
            if c["split"] != "development":
                continue
            table = torch.from_numpy(np.array(prefix[c["offset"]:c["stop"]], copy=True))
            copied = dict(c, input_ids=list(range(offset, offset+len(c["input_ids"]))))
            cases.append(copied); tables.append(table); offset += len(table)
    finally:
        prefix._mmap.close()
    hidden_table = torch.cat(tables)
    embedding, norm, _ = load_readout_norm(original["model"]["path"], device="cpu", dtype=torch.bfloat16)
    embedding, norm = embedding.float(), norm.float()
    rotary = recipe.rotary_module(original, device)
    calls = 0
    def evaluate(block):
        nonlocal calls
        snapshots = [deepcopy(block).pack(dtype=torch.float32, buffer_dtype=torch.float32)]
        def loader(local):
            guard()
            return snapshots.pop() if local == 0 else load_native_block(original["model"]["path"],
                plan["layer"]+local, device=device, dtype=torch.bfloat16).float()
        calls += 1
        value = single.stream_score(cases, input_embedding=hidden_table, block_loader=loader,
            final_readout=(norm, embedding), label_ids=original["label_ids"],
            scratch_dir=Path(directory)/f"scratch_{calls:02d}", rotary=rotary,
            layer_count=36-plan["layer"], device=device, guard=guard, microbatch=4)
        value["metrics"].update(max_live_decoder_blocks=2, resident_training_optimizer_included_in_peak=True,
            input_scope="virtual row addresses into actual accepted-prefix states; not vocabulary tokens",
            original_prompt_lengths=[len(c["input_ids"]) for c in items if c["split"] == "development"])
        return value
    return evaluate


def completed_training(root, plan):
    """Authenticate selection and its development reference before reservation."""
    from .jspace_single_block_verify import validate_records
    root = Path(root).resolve()
    result = recipe.read(root/"train/result.json")
    manifest = recipe.read(root/"train/manifest.json")
    checkpoint = root/"train/selected.pt"
    if (manifest.get("status") != "completed" or manifest.get("systems_only") is not False
            or manifest.get("fingerprint") != plan["fingerprint"]
            or manifest.get("result_sha256") != recipe.sha(root/"train/result.json")
            or result.get("status") != "completed" or result.get("systems_only") is not False
            or result.get("strict_reload_exact") is not True
            or Path(result["selected_checkpoint"]).resolve() != checkpoint
            or result.get("selected_checkpoint_sha256") != recipe.sha(checkpoint)):
        raise ValueError("Completed authenticated selected checkpoint required before verification")
    history = result["history"]
    if [row["epoch"] for row in history] != list(range(plan["config"]["epochs"]+1)):
        raise ValueError("Complete ordered epoch history required before verification")
    for row in history:
        metrics = row["metrics"]
        if (metrics.get("count") != 64 or type(metrics.get("correct")) is not int
                or not 0 <= metrics["correct"] <= 64
                or not math.isfinite(metrics["relative_mse"]) or metrics["relative_mse"] < 0):
            raise ValueError("Valid64-case development metrics required for every epoch")
    winner = max(history, key=lambda r:(r["metrics"]["correct"], -r["metrics"]["relative_mse"]))
    if winner["epoch"] != result["selected_epoch"] or winner["metrics"] != result["selected"]:
        raise ValueError("Selected checkpoint violates the fixed development selection rule")
    if winner.get("checkpoint_sha256") != result["selected_checkpoint_sha256"]:
        raise ValueError("Winning epoch checkpoint hash differs from selected bytes")
    reference_path = root/"train"/f"epoch_{winner['epoch']:02d}_development.json"
    if recipe.sha(reference_path) != winner.get("development_sha256"):
        raise ValueError("Selected development reference hash changed before verification")
    reference = recipe.read(reference_path)
    if reference["metrics"] != result["selected"]:
        raise ValueError("Selected development reference metrics differ from result")
    cases = [c for c in recipe.read(root/"items.json")["items"] if c["split"] == "development"]
    if len(cases) != 64:
        raise ValueError("Original64-case development population required")
    validate_records(reference["records"], cases)
    correct = sum(r["prediction"] == r["answer"] for r in reference["records"])
    if correct != result["selected"]["correct"] or result["quality_pass"] != (correct >= plan["minimum_correct"]):
        raise ValueError("Selected development predictions or qualification disagree with result")
    return result


def run(root, stage):
    if stage not in ("prefix", "pilot", "train", "verify"):
        raise ValueError("Only explicit prefix/pilot/train/verify stages are supported")
    root = Path(root).resolve()
    plan = verify(root, prefix=stage != "prefix")
    directory = root/("verification" if stage == "verify" else stage)
    if directory.exists():
        raise FileExistsError(directory)
    if stage == "train":
        joint.verify_pilot(root, plan["fingerprint"])
    selected_result = None
    if stage == "verify":
        selected_result = completed_training(root, plan)
    from .jspace_full_lease import wait_ready
    from .jspace_qwen_parallel_model import load_native_block
    lease = data = None
    cuda_started = False
    started = time.monotonic()
    try:
        lease = wait_ready([0]); allocation = dict(lease["allocation"])
        def guard():
            current = wait_ready([0])
            if current["id"] != lease["id"] or current["allocation"] != allocation:
                raise RuntimeError("Next-layer reservation changed")
            joint.check_reserved_peak(allocation, torch.cuda.max_memory_reserved())
        allocator = cap_allocator(lease)
        torch.set_num_threads(4); torch.manual_seed(plan["config"]["seed"])
        torch.cuda.init(); cuda_started = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
        original = recipe.read(Path(plan["source"])/"plan.json")
        items = recipe.read(root/"items.json")["items"]
        layer = plan["layer"]-1 if stage == "prefix" else plan["layer"]
        fit_path = plan["predecessor_fit_path"] if stage == "prefix" else plan["fit_path"]
        native = load_native_block(original["model"]["path"], layer, device="cuda:0", dtype=torch.bfloat16)
        block = single.build_enriched_block(native, recipe.load(fit_path), parameter_dtype=torch.float32)
        del native
        if (block.width, block.rank, block.head_dim, block.intermediate_size) != (2048, 1024, 128, 11008):
            raise ValueError("Runtime geometry differs from frozen half-rank/full-head plan")
        rotary = recipe.rotary_module(original, "cuda:0")
        if stage == "prefix":
            directory.mkdir(exist_ok=False)
            block.load_state_dict(recipe.load(plan["predecessor_checkpoint"])["block"], strict=True)
            block.pack(dtype=torch.float32, buffer_dtype=torch.float32)
            write_prefix(block, rotary, items, plan["predecessor_input_path"], directory/"input.npy",
                device="cuda:0", guard=guard)
            result = dict(status="completed", artifact_sha256=recipe.sha(directory/"input.npy"),
                predecessor_checkpoint_sha256=recipe.sha(plan["predecessor_checkpoint"]),
                input_bank_sha256=recipe.sha(plan["predecessor_input_path"]),
                items_digest=plan["items_digest"], shape=[items[-1]["stop"], 2048], dtype="float32")
        else:
            data = PrefixPairs(items, root/"prefix/input.npy", plan["native_input_path"], plan["target_path"])
            config = single.Config(**plan["config"])
            quality = make_quality(plan, items, directory, guard)
            trainer = Trainer(block, rotary, data, quality, config=config, guard=guard)
            if stage == "verify":
                from .jspace_single_block_verify import compare
                directory.mkdir(exist_ok=False)
                checkpoint = recipe.load(root/"train/selected.pt")
                if (checkpoint["epoch"] != selected_result["selected_epoch"]
                        or checkpoint["normalization"] != data.normalization
                        or checkpoint["layer"] != plan["layer"]):
                    raise ValueError("Selected epoch, layer, or original-teacher normalization changed")
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
                    training_quality_pass=selected_result["quality_pass"], confirmed=False, process_id=os.getpid(),
                    scope="Fresh-process development-only reload and packed native-suffix scoring; no confirmation claim")
                recipe.write(directory/"result.json", result)
                if not result["passed"]:
                    raise RuntimeError("Independent next-layer selected checkpoint parity failed")
            elif stage == "pilot":
                directory.mkdir(exist_ok=False)
                initial, _ = trainer.development()
                order = sorted(data.training, key=lambda i:len(items[i]["input_ids"]), reverse=True)
                update = trainer.train_window(order[:config.microbatch*config.accumulation])
                metrics, _ = trainer.development()
                result = dict(status="completed", systems_only=True, updates=1, update=update,
                    development=metrics, initial=initial, normalization=data.normalization)
                recipe.write(directory/"result.json", result)
            else:
                result = trainer.fit(directory, layer=plan["layer"],
                    event=lambda kind, **v:recipe.event(root, kind, stage=stage, **v))
        guard()
        manifest = dict(status="completed", systems_only=stage == "pilot", fingerprint=plan["fingerprint"],
            lease_id=lease["id"], allocation=allocation, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            allocator=allocator, elapsed_seconds=time.monotonic()-started)
        if stage == "prefix":
            manifest.update(result)
        else:
            manifest["result_sha256"] = recipe.sha(directory/"result.json")
            if stage == "verify":
                manifest.update(passed=result["passed"], checkpoint_sha256=result["checkpoint_sha256"])
        recipe.write(directory/"manifest.json", manifest)
        return dict(status="completed", stage=stage, layer=plan["layer"],
            quality_pass=result.get("quality_pass"), fingerprint=plan["fingerprint"])
    except Exception as error:
        directory.mkdir(parents=True, exist_ok=True)
        recipe.write(directory/"failure.json", dict(status="failed", fingerprint=plan["fingerprint"],
            error_type=type(error).__name__, error=str(error), lease_id=None if lease is None else lease["id"],
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_started else None))
        raise
    finally:
        if data is not None:
            data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "prefix", "pilot", "train", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--basis-root", type=Path)
    parser.add_argument("--verification-root", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.predecessor is None or args.basis_root is None or args.verification_root is None:
            parser.error("prepare requires --predecessor, --basis-root and --verification-root")
        value = prepare(args.root, args.predecessor, args.basis_root, args.verification_root)
    else:
        value = run(args.root, args.stage)
    print(json.dumps(value, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
