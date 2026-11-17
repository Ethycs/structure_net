"""TRAIN-only fixed-basis context extension; qualification is a separate real pilot.

The learned function is unchanged on existing rows. New affine complements use
native incoming maps, while recovered coordinate weights/offsets remain frozen.
Polynomial activation normalizers are channel tables, not position tables: they
must NOT be recalibrated during this intervention. This is not a basis refit or
a claim that the original approximation generalizes to long contexts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from . import jspace_qwen3b_progressive_model as builder
from . import jspace_3b_partial_generation as partial
from . import jspace_qwen3_capture as capture
from experiments.neural_architecture_lab.distillation_recovery import module

SCHEMA = "jspace.recovery-reset.geometry.v1"
TABLES = ("anchor", "anchor_energy", "qkv_shift", "gate_up_shift")
data = module("distillation_data")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def tensor_hash(value):
    t = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((tuple(t.shape), t.dtype)).encode())
    h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _guard(device, guard):
    device = torch.device(device)
    if device.type == "cuda" and guard is None:
        raise ValueError("CUDA geometry requires caller-owned reservation guard")
    if guard is not None:
        guard()
    return device


def _source(spec):
    if spec.get("schema") != "jspace.recovery-reset.spec.v1":
        raise ValueError("Unknown study specification")
    if sha(spec["initial_checkpoint"]) != spec["initial_checkpoint_sha256"]:
        raise ValueError("Initializer hash mismatch")
    plan = read(spec["teacher_plan"])
    if plan["model"]["path"] != spec["native_model_path"] or plan["model"]["revision"] != spec["native_model_revision"]:
        raise ValueError("Native teacher identity mismatch")
    if sorted(map(int, plan["fits"])) != list(range(plan["cut"], 36)):
        raise ValueError("Incomplete contiguous fitted suffix")
    return plan


def load_initial(spec, *, device="cpu", guard=None):
    """Reconstruct the actual unpacked model-only initializer, strictly."""
    spec = read(spec) if isinstance(spec, (str, Path)) else spec
    device = _guard(device, guard)
    plan = _source(spec)
    stack, norm, embedding, old_rotary, proof = builder.build_stack(
        spec["native_model_path"], {int(k): v for k, v in plan["fits"].items()},
        plan["cut"], previous=spec["initial_checkpoint"], device="cpu", trainable=True)
    del old_rotary
    if proof["new_layer"] is not None or not proof["norm_exact"]:
        raise ValueError("Initializer was not strictly reconstructed")
    rotary = partial.training_rotary(builder.native._config(spec["native_model_path"]), device, guard=guard)
    return stack.to(device), norm.to(device), embedding.to(device), rotary


def _dependencies(spec, plan):
    paths = [spec["initial_checkpoint"], spec["teacher_plan"], str(Path(__file__).resolve())]
    for directory in plan["fits"].values():
        paths.extend(str(Path(directory) / name) for name in ("fit.pt", "relation.pt"))
    paths.extend(str(Path(spec["native_model_path"]) / name) for name in plan["model"]["files"])
    # Freeze the actual constructors and execution code, not only fit data.
    paths.extend(str(Path(m.__file__).resolve()) for m in (builder, builder.native, partial, capture))
    from . import jspace_qwen_affine_block, jspace_qwen_compact_stack, jspace_qwen_affine_fast
    paths.extend(str(Path(m.__file__).resolve()) for m in
                 (jspace_qwen_affine_block, jspace_qwen_compact_stack, jspace_qwen_affine_fast))
    hashes = {p: sha(p) for p in sorted(set(paths))}
    for name, expected in plan["model"]["files"].items():
        if hashes[str(Path(spec["native_model_path"]) / name)] != expected:
            raise ValueError(f"Pinned native teacher file changed: {name}")
    return hashes


def _verify_hashes(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError(f"Dependency hash mismatch: {path}")


def prepare(root, spec_path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU prepare requires CUDA_VISIBLE_DEVICES=''")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = read(spec_path)
    source = _source(spec)
    state = torch.load(spec["initial_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    extra = state["stack"]["_extra_state"]
    if extra["packed"] or extra["stripped"]:
        raise ValueError("Only preserved unpacked initializer is extensible")
    inventory = {}
    counts = []
    for i, layer in enumerate(extra["layer_indices"]):
        prefix = f"blocks.{i}."
        rows = state["stack"][prefix + "anchor"].shape[0]
        counts.append(rows)
        inventory[str(layer)] = {name: dict(shape=list(t.shape), dtype=str(t.dtype), sha256=tensor_hash(t))
                                for key, t in state["stack"].items() if key.startswith(prefix)
                                for name in [key[len(prefix):]] if isinstance(t, torch.Tensor)}
        if any(state["stack"][prefix + name].shape[0] != rows for name in TABLES):
            raise ValueError("Position tables disagree")
    if len(set(counts)) != 1 or spec["target_context"] <= counts[0]:
        raise ValueError("Expected a common original range and a larger target")
    p = dict(schema=SCHEMA, spec=spec, spec_path=str(Path(spec_path).resolve()),
             spec_sha256=sha(spec_path), dependencies=_dependencies(spec, source),
             original_length=counts[0], max_length=spec["target_context"], inventory=inventory,
             bridge_inventory={name: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=tensor_hash(value))
                               for name, value in state["stack"].items() if name.startswith("bridges.")},
             layers=list(extra["layer_indices"]), geometry_mode=extra["geometry_mode"],
             calibration_examples=int(spec.get("context_calibration_examples", 16)),
             minimum_position_samples=int(spec.get("context_minimum_position_samples", 2)),
             parity_atol=0.0, parity_rtol=0.0,
             support_proof=dict(position_tables=list(TABLES), bridge_tables="bridges.*.shift",
                 fixed_channel_tables=["polynomial_anchor_mean", "polynomial_anchor_scale", "basis", "value_basis"],
                 learned_row_semantics="Recovered incoming weights act on coordinates z; new orthogonal-anchor terms use frozen native RMS-gained maps, projected heads and polynomial-kept channels. Shared learned offsets, polynomial coefficients and activation normalizers are unchanged; long-context approximation quality is unmeasured.",
                 normalizer="RMS energy=(sum(z²)+anchor_energy)/native_width; native eps unchanged",
                 rotary="Original FP32 training rotary; no frequency scaling",
                 extension="Orthogonal TRAIN means with original FP32 fitted bases; native affine folds and preserved polynomial retained-channel ordering",
                 basis_refit_required=False, unsupported="Lengths above max_length; quality/speed and cached decoding not certified"))
    if not 1 <= p["minimum_position_samples"] <= p["calibration_examples"]:
        raise ValueError("Invalid calibration coverage policy")
    p["dependencies"][p["spec_path"]] = p["spec_sha256"]
    write_json(root / "plan.json", p)
    return dict(status="prepared", qualified=False, original_length=counts[0], max_length=p["max_length"],
                corpus_available=Path(spec["corpus_manifest"]).exists(), plan=str(root / "plan.json"))


def calibration_records(manifest_path, *, target, maximum, minimum):
    manifest = read(manifest_path)
    if manifest.get("schema") != "jspace.recovery-reset.corpus.v1":
        raise ValueError("Unadmitted corpus schema")
    path = manifest.get("calibration_records", manifest["train_records"])
    if manifest["sha256"].get(path) != sha(path):
        raise ValueError("TRAIN shard hash mismatch")
    chosen = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("Geometry calibration is TRAIN only")
            if "calibration_records" in manifest:
                ids, provenance = row["input_ids"], row["provenance"]
                required = ("source_uri", "source_revision", "tokenizer_sha256", "chat_template_sha256",
                            "source_complete_tokens_sha256", "source_complete_token_count")
                if (any(not provenance.get(key) for key in required)
                        or "source_row" not in provenance or "historical_lineage" not in provenance
                        or provenance.get("derivation") != "exact_prefix_crop_for_geometry_only"):
                    raise ValueError("Calibration prefix lacks complete TRAIN source lineage")
                if provenance["source_complete_token_count"] < len(ids):
                    raise ValueError("Calibration prefix exceeds its actual source")
            else:
                record = data.ResponseRecord(**row)
                ids, provenance = list(record.input_ids), record.provenance
            if not ids or any(type(t) is not int or t < 0 for t in ids):
                raise ValueError("Calibration requires actual nonnegative token IDs")
            if len(ids) >= target:
                chosen.append((row, ids, provenance))
                if len(chosen) == maximum:
                    break
    if len(chosen) < minimum:
        raise ValueError(f"Need at least {minimum} admitted TRAIN examples covering all {target} positions; found {len(chosen)}")
    if len({r["group_id"] for r, _, _ in chosen}) < minimum:
        raise ValueError("Calibration coverage requires distinct TRAIN source groups")
    if len({r["example_id"] for r, _, _ in chosen}) != len(chosen):
        raise ValueError("Repeated calibration example")
    rows = []
    for i, (record, tokens, provenance) in enumerate(chosen):
        ids = list(tokens[:target])
        rows.append(dict(identity=record["example_id"], group_id=record["group_id"], split="train", input_ids=ids,
                         offset=i * target, stop=(i + 1) * target, loss_positions=[target - 1], targets=[ids[-1]],
                         record_sha256=data.digest(record), prefix_sha256=data.digest(ids), provenance=provenance,
                         source_tokens=len(tokens), purpose="Geometry calibration prefix, never a complete response target"))
    return rows, dict(manifest_path=str(Path(manifest_path).resolve()), manifest_sha256=sha(manifest_path),
                      calibration_records=path, calibration_sha256=sha(path),
                      example_ids=[r["example_id"] for r, _, _ in chosen],
                      group_ids=[r["group_id"] for r, _, _ in chosen], samples_per_position=len(rows))


@torch.no_grad()
def extend_block(block, original, basis, means):
    """Append affine rows; all learned parameters and polynomial transforms survive.

    Incoming recovered weights operate only on z. The orthogonal anchor term is
    still the original native map, exactly as at the original compilation. Head
    projection and polynomial channel removal must be applied to these new rows.
    """
    if block.packed or block.position_policy != "error" or block.anchor is None:
        raise ValueError("Extension requires unpacked, unclamped native-anchor geometry")
    if means.ndim != 2 or means.shape[1] != block.width or not len(means) or not torch.isfinite(means).all():
        raise ValueError("Finite observed position means required")
    dtype = torch.float64 if basis.dtype == torch.float64 else torch.float32
    basis, means = basis.to(dtype), means.to(device=basis.device, dtype=dtype)
    if not torch.equal(block.basis.cpu(), basis.to(block.basis.dtype).cpu()):
        raise ValueError("Original fitted basis differs from saved basis")
    if block.rank == block.width:
        raise ValueError("Global full-rank anchors need no position extension")
    anchor = means - (means @ basis) @ basis.T
    def incoming(weight, gain):
        return anchor @ (weight.detach().to(dtype) * gain.detach().to(dtype)[None]).T
    attn = original.self_attn
    shift = incoming(torch.cat([p.weight for p in (attn.q_proj, attn.k_proj, attn.v_proj)]),
                     original.input_layernorm.weight)
    if block.has_head_projection:
        d = block.native_head_dim
        qr = torch.cat([block.query_rope_columns[h] + h*d for h in range(block.nq)])
        kr = torch.cat([block.key_rope_columns[g] + (block.nq+g)*d for g in range(block.nkv)])
        values = [shift[:, (block.nq+block.nkv+g)*d:(block.nq+block.nkv+g+1)*d] @ block.value_basis[g].to(dtype)
                  for g in range(block.nkv)]
        shift = torch.cat((shift[:, torch.cat((qr, kr))], *values), -1)
    gate = incoming(torch.cat((original.mlp.gate_proj.weight, original.mlp.up_proj.weight)),
                    original.post_attention_layernorm.weight)
    if block.has_polynomial:
        rows = torch.cat((block.polynomial_kept, block.polynomial_kept + block.native_intermediate_size))
        gate = gate[:, rows]
    new = dict(anchor=anchor, anchor_energy=anchor.square().sum(-1, keepdim=True), qkv_shift=shift, gate_up_shift=gate)
    for name, value in new.items():
        old = getattr(block, name)
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite new {name}")
        setattr(block, name, torch.cat((old, value.to(device=old.device, dtype=old.dtype)), 0))
    block.position_count = len(block.anchor)


@torch.no_grad()
def extend_bridges(stack, bases, old_length):
    for i, bridge in enumerate(stack.bridges):
        first, second = stack.blocks[i:i+2]
        delta = first.anchor[old_length:] - second.anchor[old_length:]
        tail = delta.to(bases[i+1].dtype) @ bases[i+1]
        bridge.shift = torch.cat((bridge.shift, tail.to(bridge.shift)), 0)


def _tensor_tables(stack):
    return {f"blocks.{i}.{name}" for i in range(len(stack.blocks)) for name in TABLES} | {
        f"bridges.{i}.shift" for i in range(len(stack.bridges))}


def assert_preserved(initial, extended, old_length, table_names):
    if initial.keys() != extended.keys():
        raise ValueError("Extension changed structure keys")
    for name, old in initial.items():
        actual = extended[name]
        if name in table_names:
            actual = actual[:old_length]
        equal = torch.equal(old.cpu(), actual.cpu()) if isinstance(old, torch.Tensor) else old == actual
        if not equal:
            raise ValueError(f"Extension changed existing state: {name}")


def _load_state(stack, state, target):
    for name in _tensor_tables(stack):
        module_name, attr = name.rsplit(".", 1)
        owner = stack.get_submodule(module_name)
        tensor = state[name]
        if tensor.shape[0] != target or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid extended position table: {name}")
        setattr(owner, attr, tensor.to(device=getattr(owner, attr).device))
    for block in stack.blocks:
        block.position_count = target
        block.position_policy = "error"
    stack.load_state_dict(state, strict=True)


def _manifest(path, *, qualified=True):
    manifest = read(path)
    if manifest.get("schema") != SCHEMA or qualified and manifest.get("qualified") is not True:
        raise ValueError("Context extension is not qualified")
    required = {manifest["checkpoint"], manifest["structure"]}
    if qualified:
        required.add(manifest["pilot_receipt"])
    if not required <= manifest["artifacts"].keys():
        raise ValueError("Missing authenticated geometry artifact")
    _verify_hashes(manifest["artifacts"])
    structure = read(manifest["structure"])
    _verify_hashes(structure["dependencies"])
    lineage = structure["lineage"]
    _verify_hashes({lineage["manifest_path"]: lineage["manifest_sha256"],
                    lineage["calibration_records"]: lineage["calibration_sha256"]})
    if manifest["original_checkpoint_sha256"] != structure["spec"]["initial_checkpoint_sha256"]:
        raise ValueError("Geometry initializer identity mismatch")
    if qualified:
        receipt = read(manifest["pilot_receipt"])
        if not receipt.get("passed") or receipt["checkpoint_sha256"] != manifest["artifacts"][manifest["checkpoint"]]:
            raise ValueError("Pilot does not authenticate this geometry")
    return manifest, structure


def load_extended(manifest_path, *, device="cpu", guard=None):
    """Load only actually pilot-qualified geometry; reject stale dependencies."""
    device = _guard(device, guard)
    manifest, structure = _manifest(manifest_path)
    stack, norm, embedding, rotary = load_initial(structure["spec"], device="cpu")
    state = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    assert_preserved(stack.state_dict(), state["stack"], structure["original_length"], _tensor_tables(stack))
    assert_preserved(norm.state_dict(), state["norm"], structure["original_length"], set())
    _load_state(stack, state["stack"], manifest["max_length"])
    rotary = partial.training_rotary(builder.native._config(structure["spec"]["native_model_path"]),
                                    device, guard=guard)
    return stack.to(device), norm.to(device), embedding.to(device), rotary


def run(root, *, device="cuda:0", guard=None):
    """Stream one native layer at a time; keep only two host memmap banks."""
    device = _guard(device, guard)
    guard = guard or (lambda: None)
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    _verify_hashes(plan["dependencies"])
    spec, target = plan["spec"], plan["max_length"]
    items, lineage = calibration_records(spec["corpus_manifest"], target=target,
        maximum=plan["calibration_examples"], minimum=plan["minimum_position_samples"])
    out = root / "calibration"
    out.mkdir(exist_ok=False)
    write_json(out / "items.json", items)
    started = time.perf_counter()
    stack, norm, embedding, rotary = load_initial(spec)
    config = builder.native._config(spec["native_model_path"])
    rotary = rotary.to(device)
    paths = _source(spec)["fits"]
    bases = []
    shape = (len(items) * target, stack.width)
    current_path = out / "boundary_00.npy"
    source = np.lib.format.open_memmap(current_path, mode="w+", dtype=np.float32, shape=shape)
    destination = None
    try:
        for item in items:
            source[item["offset"]:item["stop"]] = embedding[item["input_ids"]].float().numpy()
        source.flush()
        del embedding
        with capture._execution(device):
            for layer in range(config.num_hidden_layers):
                guard()
                native = builder.native.load_native_block(spec["native_model_path"], layer, device=device,
                                                          dtype=torch.bfloat16, config=config)
                destination_path = out / f"boundary_{layer+1:02d}.npy"
                destination = np.lib.format.open_memmap(destination_path, mode="w+", dtype=np.float32, shape=shape)
                for index in range(len(items)):
                    guard()
                    batch = capture._batch(items, source, [index], device=device, dtype=torch.bfloat16)
                    hidden = batch["hidden"]
                    result = native(hidden, attention_mask=batch["attention_mask"],
                        position_ids=batch["position_ids"], cache_position=batch["position_ids"][0],
                        position_embeddings=rotary(hidden, batch["position_ids"]), use_cache=False)
                    value = result[0] if isinstance(result, tuple) else result
                    if not torch.isfinite(value).all():
                        raise FloatingPointError("Nonfinite native calibration boundary")
                    item = items[index]
                    destination[item["offset"]:item["stop"]] = value.float().cpu().numpy()[0]
                    del batch, hidden, result, value
                destination.flush()
                if layer in stack.layer_indices:
                    block = stack.blocks[layer-stack.layer_idx]
                    fit = builder.native._load(Path(paths[str(layer)]) / "fit.pt")
                    basis = builder.native._fitted_basis(fit, block.width, block.rank, "cpu")
                    bases.append(basis)
                    # Accumulate observed means in FP64; no tail padding/extrapolation.
                    means = np.zeros((target, block.width), dtype=np.float64)
                    for item in items:
                        means += source[item["offset"]:item["stop"]]
                    means /= len(items)
                    native.cpu()
                    extend_block(block, native, basis, torch.from_numpy(means[plan["original_length"]:]).float())
                    del fit, means
                source._mmap.close()
                if layer != stack.layer_idx:
                    current_path.unlink()
                source, destination = destination, None
                current_path = destination_path
                del native
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        source._mmap.close()
        source = None
        extend_bridges(stack, bases, plan["original_length"])
        initial = torch.load(spec["initial_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
        assert_preserved(initial["stack"], stack.state_dict(), plan["original_length"], _tensor_tables(stack))
        assert_preserved(initial["norm"], norm.state_dict(), plan["original_length"], set())
        checkpoint = root / "extended.pt"
        with checkpoint.open("xb") as stream:
            torch.save(dict(stack=stack.state_dict(), norm=norm.state_dict()), stream)
        boundary = out / f"boundary_{stack.layer_idx:02d}.npy"
        structure = root / "structure.json"
        write_json(structure, dict(plan, lineage=lineage, boundary=str(boundary), items=str(out / "items.json")))
        artifacts = {str(p): sha(p) for p in (checkpoint, structure, boundary, out / "items.json")}
        manifest = dict(schema=SCHEMA, qualified=False, original_checkpoint_sha256=spec["initial_checkpoint_sha256"],
            spec_sha256=plan["spec_sha256"],
            checkpoint=str(checkpoint), structure=str(structure), artifacts=artifacts,
            max_length=target, original_length=plan["original_length"], frozen_basis=True,
            calibration=lineage, short_range_parity=None, largest_output_allowance="max_length minus exact templated prompt token count",
            intervention="TRAIN-only fixed-basis affine-row extension",
            elapsed_seconds=time.perf_counter()-started, owned_native_blocks_max=1)
        write_json(root / "candidate.json", manifest)
        return dict(status="exported_unqualified", candidate=str(root / "candidate.json"), max_length=target)
    finally:
        for bank in (source, destination):
            if bank is not None and not bank._mmap.closed:
                bank._mmap.close()


def _forward(stack, norm, rotary, x, *, masked_key=None):
    length = x.shape[1]
    positions = torch.arange(length, device=x.device)
    allowed = (positions[None, :] <= positions[:, None])[None, None]
    if masked_key is not None:
        allowed[..., masked_key] = False
    mask = torch.zeros(allowed.shape, device=x.device, dtype=x.dtype).masked_fill_(~allowed, torch.finfo(x.dtype).min)
    return norm(stack(x, attention_mask=mask, position_ids=positions[None], cache_position=positions,
                      position_embeddings=rotary(x, positions[None]), use_cache=False)[0])


def _gradients(stack, norm, rotary, boundary):
    stack.zero_grad(set_to_none=True)
    norm.zero_grad(set_to_none=True)
    x = boundary.detach().clone().requires_grad_(True)
    output = _forward(stack, norm, rotary, x)
    # A nonconstant readout direction avoids RMS-normalized squared-norm degeneracy.
    direction = torch.linspace(-1, 1, output.shape[-1], device=x.device)
    (output.float() * direction).mean().backward()
    gradients = {f"{prefix}.{name}": p.grad for prefix, module in (("stack", stack), ("norm", norm))
                 for name, p in module.named_parameters() if p.requires_grad}
    if x.grad is None or any(g is None or not torch.isfinite(g).all() for g in gradients.values()) or not torch.isfinite(x.grad).all():
        raise FloatingPointError("Missing or nonfinite gradient")
    if not torch.isfinite(output).all() or not x.grad.count_nonzero():
        raise FloatingPointError("Nonfinite output or zero input gradient")
    proof = dict(output_sha256=tensor_hash(output), input_gradient_sha256=tensor_hash(x.grad),
                 parameter_gradient_sha256={name: tensor_hash(g) for name, g in gradients.items()},
                 parameters=len(gradients), finite=True)
    stack.zero_grad(set_to_none=True)
    norm.zero_grad(set_to_none=True)
    return proof


def pilot(root, *, device="cuda:0", guard=None):
    device = _guard(device, guard)
    guard = guard or (lambda: None)
    root = Path(root).resolve()
    manifest, plan = _manifest(root / "candidate.json", qualified=False)
    started = time.perf_counter()
    stack, norm, embedding, rotary = load_initial(plan["spec"], device=device, guard=guard)
    del embedding
    bank = np.load(plan["boundary"], mmap_mode="r", allow_pickle=False)
    boundary = torch.from_numpy(np.array(bank[:plan["max_length"]], copy=True))[None].to(device=device, dtype=torch.bfloat16)
    bank._mmap.close()
    stack.train(); norm.train()
    stack.checkpoint_blocks = True
    short = boundary[:, :plan["original_length"]]
    before = _gradients(stack, norm, rotary, short)
    state = torch.load(manifest["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    _load_state(stack, state["stack"], plan["max_length"])
    guard()
    after = _gradients(stack, norm, rotary, short)
    if before != after:
        raise ValueError("Short-range output/input/parameter-gradient parity failed")
    long_started = time.perf_counter()
    long_proof = _gradients(stack, norm, rotary, boundary)
    long_seconds = time.perf_counter()-long_started
    stack.eval(); norm.eval()
    with torch.no_grad():
        baseline = _forward(stack, norm, rotary, boundary)
        changed = boundary.clone()
        changed[:, -1] = -changed[:, -1] + 1
        causal = _forward(stack, norm, rotary, changed)
        if not torch.equal(baseline[:, :-1], causal[:, :-1]):
            raise ValueError("Full-range causal future-token isolation failed")
        # Mask an interior key; compare later queries, not its own changed query.
        key = plan["max_length"] // 2
        changed.copy_(boundary); changed[:, key] = -changed[:, key] + 1
        masked = _forward(stack, norm, rotary, boundary, masked_key=key)
        perturbed = _forward(stack, norm, rotary, changed, masked_key=key)
        keep = torch.arange(plan["max_length"], device=device) != key
        if not torch.equal(masked[:, keep], perturbed[:, keep]):
            raise ValueError("Masked key leaked into other query positions")
        if any(not torch.isfinite(v).all() for v in (baseline, causal, masked, perturbed)):
            raise FloatingPointError("Nonfinite full-target masked execution")
        # Exercise the same table indexing on CPU: deliberately triggering a
        # CUDA out-of-bounds kernel would poison the context and its reservation.
        try:
            stack.blocks[0]._rows(stack.blocks[0].anchor.cpu(),
                torch.tensor([[plan["max_length"]]]), 1, 1)
        except IndexError as error:
            unsupported = "CPU table index: " + type(error).__name__
        else:
            raise ValueError("Unsupported target+1 position did not fail")
    guard()
    receipt = dict(passed=True, checkpoint_sha256=sha(manifest["checkpoint"]), short_range_parity=after,
        spec_sha256=plan["spec_sha256"],
        full_target_range=dict(length=plan["max_length"], gradients=long_proof,
            causal_future_isolation=True, masked_key_isolation=True, unsupported_length_rejected=unsupported),
        device=str(device), precision="BF16 activation/parameters; original FP32 anchors and BF16 bases",
        checkpoint_blocks=True,
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        full_range_forward_backward_seconds=long_seconds, elapsed_seconds=time.perf_counter()-started,
        scope="Suffix forward/backward geometry qualification. Native-prefix rollout, optimizer updates and end-to-end timings belong to runtime pilot; no task-quality claim.")
    pilot_directory = root / "pilot"
    pilot_directory.mkdir(exist_ok=False)
    receipt_path = pilot_directory / "result.json"
    write_json(receipt_path, receipt)
    manifest.update(qualified=True, pilot_receipt=str(receipt_path), short_range_parity=after)
    manifest["artifacts"][str(receipt_path)] = sha(receipt_path)
    write_json(root / "manifest.json", manifest)
    return dict(status="qualified", manifest=str(root / "manifest.json"), max_length=manifest["max_length"])


def check(root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU check requires CUDA_VISIBLE_DEVICES=''")
    root = Path(root)
    plan = read(root / "plan.json")
    if plan.get("schema") != SCHEMA:
        raise ValueError("Unknown geometry plan")
    _verify_hashes({**plan["dependencies"], plan["spec_path"]: plan["spec_sha256"]})
    if read(plan["spec_path"]) != plan["spec"]:
        raise ValueError("Saved study specification differs from authenticated source")
    phase, qualified = "prepared", False
    for filename, require_qualified in (("candidate.json", False), ("manifest.json", True)):
        if (root / filename).exists():
            manifest, structure = _manifest(root / filename, qualified=require_qualified)
            if (manifest["spec_sha256"] != plan["spec_sha256"]
                    or structure["spec_sha256"] != plan["spec_sha256"]
                    or structure["spec"] != plan["spec"]
                    or manifest["max_length"] != plan["max_length"]):
                raise ValueError("Completed geometry artifact differs from prepared study")
            phase = "qualified" if require_qualified else "exported_unqualified"
            qualified = require_qualified
    return dict(status="authenticated", phase=phase, qualified=qualified,
                max_length=plan["max_length"], spec_sha256=plan["spec_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "pilot", "check"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--spec")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-mib", type=int, default=11008)
    args = parser.parse_args()
    try:
        if args.phase == "prepare":
            if not args.spec:
                parser.error("prepare requires --spec")
            result = prepare(args.root, args.spec)
        elif args.phase == "check":
            result = check(args.root)
        else:
            if torch.device(args.device).type != "cuda":
                raise ValueError("Real run/pilot CLI requires reserved CUDA; CPU fixtures use functions")
            from .jspace_3b_response_cache import reservation
            lease, guard = reservation(args.limit_mib)
            result = globals()[args.phase](args.root, device=args.device, guard=guard)
            result["lease_id"] = lease["id"]
        print(json.dumps(result, allow_nan=False), flush=True)
    except BaseException as error:
        root = Path(args.root)
        if root.exists():
            write_json(root / f"{args.phase}-failure-{time.time_ns()}.json",
                       dict(status="failed", phase=args.phase, error=repr(error)))
        raise


if __name__ == "__main__":
    main()
