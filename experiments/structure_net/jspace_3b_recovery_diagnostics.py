"""Frozen 10 -> 11 diagnostics, never a trainer or a quality certificate.

CPU prepare authenticates historical states and decodes existing generations.
Run requires an external GPU reservation. Native targets are recomputed on each
exact history. Full-width boundary interventions explicitly measure their extra
encode/decode seam against the unsplit production stack; that seam is not silently
attributed to insertion or recovery. All output paths are write-once.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from . import jspace_3b_full_progressive_recovery as recovery
from . import jspace_3b_packed_response_model as packed
from .jspace_qwen_compact_stack import CompactQwenStack

SCHEMA = "jspace.recovery-reset.diagnostics.v1"
ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("qa", "math", "code", "instruction", "summary")
sha, read, write = recovery.sha, recovery.read, recovery.write


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def cpu_only():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU phase requires CUDA_VISIBLE_DEVICES=''")


def family(task):
    task = task.lower()
    if any(x in task for x in ("gsm", "math", "addition")):
        return "math"
    if any(x in task for x in ("mbpp", "code", "python")):
        return "code"
    if any(x in task for x in ("summary", "summar", "xsum", "cnn")):
        return "summary"
    if any(x in task for x in ("arc", "boolq", "qa", "question")):
        return "qa"
    return "instruction"


def repetition(tokens, *, repeats=3, max_period=32):
    """Earliest adjacent periodic run; a syntactic flag, not a correctness grade.

    onset is the first repeated token (start + period), not the first token of
    the original phrase. Evidence detection happens later, at confirmation.
    """
    best = None
    for start in range(len(tokens)):
        for period in range(1, min(max_period, (len(tokens)-start)//repeats)+1):
            unit = tokens[start:start+period]
            if all(tokens[start+j*period:start+(j+1)*period] == unit for j in range(1, repeats)):
                end = start+repeats*period
                while end+period <= len(tokens) and tokens[end:end+period] == unit:
                    end += period
                candidate = dict(start=start, onset=start+period, confirmed_at=start+repeats*period-1,
                                 end=end, period=period, repeats=(end-start)//period)
                if best is None or (candidate["onset"], period) < (best["onset"], best["period"]):
                    best = candidate
    return best


def classify(tokens, eos_ids, cap, supported, prompt_length):
    if cap < 1 or supported < prompt_length or len(tokens) > cap or prompt_length+len(tokens) > supported:
        raise ValueError("Generation exceeds declared cap or positional support")
    eos = [i for i, token in enumerate(tokens) if token in eos_ids]
    if eos and eos != [len(tokens)-1]:
        raise ValueError("Generation contains tokens after EOS")
    loop = repetition(tokens[:-1] if eos else tokens)
    termination = "natural_eos" if eos else "cap_exhausted" if len(tokens) == cap else "incomplete_unknown"
    return dict(termination=termination, repetition=loop,
                phenotype="repetition_flagged" if loop else "clean_cap_exhaustion" if termination == "cap_exhausted" else termination,
                correctness="not_inferred", cap=cap, supported_total=supported)


def history(example_id, task, split, prompt, output, *, kind, generating_state=None):
    if not prompt or not output:
        raise ValueError("A trace needs a nonempty prompt and continuation")
    if split not in ("train", "development", "exposed_development"):
        raise ValueError("Diagnostics may not silently admit TEST/confirmation")
    tokens = list(prompt)+list(output)
    item = dict(example_id=example_id, family=family(task), task=task, split=split,
                prompt_ids=list(prompt), output_ids=list(output), input_ids=tokens,
                history_kind=kind, generating_state=generating_state,
                predicting_positions=list(range(len(prompt)-1, len(tokens)-1)),
                repetition=repetition(list(output)))
    item["history_sha256"] = digest(dict(input_ids=tokens, predicting_positions=item["predicting_positions"]))
    return item


def bind_history(item, receipt):
    expected = digest(dict(input_ids=item["input_ids"], predicting_positions=item["predicting_positions"]))
    if expected != item["history_sha256"] or receipt["history_sha256"] != expected:
        raise ValueError("Teacher/history identity mismatch")


def phase_at(item, step):
    loop = item["repetition"]
    if loop is None:
        return "no_loop"
    if step < loop["onset"]:
        return "before_loop"
    return "inside_loop" if step < loop["end"] else "after_loop"


def _pin(path, inputs, expected=None):
    path = Path(path).resolve()
    value = sha(path)
    if expected is not None and value != expected:
        raise ValueError("Historical artifact changed: "+str(path))
    inputs[str(path)] = value
    return dict(path=str(path), sha256=value)


def inherited_tensors(parent, child):
    """Compare actual saved inherited tensors by native layer, not filename."""
    old_layers = parent["stack"]["_extra_state"]["layer_indices"]
    new_layers = child["stack"]["_extra_state"]["layer_indices"]
    offset = list(new_layers).index(old_layers[0])
    if list(new_layers[offset:]) != list(old_layers):
        raise ValueError("Inherited layer ranges differ")
    compared, changed, absent = 0, [], []
    for name, value in parent["stack"].items():
        if not isinstance(value, torch.Tensor):
            continue
        target = name
        fields = name.split(".")
        if fields[0] in ("blocks", "bridges"):
            fields[1] = str(int(fields[1])+offset)
            target = ".".join(fields)
        elif name.startswith(("feature_i_", "feature_j_")):
            prefix, index = name.rsplit("_", 1)
            target = prefix+"_"+str(int(index)+offset)
        actual = child["stack"].get(target)
        if actual is None:
            absent.append(target)
        else:
            compared += 1
            if not torch.equal(value, actual):
                changed.append(target)
    norm_equal = set(parent["norm"]) == set(child["norm"]) and all(
        torch.equal(v, child["norm"][k]) for k, v in parent["norm"].items())
    return dict(parent_layers=list(old_layers), child_layers=list(new_layers), compared_tensors=compared,
                changed_tensors=changed, missing_tensors=absent, norm_equal=norm_equal,
                inherited_exact=not changed and not absent and norm_equal)


def inventory(spec, inputs):
    states, stages, missing = [], {}, []
    for depth in (10, 11):
        root = Path(spec["progressive_root"])/f"blocks_{depth:02d}"/"recovery"
        plan = read(root/"plan.json")
        result = read(root/"run/gpu_result.json")
        manifest = read(root/"run/gpu_manifest.json")
        if plan["cut"] != 36-depth or plan["model"]["revision"] != spec["native_model_revision"]:
            raise ValueError("Wrong historical layer range or teacher revision")
        _pin(root/"plan.json", inputs)
        if depth == 10:
            for name, wanted in plan["model"]["files"].items():
                _pin(Path(spec["native_model_path"])/name, inputs, wanted)
        _pin(root/"run/gpu_manifest.json", inputs)
        _pin(root/"run/gpu_result.json", inputs, manifest["artifacts"]["gpu_result.json"])
        snapshots = []
        for source, wanted in plan["source_sha256"].items():
            snapshot = root/"producer"/Path(source).relative_to(ROOT)
            if snapshot.exists():
                snapshots.append(_pin(snapshot, inputs, wanted))
            else:
                missing.append(dict(depth=depth, role="producer_snapshot", path=str(snapshot)))
        # All fit geometry used in strict restoration is pinned, not just a receipt.
        for directory in plan["fits"].values():
            for name in ("fit.pt", "relation.pt"):
                path = Path(directory)/name
                _pin(path, inputs, plan.get("input_sha256", {}).get(str(path)))
        candidates = []
        for candidate in result["candidates"]:
            checkpoint = candidate["checkpoint"]
            label = candidate["label"]
            role = "local_fit_before_prealignment" if label == "prefit" else "pre_joint_after_prealignment" if candidate["update"] == 0 else "post_joint"
            row = dict(id=f"blocks{depth}_{label}", depth=depth, cut=36-depth, role=role,
                       label=label, update=candidate["update"], response_tokens=candidate["response_tokens"],
                       checkpoint=checkpoint, available=Path(checkpoint["path"]).is_file(),
                       selected=checkpoint["sha256"] == result["selected"]["checkpoint"]["sha256"],
                       packed_state_sha256=candidate["packed_state_sha256"], source_root=str(root))
            if row["available"]:
                _pin(checkpoint["path"], inputs, checkpoint["sha256"])
                saved = torch.load(checkpoint["path"], map_location="cpu", weights_only=True, mmap=True)
                extra = saved["stack"]["_extra_state"]
                if list(extra["layer_indices"]) != list(range(row["cut"], 36)):
                    raise ValueError("Checkpoint layer identity differs from candidate")
                row["tensor_bytes"] = sum(v.numel()*v.element_size() for group in (saved["stack"], saved["norm"]) for v in group.values() if isinstance(v, torch.Tensor))
                row["checkpoint_schema"] = saved.get("schema", "legacy stack/norm")
                del saved
                states.append(row)
            else:
                missing.append(dict(depth=depth, role=role, path=checkpoint["path"], expected_sha256=checkpoint["sha256"]))
            candidates.append(row)
        export = root/"run/selected/export"
        export_manifest = read(export/"manifest.json")
        if export_manifest["identities"]["checkpoint_sha256"] != result["selected"]["checkpoint"]["sha256"]:
            raise ValueError("Selected export has another checkpoint")
        _pin(export/"manifest.json", inputs)
        for name, wanted in export_manifest["files"].items():
            _pin(export/name, inputs, wanted)
        stages[str(depth)] = dict(plan=plan, candidates=candidates, producer_snapshots=snapshots,
                                  selected_export=str(export), packed_tensor_bytes=export_manifest["tensor_bytes"])
        parent = plan["parent"]
        parent_root = Path(parent["root"])
        parent_result = read(parent_root/"run/gpu_result.json")
        parent_manifest = read(parent_root/"run/gpu_manifest.json")
        _pin(parent_root/"run/gpu_manifest.json", inputs)
        _pin(parent_root/"run/gpu_result.json", inputs, parent_manifest["artifacts"]["gpu_result.json"])
        parent_file = _pin(parent["checkpoint"], inputs, parent["checkpoint_sha256"])
        selected_parent = parent_result["selected"]["checkpoint"]
        lineage = dict(declared_parent=parent_file, selected_parent=selected_parent,
                       selected_parent_used=parent_file["sha256"] == selected_parent["sha256"],
                       tensor_comparisons=[])
        old_saved = torch.load(parent_file["path"], map_location="cpu", weights_only=True, mmap=True)
        # update0000 follows new-block-only local alignment, so an unchanged
        # inherited tail is observable even if the raw prefit was pruned.
        for state in candidates:
            if state["available"] and state["role"] != "post_joint":
                child = torch.load(state["checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True)
                lineage["tensor_comparisons"].append(dict(child=state["checkpoint"],
                    role=state["role"], **inherited_tensors(old_saved, child)))
                del child
        del old_saved
        stages[str(depth)]["lineage"] = lineage
    parent = stages["11"]["plan"]["parent"]
    old = [s for s in states if s["depth"] == 10 and s["checkpoint"]["sha256"] == parent["checkpoint_sha256"]]
    if len(old) != 1:
        raise ValueError("Actual inherited ten-block state is not retained; cannot substitute another checkpoint")
    return states, stages, missing, old[0]["id"]


def memory_estimate(spec, stages):
    """Count original tensor headers; workspace allowance is deliberately estimated."""
    from safetensors import safe_open
    by_layer, native_bytes = defaultdict(int), 0
    dtypes = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "BOOL": 1}
    for file in Path(spec["native_model_path"]).glob("*.safetensors"):
        with safe_open(file, framework="pt", device="cpu") as stream:
            for key in stream.keys():
                view = stream.get_slice(key)
                size = math.prod(view.get_shape())*dtypes[view.get_dtype()]
                native_bytes += size
                if key.startswith("model.layers."):
                    by_layer[int(key.split(".")[2])] += size
    if len(by_layer) != 36:
        raise ValueError("Expected actual headers for 36 native layers")
    compact = max(s["packed_tensor_bytes"] for s in stages.values())
    # One packed model, one streamed native layer, hidden sequences and chunked
    # vocabulary rows. Extra 3 GiB covers eager attention/packing/cuda allocator.
    gpu = compact+max(by_layer.values())+3*1024**3
    return dict(measured=False, native_tensor_bytes=native_bytes, largest_native_layer_bytes=max(by_layer.values()),
                largest_packed_tensor_bytes=compact, gpu_estimated_bytes=gpu,
                host_estimated_bytes=4*native_bytes+8*compact,
                workspace_allowance_bytes=3*1024**3,
                scope="Conservative tensor-based estimate, not a measured peak or hardware qualification")


def prepare(root, spec_path, *, per_family=1, cap_tokens=(64, 128, 256), limit_mib=11008):
    cpu_only()
    root = Path(root).resolve()
    if root.exists():
        raise FileExistsError(root)
    if per_family < 1 or not cap_tokens or min(cap_tokens) < 1:
        raise ValueError("Positive panel and caps required")
    spec = read(spec_path)
    if spec["schema"] != "jspace.recovery-reset.spec.v1":
        raise ValueError("Unexpected shared spec")
    inputs = {}
    _pin(spec_path, inputs)
    states, stages, missing, old = inventory(spec, inputs)
    support = min(s["plan"]["max_length"] for s in stages.values())
    panel, exclusions, counts = [], [], defaultdict(int)
    data_plan_path = Path(spec["historical_data_root"])/"plan.json"
    _pin(data_plan_path, inputs)
    data = read(data_plan_path)
    # The frozen existing panel includes TRAIN calibration ancestry and exposed
    # DEV only. No new TEST/confirmation data is opened by this adapter.
    for item in data["items"]:
        split = item["split"]
        if split not in ("train", "development"):
            continue
        key = (split, family(item["task"]))
        if counts[key] >= per_family:
            continue
        positions, targets, ids = item["loss_positions"], item["targets"], item["input_ids"]
        if not positions or positions != list(range(positions[0], positions[0]+len(targets))):
            raise ValueError("Reference trace needs complete contiguous response positions")
        prompt = ids[:positions[0]+1]
        if ids[len(prompt):] != targets or targets[-1] != item["eos_token_id"]:
            raise ValueError("Reference tokens/positions disagree")
        if len(ids) > support:
            exclusions.append(dict(example_id=item["example_id"], reason="reference_exceeds_original_support", total=len(ids)))
            continue
        panel.append(history(item["example_id"], item["task"], split, prompt, targets, kind="reference"))
        counts[key] += 1
    absent = [dict(split=split, family=f) for split in ("train", "development") for f in FAMILIES if counts[(split, f)] < per_family]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec["native_model_path"], local_files_only=True, trust_remote_code=False)
    decoded = []
    for depth in (10, 11):
        path = Path(stages[str(depth)]["selected_export"]).parent/"generation.json"
        historical_manifest = read(Path(stages[str(depth)]["selected_export"]).parents[1]/"gpu_manifest.json")
        _pin(path, inputs, historical_manifest["artifacts"]["selected/generation.json"])
        generation = read(path)
        state_sha = generation["selected_checkpoint"]["sha256"]
        selected = next(s for s in states if s["depth"] == depth and s["selected"])
        if state_sha != selected["checkpoint"]["sha256"]:
            raise ValueError("Saved generation and selected state differ")
        for row in [*generation["canaries"]["records"], *generation["corpus"]]:
            example = row.get("example_id", row.get("id"))
            task = row.get("task", row.get("id", "instruction"))
            ids, output = row["input_ids"], row["output_ids"]
            if not output:
                exclusions.append(dict(example_id=example, depth=depth, reason="empty_saved_output"))
                continue
            if len(ids)+len(output) > support:
                exclusions.append(dict(example_id=example, depth=depth, reason="saved_history_exceeds_support"))
                continue
            trace = history(example, task, "exposed_development", ids, output, kind="saved_generated", generating_state=state_sha)
            panel.append(trace)
            decoded.append(dict(depth=depth, example_id=example, task=task, checkpoint_sha256=state_sha,
                                prompt=tokenizer.decode(ids, skip_special_tokens=False),
                                response=tokenizer.decode(output, skip_special_tokens=False),
                                classification=classify(output, spec["eos_token_ids"], row["max_new_tokens"], support, len(ids)),
                                history_sha256=trace["history_sha256"], causal_evidence=False))
    sources = recovery.base.cache.prior.source.inherited.local_import_closure([Path(__file__)])
    plan = dict(schema=SCHEMA, spec=spec, spec_path=str(Path(spec_path).resolve()), spec_sha256=sha(spec_path),
                states=states, stages=stages, missing=missing, old_suffix_state=old,
                panel=panel, panel_missing=absent, exclusions=exclusions, support=support,
                cap_tokens=sorted(set(cap_tokens)), limit_mib=limit_mib,
                memory_estimate=memory_estimate(spec, stages), inputs=inputs,
                sources={str(p): sha(p) for p in sorted(sources)},
                attribution="Controlled boundary contrasts only; historical changes and task/position correlations are not causal proof",
                conditional_output_aware_fit="Not activated: requires completed diagnostic evidence and lead decision")
    plan["fingerprint"] = digest(plan)
    root.mkdir(parents=True)
    for path, wanted in plan["sources"].items():
        destination = root/"producer"/Path(path).relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        if sha(destination) != wanted:
            raise ValueError("Source changed during snapshot")
    write(root/"plan.json", plan)
    write(root/"decoded.json", dict(status="observational_only", records=decoded, missing=missing,
                                   exclusions=exclusions, no_teacher_divergence_measured=True))
    write(root/"prepare_manifest.json", dict(fingerprint=plan["fingerprint"], spec_sha256=plan["spec_sha256"],
        files={name: sha(root/name) for name in ("plan.json", "decoded.json")}))
    return dict(status="prepared", fingerprint=plan["fingerprint"], states=len(states), histories=len(panel),
                panel_missing=absent, memory_estimate=plan["memory_estimate"])


def verify(root, *, live=False):
    root = Path(root)
    p = read(root/"plan.json")
    body = dict(p)
    fingerprint = body.pop("fingerprint")
    if p["schema"] != SCHEMA or digest(body) != fingerprint:
        raise ValueError("Diagnostic plan changed")
    for path, wanted in p["inputs"].items():
        if sha(path) != wanted:
            raise ValueError("Frozen input changed: "+path)
    for path, wanted in p["sources"].items():
        if sha(root/"producer"/Path(path).relative_to(ROOT)) != wanted or live and sha(path) != wanted:
            raise ValueError("Diagnostic producer changed: "+path)
    manifest = read(root/"prepare_manifest.json")
    if manifest["fingerprint"] != fingerprint or manifest["spec_sha256"] != p["spec_sha256"]:
        raise ValueError("Prepared receipt belongs to another specification")
    for name, wanted in manifest["files"].items():
        if sha(root/name) != wanted:
            raise ValueError("Prepared artifact changed: "+name)
    return p


def _kwargs(hidden, ids, rotary):
    positions = torch.arange(ids.shape[1], device=ids.device)[None].expand_as(ids)
    allowed = torch.ones(ids.shape[1], ids.shape[1], device=ids.device, dtype=torch.bool).tril()[None, None]
    attention = torch.zeros(allowed.shape, device=ids.device, dtype=hidden.dtype).masked_fill(~allowed, torch.finfo(hidden.dtype).min)
    return dict(attention_mask=attention, position_ids=positions, position_embeddings=rotary(hidden, positions),
                cache_position=positions[0], use_cache=False)


def slice_stack(stack, start, stop):
    """Recompose only for explicit full-width boundary interventions.

    Interior bridges are copied from the saved state, never recomputed from
    rounded bases. Geometry is retained until after splitting.
    """
    blocks = deepcopy(list(stack.blocks[start:stop]))
    result = CompactQwenStack(blocks, geometry_mode=stack.geometry_mode,
                              pad_to_multiple=stack.pad_to_multiple, enabled=False)
    result.bridges = deepcopy(stack.bridges[start:max(start, stop-1)])
    return result.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16, strip_unused=True).eval().requires_grad_(False)


def restore(state, p, guard):
    builder = recovery.base.builder
    plan = p["stages"][str(state["depth"])]["plan"]
    stack, norm, embedding, rotary, proof = builder.build_stack(p["spec"]["native_model_path"],
        {int(k): v for k, v in plan["fits"].items()}, state["cut"], previous=state["checkpoint"]["path"],
        device="cpu", trainable=False)
    del rotary
    template, _ = packed.load_packed_snapshot(p["stages"][str(state["depth"])]["selected_export"], device="cpu")
    if not torch.equal(embedding, template.embedding):
        raise ValueError("Native tied embedding differs from authenticated export")
    rotary = deepcopy(template.rotary)
    del template
    insertion = slice_stack(stack, 0, 1) if state["cut"] == 25 else None
    tail = slice_stack(stack, 1, len(stack.blocks)) if state["cut"] == 25 else None
    model = packed.PackedResponseModel(builder.packed_clone(stack, device="cpu"), norm, embedding, rotary,
        config=builder.native._config(p["spec"]["native_model_path"]), max_length=p["support"])
    del stack
    actual = packed.packed_state_digest(model)["packed_state_sha256"]
    if actual != state["packed_state_sha256"]:
        raise ValueError("Reconstructed state differs from historically scored packed state")
    model.guard = guard
    return model, insertion, tail, proof


def teacher_trace(model, item, p, guard):
    """Reuse streamed prefix; compute native 25..35 with one resident layer."""
    bind_history(item, item)
    ids = torch.tensor([item["input_ids"]], device=model.embedding.device)
    original_cut = model.cut
    model.cut = 25
    try:
        with packed.attach_prefix(model, p["spec"]["native_model_path"], mode="cpu"):
            hidden = model.native_prefix(ids, torch.ones_like(ids, dtype=torch.bool))
    finally:
        model.cut = original_cut
    boundary25 = hidden.cpu()
    kwargs = _kwargs(hidden, ids, model.rotary)
    builder = recovery.base.builder
    boundary26 = None
    for index in range(25, 36):
        guard()
        block = builder.native.load_native_block(p["spec"]["native_model_path"], index,
                                                 device="cpu", config=model.config).eval().requires_grad_(False)
        block.to(model.embedding.device)
        hidden = block(hidden, **kwargs)[0]
        if index == 25:
            boundary26 = hidden.cpu()
        del block
    _, norm, rope = builder.native.load_readout_norm(p["spec"]["native_model_path"], device="cpu", config=model.config)
    hidden = norm.to(model.embedding.device)(hidden)
    del norm, rope
    return dict(history_sha256=item["history_sha256"], boundary25=boundary25, boundary26=boundary26,
                teacher_hidden=hidden.cpu(), teacher_revision=p["spec"]["native_model_revision"])


def token_metrics(student, teacher, embedding, item, receipt, *, chunk=8):
    bind_history(item, receipt)
    rows = []
    positions = item["predicting_positions"]
    for begin in range(0, len(positions), chunk):
        selected = positions[begin:begin+chunk]
        actual = F.linear(student[0, selected].to(embedding), embedding).float()
        reference = F.linear(teacher[0, selected].to(embedding), embedding).float()
        logq, logp = actual.log_softmax(-1), reference.log_softmax(-1)
        kl = (logp.exp()*(logp-logq)).sum(-1)
        for offset, position in enumerate(selected):
            step = begin+offset
            target = item["output_ids"][step]
            rows.append(dict(step=step, position=position, token_id=target, phase=phase_at(item, step),
                forward_kl=float(kl[offset]), teacher_probability=float(logp[offset, target].exp()),
                student_probability=float(logq[offset, target].exp()), teacher_nll=float(-logp[offset, target]),
                student_nll=float(-logq[offset, target]), greedy_agreement=bool(actual[offset].argmax() == reference[offset].argmax())))
    return rows


def boundary_metrics(actual, reference, item):
    if actual.shape != reference.shape:
        raise ValueError("Boundary lengths differ")
    a, b = actual.float(), reference.float()
    mse = (a-b).square().mean(-1)[0]
    scale = b.square().mean(-1)[0].clamp_min(1e-12)
    return [dict(position=position, mse=float(mse[position]), relative_rms=float((mse[position]/scale[position]).sqrt()))
            for position in item["predicting_positions"]]


def aggregate(rows):
    sums = defaultdict(lambda: defaultdict(float))
    for row in rows:
        identity = (row["contrast"], row["tail_state"], row.get("insertion_state"), row["family"], row["history_kind"], row["split"])
        for token in row["tokens"]:
            for stratum in ("all", token["phase"], f"position_{token['position']//32*32:04d}_{token['position']//32*32+31:04d}"):
                values = sums[identity+(stratum,)]
                values["tokens"] += 1
                for metric in ("forward_kl", "teacher_probability", "teacher_nll", "student_nll", "greedy_agreement"):
                    values[metric] += token[metric]
    return [dict(contrast=k[0], tail_state=k[1], insertion_state=k[2], family=k[3], history_kind=k[4], split=k[5],
                 stratum=k[6], tokens=int(v["tokens"]), **{n: x/v["tokens"] for n, x in v.items() if n != "tokens"})
            for k, v in sums.items()]


def paired_effects(rows):
    """Only exact same-history, same-tail native/compressed boundary pairs."""
    native, result = {}, []
    for row in rows:
        if row["contrast"] == "native_boundary":
            key = (row["history_sha256"], row["tail_state"], row["norm_state"])
            if key in native:
                raise ValueError("Duplicate native control")
            native[key] = row
    for row in rows:
        if row["contrast"] != "compressed_boundary":
            continue
        key = (row["history_sha256"], row["tail_state"], row["norm_state"])
        if key not in native:
            raise ValueError("Missing matched native-boundary control")
        control = native[key]
        if [t["position"] for t in row["tokens"]] != [t["position"] for t in control["tokens"]]:
            raise ValueError("Paired positions differ")
        result.append(dict(history_sha256=key[0], tail_state=key[1], norm_state=key[2],
                           insertion_state=row["insertion_state"],
                           mean_kl_difference=sum(a["forward_kl"]-b["forward_kl"] for a, b in zip(row["tokens"], control["tokens"]))/len(row["tokens"]),
                           scope="Full-width boundary intervention; consult measured production recomposition seam"))
    return result


def suffix_effects(rows, old_state):
    """Factorial tail-weight contrast with history, insertion and norm fixed."""
    controls, result = {}, []
    def key(row):
        return (row["history_sha256"], row["example_id"], row["history_kind"],
                row["generating_state"], row["split"], row["contrast"],
                row["insertion_state"], row["norm_state"])
    for row in rows:
        if row["tail_state"] == old_state and row["contrast"] in ("native_boundary", "compressed_boundary"):
            controls[key(row)] = row
    for row in rows:
        if row["tail_state"] == old_state or row["contrast"] not in ("native_boundary", "compressed_boundary"):
            continue
        control = controls.get(key(row))
        if control is None:
            raise ValueError("Suffix contrast lacks same-history/insertion/norm control")
        if [t["position"] for t in row["tokens"]] != [t["position"] for t in control["tokens"]]:
            raise ValueError("Suffix contrast positions differ")
        result.append(dict(history_sha256=row["history_sha256"], history_kind=row["history_kind"],
            insertion_state=row["insertion_state"], old_tail=old_state, updated_tail=row["tail_state"],
            norm_state=row["norm_state"], contrast=row["contrast"],
            mean_kl_difference=sum(a["forward_kl"]-b["forward_kl"] for a, b in zip(row["tokens"], control["tokens"]))/len(row["tokens"]),
            scope="Fixed full-width boundary, history and norm; changed downstream suffix weights only"))
    return result


def _emit(out, rows, item, *, contrast, state, norm_state, insertion_state, tokens, **extra):
    row = dict(example_id=item["example_id"], family=item["family"], split=item["split"],
               history_kind=item["history_kind"], history_sha256=item["history_sha256"],
               generating_state=item["generating_state"], tail_state=state, norm_state=norm_state,
               insertion_state=insertion_state, contrast=contrast, tokens=tokens, **extra)
    rows.append(row)
    with (out/"traces.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False)+"\n")


def generate_caps(model, item, p):
    prompt = item["prompt_ids"]
    room = model.max_length-len(prompt)
    caps = [c for c in p["cap_tokens"] if c <= room]
    exclusions = [dict(example_id=item["example_id"], cap=c, reason="cap_exceeds_supported_total", prompt_tokens=len(prompt), supported=model.max_length)
                  for c in p["cap_tokens"] if c > room]
    if not caps:
        return [], exclusions
    output = []
    for _ in range(max(caps)):
        ids = torch.tensor([prompt+output], device=model.embedding.device)
        token = int(model.next_logits(ids).argmax())
        output.append(token)
        if token in p["spec"]["eos_token_ids"]:
            break
    rows = [dict(cap=cap, output_ids=output[:cap],
                 classification=classify(output[:cap], p["spec"]["eos_token_ids"], cap, model.max_length, len(prompt))) for cap in caps]
    return rows, exclusions


@torch.inference_mode()
def run(root):
    root = Path(root).resolve()
    p = verify(root, live=True)
    out = root/"run"
    out.mkdir(exist_ok=False)
    lease = None
    started = time.monotonic()
    try:
        lease, guard = recovery.base.cache.reservation(p["limit_mib"])
        if len(lease["allocation"]) != 1 or os.environ.get("CUDA_VISIBLE_DEVICES") not in lease["allocation"]:
            raise ValueError("Exactly one externally reserved visible GPU required")
        if p["panel_missing"]:
            raise ValueError("Required TRAIN/DEV family panel incomplete: "+str(p["panel_missing"]))
        torch.set_num_threads(2)
        rows, cap_rows, exclusions = [], [], list(p["exclusions"])
        cache = out/"histories"
        cache.mkdir()
        panel = list(p["panel"])
        states = p["states"]
        old = next(s for s in states if s["id"] == p["old_suffix_state"])
        norm_state = old["checkpoint"]["sha256"]
        # Pin common readout norm to the actual inherited state. Own-norm
        # production rows are separate, so suffix drift is not norm drift.
        old_header = torch.load(old["checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True)
        fixed_norm = {k: v.clone() for k, v in old_header["norm"].items()}
        del old_header
        with recovery.base.model_source.execution.arithmetic_policy():
            # Fixed-weight output-cap diagnostic: one greedy trajectory per
            # state/prompt, sliced only at predeclared supported caps.
            for state in [s for s in states if s["selected"]]:
                model, insertion, tail, _ = restore(state, p, guard)
                del insertion, tail
                model.to("cuda:0")
                with packed.attach_prefix(model, p["spec"]["native_model_path"], mode="cpu"):
                    for item in [r for r in p["panel"] if r["history_kind"] == "reference"]:
                        generated, omitted = generate_caps(model, item, p)
                        exclusions.extend(dict(state=state["id"], **e) for e in omitted)
                        for row in generated:
                            cap_rows.append(dict(state=state["id"], checkpoint_sha256=state["checkpoint"]["sha256"],
                                                 example_id=item["example_id"], family=item["family"], split=item["split"], **row))
                        if generated:
                            panel.append(history(item["example_id"], item["task"], item["split"], item["prompt_ids"],
                                generated[-1]["output_ids"], kind="fresh_generated", generating_state=state["checkpoint"]["sha256"]))
                del model
                gc.collect()
                torch.cuda.empty_cache()
            write(out/"caps.json", dict(records=cap_rows, exclusions=exclusions, fixed_weights=True,
                scope="Greedy shared trajectory; clean cap exhaustion is not graded incorrect"))
            # Deduplicate exact histories for native capture, preserving all
            # origin labels in the panel. No reference cache is used.
            model, ins, tail, _ = restore(old, p, guard)
            del ins, tail
            model.to("cuda:0")
            captured = set()
            for item in panel:
                key = item["history_sha256"]
                if key not in captured:
                    torch.save(teacher_trace(model, item, p, guard), cache/(key+".pt"))
                    captured.add(key)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            # Save insertion outputs for every genuinely retained 11-state.
            insertion_states = [s for s in states if s["cut"] == 25]
            for state in insertion_states:
                model, insertion, tail, _ = restore(state, p, guard)
                del tail
                model.to("cuda:0")
                insertion.to("cuda:0")
                for key in captured:
                    receipt = torch.load(cache/(key+".pt"), weights_only=True)
                    item = next(i for i in panel if i["history_sha256"] == key)
                    ids = torch.tensor([item["input_ids"]], device="cuda:0")
                    boundary = receipt["boundary25"].to("cuda:0")
                    value = insertion(boundary, **_kwargs(boundary, ids, model.rotary))[0]
                    torch.save(dict(history_sha256=key, insertion_state=state["checkpoint"]["sha256"],
                                    boundary=value.cpu()), cache/(key+"_"+state["id"]+".pt"))
                del model, insertion
                gc.collect()
                torch.cuda.empty_cache()
            for state in states:
                model, insertion, tail, proof = restore(state, p, guard)
                del insertion
                # Keep one suffix in VRAM: first production, then sliced tail.
                model.to("cuda:0")
                for item in panel:
                    receipt = torch.load(cache/(item["history_sha256"]+".pt"), weights_only=True)
                    bind_history(item, receipt)
                    ids = torch.tensor([item["input_ids"]], device="cuda:0")
                    boundary = receipt[f"boundary{state['cut']}"].to("cuda:0")
                    actual = model.hidden_forward(dict(input_ids=ids, boundary_hidden=boundary))
                    teacher = receipt["teacher_hidden"]
                    _emit(out, rows, item, contrast="production", state=state["checkpoint"]["sha256"],
                          norm_state=state["checkpoint"]["sha256"], insertion_state=None,
                          tokens=token_metrics(actual, teacher, model.embedding, item, receipt), state_id=state["id"])
                    torch.save(actual.cpu(), cache/(item["history_sha256"]+"_production_"+state["id"]+".pt"))
                if tail is not None:
                    model.stack.to("cpu")
                    model.stack = tail.to("cuda:0")
                    model.cut = 26
                own_norm = deepcopy(model.norm.state_dict())
                model.norm.load_state_dict(fixed_norm)
                for item in panel:
                    key = item["history_sha256"]
                    receipt = torch.load(cache/(key+".pt"), weights_only=True)
                    bind_history(item, receipt)
                    ids = torch.tensor([item["input_ids"]], device="cuda:0")
                    native = receipt["boundary26"].to("cuda:0")
                    for source in [None, *insertion_states]:
                        if source is None:
                            boundary = native
                        else:
                            inserted = torch.load(cache/(key+"_"+source["id"]+".pt"), weights_only=True)
                            bind_history(item, inserted)
                            if inserted["insertion_state"] != source["checkpoint"]["sha256"]:
                                raise ValueError("Insertion state mismatch")
                            boundary = inserted["boundary"].to("cuda:0")
                        actual = model.hidden_forward(dict(input_ids=ids, boundary_hidden=boundary))
                        _emit(out, rows, item, contrast="native_boundary" if source is None else "compressed_boundary",
                              state=state["checkpoint"]["sha256"], norm_state=norm_state,
                              insertion_state=None if source is None else source["checkpoint"]["sha256"],
                              tokens=token_metrics(actual, receipt["teacher_hidden"], model.embedding, item, receipt),
                              boundary_reconstruction=boundary_metrics(boundary, native, item),
                              boundary_index=26, inserted_block_index=None if source is None else 25, state_id=state["id"])
                        if source is not None and source["id"] == state["id"]:
                            model.norm.load_state_dict(own_norm)
                            composed = model.hidden_forward(dict(input_ids=ids, boundary_hidden=boundary))
                            production = torch.load(cache/(key+"_production_"+state["id"]+".pt"), weights_only=True).to(composed)
                            seam = boundary_metrics(composed, production, item)
                            _emit(out, rows, item, contrast="recomposition_seam", state=state["checkpoint"]["sha256"],
                                  norm_state=state["checkpoint"]["sha256"], insertion_state=source["checkpoint"]["sha256"],
                                  tokens=token_metrics(composed, production, model.embedding, item, receipt),
                                  hidden_reconstruction=seam, teacher_scope="Same-state unsplit production, NOT native teacher")
                            model.norm.load_state_dict(fixed_norm)
                del model, tail, own_norm
                gc.collect()
                torch.cuda.empty_cache()
        # Identical histories can appear under multiple panel labels. Pair by
        # panel stratum to avoid accidentally matching their generating states.
        effects = []
        for identity in {(r["example_id"], r["history_kind"], r["generating_state"], r["split"]) for r in rows}:
            effects.extend(paired_effects([r for r in rows if (r["example_id"], r["history_kind"], r["generating_state"], r["split"]) == identity]))
        result = dict(schema=SCHEMA, status="completed", fingerprint=p["fingerprint"], spec_sha256=p["spec_sha256"],
            records=len(rows), histories=len(panel), paired_boundary_effects=effects,
            paired_suffix_effects=suffix_effects(rows, norm_state), summary=aggregate(rows),
            authenticated_lineage={depth: stage["lineage"] for depth, stage in p["stages"].items()},
            missing=p["missing"], exclusions=exclusions, old_suffix_state=p["old_suffix_state"],
            lease_id=lease["id"], allocation=lease["allocation"], gpu_name=torch.cuda.get_device_name(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), seconds=time.monotonic()-started,
            conclusion="Controlled traces available; mechanism attribution requires reviewing paired effects and recomposition seam, not correlation alone",
            training=False, confirmed=False)
        write(out/"result.json", result)
        write(out/"manifest.json", dict(fingerprint=p["fingerprint"], spec_sha256=p["spec_sha256"], status="completed",
            files={str(f.relative_to(out)): sha(f) for f in out.rglob("*") if f.is_file()}))
        return dict(status="completed", fingerprint=p["fingerprint"], records=len(rows), seconds=result["seconds"])
    except BaseException as error:
        write(out/"failure.json", dict(status="failed", fingerprint=p["fingerprint"], lease=lease,
              error_type=type(error).__name__, error=str(error), partial_evidence_preserved=True))
        raise


def check(root):
    cpu_only()
    p = verify(root)
    root = Path(root)
    if (root/"run/failure.json").exists():
        raise RuntimeError("Failed diagnostic retained at "+str(root/"run/failure.json"))
    if not (root/"run/manifest.json").exists():
        return dict(status="prepared_no_gpu_evidence", fingerprint=p["fingerprint"], panel_missing=p["panel_missing"])
    manifest = read(root/"run/manifest.json")
    if manifest["fingerprint"] != p["fingerprint"]:
        raise ValueError("Run belongs to another diagnostic plan")
    for name, wanted in manifest["files"].items():
        if sha(root/"run"/name) != wanted:
            raise ValueError("Diagnostic result changed: "+name)
    result = read(root/"run/result.json")
    return dict(status=result["status"], fingerprint=result["fingerprint"], records=result["records"], confirmed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--per-family", type=int, default=1)
    parser.add_argument("--caps", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--limit-mib", type=int, default=11008)
    args = parser.parse_args()
    if args.phase == "prepare":
        if args.spec is None:
            parser.error("prepare requires --spec")
        result = prepare(args.root, args.spec, per_family=args.per_family, cap_tokens=args.caps, limit_mib=args.limit_mib)
    else:
        result = run(args.root) if args.phase == "run" else check(args.root)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
