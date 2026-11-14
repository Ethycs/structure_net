"""Reserved, sequential Qwen adaptation of the quadratic-attention recipe.

The original teacher stays fixed. Each arm advances its own frozen student
prefix; no trained block or target is borrowed from the earlier joint run.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
MODULE = "experiments.structure_net.jspace_quadratic_recipe"
STUDY_ID = "JSPACE-QWEN-QUADRATIC-01"
SCHEMA = "jspace.quadratic-recipe.v1"
DEFAULT_ROOT = ROOT / "data/experiments/jspace_quadratic_recipe/20261114_v1"
TEACHER = ROOT / "data/experiments/jspace_full_compact/20261113_capture_v1"
TASK_SOURCE = ROOT / "data/experiments/jspace_qwen_parallel/20261114_v2"
PRIOR_COMPLETION = ROOT / "data/experiments/jspace_allblocks_completion/20261114_v2"
PROTOCOL = ROOT / "docs/07 - Status Reports/jspace/2026-11-14_jspace-quadratic-recipe-plan.md"
ARMS = ("softmax", "quadratic")
CONFIG = dict(seed=27861, layers=36, rank=1024, head_dim=64, krylov_block_width=1024,
    krylov_steps=1, native_width=2048, native_head_dim=128, native_mlp_width=11008,
    epochs=12, microbatch=4, accumulation=4, learning_rate=1.2e-4, temperature=2.,
    gradient_clip=1., position_capacity=416, weight_decay=0., hyperdeterminant=False,
    metric="learned positive RoPE-pair diagonal, shared per KV group; folded at deployment",
    teacher="fixed native BF16-rounded weights and immutable BF16 boundary/logit cache",
    basis_operator="supervised choice-margin Jacobian Gramian, original native capture",
    dtype="float32", minimum_size_reduction=.38, minimum_speedup=1.4,
    maximum_accuracy_drop=.05, cpu_threads=1, cpu_lengths=[128, 256],
    cpu_repeats=5, cpu_warmup=2, gpu_allocation="GPU-8cf440df=5632")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""): value.update(block)
    return value.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def save(path, value):
    with Path(path).open("xb") as stream: torch.save(value, stream)


def event(root, kind, **values):
    value = dict(time=datetime.now(timezone.utc).isoformat(), kind=kind, **values)
    with (Path(root)/"events.jsonl").open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False)+"\n"); stream.flush()
    print(json.dumps(value, allow_nan=False), flush=True)


def sources():
    from .jspace_qwen_parallel import sources as previous_sources
    paths = previous_sources() + [PROTOCOL]
    paths += list(Path(__file__).parent.glob("jspace_quadratic*.py"))
    paths += list((ROOT/"tests/structure_net").glob("test_jspace_quadratic*.py"))
    return sorted(set(paths))


def prepare(root):
    root = Path(root).resolve()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("Hide CUDA for CPU preparation")
    if root.exists(): raise FileExistsError(root)
    parent, capture = read(TEACHER/"plan.json"), read(TEACHER/"capture_manifest.json")
    if parent["scientific_fingerprint"]!=digest({k:v for k,v in parent.items() if k!="scientific_fingerprint"}):
        raise ValueError("Original teacher protocol fingerprint changed")
    if capture["status"] != "completed" or capture["systems_only"]:
        raise ValueError("Complete scientific original-teacher cache required")
    if capture["fingerprint"] != parent["scientific_fingerprint"]:
        raise ValueError("Native capture provenance mismatch")
    tasks, items = read(TASK_SOURCE/"tasks.json"), read(TEACHER/"capture/items.json")
    task_plan=read(TASK_SOURCE/"plan.json")
    if (task_plan["task_digest"]!=digest(tasks) or task_plan["fingerprint"] !=
            digest({k:v for k,v in task_plan.items() if k!="fingerprint"})):
        raise ValueError("Task-source fingerprint or identities changed")
    if {k: len(v) for k, v in tasks["splits"].items()} != dict(train=512, development=64, evaluation=512):
        raise ValueError("Fixed 512/64/512 populations required")
    for split in ("train", "development"):
        if {c["identity"] for c in tasks["splits"][split]} != {c["identity"] for c in items["items"] if c["split"] == split}:
            raise ValueError("Original teacher rows do not match task identities")
        cached={c["identity"]:c for c in items["items"] if c["split"]==split}
        for case in tasks["splits"][split]:
            if any(case[k]!=cached[case["identity"]][k] for k in ("input_ids","choices","family","answer")):
                raise ValueError("Task content changed under an existing cache identity")
    identities = [{c["identity"] for c in tasks["splits"][s]} for s in ("train", "development", "evaluation")]
    for split in ("train", "development", "evaluation"):
        rows=tasks["splits"][split]
        counts=Counter(c["family"] for c in rows)
        if (len({c["identity"] for c in rows})!=len(rows) or set(counts)!={"arc","boolq","hellaswag","winogrande"}
                or set(counts.values())!={len(rows)//4}):
            raise ValueError("Distinct, family-balanced task rows required")
    if any(identities[i] & identities[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError("Task splits overlap")
    needed = ["capture/items.json", "capture/statistics.pt", "capture/teacher_logits.npy"]
    needed += [f"capture/boundary_{i:02d}.npy" for i in range(37)]
    inputs = {}
    for name in needed:
        actual = sha(TEACHER/name)
        if capture["artifact_sha256"][name] != actual: raise ValueError("Changed teacher cache: "+name)
        inputs[str(TEACHER/name)] = actual
    for path in (TEACHER/"plan.json", TEACHER/"capture_manifest.json", TASK_SOURCE/"plan.json", TASK_SOURCE/"tasks.json"):
        inputs[str(path)] = sha(path)
    model = parent["model"]
    for name, expected in model["files"].items():
        if sha(Path(model["path"])/name) != expected: raise ValueError("Changed original teacher weight/config")
    required_names = ("jspace_quadratic_basis.py", "jspace_quadratic_qwen.py", "jspace_quadratic_train.py",
                      "jspace_quadratic_runtime.py", "jspace_quadratic_evaluate.py")
    if any(not Path(__file__).with_name(name).is_file() for name in required_names):
        raise ValueError("Recipe producers must be complete before freezing")
    # Preserve room for this experiment and the still-running prior campaign.
    free = shutil.disk_usage(root.parent if root.parent.exists() else ROOT/"data/experiments").free
    if free < 80*1024**3: raise RuntimeError("At least 80 GiB free required for this additional campaign")
    plan = dict(schema=SCHEMA, study_id=STUDY_ID, config=CONFIG, model=model,
        teacher_root=str(TEACHER), teacher_fingerprint=parent["scientific_fingerprint"],
        task_source=str(TASK_SOURCE), prior_completion=str(PRIOR_COMPLETION),
        label_ids=parent["label_ids"], task_digest=digest(tasks), items_digest=digest(items),
        input_sha256=inputs, source_sha256={str(p): sha(p) for p in sources()},
        immutable_stats={str(p):file_identity(p) for p in
            [*map(Path,inputs),*[Path(model["path"])/name for name in model["files"]]]},
        created_at=datetime.now(timezone.utc).isoformat(),
        budget=dict(free_disk_bytes=free, additional_disk_allowance_bytes=80*1024**3,
                    context_tokens=items["tokens"], prefix_bank_bytes=items["tokens"]*1024*4),
        scope="exploratory restricted four-task benchmark; two matched sequential arms; no MNIST replication claim")
    plan["fingerprint"] = digest(plan)
    root.mkdir(parents=True)
    for path in sources():
        target = root/"producer"/path.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    write(root/"plan.json", plan); write(root/"tasks.json", tasks); write(root/"items.json", items)
    return dict(status="prepared", fingerprint=plan["fingerprint"], arms=list(ARMS), layers=36)


def verify(root):
    root = Path(root); plan = read(root/"plan.json")
    if (plan["schema"] != SCHEMA or plan["config"] != CONFIG or
            plan["fingerprint"] != digest({k:v for k,v in plan.items() if k != "fingerprint"})):
        raise ValueError("Changed quadratic recipe protocol")
    if set(plan["source_sha256"]) != {str(p) for p in sources()}:
        raise ValueError("Changed producer closure")
    for name, expected in plan["source_sha256"].items():
        if sha(name) != expected or sha(root/"producer"/Path(name).relative_to(ROOT)) != expected:
            raise ValueError("Changed frozen producer: "+name)
    if digest(read(root/"tasks.json")) != plan["task_digest"] or digest(read(root/"items.json")) != plan["items_digest"]:
        raise ValueError("Changed task/cache identities")
    for path,identity in plan["immutable_stats"].items():
        if file_identity(path)!=identity:raise ValueError("Pinned input metadata changed: "+path)
    return plan


def file_identity(path):
    stat=Path(path).stat()
    return [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns]


def verify_consumed(plan, paths):
    for path in paths:
        path=Path(path)
        if sha(path)!=plan["input_sha256"][str(path)]:raise ValueError("Changed consumed teacher input: "+str(path))


def verify_layer_fit(root, layer):
    manifest=checked(root,Path(root)/"basis",hashes=False)
    for name in ("fit.pt","fit.json"):
        path=fit_path(root,layer).with_name(name)
        if sha(path)!=manifest["artifacts"][str(path.relative_to(root))]:raise ValueError("Changed fitted geometry")


def setup(root):
    from .jspace_full_lease import wait_ready
    lease = wait_ready([0]); plan = verify(root)
    torch.cuda.init(); torch.set_num_threads(4); torch.manual_seed(CONFIG["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False; torch.cuda.reset_peak_memory_stats()
    return plan, lease, lambda: wait_ready([0])


def complete(root, directory, files, **values):
    root, directory = Path(root), Path(directory); plan = verify(root)
    value = dict(status="completed", fingerprint=plan["fingerprint"],
        artifacts={str(Path(p).relative_to(root)):sha(p) for p in files}, **values)
    write(directory/"manifest.json", value)
    return {k:v for k,v in value.items() if k != "artifacts"}


def checked(root, directory, hashes=True):
    root, directory = Path(root), Path(directory); plan = verify(root); value = read(directory/"manifest.json")
    if value["status"] != "completed" or value["fingerprint"] != plan["fingerprint"]:
        raise ValueError("Stage fingerprint/completion mismatch")
    if hashes:
        for name, expected in value["artifacts"].items():
            if sha(root/name) != expected: raise ValueError("Changed completed artifact: "+name)
    return value


def load(path): return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def native_config(plan):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(plan["model"]["path"], local_files_only=True)
    config._attn_implementation = "eager"
    return config


def rotary_module(plan, device):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    return Qwen2RotaryEmbedding(config=native_config(plan), device=device).eval()


def fit_path(root, layer): return Path(root)/"basis"/f"block_{layer:02d}"/"fit.pt"
def train_path(root, arm, layer):
    if arm not in ARMS or not 0 <= layer < 36: raise ValueError("Invalid recipe arm/layer")
    return Path(root)/"arms"/arm/f"layer_{layer:02d}"


def prefix_path(root, arm, layer):
    return Path(root)/"inputs/coordinates.npy" if layer == 0 else train_path(root, arm, layer-1)/"next_coordinates.npy"


def build_block(root, arm, layer, device="cpu", *, selected=False, packed=False):
    from .jspace_qwen_parallel_model import load_native_block
    from .jspace_qwen_affine_block import AffineQwenBlock
    from .jspace_quadratic_qwen import QuadraticQwenBlock
    plan = read(Path(root)/"plan.json"); fit = load(fit_path(root, layer))
    native = load_native_block(plan["model"]["path"], layer, device=device, dtype=torch.bfloat16)
    block = AffineQwenBlock(native, fit["basis"].to(device), fit["means"].to(device),
        trainable=True, parameter_dtype=torch.float32,
        head_projection={k:fit[k].to(device) for k in ("rope_pairs", "value_basis")})
    wrapped = QuadraticQwenBlock(block, mode=arm, learn_metric=True).to(device)
    del native, fit
    if selected: wrapped.load_state_dict(load(train_path(root, arm, layer)/"selected.pt")["block"], strict=True)
    if packed: wrapped.pack(dtype=torch.float32, buffer_dtype=torch.float32).eval().requires_grad_(False)
    return wrapped


def build_readout(root, block, device="cpu", *, selected_arm=None):
    from .jspace_qwen_parallel_model import load_readout_norm
    from .jspace_quadratic_qwen import FoldedReadout
    plan = read(Path(root)/"plan.json")
    embedding, norm, _ = load_readout_norm(plan["model"]["path"], device=device, dtype=torch.bfloat16)
    value = FoldedReadout(embedding, block.basis, block.anchor, norm.weight, norm.variance_epsilon,
                         trainable=True, parameter_dtype=torch.float32)
    if selected_arm:
        value.load_state_dict(load(train_path(root, selected_arm, 35)/"selected.pt")["readout"], strict=True)
    return value.to(device)


def bridge(root, layer, device="cpu"):
    a, b = load(fit_path(root, layer)), load(fit_path(root, layer+1))
    u, v = a["basis"].to(device), b["basis"].to(device)
    ma, mb = a["means"].to(device), b["means"].to(device)
    aa, ab = ma-(ma@u)@u.T, mb-(mb@v)@v.T
    return v.T@u, (aa-ab)@v


def basis(root):
    from .jspace_quadratic_basis import fit_all_layers
    plan, lease, guard = setup(root); source = Path(plan["teacher_root"])/"capture/statistics.pt"
    if sha(source) != plan["input_sha256"][str(source)]: raise ValueError("Changed fixed Jacobian operator")
    directory = Path(root)/"basis"
    fit_all_layers(source, directory, rank=1024, block_width=1024, steps=1, head_dim=64,
        seed=CONFIG["seed"], device="cuda:0", guard=guard)
    return complete(root, directory, list(directory.glob("block_*/*")), lease_id=lease["id"])


def inputs(root):
    plan, lease, guard = setup(root); checked(root, Path(root)/"basis")
    directory = Path(root)/"inputs"; directory.mkdir(exist_ok=False)
    source = Path(plan["teacher_root"])/"capture/boundary_00.npy"
    if sha(source) != plan["input_sha256"][str(source)]: raise ValueError("Changed native embedding bank")
    original = np.load(source, mmap_mode="r"); u = load(fit_path(root, 0))["basis"].cuda()
    output = np.lib.format.open_memmap(directory/"coordinates.npy", mode="w+", dtype=np.float32,
                                     shape=(len(original), CONFIG["rank"]))
    with torch.no_grad():
        for start in range(0, len(original), 1024):
            guard(); x = torch.from_numpy(np.array(original[start:start+1024], copy=True)).view(torch.bfloat16).cuda().float()
            output[start:start+1024] = (x@u).cpu().numpy()
    output.flush(); output._mmap.close(); original._mmap.close()
    return complete(root, directory, [directory/"coordinates.npy"], lease_id=lease["id"])


def dataset_for(root, arm, layer, block, *, coordinates=None):
    from .jspace_quadratic_train import CoordinateDataset
    plan = read(Path(root)/"plan.json"); cache = Path(plan["teacher_root"])/"capture"
    return CoordinateDataset(read(Path(root)/"items.json")["items"], coordinates or prefix_path(root, arm, layer),
        cache/f"boundary_{layer+1:02d}.npy", basis=block.basis, anchor=block.anchor,
        teacher_logits_path=cache/"teacher_logits.npy" if layer == 35 else None)


def train_config():
    from .jspace_quadratic_train import SequentialConfig
    return SequentialConfig(epochs=12, microbatch=4, accumulation=4, learning_rate=1.2e-4,
        temperature=2., gradient_clip=1., seed=CONFIG["seed"], max_cpu_state_bytes=12*1024**3)


@torch.no_grad()
def advance(root, arm, layer, block, dataset, rotary, guard):
    directory = train_path(root, arm, layer)
    device=next(block.parameters()).device
    # Verify the selected checkpoint's folding on development inputs before
    # the deployed form becomes the frozen prefix for the next layer.
    indices = dataset.development[:4]; batch = dataset.batch(indices, device=device)
    cos, sin = rotary(batch["coordinates"], batch["position_ids"])
    args = dict(position_embeddings=(cos,sin), attention_mask=batch["attention_mask"], position_ids=batch["position_ids"])
    reference = block.forward_coordinates(batch["coordinates"], **args)
    block.pack(dtype=torch.float32, buffer_dtype=torch.float32).eval().requires_grad_(False)
    folded = block.forward_coordinates(batch["coordinates"], **args)
    delta = (reference-folded).float()
    parity = dict(max_abs=float(delta.abs().max()), relative_rmse=float(delta.square().mean().sqrt()/reference.float().square().mean().sqrt().clamp_min(1e-12)),
        finite=bool(torch.isfinite(folded).all()), scope="selected FP32 checkpoint before/after metric and offset folding")
    parity["passed"] = parity["finite"] and parity["relative_rmse"] < 1e-4
    write(directory/"fold_parity.json", parity)
    if not parity["passed"]: raise ValueError("FP32 metric folding changed selected outputs beyond numerical tolerance")
    del batch, reference, folded
    if layer == 35: return
    weight, shift = bridge(root, layer, device)
    output = np.lib.format.open_memmap(directory/"next_coordinates.npy", mode="w+", dtype=np.float32,
                                     shape=dataset.coordinates.shape)
    try:
        for start in range(0, len(dataset.items), 4):
            guard(); selected = list(range(start, min(start+4, len(dataset.items))))
            batch = dataset.batch(selected, device=device)
            cos, sin = rotary(batch["coordinates"], batch["position_ids"])
            z = block.forward_coordinates(batch["coordinates"], attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"], position_embeddings=(cos,sin))
            z = F.linear(z, weight, shift[0])
            if not torch.isfinite(z).all(): raise FloatingPointError("Nonfinite frozen student prefix")
            for row, i in enumerate(selected):
                case = dataset.items[i]; output[case["offset"]:case["stop"]] = z[row,:dataset.lengths[i]].cpu().numpy()
        output.flush()
    finally: output._mmap.close()


def train(root, arm, layer):
    from .jspace_quadratic_train import SequentialTrainer
    plan, lease, guard = setup(root)
    checked(root, Path(root)/"pilot")
    checked(root, Path(root)/"inputs" if layer == 0 else train_path(root, arm, layer-1))
    verify_layer_fit(root,layer)
    if layer<35:verify_layer_fit(root,layer+1)
    cache=Path(plan["teacher_root"])/"capture"
    verify_consumed(plan,[cache/f"boundary_{layer+1:02d}.npy"]+([cache/"teacher_logits.npy"] if layer==35 else []))
    block = build_block(root, arm, layer, "cuda:0").train()
    readout = build_readout(root, block, "cuda:0") if layer == 35 else None
    rotary = rotary_module(plan, "cuda:0"); dataset = dataset_for(root, arm, layer, block)
    directory = train_path(root, arm, layer); directory.parent.mkdir(parents=True, exist_ok=True)
    trainer = SequentialTrainer(block, rotary, dataset, readout=readout, label_ids=plan["label_ids"],
        config=train_config(), guard=guard)
    result = trainer.fit(directory, event=lambda kind, **values: event(root, kind, arm=arm, layer=layer, **values))
    del trainer; gc.collect(); torch.cuda.empty_cache()
    advance(root, arm, layer, block, dataset, rotary, guard)
    dataset.close(); guard(); peak = torch.cuda.max_memory_reserved()
    if peak > sum(lease["allocation"].values())*1024**2: raise RuntimeError("Recipe training exceeded reservation")
    return complete(root, directory, list(directory.glob("*")), lease_id=lease["id"], peak_reserved_bytes=peak,
                    arm=arm, layer=layer, selected_epoch=result["selected_epoch"])


def pilot(root):
    from .jspace_quadratic_train import SequentialTrainer
    plan, lease, guard = setup(root); checked(root, Path(root)/"basis")
    directory = Path(root)/"pilot"; directory.mkdir(exist_ok=False)
    block = build_block(root, "quadratic", 35, "cuda:0").train()
    readout = build_readout(root, block, "cuda:0")
    source = np.load(Path(plan["teacher_root"])/"capture/boundary_35.npy", mmap_mode="r")
    coordinates = np.lib.format.open_memmap(directory/"native_input_coordinates.npy", mode="w+", dtype=np.float32,
        shape=(len(source), 1024))
    with torch.no_grad():
        for start in range(0,len(source),1024):
            x = torch.from_numpy(np.array(source[start:start+1024],copy=True)).view(torch.bfloat16).cuda().float()
            coordinates[start:start+1024] = (x@block.basis).cpu().numpy()
    coordinates.flush(); coordinates._mmap.close(); source._mmap.close()
    dataset = dataset_for(root, "quadratic", 35, block, coordinates=directory/"native_input_coordinates.npy")
    trainer = SequentialTrainer(block, rotary_module(plan,"cuda:0"), dataset, readout=readout,
        label_ids=plan["label_ids"], config=train_config(), guard=guard)
    started=time.perf_counter(); dataset.fit_normalizer(device=trainer.device, microbatch=4, guard=guard)
    normalization_seconds=time.perf_counter()-started
    indices=sorted(dataset.training,key=lambda i:dataset.lengths[i],reverse=True)[:16]
    started=time.perf_counter(); result=trainer.train_window(indices); torch.cuda.synchronize()
    update_seconds=time.perf_counter()-started; peak=torch.cuda.max_memory_reserved(); guard()
    if peak>sum(lease["allocation"].values())*1024**2:raise RuntimeError("Final-layer pilot exceeded reservation")
    write(directory/"result.json",dict(scope="systems only: native input, not sequential student or task-quality evidence",
        update=result, normalization_seconds=normalization_seconds, update_seconds=update_seconds,
        longest_lengths=[dataset.lengths[i] for i in indices], peak_reserved_bytes=peak))
    dataset.close()
    return complete(root,directory,list(directory.glob("*")),lease_id=lease["id"],systems_only=True)


def build_model(root, arm):
    from .jspace_qwen_parallel_model import load_readout_norm
    from .jspace_quadratic_runtime import CompactTokenModel
    plan=read(Path(root)/"plan.json"); blocks=[build_block(root,arm,i,selected=True) for i in range(36)]
    readout=build_readout(root,blocks[-1],selected_arm=arm)
    e,_,_=load_readout_norm(plan["model"]["path"],dtype=torch.bfloat16)
    with torch.no_grad():embedding=e.float()@blocks[0].basis
    return CompactTokenModel(embedding,blocks,readout,rotary_module(plan,"cpu"),native_config(plan),capacity=416).eval().requires_grad_(False)


def export(root, arm):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Export is CPU only")
    plan=verify(root)
    for i in range(36):checked(root,train_path(root,arm,i))
    directory=Path(root)/"export"/arm;directory.mkdir(parents=True,exist_ok=False)
    model=build_model(root,arm)
    save(directory/"model.pt",dict(schema=SCHEMA,fingerprint=plan["fingerprint"],arm=arm,model=model))
    from .jspace_quadratic_runtime import resident_tensor_bytes
    size=resident_tensor_bytes(model)
    write(directory/"metadata.json",dict(arm=arm,fingerprint=plan["fingerprint"],tensor_bytes=size,
        model_parameters=sum(p.numel() for p in model.parameters()),
        format="trusted-local PyTorch module; reload uses frozen runtime sources, no teacher weights",
        dtype="float32",native_tensor_bytes_reference_estimate=2*6171877632,
        size_gate_uses="measured FP32 native runtime state, not this estimate",source_sha256=plan["source_sha256"]))
    return complete(root,directory,list(directory.glob("*")),tensor_bytes=size)


def evaluate(root, arm):
    from .jspace_quadratic_evaluate import stream_score
    from .jspace_qwen_parallel_model import load_native_block,load_readout_norm
    plan,lease,guard=setup(root);directory=Path(root)/"evaluation"/arm
    directory.mkdir(parents=True,exist_ok=False)
    if arm not in ("native",*ARMS):raise ValueError("Unknown evaluation arm")
    cases=read(Path(root)/"tasks.json")["splits"]["evaluation"]
    e,norm,_=load_readout_norm(plan["model"]["path"],device="cpu",dtype=torch.bfloat16)
    if arm=="native":
        embedding=e.float();head=(norm.float(),embedding)
        loader=lambda layer:load_native_block(plan["model"]["path"],layer,device="cuda:0",dtype=torch.bfloat16).float()
        bridges=None
    else:
        for layer in range(36):checked(root,train_path(root,arm,layer))
        u=load(fit_path(root,0))["basis"]
        with torch.no_grad():embedding=e.float()@u
        last=build_block(root,arm,35,selected=True)
        head=build_readout(root,last,selected_arm=arm).eval().requires_grad_(False)
        del last
        loader=lambda layer:build_block(root,arm,layer,"cuda:0",selected=True,packed=True)
        bridges=lambda layer:bridge(root,layer,"cuda:0")
    del e;gc.collect()
    value=stream_score(cases,input_embedding=embedding,block_loader=loader,final_readout=head,
        label_ids=plan["label_ids"],scratch_dir=directory/"ephemeral_banks",rotary=rotary_module(plan,"cuda:0"),
        layer_count=36,bridge_loader=bridges,device="cuda:0",guard=guard,microbatch=4,
        progress=lambda **kw:event(root,"evaluation_progress",arm=arm,**{"detail":kw}))
    value.update(arm=arm,fingerprint=plan["fingerprint"],dtype="float32",
        teacher_rounding="Native weights rounded to cached-teacher BF16 first; all evaluation arithmetic/activations FP32")
    write(directory/"result.json",value)
    return complete(root,directory,[directory/"result.json"],lease_id=lease["id"],cases=len(cases),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())


def cpu_benchmark(root, arm):
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("CPU benchmark requires hidden CUDA")
    from transformers import AutoModelForCausalLM
    from .jspace_quadratic_runtime import NativeTokenModel,benchmark_cpu
    plan=verify(root);checked(root,Path(root)/"export"/arm)
    if not any((Path(plan["prior_completion"])/name).exists() for name in ("completed.json","failed.json")):
        raise RuntimeError("Wait for prior campaign before isolated host timing")
    torch.set_num_threads(1)
    directory=Path(root)/"cpu_benchmark"/arm;directory.mkdir(parents=True,exist_ok=False)
    # Trusted local export checked above; no original fit/teacher constructor is
    # called while deserializing the compact model.
    exported=torch.load(Path(root)/"export"/arm/"model.pt",map_location="cpu",weights_only=False)
    if exported["fingerprint"]!=plan["fingerprint"] or exported["arm"]!=arm:raise ValueError("Wrong compact export")
    compact=exported["model"].eval().requires_grad_(False)
    native_source=AutoModelForCausalLM.from_pretrained(plan["model"]["path"],local_files_only=True,
        trust_remote_code=False,torch_dtype=torch.bfloat16,attn_implementation="eager").float().eval().requires_grad_(False)
    native=NativeTokenModel(native_source,capacity=416);del native_source
    prompts={};identities={}
    development=read(Path(root)/"tasks.json")["splits"]["development"]
    for length in CONFIG["cpu_lengths"]:
        selected=[case for case in development if len(case["input_ids"])>=length][:2]
        if len(selected)!=2:raise ValueError("Insufficient development prompts for fixed CPU shape")
        prompts[length]=[torch.tensor([case["input_ids"][:length]],dtype=torch.long) for case in selected]
        identities[str(length)]=[dict(identity=case["identity"],original_tokens=len(case["input_ids"]),timed_prefix_tokens=length) for case in selected]
    result=benchmark_cpu(native,compact,prompts,repeats=5,warmup=2,
        progress=lambda value:event(root,"cpu_benchmark_progress",arm=arm,detail=value))
    result.update(arm=arm,fingerprint=plan["fingerprint"],prompt_identities=identities,
        quality_scope="Primary FP32 held-out scores are separate; this is warm full-model prefill/last-logit latency, not autoregressive decode tok/s")
    write(directory/"result.json",result)
    return complete(root,directory,[directory/"result.json"],arm=arm,
                    compiled_speedup=result["compiled_speedup"],greedy_parity=result["all_compiled_greedy_equal"])


def aggregate(root):
    from .jspace_quadratic_evaluate import paired_summary
    root=Path(root);plan=verify(root)
    for layer in range(36):
        for arm in ARMS:checked(root,train_path(root,arm,layer))
    for arm in ("native",*ARMS):checked(root,root/"evaluation"/arm)
    for arm in ARMS:
        checked(root,root/"export"/arm);checked(root,root/"cpu_benchmark"/arm)
    # Recheck the immutable original cache/checkpoint at finalization, including
    # every consumed bank, after quick identity guards during individual stages.
    verify_consumed(plan,plan["input_sha256"])
    for name,expected in plan["model"]["files"].items():
        if sha(Path(plan["model"]["path"])/name)!=expected:raise ValueError("Native teacher checkpoint changed")
    rows={arm:read(root/"evaluation"/arm/"result.json")["records"] for arm in ("native",*ARMS)}
    controls={arm:paired_summary(rows["native"],rows[arm]) for arm in ARMS}
    incremental=paired_summary(rows["softmax"],rows["quadratic"])
    timing={arm:read(root/"cpu_benchmark"/arm/"result.json") for arm in ARMS}
    summary={}
    for arm in ARMS:
        size=timing[arm]["resident_tensor_bytes"]
        reduction=1-size["compact"]/size["native"]
        rates=timing[arm]["compiled_speedup"]
        gates=dict(aggregate_accuracy=controls[arm]["aggregate_quality_pass"],tensor_reduction=reduction>=.38,
            cpu_prefill_speed=min(rates.values())>=1.4,compiled_greedy_parity=timing[arm]["all_compiled_greedy_equal"])
        summary[arm]=dict(quality=controls[arm]["quality"],tensor_reduction=reduction,tensor_bytes=size,
            cpu_compiled_prefill_speedups=rates,gates=gates,targets_met=all(gates.values()))
    result=dict(schema=SCHEMA,study_id=STUDY_ID,fingerprint=plan["fingerprint"],status="completed",confirmed=False,
        targets_met=summary["quadratic"]["targets_met"],arms=summary,
        quadratic_vs_softmax=incremental["quality"],scope=plan["scope"],
        speed_gate="Minimum matched FP32 one-thread compiled CPU prefill speedup across128 and256token shapes",
        training="36 sequential frozen-prefix stages per arm, fixed original teacher,12epochs each; finalcompactreadout trained",
        limitations="Shared512-case restricted benchmark; independent training replication and long-context/summarization quality unmeasured; no CPU latency relabeled as decode tok/s")
    write(root/"campaign_results.json",result)
    report=ROOT/"docs/08 - Analysis/jspace/2026-11-14_jspace-quadratic-recipe.md"
    lines=["# Qwen sequential quadratic-attention recipe","",f"Study: {STUDY_ID}. Completed exploratory comparison; confirmed=false.","",
        f"Artifact: `{root}`. Fingerprint: `{plan['fingerprint']}`.","",
        "| Model | Accuracy | Loss vs native (pp) | Tensor reduction | CPU128 speedup | CPU256 speedup |",
        "|---|---:|---:|---:|---:|---:|"]
    native_correct=sum(r["prediction"]==r["answer"] for r in rows["native"])
    lines.append(f"| Native FP32 | {native_correct}/512 ({native_correct/512:.2%}) | 0 | 0 | 1.00x | 1.00x |")
    for arm,value in summary.items():
        q=value["quality"]["overall"];s=value["cpu_compiled_prefill_speedups"]
        lines.append(f"| {arm} | {q['compact_correct']}/512 ({q['compact_accuracy']:.2%}) | {q['accuracy_drop']*100:.3f} | {value['tensor_reduction']:.2%} | {s['128']:.3f}x | {s['256']:.3f}x |")
    lines += ["",f"Quadratic joint targets met: **{result['targets_met']}**.","",
        "Both compressed arms use identical native-task Jacobian Krylov geometry, full11,008-channel MLPs, head reduction, folded learned rotary-pair metrics, compact endpoints, and sequential distillation. Their attention normalization differs.","",
        "The original teacher targets were captured in BF16. Quality evaluation uses FP32 arithmetic from the same BF16-rounded native weights. CPU timing uses static complete-model FP32 execution, batch1, one thread, matched last-query optimization and full-graph Inductor with freezing. Compilation/warmup are excluded and recorded separately. This is prefill latency, not decoding throughput.","",
        "Per-family paired scores, both-arm training histories, compiler FX graphs, all timing samples and numerical parity diagnostics remain in the artifact root.","",result["limitations"],""]
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open("x") as stream:stream.write("\n".join(lines))
    write(root/"report_receipt.json",dict(report=str(report),sha256=sha(report),campaign_sha256=sha(root/"campaign_results.json")))
    return result


def protect_prior_timing(root):
    prior=read(Path(root)/"plan.json")["prior_completion"]
    directory=Path(prior)
    while not any((directory/name).exists() for name in ("completed.json","failed.json")):
        trace=TASK_SOURCE/"coordinator.jsonl"
        events=[json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        if not any(row.get("kind")=="phase_start" and
            (row.get("stage")=="evaluate" or (row.get("stage")=="train" and row.get("group")==5)) for row in events):return
        event(root,"yielding_to_prior_final_training_and_timing",prior=str(directory));time.sleep(30)


def run_phase(root, stage, *, arm=None, layer=None, gpu=True):
    protect_prior_timing(root)
    directory=Path(root)/"logs";directory.mkdir(exist_ok=True)
    name=f"{stage}_{arm}_{layer}";path=directory/(name+".log")
    if path.exists():raise FileExistsError("Inspect the previous attempt before any retry: "+str(path))
    command=["pixi","run","--locked","--offline","-e","ncm","python","-m",MODULE,stage,"--root",str(root)]
    if arm is not None:command += ["--arm",arm]
    if layer is not None:command += ["--layer",str(layer)]
    if gpu:
        command=["/usr/bin/python3","/usr/local/bin/inference-ctl","lease","run","--allocation",CONFIG["gpu_allocation"],
            "--minutes","120","--reason",f"{STUDY_ID} {stage} {arm} layer={layer}","--"]+command
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"","PYTHONUNBUFFERED":"1","OMP_NUM_THREADS":"4" if gpu else "1",
        "MKL_NUM_THREADS":"4" if gpu else "1","OPENBLAS_NUM_THREADS":"4" if gpu else "1",
        "PIXI_CACHE_DIR":"/tmp/structure-net-pixi-cache","UV_CACHE_DIR":"/tmp/structure-net-uv-cache",
        "TRITON_CACHE_DIR":"/tmp/structure-net-triton-cache"}
    event(root,"phase_start",stage=stage,arm=arm,layer=layer,command=command)
    with path.open("x") as stream:result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    event(root,"phase_end",stage=stage,arm=arm,layer=layer,returncode=result.returncode)
    if result.returncode:raise RuntimeError(f"{name} failed; inspect {path}")


def run(root):
    root=Path(root);verify(root)
    if os.environ.get("IC_LEASE"):raise ValueError("Coordinator must not hold a parent GPU lease")
    for stage in ("basis","inputs","pilot"):
        if (root/stage/"manifest.json").exists():checked(root,root/stage)
        else:run_phase(root,stage)
    for layer in range(36):
        for arm in ARMS:
            path=train_path(root,arm,layer)
            if (path/"manifest.json").exists():checked(root,path)
            else:run_phase(root,"train",arm=arm,layer=layer)
    for arm in ("native",*ARMS):
        run_phase(root,"evaluate",arm=arm)
    for arm in ARMS:run_phase(root,"export",arm=arm,gpu=False)
    # GPU jobs may coexist under leases; final host timings require the prior
    # campaign's trainer and large-hash closeout to have left the machine.
    waiting=False
    while not any((PRIOR_COMPLETION/name).exists() for name in ("completed.json","failed.json")):
        if not waiting:event(root,"waiting_for_prior_timing_isolation",path=str(PRIOR_COMPLETION));waiting=True
        time.sleep(30)
    for arm in ARMS:run_phase(root,"cpu_benchmark",arm=arm,gpu=False)
    result=aggregate(root)
    for stage in ("analyze","store","readback","closeout"):
        command=["pixi","run","--locked","--offline","python","-m",
            "experiments.neural_architecture_lab.finalize_jspace_quadratic",stage,"--root",str(root)]
        event(root,"closeout_start",stage=stage,command=command)
        env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
        with (root/"logs"/("closeout_"+stage+".log")).open("x") as stream:
            proc=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        event(root,"closeout_end",stage=stage,returncode=proc.returncode)
        if proc.returncode:raise RuntimeError("Quadratic closeout failed: "+stage)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","basis","inputs","pilot","train","evaluate","export","cpu_benchmark","aggregate","run"))
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--arm",choices=("native",*ARMS));parser.add_argument("--layer",type=int)
    args=parser.parse_args()
    if args.stage=="train":result=train(args.root,args.arm,args.layer)
    elif args.stage in ("evaluate","export","cpu_benchmark"):result=globals()[args.stage](args.root,args.arm)
    else:result=globals()[args.stage](args.root)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=="__main__":main()
