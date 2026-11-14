"""Read-only development oracles separating projection, handoff and recovery errors.

The native oracle uses cached BF16 hidden states and FP32 final arithmetic, not
a fresh full-FP32 native-model pass. C-B includes accumulated prefix error as
well as local recovery error; its energy is not additive with B-A unless their
cross term vanishes. No diagnostic reads the held-out evaluation population.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import jspace_quadratic_recipe as recipe

DEFAULT_LAYERS = (0, 1, 2, 10, 20, 29, 30, 34, 35)
SCHEMA = "jspace.accuracy-oracles.v1"


def select_cases(items, split):
    if split not in ("train", "development"):
        raise ValueError("Only train/development diagnostics are allowed")
    cases = [c for c in items if c["split"] == split]
    if not cases or len({c["identity"] for c in cases}) != len(cases):
        raise ValueError("Nonempty distinct diagnostic identities required")
    for case in cases:
        if (case["stop"] - case["offset"] != len(case["input_ids"])
                or case["offset"] < 0 or not case["input_ids"]
                or not 0 <= case["answer"] < len(case["choices"])):
            raise ValueError("Invalid case bank offsets or answer")
    return cases


def create_output(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    return root


def affine_anchor(basis, means):
    if means.shape != (1, basis.shape[0]):
        raise ValueError("This diagnostic requires the recipe's global anchor")
    return means - (means @ basis) @ basis.T


def handoff_coordinates(hidden, basis, anchor, next_basis, next_anchor, *, bridge=None):
    """Return direct-next A and projected-current-then-bridged B."""
    a = (hidden - next_anchor) @ next_basis
    z = (hidden - anchor) @ basis
    weight, shift = bridge if bridge is not None else (next_basis.T @ basis, ((anchor-next_anchor) @ next_basis)[0])
    b = F.linear(z, weight, shift)
    return a, b


def error_sums(a, b, c):
    if a.shape != b.shape or a.shape != c.shape or a.ndim != 2:
        raise ValueError("A/B/C must have matching [positions, rank] shapes")
    if not all(torch.isfinite(t).all() for t in (a, b, c)):
        raise FloatingPointError("Nonfinite oracle coordinates")
    floor, learned, total = (b-a).double(), (c-b).double(), (c-a).double()
    return dict(positions=a.shape[0], elements=a.numel(), target_energy=float(a.double().square().sum()),
        handoff_squared=float(floor.square().sum()), student_squared=float(learned.square().sum()),
        total_squared=float(total.square().sum()), cross_twice=float(2*(floor*learned).sum()))


def finish_errors(sums):
    result = dict(sums)
    energy = sums["target_energy"]
    for name in ("handoff", "student", "total"):
        ratio = sums[name+"_squared"] / energy if energy > 0 else None
        result[name+"_normalized_mse"] = ratio
        result[name+"_relative_rms"] = math.sqrt(ratio) if ratio is not None else None
    result["energy_identity_residual"] = (sums["total_squared"]-sums["handoff_squared"]
        -sums["student_squared"]-sums["cross_twice"])
    return result


def score_records(cases, logits):
    if logits.ndim != 2 or logits.shape[0] != len(cases) or not torch.isfinite(logits).all():
        raise ValueError("Finite case-aligned label logits required")
    records = []
    for case, row in zip(cases, logits.detach().float().cpu()):
        count = len(case["choices"])
        if count > row.numel():
            raise ValueError("Readout has fewer label rows than choices")
        records.append(dict(identity=case["identity"], family=case["family"], answer=case["answer"],
            prediction=int(row[:count].argmax()), logits=row[:count].tolist()))
    return records


def summarize(records):
    def counts(rows):
        correct = sum(r["prediction"] == r["answer"] for r in rows)
        return dict(correct=correct, total=len(rows), accuracy=correct/len(rows))
    return dict(**counts(records), by_family={f:counts([r for r in records if r["family"] == f])
        for f in sorted({r["family"] for r in records})})


def _close(bank):
    if bank is not None:
        bank._mmap.close()


def _native_rows(bank, rows, device):
    return torch.from_numpy(np.array(bank[rows], copy=True)).view(torch.bfloat16).to(device, torch.float32)


def _coordinates_batch(bank, cases, device):
    lengths = [c["stop"]-c["offset"] for c in cases]
    storage = np.zeros((len(cases), max(lengths), bank.shape[1]), dtype=np.float32)
    for i, case in enumerate(cases):
        storage[i, :lengths[i]] = bank[case["offset"]:case["stop"]]
    z = torch.from_numpy(storage).to(device)
    positions = torch.arange(max(lengths), device=device).expand(len(cases), -1)
    valid = positions < torch.tensor(lengths, device=device)[:, None]
    mask = ((positions[:, :, None] >= positions[:, None, :])
            & valid[:, :, None] & valid[:, None, :])[:, None]
    return z, positions, mask, torch.tensor(lengths, device=device)-1


def _consume(path, expected, consumed):
    """Hash each consumed artifact once, without traversing unrelated caches."""
    path = Path(path).resolve()
    if str(path) not in consumed:
        actual = recipe.sha(path)
        if actual != expected:
            raise ValueError("Changed consumed source artifact: "+str(path))
        consumed[str(path)] = dict(sha256=actual, identity=recipe.file_identity(path))


def _completed(source, directory, fingerprint):
    value = recipe.read(Path(directory)/"manifest.json")
    if value["status"] != "completed" or value["fingerprint"] != fingerprint:
        raise ValueError("Incomplete or mismatched source stage: "+str(directory))
    return value


@torch.no_grad()
def final_oracles(source, plan, cases, device, guard, consume_stage, consume_teacher, progress):
    from .jspace_qwen_parallel_model import load_readout_norm
    from .jspace_quadratic_qwen import FoldedReadout
    source = Path(source)
    consume_stage(source/"basis", recipe.fit_path(source, 35))
    fit = recipe.load(recipe.fit_path(source, 35))
    u, means = fit["basis"].to(device), fit["means"].to(device)
    anchor = affine_anchor(u, means)
    embedding, norm, _ = load_readout_norm(plan["model"]["path"], device="cpu", dtype=torch.bfloat16)
    labels = torch.tensor(plan["label_ids"])
    weight = embedding.index_select(0, labels).to(device, torch.float32)
    gain = norm.weight.detach().to(device, torch.float32)
    eps = norm.variance_epsilon
    del embedding, norm
    frozen = FoldedReadout(weight, u, anchor, gain, eps, trainable=False).eval()
    teacher_path = Path(plan["teacher_root"])/"capture/boundary_36.npy"
    consume_teacher(teacher_path)
    native_bank = np.load(teacher_path, mmap_mode="r", allow_pickle=False)
    if native_bank.dtype != np.int16 or native_bank.shape[1] != u.shape[0]:
        raise ValueError("Invalid BF16 final-boundary bank")
    try:
        native = _native_rows(native_bank, [c["stop"]-1 for c in cases], device)
    finally:
        _close(native_bank)
    oracle = (native-anchor) @ u
    native_logits = F.linear(native*torch.rsqrt(native.square().mean(-1, keepdim=True)+eps)*gain, weight)
    records = {"native_cached":score_records(cases, native_logits),
               "projected_native_frozen_head":score_records(cases, frozen(oracle))}
    results = {k:summarize(v) for k,v in records.items()}
    for arm in recipe.ARMS:
        guard()
        checkpoint = recipe.train_path(source, arm, 35)/"selected.pt"
        consume_stage(checkpoint.parent, checkpoint)
        state = recipe.load(checkpoint)
        trained = deepcopy(frozen)
        head_state = dict(state["readout"])
        head_state["weight"] = head_state["weight"].index_select(0, labels)
        head_state["bias"] = head_state["bias"].index_select(0, labels)
        trained.load_state_dict(head_state, strict=True)
        records[arm+"_oracle_trained_head"] = score_records(cases, trained(oracle))
        block = recipe.build_block(source, arm, 35, device=device, selected=False)
        block.load_state_dict(state["block"], strict=True)
        block.pack(dtype=torch.float32, buffer_dtype=torch.float32).eval().requires_grad_(False)
        del state, head_state
        prefix = recipe.prefix_path(source, arm, 35)
        consume_stage(prefix.parent, prefix)
        bank = np.load(prefix, mmap_mode="r", allow_pickle=False)
        if bank.dtype != np.float32 or bank.shape[1] != u.shape[1]:
            raise ValueError("Invalid final student-prefix bank")
        outputs = []
        rotary = recipe.rotary_module(plan, device)
        try:
            for start in range(0, len(cases), 4):
                guard()
                batch = cases[start:start+4]
                z, positions, mask, last = _coordinates_batch(bank, batch, device)
                output = block.forward_coordinates(z, attention_mask=mask, position_ids=positions,
                    position_embeddings=rotary(z, positions))
                outputs.append(output[torch.arange(len(batch), device=device), last].detach())
        finally:
            _close(bank)
        student = torch.cat(outputs)
        records[arm+"_student_frozen_head"] = score_records(cases, frozen(student))
        records[arm+"_student_trained_head"] = score_records(cases, trained(student))
        results[arm+"_final_coordinate_error"] = finish_errors(error_sums(oracle, oracle, student))
        for key in (arm+"_oracle_trained_head", arm+"_student_frozen_head", arm+"_student_trained_head"):
            results[key] = summarize(records[key])
        progress("final_arm", arm=arm, scores={k:v for k,v in results.items() if k.startswith(arm)})
        del block, trained, rotary, outputs, student, output, z
        gc.collect()
        torch.cuda.empty_cache()
    return dict(scores=results, records=records,
        native_scope="Cached BF16 final hidden; original BF16-rounded norm/readout evaluated in FP32")


@torch.no_grad()
def handoff_oracles(source, plan, cases, layers, device, guard, consume_stage, consume_teacher, progress):
    results = {}
    for layer in layers:
        if layer == 35:
            results[str(layer)] = dict(skipped="Terminal boundary has no next basis; covered by final oracle")
            continue
        guard()
        fits = []
        for index in (layer, layer+1):
            path = recipe.fit_path(source, index)
            consume_stage(Path(source)/"basis", path)
            fits.append(recipe.load(path))
        u, v = [f["basis"].to(device) for f in fits]
        a, b = [affine_anchor(w, f["means"].to(device)) for w,f in zip((u,v), fits)]
        fixed_bridge = (v.T @ u, ((a-b) @ v)[0])
        native_path = Path(plan["teacher_root"])/f"capture/boundary_{layer+1:02d}.npy"
        consume_teacher(native_path)
        native = np.load(native_path, mmap_mode="r", allow_pickle=False)
        banks = {}
        try:
            for arm in recipe.ARMS:
                path = recipe.train_path(source, arm, layer)/"next_coordinates.npy"
                consume_stage(path.parent, path)
                banks[arm] = np.load(path, mmap_mode="r", allow_pickle=False)
            if (native.dtype != np.int16 or native.shape[1] != u.shape[0]
                    or any(c.dtype != np.float32 or c.shape != (native.shape[0], v.shape[1]) for c in banks.values())):
                raise ValueError("Handoff token bank shapes/dtypes disagree")
            totals = {arm:{scope:{} for scope in ("all_valid", "answer_positions")} for arm in recipe.ARMS}
            for start in range(0, len(cases), 4):
                guard()
                batch = cases[start:start+4]
                rows = np.concatenate([np.arange(c["offset"], c["stop"]) for c in batch])
                last = torch.tensor(np.cumsum([c["stop"]-c["offset"] for c in batch])-1, device=device)
                hidden = _native_rows(native, rows, device)
                direct, bridged = handoff_coordinates(hidden, u, a, v, b, bridge=fixed_bridge)
                for arm, bank in banks.items():
                    student = torch.from_numpy(np.array(bank[rows], copy=True)).to(device)
                    for scope, values in (("all_valid", (direct, bridged, student)),
                            ("answer_positions", (direct[last], bridged[last], student[last]))):
                        for key, value in error_sums(*values).items():
                            totals[arm][scope][key] = totals[arm][scope].get(key, 0)+value
            results[str(layer)] = {arm:{scope:finish_errors(s) for scope,s in scopes.items()} for arm,scopes in totals.items()}
            progress("handoff", layer=layer, metrics=results[str(layer)])
        finally:
            _close(native)
            for bank in banks.values(): _close(bank)
    return results


def run(root, source, *, layers=DEFAULT_LAYERS, split="development"):
    # Reject unsafe scope/output before reservation or CUDA initialization.
    if split not in ("train", "development"):
        raise ValueError("Only train/development diagnostics are allowed")
    if not layers or len(set(layers)) != len(layers) or any(type(i) is not int or not 0 <= i < 36 for i in layers):
        raise ValueError("Distinct layer indices in [0,35] required")
    if Path(root).exists(): raise FileExistsError(root)
    source = Path(source).resolve()
    plan = recipe.verify(source)
    cases = select_cases(recipe.read(source/"items.json")["items"], split)
    root = create_output(root)
    recipe.write(root/"plan.json", dict(schema=SCHEMA, source=str(source), source_fingerprint=plan["fingerprint"],
        source_config=plan["config"], split=split, layers=list(layers), identities=[c["identity"] for c in cases],
        source_plan_sha256=recipe.sha(source/"plan.json"), source_items_sha256=recipe.sha(source/"items.json"),
        producer_sha256=recipe.sha(__file__), dtype="float32", microbatch=4,
        interpretation="B-A is oracle handoff error; C-B includes local recovery and accumulated prefix drift"))
    from .jspace_full_lease import wait_ready
    lease = wait_ready([0])
    guard = lambda: wait_ready([0])
    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    consumed, manifests = {}, {}
    def consume_stage(directory, path):
        key = str(directory)
        if key not in manifests:
            manifests[key] = _completed(source, directory, plan["fingerprint"])
        relative = str(Path(path).relative_to(source))
        _consume(path, manifests[key]["artifacts"][relative], consumed)
    def consume_teacher(path):
        _consume(path, plan["input_sha256"][str(path)], consumed)
    progress = lambda kind, **values: recipe.event(root, kind, **values)
    started = time.monotonic()
    final = final_oracles(source, plan, cases, device, guard, consume_stage, consume_teacher, progress)
    recipe.write(root/"final_oracles.json", final)
    handoffs = handoff_oracles(source, plan, cases, layers, device, guard, consume_stage, consume_teacher, progress)
    guard()
    for path, entry in consumed.items():
        if recipe.file_identity(path) != entry["identity"]:
            raise ValueError("Source artifact changed during diagnostic: "+path)
    result = dict(schema=SCHEMA, status="completed", split=split, cases=len(cases), source_fingerprint=plan["fingerprint"],
        final_scores=final["scores"], handoffs=handoffs, elapsed_seconds=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    recipe.write(root/"result.json", result)
    recipe.write(root/"manifest.json", dict(status="completed", source_fingerprint=plan["fingerprint"],
        consumed=consumed, artifacts={p.name:recipe.sha(p) for p in (root/"plan.json",root/"final_oracles.json",root/"result.json")},
        lease_id=lease["id"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development"), default="development")
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), allow_nan=False), flush=True)


if __name__ == "__main__": main()
