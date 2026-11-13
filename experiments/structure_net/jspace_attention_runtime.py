"""Attention-output J-lens fitting, compact edits, and contiguous residual traces."""
from contextlib import contextmanager

import torch

from .jspace_lens import load_small_qwen
from .jspace_relation_capture import _prompt
from .jspace_runtime import generate_answer
from .jspace_tasks import normalize_answer, orthonormal_basis


def residual_nodes(start, stop=35):
    return [f"{layer}.{kind}" for layer in range(start, stop+1) for kind in ("residual", "block")]


@contextmanager
def boundaries(model, start, callback, *, graph=False):
    """Capture attention writes, post-addition/pre-MLP residuals and block outputs."""
    handles = []
    def hook_for(key, root=False):
        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            if root and graph:
                h.requires_grad_(True)
            replacement = callback(key, h)
            if replacement is not None:
                return replacement if torch.is_tensor(output) else (replacement, *output[1:])
        return hook
    def pre_for(key):
        def pre(module, inputs):
            callback(key, inputs[0])
        return pre
    try:
        for layer in range(start, len(model.model.layers)):
            block = model.model.layers[layer]
            handles.append(block.self_attn.register_forward_hook(hook_for(f"{layer}.attention", layer == start)))
            handles.append(block.post_attention_layernorm.register_forward_pre_hook(pre_for(f"{layer}.residual")))
            handles.append(block.register_forward_hook(hook_for(f"{layer}.block")))
        yield
    finally:
        for handle in handles:
            handle.remove()


def fit_dictionary(model, tokenizer, prompts, token_ids, start=11, batch=2, progress=None):
    """Selected rows of the global positional mean J, at actual attention boundaries.

    Sum target cotangents over positions 1..T-2, average the same source positions,
    then average 64 generic contexts. Target is the final pre-norm block output.
    This is a compact selected-token bank, not the paper's full sparse J-space.
    """
    model.eval().requires_grad_(False)
    sums = {}
    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)["input_ids"].to(model.device)
        if ids.shape[1] < 8:
            raise ValueError("Calibration text too short")
        states = {}
        def capture(key, h):
            states[key] = h
        with torch.enable_grad(), boundaries(model, start, capture, graph=True):
            model.model(input_ids=ids.expand(batch, -1), use_cache=False)
            target = states[f"{len(model.model.layers)-1}.block"]
            keys = list(states)
            for offset in range(0, len(token_ids), batch):
                subset = token_ids[offset:offset+batch]
                cotangent = torch.zeros_like(target)
                cotangent[:len(subset), 1:-1] = model.lm_head.weight[subset].detach()[:, None].to(target)
                grads = torch.autograd.grad(target, [states[k] for k in keys], grad_outputs=cotangent,
                                            retain_graph=offset+batch < len(token_ids))
                for key, gradient in zip(keys, grads):
                    rows = gradient[:len(subset), 1:-1].float().mean(1).cpu()
                    if not torch.isfinite(rows).all():
                        raise FloatingPointError("Nonfinite attention Jacobian")
                    if key not in sums:
                        sums[key] = torch.zeros(len(token_ids), rows.shape[-1])
                    sums[key][offset:offset+len(subset)] += rows
                del grads, cotangent
        del states, target
        if progress:
            progress("dictionary", {"completed": pi+1, "total": len(prompts)})
    raw = {k: v/len(prompts) for k, v in sums.items()}
    normalized = {}
    for key, value in raw.items():
        norms = value.norm(dim=1)
        if torch.any(norms < 1e-12):
            raise ValueError("Zero dictionary row")
        normalized[key] = value/norms[:, None]
    return {"raw": raw, "rows": normalized}


def pooled(h, start):
    if h.ndim != 2 or not 0 <= start < len(h)-1:
        raise ValueError("Expected a sequence and nonempty pre-final question span")
    return torch.stack((h[start:-1].float().mean(0), h[-1].float()))


def compact_delta(source, donor, bank, method, source_country, target_country, *, seed=557):
    """Whole-prefill attention edit. Rank and actual coefficients stay explicit."""
    source, donor, bank = source.float(), donor.float(), bank.float()
    if source.shape != donor.shape or source.shape[-1] != bank.shape[1]:
        raise ValueError("Attention donor/coordinate shape mismatch")
    q = orthonormal_basis(bank)
    difference = donor-source
    projected = (difference@q.T)@q
    if method == "dense":
        return difference
    if method == "remainder":
        return difference-projected
    base = method.removeprefix("random_")
    if base == "projected_donor":
        delta = projected
    elif base == "concept_swap":
        ids = [source_country*2, source_country*2+1, target_country*2, target_country*2+1]
        pair = bank[ids]
        if len(orthonormal_basis(pair)) != 4:
            raise ValueError("Lexical concept directions not independent")
        coefficients = source@torch.linalg.pinv(pair, rtol=1e-6)
        delta = (coefficients[:, [2, 3, 0, 1]]-coefficients)@pair
    else:
        raise ValueError(method)
    if method.startswith("random_"):
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(delta.shape[-1], generator=generator)
        signs = (2*torch.randint(2, (delta.shape[-1],), generator=generator)-1).to(delta)
        delta = delta[:, permutation]*signs
    return delta


def forward(model, tokenizer, case, question_start, dictionary, *, observe_start=11, keep_attention=(),
            patch_layer=None, donor=None, method=None):
    observed, raw, diagnostics = {}, {}, {}
    device_rows = {k: v.to(model.device) for k, v in dictionary.items()}
    def capture(key, h):
        if h.shape[1] == 1:  # Hooks must not edit cached continuation tokens.
            return None
        work = h
        layer, kind = key.split(".")
        if kind == "attention" and int(layer) == patch_layer:
            before = h[0].detach().float().cpu()
            from .jspace_relation_tasks import COUNTRIES
            delta = compact_delta(before, donor, dictionary[key], method,
                                  COUNTRIES.index(case["intermediate"]), COUNTRIES.index(case["swap_to"]), seed=557+int(layer))
            work = (before+delta).to(h)[None]
            actual = work[0].float().cpu()-before
            q = orthonormal_basis(dictionary[key])
            diagnostics.update({"requested_l2": float(delta.norm()), "actual_l2": float(actual.norm()),
                                "relative_l2": float(actual.norm()/before.norm().clamp_min(1e-12)),
                                "changed_positions": int((actual.norm(dim=-1)>0).sum()), "rank": len(q),
                                "rounding_relative_error": float((actual-delta).norm()/delta.norm().clamp_min(1e-12)),
                                "actual_outside_J_relative_norm": float((actual-(actual@q.T)@q).norm()/actual.norm().clamp_min(1e-12))})
        if not torch.isfinite(work).all():
            raise FloatingPointError("Nonfinite traced activation")
        observed[key] = (pooled(work[0], question_start)@device_rows[key].T).detach().cpu()
        if kind == "attention" and int(layer) in keep_attention:
            raw[int(layer)] = work[0].detach().cpu().clone()
        return work if work is not h else None
    with boundaries(model, observe_start, capture):
        result = generate_answer(model, tokenizer, _prompt(tokenizer, case), max_new_tokens=8)
    row = {k: v for k, v in result.items() if k != "observed"}
    row.update({"case_id": case["case_id"], "expected_answer": case["answer"], "expected_target": case["swap_answer"],
                "original_correct": normalize_answer(row["answer"] or "") == normalize_answer(case["answer"]),
                "target_correct": normalize_answer(row["answer"] or "") == normalize_answer(case["swap_answer"]),
                "patch": diagnostics})
    return {"row": row, "scores": observed, "attention": raw}
