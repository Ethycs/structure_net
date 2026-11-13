"""Position-aware Jacobian dictionaries, donor-component edits and tiny heads."""
from __future__ import annotations

import torch

from .jspace_lens import _record


def bounds(length, scope, suffix_tokens=21):
    if length < suffix_tokens or scope not in ("question", "final"):
        raise ValueError("Invalid aligned suffix or scope")
    return (length-suffix_tokens, length-1) if scope == "question" else (length-1, length)


def flatten_state(state, scope, suffix_tokens=21):
    first, stop = bounds(len(state), scope, suffix_tokens)
    return state[first:stop].flatten().clone()


def task_dictionary(model, tokenizer, prompts, token_ids, sites, *, suffix_tokens=21, batch=2, progress=None):
    """Mean per-position VJP of final pre-norm h against selected W_U rows.

    Unlike a global Jacobian lens, source positions are retained separately and
    the only target is the last prompt position. Calibration uses training only.
    """
    if not prompts or not token_ids or len(set(token_ids)) != len(token_ids):
        raise ValueError("Nonempty prompts and distinct vocabulary rows required")
    decoder = model.model
    device = model.get_input_embeddings().weight.device
    layers = sorted(sites)
    last = len(decoder.layers)-1
    if min(layers) < 0 or max(layers) >= last:
        raise ValueError("Source sites must precede the final decoder block")
    totals = {}
    model.eval().requires_grad_(False)
    for pi, prompt in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        for scope in sites.values():
            bounds(ids.shape[1], scope, suffix_tokens)
        with torch.enable_grad(), _record(decoder.layers, [*layers, last], min(layers)) as captured:
            decoder(input_ids=ids.expand(batch, -1), use_cache=False)
            target = captured[last]
            sources = [captured[l] for l in layers]
            for offset in range(0, len(token_ids), batch):
                subset = token_ids[offset:offset+batch]
                cotangent = torch.zeros_like(target)
                cotangent[:len(subset), -1] = model.get_output_embeddings().weight[subset].detach().to(target)
                gradients = torch.autograd.grad(target, sources, grad_outputs=cotangent,
                                                 retain_graph=offset+batch < len(token_ids))
                for layer, gradient in zip(layers, gradients):
                    first, stop = bounds(ids.shape[1], sites[layer], suffix_tokens)
                    rows = gradient[:len(subset), first:stop].reshape(len(subset), -1).float().cpu()
                    if not torch.isfinite(rows).all():
                        raise FloatingPointError("Nonfinite task VJP")
                    if layer not in totals:
                        totals[layer] = torch.zeros(len(token_ids), rows.shape[1])
                    totals[layer][offset:offset+len(subset)] += rows
                del gradients, cotangent
        del target, sources, captured
        if progress:
            progress({"completed": pi+1, "total": len(prompts)})
    return {layer: value/len(prompts) for layer, value in totals.items()}


def component_delta(source, donor, fitted, arm):
    difference = donor.float()-source.float()
    if arm == "dense":
        return difference
    if arm == "colour":
        return (difference @ fitted["colour_inverse"].float()) @ fitted["colour"].float()
    coefficients = difference @ fitted["country_inverse"].float()
    parts = {key: coefficients @ fitted[key].float() for key in ("full", "J", "remainder")}
    if arm in parts:
        return parts[arm]
    if arm in ("random_full", "random_J"):
        return parts[arm[7:]][fitted["permutation"]]*fitted["sign"]
    if arm == "remainder_norm_J":
        rnorm, jnorm = parts["remainder"].norm(), parts["J"].norm()
        if rnorm <= 1e-12 and jnorm > 1e-8:
            raise ValueError("Cannot norm-match a zero remainder")
        return parts["remainder"]*(jnorm/rnorm.clamp_min(1e-12))
    raise ValueError("Unknown component arm")


def make_patch(donor, fitted, scope, arm, *, suffix_tokens=21):
    diagnostics = {}
    def patch(hidden):
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("Expected a single sequence")
        first, stop = bounds(hidden.shape[1], scope, suffix_tokens)
        source = hidden[0, first:stop].detach().float().cpu().flatten()
        if source.shape != donor.shape:
            raise ValueError("Donor dimensions do not match the declared positions")
        delta = component_delta(source, donor, fitted, arm)
        changed = (source+delta).reshape(stop-first, hidden.shape[2]).to(hidden)
        if not torch.isfinite(changed).all():
            raise FloatingPointError("Nonfinite component edit")
        output = hidden.clone()
        output[0, first:stop] = changed
        if not torch.equal(output[:, :first], hidden[:, :first]) or not torch.equal(output[:, stop:], hidden[:, stop:]):
            raise ValueError("Edit changed an undeclared position")
        diagnostics.update({"positions": stop-first, "first": first, "stop": stop,
                            "requested_l2": float(delta.norm()),
                            "actual_l2": float((changed.float().cpu().flatten()-source).norm()),
                            "relative_l2": float(delta.norm()/source.norm().clamp_min(1e-12))})
        return output
    return patch, diagnostics


def quadratic(values):
    # Fixed feature order and implicit constant; no fitted monomial selection.
    width = values.shape[-1]
    return torch.cat((torch.ones_like(values[..., :1]), values,
                      torch.stack([values[..., i]*values[..., j] for i in range(width) for j in range(i, width)], -1)), -1)


def encode(encoder, values):
    return ((values-encoder["mean"]) @ encoder["basis"].T)/encoder["scale"]


def head_scores(head, values):
    kind = head["kind"]
    if kind == "linear_full":
        features = values
    else:
        features = encode(head["encoder"], values)
    if kind == "quadratic":
        features = quadratic(features)
    elif kind == "vanishing":
        features = (quadratic(features) @ head["polynomials"].T).abs()
    elif kind == "mlp":
        for index in (0, 2):
            features = features @ head["state"][f"{index}.weight"].T + head["state"][f"{index}.bias"]
            if index == 0:
                features = torch.nn.functional.gelu(features)
        return features
    return (features-head["feature_mean"]) @ head["weight"] + head["prior"]


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel()*value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    return 0


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value
