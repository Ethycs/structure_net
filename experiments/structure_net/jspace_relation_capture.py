"""Frozen-model captures for continuous relation propagation, without file I/O.

The caller owns reservation validation and model loading. All returned hidden
states and fitted control banks are FP32 CPU tensors; no discretization is used.
"""
from __future__ import annotations

import json

import torch

from .jspace_runtime import generate_answer
from .jspace_tasks import normalize_answer, pca_basis, random_basis


COUNTRIES = ("China", "France", "Germany", "Russia")
ARMS = ("jacobian", "pca", "random")


def _inputs(model, cases, layers):
    cases, layers = list(cases), list(layers)
    if (not layers or len(set(layers)) != len(layers)
            or any(type(layer) is not int or not 0 <= layer < len(model.model.layers) for layer in layers)):
        raise ValueError("Observation layers must be distinct actual decoder block indices")
    identifiers = set()
    for case in cases:
        if not isinstance(case, dict) or any(
            not isinstance(case.get(key), str) or not case[key].strip()
            for key in ("case_id", "prompt", "answer", "intermediate", "swap_to", "swap_answer")
        ):
            raise ValueError("Every case requires case_id, prompt, original/swapped answers and concepts")
        if case["case_id"] in identifiers:
            raise ValueError("Duplicate case_id would duplicate evidence")
        identifiers.add(case["case_id"])
        json.dumps(case, allow_nan=False)
    width = model.get_input_embeddings().weight.shape[1]
    return cases, layers, width


def _prompt(tokenizer, case):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": case["prompt"]}],
        tokenize=False, add_generation_prompt=True,
    )


def _append(captured, result, case, layers, width, *, arm, site, diagnostics, progress):
    observed = result["observed"]
    if set(observed) != set(layers):
        raise ValueError("Missing or unexpected prefill observations")
    hidden = torch.stack([observed[layer] for layer in layers]).detach().float().cpu()
    if hidden.shape != (len(layers), width) or not bool(torch.isfinite(hidden).all()):
        raise ValueError("Invalid captured hidden-state shape or nonfinite values")
    answer = result["answer"]
    original = bool(answer) and normalize_answer(answer) == normalize_answer(case["answer"])
    counterfactual = bool(answer) and normalize_answer(answer) == normalize_answer(case["swap_answer"])
    row = {
        "case_id": case["case_id"], "case": dict(case),
        "intermediate": case["intermediate"], "swap_to": case["swap_to"],
        "expected_answer": case["answer"], "expected_swap_answer": case["swap_answer"],
        **{key: value for key, value in result.items() if key != "observed"},
        "original_correct": original, "counterfactual_correct": counterfactual,
        "correct": original if arm == "natural" else counterfactual,
        "arm": arm, "source_site": site, "observation_layers": list(layers),
        "hidden_row_index": len(captured["records"]), **diagnostics,
    }
    json.dumps(row, allow_nan=False)
    captured["records"].append(row)
    captured["hidden"].append(hidden)
    if progress is not None:
        progress(dict(row))


def _finish(captured, layers, width):
    captured["hidden"] = (torch.stack(captured["hidden"]) if captured["hidden"]
                          else torch.empty((0, len(layers), width), dtype=torch.float32))
    return captured


def capture_natural(model, tokenizer, cases, *, layers, progress=None):
    """Capture one greedy chat answer and final-position state per case/layer.

    Inputs may be any caller-selected split. No competence filter is applied.
    ``progress(row)`` is called once per completed case. Tensor rows and records
    have identical ordering; the second tensor axis follows ``layers`` exactly.
    """
    cases, layers, width = _inputs(model, cases, layers)
    captured = {"records": [], "hidden": []}
    for case in cases:
        result = generate_answer(model, tokenizer, _prompt(tokenizer, case),
                                 observe_layers=layers, max_new_tokens=8)
        _append(captured, result, case, layers, width, arm="natural", site=None,
                diagnostics={}, progress=progress)
    return _finish(captured, layers, width)


def build_control_banks(training_hidden, *, seed=7):
    """Fit four PCA axes per layer using only the supplied training states.

    PCA uses centered exact FP64 SVD on CPU, with a deterministic sign convention.
    Runtime projections are origin-based, as for J rows. ``mean`` is returned
    separately. All layers share the same seeded orthonormal random directions.
    """
    if (not isinstance(training_hidden, torch.Tensor) or training_hidden.ndim != 3
            or not training_hidden.is_floating_point() or training_hidden.shape[0] < 5
            or training_hidden.shape[1] < 1 or training_hidden.shape[2] < 4
            or not bool(torch.isfinite(training_hidden).all())):
        raise ValueError("Training states must be finite floating [N>=5, layers>=1, width>=4]")
    work = training_hidden.detach().cpu().double()
    banks = []
    for index in range(work.shape[1]):
        basis = pca_basis(work[:, index], 4)
        # SVD axes have an arbitrary sign; pin the largest-magnitude entry positive.
        pivots = basis.abs().argmax(dim=1)
        signs = basis[torch.arange(4), pivots].sign()
        banks.append((basis * signs[:, None]).float())
    random = random_basis(work.shape[2], 4, seed=seed, dtype=torch.float64).float()
    return {"pca": torch.stack(banks),
            "random": random.unsqueeze(0).repeat(work.shape[1], 1, 1),
            "mean": work.mean(dim=0).float()}


def _banks(banks, layers, width):
    output = {}
    for arm in ARMS:
        bank = banks.get(arm)
        if (not isinstance(bank, torch.Tensor) or bank.shape != (len(layers), 4, width)
                or not bank.is_floating_point() or not bool(torch.isfinite(bank).all())):
            raise ValueError(f"{arm} bank must be finite floating [layers, 4, hidden_width]")
        bank = bank.detach().cpu().double()
        norms = bank.norm(dim=-1, keepdim=True)
        if bool((norms <= 1e-12).any()):
            raise ValueError(f"{arm} bank contains a zero concept row")
        output[arm] = bank / norms
    return output


def _pair(bank, source, target):
    pair = bank[[source, target]]
    if int(torch.linalg.matrix_rank(pair, rtol=1e-6)) != 2:
        raise ValueError("Source and target concept directions must be independent")
    return pair.float(), torch.linalg.pinv(pair, rtol=1e-6).float()


def _final_position_patch(jacobian_pair, control_pair, *, arm):
    """Prepare pair-coordinate swaps; diagnostics distinguish FP32 and cast norms."""
    diagnostics = {}

    def patch(hidden):
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("Expected one prompt's residual tensor")
        work = hidden[:, -1, :].float()

        def delta_for(pair):
            vectors, inverse = (part.to(work.device) for part in pair)
            coefficients = work @ inverse
            return (coefficients.flip(-1) - coefficients) @ vectors

        reference = delta_for(jacobian_pair)
        reference_norm = reference.norm()
        delta = reference if arm == "jacobian" else delta_for(control_pair)
        if arm != "jacobian":
            norm = delta.norm()
            if float(norm) <= 1e-12 and float(reference_norm) > 1e-12:
                raise ValueError("Degenerate control cannot match a nonzero J intervention")
            delta = delta * (reference_norm / norm.clamp_min(1e-12))
        changed = (work + delta).to(hidden.dtype)
        diagnostics.update({
            "alpha": 1.0, "intervention": "pair_coefficient_swap_final_prompt_position",
            "jacobian_reference_l2": float(reference_norm),
            "requested_perturbation_l2": float(delta.norm()),
            "perturbation_l2": float((changed.float() - work).norm()),
        })
        updated = hidden.clone()
        updated[:, -1, :] = changed
        return updated

    return patch, diagnostics


def capture_interventions(model, tokenizer, cases, *, layers, banks, concept_names,
                          sites=(11, 23), progress=None):
    """Capture every case × site × J/PCA/random prefill intervention.

    Banks use fixed country row order China, France, Germany, Russia. The named
    pair's pseudoinverse coefficients are swapped, not its dot-product scores.
    Only the final prompt position changes. Norm-matched controls use the same
    unpatched state and country indices. Earlier positions, other layers, model
    parameters, and cached generation steps retain the original model behavior.
    """
    cases, layers, width = _inputs(model, cases, layers)
    names = [name.casefold() for name in concept_names]
    if names != [name.casefold() for name in COUNTRIES]:
        raise ValueError("Concept rows must be ordered China, France, Germany, Russia")
    sites = list(sites)
    if (not sites or len(set(sites)) != len(sites)
            or any(type(site) is not int or site not in layers for site in sites)):
        raise ValueError("Distinct intervention sites must appear in observation layers")
    banks = _banks(banks, layers, width)
    indices = {name: index for index, name in enumerate(names)}
    pairs = {}
    for case in cases:
        source, target = case["intermediate"].casefold(), case["swap_to"].casefold()
        if source not in indices or target not in indices or source == target:
            raise ValueError("Cases must specify two distinct known countries")
        for site in sites:
            for arm in ARMS:
                key = (site, arm, source, target)
                if key not in pairs:
                    pairs[key] = _pair(banks[arm][layers.index(site)], indices[source], indices[target])
    captured = {"records": [], "hidden": []}
    for case in cases:
        source, target = case["intermediate"].casefold(), case["swap_to"].casefold()
        prompt = _prompt(tokenizer, case)
        for site in sites:
            for arm in ARMS:
                patch, diagnostics = _final_position_patch(
                    pairs[(site, "jacobian", source, target)], pairs[(site, arm, source, target)], arm=arm)
                result = generate_answer(model, tokenizer, prompt, layer=site, patch=patch,
                                         observe_layers=layers, max_new_tokens=8)
                if not diagnostics:
                    raise RuntimeError("Intervention hook did not run")
                _append(captured, result, case, layers, width, arm=arm, site=site,
                        diagnostics=diagnostics, progress=progress)
    return _finish(captured, layers, width)
