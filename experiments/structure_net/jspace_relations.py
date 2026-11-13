"""Bounded approximate relation discovery, with separately labelled exact algebra.

Inputs are continuous score matrices [examples, variables]. No Booleanization,
thresholded concept states, activation reconstruction or neural closure is assumed.
All returned models are JSON-serializable; discovery uses train, selection uses
development, and evaluation never changes coefficients or cutoffs.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations_with_replacement
import multiprocessing as mp
import time

import numpy as np


def _matrix(values, width=None):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or (width is not None and values.shape[1] != width):
        raise ValueError("Expected a nonempty matrix with the declared variable width")
    if not np.isfinite(values).all():
        raise ValueError("Score matrices must be finite")
    return values


def monomial_exponents(width, degree=2):
    """Graded deterministic order, including the constant: 4D/2 -> 15 terms."""
    if width < 1 or degree not in (1, 2):
        raise ValueError("Only positive width and degrees one or two are supported")
    result = [[0] * width]
    for d in range(1, degree + 1):
        for indices in combinations_with_replacement(range(width), d):
            powers = [0] * width
            for index in indices:
                powers[index] += 1
            result.append(powers)
    return result


def _monomials(values, exponents):
    return np.prod(values[:, None, :] ** np.asarray(exponents)[None, :, :], axis=2)


def _features(bank, values):
    values = _matrix(values, len(bank["variable_names"]))
    z = (values - np.asarray(bank["mean"])) / np.asarray(bank["scale"])
    return _monomials(z, bank["exponents"])


def _rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def fit_relation_bank(train, *, degree=2, variable_names=None, groups=None,
                      max_candidates=8, minimum_relative_std=1e-6):
    """Discover low-variance combinations of standardized nonconstant monomials.

    A coefficient has unit L2 norm in the individually standardized monomial
    basis. Its residual cannot be made small by multiplying the polynomial by
    a tiny scalar. Constants and low-variation variables/terms are excluded.
    ``groups=p`` identifies a source block of p columns followed by destination.
    """
    train = _matrix(train)
    width = train.shape[1]
    names = list(variable_names or [f"z{i}" for i in range(width)])
    if len(names) != width or len(set(names)) != width:
        raise ValueError("Variable names must be unique and match matrix width")
    if groups is not None and not 0 < groups < width:
        raise ValueError("groups must split source and destination columns")
    exponents = monomial_exponents(width, degree)
    if len(train) < 2 * len(exponents):
        raise ValueError("Need at least twice as many training rows as polynomial terms")
    mean, std = train.mean(0), train.std(0)
    variable_ok = std > minimum_relative_std * np.maximum(1.0, np.sqrt(np.mean(train**2, 0)))
    scale = np.where(variable_ok, std, 1.0)
    raw = _monomials((train - mean) / scale, exponents)
    feature_mean, feature_std = raw.mean(0), raw.std(0)
    active = np.array([sum(p) > 0 and all(variable_ok[j] or n == 0 for j, n in enumerate(p))
                       for p in exponents]) & (feature_std > minimum_relative_std)
    candidates = []
    for d in range(1, degree + 1):
        indices = np.flatnonzero(active & (np.sum(exponents, axis=1) <= d))
        if not len(indices):
            continue
        features = (raw[:, indices] - feature_mean[indices]) / feature_std[indices]
        _, singular, vt = np.linalg.svd(features, full_matrices=False)
        for offset in range(min(max_candidates, len(indices))):
            vector = vt[-1 - offset].copy()
            vector *= 1 if vector[np.argmax(np.abs(vector))] >= 0 else -1
            unit = np.zeros(len(exponents))
            unit[indices] = vector
            if any(abs(np.dot(unit, c["feature_coefficients"])) > 1 - 1e-8 for c in candidates):
                continue
            coefficients = np.zeros(len(exponents))
            coefficients[indices] = vector / feature_std[indices]
            coefficients[0] = -float(np.dot(coefficients[indices], feature_mean[indices]))
            support = np.asarray(exponents) > 0
            mixed = True
            if groups is not None:
                source_weight = float(np.sum(unit[np.any(support[:, :groups], axis=1)] ** 2))
                dest_weight = float(np.sum(unit[np.any(support[:, groups:], axis=1)] ** 2))
                mixed = min(source_weight, dest_weight) >= .05
            candidates.append({"id": f"degree{d}-sv{offset}", "degree": d,
                               "coefficients": coefficients.tolist(),
                               "feature_coefficients": unit.tolist(), "mixed": mixed,
                               "train_rms": float(singular[-1-offset] / np.sqrt(len(train)))})
    return {"version": 1, "kind": "approximate_continuous_polynomial_relations",
            "variable_names": names, "mean": mean.tolist(), "scale": scale.tolist(),
            "exponents": exponents, "groups": groups, "train_rows": len(train),
            "discovery": {"maximum_degree": degree, "max_candidates_per_degree": max_candidates,
                          "minimum_relative_std": minimum_relative_std,
                          "scope": "bounded SVD candidate search; no exhaustive absence claim"},
            "excluded_low_variation_variables": [names[i] for i in range(width) if not variable_ok[i]],
            "candidates": candidates, "validated": False,
            "normalization": "train-standardized variables; unit coefficient norm in train-standardized monomials"}


def _control_scores(bank, values, coefficients, *, seed, controls):
    if controls < 1:
        raise ValueError("At least one shuffle control is required")
    rng = np.random.default_rng(seed)
    coordinate, pairing = [], []
    for _ in range(controls):
        shuffled = np.column_stack([rng.permutation(values[:, j]) for j in range(values.shape[1])])
        coordinate.append(_rms(_features(bank, shuffled) @ coefficients))
        if bank.get("groups") is not None:
            shuffled = values.copy()
            split = bank["groups"]
            shuffled[:, split:] = values[rng.permutation(len(values)), split:]
            pairing.append(_rms(_features(bank, shuffled) @ coefficients))
    return coordinate, pairing


def _control_summary(observed, controls):
    median = float(np.median(controls))
    # A within-source + within-destination relation can have tiny residual under
    # every pairing. It supplies no pairing evidence even when floating-point
    # residuals differ by a factor of three. Never reward division by a floor.
    informative = median > 1e-8
    return {"median_rms": median, "rms_interval": np.quantile(controls, [.025, .975]).tolist(),
            "observed_to_control": observed / median if informative else max(1., observed/1e-8),
            "nontrivial_control_residual": informative,
            "empirical_p": float((1 + np.sum(np.asarray(controls) <= observed)) / (len(controls) + 1)),
            "permutations": len(controls)}


def _ideal_span_residual(candidate, accepted, exponents):
    """Coefficient distance to accepted relations times low-degree monomials.

    This is a finite-degree numeric redundancy diagnostic, not an exact ideal
    membership certificate for a noisy relation or a complete Macaulay method.
    In particular, quadratics generated by linear constraints are not new
    nonlinear structure merely because an SVD vector has degree-two entries.
    """
    if not accepted:
        return 1.
    exponents = [tuple(p) for p in exponents]
    lookup = {p: i for i, p in enumerate(exponents)}
    maximum = candidate["degree"]
    span = []
    for relation in accepted:
        coefficients = np.asarray(relation["coefficients"])
        for multiplier in exponents:
            if sum(multiplier) + relation["degree"] > maximum:
                continue
            shifted = np.zeros(len(exponents))
            for exponent, coefficient in zip(exponents, coefficients):
                if coefficient:
                    target = tuple(a+b for a, b in zip(exponent, multiplier))
                    if target in lookup:
                        shifted[lookup[target]] += coefficient
            norm = np.linalg.norm(shifted)
            if norm > 1e-12:
                span.append(shifted / norm)
    if not span:
        return 1.
    matrix = np.asarray(span).T
    target = np.asarray(candidate["coefficients"])
    fitted = matrix @ np.linalg.lstsq(matrix, target, rcond=1e-8)[0]
    return float(np.linalg.norm(target-fitted)/max(np.linalg.norm(target), 1e-12))


def validate_relation_bank(bank, dev, *, seed=7, controls=32, max_relations=3,
                           max_rms=.10, max_control_ratio=1/3):
    """Freeze candidates on dev; return a new dict, preserving the train fit.

    Absolute RMS <= .10 and one-third the shuffled RMS are prespecified defaults,
    not universal significance thresholds. Permutation p-values are descriptive
    and uncorrected for candidate/layer searches. No claims of confirmation.
    """
    if bank["validated"]:
        raise ValueError("Bank already validated; never retune a frozen bank")
    dev = _matrix(dev, len(bank["variable_names"]))
    features = _features(bank, dev)
    judged = []
    for candidate in bank["candidates"]:
        residuals = features @ candidate["coefficients"]
        rms = _rms(residuals)
        coordinate, pairing = _control_scores(bank, dev, candidate["coefficients"], seed=seed, controls=controls)
        coord_summary = _control_summary(rms, coordinate)
        pair_summary = _control_summary(rms, pairing) if pairing else None
        passes = (candidate["mixed"] and rms <= max_rms
                  and coord_summary["observed_to_control"] <= max_control_ratio
                  and (pair_summary is None or pair_summary["observed_to_control"] <= max_control_ratio))
        judged.append({**candidate, "dev_rms": rms,
                       "dev_abs95": float(np.quantile(np.abs(residuals), .95)),
                       "dev_coordinate_shuffle": coord_summary, "dev_pairing_shuffle": pair_summary,
                       "passes_dev": bool(passes), "accepted": False})
    # Prefer the smallest degree; an equally vanishing quadratic multiple of a
    # linear relation is a consequence, not another nonlinear discovery.
    accepted = []
    for i in sorted(range(len(judged)), key=lambda i: (judged[i]["degree"], judged[i]["dev_rms"])):
        residual = _ideal_span_residual(judged[i], [judged[j] for j in accepted], bank["exponents"])
        judged[i]["ideal_span_relative_residual"] = residual
        judged[i]["redundant_at_degree_bound"] = residual <= 1e-6
        if judged[i]["passes_dev"] and residual > 1e-6 and len(accepted) < max_relations:
            judged[i]["accepted"] = True
            accepted.append(i)
    return {**bank, "candidates": judged, "validated": True, "dev_rows": len(dev),
            "selection": {"max_rms": max_rms, "max_control_ratio": max_control_ratio,
                          "max_relations": max_relations, "seed": seed, "controls": controls},
            "redundancy": "degree-bounded numeric coefficient span; tolerance 1e-6; not exact noisy ideal membership",
            "accepted_ids": [judged[i]["id"] for i in accepted]}


def evaluate_relation_bank(bank, test, *, seed=701, controls=32):
    """Evaluate frozen accepted relations; all rows remain in the denominator."""
    if not bank["validated"]:
        raise ValueError("Validate/freeze the bank before heldout evaluation")
    test = _matrix(test, len(bank["variable_names"]))
    results = []
    features = _features(bank, test)
    for candidate in bank["candidates"]:
        if not candidate["accepted"]:
            continue
        residuals = features @ candidate["coefficients"]
        rms = _rms(residuals)
        coordinate, pairing = _control_scores(bank, test, candidate["coefficients"], seed=seed, controls=controls)
        coordinate_summary = _control_summary(rms, coordinate)
        pairing_summary = _control_summary(rms, pairing) if pairing else None
        passes_test = (rms <= bank["selection"]["max_rms"]
                       and coordinate_summary["observed_to_control"] <= bank["selection"]["max_control_ratio"]
                       and (pairing_summary is None or pairing_summary["observed_to_control"] <= bank["selection"]["max_control_ratio"]))
        results.append({"id": candidate["id"], "normalized_rms": rms, "passes_frozen_test_gate": bool(passes_test),
                        "absolute_residuals": np.abs(residuals).tolist(),
                        "frozen_dev_abs95": candidate["dev_abs95"],
                        "fraction_within_dev_abs95": float(np.mean(np.abs(residuals) <= max(candidate["dev_abs95"], 1e-12))),
                        "coordinate_shuffle": coordinate_summary,
                        "pairing_shuffle": pairing_summary})
    return {"rows": len(test), "accepted_count": len(results), "relations": results,
            "interpretation": "exploratory heldout approximate relations, not exact neural ideals"}


def fit_transition_baselines(train_source, train_destination):
    """Train-only OLS with intercept; linear and quadratic explicit baselines."""
    source, destination = _matrix(train_source), _matrix(train_destination)
    if len(source) != len(destination):
        raise ValueError("Source/destination rows must be aligned")
    mean, scale = source.mean(0), source.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    destination_scale = np.maximum(destination.std(0), 1e-12)
    arms = {}
    for degree in (1, 2):
        exponents = monomial_exponents(source.shape[1], degree)
        if len(source) < 2 * len(exponents):
            raise ValueError("Insufficient training rows for transition regression")
        features = _monomials((source-mean)/scale, exponents)
        coefficients, _, rank, _ = np.linalg.lstsq(features, destination, rcond=1e-10)
        arms[str(degree)] = {"exponents": exponents, "coefficients": coefficients.tolist(), "rank": int(rank)}
    return {"mean": mean.tolist(), "scale": scale.tolist(), "destination_scale": destination_scale.tolist(),
            "destination_mean": destination.mean(0).tolist(), "arms": arms}


def evaluate_transition_baselines(model, source, destination):
    source, destination = _matrix(source, len(model["mean"])), _matrix(destination, len(model["destination_mean"]))
    if len(source) != len(destination):
        raise ValueError("Source/destination rows must be aligned")
    scale = np.asarray(model["destination_scale"])
    results = {"train_mean": {"normalized_rmse": _rms((destination-model["destination_mean"])/scale)}}
    for degree, arm in model["arms"].items():
        features = _monomials((source-model["mean"])/model["scale"], arm["exponents"])
        residual = (features @ np.asarray(arm["coefficients"]) - destination) / scale
        results["linear" if degree == "1" else "quadratic"] = {
            "normalized_rmse": _rms(residual), "per_variable_rmse": np.sqrt(np.mean(residual**2, 0)).tolist()}
    return {"rows": len(source), "baselines": results,
            "interpretation": "predictive regression baseline; no Groebner or exact neural claim"}


def _groebner_worker(connection, payload):
    """Killable subprocess: symbolic work never executes in the caller process."""
    try:
        import sympy as sp
        variables = sp.symbols(f"v0:{payload['width']}")
        expressions = []
        for relation in payload["relations"]:
            expression = sum(sp.Rational(c) * sp.prod(v**p for v, p in zip(variables, powers))
                             for c, powers in zip(relation["rational_coefficients"], relation["exponents"]))
            expressions.append(expression)
        basis = sp.groebner(expressions, *variables, order="lex", domain=sp.QQ)
        unit = any(p.as_expr() == 1 for p in basis.polys)
        degree = max((int(p.total_degree()) for p in basis.polys), default=0)
        exploded = degree > 12 or len(basis.polys) > 32 or sum(len(str(p.as_expr())) for p in basis.polys) > 50000
        eliminated = []
        if not unit and not exploded:
            source = set(variables[:payload["source_width"]])
            for poly in basis.polys:
                if not poly.as_expr().free_symbols.intersection(source):
                    reduced = sp.Poly(poly.as_expr(), *variables[payload["source_width"]:])
                    eliminated.append({"polynomial": str(poly.as_expr()),
                                       "exponents": [list(p) for p, c in reduced.terms()],
                                       "rational_coefficients": [str(c) for p, c in reduced.terms()],
                                       "coefficients": [float(c) for p, c in reduced.terms()]})
        connection.send({"status": "unit_ideal" if unit else "degree_explosion" if exploded else "ok",
                         "unit_ideal": unit, "empty_elimination": not eliminated,
                         "max_basis_degree": degree, "basis_size": len(basis.polys),
                         "basis": [str(p.as_expr()) for p in basis.polys] if not exploded else [],
                         "consequences": eliminated})
    except Exception as exc:
        connection.send({"status": "symbolic_error", "error_type": type(exc).__name__, "consequences": []})
    finally:
        connection.close()


def _bounded_groebner(payload, timeout_s):
    if not 0 < timeout_s <= 5:
        raise ValueError("Symbolic timeout must be positive and at most five seconds")
    started = time.monotonic()
    ctx = mp.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_groebner_worker, args=(send, payload), daemon=True)
    process.start()
    send.close()
    try:
        if receive.poll(max(0., timeout_s - (time.monotonic()-started))):
            try:
                result = receive.recv()
            except EOFError:
                result = {"status": "worker_failed", "consequences": []}
        else:
            result = {"status": "timeout", "consequences": []}
    finally:
        if process.is_alive():
            process.terminate()
        process.join(.1)
        if process.is_alive():
            process.kill()
            process.join(.1)
        receive.close()
    return {**result, "elapsed_seconds": time.monotonic()-started, "timeout_seconds": timeout_s}


def groebner_propagation(source_bank, joint_bank, source_dev, destination_dev, *,
                         max_relations=7, timeout_s=5, max_denominator=1000,
                         rationalization_tolerance=.01):
    """Eliminate source variables from <=7 accepted, dev-checked surrogates.

    There must be >=1 accepted source and >=1 mixed transition relation. Input
    banks must use the same training rows for compatible source normalization.
    Rationalization cannot certify an exact relation in the neural network.
    Its numeric error and any inconsistent unit ideal remain explicit outputs.
    """
    if not 2 <= max_relations <= 7:
        raise ValueError("Bounded algebra uses two to seven relations: at most three source and four mixed")
    if not 0 < timeout_s <= 5:
        raise ValueError("Symbolic timeout must be positive and at most five seconds")
    if not source_bank["validated"] or not joint_bank["validated"]:
        raise ValueError("Freeze both banks on development before algebra")
    width = len(source_bank["variable_names"])
    if joint_bank["groups"] != width:
        raise ValueError("Joint source block must match source bank")
    for field in ("mean", "scale"):
        if not np.allclose(source_bank[field], joint_bank[field][:width], rtol=1e-10, atol=1e-12):
            raise ValueError("Source and joint coordinate normalization must match")
    source_dev = _matrix(source_dev, width)
    destination_dev = _matrix(destination_dev, len(joint_bank["variable_names"])-width)
    if len(source_dev) != len(destination_dev):
        raise ValueError("Development rows must be aligned")
    source_candidates = sorted((c for c in source_bank["candidates"] if c["accepted"]), key=lambda c: c["dev_rms"])
    joint_candidates = sorted((c for c in joint_bank["candidates"] if c["accepted"] and c["mixed"]), key=lambda c: c["dev_rms"])
    base = {"interpretation": "exact elimination of an approximate rational surrogate only",
            "exact_neural_claim": False, "coefficient_field": "QQ", "monomial_order": "lex",
            "variable_names": joint_bank["variable_names"], "source_width": width,
            "selection": {"max_rms": source_bank["selection"]["max_rms"],
                          "max_control_ratio": source_bank["selection"]["max_control_ratio"]},
            "destination_mean": joint_bank["mean"][width:], "destination_scale": joint_bank["scale"][width:]}
    if not source_candidates or not joint_candidates:
        return {**base, "status": "insufficient_accepted_relations", "consequences": [], "rationalization": []}
    # Reserve >=1 source constraint and >=1 transition. Prefer source constraints
    # then mixed equations; never exceed three source or four mixed generators.
    # Four bridge equations can be necessary to eliminate four source variables.
    source_count = min(3, len(source_candidates), max_relations-1)
    selected = [(source_bank, c, source_dev) for c in source_candidates[:source_count]]
    selected += [(joint_bank, c, np.column_stack((source_dev, destination_dev)))
                 for c in joint_candidates[:min(4, max_relations-source_count)]]
    relations, checks = [], []
    for bank, candidate, dev in selected:
        coefficients = np.asarray(candidate["coefficients"])
        rational = [str(Fraction(float(c)).limit_denominator(max_denominator)) for c in coefficients]
        rounded = np.array([float(Fraction(c)) for c in rational])
        features = _features(bank, dev)
        original_rms, rounded_rms = _rms(features @ coefficients), _rms(features @ rounded)
        delta = _rms(features @ (rounded-coefficients))
        accepted = (bool(np.any(rounded[1:] != 0)) and delta <= rationalization_tolerance
                    and rounded_rms <= max(original_rms*1.5, original_rms+rationalization_tolerance))
        checks.append({"id": candidate["id"], "kind": "source" if bank is source_bank else "mixed",
                       "original_dev_rms": original_rms, "rationalized_dev_rms": rounded_rms,
                       "change_dev_rms": delta, "accepted": accepted,
                       "max_denominator": max_denominator})
        exponents = [p + [0]*(len(joint_bank["variable_names"])-len(p)) for p in bank["exponents"]]
        relations.append({"exponents": exponents, "rational_coefficients": rational})
    if not all(c["accepted"] for c in checks):
        return {**base, "status": "rationalization_rejected", "rationalization": checks, "consequences": []}
    result = _bounded_groebner({"relations": relations, "width": len(joint_bank["variable_names"]), "source_width": width}, timeout_s)
    consequences = []
    standardized = (destination_dev-np.asarray(base["destination_mean"]))/np.asarray(base["destination_scale"])
    for consequence in result["consequences"]:
        features = _monomials(standardized, consequence["exponents"])
        coefficients = np.asarray(consequence["coefficients"])
        nonconstant = np.sum(consequence["exponents"], axis=1) > 0
        contribution_scale = float(np.sqrt(np.mean(np.sum((features[:, nonconstant]*coefficients[nonconstant])**2, axis=1))))
        if contribution_scale <= 1e-12 or not np.isfinite(contribution_scale):
            continue
        consequences.append({**consequence, "normalizer_dev": contribution_scale,
                             "normalized_dev_rms": _rms(features@coefficients)/contribution_scale,
                             "passes_dev_residual_gate": _rms(features@coefficients)/contribution_scale <= base["selection"]["max_rms"]})
    return {**base, **result, "rationalization": checks, "generators": relations,
            "consequences": consequences, "empty_elimination": not consequences}


def evaluate_propagated_relations(result, destination_test, *, seed=701, controls=32):
    """Test actual destination-only elimination consequences, without refitting."""
    if controls < 1:
        raise ValueError("At least one shuffle control is required")
    values = _matrix(destination_test, len(result["destination_mean"]))
    z = (values-np.asarray(result["destination_mean"]))/np.asarray(result["destination_scale"])
    rng = np.random.default_rng(seed)
    rows = []
    for consequence in result["consequences"]:
        coefficients = np.asarray(consequence["coefficients"])
        normalizer = consequence["normalizer_dev"]
        residuals = _monomials(z, consequence["exponents"]) @ coefficients / normalizer
        shuffled_rms = []
        for _ in range(controls):
            shuffled = np.column_stack([rng.permutation(z[:, j]) for j in range(z.shape[1])])
            shuffled_rms.append(_rms(_monomials(shuffled, consequence["exponents"]) @ coefficients / normalizer))
        control = _control_summary(_rms(residuals), shuffled_rms)
        selection = result.get("selection", {"max_rms": .10, "max_control_ratio": 1/3})
        passes = (_rms(residuals) <= selection["max_rms"]
                  and control["observed_to_control"] <= selection["max_control_ratio"])
        rows.append({"polynomial": consequence["polynomial"], "normalized_rms": _rms(residuals),
                     "passes_frozen_test_gate": bool(passes),
                     "passes_dev_residual_gate": consequence.get("passes_dev_residual_gate"),
                     "absolute_residuals": np.abs(residuals).tolist(),
                     "coordinate_shuffle": control})
    return {"rows": len(values), "propagated_relation_count": len(rows), "relations": rows,
            "exact_neural_claim": False,
            "interpretation": "heldout numeric check of destination consequences of an approximate surrogate"}
