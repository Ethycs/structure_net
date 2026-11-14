"""Frozen exploratory mask search and deterministic three-layer repair selection.

True means the independently recovered compressed block; False means native.
All ranking is conditional on the named background and development population.
"""
from itertools import combinations
import math

LAYERS = 36


def mask_id(mask):
    if len(mask) != LAYERS or any(type(x) is not bool for x in mask):
        raise ValueError("Exactly 36 boolean decisions required")
    return "m" + "".join("1" if x else "0" for x in mask)


def compressed(layers):
    layers = tuple(layers)
    if len(set(layers)) != len(layers) or any(type(i) is not int or not 0 <= i < LAYERS for i in layers):
        raise ValueError("Unique absolute decoder indices 0..35 required")
    return tuple(i in layers for i in range(LAYERS))


def restored(layers):
    return tuple(not x for x in compressed(layers))


def six_groups():
    return [tuple(range(i, i+6)) for i in range(0, LAYERS, 6)] + [tuple(range(i, LAYERS, 6)) for i in range(6)]


def screen_masks():
    masks = [compressed(()), compressed((0, 1)), restored(())]
    masks += [restored((i,)) for i in range(LAYERS)]
    masks += [restored(pair) for pair in combinations(range(LAYERS), 2)]
    masks += [restored(group) for group in six_groups()]
    return {mask_id(m): m for m in masks}


def metric_key(result):
    values = result["metrics"]
    if (type(values["correct"]) is not int or type(values["teacher_agreement_count"]) is not int
            or not math.isfinite(values["choice_ce"]) or not 0 <= values["correct"] <= 64
            or not 0 <= values["teacher_agreement_count"] <= 64):
        raise ValueError("Finite 64-case decision metrics required")
    return values["correct"], values["teacher_agreement_count"], -values["choice_ce"]


def result_for(results, layers):
    return results[mask_id(restored(layers))]


def rank_restorations(results, groups):
    return sorted((tuple(g) for g in groups), key=lambda g: tuple(-x for x in metric_key(result_for(results, g))) + (g,))


def select_candidates(results):
    """Use higher-order rescue or two singletons plus interacting pair endpoints."""
    if not set(screen_masks()) <= set(results):
        raise ValueError("Complete prospective screen required before adaptive selection")
    base = metric_key(result_for(results, ()))[0]
    singles = rank_restorations(results, [(i,) for i in range(LAYERS)])
    pairs = rank_restorations(results, combinations(range(LAYERS), 2))
    groups = rank_restorations(results, six_groups())
    if metric_key(result_for(results, groups[0]))[0] >= metric_key(result_for(results, pairs[0]))[0] + 2:
        chosen = groups[0]
        method = "six_layer_group_rescue_exceeds_best_pair_by_at_least_two_correct"
    else:
        chosen = [g[0] for g in singles[:2]]
        def pair_key(pair):
            score = metric_key(result_for(results, pair))
            excess = score[0] - sum(metric_key(result_for(results, (i,)))[0] for i in pair) + base
            return (-excess, *(-x for x in score), pair)
        beneficial = [p for p in pairs if metric_key(result_for(results, p))[0] > base]
        for pair in sorted(beneficial, key=pair_key):
            for i in pair:
                if i not in chosen and len(chosen) < 6:
                    chosen.append(i)
            if len(chosen) == 6:
                break
        for (i,) in singles:
            if i not in chosen and len(chosen) < 6:
                chosen.append(i)
        method = "top_two_single_rescues_then_positive_pair_excess_rescue_then_single_fill"
    return dict(layers=sorted(chosen), method=method, best_single=list(singles[0]),
        best_pair=list(pairs[0]), best_six_group=list(groups[0]), baseline_correct=base)


def refinement_masks(candidates):
    if len(candidates) != 6 or len(set(candidates)) != 6:
        raise ValueError("Six distinct candidate layers required")
    compressed(candidates)
    masks = {}
    for size in range(7):
        for group in combinations(sorted(candidates), size):
            for m in (compressed(group), restored(group)):
                masks[mask_id(m)] = m
    return masks


def selection(results, candidates):
    expected = refinement_masks(candidates)
    if not set(expected) <= set(results):
        raise ValueError("Both complete candidate-background enumerations required")
    ranked = rank_restorations(results, combinations(sorted(candidates), 3))
    chosen = ranked[0]
    baseline = metric_key(result_for(results, ()))[0]
    rescue = metric_key(result_for(results, chosen))[0] - baseline
    return dict(layers=list(chosen), restored_mask_id=mask_id(restored(chosen)),
        restored_correct=metric_key(result_for(results, chosen))[0], baseline_correct=baseline,
        rescue_correct=rescue, positive_rescue=rescue > 0, group_size=3,
        rule="best native-restoration triple in full-compact background; correct, teacher agreement, lower choice CE, lexical indices",
        candidates=list(sorted(candidates)), scope="Conditional repair target; not a proof of learnability or a globally worst subset")


def paired(reference, candidate):
    a, b = reference["records"], candidate["records"]
    if len(a) != 64 or len(b) != 64 or any((x["identity"], x["answer"]) != (y["identity"], y["answer"]) for x, y in zip(a, b)):
        raise ValueError("Exactly paired original 64 decisions required")
    gains = [x["identity"] for x, y in zip(a, b) if x["prediction"] != x["answer"] and y["prediction"] == y["answer"]]
    losses = [x["identity"] for x, y in zip(a, b) if x["prediction"] == x["answer"] and y["prediction"] != y["answer"]]
    return dict(gained=len(gains), lost=len(losses), gained_identities=gains, lost_identities=losses)


def summarize(results, candidate_record):
    candidates = candidate_record["layers"]
    selected = selection(results, candidates)
    base = result_for(results, ())
    rows = []
    for context in ("outside_native", "outside_compact"):
        for size in range(7):
            groups = list(combinations(candidates, size))
            def identifier(g): return mask_id(compressed(g) if context == "outside_native" else restored(g))
            ordered = sorted(groups, key=lambda g: tuple(-x for x in metric_key(results[identifier(g)])) + (g,))
            best, worst = ordered[0], ordered[-1]
            rows.append(dict(context=context, candidate_switch_count=size,
                compressed_count=size if context == "outside_native" else LAYERS-size,
                tested=len(groups), best_group=list(best), worst_group=list(worst),
                best_mask=identifier(best), worst_mask=identifier(worst),
                best_correct=metric_key(results[identifier(best)])[0], worst_correct=metric_key(results[identifier(worst)])[0]))
    interactions = []
    baseline = metric_key(base)[0]
    for i, j in combinations(range(LAYERS), 2):
        score = metric_key(result_for(results, (i, j)))[0]
        singles = [metric_key(result_for(results, (x,)))[0] for x in (i, j)]
        interactions.append(dict(layers=[i,j], restored_correct=score, restoration_gain=score-baseline,
            excess_rescue=score-sum(singles)+baseline))
    qualified = [key for key in results if metric_key(results[key])[0] >= 47]
    qualified.sort(key=lambda key: (-key.count("1"), *(-x for x in metric_key(results[key])), key))
    return dict(selected=selected, candidate_record=candidate_record, same_count_rankings=rows,
        paired_selected_restoration=paired(base, results[selected["restored_mask_id"]]),
        pair_interactions=interactions, unique_masks=len(results), qualified_mask_count=len(qualified),
        most_compressed_qualified_mask=qualified[0] if qualified else None,
        complete_global_search=False, confirmed=False)
