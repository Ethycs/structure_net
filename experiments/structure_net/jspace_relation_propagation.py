"""Exact CPU control for relation propagation; no neural/compression evidence."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

import sympy as sp


def eliminate(generators, eliminated, retained):
    basis = sp.groebner(generators, *eliminated, *retained, order="lex", domain=sp.QQ)
    relations = [p.as_expr() for p in basis.polys if not p.as_expr().free_symbols.intersection(eliminated)]
    return sp.groebner(relations, *retained, order="lex", domain=sp.QQ)


def allowed_boolean_states(basis, variables):
    return [list(values) for values in product((0, 1), repeat=len(variables))
            if all(p.as_expr().subs(dict(zip(variables, values))) == 0 for p in basis.polys)]


def exact_control():
    a, b, c, d, e, f = sp.symbols("a b c d e f")
    initial = [a*a-a, b*b-b]
    first = eliminate(initial + [c-a*b, d-a-b+a*b], (a, b), (c, d))
    second = eliminate([p.as_expr() for p in first.polys] + [e-d+c, f-c], (c, d), (e, f))
    altered = eliminate([p.as_expr() for p in first.polys] + [e-d, f-c], (c, d), (e, f))
    actual_first = sorted({(x*y, x+y-x*y) for x, y in product((0, 1), repeat=2)})
    actual_second = sorted({(y-x, x) for x, y in actual_first})
    assert allowed_boolean_states(first, (c, d)) == [list(x) for x in actual_first]
    assert allowed_boolean_states(second, (e, f)) == [list(x) for x in actual_second]
    assert first.reduce(sp.expand(c*(d-c)))[1] == 0
    assert altered.reduce(e*f)[1] != 0
    assert [1, 1] in allowed_boolean_states(altered, (e, f))
    return {
        "study_id": "JSPACE-RELATION-PROPAGATION-01",
        "status": "systems_algebra_control_only_not_neural_evidence",
        "coefficient_field": "QQ", "monomial_order": "lex",
        "layer0": {"variables": ["a", "b"], "relations": [str(x) for x in initial]},
        "layer1": {"meaning": "c=AND(a,b); d=OR(a,b)",
                   "groebner_basis": [str(p.as_expr()) for p in first.polys],
                   "allowed_states": allowed_boolean_states(first, (c, d))},
        "layer2": {"meaning": "e=XOR(a,b)=d-c; f=AND(a,b)=c",
                   "groebner_basis": [str(p.as_expr()) for p in second.polys],
                   "allowed_states": allowed_boolean_states(second, (e, f))},
        "negative_control": {"meaning": "replace XOR output by OR output e=d",
                             "groebner_basis": [str(p.as_expr()) for p in altered.polys],
                             "ef_remainder": str(altered.reduce(e*f)[1])},
        "checks": {"exhaustive_state_sets_match": True, "ef_relation_transports": True,
                   "changed_transition_detected": True},
        "neural_relations_measured": False, "compression_tested": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = exact_control()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
