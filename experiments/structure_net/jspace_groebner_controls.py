"""Exact, deliberately small controls for what algebra can and cannot certify.

These are constructed systems, not measurements of a language model. Float64
payload accounting is confined to a finite, exactly representable dyadic grid.
"""
from __future__ import annotations

import math
import struct

import sympy as sp

from .jspace_relation_propagation import eliminate, exact_control


CONTROL_IDS = (
    "constraint_created_from_zero_source_ideal", "constraint_transport",
    "unique_coordinate_decoder", "ambiguous_coordinate_decoder",
    "unstable_unique_decoder", "closed_reduced_dynamics",
    "nonclosed_projection", "real_domain_not_encoded_by_ideal",
    "near_relations_can_generate_unit_ideal", "polynomial_scaling",
    "normal_form_can_increase_work",
)


def expressions(basis):
    return [str(p.as_expr()) for p in basis.polys]


def remainder(expression, generators, variables):
    return sp.groebner(generators, *variables, order="lex", domain=sp.QQ).reduce(sp.expand(expression))[1]


def grid():
    return [sp.Rational(i, 32) for i in range(-32, 33)]


def pack_points(points):
    return b"".join(struct.pack("<dd", float(x), float(y)) for x, y in points)


def coordinate_payloads():
    """Actual payload round trips; fixed decoder code and framing are excluded."""
    points = [(x, x*x) for x in grid()]
    original = pack_points(points)
    x_payload = b"".join(struct.pack("<d", float(x)) for x, _ in points)
    x_values = [row[0] for row in struct.iter_unpack("<d", x_payload)]
    x_restored = pack_points([(x, x*x) for x in x_values])
    y_payload = b"".join(struct.pack("<d", float(y)) for _, y in points)
    signs = bytearray(math.ceil(len(points)/8))
    for i, (x, _) in enumerate(points):
        signs[i//8] |= int(bool(x < 0)) << (i % 8)
    y_values = [row[0] for row in struct.iter_unpack("<d", y_payload)]
    restored = [(math.sqrt(y) * (-1 if signs[i//8] & (1 << (i % 8)) else 1), y)
                for i, y in enumerate(y_values)]
    return {"states": len(points), "original_bytes": len(original),
            "retain_x_bytes": len(x_payload), "retain_y_and_branch_bytes": len(y_payload)+len(signs),
            "retain_x_bitwise_roundtrip": original == x_restored,
            "retain_y_and_branch_bitwise_roundtrip": original == pack_points(restored),
            "payload_dtype": "little-endian float64; branch bits packed into bytes",
            "scope": "finite dyadic grid; payload only; fixed decoder/framing excluded; no model weights removed"}


def run_controls():
    x, y, u, v = sp.symbols("x y u v")
    rows = []

    created = eliminate([u-x, v-x*x], (x,), (u, v))
    rows.append({"id": CONTROL_IDS[0], "source_ideal": [], "graph": ["u-x", "v-x**2"],
                 "destination_basis": expressions(created),
                 "passed": created.reduce(v-u*u)[1] == 0 and bool(created.polys),
                 "classification": "graph_can_create_destination_constraints"})

    transported = exact_control()
    rows.append({"id": CONTROL_IDS[1], "details": transported,
                 "passed": all(transported["checks"].values()),
                 "classification": "exact_relation_transport_on_constructed_boolean_system"})

    payloads = coordinate_payloads()
    decoder = remainder(y, [y-x*x], (y, x))
    rows.append({"id": CONTROL_IDS[2], "relation": "y-x**2", "retained": ["x"],
                 "decoder": str(decoder), "payload": payloads,
                 "passed": decoder == x*x and payloads["retain_x_bitwise_roundtrip"],
                 "classification": "coordinate_compression_with_exact_decoder_on_declared_domain"})

    fibres = {}
    for value in grid():
        fibres.setdefault(value*value, []).append(value)
    collisions = [values for values in fibres.values() if len(values) > 1]
    rows.append({"id": CONTROL_IDS[3], "retained": ["y"], "relation": "y-x**2",
                 "ambiguous_fibres": len(collisions), "total_fibres": len(fibres),
                 "witness": [{"x": -1, "y": 1}, {"x": 1, "y": 1}],
                 "repair": "retain one sign bit, or impose a known one-sided domain",
                 "passed": len(collisions) == 32 and payloads["retain_y_and_branch_bitwise_roundtrip"],
                 "classification": "relation_does_not_imply_unique_inverse"})

    inverse = 1/x
    samples = []
    for value in (sp.Integer(1), sp.Rational(1, 1000), sp.Rational(1, 1000000)):
        samples.append({"x": str(value), "y": str(inverse.subs(x, value)),
                        "absolute_sensitivity": str(abs(sp.diff(inverse, x).subs(x, value)))})
    rows.append({"id": CONTROL_IDS[4], "relation": "x*y-1", "decoder": "1/x; x != 0",
                 "samples": samples, "relative_condition_number": 1,
                 "passed": samples[-1]["absolute_sensitivity"] == "1000000000000",
                 "classification": "absolute_error_amplification_near_zero_despite_unique_inverse",
                 "qualification": "relative conditioning is benign; this is an absolute-error/quantization counterexample"})

    valid_transition = {x: x+1, y: (x+1)**2}
    bad_transition = {x: x+1, y: y+1}
    preserved = remainder((y-x*x).subs(valid_transition, simultaneous=True), [y-x*x], (y, x))
    broken = remainder((y-x*x).subs(bad_transition, simultaneous=True), [y-x*x], (y, x))
    transitions, mismatches = 0, 0
    for start in grid():
        state, compressed = (start, start*start), start
        for _ in range(8):
            state = (state[0]+1, (state[0]+1)**2)
            compressed += 1
            transitions += 1
            mismatches += state != (compressed, compressed*compressed)
    rows.append({"id": CONTROL_IDS[5], "source_ideal": ["y-x**2"],
                 "full_transition": ["x+1", "(x+1)**2"], "reduced_transition": "x+1",
                 "pullback_normal_form": str(preserved), "altered_transition_normal_form": str(broken),
                 "transitions_checked": transitions, "mismatches": mismatches,
                 "passed": preserved == 0 and broken != 0 and mismatches == 0,
                 "classification": "closed_reduced_dynamics_for_specified_polynomial_system"})

    unclosed = remainder(x+u, [y-x*x], (y, u, x))
    witness = [{"state": [1, 1, 0], "retained_x": 1, "next_retained_x": 1},
               {"state": [1, 1, 1], "retained_x": 1, "next_retained_x": 2}]
    rows.append({"id": CONTROL_IDS[6], "relation": "y-x**2; u is unconstrained",
                 "transition": ["x+u", "(x+u)**2", "u"], "retained": ["x"],
                 "next_x_normal_form": str(unclosed), "witness": witness,
                 "passed": u in unclosed.free_symbols and witness[0]["next_retained_x"] != witness[1]["next_retained_x"],
                 "classification": "valid_relation_but_discarded_variable_controls_future"})

    image = eliminate([y-x*x], (x,), (y,))
    real_roots = sp.solveset(x*x+1, x, domain=sp.S.Reals)
    rows.append({"id": CONTROL_IDS[7], "map": "y=x**2 with real x",
                 "destination_basis": expressions(image), "witness_y": -1,
                 "real_preimage": str(real_roots),
                 "passed": not image.polys and real_roots == sp.EmptySet,
                 "classification": "elimination_gives_algebraic_closure_not_real_domain_inequalities"})

    epsilon = sp.Rational(1, 1000000)
    noisy = sp.groebner([x, x-epsilon], x, domain=sp.QQ)
    rows.append({"id": CONTROL_IDS[8], "approximate_generators": ["x", f"x-{epsilon}"],
                 "sample": str(epsilon/2), "maximum_absolute_residual": str(epsilon/2),
                 "exact_basis": expressions(noisy),
                 "passed": expressions(noisy) == ["1"],
                 "classification": "small_numeric_residuals_can_be_exactly_inconsistent"})

    p = y-x*x
    sample = {x: sp.Rational(1, 2), y: sp.Rational(1, 4)+sp.Rational(1, 100)}
    scale = sp.Rational(1, 10**12)
    def normalized_square(poly):
        return sp.cancel(poly.subs(sample)**2 / sum(sp.diff(poly, z).subs(sample)**2 for z in (x, y)))
    first, second = normalized_square(p), normalized_square(scale*p)
    rows.append({"id": CONTROL_IDS[9], "polynomial": str(p), "rescaled_by": str(scale),
                 "raw_absolute_residuals": [str(abs(poly.subs(sample))) for poly in (p, scale*p)],
                 "gradient_normalized_squared_residuals": [str(first), str(second)],
                 "passed": first == second and first > 0,
                 "classification": "gradient_normalization_blocks_scalar_shrinkage_in_this_control",
                 "qualification": "one regular point, not a general robustness guarantee or algorithm replication"})

    high_degree = remainder(y, [y-x**8], (y, x))
    rows.append({"id": CONTROL_IDS[10], "relation": "y-x**8", "requested_output": "y",
                 "reduced_output": str(high_degree), "retaining_y_output_multiplications": 0,
                 "retaining_x_repeated_squaring_multiplications": 3,
                 "passed": high_degree == x**8,
                 "classification": "fewer_stored_coordinates_can_require_more_output_computation",
                 "qualification": "local operation accounting, not an end-to-end runtime benchmark"})

    if [row["id"] for row in rows] != list(CONTROL_IDS):
        raise RuntimeError("Control coverage differs from declared schedule")
    for row in rows:
        row["passed"] = bool(row["passed"])
    return {"controls": rows, "count": len(rows), "passed": sum(bool(r["passed"]) for r in rows),
            "all_passed": all(bool(r["passed"]) for r in rows),
            "scope": "constructed algebra and a finite payload codec; not neural accuracy",
            "neural_compression_tested": False, "neural_speedup": None}
