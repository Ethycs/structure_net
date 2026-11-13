"""Polynomial stack decoders and an explicitly finite country quotient codec."""
from contextlib import contextmanager
from itertools import combinations, combinations_with_replacement
import signal
import struct

import numpy as np
import torch

from .jspace_attention_transport import regress


DENOMINATOR = 2**24


def exponents(degree):
    if degree not in (1, 2):
        raise ValueError("Only the declared linear/quadratic families are supported")
    rows = [(0, 0, 0, 0)]
    for order in range(1, degree+1):
        for indices in combinations_with_replacement(range(4), order):
            rows.append(tuple(indices.count(i) for i in range(4)))
    return rows


def features(x, powers):
    return np.stack([np.prod(x**np.asarray(p)[None], axis=1) for p in powers], axis=1)


def reduce_boolean(numerators, powers):
    """Exact normal form modulo u^2-u, using a shared integer denominator."""
    canonical, positions = [], {}
    for p in powers:
        key = tuple(p[:3])+(min(p[3], 1),)
        if key not in positions:
            positions[key] = len(canonical)
            canonical.append(key)
    result = np.zeros(numerators.shape[:-1]+(len(canonical),), dtype=np.int64)
    for i,p in enumerate(powers):
        result[..., positions[tuple(p[:3])+(min(p[3], 1),)]] += numerators[..., i]
    return result, canonical


def fit_family(train, attribute, dev, dev_attribute, degree):
    powers = exponents(degree)
    x = np.column_stack((train[:, 0, :3], attribute))
    xd = np.column_stack((dev[:, 0, :3], dev_attribute))
    f, fd = features(x, powers), features(xd, powers)
    scale = np.maximum(f[:, 1:].std(0), 1e-8)
    target = train[:, 1:, :3].reshape(len(train), -1)
    targetd = dev[:, 1:, :3].reshape(len(dev), -1)
    fitted, metric = regress(f[:, 1:]/scale, target, fd[:, 1:]/scale, targetd)
    weight = fitted["weight"].numpy()/scale[:, None]
    intercept = fitted["prior"].numpy()-fitted["mean"].numpy()@fitted["weight"].numpy()
    coefficients = np.concatenate((intercept[None], weight), axis=0).T.reshape(train.shape[1]-1, 3, len(powers))
    if not np.isfinite(coefficients).all() or np.abs(coefficients).max()*DENOMINATOR > 2**50:
        raise ValueError("Coefficient quantization outside declared safe range")
    numbers = np.zeros((train.shape[1], 4, len(powers)), dtype=np.int64)
    numbers[1:, :3] = np.rint(coefficients*DENOMINATOR).astype(np.int64)
    for i in range(3):
        numbers[0, i, powers.index(tuple(int(k == i) for k in range(4)))] = DENOMINATOR
    numbers[:, 3] = -numbers[:, :3].sum(1)
    numbers[:, 3, 0] += DENOMINATOR
    reduced, canonical = reduce_boolean(numbers, powers)
    model = {"degree":degree, "denominator":DENOMINATOR, "powers":[list(p) for p in canonical],
             "numerators":torch.from_numpy(reduced), "unreduced_powers":[list(p) for p in powers],
             "unreduced_numerators":torch.from_numpy(numbers)}
    raw = (fd[:, 1:]/scale-fitted["mean"].numpy())@fitted["weight"].numpy()+fitted["prior"].numpy()
    quantized = decode(model, dev[:, 0, :3], dev_attribute)[:, 1:, :3].reshape(raw.shape)
    delta = float(np.sqrt(np.mean((raw-quantized)**2)))
    if delta > 1e-5:
        raise ValueError("Rationalization changed development predictions excessively")
    return model, {**metric, "development_quantization_rms":delta,
                   "maximum_absolute_coefficient":float(np.abs(coefficients).max())}


def decode(model, anchor, attribute):
    x = np.column_stack((anchor, attribute))
    coefficient = model["numerators"].numpy()/model["denominator"]
    return np.einsum("nf,lcf->nlc", features(x, model["powers"]), coefficient)


def decode_portable(payload, anchor_J_scores, attribute):
    encoder=payload["encoder"]; probe=encoder["probe"]
    z=(anchor_J_scores-encoder["mean"].numpy())/encoder["scale"].numpy()
    anchor=((z-probe["mean"].numpy())@probe["weight"].numpy()+probe["prior"].numpy())[:,:3]
    x=np.column_stack((anchor,attribute))
    later=np.einsum("nf,lcf->nlc",features(x,payload["powers"]),payload["decoder_numerators"].numpy()/payload["denominator"])
    first_three=np.concatenate((anchor[:,None],later),1)
    return np.concatenate((first_three,1-first_three.sum(-1,keepdims=True)),-1)


@contextmanager
def deadline(seconds):
    def expired(signum, frame):
        raise TimeoutError("Bounded symbolic cell timed out")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def graph_basis(model, layers, seconds=30):
    import sympy as sp
    with deadline(seconds):
        roots = sp.symbols("x_China x_France x_Germany u")
        ys = sp.symbols(" ".join(f"s{layer}_{name}" for layer in layers for name in ("China", "France", "Germany", "Russia")))
        raw = model["unreduced_numerators"].numpy().reshape(len(ys), -1)
        reduced = model["numerators"].numpy().reshape(len(ys), -1)
        def expressions(values, powers):
            monomials = [sp.prod(x**int(e) for x,e in zip(roots,p)) for p in powers]
            return [sum(sp.Rational(int(c), model["denominator"])*m for c,m in zip(row,monomials) if c) for row in values]
        raw_polys = expressions(raw, model["unreduced_powers"])
        reduced_polys = expressions(reduced, model["powers"])
        generators = [y-f for y,f in zip(ys,raw_polys)]+[roots[-1]**2-roots[-1]]
        basis = sp.groebner(generators, *ys, *roots, order="lex", domain=sp.QQ)
        normal = [basis.reduce(y)[1] for y in ys]
        if any(sp.expand(a-b) != 0 for a,b in zip(normal,reduced_polys)):
            raise ValueError("Symbolic normal forms and numerical decoder disagree")
        if any(basis.reduce(g)[1] != 0 for g in generators):
            raise ValueError("Generator membership failed")
        compressed = [p.as_expr() for p in basis.polys if not (p.as_expr().free_symbols & set(ys))]
        return {"status":"certified_for_rational_surrogate", "field":"QQ", "order":"lex",
                "variables":[str(x) for x in (*ys,*roots)], "generators":[str(g) for g in generators],
                "groebner_basis":[str(p.as_expr()) for p in basis.polys],
                "readout_normal_forms":dict(zip(map(str,ys),map(str,normal))),
                "compressed_basis":[str(g) for g in compressed], "retained_variables":list(map(str,roots)),
                "quotient_zero_dimensional":bool(basis.is_zero_dimensional),
                "all_generator_remainders_zero":True, "decoder_normal_forms_exact":True,
                "interpretation":"Exact graph elimination for the fitted rational polynomial surrogate; neural score fidelity is measured separately."}


def country_quotient(layers, seconds=30):
    import sympy as sp
    with deadline(seconds):
        a,b = sp.symbols("a b")
        base = sp.groebner([a*a-a,b*b-b],a,b,order="lex",domain=sp.QQ)
        indicators = [(1-a)*(1-b),(1-a)*b,a*(1-b),a*b]
        ys = sp.symbols(" ".join(f"c{l}_{i}" for l in layers for i in range(4)))
        generators = [y-indicators[i%4] for i,y in enumerate(ys)]+[a*a-a,b*b-b]
        full = sp.groebner(generators,*ys,a,b,order="lex",domain=sp.QQ)
        if any(sp.expand(full.reduce(y)[1]-indicators[i%4]) != 0 for i,y in enumerate(ys)):
            raise ValueError("Country stack decoder certificate failed")
        monomials = [1,a,b,a*b]
        matrices = []
        for variable in (a,b):
            cols = [sp.Poly(base.reduce(variable*m)[1],a,b).as_dict() for m in monomials]
            matrices.append(sp.Matrix([[col.get(ex,0) for col in cols] for ex in ((0,0),(1,0),(0,1),(1,1))]))
        ma,mb = matrices
        assert ma*mb == mb*ma and ma*ma == ma and mb*mb == mb
        actions = []
        for first,second in combinations(range(4),2):
            permutation = list(range(4)); permutation[first],permutation[second] = second,first
            fa = base.reduce(sum((p//2)*indicators[i] for i,p in enumerate(permutation)))[1]
            fb = base.reduce(sum((p%2)*indicators[i] for i,p in enumerate(permutation)))[1]
            for i,p in enumerate(permutation):
                point = {a:i//2,b:i%2}
                assert int(fa.subs(point))*2+int(fb.subs(point)) == p
            actions.append({"swap":[first,second],"permutation":permutation,"a_next":str(fa),"b_next":str(fb)})
        return {"status":"exact_constructed_country_algebra", "country_codes":{"China":[0,0],"France":[0,1],"Germany":[1,0],"Russia":[1,1]},
                "stack_layers":layers,"stack_variables":list(map(str,ys)),"stack_groebner_basis":[str(p.as_expr()) for p in full.polys],
                "compressed_groebner_basis":list(map(str,base)),"standard_monomials":[str(m) for m in monomials],
                "quotient_dimension":4,"indicator_decoders":list(map(str,indicators)),
                "multiplication_matrices":[[[int(v) for v in row] for row in matrix.tolist()] for matrix in matrices],
                "swaps":actions,"swap_checks":24,"neural_dynamics_proven":False,
                "interpretation":"Boolean country indicators are a declared finite semantic encoding, not equations inferred for continuous neural activations."}


def pack_labels(labels, compressed):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0) or np.any(labels>3):
        raise ValueError("Expected an N-by-L array of country indices 0..3")
    if not 0 < labels.shape[1] < 65536:
        raise ValueError("Unsupported stack length")
    bits = []
    repeated = 0
    for row in labels:
        same = bool(np.all(row == row[0]))
        repeated += same
        if compressed:
            bits.append(int(not same))
            values = row[:1] if same else row
        else:
            values = row
        for value in values:
            bits.extend([int(value)&1,int(value)>>1])
    header = struct.pack("<4sIH",b"GBC1" if compressed else b"GBR1",*labels.shape)
    return header+np.packbits(np.asarray(bits,dtype=np.uint8),bitorder="little").tobytes(), {"logical_bits":len(bits),"constant_worlds":repeated,"total_worlds":len(labels),"header_bytes":len(header)}


def unpack_labels(payload):
    if len(payload) < 10:
        raise ValueError("Truncated label payload")
    magic,n,l = struct.unpack("<4sIH",payload[:10])
    if magic not in (b"GBC1",b"GBR1") or l == 0 or n*l > 10**7:
        raise ValueError("Invalid label payload header")
    bits = np.unpackbits(np.frombuffer(payload[10:],dtype=np.uint8),bitorder="little")
    result = np.empty((n,l),dtype=np.uint8); offset = 0
    for i in range(n):
        mixed = True
        if magic == b"GBC1":
            if offset >= len(bits):
                raise ValueError("Truncated label payload")
            mixed = bool(bits[offset]); offset += 1
        count = l if mixed else 1
        if offset+2*count > len(bits):
            raise ValueError("Truncated label payload")
        values = bits[offset:offset+2*count:2]+2*bits[offset+1:offset+2*count:2]
        result[i] = values if mixed else values[0]
        offset += 2*count
    if len(bits)-offset > 7 or bits[offset:].any():
        raise ValueError("Noncanonical padding/trailing payload")
    return result
