"""Polynomial normal-form tensors and a retained-neuron residual implementation.

The covered relations are monic graph generators y_i-p_i(z). Their distinct
leading variables give a Groebner basis by the relatively-prime-leading-term
criterion. Neural relation defects remain empirical errors, never exact zeros.
"""
from itertools import combinations_with_replacement
import torch
from torch import nn
from torch.nn import functional as F


def features(z, degree):
    columns = [torch.ones_like(z[..., :1]), z]
    if degree == 2:
        columns.append(torch.stack([z[..., i] * z[..., j]
                                    for i, j in combinations_with_replacement(range(z.shape[-1]), 2)], -1))
    elif degree != 1:
        raise ValueError("Only declared degree-one/two maps are supported")
    return torch.cat(columns, -1)


def fit_coefficients(z, anchors, degree, ridge=0.01):
    x = features(z[:, anchors], degree).double()
    penalty = torch.eye(x.shape[1], device=x.device, dtype=x.dtype) * ridge
    penalty[0, 0] = 0
    coefficient = torch.linalg.solve(x.T @ x / len(x) + penalty, x.T @ z.double() / len(x)).T
    # The rational graph used by the compiler has this declared denominator.
    return (coefficient * 2**20).round().div(2**20).float()


def anchor_indices(jacobian, scale, count):
    weighted = jacobian.float() * scale
    weighted = weighted / weighted.norm(dim=1, keepdim=True).clamp_min(1e-12)
    _, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
    remaining = vh[:count].clone()
    selected = []
    for _ in range(count):
        norms = remaining.square().sum(0)
        if selected:
            norms[selected] = -1
        index = int(norms.argmax())
        if norms[index] < 1e-12:
            raise ValueError("Insufficient task-Jacobian rank for the fixed anchor count")
        q = remaining[:, index] / norms[index].sqrt()
        remaining -= q[:, None] * (q @ remaining)[None, :]
        selected.append(index)
    return selected, singular, vh[:count]


def graph_certificate(coefficient, anchors, removed, degree):
    """Certify the triangular form, plus a representative exact CAS audit."""
    import sympy as sp
    if set(anchors) & set(removed) or len(set(removed)) != len(removed):
        raise ValueError("Graph outputs and independent anchors must be disjoint")
    k = len(anchors)
    expected = 1 + k + (k * (k + 1) // 2 if degree == 2 else 0)
    if coefficient.shape[1] != expected or not torch.isfinite(coefficient).all():
        raise ValueError("Polynomial coefficient layout mismatch")
    z = sp.symbols(f"z0:{k}")
    monomials = [sp.Integer(1), *z]
    if degree == 2:
        monomials += [z[i] * z[j] for i, j in combinations_with_replacement(range(k), 2)]
    # Full graph has at most 1024 outputs. The structural criterion applies to
    # every row; constructing all pairwise S-polynomials is unnecessary.
    selected = removed[:2]
    ys = sp.symbols(f"y0:{len(selected)}")
    numbers = (coefficient[selected].double().cpu() * 2**20).round().long()
    polys = [sum(sp.Rational(int(c), 2**20) * m for c, m in zip(row, monomials) if c)
             for row in numbers]
    g = sp.groebner([y-p for y,p in zip(ys,polys)], *ys, *z, domain=sp.QQ, order="lex")
    assert all(sp.expand(g.reduce(y)[1] - p) == 0 for y,p in zip(ys,polys))
    return {"field": "QQ", "coefficient_denominator": 2**20,
            "generator_form": "y_i-p_i(z)", "generator_count": len(removed),
            "monomial_order": "lex; all eliminated y variables before anchor z variables",
            "distinct_monic_leading_variables": True,
            "buchberger_product_criterion": True, "representative_cas_checks": len(selected),
            "standard_monomials": "all monomials in the retained anchor variables; infinite dimensional",
            "compiled_degree": degree, "packed_feature_count": expected,
            "polynomial_identity": "W_keep a_keep + W_elim p(z) + W_elim (a_elim-p(z))",
            "neural_relation_defect": "last term is measured; not assumed zero or silently evaluated via the original layer",
            "certificate_scope": "exact for the rational graph surrogate, not the pretrained model"}


def normal_form_split(groebner, expression, retained_monomials):
    """Exact polynomial core + overflow + relation defect, even off the variety."""
    import sympy as sp
    quotients, normal = groebner.reduce(sp.expand(expression))
    poly = sp.Poly(normal, *groebner.gens, domain=sp.QQ)
    coefficients = [poly.coeff_monomial(m) for m in retained_monomials]
    covered = sum(c*m for c,m in zip(coefficients, retained_monomials))
    overflow = sp.expand(normal-covered)
    defect = sum(q*g.as_expr() for q,g in zip(quotients, groebner.polys))
    if sp.expand(expression-covered-overflow-defect) != 0:
        raise ValueError("Normal-form residual identity failed")
    return coefficients, overflow, defect


def product_tensor(groebner, monomials):
    """C[k,i,j] are structure constants; retain overflow/defect per product."""
    import sympy as sp
    n = len(monomials)
    tensor = sp.MutableDenseNDimArray.zeros(n,n,n)
    residual = {}
    for i,a in enumerate(monomials):
        for j,b in enumerate(monomials):
            coeff, overflow, defect = normal_form_split(groebner, a*b, monomials)
            for k,c in enumerate(coeff):
                tensor[k,i,j] = c
            residual[i,j] = (overflow, defect)
    return tensor, residual


class PolynomialResidualMLP(nn.Module):
    """Compute uncovered channels and a polynomial tensor for removed channels.

No original activation of a removed channel is evaluated. A nonlinear relation
defect is not claimed to be exactly corrected: combined task quality is tested.
"""
    def __init__(self, original, fit, removed, mode="quadratic", compiled=True):
        super().__init__()
        if mode not in ("quadratic", "linear", "zero"):
            raise ValueError(mode)
        self.mode, self.compiled = mode, compiled
        device, dtype = original.down_proj.weight.device, original.down_proj.weight.dtype
        compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        n = original.down_proj.weight.shape[1]
        anchors = list(fit["anchors"])
        if not removed or set(removed) & set(anchors):
            raise ValueError("Remove nonanchor channels only")
        keep = [i for i in range(n) if i not in set(removed)]
        positions = [keep.index(i) for i in anchors]
        mean, scale = fit["mean"].to(device=device,dtype=compute_dtype), fit["scale"].to(device=device,dtype=compute_dtype)
        self.register_buffer("anchor_positions", torch.tensor(positions,device=device))
        self.register_buffer("anchor_mean", mean[anchors])
        self.register_buffer("anchor_scale", scale[anchors])
        self.degree = 2 if mode == "quadratic" else 1
        self.compute_dtype = compute_dtype
        self.gate_proj = self._slice(original.gate_proj, keep, 0)
        self.up_proj = self._slice(original.up_proj, keep, 0)
        self.act_fn = original.act_fn
        full_weight = original.down_proj.weight.detach().to(compute_dtype)
        kept_weight = full_weight[:, keep].clone()
        dropped_weight = full_weight[:, removed]
        bias = (torch.zeros(full_weight.shape[0],device=device,dtype=compute_dtype)
                if original.down_proj.bias is None else original.down_proj.bias.detach().to(compute_dtype).clone())
        coefficients = fit["coefficients"][str(self.degree)][removed].to(device=device,dtype=compute_dtype)
        if mode == "zero":
            coefficients = coefficients*0
            coefficients[:,0] = -mean[removed]/scale[removed]
        if compiled:
            combined = (dropped_weight*scale[removed]) @ coefficients
            bias += dropped_weight @ mean[removed] + combined[:,0]
            linear = combined[:,1:len(anchors)+1]/scale[anchors]
            kept_weight[:,positions] += linear
            bias -= linear @ mean[anchors]
            quadratic = combined[:,len(anchors)+1:]
            self.register_buffer("quadratic_weight", quadratic)
        else:
            self.register_buffer("dropped_weight", dropped_weight)
            self.register_buffer("coefficient", coefficients)
            self.register_buffer("dropped_mean", mean[removed])
            self.register_buffer("dropped_scale", scale[removed])
        self.down_proj = nn.Linear(len(keep),full_weight.shape[0],bias=False,device=device,dtype=dtype)
        self.down_proj.weight = nn.Parameter(kept_weight.to(dtype), requires_grad=False)
        self.register_buffer("output_bias", bias)
        self.requires_grad_(False)

    @staticmethod
    def _slice(module, indices, dim):
        if dim != 0:
            raise ValueError("Only input-neuron rows are sliced")
        result = nn.Linear(module.in_features,len(indices),bias=module.bias is not None,
                           device=module.weight.device,dtype=module.weight.dtype)
        result.weight = nn.Parameter(module.weight.detach()[indices].clone(),requires_grad=False)
        if module.bias is not None:
            result.bias = nn.Parameter(module.bias.detach()[indices].clone(),requires_grad=False)
        return result

    def forward(self, x):
        hidden = self.act_fn(self.gate_proj(x))*self.up_proj(x)
        z = (hidden[...,self.anchor_positions].to(self.compute_dtype)-self.anchor_mean)/self.anchor_scale
        phi = features(z,self.degree)
        value = self.down_proj(hidden).to(self.compute_dtype) + self.output_bias
        if self.compiled:
            if self.degree == 2:
                value += F.linear(phi[...,len(self.anchor_positions)+1:],self.quadratic_weight)
        else:
            predicted = F.linear(phi,self.coefficient)*self.dropped_scale+self.dropped_mean
            value += F.linear(predicted,self.dropped_weight)
        return value.to(x.dtype)
