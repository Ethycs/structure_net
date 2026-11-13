"""Posthoc semantic-grid identifiability audit; does not retune the stack decoder."""
import json
from itertools import product

import numpy as np
import sympy as sp
import torch

from .jspace_country_stack import DEFAULT_ROOT, PARENT, verify, checked, observed
from .jspace_causal_components import cpu_only
from .jspace_relation_study import file_hash, write_json


def semantic_grid_audit():
    a,b,u=sp.symbols("a b u")
    monomials=[1,a,b,a*b,u,a*u,b*u,a*b*u]
    points=list(product((0,1),repeat=3))
    missing=[(0,0,1),(1,1,0)]
    seen=[point for point in points if point not in missing]
    def row(point):
        return [sp.sympify(m).subs(dict(zip((a,b,u),point))) for m in monomials]
    design=sp.Matrix([row(point) for point in seen])
    witnesses=[(1-a)*(1-b)*u,a*b*(1-u)]
    generators=[a*a-a,b*b-b,u*u-u,*witnesses]
    basis=sp.groebner(generators,a,b,u,order="lex",domain=sp.QQ)
    truth=[[int(w.subs(dict(zip((a,b,u),point)))) for point in points] for w in witnesses]
    assert design.rank()==6 and len(design.nullspace())==2
    assert all(w.subs(dict(zip((a,b,u),point)))==0 for point in seen for w in witnesses)
    assert all(sum(values)==1 for values in truth)
    leading=[p.LM(order=basis.order).exponents for p in basis.polys]
    standard=[powers for powers in product(range(2),repeat=3)
              if not any(all(p>=l for p,l in zip(powers,lm)) for lm in leading)]
    assert len(standard)==6
    return {"variables":["a","b","u"],"country_bits":{"China":[0,0],"France":[0,1],"Germany":[1,0],"Russia":[1,1]},
            "attribute":"u=0 capital; u=1 nationality","full_semantic_states":points,"observed_training_states":seen,
            "unobserved_states":missing,"monomials":list(map(str,monomials)),"training_design_rank":6,
            "full_function_dimension":8,"unidentified_dimensions":2,"nullspace":[list(map(str,v)) for v in design.nullspace()],
            "missing_cell_indicator_polynomials":list(map(str,witnesses)),"witness_values_on_full_grid":truth,
            "training_support_groebner_basis":list(map(str,basis)),"training_quotient_dimension":6,
            "training_standard_monomials":[str(sp.prod(v**e for v,e in zip((a,b,u),powers))) for powers in standard],
            "interpretation":"An arbitrary function on the eight semantic country/attribute states has two unconstrained values after observing six states. This exact statement concerns the finite semantic grid, not an exact ideal of noisy neural scores."}


def main(root=DEFAULT_ROOT):
    cpu_only();verify(root);checked(root,"fit");checked(root,"capture")
    value=semantic_grid_audit()
    heads=torch.load(PARENT/"transport.pt",map_location="cpu",weights_only=True)
    measured={}
    for split in ("train","development","evaluation"):
        stack,attribute,country=observed(split,heads)
        x=stack[:,0,:3]
        witness=np.column_stack((attribute*x[:,0],(1-attribute)*(1-x.sum(1))))
        heldout=((country==0)&(attribute==1))|((country==3)&(attribute==0))
        measured[split]={"paired_observations":len(stack),"heldout_cell_observations":int(heldout.sum()),
                         "neural_witness_rms":np.sqrt(np.mean(witness**2,axis=0)).tolist(),
                         "neural_witness_rms_seen":np.sqrt(np.mean(witness[~heldout]**2,axis=0)).tolist(),
                         "neural_witness_rms_heldout":np.sqrt(np.mean(witness[heldout]**2,axis=0)).tolist() if heldout.any() else None}
    fitted=torch.load(root/"fit.pt",map_location="cpu",weights_only=True)
    payload=torch.load(root/"compressed_readout.pt",map_location="cpu",weights_only=True)
    # Export the predeclared linear comparator without changing the primary selection.
    linear=fitted["models"]["1"]
    reference={**payload,"powers":linear["powers"],"decoder_numerators":linear["numerators"][1:,:3].clone()}
    with (root/"linear_readout_reference.pt").open("xb") as f:torch.save(reference,f)
    result={"scope":"posthoc_algebraic_diagnosis_of_training_support_not_new_accuracy_replication","confirmed":False,
            "semantic_grid":value,"continuous_score_witnesses":measured,
            "primary_selection_unchanged":True,"linear_export_scope":"predeclared baseline exported after evaluation; not relabeled as selected candidate",
            "source_sha256":file_hash(__file__),"main_result_sha256":file_hash(root/"result.json"),
            "linear_reference_sha256":file_hash(root/"linear_readout_reference.pt")}
    write_json(root/"identifiability_audit.json",result)
    return result


if __name__=="__main__":print(json.dumps(main(),allow_nan=False))
