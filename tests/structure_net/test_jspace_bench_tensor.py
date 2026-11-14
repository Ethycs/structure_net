from itertools import product
from types import SimpleNamespace
import pytest
import sympy as sp
import torch
from torch import nn
from torch.nn import functional as F

from experiments.structure_net.jspace_bench_data import normalize
from experiments.structure_net.jspace_bench_tensor import CONFIG, DATASETS, comparison, dev_pass, digest, redundancy_report
from experiments.structure_net.jspace_groebner_tensor import (
    features, normal_form_split, product_tensor, PolynomialResidualMLP, graph_certificate,
)


def test_normal_form_keeps_relation_defect_off_the_variety_and_overflow():
    x,y=sp.symbols("x y")
    g=sp.groebner([y-x*x],y,x,domain=sp.QQ,order="lex")
    c,overflow,defect=normal_form_split(g,y*y+2*x,[sp.Integer(1),x,x*x])
    assert c==[0,2,0] and overflow==x**4
    assert sp.expand(2*x+overflow+defect-(y*y+2*x))==0
    assert defect.subs({x:2,y:5})==9


def test_boolean_quotient_tensor_has_exact_product_factorization():
    a,b,t=sp.symbols("a b t")
    g=sp.groebner([a*a-a,b*b-b],a,b,domain=sp.QQ)
    full,residual=product_tensor(g,[sp.Integer(1),a,b,a*b])
    bit,_=product_tensor(sp.groebner([t*t-t],t,domain=sp.QQ),[sp.Integer(1),t])
    exponents=[(0,0),(1,0),(0,1),(1,1)]
    for k,i,j in product(range(4),repeat=3):
        assert full[k,i,j]==bit[exponents[k][0],exponents[i][0],exponents[j][0]]*bit[exponents[k][1],exponents[i][1],exponents[j][1]]
    for overflow,defect in residual.values():
        assert overflow==0
        assert all(defect.subs({a:x,b:y})==0 for x,y in product((0,1),repeat=2))


@pytest.mark.parametrize("mode",["quadratic","linear","zero"])
def test_compiled_residual_equals_explicit_neuron_substitution_with_gradients(mode):
    torch.manual_seed(17)
    original=SimpleNamespace(gate_proj=nn.Linear(7,12,bias=True).double(),up_proj=nn.Linear(7,12,bias=True).double(),
                             down_proj=nn.Linear(12,7,bias=True).double(),act_fn=F.silu)
    fit={"anchors":[0,3,7],"mean":torch.randn(12,dtype=torch.double),"scale":torch.rand(12,dtype=torch.double)+.5,
         "coefficients":{"1":torch.randn(12,4,dtype=torch.double)/16,"2":torch.randn(12,10,dtype=torch.double)/16}}
    removed=[1,5,10]
    compiled=PolynomialResidualMLP(original,fit,removed,mode)
    uncompiled=PolynomialResidualMLP(original,fit,removed,mode,False)
    x=torch.randn(2,4,7,dtype=torch.double,requires_grad=True)
    a=F.silu(original.gate_proj(x))*original.up_proj(x)
    z=(a[...,fit["anchors"]]-fit["mean"][fit["anchors"]])/fit["scale"][fit["anchors"]]
    degree=2 if mode=="quadratic" else 1
    reconstructed=features(z,degree)@fit["coefficients"][str(degree)][removed].T
    reconstructed=reconstructed*fit["scale"][removed]+fit["mean"][removed]
    replaced=a.clone();replaced[...,removed]=0 if mode=="zero" else reconstructed
    target=original.down_proj(replaced)
    actual=compiled(x)
    assert torch.allclose(actual,target,atol=1e-11,rtol=1e-11)
    assert torch.allclose(actual,uncompiled(x),atol=1e-11,rtol=1e-11)
    actual_grad=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
    target_grad=torch.autograd.grad(target.square().sum(),x)[0]
    assert torch.allclose(actual_grad,target_grad,atol=1e-10,rtol=1e-10)
    assert compiled.gate_proj.out_features==9 and compiled.down_proj.in_features==9


def test_graph_certificate_is_exact_surrogate_and_rejects_anchor_elimination():
    coefficient=torch.arange(12*10,dtype=torch.float).reshape(12,10)/16
    result=graph_certificate(coefficient,[0,3,7],[1,5,10],2)
    assert result["buchberger_product_criterion"] and result["generator_count"]==3
    with pytest.raises(ValueError,match="disjoint"):
        graph_certificate(coefficient,[0,3,7],[3],2)


def test_redundancy_counts_reuse_not_semantic_accuracy_or_ideal_dimension():
    errors=torch.tensor([[.1,.7,.1],[.2,.1,.7],[.3,.1,.7],[.4,.7,.7]])
    result=redundancy_report(errors,errors,CONFIG)
    assert result["total_task_occurrences"]==7
    assert result["distinct_shared_elements"]==3
    assert result["redundant_occurrences"]==4
    assert result["shared_by_at_least_two_tasks"]==2
    assert result["shared_by_all_tasks"]==1


def test_label_mapping_handles_numeric_arc_keys_and_boolean_targets():
    row={"question":"q","choices":{"text":["first","second"],"label":["1","2"]},"answerKey":"2"}
    assert normalize("arc",row)[2]==1
    assert normalize("boolq",{"passage":"p","question":"q","answer":False})[2]==0


def test_gate_rejects_unpaired_rows_and_fingerprint_changes_with_scope():
    rows=[{"identity":str(i),"family":f,"answer":0,"prediction":0,"logits":[2.,0.]} for i,f in enumerate(DATASETS)]
    assert dev_pass(rows,rows)
    with pytest.raises(ValueError,match="Unpaired"):
        comparison(rows,list(reversed(rows)))
    assert digest(CONFIG)!=digest({**CONFIG,"anchors":CONFIG["anchors"]+1})
