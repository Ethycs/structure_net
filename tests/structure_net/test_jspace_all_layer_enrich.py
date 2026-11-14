"""All-layer extension keeps the fixed objective and reads matching native banks."""
import numpy as np
import pytest
import torch

from experiments.structure_net.jspace_all_layer_enrich import (
    diagnostic_groups,fit_layer,sample_boundary,training_selection,
)


def test_training_selections_and_variable_layer_banks_exclude_development(tmp_path):
    items=[];offset=0
    for family in ("arc","boolq","hellaswag","winogrande"):
        for index,split in enumerate(("train","train","development")):
            items.append(dict(identity=f"{family}{index}",family=family,split=split,
                input_ids=[1,2,3,4],offset=offset,stop=offset+4));offset+=4
    fit=training_selection(items,per_family=1,positions=4)
    check=training_selection(items,start=1,per_family=1,positions=4)
    assert not set(fit["identities"])&set(check["identities"])
    assert len(fit["indices"])==16 and sum(fit["position0"])==4
    original=torch.arange(offset*6).reshape(offset,6).float()
    for case in items:
        if case["split"]=="development":original[case["offset"]:case["stop"]]=torch.nan
    for layer in (7,8):
        values=(original+layer).bfloat16()
        path=tmp_path/f"boundary_{layer:02d}.npy";np.save(path,values.view(torch.int16).numpy())
        sampled=sample_boundary(path,fit)
        assert torch.isfinite(sampled).all()
        torch.testing.assert_close(sampled,values[fit["indices"]].float(),rtol=0,atol=0)


def test_layer_fit_preserves_all_calibrated_head_tensors_and_locked_j_columns():
    old=torch.eye(12,dtype=torch.float64)[:,:6]
    value=dict(basis=old,means=torch.zeros(1,12,dtype=torch.float64),rank=6,head_dim=2,
        rope_pairs=torch.tensor([[0]]),value_basis=torch.eye(4,dtype=torch.float64)[None,:,:2],
        metadata={"retained_functional_directions":2,"captured_normalized_gram_energy":.99})
    x=torch.zeros(6,12,dtype=torch.float64);update=torch.zeros_like(x)
    zero=torch.tensor([True,False,False,True,False,False])
    x[zero,6]=2;update[zero,7]=3;x[~zero,8]=1;update[~zero,9]=1
    candidate,report=fit_layer(value,x,x+update,zero)
    assert torch.equal(candidate["basis"][:,:2],old[:,:2])
    assert candidate["means"] is value["means"]
    assert candidate["rope_pairs"] is value["rope_pairs"] and candidate["value_basis"] is value["value_basis"]
    assert candidate["head_dim"]==2 and candidate["rank"]==6
    assert candidate["metadata"]["enrichment"] is report


def test_non_position0_diagnostics_expose_error_hidden_by_large_bos_energy():
    basis=torch.eye(4)[:,:2];mean=torch.zeros(1,4)
    h0=torch.tensor([[1000.,0,0,0],[1.,0,0,0]])
    delta=torch.tensor([[1000.,0,0,0],[0,0,1.,0]])
    groups=diagnostic_groups(basis,mean,h0,h0+delta,torch.tensor([True,False]),torch.eye(4),None)
    assert groups["all_positions"]["native_update_energy_lost"]<1e-5
    assert groups["non_position0"]["native_update_energy_lost"]==1.
    assert "update_G1_residual_ratio" not in groups["non_position0"]


def test_all_layer_population_cannot_draw_extra_cases_from_development():
    with pytest.raises(ValueError,match="training-only"):
        training_selection([dict(identity="x",family="arc",split="development")],per_family=1)
