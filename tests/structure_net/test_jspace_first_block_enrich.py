"""Locked observable directions coexist with representable native updates."""
import pytest
import torch

from experiments.structure_net.jspace_first_block_enrich import enrich_basis, projection_metrics, sample_training


def test_reachable_enrichment_preserves_exact_j_columns_and_repairs_input_and_update():
    old=torch.eye(12,dtype=torch.float64)[:,:6]
    mean=torch.zeros(1,12,dtype=torch.float64)
    h0=torch.zeros(6,12,dtype=torch.float64);delta=torch.zeros_like(h0)
    zero=torch.tensor([True,False,False,True,False,False])
    h0[zero,6]=2;delta[zero,7]=3
    h0[~zero,8]=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)
    delta[~zero,9]=torch.tensor([4.,3.,2.,1.],dtype=torch.float64)
    result,report=enrich_basis(old,mean,h0,h0+delta,zero,functional_columns=2)
    assert torch.equal(result[:,:2],old[:,:2]) and result.shape==old.shape
    assert report["strong_columns_bit_identical"] and report["additional_locked_mean_directions"]==2
    torch.testing.assert_close(result.T@result,torch.eye(6,dtype=torch.float64),rtol=0,atol=1e-12)
    before=projection_metrics(old,mean,h0,h0+delta);after=projection_metrics(result,mean,h0,h0+delta)
    assert before["input_centered_energy_lost"]==before["native_update_energy_lost"]==1.
    assert after["input_centered_energy_lost"]<1e-20 and after["native_update_energy_lost"]<1e-20


def test_rank_deficient_moments_use_reported_original_weak_completion():
    old=torch.eye(8,dtype=torch.float64)[:,:6];mean=torch.zeros(1,8,dtype=torch.float64)
    h=torch.zeros(4,8,dtype=torch.float64);h[:,0]=1.;delta=torch.zeros_like(h);delta[:,6]=1.
    result,report=enrich_basis(old,mean,h,h+delta,torch.tensor([True,False,True,False]),functional_columns=1)
    assert result.shape==old.shape and report["old_weak_completion"]>0
    assert torch.equal(result[:,:1],old[:,:1])


def test_fp32_rotated_geometry_preserves_j_core_and_orthogonality():
    torch.manual_seed(984)
    rotation=torch.linalg.qr(torch.randn(32,32)).Q
    old=rotation[:,:16];mean=torch.zeros(1,32)
    zero=torch.tensor([True,False,False,True,False,False])
    x=torch.zeros(6,32);d=torch.zeros_like(x)
    x[zero,20]=1.;d[zero,22]=2.;x[~zero,21]=3.;d[~zero,23]=4.
    h0=x@rotation.T;h1=h0+d@rotation.T
    result,report=enrich_basis(old,mean,h0,h1,zero,functional_columns=4)
    assert torch.equal(result[:,:4],old[:,:4])
    assert report["orthogonality_max"]<3e-5
    metrics=projection_metrics(result,mean,h0,h1)
    assert metrics["input_centered_energy_lost"]<1e-10
    assert metrics["native_update_energy_lost"]<1e-10


def test_training_sampler_rejects_missing_train_population_without_touching_banks(tmp_path):
    items=[dict(split="development",family=f) for f in ("arc","boolq","hellaswag","winogrande")]
    with pytest.raises(ValueError,match="training-only"):
        sample_training(items,tmp_path,per_family=1)
