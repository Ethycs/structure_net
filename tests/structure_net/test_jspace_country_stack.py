from itertools import product
import numpy as np
import pytest
import torch

from experiments.structure_net.jspace_country_stack_algebra import (
    DENOMINATOR, exponents, features, reduce_boolean, fit_family, decode,
    graph_basis, country_quotient, pack_labels, unpack_labels,
)


def test_boolean_normal_form_is_exact_only_on_declared_attribute_domain():
    powers=exponents(2); values=np.arange(2*4*len(powers),dtype=np.int64).reshape(2,4,len(powers))
    reduced,canonical=reduce_boolean(values,powers)
    rng=np.random.default_rng(6); x=rng.normal(size=(16,4)); x[:,-1]=rng.integers(0,2,16)
    assert len(powers)==15 and len(canonical)==14
    assert np.allclose(features(x,powers)@values.reshape(8,-1).T,features(x,canonical)@reduced.reshape(8,-1).T,atol=1e-12)
    x[:,-1]=.4
    assert not np.allclose(features(x,powers)@values.reshape(8,-1).T,features(x,canonical)@reduced.reshape(8,-1).T)


def test_known_quadratic_country_stack_and_symbolic_decoder_roundtrip():
    rng=np.random.default_rng(12)
    def data(n):
        anchor=rng.uniform(.05,.2,size=(n,3)); u=rng.integers(0,2,n)
        initial=np.column_stack((anchor,1-anchor.sum(1)))
        nxt=anchor.copy(); nxt[:,0]+=anchor[:,1]*anchor[:,2]; nxt[:,1]+=.1*u*anchor[:,0]
        later=np.column_stack((nxt,1-nxt.sum(1)))
        return np.stack((initial,later),1),u
    train,u=data(500); dev,ud=data(80); test,ut=data(80)
    model,metric=fit_family(train,u,dev,ud,2)
    actual=decode(model,test[:,0,:3],ut)
    assert np.sqrt(np.mean((actual-test)**2))<1e-4
    assert metric["development_quantization_rms"]<1e-5
    basis=graph_basis(model,[27,28])
    assert basis["decoder_normal_forms_exact"] and basis["all_generator_remainders_zero"]
    assert basis["compressed_basis"]==["u**2 - u"]
    assert basis["quotient_zero_dimensional"] is False


def test_constructed_finite_country_quotient_and_swaps():
    value=country_quotient([27,28])
    assert value["compressed_groebner_basis"]==["a**2 - a","b**2 - b"]
    assert value["quotient_dimension"]==4 and value["swap_checks"]==24
    assert len(value["stack_groebner_basis"])==10
    assert value["neural_dynamics_proven"] is False


def test_lossless_country_codec_exhaustive_mixed_and_constant_trajectories():
    labels=np.array(list(product(range(4),repeat=3)),dtype=np.uint8)
    for compressed in (True,False):
        payload,info=pack_labels(labels,compressed)
        assert np.array_equal(unpack_labels(payload),labels)
        assert info["constant_worlds"]==4
    constant=np.repeat(np.arange(4)[:,None],9,1)
    packed,info=pack_labels(constant,True)
    assert info["logical_bits"]==12
    assert np.array_equal(unpack_labels(packed),constant)


@pytest.mark.parametrize("payload",[b"",b"invalid-header",b"GBC1\x01\0\0\0\x09\0",b"GBR1\x01\0\0\0\x01\0\xff"])
def test_country_codec_rejects_corruption(payload):
    with pytest.raises(ValueError):
        unpack_labels(payload)


def test_codec_rejects_non_country_labels():
    with pytest.raises(ValueError):
        pack_labels(np.array([[4,0]]),True)
    with pytest.raises(ValueError):
        pack_labels(np.array([[.5,0.]]),True)


def test_serialization_keeps_exact_rational_coefficients(tmp_path):
    powers=exponents(1)
    model={"numerators":torch.arange(40,dtype=torch.int64).reshape(2,4,5),"powers":powers,"denominator":DENOMINATOR}
    path=tmp_path/"codec.pt"; torch.save(model,path)
    restored=torch.load(path,map_location="cpu",weights_only=True)
    x=np.array([[.2,.3,.1],[.1,.4,.2]]); u=np.array([0.,1.])
    assert np.array_equal(decode(model,x,u),decode(restored,x,u))
