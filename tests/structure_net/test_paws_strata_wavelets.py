import numpy as np
from experiments.structure_net.paws_strata_wavelets import carrier, deterministic_landmarks, stratum_keys, strata_diagnostics, varimax, wavelet_basis


def test_varimax_preserves_orthonormality():
    q,_=np.linalg.qr(np.random.default_rng(4).normal(size=(20,5))); rotated=varimax(q)
    assert np.linalg.norm(rotated.T@rotated-np.eye(5))<1e-10


def test_landmarks_are_deterministic_and_stratified():
    keys=np.asarray(["a"]*80+["b"]*20); mask=np.ones(100,dtype=bool)
    first=deterministic_landmarks(keys,mask,20)
    assert np.array_equal(first,deterministic_landmarks(keys,mask,20))
    assert set(keys[first])=={"a","b"}


def test_synthetic_strata_and_wavelet_lifecycle():
    rng=np.random.default_rng(8);n=96;labels=np.arange(n)%2;partition=np.arange(n)%4
    atlas={"partition":partition,"embedding":rng.normal(size=(n,40)).astype(np.float32),"labels":labels,"a_prediction":labels.copy(),"b_prediction":np.where(np.arange(n)%3,labels,1-labels),"c_prediction":np.where(np.arange(n)%7,labels,1-labels)}
    rows=[{"sentence1":f"alpha token {i}","sentence2":f"token alpha {i}"} for i in range(n)]
    x,probability,success=carrier(atlas,rows);keys=stratum_keys(labels,success);construction=np.isin(partition,[1,2]);audit=partition==3
    diagnostics=strata_diagnostics(x,keys,construction,audit)
    assert diagnostics["claim_boundary"].startswith("operational")
    indices=deterministic_landmarks(keys,construction,32);metrics,arrays=wavelet_basis(x,probability,keys,indices)
    assert metrics["orthogonality_error"]<1e-5
    assert arrays["basis"].shape==(32,32)
