from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from experiments.structure_net import jspace_accuracy_oracles as oracle
from experiments.structure_net.jspace_quadratic_qwen import FoldedReadout


def test_handoff_floor_equals_discarded_component_and_error_decomposition():
    torch.manual_seed(4)
    u = torch.linalg.qr(torch.randn(6, 3, dtype=torch.float64)).Q
    v = torch.linalg.qr(torch.randn(6, 3, dtype=torch.float64)).Q
    a = oracle.affine_anchor(u, torch.randn(1, 6, dtype=torch.float64))
    b = oracle.affine_anchor(v, torch.randn(1, 6, dtype=torch.float64))
    h = torch.randn(9, 6, dtype=torch.float64)
    direct, bridged = oracle.handoff_coordinates(h, u, a, v, b)
    reconstructed = a+((h-a) @ u) @ u.T
    torch.testing.assert_close(bridged, (reconstructed-b) @ v)
    torch.testing.assert_close(bridged-direct, -((h-a)-((h-a) @ u) @ u.T) @ v)
    assert (bridged-direct).norm() > .1
    student = bridged+.2*torch.randn_like(bridged)
    stats = oracle.finish_errors(oracle.error_sums(direct, bridged, student))
    assert abs(stats["energy_identity_residual"]) < 1e-12
    assert stats["handoff_normalized_mse"] > 0
    assert stats["positions"] == 9


def test_zero_local_loss_does_not_remove_handoff_floor():
    u = torch.tensor([[1.], [0.]], dtype=torch.float64)
    v = torch.tensor([[0.], [1.]], dtype=torch.float64)
    h = torch.tensor([[2., 3.]], dtype=torch.float64)
    a = b = torch.zeros(1, 2, dtype=torch.float64)
    direct, bridged = oracle.handoff_coordinates(h, u, a, v, b)
    stats = oracle.finish_errors(oracle.error_sums(direct, bridged, bridged))
    assert stats["student_normalized_mse"] == 0
    assert stats["handoff_normalized_mse"] == stats["total_normalized_mse"] == 1


def test_final_oracle_folded_head_matches_explicit_projected_native_readout():
    torch.manual_seed(8)
    u = torch.linalg.qr(torch.randn(5, 3, dtype=torch.float64)).Q
    a = oracle.affine_anchor(u, torch.randn(1, 5, dtype=torch.float64))
    h, e, gain = torch.randn(4,5,dtype=torch.float64), torch.randn(7,5,dtype=torch.float64), torch.randn(5,dtype=torch.float64)
    head = FoldedReadout(e, u, a, gain, .03, trainable=False, parameter_dtype=torch.float64)
    z = (h-a) @ u
    reconstructed = a+z @ u.T
    expected = F.linear(reconstructed/(reconstructed.square().mean(-1,keepdim=True)+.03).sqrt()*gain,e)
    torch.testing.assert_close(head(z), expected)
    labels = torch.tensor([1,3,6])
    small = FoldedReadout(e[labels],u,a,gain,.03,trainable=False,parameter_dtype=torch.float64)
    state = deepcopy(head.state_dict());state["weight"]=state["weight"][labels];state["bias"]=state["bias"][labels]
    small.load_state_dict(state,strict=True)
    torch.testing.assert_close(small(z), head(z)[:,labels])


def cases():
    return [dict(identity="a",family="x",answer=1,choices=["A","B"],input_ids=[2,3],offset=0,stop=2,split="train"),
            dict(identity="b",family="y",answer=0,choices=["A","B","C"],input_ids=[1],offset=2,stop=3,split="development")]


def test_split_and_exclusive_guards_precede_any_source_or_cuda_access(tmp_path, monkeypatch):
    monkeypatch.setattr(oracle.recipe,"verify",lambda *_:pytest.fail("Source access before guard"))
    with pytest.raises(ValueError,match="Only train/development"):
        oracle.run(tmp_path/"absent",tmp_path/"no_source",split="evaluation")
    with pytest.raises(FileExistsError):
        oracle.run(tmp_path,tmp_path/"no_source")
    with pytest.raises(ValueError,match="layer"):
        oracle.run(tmp_path/"absent",tmp_path/"no_source",layers=[36])
    root=oracle.create_output(tmp_path/"new")
    with pytest.raises(FileExistsError):oracle.create_output(root)
    assert oracle.select_cases(cases(),"development")[0]["identity"]=="b"
    with pytest.raises(ValueError):oracle.select_cases(cases(),"evaluation")


def test_scoring_only_valid_choice_columns_and_counting():
    records=oracle.score_records(cases(),torch.tensor([[1.,2.,999.],[4.,3.,2.]]))
    assert [r["prediction"] for r in records]==[1,0]
    assert oracle.summarize(records)["accuracy"]==1
    with pytest.raises(ValueError):oracle.score_records(cases(),torch.full((2,3),float("nan")))


def test_padding_mask_and_last_position_selection():
    import numpy as np
    bank=np.arange(6,dtype=np.float32).reshape(3,2)
    z,pos,mask,last=oracle._coordinates_batch(bank,cases(),"cpu")
    assert z.shape==(2,2,2) and last.tolist()==[1,0]
    assert mask[0,0].tolist()==[[True,False],[True,True]]
    assert mask[1,0].tolist()==[[True,False],[False,False]]
    torch.testing.assert_close(z[1,1],torch.zeros(2))
