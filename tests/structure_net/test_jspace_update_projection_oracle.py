from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.structure_net import jspace_update_projection_oracle as oracle


def native_banks(tmp_path):
    # The excluded training row is deliberately nonfinite. Reading only the
    # frozen development slices is part of the oracle's contract.
    h0 = torch.tensor([[float("nan")]*4, [1.,2.,3.,4.], [2.,3.,4.,5.],
                       [3.,4.,5.,6.], [4.,5.,6.,7.], [5.,6.,7.,8.]])
    update = torch.tensor([[0.]*4, [1.,2.,3.,4.], [2.,1.,4.,3.],
                          [3.,4.,1.,2.], [4.,3.,2.,1.], [1.,1.,2.,2.]])
    h1 = h0 + update
    for name, value in (("input", h0), ("target", h1)):
        np.save(tmp_path / f"{name}.npy", value.bfloat16().view(torch.int16).numpy())
    items = [dict(identity="train", family="tiny", split="train", input_ids=[9], choices=["A","B"], answer=0, offset=0, stop=1),
        dict(identity="dev0", family="tiny", split="development", input_ids=[8,7], choices=["A","B"], answer=1, offset=1, stop=3),
        dict(identity="dev1", family="tiny", split="development", input_ids=[3,5,4], choices=["A","B"], answer=0, offset=3, stop=6)]
    return h0, h1, items, items[1:], torch.eye(4)[:, :2]


def test_update_projection_is_closest_fixed_subspace_update():
    torch.manual_seed(11)
    h0, h1 = torch.randn(7, 6), torch.randn(7, 6)
    basis = torch.linalg.qr(torch.randn(6, 3)).Q
    projected = oracle.project_update(h0, h1, basis)
    residual = h1 - projected
    torch.testing.assert_close(residual @ basis, torch.zeros(7, 3), atol=8e-7, rtol=0)
    alternative = projected + torch.randn(7, 3) @ basis.T
    assert (alternative-h1).square().sum() > residual.square().sum()
    torch.testing.assert_close(oracle.project_update(h0, h1, torch.eye(6)), h1)
    with pytest.raises(ValueError): oracle.project_update(h0, h1[:, :-1], basis)


def test_dev_only_virtual_addresses_and_both_cache_arms(tmp_path):
    h0, h1, items, cases, basis = native_banks(tmp_path)
    arguments = (cases, items, tmp_path / "input.npy", tmp_path / "target.npy", basis)
    control, virtual, native = oracle.reconstruct(*arguments, arm="native_boundary", chunk_tokens=1)
    projected, other_virtual, compressed = oracle.reconstruct(*arguments, arm="projected_update", chunk_tokens=2)
    torch.testing.assert_close(control, h1[1:], atol=0, rtol=0)
    torch.testing.assert_close(projected, oracle.project_update(h0[1:], h1[1:], basis), atol=0, rtol=0)
    assert virtual == other_virtual and virtual[0]["input_ids"] == [0,1] and virtual[1]["input_ids"] == [2,3,4]
    assert cases[0]["input_ids"] == [8,7]
    assert native["mean_case_relative_mse"] == 0 and compressed["mean_case_relative_mse"] > 0
    assert native["total_tokens"] == 5 and native["records"][0]["native_offset"] == 1
    assert compressed["basis_orthogonality_max"] == 0


def test_zero_teacher_update_is_zero_projection_error(tmp_path):
    h0, _, items, cases, basis = native_banks(tmp_path)
    np.save(tmp_path / "target.npy", h0.bfloat16().view(torch.int16).numpy())
    _, _, result = oracle.reconstruct(cases, items, tmp_path / "input.npy", tmp_path / "target.npy", basis, arm="projected_update")
    assert result["mean_case_relative_mse"] == 0 and sum(r["zero_update_tokens"] for r in result["records"]) == 5


def test_basis_and_development_identity_gates(tmp_path):
    _, _, items, cases, basis = native_banks(tmp_path)
    args = (cases, items, tmp_path / "input.npy", tmp_path / "target.npy")
    with pytest.raises(ValueError, match="orthonormal"):
        oracle.reconstruct(*args, basis * 2, arm="projected_update")
    with pytest.raises(ValueError, match="reservation"):
        oracle.reconstruct(*args, basis, arm="projected_update", device="cuda:0")
    changed = deepcopy(cases); changed[0]["split"] = "train"
    with pytest.raises(ValueError, match="split"):
        oracle.reconstruct(changed, *args[1:], basis, arm="native_boundary")


def test_virtual_table_suffix_uses_original_positions_and_preserves_labels(tmp_path):
    _, h1, items, cases, basis = native_banks(tmp_path)
    table, virtual, _ = oracle.reconstruct(cases, items, tmp_path / "input.npy", tmp_path / "target.npy", basis, arm="native_boundary")
    class Suffix(torch.nn.Module):
        def forward(self, hidden, position_ids=None, **_kwargs):
            return hidden + position_ids[..., None].float()
    result = oracle.single.stream_score(virtual, input_embedding=table, block_loader=lambda _i:Suffix(),
        final_readout=lambda h, _labels:h[:, :2], label_ids=[0,1], scratch_dir=tmp_path / "scratch",
        rotary=lambda h,p:(torch.ones_like(h),torch.zeros_like(h)), layer_count=1, device="cpu", microbatch=2)
    assert [r["identity"] for r in result["records"]] == [c["identity"] for c in cases]
    for row, case in zip(result["records"], cases):
        expected = h1[case["stop"]-1, :2] + len(case["input_ids"])-1
        assert row["logits"] == expected.tolist() and row["answer"] == case["answer"]


def test_reservation_before_cuda_and_append_only_failure(tmp_path, monkeypatch):
    from experiments.structure_net import jspace_full_lease
    monkeypatch.setattr(oracle, "verify", lambda _root:(dict(arms=list(oracle.ARMS), fingerprint="x"), [], []))
    def denied(_devices): raise RuntimeError("No reservation")
    monkeypatch.setattr(jspace_full_lease, "wait_ready", denied)
    monkeypatch.setattr(torch.cuda, "init", lambda:pytest.fail("CUDA initialized without reservation"))
    with pytest.raises(RuntimeError, match="reservation"): oracle.run(tmp_path, "native_boundary")
    assert oracle.recipe.read(tmp_path / "native_boundary/failure.json")["status"] == "failed"
    with pytest.raises(FileExistsError): oracle.run(tmp_path, "native_boundary")
