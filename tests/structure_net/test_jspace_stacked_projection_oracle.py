from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import jspace_stacked_projection_oracle as oracle


def cases():
    return [dict(identity="short", family="one", split="development", offset=0, stop=1,
                 input_ids=[1], choices=["A", "B"], answer=0),
            dict(identity="long", family="two", split="development", offset=1, stop=4,
                 input_ids=[2,3,4], choices=["A", "B"], answer=1)]


def test_recurrence_projects_from_actual_accumulated_state_not_native_delta():
    initial = torch.zeros(1, 2, dtype=torch.float64)
    first = torch.tensor([[1., 2.]], dtype=torch.float64)
    final = torch.tensor([[3., 4.]], dtype=torch.float64)
    x, y = torch.eye(2, dtype=torch.float64)[:, :1], torch.eye(2, dtype=torch.float64)[:, 1:]
    state = oracle.project_step(initial, first, x)
    actual = oracle.project_step(state, final, y)
    torch.testing.assert_close(actual, torch.tensor([[1., 4.]], dtype=torch.float64), atol=0, rtol=0)
    incorrect_native_delta = state + ((final-first)@y)@y.T
    assert not torch.equal(actual, incorrect_native_delta)


def test_projector_preserves_input_complement_and_removes_reachable_target_error():
    torch.manual_seed(24)
    current, target = torch.randn(5, 6, dtype=torch.float64), torch.randn(5, 6, dtype=torch.float64)
    basis = torch.linalg.qr(torch.randn(6, 3, dtype=torch.float64)).Q
    result = oracle.project_step(current, target, basis)
    torch.testing.assert_close((result-target)@basis, torch.zeros(5,3,dtype=torch.float64), atol=2e-15, rtol=0)
    complement = torch.eye(6, dtype=torch.float64)-basis@basis.T
    torch.testing.assert_close(result@complement, current@complement, atol=2e-15, rtol=0)
    torch.testing.assert_close(oracle.project_step(current,target,torch.eye(6,dtype=torch.float64)),target,atol=5e-16,rtol=0)
    with pytest.raises(ValueError, match="orthonormal"):
        oracle.project_step(current, target, torch.ones(6,3,dtype=torch.float64))


def test_relative_RMS_weights_cases_equally_despite_different_prompt_lengths():
    target = torch.ones(4, 1)
    current = target + torch.tensor([[1.], [3.], [3.], [3.]])
    metrics = oracle.relative_rms(current, target, cases())
    assert metrics["valid_token_relative_rms"] == pytest.approx(5**.5)
    assert metrics["answer_relative_rms"] == pytest.approx(5**.5)
    assert metrics["valid_token_relative_rms"] != pytest.approx(7**.5)  # Token-weighted would be wrong.


def test_cached_two_layer_oracle_scores_original_readout_and_reports_native_control():
    banks = [torch.zeros(4,2), torch.tensor([[1.,2.],[3.,1.],[2.,3.],[4.,2.]]),
             torch.tensor([[3.,4.],[1.,4.],[5.,6.],[7.,8.]])]
    bases = [torch.eye(2)[:,:1], torch.eye(2)[:,1:]]
    reads, fitted, guarded = [], [], []
    def bank(index): reads.append(index); return banks[index].clone()
    def basis(index): fitted.append(index); return bases[index]
    result = oracle.evaluate(cases(), boundary_loader=bank, basis_loader=basis, layers=2,
        reader=lambda hidden:hidden, guard=lambda:guarded.append(True))
    assert reads == [0,1,2] and fitted == [0,1] and guarded
    assert result["native"]["records"][0]["logits"] == [3.,4.]
    assert result["oracle"]["records"][0]["logits"] == [1.,4.]
    assert result["oracle"]["records"][1]["logits"] == [4.,8.]
    expected = banks[1].clone(); expected[:,1] = banks[2][:,1]
    assert result["trajectory"][-1] == dict(boundary=2, **oracle.relative_rms(expected,banks[2],cases()))
    assert not result["confirmed"] and "not a task-accuracy upper bound" in result["scope"]
    assert result["paired"]["overall"]["count"] == 2


def test_full_rank_oracle_recovers_all_native_states_and_predictions():
    torch.manual_seed(54)
    banks = [torch.randn(4,2) for _ in range(3)]
    result = oracle.evaluate(cases(), boundary_loader=lambda i:banks[i], basis_loader=lambda _:torch.eye(2),
        reader=lambda h:h, layers=2)
    assert result["paired"]["overall"]["agreement"] == 1.
    assert result["trajectory"][-1]["valid_token_relative_rms"] < 1e-6
    for a,b in zip(result["native"]["records"],result["oracle"]["records"]):
        torch.testing.assert_close(torch.tensor(a["logits"]),torch.tensor(b["logits"]),atol=2e-7,rtol=1e-6)


def test_only_contiguous_development_rows_and_guarded_GPU_execution_allowed():
    bad = deepcopy(cases()); bad[0]["split"] = "evaluation"
    with pytest.raises(ValueError, match="development-only"): oracle.validate_layout(bad,4)
    bad = deepcopy(cases()); bad[1]["offset"] = 2
    with pytest.raises(ValueError, match="Contiguous"): oracle.validate_layout(bad,4)
    with pytest.raises(ValueError, match="reservation guard"):
        oracle.evaluate(cases(),boundary_loader=lambda _:pytest.fail("Read before guard"),
            basis_loader=lambda _:None,reader=lambda _:None,device="cuda:0")


def test_exact_2060_allocator_cap_and_denied_lease_precede_CUDA(tmp_path,monkeypatch):
    from experiments.structure_net import jspace_full_lease
    events = []
    fake = SimpleNamespace(is_initialized=lambda:False,
        get_device_properties=lambda _:SimpleNamespace(total_memory=6*1024**3),
        memory=SimpleNamespace(get_allocator_backend=lambda:"native"),
        set_per_process_memory_fraction=lambda fraction,device:events.append((fraction,device)))
    uuid = "GPU-8cf440df-test"
    settings = oracle.cap_allocator(dict(id="lease",allocation={uuid:3072}),cuda=fake,visible=uuid)
    assert settings["allocator_limit_bytes"] == 2816*1024**2
    assert events == [(2816/6144,0)]
    with pytest.raises(ValueError,match="leased2060"):
        oracle.cap_allocator(dict(id="wrong",allocation={uuid:7168}),cuda=fake,visible=uuid)
    monkeypatch.setattr(oracle,"verify",lambda _:dict(fingerprint="oracle"))
    def denied(_): raise RuntimeError("No reservation")
    monkeypatch.setattr(jspace_full_lease,"wait_ready",denied)
    monkeypatch.setattr(oracle,"cap_allocator",lambda *_:pytest.fail("CUDA before lease"))
    with pytest.raises(RuntimeError,match="reservation"): oracle.run(tmp_path)
    receipt = oracle.recipe.read(tmp_path/"evaluation/failure.json")
    assert receipt["lease_id"] is None and receipt["allocator"] is None


def test_prepare_is_CPU_only_before_reading_parent_sources(tmp_path,monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES",raising=False)
    monkeypatch.setattr(oracle.assembly,"verify",lambda _:pytest.fail("Read before CPU preparation guard"))
    with pytest.raises(ValueError,match="Hide CUDA"):
        oracle.prepare(tmp_path/"new",tmp_path/"assembly",tmp_path/"cache")
