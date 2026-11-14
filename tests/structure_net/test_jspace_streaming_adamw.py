"""CPU tests for optimizer precision, clipping, bounded transfer, and resume."""
import json

import pytest
import torch

from experiments.structure_net.jspace_streaming_adamw import StreamingCPUAdamW


def test_fp32_training_matches_pytorch_adamw_with_missing_gradients_and_decay():
    torch.manual_seed(1701)
    model = torch.nn.Linear(4, 3)
    reference = torch.nn.Linear(4, 3); reference.load_state_dict(model.state_dict())
    actual = StreamingCPUAdamW(model.parameters(), lr=.003, weight_decay=.07, chunk_bytes=20)
    expected = torch.optim.AdamW(reference.parameters(), lr=.003, weight_decay=.07, foreach=False)
    x, target = torch.randn(7, 4), torch.randn(7, 3)
    for step in range(12):
        actual.zero_grad(); expected.zero_grad()
        (model(x) - target).square().mean().backward()
        (reference(x) - target).square().mean().backward()
        if step == 4:
            model.bias.grad = reference.bias.grad = None
        actual.param_groups[0]["lr"] = expected.param_groups[0]["lr"] = .003 / (step + 1)
        diagnostics = actual.step(); expected.step()
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(actual.state[p][key], expected.state[q][key], rtol=0, atol=0)
    assert diagnostics["chunks"] == 4
    assert diagnostics["cpu_state_bytes"] == 12 * 15
    assert diagnostics["device_to_host_bytes"] == diagnostics["host_to_device_bytes"] == 0
    assert not actual.pin_memory


def test_bf16_small_updates_accumulate_in_master_and_resume_exactly(tmp_path):
    p = torch.nn.Parameter(torch.ones(7, dtype=torch.bfloat16))
    optimizer = StreamingCPUAdamW([p], lr=1e-4, chunk_bytes=12)
    for _ in range(10):
        p.grad = torch.ones_like(p); optimizer.step()
    assert torch.equal(p, torch.ones_like(p))  # Update still below BF16 rounding threshold.
    assert torch.all(optimizer.state[p]["master"] < .9991)
    optimizer.save_checkpoint(tmp_path / "resume")
    q = torch.nn.Parameter(torch.full_like(p, 8.))
    restored = StreamingCPUAdamW([q], lr=.9, chunk_bytes=12)
    restored.load_checkpoint(tmp_path / "resume")
    for key in ("master", "exp_avg", "exp_avg_sq"):
        assert torch.equal(optimizer.state[p][key], restored.state[q][key])
        assert restored.state[q][key].dtype == torch.float32
        assert restored.state[q][key].device.type == "cpu"
    for _ in range(90):
        p.grad = torch.ones_like(p); q.grad = torch.ones_like(q)
        optimizer.step(); restored.step()
    assert torch.equal(p, q) and torch.all(p < 1)
    assert torch.equal(optimizer.state[p]["master"], restored.state[q]["master"])
    assert optimizer.state[p]["step"] == restored.state[q]["step"] == 100


def test_bf16_accumulated_training_matches_fp32_master_reference():
    torch.manual_seed(4)
    p = torch.nn.Parameter(torch.randn(5, 2, dtype=torch.bfloat16))
    master = torch.nn.Parameter(p.detach().float().clone())
    actual = StreamingCPUAdamW([p], lr=.002, chunk_bytes=16)
    expected = torch.optim.AdamW([master], lr=.002, weight_decay=0., foreach=False)
    x = torch.randn(4, 5).bfloat16(); target = torch.randn(4, 2)
    for _ in range(8):
        actual.zero_grad(); expected.zero_grad()
        reference_weights = torch.nn.Parameter(master.detach().bfloat16())
        # Four microbatches accumulate in the declared device parameter dtype.
        for sample in range(4):
            ((x[sample:sample + 1] @ p).float() - target[sample:sample + 1]).square().mean().div(4).backward()
            ((x[sample:sample + 1] @ reference_weights).float() - target[sample:sample + 1]).square().mean().div(4).backward()
        master.grad = reference_weights.grad.float()
        # Independent low-precision forward/backward, FP32 master optimizer.
        expected.step(); actual.step(max_grad_norm=None)
        assert torch.equal(actual.state[p]["master"], master)
        assert torch.equal(p, master.bfloat16())


def test_global_clipping_and_explicit_loss_scale_match_unscaled_reference():
    p = torch.nn.Parameter(torch.tensor([1., -2., 3., 4., -1.]))
    q = torch.nn.Parameter(p.detach().clone())
    actual = StreamingCPUAdamW([p], lr=.01, chunk_bytes=8)
    expected = torch.optim.AdamW([q], lr=.01, weight_decay=0., foreach=False)
    q.grad = torch.tensor([3., 4., 0., 12., 0.]); p.grad = q.grad * 32
    reference_norm = torch.nn.utils.clip_grad_norm_([q], max_norm=2.)
    diagnostics = actual.step(max_grad_norm=2., grad_scale=32.)
    expected.step()
    assert diagnostics["unclipped_gradient_norm"] == pytest.approx(float(reference_norm), rel=1e-6)
    assert diagnostics["clip_coefficient"] == pytest.approx(2. / (13. + 1e-6))
    torch.testing.assert_close(p, q, rtol=1e-6, atol=1e-7)


def test_nonfinite_gradient_rejects_entire_update_before_state_mutation():
    a, b = (torch.nn.Parameter(torch.ones(4)) for _ in range(2))
    optimizer = StreamingCPUAdamW([a, b], lr=.1, weight_decay=.2, chunk_bytes=8)
    a.grad = torch.ones_like(a); b.grad = torch.full_like(b, float("inf"))
    with pytest.raises(FloatingPointError):
        optimizer.step(max_grad_norm=1.)
    assert torch.equal(a, torch.ones_like(a))
    assert all(s["step"] == 0 and torch.count_nonzero(s["exp_avg"]) == 0 for s in optimizer.state.values())


def test_checkpoint_chunks_have_independent_bounded_storage_and_detect_corruption(tmp_path):
    p = torch.nn.Parameter(torch.arange(100.).bfloat16())
    optimizer = StreamingCPUAdamW([p], chunk_bytes=20)
    optimizer.save_checkpoint(tmp_path / "state")
    manifest = json.loads((tmp_path / "state/manifest.json").read_text())
    chunks = manifest["parameters"][0]["chunks"]
    assert len(chunks) == 20
    for chunk in chunks:
        payload = torch.load(tmp_path / "state" / chunk["file"], weights_only=True)
        assert all(t.untyped_storage().nbytes() <= 20 for t in payload.values())
    first = tmp_path / "state" / chunks[0]["file"]
    first.write_bytes(first.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Changed optimizer checkpoint"):
        optimizer.load_checkpoint(tmp_path / "state")


def test_state_budget_and_monolithic_resume_are_explicitly_guarded():
    p = torch.nn.Parameter(torch.ones(8))
    with pytest.raises(MemoryError):
        StreamingCPUAdamW([p], max_cpu_state_bytes=95)
    optimizer = StreamingCPUAdamW([p], chunk_bytes=16)
    with pytest.raises(RuntimeError, match="save_checkpoint"):
        optimizer.state_dict()
    assert optimizer.memory_stats()["maximum_checkpoint_payload_bytes"] == 48
    with pytest.raises(ValueError):
        StreamingCPUAdamW([p, p])
