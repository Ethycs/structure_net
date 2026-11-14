"""CPU proofs for grouped VJPs and lossless streamed derivative storage."""
import copy
import math

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_qwen_progressive_calibration as capture
from experiments.structure_net.jspace_benchmark_groebner import select_nested_anchors
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock


def model_fixture(compact_suffix=True):
    torch.manual_seed(831)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_attention_heads=2,
                        num_key_value_heads=1, num_hidden_layers=3, vocab_size=16,
                        attention_dropout=0., max_position_embeddings=32)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).bfloat16().eval().requires_grad_(False)
    if compact_suffix:
        block = AffineQwenBlock(model.model.layers[2], torch.eye(8)[:, :4], torch.zeros(1, 8), trainable=False)
        block.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        model.model.layers[2] = block
    return model


def cases_fixture():
    return [{"identity": "cal", "family": "a", "split": "train", "input_ids": [1, 2, 3, 4], "loss_positions": [1, 3]},
            {"identity": "train", "family": "a", "split": "train", "input_ids": [4, 3, 2], "loss_positions": [1, 2]},
            {"identity": "dev", "family": "a", "split": "development", "input_ids": [1, 5, 2], "loss_positions": [1, 2]}]


def reader(path):
    return torch.from_numpy(np.array(np.load(path, mmap_mode="r"), copy=True)).view(torch.bfloat16)


def test_bit_writer_requires_existing_bfloat16_and_reader_preserves_exact_values(tmp_path):
    torch.manual_seed(724)
    values = torch.randn(19, 12).bfloat16(); values[0, 0] = -0.
    path = tmp_path / "bits.npy"
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.int16, shape=tuple(values.shape))
    capture.write_bfloat16_bits(matrix, slice(None), values); matrix.flush()
    with pytest.raises(ValueError, match="already-bfloat16"):
        capture.write_bfloat16_bits(matrix, slice(None), values.float())
    source = capture.StreamedBFloat16JacobianRows(path, max_rows=7)
    assert source.dtype == torch.float32 and source.readonly and source.is_floating_point()
    chunks = [source[start:start + 7] for start in range(0, 19, 7)]
    assert torch.equal(torch.cat(chunks).view(torch.int32), values.float().view(torch.int32))
    assert source.bytes_read == values.numel() * 2 and source.maximum_read_rows == 7
    with pytest.raises(ValueError, match="exceeds"):
        source[:8]
    with pytest.raises(TypeError): source[0]
    source.close(); chunks[0].zero_()
    assert torch.equal(reader(path), values)
    with pytest.raises(ValueError, match="closed"): source[:1]


def test_streamed_bfloat16_anchor_fit_matches_exact_resident_conversion(tmp_path):
    torch.manual_seed(829)
    raw = torch.randn(57, 16).bfloat16(); path = tmp_path / "jacobian.npy"
    np.save(path, raw.view(torch.int16).numpy())
    scale = torch.rand(16) + .3
    options = dict(max_anchors=4, seed=91, svd_rank=8, oversample=4, power_iterations=1, chunk_rows=8)
    expected, expected_meta = select_nested_anchors(raw.float(), scale, **options)
    with capture.StreamedBFloat16JacobianRows(path, max_rows=8) as rows:
        actual, metadata = select_nested_anchors(rows, scale, **options)
        assert actual == expected
        assert metadata["captured_energy_fraction"] == pytest.approx(expected_meta["captured_energy_fraction"], rel=1e-5)
        assert rows.rows_read == 4 * len(raw) and rows.read_calls == 4 * math.ceil(len(raw) / 8)


def test_real_group_storage_budget_uses_two_bytes_per_derivative():
    lengths = [135] * 127 + [160] + [136] * 330 + [135] * 118
    cases = [{"length": n, "loss_positions": [1, 2, 3, 4], "split": "train" if i < 512 else "development"} for i, n in enumerate(lengths)]
    shapes = {i: {"width": 2048, "intermediate": 11008} for i in range(30, 36)}
    estimate = capture.estimate_group_bytes(cases, range(128), shapes)
    assert estimate["context_tokens"] == 78115 and estimate["calibration_context_tokens"] == 17305
    assert estimate["blocks"]["35"]["mlp_jacobian"] == 17305 * 16 * 11008 * 2
    assert estimate["blocks"]["35"]["boundary"] == 78115 * 2048 * 2
    assert estimate["accepted_logits"] == 576 * 4 * 151936 * 2


def test_one_forward_joint_vjps_match_separate_autograd_and_include_earlier_context(tmp_path, monkeypatch):
    model = model_fixture(); cases = cases_fixture(); vectors_seen = []; forward_calls = []
    original = capture.project_sign_probe_groups
    def record_vectors(*args, **kwargs):
        vectors = original(*args, **kwargs); vectors_seen.append(vectors.clone()); return vectors
    monkeypatch.setattr(capture, "project_sign_probe_groups", record_vectors)
    handle = model.model.register_forward_hook(lambda module, args, output:
        forward_calls.append((torch.is_grad_enabled(), output.last_hidden_state.requires_grad)))
    with capture.GroupedTeacherCapture(tmp_path / "group", model, (0, 1), cases, [0], teacher_fingerprint="accepted",
                                      probe_groups=2, probes_per_group=2, activation_positions=3, position_capacity=6) as group:
        for index, case in enumerate(cases):
            # Deliberately call inside no_grad; the helper must establish its VJP graph.
            with torch.no_grad():
                group.capture_case(index, lambda case=case: model.model(input_ids=torch.tensor([case["input_ids"]]), use_cache=False).last_hidden_state)
        metadata = group.finish()
        with pytest.raises(ValueError, match="finished"): group.capture_case(0, lambda: None)
    handle.remove()
    assert len(forward_calls) == len(cases) and len(vectors_seen) == 1
    assert forward_calls == [(True, True), (True, False), (True, False)]
    assert metadata["calibration_families"] == ["a"] * 3 and metadata["development_families"] == ["a"] * 3
    assert not any(p.grad is not None for p in model.parameters())

    values, handles = {}, []
    for layer in (0, 1):
        def hook(module, args, layer=layer):
            x = args[0].detach().clone().requires_grad_(True) if layer == 0 else args[0]
            values[layer, "input"] = x; return (x, *args[1:])
        handles.append(model.model.layers[layer].register_forward_pre_hook(hook))
        handles.append(model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(lambda m, a, i=layer: values.__setitem__((i, "mlp"), a[0])))
    output = model.model(input_ids=torch.tensor([cases[0]["input_ids"]]), use_cache=False).last_hidden_state
    for h in handles: h.remove()
    expected = {i: [] for i in (0, 1)}; grams = {i: torch.zeros(2, 8, 8) for i in (0, 1)}
    for g in range(2):
        for p in range(2):
            scalar = (output[0, cases[0]["loss_positions"]].float() * vectors_seen[0][g, p]).sum()
            derivatives = torch.autograd.grad(scalar, [values[i, key] for i in (0, 1) for key in ("input", "mlp")], retain_graph=not (g == p == 1))
            for offset, layer in enumerate((0, 1)):
                dx, dm = derivatives[2 * offset:2 * offset + 2]
                expected[layer].append(dm[0]); x = dx[0].float(); grams[layer][g].addmm_(x.T, x, alpha=.5)
    for layer in (0, 1):
        paths = metadata["blocks"][str(layer)]
        resident = torch.cat(expected[layer]).float()
        with capture.StreamedBFloat16JacobianRows(paths["mlp_jacobian_bf16"], max_rows=16) as source:
            assert source.shape == (16, 12)
            assert torch.equal(source[:16], resident)
        assert resident[0].abs().sum() > 0  # token0 is not a selected output position.
        stats = torch.load(paths["statistics"], weights_only=True)
        torch.testing.assert_close(stats["grams"], grams[layer], rtol=1e-5, atol=1e-6)
        assert stats["counts"].tolist() == [2., 2., 2., 1., 0., 0.]
        torch.testing.assert_close(stats["mask_grams"].diagonal(dim1=-2, dim2=-1).reshape_as(stats["pair_scores"]), stats["pair_scores"])
        cached = reader(paths["boundary"]).float(); expected_mean = (cached[:3] + cached[4:7]) / 2
        torch.testing.assert_close(stats["means"][:3], expected_mean)
        assert all(len(getattr(model.model.layers[layer].self_attn, key + "_proj")._forward_hooks) == 0 for key in ("q", "k", "v"))


def test_final_block_retains_zero_context_rows_instead_of_slicing_to_outputs(tmp_path):
    model, cases = model_fixture(False), cases_fixture()
    with capture.GroupedTeacherCapture(tmp_path / "last", model, (2,), cases, [0], teacher_fingerprint="native",
                                      probe_groups=1, probes_per_group=1, activation_positions=2) as group:
        for i, case in enumerate(cases):
            group.capture_case(i, lambda case=case: model.model(input_ids=torch.tensor([case["input_ids"]]), use_cache=False).last_hidden_state)
        metadata = group.finish()
    values = reader(metadata["blocks"]["2"]["mlp_jacobian_bf16"])
    assert values.shape == (4, 12)
    assert not values[[0, 2]].any() and values[[1, 3]].abs().sum() > 0


def test_rejects_evaluation_calibration_and_changed_teacher(tmp_path):
    model, cases = model_fixture(), cases_fixture()
    with pytest.raises(ValueError, match="training cases only"):
        capture.GroupedTeacherCapture(tmp_path / "invalid", model, (0,), cases, [2], teacher_fingerprint="fixed")
    with pytest.raises(ValueError, match="at most six"):
        capture.GroupedTeacherCapture(tmp_path / "too_many", model, range(7), cases, [0], teacher_fingerprint="fixed")
    with capture.GroupedTeacherCapture(tmp_path / "changed", model, (0,), cases, [0], teacher_fingerprint="fixed") as group:
        with torch.no_grad(): next(model.parameters()).add_(1)
        with pytest.raises(ValueError, match="teacher changed"):
            group.capture_case(0, lambda: None)
        with pytest.raises(ValueError, match="complete successful"):
            group.finish()
