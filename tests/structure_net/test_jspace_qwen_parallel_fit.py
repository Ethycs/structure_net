"""Small CPU proofs for grouped-statistic compilation and independent fits."""
import json

import numpy as np
import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_qwen_parallel_fit as parallel
from experiments.structure_net.jspace_bench_data import sha, write


@pytest.fixture
def captured(tmp_path):
    torch.manual_seed(834)
    capture = tmp_path / "capture"; source = capture / "block_01"; source.mkdir(parents=True)
    shapes = {"width": 8, "intermediate": 12, "heads": 2, "kv_heads": 1, "head_dim": 4}
    lengths = [3, 4, 5, 3]; offset = 0; items = []
    for index, length in enumerate(lengths):
        items.append({"identity": str(index), "family": "ab"[index % 2],
                      "split": "train" if index < 2 else "development", "length": length,
                      "input_ids": [1] * length, "loss_positions": [1, length - 1],
                      "activation_positions": [0, 1, length - 1], "offset": offset, "stop": offset + length})
        offset += length
    paths = {"shape": shapes}
    for name, shape in (("boundary", (15, 8)), ("mlp_calibration", (6, 12)),
                        ("mlp_development", (6, 12)), ("mlp_jacobian_bf16", (28, 12))):
        path = source / (name + ".npy")
        np.save(path, torch.randn(shape).bfloat16().view(torch.int16).numpy()); paths[name] = str(path)
    basis = torch.linalg.qr(torch.randn(8, 8))[0]
    gram = basis @ torch.diag(torch.arange(1., 9.)) @ basis.T
    pair_scores = torch.tensor([[[1., 4.]], [[3., 2.]]])
    stats = {"grams": torch.stack((gram, 2 * gram)), "means": torch.zeros(8, 8),
             "counts": torch.tensor([2., 2., 2., 1., 0., 0., 0., 0.]),
             "pair_scores": pair_scores, "mask_grams": torch.diag_embed(pair_scores[:, 0]),
             "value_moments": torch.diag(torch.tensor([1., 2., 4., 8.]))[None],
             "value_means": torch.tensor([[1., 2., 3., 4.]])}
    paths["statistics"] = str(source / "statistics.pt"); torch.save(stats, paths["statistics"])
    metadata = {"status": "completed", "teacher_fingerprint": "fixed-teacher", "layers": [1],
                "blocks": {"1": paths}, "normalization": {"input_grams": "already averaged over cases and probes"},
                "calibration_indices": [0, 1], "development_indices": [2, 3], "activation_positions": 3,
                "calibration_families": ["a"] * 3 + ["b"] * 3, "development_families": ["a"] * 3 + ["b"] * 3}
    write(capture / "metadata.json", metadata); write(capture / "items.json", {"items": items})
    return capture, tmp_path / "group", metadata


def compile_interface(captured):
    capture, output, _ = captured
    parallel.finalize_basis(capture, output, rank=4, head_dim=2, progress=lambda value: None)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_attention_heads=2,
                        num_key_value_heads=1, num_hidden_layers=3, vocab_size=16,
                        attention_dropout=0., max_position_embeddings=32)
    config._attn_implementation = "eager"
    native = Qwen2DecoderLayer(config, layer_idx=1).bfloat16().eval().requires_grad_(False)
    rotary = Qwen2RotaryEmbedding(config)
    def forward(block, x, case):
        positions = torch.arange(x.shape[1])[None]
        mask = torch.full((x.shape[1], x.shape[1]), torch.finfo(x.dtype).min, dtype=x.dtype).triu(1)[None, None]
        return block(x, attention_mask=mask, position_ids=positions, cache_position=positions[0],
                     position_embeddings=rotary(x, positions), use_cache=False)
    directory = output / "block_01"; before = parallel._module_digest(native)
    interface = parallel.capture_compiled_development(directory, capture, 1, native, forward, progress=lambda value: None)
    assert parallel._module_digest(native) == before
    return directory, native, interface


def test_basis_preserves_averaged_gramian_and_head_projection_math(captured):
    capture, output, metadata = captured; events = []
    result = parallel.finalize_basis(capture, output, rank=4, head_dim=2, progress=events.append)
    fit = torch.load(result["1"]["fit"], weights_only=True)
    report = json.loads((output / "block_01" / "fit.json").read_text())
    torch.testing.assert_close(fit["eigenvalues"], 1.5 * torch.arange(8., 0., -1))
    assert fit["rope_pairs"].tolist() == [[1]] and fit["value_basis"].shape == (1, 4, 2)
    stability = report["stability"]["4"]
    assert stability["combined_energy"] == pytest.approx(26 / 36)
    assert stability["projector_overlap"] == pytest.approx(1, abs=1e-6)
    assert stability["group0_basis_on_group1_energy"] == pytest.approx(26 / 36)
    head = report["head_projection"]
    assert head["joint_removed_mask_linearized_output_energy"] == 2
    assert head["joint_removed_mask_energy_over_mask_gramian_trace"] == .4
    assert head["value_second_moment_fraction"] == pytest.approx([.8])
    assert head["value_mean_projection_loss_fraction"] == pytest.approx([5 / 30])
    assert report["unobserved_positions"] == [4, 5, 6, 7]
    assert report["teacher_fingerprint"] == metadata["teacher_fingerprint"]
    assert [x["phase"] for x in events] == ["basis_eigendecomposition", "basis_complete"]
    with pytest.raises(FileExistsError):
        parallel.finalize_basis(capture, output, rank=4, head_dim=2, progress=lambda value: None)


def test_compiled_interface_uses_actual_bfloat16_output_map_and_native_provenance(captured):
    directory, native, interface = compile_interface(captured)
    fit = torch.load(directory / "fit.pt", weights_only=True)
    unrounded = fit["basis"][:, :4].T @ native.mlp.down_proj.weight.float()
    weight = torch.load(directory / "output_weight.pt", weights_only=True)
    assert torch.equal(weight, unrounded.bfloat16().float())
    assert not torch.equal(weight, unrounded)
    bits = np.load(directory / "compiled_dev.npy")
    assert bits.dtype == np.int16 and bits.shape == (6, 12)
    assert torch.isfinite(torch.from_numpy(bits).view(torch.bfloat16)).all()
    assert interface["native_block_state_sha256"] == parallel._module_digest(native)
    assert interface["development_families"] == ["a"] * 3 + ["b"] * 3
    assert interface["input_sha256"][str(directory / "output_weight.pt")] == sha(directory / "output_weight.pt")


def test_independent_relation_fit_streams_all_rows_and_keeps_graph_scope(captured):
    directory, _, _ = compile_interface(captured); events = []
    report = parallel.fit_relation(directory, anchors=2, remove_count=3, pad_to_multiple=8,
                                   svd_rank=4, oversample=2, chunk_rows=8, progress=events.append)
    fit = torch.load(directory / "relation.pt", weights_only=True)
    assert len(fit["anchors"]) == 2 and fit["coefficients"]["2"].shape == (12, 6)
    assert len(fit["removed"]) == 3 and fit["removed"].dtype == torch.long
    assert report["selected_anchor_width"] == 2
    assert report["runtime_padding"] == {"multiple": 8, "logical_width": 15, "padded_width": 16}
    assert report["streamed_jacobian"]["rows_read"] == 4 * 28
    assert report["streamed_jacobian"]["bytes_read"] == 4 * 28 * 12 * 2
    assert report["training_families"] == report["development_families"] == {"a": 3, "b": 3}
    assert report["metrics"]["selection_interface"] == "compressed_dev"
    assert "not identities" in fit["metrics"]["certificate"]["scope"]
    assert report["relation_sha256"] == sha(directory / "relation.pt")
    assert [e["pass_number"] for e in events if e["phase"] == "jacobian_pass_complete"] == [1, 2, 3, 4]
    with pytest.raises(FileExistsError): parallel.fit_relation(directory)


def test_changed_artifact_and_unrounded_map_are_rejected(captured):
    directory, _, interface = compile_interface(captured)
    weight = torch.load(directory / "output_weight.pt", weights_only=True)
    weight[0, 0] += .000013; torch.save(weight, directory / "output_weight.pt")
    with pytest.raises(ValueError, match="Changed fitting input"):
        parallel.fit_relation(directory, progress=lambda value: None)
    interface["input_sha256"][str(directory / "output_weight.pt")] = sha(directory / "output_weight.pt")
    (directory / "interface.json").write_text(json.dumps(interface))
    with pytest.raises(ValueError, match="BF16 rounding"):
        parallel.fit_relation(directory, progress=lambda value: None)


def test_reservation_guard_precedes_any_cuda_work_or_artifact_access(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="reservation guard"):
        parallel.finalize_basis(tmp_path / "missing", tmp_path / "out", device="cuda")
    events = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: events.append("resolve") or 0)
    assert parallel._guard("cuda", lambda: events.append("guard")) == torch.device("cuda:0")
    assert events == ["guard", "resolve"]
