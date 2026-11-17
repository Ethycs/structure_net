"""Consumer-visible geometry tests with real tiny Qwen affine/polynomial blocks."""
from copy import deepcopy
import json

import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_3b_context_qualification as geometry
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


def fixture(count=2):
    torch.manual_seed(930)
    config = Qwen2Config(hidden_size=16, intermediate_size=24, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=count, vocab_size=32,
                         attention_dropout=0.0, max_position_embeddings=32)
    config._attn_implementation = "eager"
    blocks, natives, bases, means = [], [], [], []
    for index in range(count):
        native = Qwen2DecoderLayer(config, layer_idx=index).double().eval().requires_grad_(False)
        basis = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64)).Q[:, :8]
        mean = torch.randn(4, 16, dtype=torch.float64) / 4
        projection = dict(rope_pairs=torch.tensor([[0], [1]]),
                          value_basis=torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)[:, :, :2])
        block = AffineQwenBlock(native, basis, mean, head_projection=projection)
        fit = dict(anchors=[0, 2], mean=torch.randn(24, dtype=torch.float64) / 5,
                   scale=torch.rand(24, dtype=torch.float64) + 0.7,
                   coefficients={"2": torch.randn(24, 6, dtype=torch.float64) / 30})
        block.compile_polynomial(fit, [1, 3, 6, 8])
        # Recovery changes coordinate weights and shared offsets, NOT the native
        # orthogonal map. Ensure extension does not silently recompile these.
        with torch.no_grad():
            block.qkv_weight.add_(0.01)
            block.gate_up_offset.add_(0.03)
            block.polynomial_output.mul_(0.9)
        blocks.append(block); natives.append(native); bases.append(basis); means.append(mean)
    stack = CompactQwenStack(blocks, pad_to_multiple=0)
    norm = Qwen2RMSNorm(16, eps=config.rms_norm_eps).double()
    rotary = Qwen2RotaryEmbedding(config=config, device="cpu").double()
    return stack, norm, rotary, natives, bases, means


def extend(stack, natives, bases, target=9):
    for block, native, basis in zip(stack.blocks, natives, bases):
        geometry.extend_block(block, native, basis, torch.randn(target-4, 16, dtype=torch.float64))
    geometry.extend_bridges(stack, bases, 4)


def test_extension_preserves_old_outputs_and_every_gradient_exactly():
    stack, norm, rotary, natives, bases, _ = fixture()
    boundary = torch.randn(1, 4, 16, dtype=torch.float64)
    saved = deepcopy(stack.state_dict())
    before = geometry._gradients(stack, norm, rotary, boundary)
    extend(stack, natives, bases)
    after = geometry._gradients(stack, norm, rotary, boundary)
    assert before == after
    geometry.assert_preserved(saved, stack.state_dict(), 4, geometry._tensor_tables(stack))
    for block in stack.blocks:
        assert block.position_count == 9
        assert block.position_policy == "error"
    long = torch.randn(1, 9, 16, dtype=torch.float64)
    assert geometry._gradients(stack, norm, rotary, long)["finite"]


def test_new_affine_rows_match_native_fold_with_head_and_polynomial_order():
    stack, _, _, natives, bases, _ = fixture(count=1)
    block, native, basis = stack.blocks[0], natives[0], bases[0]
    means = torch.randn(5, 16, dtype=torch.float64)
    before_parameters = {name: p.clone() for name, p in block.named_parameters()}
    normalizers = (block.polynomial_anchor_mean.clone(), block.polynomial_anchor_scale.clone())
    geometry.extend_block(block, native, basis, means)
    anchor = means - (means @ basis) @ basis.T
    torch.testing.assert_close(block.anchor[4:], anchor, atol=0, rtol=0)
    torch.testing.assert_close(anchor @ basis, torch.zeros(5, 8, dtype=torch.float64), atol=2e-15, rtol=0)
    qkv = torch.cat([p.weight for p in (native.self_attn.q_proj, native.self_attn.k_proj, native.self_attn.v_proj)])
    raw = anchor @ (qkv * native.input_layernorm.weight[None]).T
    qr = torch.cat([block.query_rope_columns[h] + h*4 for h in range(4)])
    kr = torch.cat([block.key_rope_columns[g] + (4+g)*4 for g in range(2)])
    expected = torch.cat((raw[:, torch.cat((qr, kr))], raw[:, 24:28] @ block.value_basis[0],
                          raw[:, 28:32] @ block.value_basis[1]), -1)
    torch.testing.assert_close(block.qkv_shift[4:], expected, atol=0, rtol=0)
    gate = anchor @ (torch.cat((native.mlp.gate_proj.weight, native.mlp.up_proj.weight)) *
                     native.post_attention_layernorm.weight[None]).T
    rows = torch.cat((block.polynomial_kept, block.polynomial_kept + 24))
    torch.testing.assert_close(block.gate_up_shift[4:], gate[:, rows], atol=0, rtol=0)
    torch.testing.assert_close(block.anchor_energy[4:], anchor.square().sum(-1, keepdim=True), atol=0, rtol=0)
    for name, p in block.named_parameters():
        assert torch.equal(before_parameters[name], p)
    assert torch.equal(block.polynomial_anchor_mean, normalizers[0])
    assert torch.equal(block.polynomial_anchor_scale, normalizers[1])
    assert not torch.equal(block.anchor[4:], block.anchor[3:4].expand(5, -1))
    assert torch.count_nonzero(block.anchor[4:]) > 0


def test_bridge_tail_uses_unrounded_fit_basis_and_preserves_old_rows():
    stack, _, _, natives, bases, _ = fixture()
    before = stack.bridges[0].shift.clone()
    weight = stack.bridges[0].weight.clone()
    extend(stack, natives, bases)
    assert torch.equal(stack.bridges[0].shift[:4], before)
    assert torch.equal(stack.bridges[0].weight, weight)
    expected = (stack.blocks[0].anchor[4:]-stack.blocks[1].anchor[4:]) @ bases[1]
    torch.testing.assert_close(stack.bridges[0].shift[4:], expected, atol=0, rtol=0)


def test_full_support_causal_and_masked_key_isolation_and_no_clamping():
    stack, norm, rotary, natives, bases, _ = fixture()
    extend(stack, natives, bases)
    x = torch.randn(1, 9, 16, dtype=torch.float64)
    changed = x.clone(); changed[:, -1] += 3
    actual = geometry._forward(stack, norm, rotary, x)
    other = geometry._forward(stack, norm, rotary, changed)
    assert torch.equal(actual[:, :-1], other[:, :-1])
    changed = x.clone(); changed[:, 4] -= 8
    masked = geometry._forward(stack, norm, rotary, x, masked_key=4)
    perturbed = geometry._forward(stack, norm, rotary, changed, masked_key=4)
    keep = torch.arange(9) != 4
    assert torch.equal(masked[:, keep], perturbed[:, keep])
    assert torch.isfinite(masked).all()
    with pytest.raises((IndexError, RuntimeError)):
        geometry._forward(stack, norm, rotary, torch.randn(1, 10, 16, dtype=torch.float64))


def test_strict_resize_reload_reproduces_extended_output():
    stack, norm, rotary, natives, bases, _ = fixture()
    restored = deepcopy(stack)
    extend(stack, natives, bases)
    geometry._load_state(restored, stack.state_dict(), 9)
    x = torch.randn(1, 9, 16, dtype=torch.float64)
    assert torch.equal(geometry._forward(stack, norm, rotary, x), geometry._forward(restored, norm, rotary, x))
    with pytest.raises(ValueError, match="Invalid extended"):
        geometry._load_state(restored, stack.state_dict(), 10)


def test_extension_rejects_clamping_changed_basis_and_missing_observations():
    stack, _, _, natives, bases, _ = fixture(count=1)
    block = stack.blocks[0]
    means = torch.randn(3, 16, dtype=torch.float64)
    with pytest.raises(ValueError, match="basis differs"):
        geometry.extend_block(block, natives[0], bases[0] + 0.01, means)
    with pytest.raises(ValueError, match="Finite observed"):
        geometry.extend_block(block, natives[0], bases[0], means[:0])
    block.position_policy = "clamp"
    with pytest.raises(ValueError, match="unclamped"):
        geometry.extend_block(block, natives[0], bases[0], means)


def calibration_manifest(tmp_path, *, split="train", length=9, groups=2):
    records = []
    for index in range(groups):
        records.append(dict(example_id=f"case{index}", group_id=f"group{index}", split=split,
            input_ids=list(range(length)), provenance=dict(source_uri="local:source", source_revision="pinned",
                tokenizer_sha256="tokenizer", chat_template_sha256="template", source_row=index,
                source_complete_tokens_sha256="full-source-digest", source_complete_token_count=length+3,
                derivation="exact_prefix_crop_for_geometry_only", historical_lineage={"split": "train"})))
    path = tmp_path / "calibration.jsonl"
    path.write_text("".join(json.dumps(row)+"\n" for row in records))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(schema="jspace.recovery-reset.corpus.v1", train_records=str(tmp_path / "unused.jsonl"),
        calibration_records=str(path), sha256={str(path): geometry.sha(path)})))
    return manifest, path


def test_calibration_admits_exact_train_prefixes_with_lineage(tmp_path):
    manifest, _ = calibration_manifest(tmp_path, length=12)
    items, lineage = geometry.calibration_records(manifest, target=9, maximum=16, minimum=2)
    assert [r["input_ids"] for r in items] == [list(range(9)), list(range(9))]
    assert items[1]["offset"] == 9 and items[1]["stop"] == 18
    assert lineage["samples_per_position"] == 2
    assert items[0]["provenance"]["source_complete_token_count"] == 15
    assert items[0]["prefix_sha256"] == geometry.data.digest(list(range(9)))


@pytest.mark.parametrize("split,length,error", [("development", 9, "TRAIN only"), ("train", 8, "covering all")])
def test_calibration_never_uses_development_or_pads_missing_positions(tmp_path, split, length, error):
    manifest, _ = calibration_manifest(tmp_path, split=split, length=length)
    with pytest.raises(ValueError, match=error):
        geometry.calibration_records(manifest, target=9, maximum=16, minimum=2)


def test_calibration_authenticates_source_shard(tmp_path):
    manifest, path = calibration_manifest(tmp_path)
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        geometry.calibration_records(manifest, target=9, maximum=16, minimum=2)


def test_unqualified_export_cannot_be_loaded_by_runtime(tmp_path):
    manifest = tmp_path / "candidate.json"
    manifest.write_text(json.dumps(dict(schema=geometry.SCHEMA, qualified=False)))
    with pytest.raises(ValueError, match="not qualified"):
        geometry.load_extended(manifest)


def test_cuda_loader_requires_guard_before_model_io():
    with pytest.raises(ValueError, match="reservation guard"):
        geometry.load_initial({}, device="cuda:0")


def test_preservation_rejects_changed_learned_weight():
    stack, _, _, _, _, _ = fixture(count=1)
    original = deepcopy(stack.state_dict())
    with torch.no_grad():
        stack.blocks[0].polynomial_output.add_(0.125)
    with pytest.raises(ValueError, match="changed existing state"):
        geometry.assert_preserved(original, stack.state_dict(), 4, geometry._tensor_tables(stack))


def test_prepared_only_check_authenticates_spec_and_nested_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"schema": "jspace.recovery-reset.spec.v1"}))
    nested = tmp_path / "checkpoint.pt"
    nested.write_bytes(b"authenticated checkpoint bytes")
    plan = dict(schema=geometry.SCHEMA, spec_path=str(spec), spec_sha256=geometry.sha(spec),
                spec=json.loads(spec.read_text()), max_length=2048,
                dependencies={str(nested): geometry.sha(nested)})
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    result = geometry.check(tmp_path)
    assert result["phase"] == "prepared"
    assert not result["qualified"]
    nested.write_bytes(b"altered checkpoint bytes")
    with pytest.raises(ValueError, match="hash mismatch"):
        geometry.check(tmp_path)


def test_prepared_only_check_rejects_changed_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"target_context": 2048}))
    plan = dict(schema=geometry.SCHEMA, spec_path=str(spec), spec_sha256=geometry.sha(spec),
                spec=json.loads(spec.read_text()), max_length=2048, dependencies={})
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    spec.write_text(json.dumps({"target_context": 416}))
    with pytest.raises(ValueError, match="hash mismatch"):
        geometry.check(tmp_path)
