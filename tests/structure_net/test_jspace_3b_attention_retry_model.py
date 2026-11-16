"""Attention-only intervention, original Torch control and real NAL updates."""
from copy import deepcopy
import json

import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_3b_attention_retry_model as study
from experiments.structure_net import jspace_3b_response_train as training
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


def fixture(dtype=torch.float64, *, padding=0):
    torch.manual_seed(9187)
    cfg = Qwen2Config(hidden_size=16, intermediate_size=28, num_attention_heads=4,
        num_key_value_heads=2, num_hidden_layers=36, max_position_embeddings=32,
        vocab_size=32, tie_word_embeddings=True, attention_dropout=0.)
    cfg._attn_implementation = "eager"
    blocks = []
    geometric_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    for layer in (33, 34, 35):
        native = Qwen2DecoderLayer(cfg, layer).to(dtype).eval()
        basis = torch.linalg.qr(torch.randn(16, 16, dtype=geometric_dtype)).Q[:, :8]
        block = AffineQwenBlock(native, basis, torch.randn(8, 16, dtype=geometric_dtype)/5,
            parameter_dtype=dtype, head_projection={"rope_pairs":torch.tensor([[0], [1]]),
                "value_basis":torch.eye(4, dtype=geometric_dtype).repeat(2, 1, 1)[:, :, :2]})
        block.compile_polynomial(dict(anchors=[0, 1, 2], mean=torch.randn(28, dtype=geometric_dtype)/10,
            scale=torch.ones(28, dtype=geometric_dtype),
            coefficients={"2":torch.randn(28, 10, dtype=geometric_dtype)/25}), [24, 25, 26, 27])
        for parameter in block.parameters():
            parameter.data = parameter.data.to(dtype)
        with torch.no_grad():
            block.qkv_bias.add_(torch.randn_like(block.qkv_bias)/5)
            block.qkv_offset.add_(.03125)
            block.gate_up_offset.add_(.0625)
        blocks.append(block)
    stack = CompactQwenStack(blocks, geometry_mode="bf16" if dtype == torch.bfloat16 else "fp32",
        pad_to_multiple=padding, enabled=False, checkpoint_blocks=False)
    return cfg, stack


def inputs(dtype, length=4):
    x = torch.randn(2, length, 16).to(dtype)
    positions = torch.arange(length)[None].expand(2, -1)
    phase = torch.arange(length, dtype=torch.float64)[:, None] * torch.tensor([.13, .41])
    phase = torch.cat((phase, phase), -1)[None].to(dtype)
    return x, dict(position_ids=positions, position_embeddings=(phase.cos(), phase.sin()))


def equal_tensors(first, second):
    assert set(first) == set(second)
    for key in first:
        if isinstance(first[key], torch.Tensor):
            assert torch.equal(first[key], second[key]), key


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
@pytest.mark.parametrize("padding", [0, 64])
@pytest.mark.parametrize("packed", [False, True])
def test_softmax_control_exact_forward_attention_and_gradients(dtype, padding, packed):
    _, original = fixture(dtype, padding=padding)
    if packed:
        original.pack(dtype=dtype, buffer_dtype=dtype, strip_unused=True).eval()
    control = study.AttentionRetryStack.from_stack(original, mode="softmax", layers=(35,))
    x, kwargs = inputs(dtype)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    expected, expected_attn = original(a, output_attentions=True, **kwargs)
    actual, actual_attn = control(b, output_attentions=True, **kwargs)
    assert torch.equal(expected, actual)
    assert all(torch.equal(u, v) for u, v in zip(expected_attn, actual_attn))
    expected.float().square().sum().backward(); actual.float().square().sum().backward()
    assert torch.equal(a.grad, b.grad)
    for (name, old), (new_name, new) in zip(original.named_parameters(), control.named_parameters()):
        assert name == new_name
        assert old.grad is None and new.grad is None or torch.equal(old.grad, new.grad)
    equal_tensors(original.state_dict(), control.state_dict())
    assert not original.enabled and not control._can_fuse(x)


def test_only_layer35_changes_and_polynomial_mlp_remains_the_existing_operation(monkeypatch):
    _, original = fixture()
    changed = study.AttentionRetryStack.from_stack(original, layers=(35,))
    before = deepcopy(changed.state_dict())
    calls = []
    target = changed.blocks[-1]
    original_mlp = target._mlp_update
    def record_mlp(*args):
        calls.append(args[0].shape)
        return original_mlp(*args)
    monkeypatch.setattr(target, "_mlp_update", record_mlp)
    x, kwargs = inputs(torch.float64)
    _, old_attn = original(x, output_attentions=True, **kwargs)
    _, new_attn = changed(x, output_attentions=True, **kwargs)
    assert torch.equal(old_attn[0], new_attn[0]) and torch.equal(old_attn[1], new_attn[1])
    assert not torch.allclose(old_attn[2], new_attn[2], atol=1e-9, rtol=1e-9)
    assert calls == [torch.Size([2, 4, 24])]
    assert all(b.has_polynomial and b.polynomial_feature_count == 10 for b in changed.blocks)
    assert all(b.head_dim == 2 and b.native_head_dim == 4 and b.scale == .5 for b in changed.blocks)
    equal_tensors(before, changed.state_dict())


def test_quadratic_masked_reference_and_all_masked_gradients():
    torch.manual_seed(127)
    scores = torch.randn(2, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    allowed = torch.ones(2, 1, 4, 4, dtype=torch.bool).tril()
    allowed[0, :, :, 2:] = False
    allowed[1, :, 0] = False
    mask = torch.where(allowed, 0., -torch.inf)
    actual = study.quadratic_probabilities(scores, attention_mask=mask)
    safe = torch.where(allowed, scores, 0.)
    weights = torch.where(allowed, 1 + safe + .5 * safe.square(), 0.)
    expected = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)
    assert torch.equal(actual, expected)
    assert not torch.count_nonzero(actual[1, :, 0])
    assert torch.autograd.gradcheck(lambda s: study.quadratic_probabilities(s, attention_mask=mask), (scores,))
    gradient, = torch.autograd.grad(actual.square().sum(), scores)
    assert torch.isfinite(gradient).all() and not torch.count_nonzero(gradient.masked_select(~allowed.expand_as(scores)))
    # Native-rounded BF16 scores are promoted only for the polynomial.
    rounded = scores.detach().bfloat16()
    assert torch.equal(study.quadratic_probabilities(rounded, attention_mask=mask),
        study.quadratic_probabilities(rounded.float(), attention_mask=mask))


def test_causal_and_padding_support_with_polynomial_mlp():
    _, original = fixture()
    changed = study.AttentionRetryStack.from_stack(original)
    x, kwargs = inputs(torch.float64)
    allowed = torch.ones(2, 1, 4, 4, dtype=torch.bool).tril()
    allowed[0, :, :, 2:] = False
    mask = torch.where(allowed, 0., -torch.inf)
    first, attention = changed(x, attention_mask=mask, output_attentions=True, **kwargs)
    altered = x.clone(); altered[0, 2:] += 30; altered[1, 3:] -= 30
    second = changed(altered, attention_mask=mask, **kwargs)[0]
    assert torch.equal(first[0, :2], second[0, :2])
    assert torch.equal(first[1, :3], second[1, :3])
    assert not torch.count_nonzero(attention[-1].masked_select(~allowed.expand_as(attention[-1])))
    assert torch.isfinite(first).all()


@pytest.mark.parametrize("mode", ["softmax", "quadratic"])
def test_checkpointed_and_ordinary_gradients_are_exact(mode):
    _, original = fixture()
    first = study.AttentionRetryStack.from_stack(original, mode=mode)
    second = deepcopy(first); second.checkpoint_blocks = True
    x, kwargs = inputs(torch.float64)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    ya, yb = first(a, **kwargs)[0], second(b, **kwargs)[0]
    assert torch.equal(ya, yb)
    ya.square().sum().backward(); yb.square().sum().backward()
    assert torch.equal(a.grad, b.grad)
    assert all(torch.equal(p.grad, q.grad) for p, q in zip(first.parameters(), second.parameters()))


@pytest.mark.parametrize("mode", ["softmax", "quadratic"])
def test_packed_fresh_reload_preserves_mode_and_rejects_old_loader(tmp_path, mode):
    cfg, original = fixture(torch.bfloat16, padding=64)
    stack = study.AttentionRetryStack.from_stack(original, mode=mode)
    stack.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16, strip_unused=True).eval()
    model = study.PackedAttentionRetryModel(stack, Qwen2RMSNorm(16).bfloat16(),
        torch.randn(32, 16).bfloat16(), Qwen2RotaryEmbedding(cfg), config=cfg, max_length=8)
    batch = dict(input_ids=torch.tensor([[1, 3, 5, 7]]), boundary_hidden=torch.randn(1, 4, 16).bfloat16())
    with torch.no_grad():
        expected = model.readout(model.hidden_forward(batch)[:, -1])
    receipt = study.export_packed_snapshot(model, tmp_path / "export", identities={"fixture":"native"})
    assert isinstance(model.stack, study.AttentionRetryStack)  # Serialization view did not replace live stack.
    assert receipt["attention_retry"]["layers"] == [35] and receipt["attention_retry"]["mode"] == mode
    assert receipt["embedding_copies"] == 1
    assert not (tmp_path / "export/tensor_store").exists()
    with study.packed.safe_open(tmp_path / "export/weights.safetensors", framework="pt") as stream:
        assert stream.metadata()["schema"] == study.SCHEMA
    fresh, proof = study.load_packed_snapshot(tmp_path / "export")
    with torch.no_grad():
        actual = fresh.readout(fresh.hidden_forward(batch)[:, -1])
    assert torch.equal(actual, expected) and proof["packed_state_exact"]
    assert fresh.stack.get_extra_state() == model.stack.get_extra_state()
    assert all(b.has_polynomial for b in fresh.stack.blocks) and fresh.stack.stripped
    assert not fresh.stack.enabled and fresh.inference_backend == "packed_bf16_torch_attention_retry"
    with pytest.raises(ValueError, match="completed packed snapshot"):
        study.packed.load_packed_snapshot(tmp_path / "export")
    manifest = tmp_path / "export/manifest.json"
    tampered = json.loads(manifest.read_text()); tampered["attention_retry"]["layers"] = [34]
    manifest.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="mode, layers"):
        study.load_packed_snapshot(tmp_path / "export")


class Dataset:
    def __init__(self):
        self.training, self.development = [0, 1], []
        data = training.module("distillation_data")
        provenance = dict(source_uri="fixture", source_revision="1", tokenizer_sha256="t",
            chat_template_sha256="c", prompt_provenance="fixture", response_provenance="reference",
            assistant_boundary_verified=True)
        self.records = [data.ResponseRecord(example_id=f"r{i}", group_id=f"g{i}", split="train", task="fixture",
            prompt_ids=(1, 3+i), response_ids=(6+i, 31), provenance=provenance,
            finish_reason="eos", eos_token_id=31) for i in range(2)]
        self.hidden = [torch.randn(4, 16).bfloat16() for _ in self.records]
        self.boundary = [torch.randn(4, 16).bfloat16() for _ in self.records]

    def batch(self, indices):
        index, = indices
        record = self.records[index]
        batch = training.module("distillation_data").response_batch([record], pad_token_id=0, max_length=8)
        return dict(batch, teacher_hidden=self.hidden[index][None], boundary_hidden=self.boundary[index][None],
            teacher_input_ids=batch["input_ids"].clone(), teacher_record_ids=[record.example_id])


@pytest.mark.parametrize("mode", ["softmax", "quadratic"])
def test_real_response_trainer_backward_update_and_raw_reload(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU preparation initialized CUDA"))
    cfg, original = fixture(torch.bfloat16)
    stack = study.AttentionRetryStack.from_stack(original, mode=mode)
    norm, E, rotary = Qwen2RMSNorm(16).bfloat16(), torch.randn(32, 16).bfloat16(), Qwen2RotaryEmbedding(cfg)
    dataset = Dataset()
    trainer = training.ResponseTrainer(stack, norm, E, rotary, dataset, teacher_id="native",
        config=training.ResponseConfig(learning_rate=2e-5, max_supervised_tokens=4,
            max_length=8, token_chunk_size=1, optimizer_chunk_bytes=64), host_reserve_bytes=0)
    try:
        before = trainer.parameter_fingerprints()
        rows = [trainer.train_window([i]) for i in (0, 1)]
        change = trainer.change_report(before)
        assert all(row["gradients_present_by_block"] == [True]*3 for row in rows)
        assert all(torch.isfinite(torch.tensor(row["optimizer"]["unclipped_gradient_norm"])) for row in rows)
        assert change["bf16_changed_by_block"] == [True]*3
        assert trainer.supervised_tokens == 4 and trainer.visited == {0, 1}
        assert training.offload.fixed_state(stack, E, rotary) == trainer.fixed
        assert all(torch.equal(p.detach().cpu().reshape(-1), trainer.optimizer.state[p]["master"].to(p.dtype).reshape(-1))
            for p in trainer.parameters)
        cp = trainer.save_checkpoint(tmp_path / "updated.pt", identities={"mode":mode})
        saved = torch.load(cp["path"], map_location="cpu", weights_only=True)
        assert saved["stack"]["_extra_state"]["attention_retry"]["mode"] == mode
        packed_stack = deepcopy(stack).pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16, strip_unused=True)
        packed_model = study.PackedAttentionRetryModel(packed_stack, deepcopy(norm), E, deepcopy(rotary), config=cfg, max_length=8)
        description = study.packed._describe(study._carrier(packed_model))
        fresh = study.restore_unpacked(saved["stack"], description)
        equal_tensors(saved["stack"], fresh.state_dict())
        assert fresh.get_extra_state() == stack.get_extra_state()
        fresh_model = training.ResponseModel(fresh, deepcopy(norm), E, deepcopy(rotary), max_length=8)
        trainer.model.eval(); fresh_model.eval()
        with torch.no_grad():
            assert torch.equal(trainer.model.hidden_forward(dataset.batch([0])), fresh_model.hidden_forward(dataset.batch([0])))
    finally:
        trainer.close()


def test_layer_identity_and_cross_mode_state_are_strict():
    _, original = fixture()
    for layers in ((36,), (35, 35), (35, 34), (True,), ()):
        with pytest.raises(ValueError, match="target layers"):
            study.AttentionRetryStack.from_stack(original, layers=layers)
    quadratic = study.AttentionRetryStack.from_stack(original)
    control = study.AttentionRetryStack.from_stack(original, mode="softmax")
    with pytest.raises(ValueError, match="mode/layers"):
        control.load_state_dict(quadratic.state_dict(), strict=True)
    x, kwargs = inputs(torch.float64)
    for model in (control, quadratic):
        with pytest.raises(ValueError, match="cache-free"):
            model(x, use_cache=True, **kwargs)
        with pytest.raises(ValueError, match="cache-free"):
            model(x, past_key_value=object(), **kwargs)
