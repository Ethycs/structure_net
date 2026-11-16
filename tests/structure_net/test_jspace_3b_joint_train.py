"""Actual compact Qwen gradients, bounded CPU accumulation, and Torch scoring."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_3b_joint_train as study
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


class ToyDataset:
    width = 8
    vocabulary = 16

    def __init__(self):
        self.training = list(range(16))
        self.development = [16, 17]
        self.lengths = [3 if i % 2 == 0 else 5 for i in range(18)]
        self.items = [dict(identity=f"case{i}", family="arc" if i % 2 else "boolq",
                           split="train" if i < 16 else "development", choices=["first", "second"],
                           answer=i % 2, targets=[1, 2, 3, [2, 7][i % 2]]) for i in range(18)]
        self.hidden = torch.randn(18, 5, 8).bfloat16()
        self.teacher = torch.randn(18, 4, 16).bfloat16()
        self.calls = []

    def batch(self, indices, *, device="cpu", dtype=torch.bfloat16):
        self.calls.append(list(indices))
        lengths = [self.lengths[i] for i in indices]
        positions = torch.arange(max(lengths), device=device)[None].expand(len(indices), -1)
        valid = positions < torch.tensor(lengths, device=device)[:, None]
        allowed = (positions[:, None, :] <= positions[:, :, None]) & valid[:, None, :]
        mask = torch.zeros(len(indices), 1, max(lengths), max(lengths), device=device, dtype=dtype)
        mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
        return dict(indices=list(indices), lengths=lengths,
                    hidden=self.hidden[indices, :max(lengths)].to(device=device, dtype=dtype),
                    position_ids=positions, attention_mask=mask,
                    loss_positions=torch.tensor([[0, 1, n - 2, n - 1] for n in lengths], device=device),
                    targets=torch.tensor([self.items[i]["targets"] for i in indices], device=device),
                    teacher=self.teacher[indices].to(device))


def components(checkpointing=True):
    torch.manual_seed(6621)
    cfg = Qwen2Config(hidden_size=8, intermediate_size=12, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=32, attention_dropout=0.)
    cfg._attn_implementation = "eager"
    blocks = []
    for layer in range(2):
        native = Qwen2DecoderLayer(cfg, layer).bfloat16().eval()
        basis = torch.linalg.qr(torch.randn(8, 8)).Q[:, :4]
        block = AffineQwenBlock(native, basis, torch.randn(6, 8) / 10,
                                trainable=True, parameter_dtype=torch.bfloat16)
        fit = dict(anchors=[0, 1], mean=torch.zeros(12), scale=torch.ones(12),
                   coefficients={"2": torch.randn(12, 6) / 20})
        block.compile_polynomial(fit, [10, 11])
        for parameter in block.parameters():
            parameter.data = parameter.data.bfloat16()
        blocks.append(block)
    stack = CompactQwenStack(blocks, geometry_mode="bf16", pad_to_multiple=0,
                             checkpoint_blocks=checkpointing)
    norm = Qwen2RMSNorm(8).bfloat16()
    E = torch.randn(16, 8).bfloat16().requires_grad_(False)
    rotary = Qwen2RotaryEmbedding(cfg).eval()
    data = ToyDataset()
    targets = study.feature.AnswerTargets(torch.randn(16, 4).bfloat16(), data.training)
    config = replace(study.CONFIG, optimizer_chunk_bytes=64, checkpoint_blocks=checkpointing)
    return stack, norm, E, rotary, data, targets, config


@pytest.fixture
def trainer():
    stack, norm, E, rotary, data, targets, config = components()
    instance = study.JointTrainer(stack, norm, E, rotary, data, [2, 7], targets=targets, alpha=study.broad.cache.ALPHA,
                                  config=config, host_reserve_bytes=0)
    try:
        yield instance
    finally:
        instance.close()


def masters(instance):
    return {id(p): s["master"].clone() for p, s in instance.optimizer.state.items()}


@pytest.mark.parametrize("checkpointing", [False, True])
def test_all_microbatch_leaf_gradients_reach_cpu_banks_and_change_fp32_masters(monkeypatch, checkpointing):
    stack, norm, E, rotary, data, targets, config = components(checkpointing)
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CPU training initialized CUDA"))
        instance = study.JointTrainer(stack, norm, E, rotary, data, [2, 7], targets=targets, alpha=study.broad.cache.ALPHA,
                                      config=config, host_reserve_bytes=0)
    try:
        before = masters(instance)
        fixed = study.offload.fixed_state(stack, E, rotary)
        identities = tuple(map(id, instance.parameters))
        observed = []
        step = instance.optimizer.step
        def inspect_step(**kwargs):
            assert all(p.grad is None for p in instance.parameters)
            assert all(instance.optimizer.received[p] == 16 for p in instance.parameters)
            assert all(instance.optimizer.cpu_gradients[p].device.type == "cpu" for p in instance.parameters)
            assert all(instance.optimizer.cpu_gradients[p].dtype == p.dtype for p in instance.parameters)
            observed.append(True)
            return step(**kwargs)
        monkeypatch.setattr(instance.optimizer, "step", inspect_step)
        result = instance.train_window(range(16))
        assert observed == [True] and data.calls == [[i] for i in range(16)]
        assert result["gradients_present_by_block"] == [True, True]
        assert result["update"] == 1 and result["cases"] == 16
        assert result["loss"] == pytest.approx(result["endpoint_loss"] + instance.alpha * result["feature_loss"], rel=1e-6)
        for block in stack.blocks:
            assert any(not torch.equal(instance.optimizer.state[p]["master"], before[id(p)]) for p in block.parameters())
            assert not torch.equal(instance.optimizer.state[block.polynomial_bias]["master"], before[id(block.polynomial_bias)])
        assert not torch.equal(instance.optimizer.state[norm.weight]["master"], before[id(norm.weight)])
        assert all(s["step"] == 1 for s in instance.optimizer.state.values())
        assert tuple(map(id, instance.parameters)) == identities
        assert study.offload.fixed_state(stack, E, rotary) == fixed
        assert all(not b.requires_grad and b.grad is None for b in stack.buffers())
        assert E.grad is None and not E.requires_grad
        assert all(p.grad is None for p in instance.parameters)
        assert instance.optimizer.cpu_gradients == instance.optimizer.received == {}
    finally:
        instance.close()


def test_rejects_nontraining_or_replaced_parameters_before_any_forward(trainer, monkeypatch):
    monkeypatch.setattr(trainer.optimizer, "step", lambda **kw: pytest.fail("invalid update stepped"))
    for indices in ([16], [], list(range(16)) + [0]):
        with pytest.raises(ValueError, match="TRAIN-only"):
            trainer.train_window(indices)
    assert not trainer.dataset.calls
    trainer.norm.weight = torch.nn.Parameter(trainer.norm.weight.detach().clone())
    with pytest.raises(RuntimeError, match="identities changed"):
        trainer.train_window([0])
    assert not trainer.dataset.calls


@pytest.mark.parametrize("failure", ["objective", "gradient"])
def test_nonfinite_training_never_updates_master_or_moments(trainer, monkeypatch, failure):
    before = {id(p): deepcopy(state) for p, state in trainer.optimizer.state.items()}
    if failure == "objective":
        trainer.dataset.teacher[0, 0, 0] = float("nan")
        monkeypatch.setattr(trainer.optimizer, "step", lambda **kw: pytest.fail("nonfinite objective stepped"))
        handle = None
    else:
        handle = trainer.parameters[0].register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    try:
        with pytest.raises(FloatingPointError, match="Nonfinite"):
            trainer.train_window([0])
    finally:
        if handle is not None:
            handle.remove()
    assert trainer.updates == 0
    for p, state in trainer.optimizer.state.items():
        for key, value in state.items():
            expected = before[id(p)][key]
            assert torch.equal(value, expected) if torch.is_tensor(value) else value == expected


def test_development_is_actual_unpacked_torch_model_and_original_readout(trainer, monkeypatch):
    expected_rows = []
    losses = []
    trainer.stack.eval()
    trainer.norm.eval()
    with torch.inference_mode():
        for index in trainer.dataset.development:
            batch = trainer.dataset.batch([index])
            output = trainer.stack(batch["hidden"], attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"], cache_position=batch["position_ids"][0],
                position_embeddings=trainer.rotary(batch["hidden"], batch["position_ids"]), use_cache=False)[0]
            selected = output[:, batch["loss_positions"][0]]
            logits = F.linear(trainer.norm(selected), trainer.readout).float()
            ce = F.cross_entropy(logits.reshape(-1, 16), batch["targets"].reshape(-1))
            logp = (logits / 2).log_softmax(-1)
            target = (batch["teacher"].float() / 2).log_softmax(-1)
            kl = (target.exp() * (target - logp)).sum(-1).mean()
            losses.append((float(.7 * 4 * kl + .3 * ce), float(kl), float(ce)))
            expected_rows.append(study.broad.cache.sanity._record(
                trainer.dataset.items[index], trainer.norm(output[:, -1]), trainer.readout, [2, 7]))
    state = study.offload.fixed_state(trainer.stack, trainer.readout, trainer.rotary)
    monkeypatch.setattr(trainer.stack, "pack", lambda *a, **k: pytest.fail("evaluation packed a copy"))
    monkeypatch.setattr(study.broad.cache.builder, "packed_clone", lambda *a, **k: pytest.fail("evaluation cloned packed model"))
    metrics, rows = trainer.development()
    assert rows == expected_rows and metrics["count"] == 2
    for column, key in enumerate(("loss", "kl", "ce")):
        assert metrics[key] == pytest.approx(sum(value[column] for value in losses) / 2, abs=0., rel=0.)
    assert metrics["correct"] == sum(r["answer"] == r["prediction"] for r in rows)
    assert study.offload.fixed_state(trainer.stack, trainer.readout, trainer.rotary) == state
    assert not trainer.stack.packed and not trainer.stack.stripped


def test_host_admission_and_reservation_precede_optimizer_allocation(monkeypatch):
    stack, norm, E, rotary, data, targets, config = components()
    monkeypatch.setattr(study.offload, "CPUGradientAdamW", lambda *a, **k: pytest.fail("optimizer allocated"))
    def denied():
        raise RuntimeError("lease denied")
    with pytest.raises(RuntimeError, match="lease denied"):
        study.JointTrainer(stack, norm, E, rotary, data, [2, 7], targets=targets, alpha=study.broad.cache.ALPHA,
                           config=config, guard=denied, host_reserve_bytes=0)
    monkeypatch.setattr(study.offload, "require_host_memory",
                        lambda *a, **k: (_ for _ in ()).throw(MemoryError("host limit")))
    with pytest.raises(MemoryError, match="host limit"):
        study.JointTrainer(stack, norm, E, rotary, data, [2, 7], targets=targets, alpha=study.broad.cache.ALPHA,
                           config=config, host_reserve_bytes=0)


def test_memory_accounting_includes_cpu_gradients_and_snapshot(trainer):
    n = sum(p.numel() for p in trainer.parameters)
    gradients = sum(p.numel() * p.element_size() for p in trainer.parameters)
    report = trainer.memory
    assert report["cpu_master_moments_bytes"] == n * 12
    assert report["cpu_gradient_bank_bytes"] == gradients
    assert report["cpu_additional_bound"] == n * 12 + gradients + report["snapshot_bytes"] + 6 * 64
    assert report["persistent_gpu_parameter_gradients"] is False


@pytest.mark.parametrize("defect", ["polynomial_dtype", "answer_bounds"])
def test_constructor_rejects_noncanonical_parameter_precision_and_labels(monkeypatch, defect):
    stack, norm, E, rotary, data, targets, config = components()
    if defect == "polynomial_dtype":
        stack.blocks[0].polynomial_bias.data = stack.blocks[0].polynomial_bias.data.float()
    else:
        data.items[0]["answer"] = 2
    monkeypatch.setattr(study.offload, "CPUGradientAdamW", lambda *a, **k: pytest.fail("invalid model allocated optimizer"))
    with pytest.raises(ValueError):
        study.JointTrainer(stack, norm, E, rotary, data, [2, 7], targets=targets, alpha=study.broad.cache.ALPHA,
                           config=config, host_reserve_bytes=0)


def test_endpoint_only_default_reaches_every_block_without_a_feature_target():
    stack, norm, E, rotary, data, _, config = components()
    instance = study.JointTrainer(stack, norm, E, rotary, data, [2, 7], config=config, host_reserve_bytes=0)
    try:
        before = masters(instance)
        result = instance.train_window(range(16))
        assert result["alpha"] == 0 and result["feature_loss"] == 0
        assert result["normalization_energy"] is None and instance.feature_targets is None
        assert result["loss"] == result["endpoint_loss"]
        assert result["gradients_present_by_block"] == [True, True]
        for block in stack.blocks:
            assert any(not torch.equal(before[id(p)], instance.optimizer.state[p]["master"]) for p in block.parameters())
        assert not torch.equal(before[id(norm.weight)], instance.optimizer.state[norm.weight]["master"])
    finally:
        instance.close()


@pytest.mark.parametrize("use_targets,alpha", [(True, 0.), (False, .2), (True, -1.), (False, float("nan"))])
def test_feature_target_and_coefficient_must_define_one_explicit_objective(monkeypatch, use_targets, alpha):
    stack, norm, E, rotary, data, targets, config = components()
    monkeypatch.setattr(study.offload, "CPUGradientAdamW", lambda *a, **k: pytest.fail("invalid objective allocated optimizer"))
    with pytest.raises(ValueError, match="targets disagree"):
        study.JointTrainer(stack, norm, E, rotary, data, [2, 7], targets=targets if use_targets else None,
                           alpha=alpha, config=config, host_reserve_bytes=0)
