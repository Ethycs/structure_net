"""CPU contracts; actual two-card transfers remain a reserved-pilot gate."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_joint_recovery as original
from experiments.structure_net import jspace_recovered_joint as recovered
from experiments.structure_net import jspace_semantic_recovery as semantic
from experiments.structure_net.jspace_semantic_loss import semantic_recovery_loss


class TinyData:
    """In-memory full-vocabulary FP32 targets, with binary and four-choice cases."""
    def __init__(self):
        self.items = []
        for i, length in enumerate((4, 6, 5, 4)):
            count = 2 if i % 2 == 0 else 4
            self.items.append(dict(identity=str(i), family=str(count),
                split="train" if i < 2 else "development", input_ids=[(i+j) % 19 for j in range(length)],
                choices=list("ABCD"[:count]), answer=(i+1) % count,
                loss_positions=[0, 1, length-2, length-1]))
        self.training, self.development = [0, 1], [2, 3]
        self.label_ids, self.vocabulary = [19, 20, 21, 22], 23
        self.targets = torch.randn(4, 2, self.vocabulary)

    def batch(self, indices, device):
        cases = [self.items[i] for i in indices]
        width = max(len(c["input_ids"]) for c in cases)
        ids = torch.zeros(len(cases), width, dtype=torch.long)
        for i, case in enumerate(cases): ids[i, :len(case["input_ids"])] = torch.tensor(case["input_ids"])
        positions = torch.arange(width).expand(len(cases), -1)
        valid = positions < torch.tensor([len(c["input_ids"]) for c in cases])[:, None]
        mask = (positions[:, :, None] >= positions[:, None, :]) & valid[:, :, None] & valid[:, None, :]
        return dict(indices=indices, input_ids=ids.to(device), position_ids=positions.to(device),
            attention_mask=mask[:, None].to(device), teacher=self.targets[indices].to(device),
            loss_positions=torch.tensor([c["loss_positions"] for c in cases], device=device))


def tiny_model(*, checkpoint=True):
    torch.manual_seed(628)
    config = Qwen2Config(hidden_size=16, intermediate_size=24, num_attention_heads=4,
        num_key_value_heads=2, num_hidden_layers=2, vocab_size=23,
        max_position_embeddings=32, attention_dropout=0.)
    config._attn_implementation = "eager"
    blocks = []
    for layer in range(2):
        native = Qwen2DecoderLayer(config, layer_idx=layer).bfloat16().float()
        fit = dict(basis=torch.linalg.qr(torch.randn(16, 8)).Q, means=torch.randn(1, 16)/7)
        blocks.append(semantic.build_enriched_block(native, fit, parameter_dtype=torch.float32))
    model = recovered.PartitionedCarryModel(torch.randn(23, 16)/3, blocks, torch.ones(16), 1e-6,
        Qwen2RotaryEmbedding(config=config), split=1, checkpoint_blocks=checkpoint)
    return model, TinyData()


def test_actual_two_block_routed_forward_and_checkpoint_gradients_match_sequential():
    model, data = tiny_model()
    sequential = original.JointCarryModel(model.embedding.clone(), deepcopy(list(model.blocks)),
        model.norm_gain.detach(), model.epsilon, deepcopy(model.rotary), checkpoint_blocks=False)
    batch = data.batch([0, 1], "cpu")
    args = [batch[k] for k in ("input_ids", "position_ids", "attention_mask")]
    positions = batch["loss_positions"][:, [1, 3]]
    actual, expected = model(*args, positions), sequential(*args, positions)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (2, 2, 23)
    weight = torch.randn_like(actual)
    (actual*weight).sum().backward(); (expected*weight).sum().backward()
    reference = dict(sequential.named_parameters())
    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, reference[name].grad, atol=0, rtol=0)
    assert all(sum(float(p.grad.abs().sum()) for p in b.parameters()) > 0 for b in model.blocks)
    assert model.norm_gain.grad.abs().sum() > 0
    assert not model.embedding.requires_grad and model.embedding.grad is None


def test_one_optimizer_window_changes_every_block_and_norm_but_not_tied_E_or_geometry():
    model, data = tiny_model()
    fixed = {name: value.clone() for name, value in model.named_buffers()}
    fixed_digest = recovered.fixed_fingerprint(model)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    trainer = recovered.Trainer(model, data, config=recovered.Config(
        epochs=1, microbatch=1, accumulation=2, learning_rate=1e-3, optimizer_chunk_bytes=128))
    with pytest.raises(ValueError, match="training identities"):
        trainer.train_window(data.development)
    result = trainer.train_window(data.training)
    assert result["update"] == 1
    for layer in range(2):
        assert any(not torch.equal(parameter, initial[name]) for name, parameter in model.named_parameters()
            if name.startswith(f"blocks.{layer}."))
    assert not torch.equal(model.norm_gain, initial["norm_gain"])
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, fixed[name], atol=0, rtol=0)
        assert not value.requires_grad
    assert "embedding" not in dict(model.named_parameters())
    assert all("lm_head" not in name for name, _ in model.named_parameters())
    assert result["gradient_parameter_count"] == len(initial)
    assert recovered.fixed_fingerprint(model) == fixed_digest
    with torch.no_grad(): model.blocks[0].basis[0, 0].add_(.001)
    assert recovered.fixed_fingerprint(model) != fixed_digest


def test_prospective_objective_keeps_conditional_choices_and_excludes_format_positions():
    config = recovered.Config()
    assert (config.epochs, config.microbatch, config.accumulation, config.learning_rate) == (3, 1, 16, 1e-5)
    assert config.seed == 27865 and 512 // (config.microbatch*config.accumulation)*config.epochs == 96
    assert (config.objective.temperature, config.objective.full_kl_weight,
        config.objective.choice_kl_weight, config.objective.choice_ce_weight) == (2., .2, .5, .3)
    torch.manual_seed(727)
    student = torch.randn(2, 4, 11, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    ids, counts, answers = [2, 4, 6, 8], torch.tensor([2, 4]), torch.tensor([1, 3])
    values = semantic_recovery_loss(student, teacher, label_ids=ids, choice_counts=counts,
        answers=answers, config=config.objective)
    manual = []
    for i, count in enumerate(counts.tolist()):
        s, t = student[i, [1, 3]], teacher[i, [1, 3]].detach()
        p, logp = (t/2).softmax(-1), (t/2).log_softmax(-1)
        sc, tc = s[-1, ids[:count]], t[-1, ids[:count]]
        full = (p*(logp-(s/2).log_softmax(-1))).sum(-1).mean()
        choice = ((tc/2).softmax(-1)*((tc/2).log_softmax(-1)-(sc/2).log_softmax(-1))).sum()
        manual.append(.8*full + 2*choice + .3*F.cross_entropy(sc[None], answers[i:i+1]))
    torch.testing.assert_close(values["loss"], torch.stack(manual).mean())
    values["loss"].backward()
    assert teacher.grad is None and not student.grad[:, [0, 2]].count_nonzero()
    assert student.grad[:, 1, 10].abs().sum() > 0  # Full vocabulary, not a four-row replacement.


def test_selection_is_accuracy_first_then_conditional_CE_not_full_KL():
    good = dict(correct=31, choice_ce=2., full_kl=100.)
    wrong = dict(correct=30, choice_ce=.001, full_kl=.001)
    assert recovered.selection_key(good) > recovered.selection_key(wrong)
    assert recovered.selection_key(dict(correct=31, choice_ce=1., full_kl=100.)) > recovered.selection_key(good)
    assert recovered.selection_key(dict(correct=31, choice_ce=2., full_kl=.001)) == recovered.selection_key(good)


def test_initial_block_loads_recovered_values_and_the_distinct_block_zero_basis(tmp_path):
    torch.manual_seed(516)
    config = Qwen2Config(hidden_size=16, intermediate_size=24, num_attention_heads=4,
        num_key_value_heads=2, num_hidden_layers=2, vocab_size=23, attention_dropout=0.)
    config._attn_implementation = "eager"
    native = Qwen2DecoderLayer(config, layer_idx=0).bfloat16().float()
    fits = [dict(basis=torch.linalg.qr(torch.randn(16, 8)).Q, means=torch.randn(1, 16)/4)
        for _ in range(2)]
    for layer, fit in enumerate(fits):
        selected = semantic.build_enriched_block(deepcopy(native), fit, parameter_dtype=torch.float32)
        initial = next(selected.parameters()).detach().clone()
        with torch.no_grad(): next(selected.parameters()).add_(.125*(layer+1))
        fit_path = tmp_path/("first-block-fit.pt" if layer == 0 else "all-layer-fit.pt")
        checkpoint = tmp_path/f"selected-{layer}.pt"
        original.recipe.save(fit_path, fit)
        original.recipe.save(checkpoint, dict(block=original._cpu(selected.state_dict()), epoch=7-layer, layer=layer))
        reference = dict(fit_path=str(fit_path), checkpoint=str(checkpoint),
            checkpoint_sha256=original.recipe.sha(checkpoint), selected_epoch=7-layer)
        loaded = recovered.load_initial_block(deepcopy(native), reference, layer=layer)
        torch.testing.assert_close(loaded.basis, fit["basis"], atol=0, rtol=0)
        assert not torch.equal(next(loaded.parameters()), initial)
        for (name, value), (other, expected) in zip(loaded.named_parameters(), selected.named_parameters()):
            assert name == other and value.dtype == torch.float32 and value.requires_grad
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
        x = torch.randn(1, 4, 16); positions = torch.arange(4)[None]
        rotary = Qwen2RotaryEmbedding(config=config)
        kwargs = dict(position_ids=positions, position_embeddings=rotary(x, positions),
            attention_mask=torch.ones(1, 1, 4, 4, dtype=torch.bool).tril())
        torch.testing.assert_close(loaded(x, **kwargs), selected(x, **kwargs), atol=0, rtol=0)
    with open(checkpoint, "ab") as stream: stream.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        recovered.load_initial_block(deepcopy(native), reference, layer=1)


def test_initializer_is_eligible_and_selected_checkpoint_reloads_exactly(tmp_path, monkeypatch):
    model, data = tiny_model()
    trainer = recovered.Trainer(model, data, config=recovered.Config(
        epochs=1, microbatch=1, accumulation=2, learning_rate=1e-3, optimizer_chunk_bytes=128))
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    _, initial_records = trainer.development()
    for index, record in zip(data.development, initial_records):
        data.items[index]["answer"] = record["prediction"]
    baseline, baseline_records = trainer.development()
    assert baseline["correct"] == 2
    original_development = trainer.development
    # Deliberately make the trained epoch worse, isolating the selection
    # contract from whether this tiny random task happens to learn in one step.
    calls = []
    def scripted_development():
        metrics, records = original_development()
        calls.append(1)
        if len(calls) == 2:
            metrics = dict(metrics, correct=0, accuracy=0., choice_ce=0.)
        return metrics, records
    monkeypatch.setattr(trainer, "development", scripted_development)
    result = trainer.fit(tmp_path/"fit")
    assert result["selected_epoch"] == 0 and result["updates"] == 1
    assert result["strict_reload_exact"] and result["selected"] == baseline
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, initial[name], atol=0, rtol=0)
    fresh, _ = tiny_model()
    original.load_selected(fresh, result["selected_directory"])
    fresh_trainer = recovered.Trainer(fresh, data, config=trainer.config)
    assert fresh_trainer.development() == (baseline, baseline_records)


class FakeCuda:
    def __init__(self):
        self.events = []
        self.memory = SimpleNamespace(get_allocator_backend=lambda:"native")
    def is_initialized(self): return False
    def device_count(self): return 2
    def get_device_properties(self, index):
        self.events.append(("properties", index))
        return SimpleNamespace(total_memory=(8 if index == 0 else 12)*1024**3)
    def set_per_process_memory_fraction(self, fraction, device):
        self.events.append(("fraction", fraction, device))


def test_two_gpu_allocator_maps_reversed_UUIDs_and_caps_each_lease_separately():
    front, tail = "GPU-0370fdb0-test", "GPU-7b1db3cb-test"
    lease = dict(id="test-lease", allocation={front:11264, tail:7168})
    fake = FakeCuda()
    result = recovered.allocator_settings(lease, cuda=fake, visible=f"{tail},{front}")
    settings = {row["physical_uuid"]:row for row in result["settings"]}
    assert settings[front]["allocator_limit_bytes"] == (11264-256)*1024**2
    assert settings[tail]["allocator_limit_bytes"] == (7168-256)*1024**2
    assert settings[front]["memory_fraction"] == pytest.approx(11008/12288)
    assert settings[tail]["memory_fraction"] == pytest.approx(6912/8192)
    assert result["physical_by_logical"] == [tail, front]
    assert set(result["devices"]) == {"3060", "2060_SUPER"}
    assert len([e for e in fake.events if e[0] == "fraction"]) == 2
    unleased = FakeCuda()
    with pytest.raises((ValueError, RuntimeError)):
        recovered.allocator_settings(dict(id="bad", allocation={front:11264}),
            cuda=unleased, visible=f"{tail},{front}")
    assert unleased.events == []  # No device properties or CUDA allocator touched.


def test_denied_reservation_cannot_construct_model_or_initialize_CUDA(tmp_path, monkeypatch):
    from experiments.structure_net import jspace_full_lease
    monkeypatch.setattr(recovered, "verify", lambda _:dict(fingerprint="test", config={}))
    def denied(_): raise RuntimeError("reservation denied")
    monkeypatch.setattr(jspace_full_lease, "wait_ready", denied)
    monkeypatch.setattr(recovered, "allocator_settings", lambda *_a, **_k:pytest.fail("CUDA before lease"))
    monkeypatch.setattr(recovered, "build_model", lambda *_a, **_k:pytest.fail("Model before lease"))
    monkeypatch.setattr(torch.cuda, "init", lambda:pytest.fail("CUDA before lease"))
    with pytest.raises(RuntimeError, match="reservation denied"):
        recovered.run(tmp_path, "pilot")
    receipt = original.recipe.read(tmp_path/"pilot/failure.json")
    assert receipt["status"] == "failed" and receipt["lease_id"] is None


def test_initializer_control_requires_27_correct_and_actual_reference_logits():
    cases = [dict(identity=str(i), family="tiny", answer=0, choices=["A", "B"]) for i in range(64)]
    records = [dict(identity=c["identity"], family=c["family"], answer=0, prediction=int(i >= 27),
        logits=[1., 0.] if i < 27 else [0., 1.]) for i, c in enumerate(cases)]
    reference = dict(records=deepcopy(records))
    assert recovered.baseline_control(records, reference, cases)["passed"]
    records[0]["logits"][0] = 2.
    failed = recovered.baseline_control(records, reference, cases)
    assert failed["predictions_exact"] and not failed["logits_pass"] and not failed["passed"]


def test_training_authenticates_successful_pilot_before_reservation(tmp_path, monkeypatch):
    from experiments.structure_net import jspace_full_lease
    plan = dict(fingerprint="matching")
    monkeypatch.setattr(recovered, "verify", lambda _:plan)
    monkeypatch.setattr(jspace_full_lease, "wait_ready", lambda *_:pytest.fail("Lease before pilot gate"))
    with pytest.raises(FileNotFoundError): recovered.run(tmp_path, "train")
    pilot = tmp_path/"pilot"; pilot.mkdir()
    result = dict(status="completed", systems_only=True, updates=1,
        initial_control=dict(passed=True, correct=27), fixed_state_unchanged=True,
        parameter_changes=dict(changed_blocks=list(range(36)), unchanged_blocks=[]))
    original.recipe.write(pilot/"result.json", result)
    original.recipe.write(pilot/"manifest.json", dict(status="completed", fingerprint="matching",
        result_sha256=original.recipe.sha(pilot/"result.json")))
    recovered.verify_pilot(tmp_path, plan)
    changed = dict(result, updates=2)
    (pilot/"result.json").write_text(__import__("json").dumps(changed))
    with pytest.raises(ValueError, match="pilot required"):
        recovered.run(tmp_path, "train")
