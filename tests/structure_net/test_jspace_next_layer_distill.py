from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.structure_net import jspace_next_layer_distill as stage


def tiny(tmp_path):
    # Reuse the frozen actual-Qwen fixture, not a mocked training mechanism.
    path = Path(__file__).with_name("test_jspace_single_block_distill.py")
    spec = importlib.util.spec_from_file_location("single_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    block, rotary, native_pairs, _, native, suffix = module.fixture(tmp_path)
    inputs = np.load(tmp_path/"input.npy").view(np.int16)
    prefix = torch.from_numpy(inputs.copy()).view(torch.bfloat16).float().numpy()
    prefix = prefix + np.float32(.025)
    np.save(tmp_path/"prefix.npy", prefix)
    data = stage.PrefixPairs(native_pairs.items, tmp_path/"prefix.npy", tmp_path/"input.npy", tmp_path/"target.npy")
    native_pairs.close()
    def quality(live):
        frozen = deepcopy(live).pack(dtype=torch.float32, buffer_dtype=torch.float32)
        records = []
        for i in data.development:
            batch = data.batch([i], "cpu")
            h = frozen(batch["hidden"], attention_mask=batch["mask"], position_ids=batch["positions"],
                position_embeddings=rotary(batch["hidden"], batch["positions"]))
            mask = torch.zeros_like(batch["mask"], dtype=torch.float32).masked_fill(~batch["mask"], torch.finfo(torch.float32).min)
            h = suffix(h, attention_mask=mask, position_embeddings=rotary(h, batch["positions"]))[0]
            logits = h[0, -1, :2]
            case = data.items[i]
            records.append(dict(identity=case["identity"], family=case["family"], answer=case["answer"],
                prediction=int(logits.argmax()), logits=logits.tolist()))
        return dict(records=records)
    return block, rotary, data, quality, native, suffix


def test_original_update_normalizer_is_independent_of_student_prefix(tmp_path):
    block, rotary, data, quality, _, _ = tiny(tmp_path)
    config = stage.single.Config(epochs=1, microbatch=1, accumulation=2)
    normalization = data.fit_floor("cpu", config, lambda:None)
    expected = []
    for i in data.training:
        batch = data.batch([i], "cpu")
        assert not torch.equal(batch["hidden"], batch["native_hidden"])
        expected.extend((batch["target"]-batch["native_hidden"]).square().mean(-1)[batch["valid"]].tolist())
    assert normalization["floor"] == pytest.approx(torch.tensor(expected).median().item()*config.floor_fraction)
    data.close()
    changed = np.load(tmp_path/"prefix.npy")*1000
    changed[-4:] = np.nan  # Even poisoned dev inputs cannot influence training normalization.
    np.save(tmp_path/"prefix.npy", changed)
    data = stage.PrefixPairs(data.items, tmp_path/"prefix.npy", tmp_path/"input.npy", tmp_path/"target.npy")
    assert data.fit_floor("cpu", config, lambda:None) == normalization
    data.close()


def test_outside_floor_includes_prefix_error_but_denominator_does_not():
    native = torch.zeros(1, 2, 2)
    target = torch.tensor([[[1., 2.], [2., 3.]]])
    prefix = torch.tensor([[[0., 7.], [0., 8.]]])
    v = torch.tensor([[[.5], [1.]]], requires_grad=True)
    u = torch.tensor([[1.], [0.]])
    output = prefix+v@u.T
    batch = dict(hidden=prefix, native_hidden=native, target=target, valid=torch.ones(1, 2, dtype=torch.bool))
    values = stage.feature_losses(output, batch, floor=.01, basis=u)
    energy = target.square().mean(-1)
    expected_floor = torch.tensor([[25., 25.]])/2/energy
    torch.testing.assert_close(values["irreducible_outside_relative_mse"], expected_floor.mean())
    expected_loss = (output-target).square().mean(-1)/energy
    torch.testing.assert_close(values["loss"], expected_loss.mean())
    grad = torch.autograd.grad(values["loss"], v, retain_graph=True)[0]
    projected = (((output-target)@u).square().sum(-1)/2/energy).mean()
    torch.testing.assert_close(grad, torch.autograd.grad(projected, v)[0])


def test_actual_qwen_training_keeps_one_best_and_original_suffix_frozen(tmp_path):
    block, rotary, data, quality, native, suffix = tiny(tmp_path)
    suffix_before = deepcopy(suffix.state_dict())
    prefix_hash = stage.recipe.sha(tmp_path/"prefix.npy")
    trainer = stage.Trainer(block, rotary, data, quality,
        config=stage.single.Config(epochs=2, microbatch=1, accumulation=2))
    result = trainer.fit(tmp_path/"fit", native_correct=2, minimum_correct=0, layer=1)
    assert result["strict_reload_exact"] and result["updates"] == 2
    assert sorted(p.name for p in (tmp_path/"fit").glob("*.pt")) == ["selected.pt"]
    assert len(list((tmp_path/"fit").glob("epoch_*_development.json"))) == 3
    selected = stage.recipe.load(tmp_path/"fit/selected.pt")
    assert selected["epoch"] == result["selected_epoch"] and selected["layer"] == 1
    assert result["selected"]["correct"] == max(h["metrics"]["correct"] for h in result["history"])
    assert result["selected_checkpoint_sha256"] == stage.recipe.sha(tmp_path/"fit/selected.pt")
    assert stage.recipe.sha(tmp_path/"prefix.npy") == prefix_hash
    for key, tensor in suffix.state_dict().items():
        torch.testing.assert_close(tensor, suffix_before[key], atol=0, rtol=0)
    assert all(p.grad is None and not p.requires_grad for p in suffix.parameters())
    data.close()


def test_prefix_writer_matches_actual_packed_block_without_coordinate_reconstruction(tmp_path):
    block, rotary, data, _, _, _ = tiny(tmp_path)
    packed = deepcopy(block).pack(dtype=torch.float32, buffer_dtype=torch.float32)
    expected = []
    with torch.no_grad():
        for i in range(len(data.items)):
            b = data.batch([i], "cpu")
            value = packed(b["hidden"], attention_mask=b["mask"], position_ids=b["positions"],
                position_embeddings=rotary(b["hidden"], b["positions"]))
            expected.append(value[0])
    stage.write_prefix(packed, rotary, data.items, tmp_path/"prefix.npy", tmp_path/"next.npy", device="cpu", microbatch=1)
    actual = torch.from_numpy(np.load(tmp_path/"next.npy"))
    torch.testing.assert_close(actual, torch.cat(expected), atol=0, rtol=0)
    assert actual.shape[1] == 16 and actual.dtype == torch.float32
    with pytest.raises(FileExistsError):
        stage.write_prefix(packed, rotary, data.items, tmp_path/"prefix.npy", tmp_path/"next.npy", device="cpu")
    data.close()


def test_native_suffix_quality_uses_actual_prefix_and_correct_layer_offset(tmp_path, monkeypatch):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from experiments.structure_net import jspace_qwen_parallel_model as loaders
    block, rotary, data, _, _, suffix = tiny(tmp_path)
    root, source = tmp_path/"stage", tmp_path/"source"
    (root/"prefix").mkdir(parents=True); source.mkdir()
    np.save(root/"prefix/input.npy", np.load(tmp_path/"prefix.npy"))
    stage.recipe.write(source/"plan.json", dict(model=dict(path="tiny-native"), label_ids=[20, 21]))
    torch.manual_seed(31)
    embedding = torch.randn(23, 16).bfloat16().float()
    norm = Qwen2RMSNorm(16).eval()
    observed = []
    def native_loader(path, layer, **_):
        assert path == "tiny-native"
        observed.append(layer)
        return deepcopy(suffix)
    monkeypatch.setattr(loaders, "load_native_block", native_loader)
    monkeypatch.setattr(loaders, "load_readout_norm", lambda *_a, **_k:(embedding, norm, None))
    monkeypatch.setattr(stage.recipe, "rotary_module", lambda *_a, **_k:rotary)
    quality = stage.make_quality(dict(source=str(source), layer=34), data.items, root/"train", lambda:None, device="cpu")
    scored = quality(block)
    assert observed == [35] and len(scored["metrics"]["layers"]) == 2
    packed = deepcopy(block).pack(dtype=torch.float32, buffer_dtype=torch.float32)
    with torch.no_grad():
        for record, index in zip(scored["records"], data.development):
            batch = data.batch([index], "cpu")
            rope = rotary(batch["hidden"], batch["positions"])
            hidden = packed(batch["hidden"], attention_mask=batch["mask"],
                position_ids=batch["positions"], position_embeddings=rope)
            mask = torch.zeros_like(batch["mask"], dtype=torch.float32).masked_fill(~batch["mask"], torch.finfo(torch.float32).min)
            hidden = suffix(hidden, attention_mask=mask, position_embeddings=rope)[0]
            expected = torch.nn.functional.linear(norm(hidden[:, -1]), embedding[[20, 21]])[0]
            torch.testing.assert_close(torch.tensor(record["logits"]), expected, atol=2e-6, rtol=2e-6)
            assert record["identity"] == data.items[index]["identity"]
    data.close()


def accepted_fixture(tmp_path, monkeypatch):
    root = tmp_path/"previous"
    (root/"train").mkdir(parents=True)
    checkpoint = root/"train/epoch_00.pt"
    stage.recipe.save(checkpoint, dict(block={"tiny":torch.ones(1)}))
    records = [dict(identity=str(i), family="tiny", answer=0, prediction=int(i>=47),
        logits=[1., 0.] if i < 47 else [0., 1.]) for i in range(64)]
    metrics = stage.single.ranking_summary(records)
    metrics["relative_mse"] = .1
    row = root/"train/epoch_00_development.json"
    stage.recipe.write(row, dict(metrics=metrics, records=records))
    plan = dict(schema=stage.single.SCHEMA, fingerprint="parent", input_path=str(tmp_path/"input.npy"), config=dict(epochs=0))
    stage.recipe.write(root/"plan.json", plan)
    result = dict(status="completed", systems_only=False, strict_reload_exact=True, quality_pass=True,
        native_correct=50, minimum_correct=47, selected=metrics, selected_epoch=0,
        selected_checkpoint=str(checkpoint), history=[dict(epoch=0, metrics=metrics, checkpoint_sha256=stage.recipe.sha(checkpoint),
            development_sha256=stage.recipe.sha(row))])
    stage.recipe.write(root/"train/result.json", result)
    stage.recipe.write(root/"train/manifest.json", dict(status="completed", systems_only=False,
        fingerprint="parent", result_sha256=stage.recipe.sha(root/"train/result.json")))
    stage.recipe.write(root/"items.json", dict(items=[dict(identity=str(i), family="tiny", answer=0,
        choices=["A", "B"], input_ids=[1, 2], split="development") for i in range(64)]))
    verification = tmp_path/"independent"
    (verification/"verification").mkdir(parents=True)
    vplan = dict(schema="jspace.single-block-independent-verification.v1", campaign_fingerprint="parent",
        campaign=str(root), source_sha256={})
    vplan["fingerprint"] = stage.recipe.digest(vplan)
    stage.recipe.write(verification/"plan.json", vplan)
    stage.recipe.write(verification/"verification/result.json", dict(status="completed", passed=True,
        fingerprint=vplan["fingerprint"],
        campaign_fingerprint="parent", checkpoint_sha256=stage.recipe.sha(checkpoint), selected_epoch=0,
        strict_state_load=True, training_quality_pass=True, records=records))
    stage.recipe.write(verification/"verification/manifest.json", dict(status="completed", passed=True,
        fingerprint=vplan["fingerprint"], result_sha256=stage.recipe.sha(verification/"verification/result.json")))
    monkeypatch.setattr(stage.single, "verify", lambda _:plan)
    return root, verification, result


def test_next_stage_requires_passing_reloaded_predecessor_and_exact_checkpoint(tmp_path, monkeypatch):
    root, verification, result = accepted_fixture(tmp_path, monkeypatch)
    assert stage.accepted_predecessor(root, verification)["layer"] == 0
    with (root/"train/epoch_00.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint/qualification"):
        stage.accepted_predecessor(root, verification)
    result["quality_pass"] = False
    (root/"train/result.json").write_text(__import__("json").dumps(result))
    manifest = stage.recipe.read(root/"train/manifest.json")
    manifest["result_sha256"] = stage.recipe.sha(root/"train/result.json")
    (root/"train/manifest.json").write_text(__import__("json").dumps(manifest))
    with pytest.raises(ValueError, match="cumulative"):
        stage.accepted_predecessor(root, verification)


def test_independent_verification_is_required_even_after_internal_reload_pass(tmp_path, monkeypatch):
    root, verification, _ = accepted_fixture(tmp_path, monkeypatch)
    path = verification/"verification/manifest.json"
    manifest = stage.recipe.read(path)
    manifest["passed"] = False
    path.write_text(__import__("json").dumps(manifest))
    with pytest.raises(ValueError, match="independent verification"):
        stage.accepted_predecessor(root, verification)


def test_cpu_prepare_and_explicit_stage_guards_precede_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(stage, "accepted_predecessor", lambda *_:pytest.fail("Predecessor accessed before CPU guard"))
    with pytest.raises(ValueError, match="Hide CUDA"):
        stage.prepare(tmp_path, tmp_path, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="explicit"):
        stage.run(tmp_path, "all")


def test_cuda_allocator_cap_respects_visible_gpu_lease_and_keeps_guard_margin():
    calls = []
    cuda = SimpleNamespace(is_initialized=lambda:False,
        get_device_properties=lambda device:SimpleNamespace(total_memory=12*1024**3),
        set_per_process_memory_fraction=lambda fraction, device:calls.append((fraction, device)),
        memory=SimpleNamespace(get_allocator_backend=lambda:"native"))
    result = stage.cap_allocator(dict(id="tiny", allocation={"GPU-test":7168, "GPU-unrelated":8192}),
        cuda=cuda, visible="GPU-test")
    assert result["allocator_limit_bytes"] == (7168-256)*1024**2
    assert calls == [((7168-256)/(12*1024), 0)]
    with pytest.raises(ValueError, match="headroom"):
        stage.cap_allocator(dict(id="tiny", allocation={"GPU-test":128}), cuda=cuda, visible="GPU-test")
    assert len(calls) == 1


def test_reserved_predecessor_requires_its_runtime_completion_receipt(tmp_path, monkeypatch):
    from experiments.structure_net import jspace_single_block_reserved as reserved
    root, verification, _ = accepted_fixture(tmp_path, monkeypatch)
    (root/"runtime_policy.json").write_text("{}")
    monkeypatch.setattr(reserved, "verify", lambda _: (stage.recipe.read(root/"plan.json"), {}))
    def missing(*_):
        raise ValueError("missing runtime completion")
    monkeypatch.setattr(reserved, "verify_runtime_receipt", missing)
    with pytest.raises(ValueError, match="runtime completion"):
        stage.accepted_predecessor(root, verification)


@pytest.mark.parametrize("tamper,match", [
    ("development", "development reference hash"),
    ("winner", "selection rule"),
    ("order", "ordered epoch history"),
    ("count", "64-case development metrics"),
])
def test_verification_selection_preflight_blocks_reservation_on_tampering(tmp_path, monkeypatch, tamper, match):
    from experiments.structure_net import jspace_full_lease
    root, _, result = accepted_fixture(tmp_path, monkeypatch)
    (root/"train/epoch_00.pt").rename(root/"train/selected.pt")
    result["selected_checkpoint"] = str(root/"train/selected.pt")
    result["selected_checkpoint_sha256"] = stage.recipe.sha(root/"train/selected.pt")
    plan = dict(fingerprint="parent", config=dict(epochs=0), minimum_correct=47)
    def write_result():
        (root/"train/result.json").write_text(json.dumps(result))
        manifest = stage.recipe.read(root/"train/manifest.json")
        manifest["result_sha256"] = stage.recipe.sha(root/"train/result.json")
        (root/"train/manifest.json").write_text(json.dumps(manifest))
    write_result()
    assert stage.completed_training(root, plan)["selected_epoch"] == 0
    if tamper == "development":
        with (root/"train/epoch_00_development.json").open("a") as stream:
            stream.write("\n")
    elif tamper == "winner":
        result["selected_epoch"] = 1
    elif tamper == "order":
        result["history"][0]["epoch"] = 1
    else:
        result["history"][0]["metrics"]["count"] = 63
    write_result()
    monkeypatch.setattr(stage, "verify", lambda *_a, **_k:plan)
    monkeypatch.setattr(jspace_full_lease, "wait_ready", lambda *_:pytest.fail("Reservation attempted after tampered selection"))
    with pytest.raises(ValueError, match=match):
        stage.run(root, "verify")
    assert not (root/"verification").exists()
