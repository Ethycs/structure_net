from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments.structure_net import jspace_next_projection as diagnostic


def banks(tmp_path):
    native = torch.tensor([[float("nan")]*4, [1., 2., 3., 4.], [2., 3., 4., 5.],
                           [3., 4., 5., 6.], [4., 5., 6., 7.], [5., 6., 7., 8.]])
    target = native + torch.tensor([1., 2., 3., 4.])
    # A non-BF16-representable prefix error must survive outside the update basis.
    prefix = native + torch.tensor([.125, -.25, .0012345, .1234567])
    np.save(tmp_path/"prefix.npy", prefix.numpy())
    for name, tensor in (("native", native), ("target", target)):
        np.save(tmp_path/f"{name}.npy", tensor.bfloat16().view(torch.int16).numpy())
    items = [dict(identity="train", family="tiny", split="train", input_ids=[7], choices=["a", "b"], answer=0, offset=0, stop=1),
        dict(identity="dev0", family="tiny", split="development", input_ids=[9, 8], choices=["a", "b"], answer=1, offset=1, stop=3),
        dict(identity="dev1", family="tiny", split="development", input_ids=[8, 6, 7], choices=["a", "b"], answer=0, offset=3, stop=6)]
    return prefix, native, target, items, torch.eye(4)[:, :2]


def arguments(tmp_path, items, basis):
    return (items[1:], items, tmp_path/"prefix.npy", tmp_path/"native.npy", tmp_path/"target.npy", basis)


def test_surrogate_preserves_prefix_complement_and_native_normalization(tmp_path):
    prefix, native, target, items, basis = banks(tmp_path)
    table, virtual, geometry = diagnostic.reconstruct(*arguments(tmp_path, items, basis),
        arm="projected_correction", normalization_floor=.01, chunk_tokens=1)
    expected = prefix[1:] + ((target[1:]-prefix[1:])@basis)@basis.T
    torch.testing.assert_close(table, expected, atol=0, rtol=0)
    torch.testing.assert_close(table[:, 2:], prefix[1:, 2:], atol=0, rtol=0)
    assert not torch.equal(table[:, 2:], table[:, 2:].bfloat16().float())
    torch.testing.assert_close((target[1:]-table)@basis, torch.zeros(5, 2), atol=0, rtol=0)
    for row, case in zip(geometry["records"], items[1:]):
        x, t, h = (v[case["offset"]:case["stop"]] for v in (prefix, target, native))
        y = diagnostic.project_update(x, t, basis)
        relative = (y-t).square().mean(-1)/(t-h).square().mean(-1)
        assert row["relative_mse"] == pytest.approx(float(relative.mean()))
        assert row["original_teacher_update_energy"] == 7.5
        assert row["student_to_target_energy"] != row["original_teacher_update_energy"]
    assert [c["input_ids"] for c in virtual] == [[0, 1], [2, 3, 4]]
    assert items[1]["input_ids"] == [9, 8]
    assert geometry["total_tokens"] == 5  # Poisoned training row was never read.


def test_native_control_and_full_rank_projection_match_teacher(tmp_path):
    _, _, target, items, basis = banks(tmp_path)
    control, virtual, geometry = diagnostic.reconstruct(*arguments(tmp_path, items, basis), arm="native_boundary", normalization_floor=.01)
    torch.testing.assert_close(control, target[1:], atol=0, rtol=0)
    assert geometry["mean_case_mse"] == geometry["mean_case_relative_mse"] == 0
    full, other, _ = diagnostic.reconstruct(*arguments(tmp_path, items, torch.eye(4)), arm="projected_correction", normalization_floor=.01)
    torch.testing.assert_close(full, control, atol=1e-6, rtol=0)
    assert virtual == other


def test_surrogate_feature_optimum_does_not_imply_task_accuracy_upper_bound():
    # Native target prefers label0. Euclidean projection removes that direction;
    # a non-optimal in-basis update restores label0, so accuracy is not bounded.
    x, target, basis = torch.zeros(1, 2), torch.tensor([[1., 4.]]), torch.tensor([[1.], [0.]])
    surrogate = diagnostic.project_update(x, target, basis)
    alternative = torch.tensor([[3., 0.]])
    head = lambda h: torch.stack((h[:, 0]+h[:, 1], torch.full_like(h[:, 0], 2.)), -1)
    assert int(head(target).argmax()) == int(head(alternative).argmax()) == 0
    assert int(head(surrogate).argmax()) == 1
    assert (alternative-target).square().sum() > (surrogate-target).square().sum()


def test_projection_rejects_changed_split_basis_and_prefix_precision(tmp_path):
    _, _, _, items, basis = banks(tmp_path)
    with pytest.raises(ValueError, match="orthonormal"):
        diagnostic.reconstruct(*arguments(tmp_path, items, basis*2), arm="projected_correction", normalization_floor=.01)
    with pytest.raises(ValueError, match="reservation"):
        diagnostic.reconstruct(*arguments(tmp_path, items, basis), arm="projected_correction", normalization_floor=.01, device="cuda:0")
    cases = deepcopy(items[1:]); cases[0]["split"] = "train"
    with pytest.raises(ValueError, match="split"):
        diagnostic.reconstruct(cases, *arguments(tmp_path, items, basis)[1:], arm="native_boundary", normalization_floor=.01)
    np.save(tmp_path/"prefix.npy", np.load(tmp_path/"prefix.npy").astype(np.float16))
    with pytest.raises(ValueError, match="FP32 prefix"):
        diagnostic.reconstruct(*arguments(tmp_path, items, basis), arm="native_boundary", normalization_floor=.01)


def test_real_causal_suffix_matches_direct_variable_length_execution(tmp_path):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding, Qwen2RMSNorm
    _, _, _, items, basis = banks(tmp_path)
    table, virtual, _ = diagnostic.reconstruct(*arguments(tmp_path, items, basis), arm="projected_correction", normalization_floor=.01)
    torch.manual_seed(81)
    config = Qwen2Config(hidden_size=4, intermediate_size=12, num_attention_heads=2,
        num_key_value_heads=1, num_hidden_layers=4, vocab_size=13)
    config._attn_implementation = "eager"
    layers = [Qwen2DecoderLayer(config, i).eval() for i in (2, 3)]
    rotary, norm, embedding = Qwen2RotaryEmbedding(config), Qwen2RMSNorm(4), torch.randn(13, 4)
    seen = []
    def loader(i):
        seen.append(i+2)
        return deepcopy(layers[i])
    result = diagnostic.stage.single.stream_score(virtual, input_embedding=table, block_loader=loader,
        final_readout=(norm, embedding), label_ids=[10, 11], scratch_dir=tmp_path/"scratch",
        rotary=rotary, layer_count=2, device="cpu", microbatch=2)
    assert seen == [2, 3]
    with torch.no_grad():
        for record, case in zip(result["records"], virtual):
            hidden = table[case["input_ids"]][None]
            positions = torch.arange(hidden.shape[1])[None]
            causal = positions[:, :, None] >= positions[:, None, :]
            mask = torch.zeros_like(causal[:, None], dtype=torch.float32).masked_fill(~causal[:, None], torch.finfo(torch.float32).min)
            for layer in layers:
                hidden = layer(hidden, attention_mask=mask, position_embeddings=rotary(hidden, positions))[0]
            expected = torch.nn.functional.linear(norm(hidden[:, -1]), embedding[[10, 11]])[0]
            torch.testing.assert_close(torch.tensor(record["logits"]), expected, atol=2e-6, rtol=2e-6)
            assert all(record[k] == case[k] for k in ("identity", "family", "answer"))
    assert not (tmp_path/"scratch").exists()


def test_prepare_requires_cuda_hidden_and_completed_prefix_before_writing(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(diagnostic.stage, "verify", lambda *_a, **_k:pytest.fail("Parent checked before CPU guard"))
    with pytest.raises(ValueError, match="Hide CUDA"):
        diagnostic.prepare(tmp_path/"new", tmp_path/"campaign")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    def incomplete(_path, *, prefix):
        assert prefix is True
        raise ValueError("A completed matching accepted-prefix cache is required")
    monkeypatch.setattr(diagnostic.stage, "verify", incomplete)
    with pytest.raises(ValueError, match="completed"):
        diagnostic.prepare(tmp_path/"new", tmp_path/"campaign")
    assert not (tmp_path/"new").exists()


def test_reservation_denial_precedes_cuda_and_preserves_failure_receipt(tmp_path, monkeypatch):
    from experiments.structure_net import jspace_full_lease
    monkeypatch.setattr(diagnostic, "verify", lambda _:(dict(arms=list(diagnostic.ARMS), fingerprint="frozen"), [], []))
    def denied(_devices):
        raise RuntimeError("No reservation")
    monkeypatch.setattr(jspace_full_lease, "wait_ready", denied)
    monkeypatch.setattr(torch.cuda, "init", lambda:pytest.fail("CUDA before reservation"))
    with pytest.raises(RuntimeError, match="reservation"):
        diagnostic.run(tmp_path, "native_boundary")
    receipt = diagnostic.recipe.read(tmp_path/"native_boundary/failure.json")
    assert receipt["status"] == "failed" and receipt["lease_id"] is None and receipt["peak_reserved_bytes"] is None
    with pytest.raises(FileExistsError):
        diagnostic.run(tmp_path, "native_boundary")


def test_input_hash_guard_rejects_mutation_before_execution(tmp_path, monkeypatch):
    input_file = tmp_path/"frozen.bin"; input_file.write_bytes(b"original")
    plan = dict(schema=diagnostic.SCHEMA, campaign="campaign", campaign_fingerprint="parent",
        input_sha256={str(input_file):diagnostic.recipe.sha(input_file)})
    plan["fingerprint"] = diagnostic.recipe.digest(plan)
    diagnostic.recipe.write(tmp_path/"plan.json", plan)
    monkeypatch.setattr(diagnostic.stage, "verify", lambda *_a, **_k:dict(fingerprint="parent"))
    input_file.write_bytes(b"modified")
    with pytest.raises(ValueError, match="Changed diagnostic input"):
        diagnostic.verify(tmp_path)


def test_training_floor_is_explicit_and_matches_training_loss_without_dev_fit(tmp_path):
    prefix, native, target, items, basis = banks(tmp_path)
    table, _, geometry = diagnostic.reconstruct(*arguments(tmp_path, items, basis),
        arm="projected_correction", normalization_floor=100.)
    assert geometry["training_derived_normalization_floor"] == 100.
    losses = []
    cursor = 0
    for case in items[1:]:
        length = len(case["input_ids"])
        actual = diagnostic.stage.single.feature_losses(table[cursor:cursor+length][None],
            native[case["offset"]:case["stop"]][None], target[case["offset"]:case["stop"]][None],
            torch.ones(1, length, dtype=torch.bool), floor=100.)
        losses.append(float(actual["loss"]))
        cursor += length
    assert geometry["loss"] == pytest.approx(sum(losses)/len(losses))
    assert geometry["loss"] == geometry["mean_case_relative_mse"]
    for invalid in (0, -1, float("nan"), True):
        with pytest.raises(ValueError, match="normalization floor"):
            diagnostic.reconstruct(*arguments(tmp_path, items, basis), arm="native_boundary", normalization_floor=invalid)


def test_normalizer_is_loaded_only_from_authenticated_matching_pilot(tmp_path, monkeypatch):
    pilot = tmp_path/"pilot"; pilot.mkdir()
    normalization = dict(train_case_count=512, train_token_count=69130,
        median_teacher_update_energy=.05, floor=.05*1e-4, floor_fraction=1e-4)
    result = dict(status="completed", systems_only=True, updates=1, normalization=normalization)
    diagnostic.recipe.write(pilot/"result.json", result)
    manifest = dict(status="completed", systems_only=True, fingerprint="campaign",
        result_sha256=diagnostic.recipe.sha(pilot/"result.json"), allocation={"GPU-test":7168}, peak_reserved_bytes=0)
    diagnostic.recipe.write(pilot/"manifest.json", manifest)
    monkeypatch.setattr(np, "load", lambda *_a, **_k:pytest.fail("Diagnostic tried fitting a new normalizer"))
    parent = dict(fingerprint="campaign", config=dict(floor_fraction=1e-4))
    assert diagnostic.pilot_normalization(tmp_path, parent) == normalization
    with pytest.raises(ValueError, match="matching"):
        diagnostic.pilot_normalization(tmp_path, dict(parent, fingerprint="other"))
    result["normalization"]["floor"] = 1e-12
    (pilot/"result.json").write_text(__import__("json").dumps(result))
    manifest["result_sha256"] = diagnostic.recipe.sha(pilot/"result.json")
    (pilot/"manifest.json").write_text(__import__("json").dumps(manifest))
    with pytest.raises(ValueError, match="TRAIN-only"):
        diagnostic.pilot_normalization(tmp_path, parent)
