"""Contracts for the expanded-data, single-new-block calibration adapter."""
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_broad_step_capture as study
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock


def tiny_rows():
    rows = []
    offset = 0
    for index, ids in enumerate(([1, 2, 3, 4], [3, 1, 2], [2, 5, 1])):
        length = len(ids)
        legacy = [0, 0, length - 2, length - 1]
        content = [1, length - 2, 1, length - 1]
        row = dict(identity=f"case{index}", family="arc", split="development" if index == 2 else "train",
                   input_ids=list(ids), choices=["one", "two"], answer=index % 2,
                   offset=offset, stop=offset + length,
                   legacy_loss_positions=legacy, content_loss_positions=content,
                   legacy_targets=[ids[p + 1] for p in legacy[:-1]] + [6 + index % 2],
                   content_targets=[ids[p + 1] for p in content[:-1]] + [6 + index % 2])
        row.update(loss_positions=legacy.copy(), targets=row["legacy_targets"].copy())
        rows.append(row)
        offset += length
    return rows


def tiny_teacher(tmp_path):
    torch.manual_seed(832)
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_attention_heads=2,
                        num_key_value_heads=1, num_hidden_layers=3, vocab_size=16,
                        attention_dropout=0., max_position_embeddings=32, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).bfloat16().eval().requires_grad_(False)
    parent = AffineQwenBlock(model.model.layers[2], torch.eye(8)[:, :4], torch.zeros(1, 8),
                             trainable=False)
    parent.pack(dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
    model.model.layers[2] = parent
    rows = tiny_rows()
    boundaries = []
    for case in rows:
        with torch.no_grad():
            output = model.model(input_ids=torch.tensor([case["input_ids"]]),
                                 use_cache=False, output_hidden_states=True)
            boundaries.append(output.hidden_states[1][0].clone())
    path = tmp_path / "boundary.npy"
    np.save(path, torch.cat(boundaries).view(torch.int16).numpy())
    return model, rows, path


def test_content_aliases_preserve_prompt_order_offsets_and_original_targets():
    original = tiny_rows()
    before = deepcopy(original)
    rows = study.content_items(original)
    assert original == before
    for source, row in zip(original, rows):
        assert row["input_ids"] == source["input_ids"]
        assert (row["offset"], row["stop"], row["identity"]) == (
            source["offset"], source["stop"], source["identity"])
        assert row["loss_positions"] == source["content_loss_positions"]
        assert row["targets"] == source["content_targets"]
        assert row["legacy_loss_positions"] == source["legacy_loss_positions"]
    rows[0]["loss_positions"][0] = 999
    rows[0]["targets"][0] = 999
    assert original == before


def production_fixture(tmp_path, monkeypatch):
    rows = []
    calibration = []
    template = tiny_rows()[0]
    for split, count in (("train", 1152), ("development", 64)):
        for family in study.FAMILIES:
            for index in range(count):
                row = deepcopy(template)
                row.update(identity=f"{split}-{family}-{index}", family=family, split=split,
                           offset=4 * len(rows), stop=4 * (len(rows) + 1))
                if split == "train" and index < 32:
                    calibration.append(len(rows))
                rows.append(row)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    study.base.write(cache_root / "items.json", {"items": rows})
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"authenticated selected parent")
    plan = dict(cut=21, allocation_mib=11264, allocator_mib=11008,
                cache_root=str(cache_root), data_root=str(tmp_path / "data"),
                model={"path": "native-model"}, label_ids=[6, 7, 8, 9],
                training_order=list(reversed(range(4608))), calibration_indices=calibration,
                parent_fits={str(i): f"fit-{i}" for i in range(22, 36)},
                parent_checkpoint=str(checkpoint), parent_checkpoint_sha256=study.base.sha(checkpoint),
                expected_parent_fixed_state={"sha256": "a" * 64, "tensor_bytes": 100},
                boundary_root=str(tmp_path / "boundary"), capture_root=str(tmp_path / "capture"),
                fit_root=str(tmp_path / "fit"))
    cache_plan = {k: deepcopy(plan[k]) for k in ("model", "label_ids", "data_root", "training_order")}
    cache_plan["fits"] = deepcopy(plan["parent_fits"])
    monkeypatch.setattr(study.cache, "verify", lambda root: cache_plan)
    return plan, rows


def test_production_accepts_expanded_data_without_relaxing_train_only_calibration(tmp_path, monkeypatch):
    plan, source = production_fixture(tmp_path, monkeypatch)
    rows = study.validate_plan_items(plan)
    assert len(rows) == 4864 and len(plan["calibration_indices"]) == 128
    assert all(rows[i]["split"] == "train" for i in plan["calibration_indices"])
    assert rows[0]["loss_positions"] == source[0]["content_loss_positions"]
    assert study.calibration.calibration_seed(21) == 820861
    fingerprint = study.teacher_fingerprint(plan, rows)
    changed = deepcopy(rows)
    changed[0]["loss_positions"][0] = 0
    assert study.teacher_fingerprint(plan, changed) != fingerprint
    assert study.teacher_fingerprint(dict(plan, parent_checkpoint_sha256="b" * 64), rows) != fingerprint


@pytest.mark.parametrize("defect", ["development_j", "duplicate_j", "unbalanced_j", "parent", "nested", "small"])
def test_production_rejects_dataset_or_parent_substitution(tmp_path, monkeypatch, defect):
    plan, rows = production_fixture(tmp_path, monkeypatch)
    if defect == "development_j":
        plan["calibration_indices"][0] = 4608
    elif defect == "duplicate_j":
        plan["calibration_indices"][0] = plan["calibration_indices"][1]
    elif defect == "unbalanced_j":
        plan["calibration_indices"][0] = 1152 + 32
    elif defect == "parent":
        Path(plan["parent_checkpoint"]).write_bytes(b"changed selected weights")
    elif defect == "nested":
        plan["fit_root"] = str(tmp_path / "capture" / "fit")
    else:
        small = rows[:512] + rows[-64:]
        for index, row in enumerate(small):
            row.update(offset=4 * index, stop=4 * (index + 1))
        (tmp_path / "cache" / "items.json").write_text(json.dumps({"items": small}))
    with pytest.raises(ValueError):
        study.validate_plan_items(plan)


def test_actual_native_boundaries_are_singleton_bf16_and_hooks_are_removed(tmp_path):
    model, rows, _ = tiny_teacher(tmp_path)
    for row in rows:
        output = model.model(input_ids=torch.tensor([row["input_ids"]]), use_cache=False,
                             output_hidden_states=True)
        captured = study.native_boundaries(model, row, 1, lambda: None)
        assert set(captured) == {1, 2}
        for layer in captured:
            assert captured[layer].dtype == torch.bfloat16
            assert torch.equal(captured[layer], output.hidden_states[layer])
        assert not any(block._forward_pre_hooks for block in model.model.layers)
    def denied():
        raise RuntimeError("no reservation")
    with pytest.raises(RuntimeError, match="reservation"):
        study.native_boundaries(model, rows[0], 1, denied)
    assert not any(block._forward_pre_hooks for block in model.model.layers)


def test_actual_content_vjp_capture_keeps_frozen_parent_and_full_prefix_gradient(tmp_path):
    model, source_rows, boundary = tiny_teacher(tmp_path)
    rows = study.content_items(source_rows)
    with study.CachedPrefixTeacher(rows, boundary, 1, model.model.layers[1], model.model.layers[2],
                                   model.model.norm, model.lm_head.weight, model.model.rotary_emb) as teacher:
        state = {k: v.clone() for k, v in teacher.state_dict().items() if isinstance(v, torch.Tensor)}
        for index, case in enumerate(rows):
            expected = model.model(input_ids=torch.tensor([case["input_ids"]]), use_cache=False).last_hidden_state
            assert torch.equal(teacher.forward_case(index), expected)
        inputs = {}
        def watch(_module, arguments):
            value = arguments[0].detach().requires_grad_()
            inputs["hidden"] = value
            return (value, *arguments[1:])
        hook = teacher.model.layers[1].register_forward_pre_hook(watch)
        actual = teacher.forward_case(0)
        gradient = torch.autograd.grad(actual[0, -1].float().square().sum(), inputs["hidden"])[0]
        hook.remove()
        assert torch.isfinite(gradient).all() and gradient[0, 0].abs().sum() > 0
        capture = tmp_path / "tiny_capture"
        with study.GroupedTeacherCapture(capture, teacher, [1], rows, [0, 1], teacher_fingerprint="tiny",
                probe_groups=2, probes_per_group=8, activation_positions=16, position_capacity=416,
                seed=study.calibration.calibration_seed(1)) as recorder:
            for index in range(len(rows)):
                recorder.capture_case(index, lambda index=index: teacher.forward_case(index))
            metadata = recorder.finish()
        logits = torch.from_numpy(np.load(capture / "accepted_teacher_logits.npy").copy()).view(torch.bfloat16)
        expected = torch.nn.functional.linear(teacher.forward_case(0)[0, rows[0]["content_loss_positions"]],
                                              model.lm_head.weight)
        assert torch.equal(logits[0], expected)
        assert metadata["calibration_indices"] == [0, 1] and metadata["development_indices"] == [2]
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
        assert all(torch.equal(teacher.state_dict()[k], v) for k, v in state.items())


@pytest.mark.parametrize("phase", ["boundary", "capture", "fit"])
def test_denied_lease_precedes_validation_and_cuda(tmp_path, monkeypatch, phase):
    monkeypatch.setattr(study, "validate_plan_items", lambda _: pytest.fail("validated after denied lease"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("initialized CUDA"))
    def denied():
        raise RuntimeError("reservation denied")
    with pytest.raises(RuntimeError, match="reservation denied"):
        getattr(study, "run_" + phase)(tmp_path / phase, {}, denied)


def fit_fixture(tmp_path, monkeypatch):
    items = study.content_items(tiny_rows())
    root = tmp_path / "capture"
    root.mkdir()
    plan = dict(capture_root=str(root), fit_root=str(tmp_path / "fit"),
                parent_checkpoint_sha256="b" * 64, calibration_indices=[0, 1],
                model={"path": "model"})
    metadata = dict(layers=[21], calibration_indices=[0, 1], capture_order=list(range(len(items))),
                    probe_groups=2, probes_per_group=8, activation_positions=16, seed=820861,
                    teacher_fingerprint=study.teacher_fingerprint(plan, items))
    study.base.write(root / "metadata.json", metadata)
    study._save_receipt(root, plan, "capture", items,
                        artifacts={"metadata.json": study.base.sha(root / "metadata.json")})
    monkeypatch.setattr(study, "validate_plan_items", lambda _: items)
    monkeypatch.setattr(study, "DEVICE", "cpu")
    monkeypatch.setattr(study.trial, "_model_device", lambda _: nullcontext())
    monkeypatch.setattr(study.cache.sanity.execution, "arithmetic_policy", lambda: nullcontext())
    return plan, items


def test_fit_preserves_basis_then_nonpolynomial_interface_then_train_relation(tmp_path, monkeypatch):
    plan, _ = fit_fixture(tmp_path, monkeypatch)
    calls = []
    def basis(capture, out, **kwargs):
        assert kwargs["layer_ids"] == [21] and kwargs["rank"] == 1024 and kwargs["head_dim"] == 64
        calls.append("basis")
        kwargs["progress"]({"phase": "basis_eigendecomposition", "layer": 21})
        (out / "block_21").mkdir(parents=True)
        (out / "block_21" / "fit.pt").write_bytes(b"new basis")
    def interface(directory, capture, cut, native, callback, **kwargs):
        assert calls == ["basis"] and cut == 21 and (directory / "fit.pt").exists()
        calls.append("compiled")
        (directory / "interface.json").write_text("{}")
    def relation(directory, **kwargs):
        assert calls == ["basis", "compiled"] and (directory / "interface.json").exists()
        calls.append("relation")
        expected = dict(seed=861, anchors=24, remove_count=1024, pad_to_multiple=128,
                        svd_rank=64, oversample=16, power_iterations=1, ridge=.01,
                        eligibility_nrmse=.5, chunk_rows=256)
        assert {k: kwargs[k] for k in expected} == expected
        (directory / "relation.pt").write_bytes(b"TRAIN-only relation")
    monkeypatch.setattr(study.fitting, "finalize_basis", basis)
    monkeypatch.setattr(study.fitting, "capture_compiled_development", interface)
    monkeypatch.setattr(study.rankfit, "fit_relation", relation)
    monkeypatch.setattr(study.builder.native, "load_native_block", lambda *_args, **_kw: object())
    monkeypatch.setattr(study.repair, "restore_rotary", lambda *_args, **_kw: object())
    result = study.run_fit(plan["fit_root"], plan, lambda: None)
    assert calls == ["basis", "compiled", "relation"]
    assert result["order"] == ["basis", "compiled_nonGB_interface", "TRAIN_relation"]
    assert result["confirmed"] is False
    assert result["artifacts"]["block_21/relation.pt"] == study.base.sha(
        Path(plan["fit_root"]) / "block_21" / "relation.pt")


def test_changed_capture_receipt_blocks_fit_before_model_loading(tmp_path, monkeypatch):
    plan, _ = fit_fixture(tmp_path, monkeypatch)
    (Path(plan["capture_root"]) / "metadata.json").write_text("{}")
    monkeypatch.setattr(study.builder.native, "load_native_block",
                        lambda *_a, **_kw: pytest.fail("native model loaded before capture authentication"))
    monkeypatch.setattr(study.fitting, "finalize_basis",
                        lambda *_a, **_kw: pytest.fail("fit started before capture authentication"))
    with pytest.raises(ValueError, match="artifact changed"):
        study.run_fit(plan["fit_root"], plan, lambda: None)
    assert not Path(plan["fit_root"]).exists()


def test_split_rotary_preserves_two_actual_paths_and_input_gradient(tmp_path):
    model, source_rows, boundary = tiny_teacher(tmp_path)
    rows = study.content_items(source_rows)
    native = model.model.layers[1]
    with torch.no_grad():
        native.self_attn.q_proj.weight.mul_(16)
        native.self_attn.k_proj.weight.mul_(16)
    native_rope = deepcopy(model.model.rotary_emb).eval()
    parent_rope = deepcopy(model.model.rotary_emb).eval()
    parent_rope.inv_freq.mul_(13)
    parent = model.model.layers[2]
    with study.SplitRotaryTeacher(rows, boundary, 1, native, parent, model.model.norm,
            model.lm_head.weight, parent_rope, native_rotary=native_rope) as teacher:
        x = teacher.boundary(0).detach().requires_grad_()
        entrance = study.calibration._hidden(native(x, **study.calibration.block_arguments(x, native_rope)))
        reference = model.model.norm(study.calibration._hidden(
            parent(entrance, **study.calibration.block_arguments(entrance, parent_rope))))
        expected_gradient = torch.autograd.grad(reference[0, -1].float().square().sum(), x)[0]
        captured = {}
        def watch(_module, arguments):
            value = arguments[0].detach().requires_grad_()
            captured["input"] = value
            return (value, *arguments[1:])
        handle = native.register_forward_pre_hook(watch)
        try:
            actual = teacher.forward_case(0)
            actual_gradient = torch.autograd.grad(actual[0, -1].float().square().sum(), captured["input"])[0]
        finally:
            handle.remove()
        assert torch.equal(actual, reference)
        assert torch.equal(actual_gradient, expected_gradient)
        assert actual_gradient[0, 0].abs().sum() > 0
        wrong_entrance = study.calibration._hidden(native(x.detach(), **study.calibration.block_arguments(x, parent_rope)))
        assert not torch.equal(wrong_entrance, entrance), "The fixture must distinguish the two rotary paths"
        assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())


def test_native_seam_requires_every_valid_cached_value_to_match(tmp_path):
    model, source_rows, boundary = tiny_teacher(tmp_path)
    with study.SplitRotaryTeacher(study.content_items(source_rows), boundary, 1, model.model.layers[1],
            model.model.layers[2], model.model.norm, model.lm_head.weight, model.model.rotary_emb,
            native_rotary=deepcopy(model.model.rotary_emb)) as teacher:
        values = torch.cat([teacher.native_forward(i)[0] for i in range(len(source_rows))]).detach()
        path = tmp_path / "next_boundary.npy"
        np.save(path, values.view(torch.int16).numpy())
        assert study.check_native_seam(teacher, path) == dict(passed=True, bit_exact=True, count=3, max_abs=0.)
        values[-1, 0] += 1
        np.save(path, values.view(torch.int16).numpy())
        with pytest.raises(ValueError):
            study.check_native_seam(teacher, path)


def test_native_rotary_guard_precedes_configuration_or_device_initialization(monkeypatch):
    monkeypatch.setattr(study.builder.native, "_config", lambda *_a: pytest.fail("config loaded"))
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("CUDA initialized"))
    def denied():
        raise RuntimeError("no lease")
    with pytest.raises(RuntimeError, match="no lease"):
        study.native_rotary({"model": {"path": "unused"}}, "cuda:0", denied)


def test_recovery_dataset_keeps_native_content_teacher_with_distinct_student_and_parent_inputs(tmp_path, monkeypatch):
    rows = study.content_items(tiny_rows())
    native = tmp_path / "cache" / "native"
    entrance = tmp_path / "boundary"
    native.mkdir(parents=True)
    entrance.mkdir()
    hidden = torch.arange(rows[-1]["stop"] * 8).reshape(-1, 8).bfloat16()
    logits = (torch.arange(len(rows) * 8 * 16).reshape(len(rows), 8, 16) % 31).bfloat16()
    np.save(entrance / "boundary_21.npy", hidden.view(torch.int16).numpy())
    np.save(native / "boundary_22.npy", (hidden + 100).view(torch.int16).numpy())
    np.save(native / "teacher_logits.npy", logits.view(torch.int16).numpy())
    plan = dict(cache_root=str(native.parent), boundary_root=str(entrance), label_ids=[6, 7, 8, 9])
    monkeypatch.setattr(study, "validate_plan_items", lambda _: rows)
    for parent_input in (False, True):
        dataset = study.recovery_dataset(plan, parent_input=parent_input)
        try:
            batch = dataset.batch([2, 0], device="cpu")
            assert torch.equal(batch["teacher"], logits[[2, 0], 4:8])
            for b, index in enumerate((2, 0)):
                case = rows[index]
                expected = hidden[case["offset"]:case["stop"]] + (100 if parent_input else 0)
                assert torch.equal(batch["hidden"][b, :len(expected)], expected)
            assert batch["loss_positions"].tolist() == [rows[i]["content_loss_positions"] for i in (2, 0)]
            assert batch["targets"].tolist() == [rows[i]["content_targets"] for i in (2, 0)]
        finally:
            dataset.close()
