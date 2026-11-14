from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.structure_net import jspace_independent_layer_distill as worker


def fixture(tmp_path):
    path = Path(__file__).with_name("test_jspace_single_block_distill.py")
    spec = importlib.util.spec_from_file_location("independent_single_fixture", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.fixture(tmp_path)


def quality_fixture(tmp_path, monkeypatch):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from experiments.structure_net import jspace_qwen_parallel_model as loaders
    block, rotary, data, _, native, suffix = fixture(tmp_path)
    source, cache = tmp_path/"source", tmp_path/"cache"
    source.mkdir(); (cache/"cache").mkdir(parents=True)
    worker.recipe.write(source/"plan.json", dict(model=dict(path="tiny-native"), label_ids=[20, 21]))
    quality_items, rows, cursor = [], [], 0
    for i in data.development:
        case = data.items[i]
        # Deliberately distinguish true FP32 quality input from BF16 training.
        value = data.batch([i], "cpu")["hidden"][0]+.0012345
        rows.append(value)
        quality_items.append(dict(case, offset=cursor, stop=cursor+len(case["input_ids"])))
        cursor += len(case["input_ids"])
    worker.recipe.write(cache/"items.json", dict(items=quality_items))
    boundary = cache/"cache/boundary_34.npy"
    np.save(boundary, torch.cat(rows).numpy())
    torch.manual_seed(75)
    norm, embedding = Qwen2RMSNorm(16), torch.randn(23, 16).bfloat16().float()
    seen = []
    def load_native(path, layer, **_):
        assert path == "tiny-native"
        seen.append(layer)
        return deepcopy(native if layer == 34 else suffix)
    monkeypatch.setattr(loaders, "load_native_block", load_native)
    monkeypatch.setattr(loaders, "load_readout_norm", lambda *_a, **_k:(embedding, norm, None))
    monkeypatch.setattr(worker.recipe, "rotary_module", lambda *_a, **_k:rotary)
    plan = dict(source=str(source), layer=34, quality_prefix_root=str(cache), quality_boundary_path=str(boundary))
    quality = worker.make_quality(plan, data.items, tmp_path/"quality", lambda:None, device="cpu")
    return block, rotary, data, quality, native, suffix, norm, embedding, rows, seen, plan


def test_native_quality_uses_fp32_prefix_and_exact_absolute_layer_offsets(tmp_path, monkeypatch):
    block, rotary, data, quality, native, suffix, norm, embedding, rows, seen, plan = quality_fixture(tmp_path, monkeypatch)
    native_result = quality(None)
    assert seen == [34, 35]
    compact_result = quality(block)
    assert seen == [34, 35, 35]
    assert all(p.requires_grad for p in block.parameters())
    with torch.no_grad():
        for result, first in ((native_result, native), (compact_result, deepcopy(block).pack(dtype=torch.float32, buffer_dtype=torch.float32))):
            for record, i, x in zip(result["records"], data.development, rows):
                hidden = x[None]
                positions = torch.arange(len(x))[None]
                allowed = (positions[:, :, None] >= positions[:, None, :])[:, None]
                mask = torch.zeros_like(allowed, dtype=torch.float32).masked_fill(~allowed, torch.finfo(torch.float32).min)
                rope = rotary(hidden, positions)
                hidden = first(hidden, attention_mask=mask, position_ids=positions, position_embeddings=rope)
                hidden = hidden[0] if isinstance(hidden, tuple) else hidden
                hidden = suffix(hidden, attention_mask=mask, position_embeddings=rope)[0]
                logits = torch.nn.functional.linear(norm(hidden[:, -1]), embedding[[20, 21]])[0]
                torch.testing.assert_close(torch.tensor(record["logits"]), logits, atol=2e-6, rtol=2e-6)
                assert record["identity"] == data.items[i]["identity"]
                assert record["answer"] == data.items[i]["answer"]
    assert not torch.equal(torch.cat(rows), torch.cat(rows).bfloat16().float())
    assert all(p.grad is None and not p.requires_grad for p in native.parameters())
    data.close()


def test_independent_feature_training_freezes_other_blocks_and_native_pairs(tmp_path):
    block, rotary, data, quality, native, suffix = fixture(tmp_path)
    other = deepcopy(block)
    frozen = [deepcopy(m.state_dict()) for m in (native, suffix, other)]
    input_hash = worker.recipe.sha(tmp_path/"input.npy")
    trainer = worker.Trainer(block, rotary, data, quality,
        config=worker.single.Config(epochs=1, microbatch=1, accumulation=2))
    batch = data.batch([0, 1], "cpu")
    expected = worker.single.feature_losses(trainer.forward(batch), batch["hidden"], batch["target"], batch["valid"],
        floor=data.normalization["floor"], basis=block.basis)
    observed = trainer.losses(batch, diagnostics=True)
    for key in expected:
        torch.testing.assert_close(expected[key], observed[key], atol=0, rtol=0)
    observed["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    assert not batch["hidden"].requires_grad and not batch["target"].requires_grad
    trainer.optimizer.zero_grad(set_to_none=True)
    result = trainer.fit(tmp_path/"train", layer=34, native_correct=2, minimum_correct=3)
    assert result["strict_reload_exact"] and not result["quality_pass"] and result["updates"] == 1
    assert result["trained_layer"] == 34 and result["confirmed"] is False
    assert sorted(p.name for p in (tmp_path/"train").glob("*.pt")) == ["selected.pt"]
    assert result["selected_checkpoint_sha256"] == worker.recipe.sha(tmp_path/"train/selected.pt")
    for row in result["history"]:
        assert row["development_sha256"] == worker.recipe.sha(tmp_path/"train"/f"epoch_{row['epoch']:02d}_development.json")
    for module, state in zip((native, suffix, other), frozen):
        for key, tensor in module.state_dict().items():
            if isinstance(tensor, torch.Tensor):
                torch.testing.assert_close(tensor, state[key], atol=0, rtol=0)
            else:
                assert tensor == state[key]
        assert all(p.grad is None for p in module.parameters())
    assert input_hash == worker.recipe.sha(tmp_path/"input.npy")
    data.close()


def test_quality_table_rejects_quantized_or_reordered_prefix(tmp_path, monkeypatch):
    _, _, data, _, _, _, _, _, _, _, plan = quality_fixture(tmp_path, monkeypatch)
    path = Path(plan["quality_prefix_root"])/"items.json"
    items = worker.recipe.read(path)["items"]
    with pytest.raises(ValueError, match="identities, order"):
        worker.quality_table(data.items, list(reversed(items)), plan["quality_boundary_path"])
    raw = np.load(plan["quality_boundary_path"])
    np.save(plan["quality_boundary_path"], raw.astype(np.float16))
    with pytest.raises(ValueError, match="FP32"):
        worker.quality_table(data.items, items, plan["quality_boundary_path"])
    data.close()


def test_native_substitution_control_checks_predictions_and_identity():
    cases = [dict(identity="a", family="tiny", answer=0, choices=["a", "b"])]
    reference = dict(records=[dict(identity="a", family="tiny", answer=0, prediction=0, logits=[1., 0.])])
    equal = deepcopy(reference); equal["records"][0]["logits"][0] += 1e-6
    control = worker.native_control(equal, reference, cases, "reference-sha")
    assert control["passed"] and control["predictions_exact"] and control["logits_pass"] and control["logit_max_abs_delta"] > 0
    assert control["reference_sha256"] == "reference-sha"
    wrong_states = deepcopy(reference); wrong_states["records"][0]["logits"] = [10., 0.]
    invalid = worker.native_control(wrong_states, reference, cases, "reference-sha")
    assert invalid["predictions_exact"] and not invalid["logits_pass"] and not invalid["passed"]
    failed = deepcopy(reference); failed["records"][0].update(prediction=1, logits=[0., 1.])
    assert not worker.native_control(failed, reference, cases, "sha")["passed"]
    failed["records"][0]["identity"] = "other"
    with pytest.raises(ValueError, match="identities"):
        worker.native_control(failed, reference, cases, "sha")


def test_prepare_layer_and_cpu_guards_precede_source_access(tmp_path, monkeypatch):
    monkeypatch.setattr(worker.recipe, "verify", lambda *_a:pytest.fail("Source read before scope guard"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="Hide CUDA"):
        worker.prepare(tmp_path, tmp_path, tmp_path, tmp_path, 2)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for layer in (0, 1, 36, True):
        with pytest.raises(ValueError, match="layers2..35"):
            worker.prepare(tmp_path, tmp_path, tmp_path, tmp_path, layer)


def test_native_pilot_gate_blocks_training_before_reservation(tmp_path, monkeypatch):
    from experiments.structure_net import jspace_full_lease
    (tmp_path/"pilot").mkdir()
    worker.recipe.write(tmp_path/"pilot/result.json", dict(native_control=dict(passed=False)))
    plan = dict(fingerprint="frozen", quality_reference_sha256="native")
    monkeypatch.setattr(worker, "verify", lambda _:plan)
    monkeypatch.setattr(worker.joint, "verify_pilot", lambda *_:None)
    monkeypatch.setattr(jspace_full_lease, "wait_ready", lambda *_:pytest.fail("Reservation before native pilot gate"))
    with pytest.raises(ValueError, match="native substitution"):
        worker.run(tmp_path, "train")
    assert not (tmp_path/"train").exists()


def test_fresh_verification_allows_quality_miss_and_still_requires_completed_selection(tmp_path, monkeypatch):
    from experiments.structure_net import jspace_full_lease
    events = []
    monkeypatch.setattr(worker, "verify", lambda _:dict(fingerprint="frozen"))
    def complete(root, plan):
        events.append("authenticated-selection")
        return dict(quality_pass=False)
    monkeypatch.setattr(worker.sequential, "completed_training", complete)
    def denied(_):
        events.append("lease")
        raise RuntimeError("No reservation")
    monkeypatch.setattr(jspace_full_lease, "wait_ready", denied)
    monkeypatch.setattr(torch.cuda, "init", lambda:pytest.fail("CUDA without reservation"))
    with pytest.raises(RuntimeError, match="reservation"):
        worker.run(tmp_path, "verify")
    assert events == ["authenticated-selection", "lease"]
    receipt = worker.recipe.read(tmp_path/"verification/failure.json")
    assert receipt["lease_id"] is None and receipt["reservation_guard"]["calls"] == 0
    with pytest.raises(FileExistsError):
        worker.run(tmp_path, "verify")
