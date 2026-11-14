from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_single_block_verify as verification


def campaign(tmp_path, monkeypatch):
    root = tmp_path / "campaign"; directory = root / "train"; directory.mkdir(parents=True)
    recipe = verification.recipe
    cases = [dict(identity=str(i), family="tiny", answer=i % 2, choices=["a", "b"], split="development") for i in range(64)]
    records = [dict(identity=c["identity"], family=c["family"], answer=c["answer"], prediction=c["answer"],
                    logits=[1., 0.] if c["answer"] == 0 else [0., 1.]) for c in cases]
    metrics = dict(correct=64, relative_mse=.2, mse=.5, loss=.2)
    reference = dict(metrics=metrics, records=records)
    normalization = dict(floor=.001)
    checkpoint = directory / "epoch_00.pt"
    recipe.save(checkpoint, dict(epoch=0, normalization=normalization, block={"weight": torch.ones(2, 2)}))
    refpath = directory / "epoch_00_development.json"; recipe.write(refpath, reference)
    row = dict(epoch=0, metrics=metrics, checkpoint=checkpoint.name, checkpoint_sha256=recipe.sha(checkpoint),
               development_sha256=recipe.sha(refpath))
    result = dict(status="completed", systems_only=False, strict_reload_exact=True, selected_epoch=0,
        selected_checkpoint=str(checkpoint.resolve()), history=[row], selected=metrics, normalization=normalization,
        quality_pass=False)
    recipe.write(directory / "result.json", result)
    manifest = dict(status="completed", systems_only=False, fingerprint="parent",
                    result_sha256=recipe.sha(directory / "result.json"))
    recipe.write(directory / "manifest.json", manifest); recipe.write(root / "items.json", dict(items=cases))
    plan = dict(fingerprint="parent", config=dict(epochs=0), input_sha256={})
    for key in ("fit_path", "input_path", "target_path"):
        path = tmp_path / key; path.write_bytes(key.encode()); plan[key] = str(path)
        plan["input_sha256"][str(path)] = recipe.sha(path)
    recipe.write(root / "plan.json", plan)
    monkeypatch.setattr(verification.single, "verify", lambda _root: plan)
    return root, plan, result, reference, cases


def test_completed_gate_accepts_scientific_failure_and_authenticates_selection(tmp_path, monkeypatch):
    root, _, _, _, _ = campaign(tmp_path, monkeypatch)
    checked = verification.completed_training(root)
    assert checked["result"]["quality_pass"] is False
    assert checked["checkpoint"].name == "epoch_00.pt"
    block = torch.nn.Linear(2, 2, bias=False)
    verification.load_selected(block, checked)
    assert torch.equal(block.weight, torch.ones(2, 2)) and not block.weight.requires_grad


def runtime_policy(root):
    from experiments.structure_net import jspace_single_block_reserved as reserved
    recipe = verification.recipe
    sources = [Path(reserved.__file__), recipe.ROOT / "tests/structure_net/test_jspace_single_block_reserved.py"]
    policy = dict(schema=reserved.SCHEMA, base_schema=verification.single.SCHEMA,
        base_plan_fingerprint="parent", base_plan_sha256=recipe.sha(root / "plan.json"),
        headroom_mib=reserved.HEADROOM_MIB, allocator_backend="native", device=0, requires_fresh_cuda_process=True,
        source_sha256={str(path):recipe.sha(path) for path in sources})
    policy["fingerprint"] = recipe.digest(policy)
    for path in sources:
        target = root / "runtime_producer" / path.relative_to(recipe.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    recipe.write(root / "runtime_policy.json", policy)
    return policy


@pytest.mark.parametrize("status", ["missing", "failed", "completed"])
def test_bounded_runtime_receipt_is_required_and_frozen(tmp_path, monkeypatch, status):
    root, _, _, _, _ = campaign(tmp_path, monkeypatch)
    policy = runtime_policy(root)
    if status != "missing":
        verification.recipe.write(root / "runtime_train.json", dict(status=status, stage="train",
            base_plan_fingerprint="parent", runtime_policy_fingerprint=policy["fingerprint"],
            base_stage_manifest_sha256=verification.recipe.sha(root / "train/manifest.json")))
    if status != "completed":
        monkeypatch.setattr(verification.recipe, "load", lambda *_:pytest.fail("Checkpoint loaded before runtime receipt gate"))
        with pytest.raises((FileNotFoundError, ValueError)):
            verification.completed_training(root)
    else:
        checked = verification.completed_training(root)
        assert checked["runtime_policy_fingerprint"] == policy["fingerprint"]
        for name in ("runtime_policy.json", "runtime_train.json"):
            assert checked["input_sha256"][str(root / name)] == verification.recipe.sha(root / name)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        receipt = tmp_path / "receipt"; verification.prepare(root, receipt)
        assert verification.verify(receipt)[0]["runtime_policy_fingerprint"] == policy["fingerprint"]
        # A removed policy cannot silently downgrade to the uncapped protocol.
        (root / "runtime_policy.json").unlink()
        with pytest.raises(ValueError, match="allocator policy"):
            verification.completed_training(root)


@pytest.mark.parametrize("changed", ["manifest", "checkpoint", "reference", "normalization"])
def test_changed_input_rejected_before_untrusted_checkpoint_loading(tmp_path, monkeypatch, changed):
    root, _, _, _, _ = campaign(tmp_path, monkeypatch)
    if changed == "normalization":
        checked = verification.completed_training(root)
        checked["result"]["normalization"] = dict(floor=9.)
        with pytest.raises(ValueError, match="normalization"):
            verification.load_selected(torch.nn.Linear(2, 2, bias=False), checked)
        return
    path = root / "train" / {"manifest": "manifest.json", "checkpoint": "epoch_00.pt",
                            "reference": "epoch_00_development.json"}[changed]
    if changed == "manifest":
        value = json.loads(path.read_text()); value["status"] = "running"; path.write_text(json.dumps(value))
    else:
        with path.open("ab") as stream: stream.write(b" ")
    monkeypatch.setattr(verification.recipe, "load", lambda *_: pytest.fail("Loaded before hash/completion gate"))
    with pytest.raises(ValueError): verification.completed_training(root)


def test_reference_identity_and_argmax_are_checked(tmp_path, monkeypatch):
    _, _, _, reference, cases = campaign(tmp_path, monkeypatch)
    changed = deepcopy(reference["records"]); changed.reverse()
    with pytest.raises(ValueError, match="identities"):
        verification.validate_records(changed, cases)
    changed = deepcopy(reference["records"]); changed[0]["prediction"] = 1
    with pytest.raises(ValueError, match="argmax"):
        verification.validate_records(changed, cases)


def test_comparison_keeps_prediction_equality_separate_from_tolerance(tmp_path, monkeypatch):
    _, _, _, reference, cases = campaign(tmp_path, monkeypatch)
    features = dict(relative_mse=.2, mse=.5, loss=.2)
    assert verification.compare(reference, features, reference["records"], cases)["passed"]
    changed = deepcopy(reference["records"]); changed[0]["logits"][0] += .01
    assert not verification.compare(reference, features, changed, cases)["logits_pass"]
    before = deepcopy(reference); before["records"][0]["logits"] = [1., 1. - 1e-6]
    after = deepcopy(before["records"]); after[0]["logits"] = [1. - 1e-6, 1.]; after[0]["prediction"] = 1
    result = verification.compare(before, features, after, cases)
    assert result["logits_pass"] and not result["predictions_exact"] and not result["passed"]
    assert not verification.compare(reference, {**features, "mse": .6}, reference["records"], cases)["feature_pass"]


def test_fresh_actual_qwen_block_strict_reload_then_detached_pack(tmp_path):
    torch.manual_seed(27864)
    config = Qwen2Config(hidden_size=16, intermediate_size=24, num_attention_heads=4,
        num_key_value_heads=2, num_hidden_layers=2, vocab_size=23, max_position_embeddings=32)
    config._attn_implementation = "eager"
    native = Qwen2DecoderLayer(config, layer_idx=0).bfloat16().float().eval().requires_grad_(False)
    fit = dict(basis=torch.linalg.qr(torch.randn(16, 8)).Q, means=torch.randn(1, 16) / 20)
    original = verification.single.build_enriched_block(deepcopy(native), fit, parameter_dtype=torch.float32)
    with torch.no_grad():
        for parameter in original.parameters(): parameter.add_(torch.randn_like(parameter) * .001)
    path = tmp_path / "selected.pt"; normalization = dict(floor=.001)
    verification.recipe.save(path, dict(block=original.state_dict(), epoch=2, normalization=normalization))
    checked = dict(checkpoint=path, input_sha256={str(path): verification.recipe.sha(path)},
                   result=dict(selected_epoch=2, normalization=normalization))
    fresh = verification.single.build_enriched_block(deepcopy(native), fit, parameter_dtype=torch.float32)
    verification.load_selected(fresh, checked)
    for key, value in original.state_dict().items():
        restored = fresh.state_dict()[key]
        assert torch.equal(value, restored) if isinstance(value, torch.Tensor) else value == restored
    packed = deepcopy(fresh).pack(dtype=torch.float32, buffer_dtype=torch.float32)
    assert not any(p.requires_grad for p in packed.parameters())
    h = torch.randn(2, 5, 16); positions = torch.arange(5)[None].expand(2, -1)
    allowed = (positions[:, :, None] >= positions[:, None, :])[:, None]
    rotary = Qwen2RotaryEmbedding(config=config); kwargs = dict(attention_mask=allowed,
        position_ids=positions, position_embeddings=rotary(h, positions))
    with torch.no_grad():
        expected = original(h, **kwargs); observed = packed(h, **kwargs)
    torch.testing.assert_close(expected, fresh(h, **kwargs), atol=0, rtol=0)
    torch.testing.assert_close(expected, observed, atol=1e-5, rtol=2e-4)
    broken = dict(checked); broken["result"] = dict(selected_epoch=3, normalization=normalization)
    with pytest.raises(ValueError, match="epoch"):
        verification.load_selected(fresh, broken)


def test_prepare_is_cpu_only_exclusive_and_freezes_receipt(tmp_path, monkeypatch):
    campaign_root, _, _, _, _ = campaign(tmp_path, monkeypatch)
    root = tmp_path / "verification"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="Hide CUDA"):
        verification.prepare(campaign_root, root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = verification.prepare(campaign_root, root)
    plan, checked = verification.verify(root)
    assert result["fingerprint"] == plan["fingerprint"]
    assert checked["checkpoint"].name == "epoch_00.pt"
    with pytest.raises(FileExistsError): verification.prepare(campaign_root, root)
    with pytest.raises(ValueError): verification.Tolerances(logit_atol=float("nan"))


def test_cuda_reservation_is_required_before_initialization(tmp_path, monkeypatch):
    from experiments.structure_net import jspace_full_lease
    monkeypatch.setattr(verification, "verify", lambda _root: (dict(fingerprint="verified"), {}))
    def refused(_devices): raise RuntimeError("No active reservation")
    monkeypatch.setattr(jspace_full_lease, "wait_ready", refused)
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CUDA initialized without a lease"))
    with pytest.raises(RuntimeError, match="reservation"):
        verification.run(tmp_path, device="cuda:0")
    assert verification.recipe.read(tmp_path / "verification/failure.json")["status"] == "failed"
