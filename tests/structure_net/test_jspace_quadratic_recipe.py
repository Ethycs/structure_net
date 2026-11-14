"""Controller integration: real checkpoints/bridges, immutable splits and leases."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_quadratic_recipe as recipe
from experiments.structure_net.jspace_quadratic_train import CoordinateDataset


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch.manual_seed(52)
    root = tmp_path / "run"; root.mkdir()
    path = tmp_path / "native"
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_attention_heads=2,
        num_key_value_heads=1, num_hidden_layers=36, vocab_size=16, attention_dropout=0.,
        tie_word_embeddings=True, max_position_embeddings=32)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).bfloat16().eval()
    model.save_pretrained(path)
    write(root / "plan.json", {"model": {"path": str(path)}, "label_ids": [0, 1]})
    for layer in (0, 1, 35):
        basis = torch.linalg.qr(torch.randn(8, 8)).Q[:, :4]
        fit = {"basis": basis, "means": torch.randn(1, 8) / 10,
               "rope_pairs": torch.tensor([[0]]), "value_basis": torch.eye(4)[None, :, :2]}
        target = recipe.fit_path(root, layer); target.parent.mkdir(parents=True)
        torch.save(fit, target)
    return root


@pytest.mark.parametrize("arm", recipe.ARMS)
def test_real_native_loading_selected_checkpoint_readout_and_metric_pack(tiny_checkpoint, arm):
    root = tiny_checkpoint
    block = recipe.build_block(root, arm, 35)
    assert block.mode == arm and block.intermediate_size == 12 and block.head_dim == 2
    assert all(p.dtype == torch.float32 for p in block.parameters())
    assert block.scale == .5  # Native sqrt(4), despite a retained head width of2.
    head = recipe.build_readout(root, block)
    with torch.no_grad():
        block.metric.log_metric.add_(.2)
        head.bias.add_(.05)
    path = recipe.train_path(root, arm, 35); path.mkdir(parents=True)
    torch.save({"block": block.state_dict(), "readout": head.state_dict()}, path / "selected.pt")
    restored = recipe.build_block(root, arm, 35, selected=True).eval()
    restored_head = recipe.build_readout(root, restored, selected_arm=arm).eval()
    packed = recipe.build_block(root, arm, 35, selected=True, packed=True)
    rotary = recipe.rotary_module(recipe.read(root / "plan.json"), "cpu")
    coordinates = torch.randn(2, 5, 4)
    positions = torch.arange(5)[None].expand(2, -1)
    kwargs = dict(position_ids=positions, position_embeddings=rotary(coordinates, positions))
    with torch.no_grad():
        reference = block.eval().forward_coordinates(coordinates, **kwargs)
        exact = restored.forward_coordinates(coordinates, **kwargs)
        folded = packed.forward_coordinates(coordinates, **kwargs)
    torch.testing.assert_close(exact, reference, atol=0, rtol=0)
    torch.testing.assert_close(folded, reference, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(restored_head(folded), head(reference), atol=5e-6, rtol=5e-5)
    assert packed.metric is None and packed.packed and not any(p.requires_grad for p in packed.parameters())


def test_arm_metadata_cannot_relabel_a_selected_checkpoint(tiny_checkpoint):
    root = tiny_checkpoint
    block = recipe.build_block(root, "quadratic", 0)
    path = recipe.train_path(root, "softmax", 0); path.mkdir(parents=True)
    torch.save({"block": block.state_dict(), "readout": None}, path / "selected.pt")
    with pytest.raises(ValueError, match="same attention mode"):
        recipe.build_block(root, "softmax", 0, selected=True)


def test_fixed_bridge_matches_native_reconstruction_without_changing_arms(tiny_checkpoint):
    root = tiny_checkpoint
    first, second = recipe.load(recipe.fit_path(root, 0)), recipe.load(recipe.fit_path(root, 1))
    u, v = first["basis"], second["basis"]
    a = first["means"] - first["means"] @ u @ u.T
    b = second["means"] - second["means"] @ v @ v.T
    coordinates = torch.randn(2, 7, 4)
    weight, shift = recipe.bridge(root, 0)
    actual = torch.nn.functional.linear(coordinates, weight, shift[0])
    expected = (coordinates @ u.T + a - b) @ v
    torch.testing.assert_close(actual, expected, atol=4e-7, rtol=3e-6)
    assert recipe.prefix_path(root, "softmax", 0) == recipe.prefix_path(root, "quadratic", 0)
    assert recipe.prefix_path(root, "softmax", 1) != recipe.prefix_path(root, "quadratic", 1)


def test_actual_advance_packs_and_streams_selected_student_prefix(tiny_checkpoint):
    root = tiny_checkpoint
    block = recipe.build_block(root, "quadratic", 0).eval()
    with torch.no_grad():
        block.metric.log_metric.add_(.15)
    rotary = recipe.rotary_module(recipe.read(root / "plan.json"), "cpu")
    lengths = [5, 7, 4, 6]
    items = []; offset = 0
    for i, length in enumerate(lengths):
        items.append(dict(identity=str(i), family="toy", split="train" if i < 2 else "development",
            input_ids=list(range(length)), offset=offset, stop=offset+length, loss_positions=[0, 1, 2, length-1]))
        offset += length
    coordinates = torch.randn(offset, 4)
    teacher = torch.randn(offset, 8).bfloat16()
    np.save(root / "coordinates.npy", coordinates.numpy())
    np.save(root / "teacher.npy", teacher.view(torch.int16).numpy())
    data = CoordinateDataset(items, root / "coordinates.npy", root / "teacher.npy", basis=block.basis, anchor=block.anchor)
    following = recipe.load(recipe.fit_path(root, 1)); v = following["basis"]
    b = following["means"] - following["means"] @ v @ v.T
    expected = []
    with torch.no_grad():
        for i, case in enumerate(items):
            z = coordinates[case["offset"]:case["stop"]][None]
            positions = torch.arange(lengths[i])[None]
            output = block.forward_coordinates(z, position_ids=positions, position_embeddings=rotary(z, positions))
            expected.append(((output @ block.basis.T + block.anchor - b) @ v)[0])
    directory = recipe.train_path(root, "quadratic", 0); directory.mkdir(parents=True)
    recipe.advance(root, "quadratic", 0, block, data, rotary, lambda: None)
    actual = torch.from_numpy(np.load(directory / "next_coordinates.npy"))
    torch.testing.assert_close(actual, torch.cat(expected), atol=4e-6, rtol=4e-5)
    assert recipe.read(directory / "fold_parity.json")["passed"]
    assert not any(p.requires_grad for p in block.parameters())
    torch.testing.assert_close(torch.from_numpy(np.load(root / "coordinates.npy")), coordinates, atol=0, rtol=0)
    data.close()


def test_gpu_children_use_reservation_cpu_children_hide_cuda_and_logs_prevent_retry(tmp_path, monkeypatch):
    prior = tmp_path / "prior"
    write(prior / "completed.json", {"status": "completed"})
    write(tmp_path / "plan.json", {"prior_completion": str(prior)})
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write("mock child completed\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(recipe.subprocess, "run", run)
    recipe.run_phase(tmp_path, "train", arm="quadratic", layer=3)
    command, options = calls[-1]
    assert command[:4] == ["/usr/bin/python3", "/usr/local/bin/inference-ctl", "lease", "run"]
    assert command[command.index("--allocation") + 1] == recipe.CONFIG["gpu_allocation"]
    assert command[command.index("-m") + 1] == recipe.MODULE
    assert command[-4:] == ["--arm", "quadratic", "--layer", "3"]
    assert "--locked" in command and "--offline" in command
    recipe.run_phase(tmp_path, "export", arm="quadratic", gpu=False)
    command, options = calls[-1]
    assert "inference-ctl" not in " ".join(command)
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "" and options["env"]["OMP_NUM_THREADS"] == "1"
    with pytest.raises(FileExistsError):
        recipe.run_phase(tmp_path, "train", arm="quadratic", layer=3)


@pytest.fixture
def preparation_inputs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"; repo.mkdir()
    teacher, source = repo / "teacher", repo / "task_source"
    teacher.mkdir(); source.mkdir()
    monkeypatch.setattr(recipe, "ROOT", repo)
    monkeypatch.setattr(recipe, "TEACHER", teacher)
    monkeypatch.setattr(recipe, "TASK_SOURCE", source)
    producer = repo / "producer.py"; producer.write_text("# fixed recipe producer\n")
    monkeypatch.setattr(recipe, "sources", lambda: [producer])
    monkeypatch.setattr(recipe.shutil, "disk_usage", lambda _: SimpleNamespace(free=100*1024**3))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    tasks = {"splits": {}, "previous_identities": [], "fresh_exclusions": {}}
    families = ("arc", "boolq", "hellaswag", "winogrande")
    for split, count in (("train", 128), ("development", 16), ("evaluation", 128)):
        tasks["splits"][split] = [dict(identity=f"{split}-{family}-{i}", family=family, input_ids=[1, 2],
            choices=["a", "b"], answer=0) for family in families for i in range(count)]
    rows = []; offset = 0
    for split in ("train", "development"):
        for case in tasks["splits"][split]:
            rows.append({**case, "split": split, "offset": offset, "stop": offset+2, "loss_positions": [0, 0, 0, 1]})
            offset += 2
    items = {"items": rows, "tokens": offset}
    cache = teacher / "capture"; cache.mkdir()
    write(cache / "items.json", items)
    torch.save({"mock": torch.ones(1)}, cache / "statistics.pt")
    np.save(cache / "teacher_logits.npy", np.zeros((576, 4, 4), dtype=np.int16))
    for layer in range(37):
        np.save(cache / f"boundary_{layer:02d}.npy", np.zeros((offset, 8), dtype=np.int16))
    model = repo / "model"; model.mkdir()
    (model / "weights").write_bytes(b"frozen native checkpoint")
    original = {"model": {"path": str(model), "files": {"weights": recipe.sha(model / "weights")}}, "label_ids": [0, 1],
        "source_sha256": {str(producer): recipe.sha(producer)}, "input_sha256": {}, "task_digest": recipe.digest(tasks)}
    original["scientific_fingerprint"] = recipe.digest(original)
    write(teacher / "plan.json", original)
    archive = teacher / "producer/producer.py"; archive.parent.mkdir(); archive.write_bytes(producer.read_bytes())
    write(teacher / "tasks.json", tasks)
    manifest = {"status": "completed", "systems_only": False, "fingerprint": original["scientific_fingerprint"],
        "artifact_sha256": {str(p.relative_to(teacher)): recipe.sha(p) for p in cache.iterdir()}}
    write(teacher / "capture_manifest.json", manifest)
    prior = {"schema": "jspace.qwen-parallel.v2", "task_digest": recipe.digest(tasks)}
    prior["fingerprint"] = recipe.digest(prior)
    write(source / "plan.json", prior); write(source / "tasks.json", tasks)
    return repo / "new", teacher, source, producer


def test_prepare_freezes_original_task_teacher_and_own_sources(preparation_inputs):
    root, teacher, source, producer = preparation_inputs
    result = recipe.prepare(root)
    assert result["status"] == "prepared"
    plan = recipe.verify(root)
    assert plan["task_digest"] == recipe.digest(recipe.read(source / "tasks.json"))
    assert plan["teacher_fingerprint"] == recipe.read(teacher / "plan.json")["scientific_fingerprint"]
    assert plan["config"]["basis_operator"].startswith("supervised choice-margin")
    assert recipe.read(root / "items.json") == recipe.read(teacher / "capture/items.json")
    assert (root / "producer/producer.py").read_bytes() == producer.read_bytes()
    producer.write_text("# changed after freeze\n")
    with pytest.raises(ValueError, match="producer"):
        recipe.verify(root)


@pytest.mark.parametrize("fault", ["duplicate_evaluation", "unbalanced_evaluation", "changed_training_content", "task_digest", "teacher_fingerprint"])
def test_prepare_rejects_inconsistent_or_leaking_task_evidence(preparation_inputs, fault):
    root, teacher, source, _ = preparation_inputs
    tasks = recipe.read(source / "tasks.json")
    if fault == "duplicate_evaluation":
        tasks["splits"]["evaluation"][-1] = deepcopy(tasks["splits"]["evaluation"][0])
    elif fault == "unbalanced_evaluation":
        tasks["splits"]["evaluation"][0]["family"] = "boolq"
    elif fault == "changed_training_content":
        tasks["splits"]["train"][0]["input_ids"] = [9, 9]
    elif fault == "task_digest":
        prior = recipe.read(source / "plan.json")
        prior["task_digest"] = "changed"
        prior.pop("fingerprint"); prior["fingerprint"] = recipe.digest(prior)
        write(source / "plan.json", prior)
    else:
        parent = recipe.read(teacher / "plan.json"); parent["label_ids"] = [2, 3]
        write(teacher / "plan.json", parent)
    if fault in {"duplicate_evaluation", "unbalanced_evaluation", "changed_training_content"}:
        write(source / "tasks.json", tasks)
        prior = recipe.read(source / "plan.json"); prior["task_digest"] = recipe.digest(tasks)
        prior.pop("fingerprint"); prior["fingerprint"] = recipe.digest(prior)
        write(source / "plan.json", prior)
    with pytest.raises(ValueError):
        recipe.prepare(root)
    assert not root.exists()
