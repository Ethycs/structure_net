from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from tokenizers.pre_tokenizers import Whitespace
import torch
from transformers import AutoModelForCausalLM

from experiments.structure_net import jspace_3b_infer as cli
from test_jspace_allblocks_artifact import fixture


@pytest.fixture
def bundle(tmp_path):
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        source, model, _ = fixture(tmp_path, polynomial=True)
        tokenizer = cli.artifact.AutoTokenizer.from_pretrained(source, local_files_only=True)
        tokenizer.backend_tokenizer.pre_tokenizer = Whitespace()
        tokenizer.add_tokens(["user", "assistant"])
        tokenizer.chat_template = ("{% for message in messages %}{{ message['role'] }} {{ message['content'] }} "
                                   "{% endfor %}{% if add_generation_prompt %}assistant{% endif %}")
        tokenizer.save_pretrained(source)
        output = tmp_path / "bundle"
        cli.artifact.export_artifact(output, model.model.layers[0], model.model.norm, source, "tiny-full36")
        yield output, source, model, tokenizer
    finally:
        torch.set_num_threads(previous)


def test_raw_text_cli_roundtrip_after_dense_source_removed(bundle, monkeypatch, capsys):
    output, source, reference, tokenizer = bundle
    inputs = cli.tokenize_prompt(tokenizer, "A B", max_new_tokens=2, capacity=8)
    assert inputs["input_ids"].tolist() == [[6, 4, 5, 7]]
    generation = cli.artifact.GenerationConfig.from_pretrained(output, local_files_only=True)
    assert generation.repetition_penalty == 1.05
    generation.do_sample = False; generation.num_beams = 1; generation.max_new_tokens = 2
    with torch.inference_mode():
        expected = reference.generate(**inputs, generation_config=generation)[0, 4:].tolist()
    original_config = reference.generation_config.to_dict()
    shutil.rmtree(source)
    original_open = io.open
    def denied_source(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file).is_relative_to(source):
            pytest.fail("Raw-text inference read deleted native source")
        return original_open(file, *args, **kwargs)
    monkeypatch.setattr(io, "open", denied_source)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: pytest.fail("Generic dense loader used"))
    monkeypatch.setattr(cli, "cuda_reservation", lambda: pytest.fail("CPU path touched CUDA reservation"))
    result = cli.main(["--bundle", str(output), "--prompt", "A B", "--max-new-tokens", "2", "--json"])
    assert json.loads(capsys.readouterr().out) == result
    assert result["generated_token_ids"] == expected
    assert result["text"] == tokenizer.decode(expected, skip_special_tokens=True)
    assert result["input_tokens"] == 4 and 0 < result["generated_tokens"] <= 2
    assert result["position_capacity"] == 8 and result["source_fingerprint"] == "tiny-full36"
    assert result["device"] == "cpu" and result["reservation"] is None and result["greedy"]
    assert reference.generation_config.to_dict() == original_config


def test_plain_completion_and_template_requirement(bundle):
    output, _, _, tokenizer = bundle
    result = cli.infer(output, "A B", plain=True, max_new_tokens=1)
    assert result["input_tokens"] == 2 and result["generated_tokens"] == 1
    tokenizer.chat_template = None
    with pytest.raises(ValueError, match="--plain"):
        cli.tokenize_prompt(tokenizer, "A B", capacity=8, max_new_tokens=1)
    assert cli.tokenize_prompt(tokenizer, "A B", plain=True, capacity=8, max_new_tokens=6)["input_ids"].shape[-1] == 2


def test_pinned_eos_stops_greedy_generation(bundle):
    output, _, _, _ = bundle
    initial = cli.infer(output, "A B", max_new_tokens=1)
    generation_path = output / "generation_config.json"
    generation = json.loads(generation_path.read_text())
    generation["eos_token_id"] = initial["generated_token_ids"][0]
    generation_path.write_text(json.dumps(generation))
    manifest_path = output / "manifest.json"; manifest = json.loads(manifest_path.read_text())
    manifest["files"]["generation_config.json"] = cli.artifact.sha(generation_path)
    manifest_path.write_text(json.dumps(manifest))
    result = cli.infer(output, "A B", max_new_tokens=3)
    assert result["generated_tokens"] == 1
    assert result["generated_token_ids"] == initial["generated_token_ids"]


def test_capacity_rejected_before_model_or_gpu_loading(bundle, monkeypatch):
    output, _, _, _ = bundle
    monkeypatch.setattr(cli.artifact, "load_artifact", lambda *a, **k: pytest.fail("Oversized request loaded weights"))
    monkeypatch.setattr(cli, "cuda_reservation", lambda: pytest.fail("Oversized request touched CUDA"))
    with pytest.raises(ValueError, match="no truncation"):
        cli.infer(output, "A B A B A B", max_new_tokens=1, device="cuda:0")
    for count in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            cli.infer(output, "A", max_new_tokens=count)


@pytest.mark.parametrize("field", ["source", "source_inventory", "version", "tokenizer"])
def test_runtime_or_text_mismatch_rejected_before_load(bundle, monkeypatch, field):
    output, _, _, _ = bundle
    path = output / "manifest.json"; manifest = json.loads(path.read_text())
    if field == "source": manifest["runtime_source_sha256"]["jspace_qwen_affine_fast.py"] = "0" * 64
    elif field == "source_inventory": manifest["runtime_source_sha256"].pop("jspace_qwen_affine_fast.py")
    elif field == "version": manifest["versions"]["torch"] = "different"
    else: (output / "tokenizer_config.json").write_text("{}")
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(cli.artifact, "load_artifact", lambda *a, **k: pytest.fail("Mismatched runtime loaded model"))
    monkeypatch.setattr(cli, "cuda_reservation", lambda: pytest.fail("Mismatched runtime touched GPU"))
    with pytest.raises(ValueError, match="mismatch"):
        cli.infer(output, "A", max_new_tokens=1, device="cuda:0")


class FakeCuda:
    def __init__(self):
        self.calls = []; self.peak = 0
        self.memory = SimpleNamespace(get_allocator_backend=lambda: "native")
    def is_initialized(self): self.calls.append("initialized"); return False
    def get_device_properties(self, device): self.calls.append("properties"); return SimpleNamespace(total_memory=12*1024**3)
    def set_per_process_memory_fraction(self, fraction, device): self.calls.append(("cap", fraction, device))
    def reset_peak_memory_stats(self, device): self.calls.append("reset")
    def max_memory_reserved(self, device): return self.peak


def test_reservation_required_before_any_cuda_query():
    cuda = FakeCuda()
    def denied(): raise RuntimeError("No active lease")
    with pytest.raises(RuntimeError, match="No active lease"):
        cli.cuda_reservation(reservation=denied, cuda=cuda, visible="GPU-test")
    assert cuda.calls == []


def test_allocator_cap_and_repeated_lease_checks_without_gpu():
    cuda = FakeCuda(); lease = dict(id="lease1", allocation={"GPU-test": 11264}); calls=[]
    def ready(): calls.append("lease"); return deepcopy(lease)
    guard, receipt = cli.cuda_reservation(reservation=ready, cuda=cuda, visible="GPU-test")
    assert calls == ["lease"]
    assert receipt["allocator_limit_bytes"] == 11008*cli.MIB
    assert ("cap", 11008*cli.MIB/(12*1024**3), 0) in cuda.calls
    guard(); assert calls == ["lease", "lease"]
    cuda.peak = receipt["allocator_limit_bytes"] + 1
    with pytest.raises(MemoryError, match="allocator limit"): guard()
    cuda.peak = 0; lease["id"] = "new"
    with pytest.raises(RuntimeError, match="reservation changed"): guard()


def test_unleased_visible_gpu_fails_before_cuda_query():
    cuda = FakeCuda()
    with pytest.raises(ValueError, match="visible GPU"):
        cli.cuda_reservation(reservation=lambda: dict(id="lease",allocation={"GPU-other":11264}),
                             cuda=cuda, visible="GPU-test")
    assert cuda.calls == []
