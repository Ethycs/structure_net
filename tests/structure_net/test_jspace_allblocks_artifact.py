from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sys

import pytest
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
import torch
from torch import nn
from transformers import DynamicCache, GenerationConfig, PreTrainedTokenizerFast, Qwen2Config, StaticCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2RMSNorm, Qwen2RotaryEmbedding

from experiments.structure_net import jspace_allblocks_artifact as artifact
from experiments.structure_net.jspace_qwen_affine_block import AffineQwenBlock
from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack


def fixture(tmp_path, *, polynomial=True, geometry="fp32"):
    torch.manual_seed(404)
    dtype = torch.float64 if geometry == "fp32" else torch.float32
    config = Qwen2Config(hidden_size=8, intermediate_size=12, num_attention_heads=2,
        num_key_value_heads=1, num_hidden_layers=36, vocab_size=10, max_position_embeddings=64,
        tie_word_embeddings=True, attention_dropout=0., pad_token_id=0, bos_token_id=1, eos_token_id=2)
    config._attn_implementation = "eager"
    source = tmp_path/"native"; source.mkdir()
    config.save_pretrained(source)
    GenerationConfig(eos_token_id=2, bos_token_id=1, pad_token_id=0, repetition_penalty=1.05).save_pretrained(source)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "A": 4, "B": 5}, unk_token="[UNK]")),
        pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.save_pretrained(source)
    embedding = torch.randn(10, 8, dtype=dtype)/4
    save_file({"model.embed_tokens.weight": embedding, "must_not_read_dense_tensor": torch.randn(7, 7)},
              source/"model.safetensors")
    native = Qwen2DecoderLayer(config, layer_idx=0).to(dtype).eval()
    basis = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64)).Q[:, :6].to(dtype)
    block = AffineQwenBlock(native, basis, torch.randn(8, 8, dtype=dtype)/100,
        head_projection={"rope_pairs": torch.tensor([[1]]), "value_basis": torch.eye(4, dtype=dtype)[None, :, :2]})
    if polynomial:
        fit = {"anchors": [0, 2], "mean": torch.zeros(12, dtype=dtype), "scale": torch.ones(12, dtype=dtype),
               "coefficients": {"2": torch.randn(12, 6, dtype=dtype)/1000}}
        block.compile_polynomial(fit, [1, 3, 4])
    with torch.no_grad():
        block.attention_output.mul_(.05); block.mlp_output.mul_(.05)
        if polynomial:
            block.polynomial_output.mul_(.05)
    blocks = [deepcopy(block) for _ in range(36)]
    for i, value in enumerate(blocks):
        value.layer_idx = i
    stack = CompactQwenStack(blocks, geometry_mode=geometry, pad_to_multiple=8).eval()
    pack_dtype = torch.bfloat16 if geometry == "bf16" else dtype
    stack.pack(dtype=pack_dtype, buffer_dtype=pack_dtype, strip_unused=True)
    norm = Qwen2RMSNorm(8, eps=config.rms_norm_eps).to(pack_dtype).eval()
    embedding = embedding.to(pack_dtype)
    if geometry == "bf16":
        save_file({"model.embed_tokens.weight": embedding, "must_not_read_dense_tensor": torch.randn(7, 7)},
                  source/"model.safetensors")
    model = Qwen2ForCausalLM(config).eval()
    model.model.layers = nn.ModuleList([stack])
    model.model.embed_tokens.weight = nn.Parameter(embedding, requires_grad=False)
    model.lm_head.weight = model.model.embed_tokens.weight
    model.model.norm = norm
    model.model.rotary_emb = Qwen2RotaryEmbedding(config)
    return source, model, config


@pytest.mark.parametrize("polynomial,geometry", [(False, "fp32"), (True, "fp32"), (True, "bf16")])
def test_standalone_logits_state_bytes_and_original_cache_indices(tmp_path, polynomial, geometry):
    source, reference, config = fixture(tmp_path, polynomial=polynomial, geometry=geometry)
    output = tmp_path/"bundle"
    report = artifact.export_artifact(output, reference.model.layers[0], reference.model.norm,
                                      source, "frozen-campaign")
    loaded, tokenizer = artifact.load_artifact(output)
    assert loaded.model.embed_tokens.weight is loaded.lm_head.weight
    assert loaded.config.num_hidden_layers == 36
    assert loaded.model.layers[0].layer_indices == tuple(range(36))
    assert loaded.compressed_position_capacity == 8 and loaded.config.max_position_embeddings == 64
    assert loaded.config.rope_theta == config.rope_theta
    assert loaded.generation_config.repetition_penalty == 1.05
    assert tokenizer.encode("A") == [4]
    assert artifact.resident_tensor_bytes(loaded) == artifact.resident_tensor_bytes(reference)
    assert report["resident_tensor_bytes"] == artifact.resident_tensor_bytes(loaded)
    assert report["total_file_bytes"] == sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    with safe_open(output/"weights.safetensors", framework="pt") as stream:
        actual_bytes = sum(stream.get_tensor(key).numel()*stream.get_tensor(key).element_size() for key in stream.keys())
        assert list(stream.keys()).count("embedding.weight") == 1
        assert not any("lm_head" in key for key in stream.keys())
    assert actual_bytes == report["tensor_payload_bytes"]
    inputs = torch.tensor([[4, 5, 4]])
    with torch.inference_mode():
        expected = reference(inputs, use_cache=False).logits
        actual = loaded(inputs, use_cache=False).logits
    assert torch.isfinite(actual).all() and torch.equal(actual, expected)
    dynamic, other = DynamicCache(), DynamicCache()
    dtype = loaded.lm_head.weight.dtype
    static = StaticCache(config, max_batch_size=1, max_cache_len=8, device="cpu", dtype=dtype)
    with torch.inference_mode():
        for position in range(inputs.shape[1]):
            ids = inputs[:, position:position+1]
            kwargs = dict(cache_position=torch.tensor([position]), use_cache=True)
            a = loaded(ids, past_key_values=dynamic, **kwargs).logits
            b = reference(ids, past_key_values=other, **kwargs).logits
            c = loaded(ids, past_key_values=static, **kwargs).logits
            assert torch.equal(a, b)
            torch.testing.assert_close(c, a, rtol=0, atol=0)
    assert len(dynamic.key_cache) == len(static.key_cache) == 36
    for i in range(36):
        assert dynamic.key_cache[i].shape == (1, 1, 3, 2)
        assert static.key_cache[i].shape == (1, 1, 8, 4)
        assert torch.count_nonzero(static.key_cache[i][..., 2:]) == 0
    original = reference.model.layers[0].state_dict()
    restored = loaded.model.layers[0].state_dict()
    assert original.keys() == restored.keys()
    for name, value in original.items():
        assert torch.equal(value, restored[name]) if isinstance(value, torch.Tensor) else value == restored[name]


def test_no_source_or_compiler_reads_after_export_and_capacity_is_enforced(tmp_path, monkeypatch):
    source, reference, _ = fixture(tmp_path)
    output = tmp_path/"standalone"
    artifact.export_artifact(output, reference.model.layers[0], reference.model.norm, source, "frozen")
    original_open = io.open
    def deny_native(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file).is_relative_to(source):
            pytest.fail("Standalone loader attempted to read native source")
        return original_open(file, *args, **kwargs)
    monkeypatch.setattr(io, "open", deny_native)
    original_safe_open = artifact.safe_open
    def deny_tensor_source(path, *args, **kwargs):
        assert not Path(path).is_relative_to(source)
        return original_safe_open(path, *args, **kwargs)
    monkeypatch.setattr(artifact, "safe_open", deny_tensor_source)
    monkeypatch.setattr(AffineQwenBlock, "__init__", lambda *a, **k: pytest.fail("Compiler invoked"))
    monkeypatch.setattr(CompactQwenStack, "__init__", lambda *a, **k: pytest.fail("Bridge compiler invoked"))
    monkeypatch.setattr(artifact, "_completed_campaign", lambda *a, **k: pytest.fail("Standalone load accessed campaign"))
    shutil.rmtree(source)
    loaded, _ = artifact.load_artifact(output)
    with pytest.raises(ValueError, match="calibrated"):
        loaded(torch.ones((1, 9), dtype=torch.long))
    with pytest.raises(ValueError, match="calibrated"):
        loaded(torch.ones((1, 1), dtype=torch.long), cache_position=torch.tensor([8]))
    with pytest.raises(ValueError, match="calibrated"):
        loaded.generate(torch.ones((1, 3), dtype=torch.long), max_new_tokens=6)
    with torch.inference_mode():
        assert loaded.generate(torch.tensor([[4, 5]]), max_new_tokens=2).shape[1] <= 4


@pytest.mark.parametrize("geometry", ["fp32", "bf16"])
def test_evaluator_export_reads_only_embedding_and_requires_schema_fingerprint(tmp_path, monkeypatch, geometry):
    source, reference, _ = fixture(tmp_path, geometry=geometry)
    if geometry == "bf16":
        # Match HF's deployment cast even when the source is stored at FP32.
        save_file({"model.embed_tokens.weight": reference.lm_head.weight.detach().float()}, source/"model.safetensors")
    checkpoint = tmp_path/"packed.pt"
    torch.save(dict(schema=artifact.PACKED_SCHEMA, fingerprint="frozen", stack=reference.model.layers[0].state_dict(),
                    norm=reference.model.norm.state_dict()), checkpoint)
    original_safe_open = artifact.safe_open
    names = []
    class Reader:
        def __init__(self, *a, **kw):
            self.inner = original_safe_open(*a, **kw)
        def __enter__(self):
            self.inner.__enter__(); return self
        def __exit__(self, *args):
            return self.inner.__exit__(*args)
        def get_tensor(self, name):
            names.append(name); return self.inner.get_tensor(name)
    monkeypatch.setattr(artifact, "safe_open", Reader)
    with pytest.raises(ValueError, match="schema/fingerprint"):
        artifact.export_packed_checkpoint(tmp_path/"bad", checkpoint, source, source_fingerprint="wrong", position_capacity=8)
    artifact.export_packed_checkpoint(tmp_path/"bundle", checkpoint, source, source_fingerprint="frozen", position_capacity=8)
    assert names == ["model.embed_tokens.weight"]
    monkeypatch.setattr(artifact, "safe_open", original_safe_open)
    restored, _ = artifact.load_artifact(tmp_path/"bundle")
    assert restored.lm_head.weight.dtype == reference.lm_head.weight.dtype
    with torch.inference_mode():
        ids = torch.tensor([[4, 5, 4]])
        assert torch.equal(reference(ids).logits, restored(ids).logits)


def test_rejects_tampered_files_invalid_cache_metadata_and_unstripped_extra_state(tmp_path):
    source, reference, _ = fixture(tmp_path)
    output = tmp_path/"bundle"
    artifact.export_artifact(output, reference.model.layers[0], reference.model.norm, source, "frozen")
    metadata = json.loads((output/"structure.json").read_text())
    metadata["stack"]["blocks"][1]["layer_idx"] = 4
    (output/"structure.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="hash mismatch"):
        artifact.load_artifact(output)
    manifest = json.loads((output/"manifest.json").read_text())
    manifest["files"]["structure.json"] = artifact.sha(output/"structure.json")
    (output/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="36 original cache"):
        artifact.load_artifact(output)
    state = deepcopy(reference.model.layers[0].state_dict())
    state["_extra_state"]["stripped"] = False
    with pytest.raises(ValueError, match="packed/stripped"):
        artifact._metadata_from_state(state, artifact._config(source))
    with pytest.raises(FileExistsError):
        artifact.export_artifact(output, reference.model.layers[0], reference.model.norm, source, "frozen")


def test_rejects_nonfinite_weights_and_nonstandard_rope(tmp_path):
    source, reference, _ = fixture(tmp_path)
    with torch.no_grad():
        reference.model.layers[0].blocks[0].qkv_weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        artifact.export_artifact(tmp_path/"bad", reference.model.layers[0], reference.model.norm, source, "frozen")
    config = json.loads((source/"config.json").read_text())
    config["rope_scaling"] = {"rope_type": "linear", "factor": 2.}
    (source/"config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="default RoPE"):
        artifact._config(source)


def test_cli_contract_and_cpu_export_guard(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        artifact.export_campaign(tmp_path, tmp_path/"bundle")
    calls = []
    monkeypatch.setattr(artifact, "export_campaign", lambda root, output: calls.append((root, output)) or {"exported": True})
    monkeypatch.setattr(sys, "argv", ["artifact", "export", "--campaign", str(tmp_path), "--output", str(tmp_path/"bundle")])
    artifact.main()
    assert calls == [(tmp_path, tmp_path/"bundle")]
    assert json.loads(capsys.readouterr().out)["exported"]
    monkeypatch.setattr(artifact, "validate_campaign", lambda *args: {"passed": False})
    monkeypatch.setattr(sys, "argv", ["artifact", "validate", "--campaign", str(tmp_path), "--output", str(tmp_path/"bundle")])
    with pytest.raises(SystemExit) as failure:
        artifact.main()
    assert failure.value.code == 1


def completed_campaign_fixture(tmp_path, monkeypatch, schema):
    from experiments.structure_net import jspace_qwen_parallel as campaign
    from experiments.structure_net import jspace_allblocks_v2 as v2

    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    root = tmp_path/"campaign"; root.mkdir()
    producer = root/"producer"; producer.mkdir()
    source = tmp_path/"runtime.py"; source.write_text("frozen producer\n")
    shutil.copy2(source, producer/source.name)
    monkeypatch.setattr(campaign, "sources", lambda: [source])
    monkeypatch.setattr(v2, "sources", lambda: [source])
    tasks = {"splits": {}}
    (root/"tasks.json").write_text(json.dumps(tasks))
    config = v2.CONFIG if schema == v2.SCHEMA else campaign.CONFIG
    plan = dict(schema=schema, config=config, source_sha256={str(source): campaign.sha(source)},
                task_digest=campaign.digest(tasks), model={"path": "native-path-not-read-by-verification"})
    plan["fingerprint"] = campaign.digest(plan)
    (root/"plan.json").write_text(json.dumps(plan))
    evaluation = root/"evaluation"; evaluation.mkdir()
    (evaluation/"manifest.json").write_text(json.dumps(dict(status="completed", fingerprint=plan["fingerprint"], artifacts={})))
    (root/"campaign_results.json").write_text(json.dumps(dict(status="completed", fingerprint=plan["fingerprint"])))
    return root, source, plan


@pytest.mark.parametrize("schema", ["jspace.qwen-parallel.v1", "jspace.qwen-parallel.v2"])
def test_export_dispatches_recorded_protocol_and_restores_source_guards(tmp_path, monkeypatch, schema):
    from experiments.structure_net import jspace_qwen_parallel as campaign
    from experiments.structure_net import jspace_qwen_parallel_evaluate as evaluator

    root, source, plan = completed_campaign_fixture(tmp_path, monkeypatch, schema)
    before = {key: getattr(campaign, key) for key in ("CONFIG", "SCHEMA", "MODULE", "sources")}
    acceptance = evaluator.acceptance_gates
    calls = []
    real_verify = campaign.verify
    def verified(*args, **kwargs):
        calls.append((campaign.SCHEMA, campaign.CONFIG["maximum_accuracy_drop"]))
        return real_verify(*args, **kwargs)
    monkeypatch.setattr(campaign, "verify", verified)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(artifact, "export_packed_checkpoint", lambda *args, **kwargs: kwargs)
    assert artifact.export_campaign(root, tmp_path/"bundle")["source_fingerprint"] == plan["fingerprint"]
    assert calls == [(schema, plan["config"]["maximum_accuracy_drop"])]*2
    for key, value in before.items():
        assert getattr(campaign, key) is value
    assert evaluator.acceptance_gates is acceptance
    source.write_text("tampered producer\n")
    with pytest.raises(ValueError, match="Frozen producer changed"):
        artifact.export_campaign(root, tmp_path/"bundle")
    for key, value in before.items():
        assert getattr(campaign, key) is value
    assert evaluator.acceptance_gates is acceptance


def test_validation_dispatches_v2_before_gpu_and_unknown_schema_is_rejected(tmp_path, monkeypatch):
    from experiments.structure_net import jspace_qwen_parallel as campaign
    from experiments.structure_net import jspace_full_lease

    root, source, plan = completed_campaign_fixture(tmp_path, monkeypatch, "jspace.qwen-parallel.v2")
    before = {key: getattr(campaign, key) for key in ("CONFIG", "SCHEMA", "MODULE", "sources")}
    monkeypatch.setattr(jspace_full_lease, "wait_ready", lambda *args: {"id": "test-lease"})
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: None)
    monkeypatch.setattr(torch, "manual_seed", lambda *args: None)
    class BeforeGPU(Exception):
        pass
    def stop_before_gpu():
        raise BeforeGPU("Protocol validation succeeded before GPU setup")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", stop_before_gpu)
    with pytest.raises(BeforeGPU):
        artifact.validate_campaign(root, tmp_path/"bundle")
    for key, value in before.items():
        assert getattr(campaign, key) is value
    source.write_text("tampered producer\n")
    with pytest.raises(ValueError, match="Frozen producer changed"):
        artifact.validate_campaign(root, tmp_path/"bundle")
    for key, value in before.items():
        assert getattr(campaign, key) is value
    plan["schema"] = "jspace.qwen-parallel.unsupported"
    (root/"plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Unsupported artifact source"):
        artifact._completed_campaign(root)
