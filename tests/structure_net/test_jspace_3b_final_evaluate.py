from copy import deepcopy
from types import SimpleNamespace
import shutil
import numpy as np

import pytest
import torch
from safetensors.torch import save_file
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_3b_final_evaluate as final
from experiments.structure_net.jspace_qwen_parallel_model import state_dict_cpu


@pytest.fixture
def components(tmp_path):
    torch.manual_seed(8007)
    cfg = Qwen2Config(hidden_size=8, intermediate_size=12, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=16,
        max_position_embeddings=32, tie_word_embeddings=True, attention_dropout=0.)
    cfg._attn_implementation = "eager"
    native = Qwen2ForCausalLM(cfg).bfloat16().eval().requires_grad_(False)
    path = tmp_path/"native"; path.mkdir(); cfg.save_pretrained(path)
    generation = GenerationConfig(eos_token_id=0, pad_token_id=0, repetition_penalty=1.05)
    generation.save_pretrained(path)
    save_file({k: v.detach().clone() for k, v in native.state_dict().items()
               if k != "lm_head.weight"}, path/"model.safetensors")
    fits = {}
    for layer in range(3):
        fit = tmp_path/f"fit_{layer}"; fit.mkdir(); fits[layer] = fit
        torch.save(dict(basis=torch.linalg.qr(torch.randn(8, 8)).Q, means=torch.randn(10, 8)/100,
            rope_pairs=torch.tensor([[0]]), value_basis=torch.eye(4)[None, :, :2]), fit/"fit.pt")
        torch.save(dict(anchors=[0, 1], mean=torch.zeros(12), scale=torch.ones(12),
            coefficients={"2": torch.randn(12, 6)/500}, removed=torch.tensor([10, 11])), fit/"relation.pt")
    stack, norm, embedding, rotary, _ = final.builder.build_stack(path, fits, 0,
        config=cfg, rank=4, trainable=False, checkpoint_blocks=False)
    with torch.no_grad():
        norm.weight.fill_(1.25)
        stack.blocks[1].qkv_weight.add_(.00390625)
    state = dict(stack=state_dict_cpu(stack), norm=state_dict_cpu(norm))
    checkpoint = tmp_path/"selected.pt"; torch.save(state, checkpoint)
    return path, fits, checkpoint, cfg, generation, stack, norm, embedding, rotary


def compact(parts):
    path, fits, checkpoint, cfg, generation, *_ = parts
    return final.build_compact_model(path, fits, checkpoint, config=cfg,
                                    generation_config=generation, rank=4)


def test_meta_full_model_retains_tied_embedding_native_cache_config_and_selected_norm(components):
    model = compact(components)
    _, _, _, cfg, generation, stack, norm, embedding, rotary = components
    assert len(model.model.layers) == 1 and model.config.num_hidden_layers == 3
    assert model.model.layers[0].layer_indices == (0, 1, 2)
    assert model.model.layers[0].packed and model.model.layers[0].stripped
    assert model.model.embed_tokens.weight is model.lm_head.weight
    assert torch.equal(model.model.norm.weight, norm.weight)
    assert model.generation_config.to_dict() == generation.to_dict()
    assert not any(t.is_meta for t in (*model.parameters(), *model.buffers()))
    assert all(not p.requires_grad for p in model.parameters())
    ids = torch.tensor([[1, 2, 3, 4]])
    x = embedding[ids]; positions = torch.arange(4)[None]
    reference = final.builder.packed_clone(stack)(x, position_ids=positions,
        position_embeddings=rotary(x, positions))[0]
    actual = model.model(input_ids=ids, use_cache=False).last_hidden_state
    assert torch.equal(actual, norm(reference))
    assert final.builder.tensor_bytes(model) == final.builder.tensor_bytes(model.model.layers[0], norm, embedding, rotary)


def test_real_compact_dynamic_static_cache_matches_full_prefill_and_zero_padding(components):
    model = compact(components)
    value = final.validate_cache(model, torch.tensor([[1, 2, 3, 4, 5]]), capacity=8)
    assert value["passed"] and value["future_cells_poisoned"] and value["static_padding_zero"]
    assert value["dynamic_head_widths"] == [2]*3
    assert value["static_head_widths"] == [4]*3
    assert value["cache_layers"] == 3


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_native_loader_explicit_backend_and_real_cached_controls(components, backend):
    path = components[0]
    native = final.load_native_model(path, attention=backend)
    assert native.config._attn_implementation == backend
    assert native.final_evaluation_backend == "native_"+backend
    assert native.lm_head.weight is native.model.embed_tokens.weight
    result = final.validate_cache(native, torch.tensor([[1, 2, 3, 4]]), capacity=8)
    assert result["passed"] and result["dynamic_head_widths"] == [4]*3


def test_raw_choice_scoring_and_fresh_build_are_exact(components):
    first, second = compact(components), compact(components)
    cases = [dict(identity="one", family="a", input_ids=[1, 3, 2], choices=["a", "b"], answer=1)]
    a = final.score_model(first, cases, [2, 5])
    b = final.score_model(second, cases, [2, 5])
    assert a == b
    hidden = first.model(input_ids=torch.tensor([[1, 3, 2]]), use_cache=False).last_hidden_state[:, -1]
    expected = torch.nn.functional.linear(hidden, first.lm_head.weight[[2, 5]])[0].float()
    assert a[0]["logits"] == expected.tolist() and a[0]["prediction"] == int(expected.argmax())


def test_partial_stack_and_missing_lease_are_rejected_before_cuda(components, monkeypatch):
    path, fits, _, cfg, generation, *_ = components
    stack, norm, embedding, rotary, _ = final.builder.build_stack(path, fits, 1, rank=4, config=cfg, trainable=False)
    with pytest.raises(ValueError, match="cut0"):
        final.attach_compact_model(final.builder.packed_clone(stack), norm, embedding, rotary, cfg, generation)
    touched = []; monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: touched.append(True))
    with pytest.raises(ValueError, match="reservation"):
        final.load_native_model(path, device="cuda", attention="sdpa")
    with pytest.raises(ValueError, match="reservation"):
        final.build_compact_model(path, fits, "absent", device="cuda")
    assert not touched
    with pytest.raises(ValueError, match="explicit"):
        final.load_native_model(path, attention="automatic")


def test_runtime_five_repeats_include_prefill_and_exclude_setup(monkeypatch):
    from experiments.structure_net import jspace_qwen_affine as base
    class Decoder:
        def capture(self): return 7.
        def prefill(self, ids): return torch.zeros(1, 8)
        def generate(self, ids, captured): return torch.zeros(32, dtype=torch.long)
    class Model:
        lm_head = SimpleNamespace(weight=torch.zeros(16, 8))
        def generate(self, ids, **kw): return torch.cat((ids, torch.zeros(1, 32, dtype=torch.long)), 1)
    monkeypatch.setattr(base, "graph_decoder", lambda model: Decoder())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(final.scoring, "_timed", lambda call: (call(), .25))
    cases = [dict(identity=str(i), input_ids=[1, 2, 3]) for i in range(8)]
    value = final._runtime(Model(), SimpleNamespace(eos_token_id=0), cases, seed=42, guard=lambda: None)
    assert value["graph_static_exact"] and value["graph_capture_seconds"] == 7.
    assert len(value["timings"]["cuda_graph"]["samples"]) == 40
    assert value["timings"]["cuda_graph"]["tokens_per_second"] == 128.
    assert len(value["prefill"]["samples"]) == 40
    assert all(value["hf_graph_tokens_equal"])


def test_final_gate_uses_aggregate_five_points_not_family_or_eager_speed():
    rows = [dict(identity=str(i), family=str(i//128), answer=0, prediction=0, logits=[1., 0.]) for i in range(512)]
    compact_rows = deepcopy(rows)
    for row in compact_rows[:25]: row["prediction"] = 1
    quality = final.scoring.paired_quality(rows, compact_rows)
    args = dict(native_bytes=10000, compact_bytes=6200,
                sdpa_before=dict(total_tokens=120, total_seconds=15.),
                sdpa_after=dict(total_tokens=120, total_seconds=10.), compact_rate=13.44,
                reload_exact=True, cache_valid=True, graph_static_exact=True)
    result = final.final_gates(quality, **args)
    assert result["passed"] and not result["family_accuracy_is_gate"]
    assert result["family_diagnostics"]["0"]["accuracy_drop"] > .05
    compact_rows[25]["prediction"] = 1
    assert not final.final_gates(final.scoring.paired_quality(rows, compact_rows), **args)["gates"]["aggregate_quality"]
    assert result["native_sdpa_pooled_rate"] == 9.6
    assert not final.final_gates(quality, **dict(args, compact_rate=13.43))["passed"]
    assert not final.final_gates(quality, **dict(args, compact_bytes=6201))["passed"]
    assert not final.final_gates(quality, **dict(args, cache_valid=False))["passed"]


def test_tf32_policy_restored_on_failure():
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with pytest.raises(RuntimeError), final.arithmetic_policy():
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        raise RuntimeError("test")
    assert before == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def test_standalone_full36_bundle_reloads_without_native_or_fit_access(components, tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from experiments.structure_net.jspace_qwen_compact_stack import CompactQwenStack
    path, fits, _, cfg, generation, stack, norm, embedding, rotary = components
    cfg = deepcopy(cfg); cfg.num_hidden_layers = 36; cfg.layer_types = ["full_attention"]*36; cfg.save_pretrained(path)
    vocabulary = {str(i): i for i in range(16)}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="3")),
        unk_token="3", pad_token="0", eos_token="2", bos_token="1")
    tokenizer.save_pretrained(path)
    prototype = deepcopy(stack.blocks[0])
    prototype.basis = torch.load(fits[0]/"fit.pt", weights_only=True)["basis"][:, :4].clone()
    with torch.no_grad():
        prototype.attention_output.mul_(.03); prototype.mlp_output.mul_(.03)
        prototype.polynomial_output.mul_(.03); prototype.polynomial_bias.zero_()
    blocks = [deepcopy(prototype) for _ in range(36)]
    for i, block in enumerate(blocks): block.layer_idx = i
    full = CompactQwenStack(blocks, geometry_mode="bf16", pad_to_multiple=128)
    original = final.attach_compact_model(final.builder.packed_clone(full), norm, embedding, rotary, cfg, generation)
    bundle = tmp_path/"standalone"
    receipt = final.export_model_bundle(bundle, original, path, "accepted-cut0", position_capacity=10)
    assert receipt["embedding_copies"] == 1
    assert receipt["resident_tensor_bytes"] == final.builder.tensor_bytes(original)
    with pytest.raises(ValueError, match="different"):
        final.load_model_bundle(bundle, expected_fingerprint="wrong")
    shutil.rmtree(path)
    for directory in fits.values(): shutil.rmtree(directory)
    monkeypatch.setattr(final.builder.native, "load_native_block", lambda *a, **kw: pytest.fail("Read native compiler source"))
    loaded, local_tokenizer = final.load_model_bundle(bundle, expected_fingerprint="accepted-cut0")
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight
    assert loaded.model.layers[0].layer_indices == tuple(range(36))
    cases = [dict(identity="bundle", family="a", input_ids=[4, 5, 4], choices=["a", "b"], answer=0)]
    assert final.score_model(loaded, cases, [4, 5]) == final.score_model(original, cases, [4, 5])
    assert final.builder.tensor_bytes(loaded) == receipt["resident_tensor_bytes"]
    assert local_tokenizer is not None


def test_development_replay_matches_original_trainer_full_vocab_batch_shapes(components, tmp_path):
    from experiments.structure_net.jspace_qwen_parallel_train import BoundaryDataset, SuffixTrainer
    model = compact(components); labels = [2, 5, 7, 9]
    cases = []; offset = 0
    for i, length in enumerate([3, 4, 3, 4, 3, 6, 4, 5]):
        ids = list(range(1, length+1)); positions = [0, 1, length-2, length-1]
        cases.append(dict(identity=str(i), family="tiny", split="train" if i<4 else "development",
            input_ids=ids, choices=["a", "b"] if i%2 else ["a", "b", "c", "d"], answer=i%2,
            loss_positions=positions, targets=[ids[1], ids[2], ids[-1], labels[i%2]], offset=offset, stop=offset+length))
        offset += length
    boundary = torch.cat([model.model.embed_tokens(torch.tensor(c["input_ids"])).detach() for c in cases])
    np.save(tmp_path/"boundary.npy", boundary.view(torch.int16).numpy())
    np.save(tmp_path/"teacher.npy", np.zeros((8, 4, 16), dtype=np.int16))
    dataset = BoundaryDataset(cases, tmp_path/"boundary.npy", tmp_path/"teacher.npy")
    probe = SimpleNamespace(stack=model.model.layers[0], norm=model.model.norm, readout=model.lm_head.weight,
                            rotary=model.model.rotary_emb, device=torch.device("cpu"))
    batch = dataset.batch(dataset.development)
    with torch.inference_mode(): expected = SuffixTrainer.logits(probe, batch)
    records = final.score_development(model, cases[4:], labels)
    for i, row in enumerate(records):
        assert row["logits"] == expected[i, -1, labels[:len(cases[i+4]["choices"]) ]].tolist()
    dataset.close()
