from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.structure_net import ncm_continuous_codec_target as run
from experiments.structure_net.ncm_blockwise_codec import to_block_major


def tiny_codec():
    spec = importlib.util.spec_from_file_location("continuous_target_test_model",
        Path("tests/structure_net/test_ncm_generated_history_core.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forecaster, original, *_ = module.model()
    return original, forecaster.generator.vqvae_local[0]


def test_actual_codec_uses_complete_independent_blocks_and_frozen_state():
    original, facade = tiny_codec()
    before = deepcopy(original.state_dict())
    words = torch.randint(0, 30, (2, facade.context_length))
    codes = facade.encode(words)
    latents = run.latent_arms(facade, words, codes)
    continuous = original.encode_z(words.reshape(8, original.context_length)).reshape_as(latents[run.ARMS[1]])
    quantized = original.vq.idx_to_z_q(to_block_major(codes, facade.vq.block_levels, 4)).reshape_as(continuous)
    assert torch.equal(latents[run.ARMS[1]], continuous)
    assert torch.equal(latents[run.ARMS[0]], quantized)
    assert not torch.equal(continuous, quantized)
    assert not any(value.requires_grad for value in latents.values())
    assert all(torch.equal(value, original.state_dict()[key]) for key, value in before.items())
    assert all(value.grad is None for value in original.parameters())


def test_incomplete_frames_or_unfrozen_codec_are_rejected():
    original, facade = tiny_codec()
    words = torch.randint(0, 30, (1, facade.context_length))
    codes = facade.encode(words)
    with pytest.raises(ValueError, match="complete"):
        run.latent_arms(facade, words[:, :-1], codes)
    original.train()
    with pytest.raises(ValueError, match="frozen"):
        run.latent_arms(facade, words, codes)


def test_neural_readout_accepts_only_latents_and_excludes_padding_vocabulary():
    original, facade = tiny_codec()
    words = torch.randint(0, 30, (1, facade.context_length))
    latent = original.encode_z(words.reshape(4, 8)).reshape(1, 32, 16)
    logits = run.neural_logits(facade, latent, vocab_limit=31)
    assert logits.shape == (1, 32, 31) and logits.argmax(-1).max() < 31
    assert not logits.requires_grad


def test_raw_output_replay_rejects_changed_ids_even_if_counts_are_unchanged():
    source = {"id": "a", "word_ids": [7, 50256] + [50257] * 190,
        "target_text": "w7", "source_sha256": "source", "diagnostic_group": "original_16"}
    arm = {"raw_word_ids": [7, 50256] + [8] * 190, "correct": 2, "total": 2,
        "text": "w7", "global_EOS_position": 1, "exact_target_text": True,
        "CE": .1, "latent_sha256": "a" * 64}
    item = {"id": "a", "target_word_ids": source["word_ids"], "target_text": "w7",
        "source_sha256": "source", "diagnostic_group": "original_16",
        "arms": {name: deepcopy(arm) for name in run.ARMS}}
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: " ".join(f"w{i}" for i in ids))
    run.replay_case(item, source, tokenizer)
    item["arms"][run.ARMS[1]]["raw_word_ids"][0] = 9
    with pytest.raises(ValueError, match="raw decoder"):
        run.replay_case(item, source, tokenizer)


def components(old=199, added=199):
    return {group: {"cases": n, "arms": {run.ARMS[1]: {"correct": correct, "total": total}}}
        for group, n, correct, total in (("original_16", 16, old, 200),
            ("additional_16", 16, added, 200), ("all_32", 32, old + added, 400))}


def test_fidelity_gate_inclusive_and_each_group_and_composition_required():
    assert run.fidelity_gate(components())
    assert not run.fidelity_gate(components(198, 200))
    broken = components()
    broken["all_32"]["arms"][run.ARMS[1]]["total"] = 399
    with pytest.raises(ValueError, match="compose"):
        run.fidelity_gate(broken)
    broken = components()
    broken["additional_16"]["cases"] = 15
    with pytest.raises(ValueError, match="complete"):
        run.fidelity_gate(broken)


def test_eos_is_counted_and_target_pad_is_ignored():
    target = torch.tensor([[5, 50256, 50257]])
    actual = torch.tensor([[5, 0, 50256]])
    assert run.clean.diagnostic.counts(actual, target) == {"correct": 1, "total": 2}


def test_cpu_contract_prevents_cuda_use(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        run.cpu_only()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(ValueError, match="CPU-only"):
        run.cpu_only()


def test_aggregation_preserves_groups_counts_and_exact_texts():
    rows = []
    for i in range(32):
        rows.append({"diagnostic_group": "original_16" if i < 16 else "additional_16",
            "arms": {arm: {"correct": 2 if arm == run.ARMS[1] else 1,
                            "total": 2, "exact_target_text": arm == run.ARMS[1]} for arm in run.ARMS}})
    values = run.aggregate(rows)
    assert values["all_32"]["arms"][run.ARMS[1]]["correct"] == 64
    assert values["all_32"]["arms"][run.ARMS[0]]["correct"] == 32
    assert values["original_16"]["arms"][run.ARMS[1]]["exact_target_texts"] == 16
    assert run.fidelity_gate(values)
