from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net.jspace_runtime import generate_answer, parse_first_answer


class FakeTokenizer:
    eos_token_id = 2
    pieces = {2: "", 3: "Bra", 4: "zil", 5: " ", 6: "Portu", 7: "guese", 8: "\n"}

    def __call__(self, prompt, **kwargs):
        return {"input_ids": torch.tensor([[1, 9, 10]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

    def decode(self, ids, **kwargs):
        return "".join(self.pieces.get(int(token), "?") for token in ids)


class ScriptedHead(torch.nn.Module):
    def __init__(self, ids):
        super().__init__()
        self.ids = list(ids)
        self.shapes = []

    def forward(self, hidden):
        self.shapes.append(tuple(hidden.shape))
        result = torch.full((*hidden.shape[:-1], 16), -10., device=hidden.device)
        result[..., self.ids[len(self.shapes) - 1]] = 10
        return result


def tiny_model(ids):
    torch.manual_seed(11)
    config = Qwen2Config(vocab_size=16, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=64, eos_token_id=2)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config)
    model.lm_head = ScriptedHead(ids)
    return model


def test_parser_waits_for_full_multitoken_word_and_does_not_search_explanations():
    assert parse_first_answer("Portu")["answer"] is None
    assert parse_first_answer("Portuguese")["status"] == "pending"
    assert parse_first_answer("  \"Portuguese\n")["answer"] == "Portuguese"
    assert parse_first_answer("Portuguese", ended=True)["answer"] == "Portuguese"
    assert parse_first_answer("The answer is Portuguese.")["answer"] == "The"
    assert parse_first_answer("12abc ")["status"] == "invalid"


def test_parser_preserves_whole_numbers_and_handles_decimal_chunk_boundaries():
    assert parse_first_answer("26 ")["answer"] == "26"
    assert parse_first_answer("26")["answer"] is None
    assert parse_first_answer("12.")["status"] == "pending"
    assert parse_first_answer("12.5 ")["answer"] == "12.5"
    assert parse_first_answer("12. ")["answer"] == "12"
    assert parse_first_answer("North-")["status"] == "pending"
    assert parse_first_answer("North-West ")["answer"] == "North-West"
    assert parse_first_answer("'", ended=True)["status"] == "empty"


def test_real_qwen_cache_continuation_is_unpatched_and_observations_are_prefill_only():
    model = tiny_model([3, 4, 5])
    model.train()
    calls = []

    def patch(hidden):
        calls.append(tuple(hidden.shape))
        return torch.full_like(hidden, 7.)

    result = generate_answer(model, FakeTokenizer(), "prompt", layer=0, patch=patch,
                             observe_layers=(0, 1), max_new_tokens=4)
    assert result["answer"] == "Brazil"
    assert result["raw"] == "Brazil "
    assert result["token_ids"] == [3, 4, 5]
    assert result["first_token_id"] == 3
    assert result["first_token_text"] == "Bra"
    assert result["total_generated_tokens"] == 3
    assert result["stopped_by"] == "answer_boundary"
    assert calls == [(1, 3, 8)]  # Cache continuation passes never receive the patch.
    assert model.lm_head.shapes == [(1, 8)] * 3  # Never prompt x vocabulary logits.
    assert torch.equal(result["observed"][0], torch.full((8,), 7.))
    assert result["observed"][1].shape == (8,)
    assert all(value.dtype == torch.float32 and value.device.type == "cpu"
               for value in result["observed"].values())
    assert all(len(block._forward_hooks) == 0 for block in model.model.layers)
    assert model.training  # Preserve caller state after an inference-only call.


def test_cap_does_not_credit_an_unfinished_word_and_eos_can_complete_one():
    capped = generate_answer(tiny_model([6]), FakeTokenizer(), "prompt", max_new_tokens=1)
    assert capped["raw"] == "Portu"
    assert capped["answer"] is None
    assert capped["answer_status"] == "truncated_or_ambiguous"
    ended = generate_answer(tiny_model([6, 7, 2]), FakeTokenizer(), "prompt")
    assert ended["answer"] == "Portuguese"
    assert ended["stopped_by"] == "eos"


def test_hooks_are_removed_on_patch_failure_and_on_invalid_patch_shape():
    model = tiny_model([3])

    def broken(hidden):
        raise RuntimeError("deliberate failure")

    with pytest.raises(RuntimeError, match="deliberate"):
        generate_answer(model, FakeTokenizer(), "prompt", layer=0, patch=broken, observe_layers=(1,))
    assert all(len(block._forward_hooks) == 0 for block in model.model.layers)
    with pytest.raises(ValueError, match="preserve"):
        generate_answer(model, FakeTokenizer(), "prompt", layer=0, patch=lambda h: h[:, -1])
    assert all(len(block._forward_hooks) == 0 for block in model.model.layers)


def test_invalid_layer_is_rejected_without_installing_hooks():
    model = tiny_model([3])
    with pytest.raises(ValueError, match="zero-based"):
        generate_answer(model, FakeTokenizer(), "prompt", observe_layers=(3,))
    with pytest.raises(ValueError, match="requires"):
        generate_answer(model, FakeTokenizer(), "prompt", patch=lambda h: h)
    assert all(len(block._forward_hooks) == 0 for block in model.model.layers)
