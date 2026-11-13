"""CPU numerical checks of the causal estimator, not learned task accuracy."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net.jspace_lens import (
    LensFit, apply_full_lens, capture_residuals, fit_full_jacobians,
    fit_token_rows, token_directions,
)


class CharacterTokenizer:
    def __call__(self, text, *, return_tensors, truncation, max_length):
        return BatchEncoding({"input_ids": torch.tensor([[ord(c) % 16 for c in text[:max_length]]])})

    def decode(self, ids):
        return str(ids[0])


class CausalSum(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer("matrix", matrix)

    def forward(self, hidden):
        return hidden + hidden.cumsum(dim=1) @ self.matrix.T


class ToyDecoder(nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, len(matrix))
        self.layers = nn.ModuleList([nn.Identity(), CausalSum(matrix)])
        self.norm = nn.Identity()

    def forward(self, *, input_ids, use_cache):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        matrix = torch.tensor([[0.2, -0.1, 0.0], [0.3, 0.5, 0.1], [0.0, 0.1, -0.2]])
        self.model = ToyDecoder(matrix)
        self.lm_head = nn.Linear(3, 16, bias=False)
        self.config = SimpleNamespace(_name_or_path="analytic_causal_toy", _commit_hash=None)

    def get_output_embeddings(self):
        return self.lm_head


def test_causal_future_sum_and_equal_prompt_average():
    model = ToyLM()
    # Valid lengths 3 and 5: average triangular sums give factors 2 and 3.
    result = fit_full_jacobians(model, CharacterTokenizer(), ["abcd", "abcdef"],
                               source_layers=[0], token_ids=[1, 2], dim_batch=2,
                               max_seq_len=8, skip_first=0)
    expected = torch.eye(3) + 2.5 * model.model.layers[1].matrix
    torch.testing.assert_close(result.jacobians[0], expected)
    torch.testing.assert_close(result.token_rows[0], model.lm_head.weight[[1, 2]] @ expected)
    assert result.metadata["valid_positions_per_prompt"] == [3, 5]
    assert not any(p.requires_grad for p in model.parameters())
    assert not any(block._forward_hooks for block in model.model.layers)


@pytest.mark.parametrize("dim_batch", [1, 3])
def test_selected_vjp_matches_full_jacobian_on_real_qwen(dim_batch):
    torch.manual_seed(11)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=16,
        attention_dropout=0.0, _attn_implementation="eager",
    ))
    tokenizer = CharacterTokenizer()
    kwargs = dict(source_layers=[0, 1], token_ids=[5, 2, 9], dim_batch=dim_batch,
                  max_seq_len=8, skip_first=1)
    full = fit_full_jacobians(model, tokenizer, ["abcdef", "fabcde"], **kwargs)
    selected = fit_token_rows(model, tokenizer, ["abcdef", "fabcde"], **kwargs)
    assert not selected.jacobians
    for layer in [0, 1]:
        torch.testing.assert_close(selected.token_rows[layer], full.token_rows[layer], rtol=1e-5, atol=1e-7)
    assert selected.metadata["token_ids"] == [5, 2, 9]
    assert selected.metadata["bounded_dictionary_pilot"]
    assert selected.metadata["backward_passes_per_prompt"] == (3 + dim_batch - 1) // dim_batch
    residuals, ids = capture_residuals(model, tokenizer, "abcdef", source_layers=[0, 1, 2], positions=[-1])
    assert ids.shape == (1, 6)
    assert residuals[0].shape == (1, 8)
    assert apply_full_lens(model, full, residuals[0], 0).shape == (1, 16)
    with pytest.raises(ValueError, match="full Jacobian"):
        apply_full_lens(model, selected, residuals[0], 0)


def test_skip_positions_metadata_and_safe_roundtrip(tmp_path):
    model = ToyLM()
    progress = []
    result = fit_token_rows(model, CharacterTokenizer(), ["abcdef"], source_layers=[0],
                            token_ids=[3, 4], dim_batch=3, skip_first=2, max_seq_len=6,
                            progress_callback=progress.append)
    expected = model.lm_head.weight[[3, 4]] @ (torch.eye(3) + 2 * model.model.layers[1].matrix)
    torch.testing.assert_close(result.token_rows[0], expected)
    assert progress[0]["completed_prompts"] == 1
    path = tmp_path / "lens.pt"
    result.save(path)
    restored = LensFit.load(path)
    assert restored.metadata == result.metadata
    torch.testing.assert_close(restored.token_rows[0], result.token_rows[0])
    torch.testing.assert_close(token_directions(restored, 0, normalize=True).norm(dim=-1), torch.ones(2))


@pytest.mark.parametrize("changes", [
    {"source_layers": [1]}, {"source_layers": [-1]}, {"source_layers": []},
    {"token_ids": [1, 1]}, {"token_ids": [-1]}, {"token_ids": [16]},
    {"dim_batch": 0}, {"skip_first": -1}, {"prompts": []}, {"prompts": ["x"]},
])
def test_invalid_pilot_config_fails_before_recording(changes):
    kwargs = dict(prompts=["abcd"], source_layers=[0], token_ids=[1], dim_batch=1,
                  max_seq_len=8, skip_first=0)
    kwargs.update(changes)
    model = ToyLM()
    with pytest.raises(ValueError):
        fit_token_rows(model, CharacterTokenizer(), **kwargs)
    assert not any(block._forward_hooks for block in model.model.layers)
