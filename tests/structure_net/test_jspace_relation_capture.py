"""CPU tests with actual tiny Qwen2 decoder blocks and synthetic training states."""
from __future__ import annotations

import json

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net import jspace_relation_capture as capture


@pytest.fixture(autouse=True)
def one_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


class Tokenizer:
    eos_token_id = 2

    def __init__(self):
        self.messages = []
        self.prompts = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.messages.append(messages)
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return {"input_ids": torch.tensor([[1, 9, 10]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long)}

    def decode(self, ids, **kwargs):
        return "".join({2: "", 3: "red", 4: "blue", 5: " "}.get(int(i), "?") for i in ids)


class Head(torch.nn.Module):
    def __init__(self, answer_id=3, delimiter=True):
        super().__init__()
        self.answer_id, self.delimiter, self.calls = answer_id, delimiter, 0

    def forward(self, hidden):
        token = 5 if self.delimiter and self.calls % 2 else self.answer_id
        self.calls += 1
        logits = torch.full((*hidden.shape[:-1], 16), -10., device=hidden.device)
        logits[..., token] = 10.
        return logits


def model(depth=24):
    torch.manual_seed(31)
    config = Qwen2Config(vocab_size=16, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=depth, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=64, eos_token_id=2)
    config._attn_implementation = "eager"
    result = Qwen2ForCausalLM(config)
    result.lm_head = Head()
    return result


def cases():
    return [{"case_id": "case-1", "prompt": "Return only the colour.", "answer": "red",
             "intermediate": "China", "swap_to": "France", "swap_answer": "blue",
             "mapping": {"China": "red", "France": "blue"}}]


def banks(layers=12):
    controls = capture.build_control_banks(torch.randn(30, layers, 8), seed=7)
    controls["jacobian"] = torch.eye(8)[:4].unsqueeze(0).repeat(layers, 1, 1)
    return controls


def test_natural_real_qwen_captures_all_layers_answers_and_chat_only():
    qwen, tokenizer = model(), Tokenizer()
    qwen.train()
    layers = list(range(1, 24, 2))
    progress = []
    result = capture.capture_natural(qwen, tokenizer, cases(), layers=layers, progress=progress.append)
    assert result["hidden"].shape == (1, 12, 8)
    assert result["hidden"].dtype == torch.float32 and result["hidden"].device.type == "cpu"
    row = result["records"][0]
    assert row["answer"] == "red" and row["raw"] == "red "
    assert row["correct"] is True and row["original_correct"] is True
    assert row["counterfactual_correct"] is False
    assert row["case"] == cases()[0] and row["hidden_row_index"] == 0
    assert row["observation_layers"] == layers and row["source_site"] is None
    assert progress == result["records"]
    assert tokenizer.messages == [[{"role": "user", "content": cases()[0]["prompt"]}]]
    assert tokenizer.prompts == ["<user>Return only the colour.<assistant>"]
    json.dumps(result["records"], allow_nan=False)
    assert qwen.training and not any(block._forward_hooks for block in qwen.model.layers)


def test_control_banks_fit_training_pca_and_share_fixed_random_axes():
    rng = torch.Generator().manual_seed(9)
    training = torch.randn(64, 3, 8, generator=rng).double()
    training[..., 4:] = 0
    training += torch.arange(8).double()
    original = training.clone()
    rng_state = torch.random.get_rng_state().clone()
    result = capture.build_control_banks(training, seed=7)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.equal(training, original)
    expected_projection = torch.diag(torch.tensor([1.] * 4 + [0.] * 4))
    for arm in ("pca", "random"):
        assert result[arm].shape == (3, 4, 8)
        assert result[arm].dtype == torch.float32 and result[arm].device.type == "cpu"
        assert torch.allclose(result[arm] @ result[arm].transpose(-2, -1), torch.eye(4), atol=1e-6)
    assert torch.allclose(result["pca"].transpose(-2, -1) @ result["pca"], expected_projection, atol=1e-6)
    assert torch.equal(result["random"][0], result["random"][1])
    assert torch.equal(result["random"][0], result["random"][2])
    assert torch.equal(result["mean"], training.mean(0).float())
    repeat = capture.build_control_banks(training.clone(), seed=7)
    assert all(torch.equal(result[key], repeat[key]) for key in result)


def test_coefficient_swap_preserves_other_positions_and_pair_complement():
    bank = torch.eye(8, dtype=torch.float64)[:4]
    bank[1] = (bank[0] + bank[1]) / 2**.5
    pair = capture._pair(bank, 0, 1)
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8) / 7
    patch, diagnostics = capture._final_position_patch(pair, pair, arm="jacobian")
    changed = patch(hidden)
    assert torch.equal(hidden[:, :-1], changed[:, :-1])
    before, after = hidden[:, -1] @ pair[1], changed[:, -1] @ pair[1]
    assert torch.allclose(after, before.flip(-1), atol=2e-6)
    projector = pair[1] @ pair[0]
    assert torch.allclose(changed[:, -1] @ (torch.eye(8) - projector),
                          hidden[:, -1] @ (torch.eye(8) - projector), atol=2e-6)
    assert diagnostics["perturbation_l2"] == pytest.approx(diagnostics["jacobian_reference_l2"], rel=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_control_swaps_match_reference_norm_and_report_actual_cast_change(dtype):
    reference = capture._pair(torch.eye(8, dtype=torch.float64)[:4], 0, 1)
    control = capture._pair(capture.random_basis(8, 4, seed=23, dtype=torch.float64), 0, 1)
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8).to(dtype) / 7
    patch, diagnostics = capture._final_position_patch(reference, control, arm="random")
    changed = patch(hidden)
    assert changed.dtype == dtype and torch.equal(changed[:, :-1], hidden[:, :-1])
    assert diagnostics["requested_perturbation_l2"] == pytest.approx(diagnostics["jacobian_reference_l2"], rel=2e-6)
    assert diagnostics["perturbation_l2"] == float((changed[:, -1].float() - hidden[:, -1].float()).norm())


def test_actual_qwen_interventions_are_prefill_final_position_only_and_keep_full_observations(monkeypatch):
    qwen, tokenizer = model(), Tokenizer()
    layers = list(range(1, 24, 2))
    natural = capture.capture_natural(qwen, tokenizer, cases(), layers=layers)
    before_weights = {key: value.clone() for key, value in qwen.state_dict().items()}
    qwen.lm_head = Head(answer_id=4)
    real_generate = capture.generate_answer
    patch_calls = []

    def inspected_generate(*args, **kwargs):
        original_patch = kwargs["patch"]

        def inspected_patch(hidden):
            changed = original_patch(hidden)
            assert torch.equal(changed[:, :-1], hidden[:, :-1])
            patch_calls.append(tuple(hidden.shape))
            return changed

        kwargs["patch"] = inspected_patch
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(capture, "generate_answer", inspected_generate)
    result = capture.capture_interventions(qwen, tokenizer, cases(), layers=layers,
                                          banks=banks(), concept_names=capture.COUNTRIES)
    assert result["hidden"].shape == (6, 12, 8)
    assert patch_calls == [(1, 3, 8)] * 6  # Not called on cached continuation tokens.
    assert [(r["source_site"], r["arm"]) for r in result["records"]] == [
        (site, arm) for site in (11, 23) for arm in capture.ARMS]
    for index, row in enumerate(result["records"]):
        assert row["answer"] == "blue" and row["correct"] and row["counterfactual_correct"]
        assert not row["original_correct"] and row["hidden_row_index"] == index
        assert row["requested_perturbation_l2"] == pytest.approx(row["jacobian_reference_l2"], rel=2e-6)
        site_index = layers.index(row["source_site"])
        assert torch.equal(result["hidden"][index, :site_index], natural["hidden"][0, :site_index])
        assert not torch.equal(result["hidden"][index, site_index], natural["hidden"][0, site_index])
    assert all(torch.equal(value, qwen.state_dict()[key]) for key, value in before_weights.items())
    assert not any(block._forward_hooks for block in qwen.model.layers)
    json.dumps(result["records"], allow_nan=False)


def test_incomplete_generation_is_incorrect_and_does_not_filter_the_case():
    qwen = model(depth=2)
    qwen.lm_head = Head(delimiter=False)
    result = capture.capture_natural(qwen, Tokenizer(), cases(), layers=(0, 1))
    assert len(result["records"]) == 1 and result["hidden"].shape == (1, 2, 8)
    row = result["records"][0]
    assert row["answer"] is None and row["answer_status"] == "truncated_or_ambiguous"
    assert not row["correct"] and row["total_generated_tokens"] == 8


def test_control_banks_refuse_nonfinite_or_rank_deficient_training_data():
    with pytest.raises(ValueError, match="rank"):
        capture.build_control_banks(torch.ones(10, 2, 8))
    with pytest.raises(ValueError, match="finite"):
        capture.build_control_banks(torch.full((10, 2, 8), float("nan")))


def test_validation_rejects_duplicate_cases_and_wrong_country_order_before_forward():
    qwen, tokenizer = model(depth=2), Tokenizer()
    with pytest.raises(ValueError, match="Duplicate"):
        capture.capture_natural(qwen, tokenizer, cases() * 2, layers=(0, 1))
    with pytest.raises(ValueError, match="ordered"):
        capture.capture_interventions(qwen, tokenizer, cases(), layers=(0, 1), banks=banks(2),
                                      concept_names=tuple(reversed(capture.COUNTRIES)), sites=(0,))
    assert tokenizer.prompts == [] and qwen.lm_head.calls == 0
