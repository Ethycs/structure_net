"""Actual tiny-Qwen and known-outcome controls for donor localization."""
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net.jspace_donor_localization import (
    CONFIG, aggregate, balanced_selection, compare_logits, forward_record, make_donor_patch, pair_alignment,
)
from experiments.structure_net.jspace_relation_tasks import COUNTRIES, COLOURS


class TinyTokenizer:
    eos_token_id = 2
    def __call__(self, prompt, **kwargs):
        ids = [1, 3, 4, 5 if prompt == "source" else 7, 6]
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, 5, dtype=torch.long)}
    def decode(self, ids, **kwargs):
        return "".join(chr(97+int(i) % 26)+" " for i in ids if int(i) != self.eos_token_id)


def model_and_states():
    torch.manual_seed(19)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64, eos_token_id=2)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config)
    tokenizer = TinyTokenizer()
    source = forward_record(model, tokenizer, "source", layers=(0, 1, 2))
    donor = forward_record(model, tokenizer, "donor", layers=(0, 1, 2))
    return model, tokenizer, source, donor


def test_terminal_copy_reproduces_actual_qwen_logits_exactly():
    model, tokenizer, source, donor = model_and_states()
    patch, diagnostics = make_donor_patch(donor["states"][2][3:], 3, "final")
    changed = forward_record(model, tokenizer, "source", layer=2, patch=patch)
    assert compare_logits(changed["logits"], donor["logits"])["exact"]
    assert diagnostics["positions"] == 1
    identity, _ = make_donor_patch(source["states"][2][3:], 3, "final")
    restored = forward_record(model, tokenizer, "source", layer=2, patch=identity)
    assert torch.equal(restored["logits"], source["logits"])
    assert all(len(block._forward_hooks) == 0 for block in model.model.layers)
    assert len(model.lm_head._forward_hooks) == 0


def test_full_suffix_reproduces_donor_at_every_layer_with_identical_prefix():
    model, tokenizer, source, donor = model_and_states()
    for layer in (0, 1, 2):
        assert torch.equal(source["states"][layer][:3], donor["states"][layer][:3])
        patch, _ = make_donor_patch(donor["states"][layer][3:], 3, "query_suffix")
        changed = forward_record(model, tokenizer, "source", layer=layer, patch=patch)
        assert torch.equal(changed["logits"], donor["logits"])
    patch, _ = make_donor_patch(donor["states"][2][3:], 3, "query_except_final")
    unchanged_output = forward_record(model, tokenizer, "source", layer=2, patch=patch)
    assert torch.equal(unchanged_output["logits"], source["logits"])


@pytest.mark.parametrize("scope,bounds", [("final", (4,5)), ("query_suffix", (2,5)), ("query_except_final", (2,4))])
def test_masks_copy_exact_positions_and_preserve_others(scope, bounds):
    hidden = torch.arange(20).reshape(1,5,4).to(torch.bfloat16)
    suffix = torch.full((3,4), 9., dtype=torch.bfloat16)
    patch, diagnostics = make_donor_patch(suffix, 2, scope)
    changed = patch(hidden)
    first, last = bounds
    assert torch.equal(changed[:, :first], hidden[:, :first])
    assert torch.equal(changed[:, last:], hidden[:, last:])
    assert torch.equal(changed[0, first:last], suffix[first-2:last-2])
    assert diagnostics["positions"] == last-first


def test_failed_patch_cleans_all_hooks():
    model, tokenizer, _, _ = model_and_states()
    bad_patch, _ = make_donor_patch(torch.zeros(2,16), 1, "final")
    with pytest.raises(ValueError, match="alignment"):
        forward_record(model, tokenizer, "source", layers=(0,1,2), layer=1, patch=bad_patch)
    assert all(len(block._forward_hooks) == 0 for block in model.model.layers)
    assert not model.lm_head._forward_hooks


def test_alignment_rejects_length_and_prefix_changes():
    class CharacterTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]+"\nAssistant:"
        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i,i+1) for i in range(len(text))]}
    tokenizer = CharacterTokenizer()
    source = {"prompt": "Same facts\nQuestion Ada?", "question": "Question Ada?"}
    donor = {"prompt": "Same facts\nQuestion Ben?", "question": "Question Ben?"}
    assert pair_alignment(tokenizer, source, donor)["question_start"] == 11
    assert pair_alignment(tokenizer, source, {"prompt": "Same facts\nQuestion Ruby?", "question": "Question Ruby?"}) is None
    assert pair_alignment(tokenizer, source, {"prompt": "Else facts\nQuestion Ben?", "question": "Question Ben?"}) is None


def test_balanced_selection_is_deterministic_and_bounded():
    candidates = [{"source": {"intermediate": country, "answer": colour, "case_id": f"{country}-{colour}-{i}"}}
                  for country in COUNTRIES for colour in COLOURS for i in range(4)]
    chosen = balanced_selection(candidates, 2)
    assert len(chosen) == 32 and chosen == balanced_selection(list(reversed(candidates)), 2)
    with pytest.raises(ValueError, match="Insufficient"):
        balanced_selection(candidates, 5)


def test_known_grid_identifies_distributed_effect_and_rejects_missing_cells():
    cases = [{"source": {"case_id": str(i)}} for i in range(16)]
    rows = []
    for layer in CONFIG["layers"]:
        for scope in CONFIG["scopes"]:
            success = scope == "query_suffix" or (layer == 35 if scope == "final" else layer < 35)
            for case in cases:
                rows.append({"case_id": case["source"]["case_id"], "layer": layer, "scope": scope,
                             "target_correct": success, "original_correct": not success,
                             "donor_logits": {"argmax_equal": success, "exact": success, "max_absolute_difference": 0 if success else 1}})
    result = aggregate(rows, cases, {"eligible": [True]*16})
    assert result["primary_metric"] == 1
    assert result["classification"] == "distributed_question_state_supported"
    assert result["usable_layers"]["final"] == [35]
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate(rows[:-1], cases, {"eligible": [True]*16})
