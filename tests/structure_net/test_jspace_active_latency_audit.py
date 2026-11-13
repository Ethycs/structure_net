"""Check the matched readout against actual tiny-Qwen logits."""
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from experiments.structure_net.jspace_active_latency_audit import teacher_choice


def test_teacher_choice_matches_full_forward_and_restricted_logits():
    torch.manual_seed(17)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    config._attn_implementation = "eager"
    model = Qwen2ForCausalLM(config).eval().requires_grad_(False)
    encoded = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    def tokenizer(prompt, **kwargs):
        return encoded
    with torch.inference_mode():
        reference = model(**encoded, use_cache=False).logits[0, -1]
    winner = int(reference.argmax())
    candidates = [winner]+[i for i in range(32) if i != winner][:3]
    weights = model.lm_head.weight[candidates].detach().clone()
    assert teacher_choice(model, tokenizer, "prompt") == winner
    assert teacher_choice(model, tokenizer, "prompt", weights) == 0
    assert torch.allclose(reference[candidates], model.lm_head(model.model(**encoded, use_cache=False).last_hidden_state)[0, -1, candidates])
