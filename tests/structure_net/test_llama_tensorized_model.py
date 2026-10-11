import torch
from transformers import LlamaConfig, LlamaForCausalLM
from experiments.structure_net.llama_tensorized_model import TensorizedLlama
from experiments.structure_net.llama_cotenancy_data import example, score


def test_complete_model_packing_matches_hf_and_cached_decode():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True)
    config._attn_implementation = "sdpa"
    hf = LlamaForCausalLM(config).eval().requires_grad_(False)
    packed = TensorizedLlama.from_hf(hf, replicas=2)
    ids = torch.randint(0, 64, (2, 10))
    with torch.inference_mode():
        expected = hf(ids).logits
        actual, cache = packed(ids, use_cache=True, all_logits=True)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        for i in range(2):
            separate, _ = packed.slice(i)(ids[i:i+1], all_logits=True)
            torch.testing.assert_close(actual[i:i+1], separate, atol=1e-6, rtol=1e-5)
        extra = torch.randint(0, 64, (2, 2))
        cached, _ = packed(extra, past=cache, all_logits=True)
        direct, _ = packed(torch.cat((ids, extra), 1), all_logits=True)
        torch.testing.assert_close(cached, direct[:, -2:], atol=1e-6, rtol=1e-5)
        packed.matrices['model.layers.0.self_attn.q_proj.weight']['cores'][0][0].mul_(2)
        changed, _ = packed(ids, all_logits=True)
        assert not torch.equal(changed[0], actual[0])
        torch.testing.assert_close(changed[1], actual[1], atol=0, rtol=0)


def test_exact_extractive_summary_scoring_penalizes_invented_lines_and_omissions():
    row = example(7, "work", "test")
    gold = '\n'.join(row['gold_fragments'])
    assert score(gold, row)['complete_summary']
    assert score('- ' + gold.replace('\n', '\n- '), row)['complete_summary']
    assert not score(gold + '\nEveryone approved the budget.', row)['complete_summary']
    assert score(row['gold_fragments'][0], row)['fragment_coverage'] == 1/3
    assert score('', row)['supported_line_rate'] == 0
