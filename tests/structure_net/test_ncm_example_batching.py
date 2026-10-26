import pytest
import torch

from experiments.structure_net import ncm_example_batching as batching
from experiments.structure_net import ncm_packed_sequential_objective as objective
from experiments.structure_net.ncm_lm1b_source_attention import SourceAttention, SourceMemory


def test_collation_preserves_all_source_values_and_masks_padding():
    values = [torch.randn(n, 6, requires_grad=True) for n in (2, 5, 3)]
    f, m, g = batching.collate(values, [[1, 2]]*3, device='cpu', width=6, positions=2)
    assert not f.requires_grad and g.dtype == torch.long
    for i, x in enumerate(values):
        assert torch.equal(f[i, :len(x)], x)
        assert m[i].tolist() == [True]*len(x) + [False]*(5-len(x))
        assert torch.count_nonzero(f[i, len(x):]) == 0
    with pytest.raises(ValueError, match='integer'):
        batching.collate(values, [[1.1, 2]]*3, device='cpu', width=6, positions=2)
    with pytest.raises(ValueError, match='FP32'):
        batching.collate([values[0].half()], [[1, 2]], device='cpu', width=6, positions=2)


def test_noise_matches_original_examples_across_different_batch_partitions():
    gold = torch.arange(5*12).reshape(5, 12).remainder(8)
    enabled = [True, False, True, True, False]
    expected_rng = torch.Generator().manual_seed(71)
    expected = [objective.history(gold[i:i+1], (1, 3, 8), 8, expected_rng, flag)
                for i, flag in enumerate(enabled)]
    for sizes in ((5,), (2, 2, 1), (1, 1, 1, 1, 1)):
        rng = torch.Generator().manual_seed(71)
        parts, changes, start = [], [], 0
        for size in sizes:
            value, count = batching.histories(gold[start:start+size], (1, 3, 8), 8,
                                               rng, enabled[start:start+size])
            parts.append(value); changes.extend(count); start += size
        assert torch.equal(torch.cat(parts), torch.cat([x[0] for x in expected]))
        assert changes == [x[1] for x in expected]
        assert torch.equal(rng.get_state(), expected_rng.get_state())


def test_uneven_microbatches_preserve_example_and_level_weight_and_gradients():
    torch.manual_seed(8)
    logits = torch.randn(5, 6, 8, requires_grad=True)
    gold = torch.randint(8, (5, 6))
    whole = batching.weighted_loss(logits, gold, (1, 2, 3), examples_in_update=5)
    split = sum(batching.weighted_loss(logits[a:b], gold[a:b], (1, 2, 3),
                examples_in_update=5) for a, b in ((0, 2), (2, 4), (4, 5)))
    assert torch.allclose(whole, split, atol=1e-6, rtol=1e-6)
    assert torch.allclose(torch.autograd.grad(whole, logits, retain_graph=True)[0],
                          torch.autograd.grad(split, logits)[0], atol=1e-7, rtol=1e-6)


def test_real_source_attention_batch_matches_separate_examples_and_ignores_nan_padding():
    torch.manual_seed(19)
    memory = SourceMemory(6, 4, 2).eval()
    attention = SourceAttention(4, 2, dropout=0.).eval()
    torch.nn.init.normal_(attention.out.weight)
    values = [torch.randn(n, 6) for n in (2, 5)]
    f, m, _ = batching.collate(values, [[1]]*2, device='cpu', width=6, positions=1)
    queries = torch.randn(2, 3, 4)
    expected = torch.cat([attention(queries[i:i+1], memory(x[None], torch.ones(1, len(x), dtype=torch.bool)))
                          for i, x in enumerate(values)])
    f[~m] = float('nan')
    actual = attention(queries, memory(f, m))
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)
