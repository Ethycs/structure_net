import torch
from experiments.structure_net.ncm_native_word_convergence import iterate


def test_causal_refinement_propagates_prefix_and_verifies_full_fixed_point():
    def step(state):
        return torch.cat((torch.ones_like(state[:, :1]), state[:, :-1] + 1), -1)
    initial = torch.zeros(1, 8, dtype=torch.long)
    _, four, _, verified_four = iterate(step, initial, 4)
    assert four[-1][0][:4] == [1, 2, 3, 4]
    assert not verified_four
    final, _, prefixes, verified = iterate(step, initial, 8)
    assert final.tolist() == [[1, 2, 3, 4, 5, 6, 7, 8]]
    assert prefixes[-1] == [7]
    assert verified
