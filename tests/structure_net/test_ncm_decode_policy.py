import ast
from pathlib import Path

import torch

from experiments.structure_net.ncm_decode_policy import pick_tokens, render_author


def test_penalty_policy_matches_actual_upstream_evaluator_function():
    path = Path("data/experiments/ncm_llama/vendor/ncm-20261021/scripts/eval_shard.py")
    function = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "pick_tokens")
    for prior in (None, torch.tensor([.1, .2, .3, .4])):
        namespace = {"torch": torch, "freq": 2., "mu": 1., "ADJ": [5., 4., .5], "ALLOW": prior}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        logits = torch.randn(3, 8, 4, generator=torch.Generator().manual_seed(72))
        actual = pick_tokens(logits, frequency=2., deficit=1., adjacency=(5., 4., .5), allowance=prior)
        expected = namespace["pick_tokens"](logits)
        assert torch.equal(actual, expected)


def test_author_render_skips_pad_without_discarding_content_before_eos():
    class Tokenizer:
        def decode(self, ids, skip_special_tokens):
            assert skip_special_tokens
            return ",".join(map(str, ids))
    assert render_author([10, 50257, 11, 50256, 12], Tokenizer()) == "10,11"
