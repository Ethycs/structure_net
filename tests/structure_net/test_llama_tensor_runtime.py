import pytest
from experiments.structure_net.llama_tensor_runtime import PromptLookup, accept, bounded, digest


def test_lookup_longest_match_prompt_only_eos_boundary():
    lookup = PromptLookup([1, 2, 3, 4, 5, 99, 2, 3, 8], {99})
    assert lookup.propose([0, 2, 3]) == [4, 5]
    assert lookup.propose([88, 77]) == []
    assert lookup.propose([1, 2, 3], 1) == [4]
    assert lookup.propose([3, 4, 5]) == []


def test_verified_prefix_and_cache_rollback():
    assert accept([10, 20, 30], [10, 21, 999, 998]) == ([10, 21], 2)
    assert accept([10, 20], [10, 20, 30]) == ([10, 20, 30], 3)
    assert accept([], [42]) == ([42], 1)
    assert accept([10, 20], [11, 999, 998]) == ([11], 1)
    with pytest.raises(ValueError):
        accept([1], [1])


def test_eos_and_cap_do_not_emit_unverified_padding():
    assert bounded([1, 99, 2], 3, {99}, True) == ([1, 99], True)
    assert bounded([1, 99, 2], 1, {99}, True) == ([1], False)
    assert bounded([1, 99, 2], 3, {99}, False) == ([1, 99, 2], False)


def test_fingerprint_changes_with_intervention():
    assert digest({"arm": "graph"}) != digest({"arm": "graph_lookup8"})


def test_quality_diagnostic_does_not_mix_fragment_facts():
    from experiments.neural_architecture_lab.analyze_llama_tensor_runtime import source_checks
    class Tokenizer:
        def encode(self, text, **kwargs):
            return text.split()
    fragments = [{"id": "F01", "reference_fact": "Project Cedar 1: Maya, $2000, November 12.",
                  "source": "Project Cedar 1: Maya, $2000, November 12.",
                  "required": {"owner":"Maya", "budget":"2000", "deadline":"November 12"}},
                 {"id": "F02", "reference_fact": "Project Elm 2: Omar, $3000, November 13.",
                  "source": "Project Elm 2: Omar, $3000, November 13.",
                  "required": {"owner":"Omar", "budget":"3000", "deadline":"November 13"}}]
    text = "F01: Project Cedar 1: Maya, $3000, November 13.\nF02: Project Elm 2: Omar, $2000, November 12."
    result = source_checks({"text":text, "output_ids":[128009]}, {"fragments":fragments}, Tokenizer())
    assert not result["diagnostic_pass"]
    assert not result["fragments"][0]["checks"]["budget"]
