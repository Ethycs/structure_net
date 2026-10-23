import pytest

from experiments.structure_net.ncm_released_xsum_benchmark import (
    TARGETS, compatible, parse_author_metrics, validate_outputs)


def test_compatibility_requires_all_endpoints_with_inclusive_margin():
    assert compatible(TARGETS)
    assert compatible({k: v+.5 for k, v in TARGETS.items()})
    assert not compatible({**TARGETS, "rouge2": TARGETS["rouge2"]+.500001})
    with pytest.raises(ValueError):
        compatible({**TARGETS, "rouge1": float("nan")})


def test_author_metrics_parsing_rejects_nonfinite():
    text = "merged 1 shards\n  R-1 R-2 R-L genPPL words\n35.00 12.80 28.00 41.70 23.1\n"
    assert parse_author_metrics(text) == {**TARGETS, "gen_ppl": 41.7, "mean_words": 23.1}
    with pytest.raises(ValueError):
        parse_author_metrics(text.replace("41.70", "nan"))


def test_benchmark_pairing_is_exact_and_complete():
    rows = {"idxs": [0, 1], "hyps": ["a", "b"], "refs": ["x", "y"]}
    validate_outputs(rows, ["x", "y"])
    for changed in ({"idxs": [0, 0]}, {"refs": ["y", "x"]}, {"hyps": ["a", ""]}):
        with pytest.raises(ValueError):
            validate_outputs({**rows, **changed}, ["x", "y"])
