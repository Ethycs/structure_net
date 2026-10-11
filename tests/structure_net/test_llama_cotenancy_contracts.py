import json
import pytest

from experiments.structure_net.llama_cotenancy_data import example, score
from experiments.structure_net.prepare_llama_cotenancy import write_json
from experiments.structure_net.run_llama_cotenancy import aggregate_quality, promotion


def row(seed, text=None):
    item = example(seed, "work", "validation")
    text = "\n".join(item["gold_fragments"]) if text is None else text
    return {"id": item["id"], "metrics": score(text, item), "reference_nll": 1.0}


def test_quality_gate_requires_complete_matching_pairs_and_no_quality_loss():
    baseline = [row(1963), row(1964)]
    assert promotion(baseline, baseline)["quality_pass"]
    assert aggregate_quality(baseline)["complete_summaries"] == 2
    candidate = [row(1963, ""), row(1964)]
    assert not promotion(candidate, baseline)["quality_pass"]
    assert not promotion(candidate, candidate)["baseline_competent"]
    for bad in ([], baseline[:1], [baseline[0], baseline[0]]):
        with pytest.raises(ValueError):
            promotion(bad, baseline)


def test_split_generation_is_deterministic_and_disjoint():
    validation = [example(seed, family, "validation") for seed in (1963, 1964) for family in ("work", "personal")]
    test = [example(seed, family, "test") for seed in (7, 17, 29, 41, 53) for family in ("work", "personal")]
    assert not {r["prompt"] for r in validation} & {r["prompt"] for r in test}
    assert len({r["prompt"] for r in validation+test}) == 14
    assert example(7, "work", "test") == example(7, "work", "test")
    assert all(set(r["gold_fragments"]) <= set(r["source_sentences"]) for r in validation+test)


def test_strict_json_rejects_nonfinite_without_corrupting_prior_artifact(tmp_path):
    target = tmp_path / "result.json"
    write_json(target, {"valid": 1})
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            write_json(target, {"invalid": value})
    assert json.loads(target.read_text()) == {"valid": 1}
