from __future__ import annotations

from experiments.structure_net.paws_dataset_contract import (
    Config,
    build_contract,
    contract_worker,
    normalize_sentence,
    pair_group_id,
    parse_output,
    render_prompt,
)
from neural_architecture_lab.core import Experiment, ExperimentResult


def test_pair_group_is_normalized_and_order_invariant() -> None:
    assert normalize_sentence("  Café  TEST ") == "café test"
    assert pair_group_id("A sentence", "Another") == pair_group_id(
        " another ", "a SENTENCE"
    )


def test_prompt_and_output_contract_is_strict() -> None:
    prompt = render_prompt("One.", "Two.")
    assert "Sentence A: One." in prompt
    assert "Sentence B: Two." in prompt
    assert parse_output("PARAPHRASE\n") == 1
    assert parse_output("DIFFERENT") == 0
    assert parse_output("paraphrase") is None
    assert parse_output("PARAPHRASE because...") is None


def test_acquired_dataset_contract_passes() -> None:
    contract = build_contract()
    assert contract["valid"] is True
    assert contract["splits"]["train"]["rows"] == 49_401
    assert contract["tokenizers"]["tinyllm"]["over_context_count"] == 0
    assert contract["tokenizers"]["smollm"]["over_context_count"] == 0


def test_nal_worker_returns_canonical_result(tmp_path) -> None:
    output = tmp_path / "contract.json"
    experiment = Experiment(
        id="contract-test",
        hypothesis_id="paws-abc-routing-dataset-contract-v1",
        name="contract test",
        parameters={"configuration": Config().__dict__, "output": str(output)},
        device_id=-1,
    )
    result = contract_worker(experiment, -1)
    assert isinstance(result, ExperimentResult)
    assert result.primary_metric == 1.0
    assert result.metrics["contract_valid"] == 1.0
    assert output.is_file()
