from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.structure_net import jspace_3b_response_eval as evaluation


def record(name="one", response=(3, 4, 6)):
    return evaluation.data.ResponseRecord(example_id=name, group_id=name, split="development",
        task="test_fixture", prompt_ids=(1, 2), response_ids=response, finish_reason="eos", eos_token_id=6,
        provenance=dict(source_uri="synthetic://evaluation", source_revision="1",
                        tokenizer_sha256="a"*64, chat_template_sha256="b"*64,
                        prompt_provenance="test fixture", response_provenance="test fixture",
                        assistant_boundary_verified=True))


class Fixture(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(23)
        self.embedding = nn.Embedding(7, 5)
        self.head = nn.Linear(5, 7)

    def hidden_forward(self, batch):
        return self.embedding(batch["input_ids"])

    def readout(self, rows):
        return self.head(rows)

    def next_logits(self, ids):
        return self.head(self.embedding(ids[:, -1]))[0]


def test_every_response_token_and_eos_exact_loss_chunking_and_teacher_kl():
    model = Fixture().train()
    teacher = deepcopy(model).eval()
    class Provider:
        def prepare(self, batch):
            self.logits = teacher.readout(teacher.hidden_forward(batch))
        def __call__(self, batch, positions):
            return self.logits[positions[:, 0], positions[:, 1]]
        def clear_cache(self):
            self.logits = None
    records = [record(), record("two", (5, 6))]
    before = deepcopy(model.state_dict())
    result = evaluation.response_losses(model, records, pad_token_id=0,
        teacher_provider=Provider(), token_chunk_size=1)
    b = evaluation.data.response_batch(records, pad_token_id=0, max_length=416)
    expected = F.cross_entropy(model.readout(model.hidden_forward(b))[b["loss_mask"]],
                               b["targets"][b["loss_mask"]])
    assert result["response_tokens"] == 5
    assert all(row["eos_supervised"] for row in result["records"])
    assert result["nll"] == pytest.approx(float(expected), rel=1e-6)
    assert result["kl"] == pytest.approx(0., abs=1e-7)
    assert model.training and all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert all(p.grad is None for p in model.parameters())


def test_loss_refuses_training_or_duplicate_records():
    model = Fixture()
    with pytest.raises(ValueError, match="Unique"):
        evaluation.response_losses(model, [record(), record()], pad_token_id=0)
    r = record()
    train = evaluation.data.ResponseRecord(**{**r.__dict__, "split": "train"})
    with pytest.raises(ValueError, match="DEV"):
        evaluation.response_losses(model, [train], pad_token_id=0)


def test_generation_uses_full_vocabulary_actual_prefixes_eos_and_complete_budget():
    prefixes = []
    def next_logits(ids):
        prefixes.append(ids.tolist()[0])
        result = torch.zeros(7)
        result[5 if len(prefixes) == 1 else 6] = 10
        return result
    result = evaluation.generate_prompt(next_logits, [1, 2], eos_token_ids=6,
                                         max_new_tokens=4, max_length=6, vocabulary_size=7)
    assert prefixes == [[1, 2], [1, 2, 5]]
    assert result["output_ids"] == [5, 6] and result["stop_reason"] == "eos"
    with pytest.raises(ValueError, match="untruncated"):
        evaluation.generate_prompt(next_logits, [1, 2, 3], eos_token_ids=6,
                                     max_new_tokens=4, max_length=6)
    result = evaluation.generate_prompt(lambda ids: torch.arange(7.), [1, 2],
                                         eos_token_ids=5, max_new_tokens=2, max_length=4)
    assert result["output_ids"] == [6, 6] and result["stop_reason"] == "token_limit"


def test_official_multiple_eos_configuration_reaches_records_and_canaries():
    model = Fixture()
    model.next_logits = lambda ids: torch.arange(7.)  # Emits end-of-text 6.
    tokenizer = type("Tokenizer", (), {"eos_token_id": 5,
        "decode": lambda self, ids, **kwargs: "42"})()
    r = evaluation.data.ResponseRecord(**{**record().__dict__, "response_ids":(3, 5), "eos_token_id":5})
    generated = evaluation.generate_record(model, r, tokenizer, max_new_tokens=4,
                                            eos_token_ids=[5, 6])
    assert generated["output_ids"] == [6] and generated["stop_reason"] == "eos"
    assert generated["eos_token_ids"] == [5, 6]
    canaries = evaluation.evaluate_canaries(model, tokenizer,
        cases=[dict(id="arith", kind="text", expected="42", prompt="17+25", input_ids=[1, 2])],
        max_new_tokens=4, eos_token_ids=[5, 6])
    assert canaries["successes"] == 1


def test_path_parity_checks_nonwinning_vocab_and_generated_prefixes():
    model = Fixture().eval()
    receipt = evaluation.full_vocabulary_path_parity(model.next_logits, model.next_logits,
        [1, 2], vocabulary_size=7, generated_prefixes=2)
    assert receipt["passed"] and len(receipt["records"]) == 3
    def corrupt(ids):
        result = model.next_logits(ids).clone()
        if ids.shape[1] > 2:
            result[int(result.argmin())] -= .1
        return result
    with pytest.raises(ValueError, match="disagree"):
        evaluation.full_vocabulary_path_parity(corrupt, model.next_logits,
            [1, 2], vocabulary_size=7)


def test_number_and_repetition_diagnostics_are_not_semantic_accuracy():
    row = dict(response="12 crates cost 1,200 dollars instead of 10 crates.",
               output_ids=[1, 2, 3, 4]*4, stop_reason="token_limit")
    result = evaluation.generation_diagnostics(row, source_text="There were 12 crates for $1,200.",
                                               reference_text="12 crates, costing 1,200 dollars.")
    assert result["numbers_absent_from_source"] == ["10"]
    assert result["reference_number_recall"] == 1.
    assert result["repetition"]["flagged"]
    assert "accuracy" not in result and "task_success" not in result
    assert evaluation.number_spans("0.50 and 50% and -3") == ["-3", "0.5", "50%"]


def test_exact_canaries_check_json_types_and_do_not_match_summaries():
    case = dict(kind="json", expected={"count": 1})
    assert evaluation.check_canary(case, '{"count": 1}')
    assert not evaluation.check_canary(case, '{"count": true}')
    assert not evaluation.check_canary(case, '```json\n{"count": 1}\n```')
    assert evaluation.check_canary(dict(kind="text", expected="42"), "42\n")


def test_failed_generation_gate_cannot_be_rescued_by_other_accuracy():
    row = dict(id="case", input_ids=[1], max_new_tokens=128, eos_token_ids=[6],
               generation_success=True)
    native = dict(records=[row])
    student = dict(records=[dict(row, generation_success=False)], accuracy=1.)
    gate = evaluation.generation_gate(native, student)
    assert not gate["passed"] and gate["semantic_evaluation_required"]
    with pytest.raises(ValueError, match="paired"):
        evaluation.generation_gate(native, dict(records=[dict(row, max_new_tokens=1)]))


def test_missing_semantic_judgments_fail_instead_of_counting_reference_overlap():
    outputs = [dict(example_id="one", response="A numerically correct but irrelevant response.")]
    provenance = dict(judge_id="fixture", judge_version="1", rubric="reviewed semantic requirements",
                      prompt_sha256="a"*64, settings={})
    result = evaluation.semantic_gate(outputs, outputs,
        [dict(example_id="one", status="timeout")], judge_provenance=provenance)
    assert not result["passed"]
    judgments = [dict(example_id="one", status="completed",
                      native=dict(task_success=True, generation_success=True),
                      student=dict(task_success=False, generation_success=True))]
    result = evaluation.semantic_gate(outputs, outputs, judgments, judge_provenance=provenance)
    assert not result["passed"] and not result["rates"]["task_success"]["passed"]
