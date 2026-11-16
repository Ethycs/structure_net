from dataclasses import asdict
import json

import pytest

from experiments.structure_net import jspace_3b_response_data as corpus


class Tokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c)+100 for c in text]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        user = next(m["content"] for m in messages if m["role"] == "user")
        prefix = [50] + self.encode(user) + [51]
        if add_generation_prompt:
            return prefix
        return prefix + self.encode(messages[-1]["content"]) + [self.eos_token_id] + self.encode("\n")


def row(i="one", context="source", response="A complete answer."):
    return dict(example_id=i, source_group="group:"+i, source_uri="fixture", source_revision="fixed",
        source_row=0, family="dolly", instruction="Explain", context=context, response=response,
        task="instruction:open_qa", reference_quality="fixture")


def encode(value=None, **kwargs):
    return corpus.tokenize_reference(Tokenizer(), value or row(), split="train", group_id="g",
        tokenization=dict(tokenizer_sha256="t", chat_template_sha256="c"), **kwargs)


def test_complete_response_has_real_eos_and_every_causal_target():
    record = encode()
    assert record.response_ids[-1] == 1
    assert len(record.prediction_positions) == len(record.response_ids)
    batch = corpus.data.response_batch([record], pad_token_id=0, max_length=416)
    assert batch["targets"][batch["loss_mask"]].tolist() == list(record.response_ids)
    assert batch["counts"]["supervised_tokens"] == len(record.response_ids)
    assert not batch["loss_mask"][0, :len(record.prompt_ids)-1].any()


@pytest.mark.parametrize("limit, reason", [(dict(max_prompt=2), "prompt_over_limit"),
    (dict(max_response=2), "response_over_limit"), (dict(max_total=2), "total_over_limit")])
def test_oversize_is_rejected_without_truncation(limit, reason):
    with pytest.raises(ValueError, match=reason):
        encode(**limit)


def test_embedded_chat_tokens_rejected():
    with pytest.raises(ValueError, match="embedded_chat_special_token"):
        encode(row(response="bad <|im_end|> tail"))


def test_unexpected_template_footer_rejected():
    class Bad(Tokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            ids = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return ids if add_generation_prompt else ids+[99]
    with pytest.raises(ValueError, match="unexpected_template_footer"):
        corpus.tokenize_reference(Bad(), row(), split="train", group_id="g",
            tokenization=dict(tokenizer_sha256="t", chat_template_sha256="c"))


def test_connected_source_variants_never_cross_splits():
    common = " ".join(f"word{i}" for i in range(30))
    rows = [row("a", common+" first"), row("b", common+" second"), row("c", "unique unrelated source")]
    groups = corpus.grouped_ids(rows)
    assert groups[0] == groups[1] != groups[2]
    assert corpus.split_for(groups[0], 2) == corpus.split_for(groups[1], 2)
    assert corpus.grouped_ids(rows) == groups


def test_exposed_questions_exclude_option_suffix_and_keep_source_receipt(tmp_path):
    path = tmp_path/"exposed.json"
    path.write_text(json.dumps(dict(items=[dict(prompt="What is the capital?\n\nA. One\nB. Two")])))
    exact, _, receipt = corpus.exposed_questions([path])
    assert "what is the capital" in exact
    assert receipt[0]["sha256"] == corpus.sha(path)


def test_sealed_tasks_cannot_be_supplied_as_exposed_inputs():
    with pytest.raises(ValueError, match="Sealed task contents"):
        corpus.exposed_questions([corpus.SEALED_TASKS])


def test_manifest_and_payload_are_authenticated(tmp_path):
    record = encode()
    path = tmp_path/"train.jsonl"
    path.write_text(json.dumps(asdict(record))+"\n")
    manifest = dict(artifacts={"train.jsonl": corpus.sha(path)})
    manifest["fingerprint"] = corpus.data.digest(manifest)
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    assert corpus.load_records(tmp_path, "train") == [record]
    path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="payload changed"):
        corpus.load_records(tmp_path, "train")
    with pytest.raises(ValueError, match="Only TRAIN/DEV"):
        corpus.load_records(tmp_path, "test")
