import pytest

from experiments.structure_net import ncm_recovered_paraphrase_metadata as study


def source():
    return study.catalog.source_record({'source_sha256':'hash','source_component':'component',
        'source':'Original passage.', 'source_cl100k_tokens':200}, 'assistant','2023-01-01')


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs): return messages, kwargs


def test_prompt_includes_original_context_without_modifying_the_passage():
    row = source(); tokenizer = study.SourceContextTokenizer(Tokenizer(), row)
    original = [{'role':'system','content':'Summarize.'},
        {'role':'user','content':'<passage>\nOriginal passage.\n</passage>'}]
    messages, kwargs = tokenizer.apply_chat_template(original, date_string='23 Oct 2026', tokenize=False)
    assert messages[0]['content'] == 'Summarize.'+study.CONTEXT_RULE
    assert messages[1]['content'].endswith(original[1]['content'])
    assert '"role":"assistant"' in messages[1]['content']
    assert '"created_at":"2023-01-01"' in messages[1]['content']
    assert kwargs == {'date_string':'23 Oct 2026','tokenize':False}
    assert original[0]['content'] == 'Summarize.'


@pytest.mark.parametrize('field', ['target_text','native_word_ids','oracle_codes','summarizer_identity'])
def test_target_fields_cannot_reach_tokenizer(field):
    with pytest.raises(ValueError, match='source-only'):
        study.SourceContextTokenizer(Tokenizer(), {**source(), field:'forbidden'})


def test_missing_date_stays_null_and_mismatched_passage_is_rejected():
    row={**source(),'original_created_at':None}
    tokenizer=study.SourceContextTokenizer(Tokenizer(),row)
    messages=[{'role':'system','content':'Summarize.'},{'role':'user','content':'<passage>\nOriginal passage.\n</passage>'}]
    assert '"created_at":null' in tokenizer.messages(messages)[1]['content']
    messages[1]['content']='A target substituted for the source.'
    with pytest.raises(ValueError, match='literal passage'): tokenizer.messages(messages)
