import pytest

from experiments.structure_net.ncm_word_final_eval import decode_words,predict,require_outputs


class Recorder:
    def __init__(self):self.calls=[]

    def decode(self,ids,skip_special_tokens):
        self.calls.append((list(ids),skip_special_tokens))
        return "full text including all blocks"


def test_decode_has_one_global_stop_and_preserves_full_unterminated_frame():
    tokenizer=Recorder()
    raw=[42]*192;raw[110]=50256
    text,end=decode_words(raw,tokenizer)
    assert text=="full text including all blocks" and end==110
    assert tokenizer.calls==[([42]*110,False)]
    raw=[43]*192
    _,end=decode_words(raw,tokenizer)
    assert end==192 and tokenizer.calls[-1]==(raw,False)
    with pytest.raises(ValueError):decode_words([42]*191,tokenizer)
    with pytest.raises(ValueError):decode_words([50257]+[42]*191,tokenizer)


def test_evaluation_rejects_target_inputs_and_source_or_decoder_changes():
    for key in ("word_ids","concept_ids","target_text"):
        with pytest.raises(ValueError,match="target labels"):
            predict(None,{key:[]},None,None)
    case={"id":"one","source_id":"session","group_id":"group","source_sha256":"hash",
        "source":"literal source","spans":[[0,14]],"split":"short_development"}
    row={**case,"source_truncated":False,"AR_fallback":False,"oracle_target_codes":False,
        "output_repair":False,"fixed_generator_passes":9}
    require_outputs([row],[case])
    for update in ({"source":"changed"},{"source_truncated":True},{"AR_fallback":True},
            {"oracle_target_codes":True},{"output_repair":True},{"fixed_generator_passes":10}):
        with pytest.raises(ValueError):require_outputs([{**row,**update}],[case])
