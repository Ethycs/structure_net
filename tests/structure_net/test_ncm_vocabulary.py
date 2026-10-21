import pytest
from experiments.structure_net.ncm_vocabulary import CachedVocabulary
from experiments.structure_net.ncm_llama import decode_ids


class Tokenizer:
    calls=0
    def __len__(self):
        self.calls+=1
        return 50258
    def decode(self,ids,skip_special_tokens=True):return ','.join(map(str,ids))


def test_cached_vocabulary_keeps_all_validation_and_termination_with_one_lookup():
    original=Tokenizer();cached=CachedVocabulary(original)
    assert original.calls==1
    for ids in ([1,2,50256,3],[3,50257],[1,2,3],[-1,50256],[50258,50256]):
        assert decode_ids(ids,cached)==decode_ids(ids,Tokenizer())
    assert original.calls==1
