import pytest
from experiments.structure_net import ncm_public_xsum_codec_recovery as recovery


def test_exact_nested_codec_and_no_generator_or_attention_keys():
    a, b = object(), object()
    state = {'concepts.codec.encoder.weight': a, 'concepts.codec.decoder.weight': b,
             'concepts.generator.weight': object(), 'word_adapters.weight': object()}
    assert recovery.codec_state(state, ['encoder.weight', 'decoder.weight']) == {'encoder.weight': a, 'decoder.weight': b}


@pytest.mark.parametrize('state', [{}, {'codec.encoder.weight': 1},
    {'concepts.codec.encoder.weight': 1, 'concepts.codec.unexpected': 2}])
def test_missing_or_extra_codec_state_cannot_silently_load(state):
    with pytest.raises(ValueError): recovery.codec_state(state, ['encoder.weight'])


def test_recovery_preserves_selection_and_uses_fresh_execution_roots():
    assert recovery.RUNS['prepare'] == recovery.original.RUNS['prepare']
    for stage in ('pilot', 'eval', 'results'):
        assert recovery.RUNS[stage] != recovery.original.RUNS[stage]
    assert recovery.CODEC_SHA == recovery.original.CODEC_SHA
