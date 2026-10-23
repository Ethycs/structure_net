from experiments.structure_net import ncm_broad_decoder_transfer as engine
from experiments.structure_net.ncm_vocabulary_balanced_codec import DATA, PLAN, redirected, synthetic_targets


def test_synthetic_stream_is_balanced_deterministic_and_has_valid_terminators():
    rows = synthetic_targets(24, vocabulary=17)
    assert rows == synthetic_targets(24, vocabulary=17)
    counts = [0]*17
    for row in rows:
        ids = row["native_target_ids"]
        assert len(ids) == 48
        end = ids.index(50256)
        assert 8 <= end <= 47 and ids[end+1:] == [50257]*(47-end)
        for token in ids[:end]:
            assert 0 <= token < 17
            counts[token] += 1
    assert min(counts) > 0 and max(counts)-min(counts) <= 1
    assert rows != synthetic_targets(24, vocabulary=17, content_seed=6043)


def test_wrapper_restores_engine_configuration_without_changing_source():
    old = (engine.DATA, engine.PLAN, engine.freeze)
    with redirected():
        assert engine.DATA == DATA and engine.PLAN == PLAN and engine.freeze is not old[2]
    assert (engine.DATA, engine.PLAN, engine.freeze) == old
