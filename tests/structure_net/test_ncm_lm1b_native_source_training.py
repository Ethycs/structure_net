import pytest

from experiments.structure_net import ncm_lm1b_native_source_training as run


def test_source_split_rejects_leakage_under_distinct_case_ids():
    def rows(prefix, count):
        return [{k: f'{prefix}-{k}-{i}' for k in ('id', 'source_id', 'group_id', 'source_sha256')}
            for i in range(count)]
    train = rows('train', 655)
    dev = {'short_development': rows('short', 64), 'long_diagnostic': rows('long', 64)}
    assert run.split_guard(train, dev)['id']['overlap'] == 0
    train[0]['group_id'] = dev['long_diagnostic'][0]['group_id']
    with pytest.raises(ValueError):
        run.split_guard(train, dev)
