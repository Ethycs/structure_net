import pytest

from experiments.structure_net import ncm_public_code_targets as targets


def test_pilot_selection_is_distinct_and_includes_short_and_long_targets():
    entries = [{'id': str(i), 'native_lexical_tokens': i+1} for i in range(126)]
    positions = targets.pilot_positions(entries)
    assert len(positions) == len(set(positions)) == 16
    assert positions[0] == 0 and 125 in positions
    with pytest.raises(ValueError): targets.pilot_positions(entries[:15])


def test_only_complete_native_TRAIN_codes_are_bound():
    row = {'id': 'id', 'split': 'TRAIN', 'source_sha256': 'source',
           'target_sha256': 'target', 'native_frame_sha256': 'native'}
    result = targets.bind(row, [0]*641)
    assert result['target_sha256'] == 'target' and result['codec_sha256'] == targets.codec_source.CODEC_SHA
    for invalid in ([0]*640, [8192]*641, [0.0]*641):
        with pytest.raises(ValueError): targets.bind(row, invalid)
    with pytest.raises(ValueError): targets.bind({**row, 'split': 'DEV'}, [0]*641)
