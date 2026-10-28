import json
import pytest
from experiments.structure_net import ncm_available_pair_source_package as source


def test_array_stream_has_exact_records_and_rejects_incomplete_input(tmp_path):
    path=tmp_path/'data.json';rows=[{'text':'x'*1100000},['a','b'],{'unicode':'😀é'}]
    path.write_text(json.dumps(rows,ensure_ascii=False))
    assert list(source.stream_array(path))==rows
    path.write_text('[{"a": 1}')
    with pytest.raises(ValueError):list(source.stream_array(path))
    path.write_text('[1,]')
    with pytest.raises(ValueError):list(source.stream_array(path))


def test_unicode_fragmentation_is_lossless_and_every_fragment_fits():
    value=('😀 café 中文 source words. '*500)
    parts=list(source.fragments(value,cap=63))
    assert ''.join(parts)==value
    assert all(0<len(source.ENCODING.encode(p,disallowed_special=()))<=63 for p in parts)
