import hashlib
import pytest
from experiments.structure_net.ncm_cached_stored_summary_join import bind


def fixtures():
    sha=lambda t:hashlib.sha256(t.encode()).hexdigest()
    source={'id':'cached','source':'12 tests passed.','source_sha256':sha('12 tests passed.')}
    original={**source,'id':'stored','split':'train','target':'12 passed','target_sha256':sha('12 passed'),
        'source_id':'family','group_id':'group'}
    metadata={**original,'original_line_number':7,'target_tokens':3,'native_target_tokens':3}
    return source,original,metadata


def test_exact_source_join_preserves_original_label_and_new_source_identity():
    source,original,metadata=fixtures();value=bind(source,original,metadata)
    assert value['id']=='cached' and value['stored_label_id']=='stored'
    assert value['target_text']=='12 passed' and value['semantic_qualified'] is False


def test_changed_source_or_label_is_rejected():
    source,original,metadata=fixtures()
    for changed in ({**original,'source':'13 tests passed.'},{**original,'target':'13 passed'},
                    {**original,'split':'test'}):
        with pytest.raises(ValueError):bind(source,changed,metadata)
