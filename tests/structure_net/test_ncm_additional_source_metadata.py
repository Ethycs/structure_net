import pytest

from experiments.structure_net import ncm_additional_source_metadata as run


def test_target_fields_rejected_before_tokenizer_access():
    for key in run.FORBIDDEN:
        with pytest.raises(ValueError, match='source-only'):
            run.prepare_one({key: []}, None, None, None)


def test_five_metadata_probes_cover_bands_and_actual_longest():
    rows = [{'id': f'{b}-{i}', 'bin': b, 'source_cl100k_tokens': b * 400 + i} for b in range(5) for i in range(8)]
    chosen = run.pilot_ids(rows)
    assert len(set(chosen)) == 5
    assert [next(r['bin'] for r in rows if r['id'] == v) for v in chosen] == list(range(5))
    assert chosen[-1] == '4-7'
