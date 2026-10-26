import pytest

from experiments.structure_net import ncm_public_xsum_data as original
from experiments.structure_net import ncm_public_xsum_data_v2 as recovery


@pytest.mark.parametrize('document,summary', [('', 'Reference'), ('Source', ''), (' ', '\n')])
def test_empty_fields_preserved_but_never_token_eligible(document, summary):
    row = {'id': '30280969', 'document': document, 'summary': summary}
    recovery.validate_shape(row)
    assert row == {'id': '30280969', 'document': document, 'summary': summary}
    reasons = original.token_reasons(300 if document.strip() else 0,
        10 if summary.strip() else 0, [200]*10 if summary.strip() else [], {0, 100, 101})
    assert reasons


@pytest.mark.parametrize('changes', [{'document': None}, {'summary': None}, {'id': ''}, {'id': '１２３'}, {'split': 'test'}])
def test_recovery_still_rejects_unknown_shapes(changes):
    with pytest.raises(ValueError):
        recovery.validate_shape({'id': '123', 'document': 'a', 'summary': 'b', **changes})
