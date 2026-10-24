"""Protect the replacement's unchanged all-control admission gate."""
import copy
import pytest

from experiments.structure_net import ncm_sol_judge_qualification as run


class Encoding:
    def encode(self, text, **kwargs):
        return text.split()


def fixture():
    packets, expected = run.profile.calibration()
    rows = {}
    for packet in packets:
        grades = []
        for candidate in packet['candidates']:
            ok = expected[candidate['candidate_id']]['expected_pass']
            grades.append({'candidate_id': candidate['candidate_id'], 'faithful': ok,
                'salient': ok, 'readable': True, 'claims': [], 'reason': 'fixture'})
        rows[packet['case_id']] = {'status': 'valid', 'evaluations': grades}
    return packets, expected, rows


def test_all_fifty_required_and_semantic_failure_is_terminal():
    packets, expected, rows = fixture()
    _, result = run.score(packets, expected, rows, Encoding())
    assert result['status'] == 'passed' and result['correct'] == 50
    altered = copy.deepcopy(rows)
    packet = packets[0]
    for grade in altered[packet['case_id']]['evaluations']:
        if expected[grade['candidate_id']]['expected_pass']:
            grade['faithful'] = False
            break
    _, result = run.score(packets, expected, altered, Encoding())
    assert result['status'] == 'failed_qualification' and result['correct'] == 49


def test_invalid_packet_never_drops_from_denominator():
    packets, expected, rows = fixture()
    rows[packets[0]['case_id']] = {'status': 'invalid'}
    scores, result = run.score(packets, expected, rows, Encoding())
    assert len(scores) == 50 and result['controls'] == 50
    assert result['status'] == 'failed_qualification'
    assert any(s['actual'] is None for s in scores)


def test_control_edits_or_omissions_are_rejected():
    packets, expected, rows = fixture()
    with pytest.raises(ValueError, match='fixed50'):
        run.score(packets[:-1], expected, rows, Encoding())
    packets[0]['candidates'][0]['text'] += ' modified'
    with pytest.raises(ValueError, match='text changed'):
        run.score(packets, expected, rows, Encoding())
