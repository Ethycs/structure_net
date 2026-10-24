"""Test real native rendering and paired semantic accounting, not string accuracy."""
import pytest

from experiments.structure_net import ncm_sol64_bert_roundtrip as run
from experiments.neural_architecture_lab.finalize_ncm_sol64_bert_roundtrip import transitions


def test_actual64_first_targets_render_without_truncation():
    run.configure()
    tokenizer = run.native.tokenizer()
    rows = run.q.read(run.teacher.teacher.GENERATE / 'labels.json')
    cases = run.expected_cases()
    raw = {r['id']: r for r in rows}
    assert len(cases) == 64 and len({r['id'] for r in cases}) == 64
    for case in cases:
        row = raw[case['id']]
        lexical = row['native_word_ids'][1:1 + len(tokenizer.encode(row['target_text'], add_special_tokens=False))]
        assert case['texts']['labels'] == tokenizer.decode(lexical, skip_special_tokens=True).strip()
    assert run.split_guard(cases)['group_id']['overlap'] == 0
    bad = {**rows[0], 'native_word_ids': [101, 101] + [0] * 126}
    with pytest.raises(ValueError):
        run.render(bad, tokenizer)


def test_paired_transitions_preserve_same_cases_and_detect_losses():
    raw = [{'id': str(i), 'valid': True, 'passed': i != 0} for i in range(64)]
    rendered = [{'id': str(i), 'valid': True, 'passed': i != 1} for i in range(64)]
    value = transitions(raw, rendered)
    assert value['lost_passes'] == value['gained_passes'] == 1
    assert value['retained_passes'] == 62 and value['all_valid']
    with pytest.raises(ValueError):
        transitions(raw, rendered[::-1])
    assert run.teacher.joint_gate(raw, rendered)['joint_passed'] == 62
