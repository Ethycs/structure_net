import ast
from pathlib import Path

import pytest

from experiments.word_games.humaneval_labels import Rename, interface, grade, wilson
from experiments.word_games.run_humaneval_labels import aggregate


PROBLEM = {'task_id': 'control', 'entry_point': 'increment',
    'prompt': 'def increment(n: int) -> int:\n    """Return n plus one."""\n',
    'canonical_solution': '    return n + 1\n',
    'test': 'def check(candidate):\n    assert candidate(0) == 1\n    assert candidate(n=-2) == -1\n'}


def test_renaming_keeps_literals_annotations_and_helper_bindings():
    source = 'def increment(n: int) -> str:\n    return str(n) + "increment n"\n'
    table, native, renamed = interface(PROBLEM)
    translated = ast.unparse(Rename(table).visit(ast.parse(source)))
    assert 'parcel_0: int' in translated and '-> str' in translated
    assert "'increment n'" in translated
    assert 'Return n plus one' not in native and 'Return n plus one' not in renamed
    ns = {}; exec(translated, ns)
    assert ns['bridge_operation'](3) == '3increment n'


@pytest.mark.parametrize('arm,code,delivered,expected', [
    ('native', 'def increment(n):\n return n+1\n', True, True),
    ('relabeled', 'def bridge_operation(parcel_0):\n return parcel_0+1\n', True, True),
    ('native', 'def increment(n):\n return n+1\n', False, False),
    ('native', 'def increment(n):\n return n\n', True, False),
    ('native', 'raise SystemExit(0)\n', True, False),
])
def test_real_isolated_grader_and_completion_gate(tmp_path, arm, code, delivered, expected):
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    (workspace / 'solution.py').write_text(code)
    result = grade(workspace, PROBLEM, arm, tmp_path / 'evidence', delivered)
    assert result['passed'] is expected
    if not delivered: assert result['artifact_passed']


def test_grader_timeout_is_failure(tmp_path):
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    (workspace / 'solution.py').write_text('while True: pass\n')
    result = grade(workspace, PROBLEM, 'native', tmp_path / 'evidence', timeout=0.5)
    assert not result['passed'] and result['error'] == 'timeout'


def test_unattempted_not_reported_as_observed_failures_and_wilson_not_certain():
    empty = aggregate([])
    assert empty['native']['tasks'] == 0 and empty['native']['unattempted'] == 32
    assert empty['native']['observed_pass_rate'] is None
    assert wilson(32,32)[0] < 0.95
    assert wilson(0,32)[1] > 0
