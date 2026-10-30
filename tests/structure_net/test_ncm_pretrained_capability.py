import pytest

from experiments.structure_net.ncm_pretrained_capability import board, score_board


def fixture():
    cells = [[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]
    solution = [v for i, row in enumerate(cells) for v in (row + ([10] if i < 8 else []))]
    puzzle = [11, *[0 if i % 3 == 0 and x != 10 else x for i, x in enumerate(solution)], 11]
    return puzzle, solution


def test_solution_and_clues():
    puzzle, solution = fixture()
    assert score_board(solution, puzzle, solution) == dict(format_valid=True, grid_valid=True,
        clues_preserved=True, solved=True, exact=True, correct_cells=81, total_cells=81)
    # A globally permuted grid is valid but does not solve this puzzle's givens.
    permutation = [x % 9 + 1 if x <= 9 else x for x in solution]
    result = score_board(permutation, puzzle, solution)
    assert result["grid_valid"] and not result["clues_preserved"] and not result["solved"]


@pytest.mark.parametrize("mutation", ["short", "long", "separator", "blank", "bos", "duplicate"])
def test_failures_stay_in_denominator(mutation):
    puzzle, solution = fixture()
    raw = solution.copy()
    if mutation == "short": raw.pop()
    if mutation == "long": raw.append(1)
    if mutation == "separator": raw[9] = 1
    if mutation == "blank": raw[0] = 0
    if mutation == "bos": raw[0] = 11
    if mutation == "duplicate": raw[0] = raw[1]
    result = score_board(raw, puzzle, solution)
    assert not result["solved"] and not result["exact"] and result["total_cells"] == 81
    if mutation != "duplicate":
        assert not result["format_valid"] and result["correct_cells"] == 0


def test_malformed_reference_rejected():
    puzzle, solution = fixture()
    with pytest.raises(ValueError):
        score_board(solution, puzzle[:-1], solution)
    assert board([11] * 89) is None
