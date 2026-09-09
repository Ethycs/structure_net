"""Selection must use paired sources and never trade away a joint gate."""
from experiments.structure_net import qwen_moba_shared_assessment as s


def rows(length, setting, speeds=(1.1, 1.2), gates=(True, True)):
    return [{"source": source, "name": setting.name, "length": length, "prefix_speedup": speed,
             "gate": gate, "metrics": {"change_pearson": .98}}
            for source, speed, gate in zip(("wikipedia", "bnc_spoken"), speeds, gates)]


def test_selection_requires_both_source_joint_gates():
    assert s.select(rows(8192, s.SETTINGS[0], gates=(True, False))) is None
    assert s.select(rows(8192, s.SETTINGS[0])[:1]) is None


def test_shortest_passing_context_precedes_speed_optimization():
    cells = rows(8192, s.SETTINGS[0]) + rows(16384, s.SETTINGS[1], speeds=(3, 3))
    result = s.select(cells)
    assert result["length"] == 8192
    assert result["setting"]["layers"] == "all"


def test_at_same_length_select_greatest_paired_mean_speed():
    cells = rows(8192, s.SETTINGS[0]) + rows(8192, s.SETTINGS[1], speeds=(1.2, 1.4))
    assert s.select(cells)["setting"]["layers"] == "last"
