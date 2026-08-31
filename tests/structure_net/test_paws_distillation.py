from experiments.structure_net.paws_distillation import partition


def test_partition_is_stable_and_bounded() -> None:
    values = [partition(f"group-{index}") for index in range(100)]
    assert values == [partition(f"group-{index}") for index in range(100)]
    assert set(values) == {0, 1, 2, 3}
