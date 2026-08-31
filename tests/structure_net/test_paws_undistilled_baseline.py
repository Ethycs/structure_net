from __future__ import annotations

import csv

import pytest

from experiments.structure_net.paws_undistilled_baseline import Config, selected


def _dataset(path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "sentence1", "sentence2", "label"], delimiter="\t")
        writer.writeheader()
        for label in (0, 1):
            for index in range(4):
                writer.writerow({"id": f"{label}-{index}", "sentence1": f"left {label} {index}", "sentence2": f"right {label} {index}", "label": label})


def test_selection_is_balanced_deterministic_and_group_unique(tmp_path) -> None:
    path = tmp_path / "dev.tsv"
    _dataset(path)
    config = Config(samples=4, dev_path=str(path))
    first = selected(config)
    second = selected(config)
    assert first == second
    assert [row["label"] for row in first].count(0) == 2
    assert [row["label"] for row in first].count(1) == 2
    assert len({row["group_id"] for row in first}) == 4


def test_selection_rejects_odd_sample_count(tmp_path) -> None:
    path = tmp_path / "dev.tsv"
    _dataset(path)
    with pytest.raises(ValueError, match="even"):
        selected(Config(samples=3, dev_path=str(path)))
