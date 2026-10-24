from dataclasses import FrozenInstanceError

import pytest

from experiments.structure_net import ncm_lm1b_native_capacity as run


class Encoding:
    def encode(self, text, **kwargs):
        return range(int(text))


def rows():
    values = []
    for band, length in enumerate((200, 600, 900, 1200, 1800)):
        for i in range(12):
            key = f"{band}-{i}"
            values.append({"id": key, "source_id": key, "group_id": key, "source_sha256": key,
                "source": str(length), "target_origin": "original_approved_target_unchanged"})
    return values


def test_control_selection_is_complete_metadata_only_and_order_independent():
    selected = run.choose(rows(), Encoding())
    assert selected == run.choose(list(reversed(rows())), Encoding())
    assert len(selected) == 16 and sum(r["capacity_control_bin"] == 0 for r in selected) == 8
    values = rows()
    for row in values:
        if row["source"] == "1800":
            row["target_origin"] = "latest_capacity35_candidate_potentially_noisy"
    with pytest.raises(ValueError, match="all16"):
        run.choose(values, Encoding())


def test_fixed256_schedule_and_budget_cannot_silently_change():
    c = run.Config()
    assert c.microbatch * c.accumulation == 16 and c.updates == 256
    assert run.rate(16) == c.max_lr and run.rate(256) == c.min_lr
    with pytest.raises(FrozenInstanceError):
        c.updates = 2
    with pytest.raises(ValueError):
        run.rate(257)
