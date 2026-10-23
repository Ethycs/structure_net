import pytest

from experiments.structure_net.ncm_application_range_sources import select, source_bin, clean_source


def test_wide_pool_disjoint_groups_strict_quotas_and_no_old_targets():
    rows = []
    for bucket in range(4):
        for i in range(5):
            rows.append({"id": f"{bucket}/{i}", "source_id": f"s{bucket}/{i}", "group_id": f"g{i}",
                         "source_sha256": f"h{bucket}/{i}", "bin": bucket})
    with pytest.raises(ValueError, match="insufficient disjoint groups"):
        select(rows, {"train": 1, "diagnostic": 1, "sealed_final": 1})
    for row in rows:
        row["group_id"] = row["source_id"]
    chosen = select(rows, {"train": 1, "diagnostic": 1, "sealed_final": 1})
    flat = [r for values in chosen.values() for r in values]
    assert len(flat) == len({r["group_id"] for r in flat}) == 12
    assert all(len(values) == 4 for values in chosen.values())
    assert [source_bin(n) for n in (511, 512, 767, 768, 1023, 1024, 1535, 1536, 2048, 2049)] == [None, 0, 0, 1, 1, 2, 2, 3, 3, None]
    source = {**rows[0], "source": "Public input.", "spans": [], "target": "Old saved target.", "target_ids": [1, 2]}
    assert not {"target", "target_ids", "bin"} & clean_source(source).keys()
