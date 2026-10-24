from copy import deepcopy

import pytest

from experiments.structure_net import ncm_additional_training_reservation as run


def rows():
    return [{"id": f"{band}-{i}", "source_sha256": f"hash-{band}-{i}", "source_component": f"group-{band}-{i}",
        "blocked": False, "candidate": True, "bin": band, "tokens": tokens, "public_origins": [{"source_id": str(i)}]}
        for band, tokens in enumerate((200, 600, 900, 1200, 1800)) for i in range(3)]


def test_complete_disjoint_selection_is_order_independent():
    data = rows()
    reserved = {"group-0-0"}
    data[4]["blocked"] = True
    quotas = {i: 1 for i in range(5)}
    selected, proof = run.choose(data, reserved, quotas)
    assert run.choose(list(reversed(data)), reserved, quotas) == (selected, proof)
    assert len(selected) == 5
    assert all(r["source_component"] not in {"group-0-0", data[4]["source_component"]} for r in selected)
    assert proof["all_reserved_components_excluded"] == 1 and proof["blocked_components_excluded"] == 1


def test_any_blocked_alias_excludes_the_whole_component():
    data = rows()
    data[1]["source_component"] = data[0]["source_component"]
    data[0]["blocked"] = True
    selected, _ = run.choose(data, set(), {i: 1 for i in range(5)})
    assert all(r["source_component"] != data[0]["source_component"] for r in selected)


def test_quota_shortfall_aborts_without_an_easier_population():
    with pytest.raises(ValueError, match="insufficient"):
        run.choose(rows(), {"group-0-0", "group-0-1", "group-0-2"}, {i: 1 for i in range(5)})


@pytest.mark.parametrize("field", ("source", "target_text", "text"))
def test_text_is_rejected_before_source_selection(field):
    data = deepcopy(rows())
    data[0][field] = "must not read source text in this control"
    with pytest.raises(ValueError, match="metadata only"):
        run.choose(data, set(), {i: 1 for i in range(5)})
