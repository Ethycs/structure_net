from copy import deepcopy

import pytest

from experiments.structure_net.ncm_budget128_fixed_baselines import same_sources, quality_gate


def test_fixed_baseline_regrading_preserves_all_sources_and_gate_denominator():
    development = [{"id": str(i), "source_id": "s" + str(i), "group_id": "g" + str(i),
                    "source_sha256": "h" + str(i), "source": "source " + str(i)} for i in range(64)]
    rows = [{**r, "NCM_model": False, "source_truncated": False} for r in development]
    same_sources(rows, development)
    assert quality_gate({"cases": 64, "valid": 64, "passed": 61})
    assert not quality_gate({"cases": 64, "valid": 63, "passed": 61})
    assert not quality_gate({"cases": 64, "valid": 64, "passed": 60})
    with pytest.raises(ValueError):
        same_sources(rows[:-1], development)
    changed = deepcopy(rows)
    changed[0]["source"] += " changed"
    with pytest.raises(ValueError):
        same_sources(changed, development)
    changed = deepcopy(rows)
    changed[0]["source_truncated"] = True
    with pytest.raises(ValueError):
        same_sources(changed, development)
