import copy

import pytest

from experiments.structure_net.ncm_source_readout_grade import packet_key, remember


def test_packet_cache_is_exact_and_separates_judge_roles():
    packet = {"case_id": "a", "source_lines": [{"line_id": "L1", "text": "source"}],
              "candidates": [{"candidate_id": "summary", "text": "text"}]}
    assert packet_key(packet, "secondary") == packet_key(copy.deepcopy(packet), "secondary")
    assert packet_key(packet, "primary") != packet_key(packet, "secondary")
    changed = copy.deepcopy(packet); changed["candidates"][0]["text"] = "different"
    assert packet_key(changed, "secondary") != packet_key(packet, "secondary")
    changed = copy.deepcopy(packet); changed["source_lines"][0]["text"] = "other source"
    assert packet_key(changed, "secondary") != packet_key(packet, "secondary")


def test_negative_and_exhausted_invalid_judgments_cannot_be_replaced():
    for first in ({"status": "valid", "passed": False}, {"status": "invalid"}):
        cache = {}
        remember(cache, "key", first)
        remember(cache, "key", copy.deepcopy(first))
        with pytest.raises(ValueError, match="cannot be overwritten"):
            remember(cache, "key", {"status": "valid", "passed": True})
        assert cache["key"] == first
