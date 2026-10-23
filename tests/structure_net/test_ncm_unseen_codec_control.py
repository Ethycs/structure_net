import pytest

from experiments.structure_net.ncm_memory_data import text_hash
from experiments.structure_net.ncm_unseen_codec_control import cohorts, validate_targets


def rows(n, offset=0):
    return [{k: str(i+offset) for k in ("id", "source_id", "group_id", "source_sha256")}
            for i in range(n)]


def test_novel_cohort_must_be_disjoint_on_every_key():
    old, new = rows(1999), rows(20, 3000)
    assert len(cohorts(old, new)["seen"]) == 20
    assert cohorts(old[::-1], new) == cohorts(old, new)
    for key in old[0]:
        changed = [dict(x) for x in new]; changed[0][key] = old[-1][key]
        with pytest.raises(ValueError, match="overlaps"):
            cohorts(old, changed)
    with pytest.raises(ValueError, match="repeats"):
        cohorts(old, new[:-1]+[new[0]])


def test_target_text_and_token_identity_cannot_silently_change():
    class Tokenizer:
        eos_token_id, pad_token_id = 50256, 50257
        def encode(self, text, add_special_tokens=False):
            return [1, 2]
    row = {"target": "text", "target_sha256": text_hash("text"),
           "native_target_ids": [1, 2, 50256]+[50257]*45}
    validate_targets([row], Tokenizer())
    with pytest.raises(ValueError, match="hash"):
        validate_targets([{**row, "target": "changed"}], Tokenizer())
    with pytest.raises(ValueError, match="IDs"):
        validate_targets([{**row, "native_target_ids": [1]*48}], Tokenizer())
