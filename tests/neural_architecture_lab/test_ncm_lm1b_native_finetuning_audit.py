import pytest
import torch

from experiments.neural_architecture_lab import finalize_ncm_lm1b_native_finetuning_pilot as audit


def generation():
    return {"codes": torch.zeros(2, 641, dtype=torch.long), "raw_word_ids": torch.full((2, 128), 30527, dtype=torch.long)}


def test_all_native_slots_including_unmapped_word_ids_are_preserved():
    audit.require_generation(generation(), 2)


@pytest.mark.parametrize("kind", ("truncated", "word-overflow", "code-overflow", "wrong-type"))
def test_incomplete_or_non_native_saved_probes_cannot_admit_systems_success(kind):
    value = generation()
    if kind == "truncated":
        value["raw_word_ids"] = value["raw_word_ids"][:, :-1]
    elif kind == "word-overflow":
        value["raw_word_ids"][0, 0] = 30528
    elif kind == "code-overflow":
        value["codes"][0, 0] = 8192
    else:
        value["codes"] = value["codes"].float()
    with pytest.raises(ValueError, match="native hard-generation"):
        audit.require_generation(value, 2)
