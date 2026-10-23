import pytest
from experiments.structure_net.ncm_isolated_quality_data import check_disjoint, frame_for_prompt, original_final_metadata


def row(name):
    return {key: key + name for key in ("id", "source_id", "group_id", "source_sha256")}


def test_complete_prompt_frames_and_no_clipping():
    assert frame_for_prompt(768) == 768
    assert frame_for_prompt(769) == 1024
    assert frame_for_prompt(2276) == 2304
    assert frame_for_prompt(2304) == 2304
    with pytest.raises(ValueError):
        frame_for_prompt(2305)


def test_all_source_identity_dimensions_exclude_heldout_and_duplicate_training():
    assert not any(check_disjoint([row("a"), row("b")], [row("c")]).values())
    for key in row("a"):
        heldout = row("c"); heldout[key] = row("a")[key]
        with pytest.raises(ValueError):
            check_disjoint([row("a")], [heldout])
    with pytest.raises(ValueError):
        check_disjoint([row("a"), row("a")], [])


def test_original_final_derivation_requires_only_public_hash_metadata():
    proof = {"exact_sources": {"train": "t", "development": "d", **{f"final{i}": str(i) for i in range(256)}}}
    final = original_final_metadata(proof, [{"id": "train"}], [{"id": "development"}])
    assert len(final) == 256 and "train" not in final and "development" not in final
