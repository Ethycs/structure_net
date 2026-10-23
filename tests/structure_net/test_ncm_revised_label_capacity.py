from experiments.structure_net.ncm_revised_label_capacity_recovery import capacity_critique, root, BASE_ROOT


def test_capacity_metadata_never_replaces_semantic_judgment_or_regenerates_drafts():
    critique = {"faithful": False, "salient": True, "readable": True, "claims": [], "reason": "wrong order"}
    enriched = capacity_critique({"stored_tokens": 141, "native_tokens_including_eos": 142}, critique)
    assert all(enriched[k] == v for k,v in critique.items()) and "draft_capacity" not in critique
    assert not enriched["draft_capacity"]["fits_both"]
    assert root(False,"draft") == BASE_ROOT(False,"draft")
    assert root(True,"critique") == BASE_ROOT(True,"critique")
    assert root(False,"revision").name.endswith('_v2')
