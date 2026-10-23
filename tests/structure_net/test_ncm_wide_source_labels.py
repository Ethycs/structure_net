from experiments.structure_net.ncm_wide_source_labels import label_fits


def test_vetted_label_requires_both_caps_and_evidence_without_repair():
    good = {"teacher_valid": True, "text": "Supported summary.", "claims": [{"claim": "Supported", "evidence_ids": ["L0"]}],
            "stored_tokens": 128, "native_tokens_including_eos": 128, "native_roundtrip_exact": True,
            "native_special_tokens_in_text": False}
    assert label_fits(good)
    for update in ({"stored_tokens": 129}, {"native_tokens_including_eos": 129}, {"claims": []},
                   {"native_roundtrip_exact": False}, {"native_special_tokens_in_text": True}, {"text": ""},
                   {"teacher_valid": False}, {"claims": [{"claim": "Supported", "evidence_ids": []}]}):
        assert not label_fits({**good, **update})
