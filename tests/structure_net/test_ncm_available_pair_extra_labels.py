from experiments.structure_net import ncm_available_pair_extra_labels as extra


def test_content_identity_ignores_namespace_but_keeps_target_and_span_changes():
    pointer={'span_text_sha256':'a'*64,'turn_text_sha256':'b'*64,'start_char':0,'end_char':20,'source_id':'first'}
    a=extra.label({'summary':'a summary','spans':[pointer]})
    b=extra.label({'summary':'a summary','pointer':{**pointer,'source_id':'second'}})
    assert a[0]==b[0]
    assert extra.label({'summary':'different','pointer':pointer})[0]!=a[0]
    assert extra.label({'summary':'a summary','pointer':{**pointer,'end_char':21}})[0]!=a[0]


def test_missing_or_invalid_source_pointers_remain_unresolved():
    assert extra.label({'summary':'a summary'}) is None
    assert extra.label({'summary':'a summary','spans':[]}) is None
    assert extra.label({'summary':'a summary','pointer':{'span_text_sha256':'a'*64}}) is None
