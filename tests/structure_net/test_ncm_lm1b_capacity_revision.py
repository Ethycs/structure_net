from experiments.structure_net import ncm_lm1b_capacity_revision as run


def test_fixed_revision_selection_keeps_all_qualified_firsts():
    run._configure()
    prior,reviews=run.review_selection()
    assert len(prior)==35 and sum(r['requires_revision'] for r in reviews.values())==12
    assert sum(not r['requires_revision'] for r in reviews.values())==23
    assert len(run.base_cache())==2424


def test_revision_payload_contains_source_and_issues_and_kept_candidate_uses_original_request():
    run._configure()
    prior,reviews=run.review_selection()
    cases=run.q.read(run.PARENT_DATA/'cases.json')
    changed=next(c for c in cases if reviews[c['id']]['requires_revision'])
    kept=next(c for c in cases if not reviews[c['id']]['requires_revision'])
    a=run.teacher_payload(changed);b=run.teacher_payload(kept)
    import json
    value=json.loads(a['messages'][1]['content'])
    assert value['source']==changed['source'] and value['previous_summary']==prior[changed['id']]['text']
    assert value['recorded_review']==reviews[changed['id']]
    assert a['messages'][0]['content']==run.SYSTEM
    assert b['messages'][0]['content']==run.PARENT_SYSTEM
    first=run.q.read(prior[kept['id']]['attempts'][-1])
    assert b==first['request']
    assert run.replay_teacher(kept,prior[kept['id']])==prior[kept['id']]['text']
