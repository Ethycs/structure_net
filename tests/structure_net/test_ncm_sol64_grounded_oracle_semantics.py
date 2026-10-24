"""Source-on qualification and canonical history survive the matched study."""
from experiments.structure_net import ncm_sol64_grounded_oracle_semantics as run


def test_source_off_alone_cannot_qualify_the_directional_hypothesis():
    assert run.classification([]) == 'fixed64_grounded_oracle_quality_failed'
    assert run.classification(['source_off']) == 'fixed64_grounded_oracle_quality_failed'
    assert run.classification(['source_on']) == 'fixed64_grounded_oracle_semantically_qualified'
    assert run.ARMS == ('source_off', 'source_on')


def test_original_completed_codec_lineage_is_not_rebound_by_new_study():
    run.configure()
    assert run.lineage.codec.GRADE == run.ROOT / '20261024_sol64_codec_semantics_grade_v1'
    assert run.lineage.COUNT == 3523
    assert run.file_hash(run.lineage.CACHE) == run.lineage.CACHE_SHA
    assert run.lineage.codec.GRADE != run.GRADE
